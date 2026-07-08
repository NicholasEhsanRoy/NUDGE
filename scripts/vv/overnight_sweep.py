#!/usr/bin/env python3
"""Overnight V&V: gate calibration + identifiability power sweep.

Pure computation over the proven generate_synthetic_perturbseq + fit code. Two
studies, decoupled so the expensive fitting runs ONCE per dataset and the gates
are swept analytically in post-processing:

1. GATE CALIBRATION.
   - Many *linear* datasets → the switch-detection gate must reject them
     (false-positive rate vs margin_k).
   - Many *switch* datasets (known mechanism) → correct / unresolved / wrong /
     missed rates vs margin_k. Together these calibrate margin_k.
2. IDENTIFIABILITY POWER SWEEP.
   - A grid over mechanism × effect-size × cells-per-condition × noise → the
     fraction correctly attributed → heatmaps ("identifiability vs noise vs cells").

Raw per-dataset losses are written incrementally to CSV (so a crash keeps partial
results); ``analyze`` reads them, sweeps margin_k, and writes figures + a summary.

Usage:
    python scripts/vv/overnight_sweep.py collect   # the long run (writes CSVs)
    python scripts/vv/overnight_sweep.py analyze    # figures + summary from CSVs
    python scripts/vv/overnight_sweep.py all        # collect then analyze
    python scripts/vv/overnight_sweep.py all --smoke  # tiny config, ~2 min
"""

from __future__ import annotations

import csv
import sys
import time
import traceback
from pathlib import Path

import jax
import numpy as np

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef
from nudge.core.vocabulary import MechanismClass
from nudge.data.synthetic import PerturbationSpec, generate_synthetic_perturbseq
from nudge.inference.classify import decide, switch_detected
from nudge.inference.fit import (
    _condition_counts,
    _log1p_distance,
    _loss_stats,
    _self_distance,
    _updated_circuit,
    fit_parameters,
)

RESULTS = Path(__file__).parent / "results"
LINEAR_CSV = RESULTS / "gate_linear.csv"
ATTR_CSV = RESULTS / "attribution.csv"

# Which parameter each mechanism moves, and a set of effect-size factors.
MECHANISMS = {
    "threshold": ("K", [1.5, 2.0, 3.0, 5.0]),
    "gain": ("n", [0.5, 0.34, 0.2, 0.1]),
    "ceiling": ("vmax", [0.7, 0.5, 0.3, 0.15]),
}
# A single "strong" factor per mechanism for the calibration study.
STRONG_FACTOR = {"threshold": 3.0, "gain": 0.2, "ceiling": 0.3}


def switch_circuit() -> Circuit:
    return Circuit(
        [SpeciesDef("IN", basal=1.0, decay=1.0), SpeciesDef("SW", basal=0.05, decay=1.0)],
        [EdgeDef(0, 1, "hill_activation", K=1.0, n=6.0, vmax=2.0)],
    )


def linear_circuit() -> Circuit:
    return Circuit(
        [SpeciesDef("IN", basal=1.0, decay=1.0), SpeciesDef("SW", basal=0.5, decay=1.0)],
        [EdgeDef(0, 1, "linear", weight=1.0)],
    )


def _wt_gate_losses(adata, circuit, n_cells, steps, seed, target_edge=0):
    """WT mechanistic + linear-baseline losses and the loss noise floor."""
    wt_counts = _condition_counts(adata, "WT")
    floor_mean, floor_std = _self_distance(wt_counts, n_cells, jax.random.key(seed + 50))
    wt_free = [("edge", target_edge, p) for p in ("K", "n", "vmax")]
    wt_vals, wt_mech_h = fit_parameters(
        adata, circuit, wt_free, condition="WT", n_cells=n_cells, steps=steps, seed=seed
    )
    wtc = _updated_circuit(circuit, wt_vals)
    _, wt_lin_h = fit_parameters(
        adata, wtc.linear_baseline(), [("edge", target_edge, "weight")],
        condition="WT", n_cells=n_cells, steps=steps, seed=seed + 1,
    )
    return wtc, floor_mean, floor_std, _loss_stats(wt_mech_h)[0], _loss_stats(wt_lin_h)[0]


def _restricted_losses(adata, wtc, condition, n_cells, steps, seed, target_edge=0):
    """The three restricted-mechanistic fit losses + the perturbed-vs-WT distance."""
    wt_counts = _condition_counts(adata, "WT")
    perturbed = _condition_counts(adata, condition)
    wt_distance = _log1p_distance(perturbed, wt_counts, n_cells, jax.random.key(seed + 7))
    losses = {}
    for j, p in enumerate(("K", "n", "vmax")):
        _, h = fit_parameters(
            adata, wtc, [("edge", target_edge, p)],
            condition=condition, n_cells=n_cells, steps=steps, seed=seed + 3 * j,
        )
        losses[p] = _loss_stats(h)[0]
    return losses, wt_distance


def _append(path: Path, row: dict, header: list[str]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()


def collect(smoke: bool = False) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    n_cells = 256 if not smoke else 96
    steps = 250 if not smoke else 60
    n_linear = 300 if not smoke else 3
    n_switch = 40 if not smoke else 2
    reps = 3 if not smoke else 1
    cells_grid = [100, 300, 1000, 3000] if not smoke else [300]
    noise_grid = [0, 1, 2] if not smoke else [1]
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    lin_header = ["seed", "weight_factor", "floor_mean", "floor_std", "wt_mech", "wt_lin"]
    attr_header = [
        "study", "mechanism", "param", "factor", "cells", "noise", "rep",
        "floor_mean", "floor_std", "wt_mech", "wt_lin",
        "loss_K", "loss_n", "loss_vmax", "wt_distance",
    ]

    # ---- Study 1a: linear datasets (false-positive calibration) ----
    log(f"GATE CALIBRATION — {n_linear} linear datasets")
    for i in range(n_linear):
        try:
            factor = float(np.linspace(0.3, 0.8, 5)[i % 5])
            adata = generate_synthetic_perturbseq(
                linear_circuit(),
                [PerturbationSpec("lin", "edge", 0, "weight", factor)],
                n_cells_per_condition=2000, realism_level=1, seed=i,
            )
            _, fmean, fstd, wtm, wtl = _wt_gate_losses(
                adata, switch_circuit(), n_cells, steps, seed=1000 + i
            )
            _append(LINEAR_CSV, {
                "seed": i, "weight_factor": round(factor, 3),
                "floor_mean": fmean, "floor_std": fstd, "wt_mech": wtm, "wt_lin": wtl,
            }, lin_header)
        except Exception:
            log(f"  linear {i} FAILED\n{traceback.format_exc()}")
        if (i + 1) % 25 == 0:
            log(f"  linear {i + 1}/{n_linear}")

    # ---- Study 1b: switch datasets (correct/unresolved/wrong calibration) ----
    log(f"GATE CALIBRATION — {n_switch} switch datasets per mechanism")
    for mech, factor in STRONG_FACTOR.items():
        param = MECHANISMS[mech][0]
        for i in range(n_switch):
            _one_switch_row("calib", mech, param, factor, 2000, 1, i, n_cells, steps, attr_header, log)

    # ---- Study 2: identifiability power sweep ----
    total = sum(len(f) for _, f in MECHANISMS.values()) * len(cells_grid) * len(noise_grid) * reps
    log(f"POWER SWEEP — {total} points")
    done = 0
    for mech, (param, factors) in MECHANISMS.items():
        for factor in factors:
            for cells in cells_grid:
                for noise in noise_grid:
                    for rep in range(reps):
                        _one_switch_row(
                            "power", mech, param, factor, cells, noise, rep,
                            min(n_cells, cells), steps, attr_header, log,
                        )
                        done += 1
            log(f"  power {mech} factor={factor}: {done}/{total}")
    log(f"DONE collect in {time.time() - t0:.0f}s")


def _one_switch_row(study, mech, param, factor, cells, noise, rep, n_cells, steps, header, log):
    try:
        seed = hash((study, mech, factor, cells, noise, rep)) % (2**31)
        adata = generate_synthetic_perturbseq(
            switch_circuit(),
            [PerturbationSpec(mech, "edge", 0, param, factor)],
            n_cells_per_condition=cells, realism_level=noise, seed=seed % 10000,
        )
        wtc, fmean, fstd, wtm, wtl = _wt_gate_losses(adata, switch_circuit(), n_cells, steps, seed)
        losses, wt_dist = _restricted_losses(adata, wtc, mech, n_cells, steps, seed)
        _append(ATTR_CSV, {
            "study": study, "mechanism": mech, "param": param, "factor": factor,
            "cells": cells, "noise": noise, "rep": rep,
            "floor_mean": fmean, "floor_std": fstd, "wt_mech": wtm, "wt_lin": wtl,
            "loss_K": losses["K"], "loss_n": losses["n"], "loss_vmax": losses["vmax"],
            "wt_distance": wt_dist,
        }, header)
    except Exception:
        log(f"  {study} {mech} f={factor} cells={cells} noise={noise} r={rep} FAILED\n{traceback.format_exc()}")


# --------------------------------------------------------------------------- #
#  Analysis                                                                    #
# --------------------------------------------------------------------------- #
_PARAM_MECH = {"K": MechanismClass.THRESHOLD, "n": MechanismClass.GAIN, "vmax": MechanismClass.CEILING}


def _verdict(row: dict, margin_k: float) -> str:
    """Apply the full gate stack to one attribution row at a given margin_k."""
    floor_std = float(row["floor_std"])
    floor_mean = float(row["floor_mean"])
    noise_margin = margin_k * floor_std
    if not switch_detected(float(row["wt_mech"]), float(row["wt_lin"]), noise_margin=noise_margin):
        return "off-model"
    losses = {"K": float(row["loss_K"]), "n": float(row["loss_n"]), "vmax": float(row["loss_vmax"])}
    call = decide(
        "p", losses, float(row["wt_distance"]),
        noise_margin=noise_margin, effect_margin=floor_mean + 3 * floor_std,
        off_model_loss=5.0 * floor_mean,
    )
    return call.mechanism.value


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def analyze() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lin = _read(LINEAR_CSV)
    attr = _read(ATTR_CSV)
    calib = [r for r in attr if r["study"] == "calib"]
    power = [r for r in attr if r["study"] == "power"]
    summary = ["# Overnight V&V results\n"]

    # ---- Calibration curve: FP (linear) + correct/wrong (switch) vs margin_k ----
    margin_ks = np.linspace(0.0, 3.0, 31)
    fp, correct, unresolved, wrong, missed = [], [], [], [], []
    for mk in margin_ks:
        fp.append(np.mean([
            switch_detected(float(r["wt_mech"]), float(r["wt_lin"]), noise_margin=mk * float(r["floor_std"]))
            for r in lin
        ]) if lin else np.nan)
        verdicts = [(r["mechanism"], _verdict(r, mk)) for r in calib]
        n = max(len(verdicts), 1)
        correct.append(sum(t == v for t, v in verdicts) / n)
        unresolved.append(sum(v == "unresolved" for _, v in verdicts) / n)
        missed.append(sum(v == "off-model" for _, v in verdicts) / n)
        wrong.append(sum(v in {"threshold", "gain", "ceiling"} and v != t for t, v in verdicts) / n)

    # recommend margin_k: lowest FP <= 0.02 that maximizes correct
    ok = [i for i, f in enumerate(fp) if f <= 0.02]
    best_i = max(ok, key=lambda i: correct[i]) if ok else int(np.argmin(fp))
    rec_mk = float(margin_ks[best_i])
    summary.append(f"**Recommended margin_k = {rec_mk:.2f}**  "
                   f"(false-positive rate on linear data {fp[best_i]:.1%}, "
                   f"correct-attribution {correct[best_i]:.1%}, "
                   f"misclassification {wrong[best_i]:.1%}, "
                   f"unresolved {unresolved[best_i]:.1%})\n")
    summary.append(f"- Calibrated against {len(lin)} synthetic linear datasets and "
                   f"{len(calib)} switch datasets.\n")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(margin_ks, fp, label="false positive (linear→switch)", color="crimson")
    ax.plot(margin_ks, correct, label="correct attribution (switch)", color="seagreen")
    ax.plot(margin_ks, wrong, label="misclassified (wrong mechanism)", color="black", ls="--")
    ax.plot(margin_ks, unresolved, label="unresolved (abstain)", color="steelblue", ls=":")
    ax.axvline(rec_mk, color="gray", ls="-", lw=1, label=f"recommended k={rec_mk:.2f}")
    ax.set_xlabel("margin_k (noise-floor multiplier)")
    ax.set_ylabel("rate")
    ax.set_title("Gate calibration")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "calibration.png", dpi=130)
    plt.close(fig)

    # ---- Identifiability heatmaps: correct fraction vs cells × noise, per mechanism ----
    if power:
        cells_vals = sorted({int(r["cells"]) for r in power})
        noise_vals = sorted({int(r["noise"]) for r in power})
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, mech in zip(axes, MECHANISMS):
            grid = np.full((len(noise_vals), len(cells_vals)), np.nan)
            for ni, noise in enumerate(noise_vals):
                for ci, cells in enumerate(cells_vals):
                    rows = [r for r in power if r["mechanism"] == mech
                            and int(r["cells"]) == cells and int(r["noise"]) == noise]
                    if rows:
                        grid[ni, ci] = np.mean([_verdict(r, rec_mk) == mech for r in rows])
            im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto", origin="lower")
            ax.set_xticks(range(len(cells_vals)), cells_vals)
            ax.set_yticks(range(len(noise_vals)), noise_vals)
            ax.set_xlabel("cells / condition")
            ax.set_ylabel("noise level")
            ax.set_title(f"{mech}: correct fraction")
            for ni in range(len(noise_vals)):
                for ci in range(len(cells_vals)):
                    if not np.isnan(grid[ni, ci]):
                        ax.text(ci, ni, f"{grid[ni, ci]:.2f}", ha="center", va="center",
                                color="white" if grid[ni, ci] < 0.6 else "black", fontsize=8)
        fig.colorbar(im, ax=axes, fraction=0.02)
        fig.suptitle(f"Identifiability (margin_k={rec_mk:.2f})")
        fig.savefig(RESULTS / "identifiability_cells_noise.png", dpi=130)
        plt.close(fig)

        # effect-size × cells (at lowest noise), per mechanism
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (mech, (_, factors)) in zip(axes, MECHANISMS.items()):
            fac_vals = sorted({float(r["factor"]) for r in power if r["mechanism"] == mech})
            grid = np.full((len(fac_vals), len(cells_vals)), np.nan)
            for fi, fac in enumerate(fac_vals):
                for ci, cells in enumerate(cells_vals):
                    rows = [r for r in power if r["mechanism"] == mech
                            and float(r["factor"]) == fac and int(r["cells"]) == cells
                            and int(r["noise"]) == min(noise_vals)]
                    if rows:
                        grid[fi, ci] = np.mean([_verdict(r, rec_mk) == mech for r in rows])
            im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto", origin="lower")
            ax.set_xticks(range(len(cells_vals)), cells_vals)
            ax.set_yticks(range(len(fac_vals)), [f"{v:g}" for v in fac_vals])
            ax.set_xlabel("cells / condition")
            ax.set_ylabel("effect size (factor)")
            ax.set_title(f"{mech}")
        fig.colorbar(im, ax=axes, fraction=0.02)
        fig.suptitle(f"Identifiability vs effect size (noise={min(noise_vals)}, margin_k={rec_mk:.2f})")
        fig.savefig(RESULTS / "identifiability_effect_cells.png", dpi=130)
        plt.close(fig)

    # ---- Confusion matrix over the switch calibration set ----
    if calib:
        classes = ["threshold", "gain", "ceiling", "unresolved", "off-model"]
        conf = np.zeros((3, len(classes)))
        truth_order = ["threshold", "gain", "ceiling"]
        for r in calib:
            v = _verdict(r, rec_mk)
            ti = truth_order.index(r["mechanism"])
            if v in classes:
                conf[ti, classes.index(v)] += 1
        conf_norm = conf / np.clip(conf.sum(1, keepdims=True), 1, None)
        summary.append("\n## Confusion (rows = true mechanism, at recommended margin_k)\n")
        summary.append("| true \\ called | " + " | ".join(classes) + " |")
        summary.append("|" + "---|" * (len(classes) + 1))
        for ti, t in enumerate(truth_order):
            summary.append(f"| {t} | " + " | ".join(f"{conf_norm[ti, j]:.2f}" for j in range(len(classes))) + " |")
        summary.append("")

    summary.append("\n## Figures\n- `calibration.png`\n- `identifiability_cells_noise.png`"
                   "\n- `identifiability_effect_cells.png`\n")
    (RESULTS / "SUMMARY.md").write_text("\n".join(summary))
    print("\n".join(summary))
    print(f"\nWrote figures + SUMMARY.md to {RESULTS}")


if __name__ == "__main__":
    args = sys.argv[1:]
    smoke = "--smoke" in args
    cmd = next((a for a in args if not a.startswith("--")), "all")
    if cmd in ("collect", "all"):
        collect(smoke=smoke)
    if cmd in ("analyze", "all"):
        analyze()
