"""Measure the Lyapunov covariance attribution on INDEPENDENT SSA toggle data.

The existing `tests/inference/test_lyapunov.py` only exercises the covariance
attribution on *inverse-crime* data (`sample_lna_mixture` — the fitter fits its own
Gaussian samples). This script closes the genuine open question: does
`attribute_lyapunov_single` / `attribute_lyapunov_multi` recover-or-abstain on data from
the INDEPENDENT tau-leaping SSA (`generate_toggle_perturbseq`), with the fail-safe
intact?

It bridges SSA counts -> activity (`inference.bridge.counts_to_activity`) exactly as the
real-data path does, then runs the single-condition and (basal-shifted) two-operating-
point attribution for each of gain / threshold / ceiling ground truth, across seeds.

Run: `uv run python scripts/vv/toggle_lyapunov_ssa.py`
"""

from __future__ import annotations

import numpy as np

from nudge.circuits import toggle
from nudge.data.stochastic import generate_toggle_perturbseq
from nudge.data.synthetic import PerturbationSpec
from nudge.inference.bridge import counts_to_activity
from nudge.inference.lyapunov import (
    OperatingPoint,
    attribute_lyapunov_multi,
    attribute_lyapunov_single,
    calibrate_from_wt,
)
from nudge.mechanisms.readout import Readout

# One marker gene per species (identity readout -> gene name == species name).
MARKERS = {"A": ["A"], "B": ["B"]}
# Deep-sequencing readout (Lambda = 0.2 + 15*activity): high mode ~ 30 counts, so
# scale*peak clears lna_reliable's >=15 depth guard (the default scale=5 trips it).
DEEP = Readout.identity(2, scale=15.0)

# A gain factor that stays bistable on ONE edge (n: 4 -> 2.4, factor 0.6); a threshold
# and ceiling likewise chosen to keep both attractors populated (so the LNA is defined).
FACTORS = {"n": 0.6, "K": 1.6, "vmax": 0.6}
MECH = {"n": "gain", "K": "threshold", "vmax": "ceiling"}


def _activity(adata: object, condition: str, circuit: object) -> np.ndarray:
    mask = np.asarray(adata.obs["condition"] == condition)  # type: ignore[attr-defined]
    return counts_to_activity(adata[mask], circuit, MARKERS)  # type: ignore[index]


def _make_adata(basal: float, param: str, seed: int, n_cells: int = 3000):
    circ = toggle(basal=basal)
    return generate_toggle_perturbseq(
        circ,
        [PerturbationSpec("cond", "edge", 0, param, FACTORS[param])],
        readout=DEEP,
        n_cells_per_condition=n_cells,
        seed=seed,
    ), circ


def run_single(seeds: range = range(3)) -> None:
    print("\n=== single-condition (independent SSA, basal=0.05) ===", flush=True)
    print(f"{'true':>10} {'seed':>4}  {'label':>18}  NLLs (n / K / vmax)", flush=True)
    for param in ("n", "K", "vmax"):
        for seed in seeds:
            adata, circ = _make_adata(0.05, param, seed)
            wt = _activity(adata, "WT", circ)
            cond = _activity(adata, "cond", circ)
            label, nlls = attribute_lyapunov_single(
                cond, circ, wt_data=wt, target_edge=0, steps=200, seed=seed
            )
            s = "  ".join(f"{nlls.get(k, float('nan')):.3f}" for k in ("n", "K", "vmax"))
            print(f"{MECH[param]:>10} {seed:>4}  {label:>18}  {s}", flush=True)


def run_multi(seeds: range = range(3)) -> None:
    print("\n=== two operating points (basal-B 0.05 + 0.30) ===", flush=True)
    print(
        f"{'true':>10} {'seed':>4}  {'label':>14}  NLLs (gain/threshold/ceiling)",
        flush=True,
    )
    for param in ("n", "K", "vmax"):
        for seed in seeds:
            points: list[OperatingPoint] = []
            for basal in (0.05, 0.30):
                circ = toggle(basal=basal)
                adata = generate_toggle_perturbseq(
                    circ,
                    [PerturbationSpec("cond", "edge", 0, param, FACTORS[param])],
                    readout=DEEP,
                    n_cells_per_condition=3000,
                    seed=seed,
                )
                wt = _activity(adata, "WT", circ)
                cond = _activity(adata, "cond", circ)
                scale, obs = calibrate_from_wt(wt, circ)
                points.append(
                    OperatingPoint(data=cond, circuit=circ, scale=scale, obs_sd=obs)
                )
            label, nlls = attribute_lyapunov_multi(
                points, target_edge=0, steps=200, seed=seed
            )
            s = "  ".join(
                f"{nlls.get(k, float('nan')):.3f}"
                for k in ("gain", "threshold", "ceiling")
            )
            print(f"{MECH[param]:>10} {seed:>4}  {label:>14}  {s}", flush=True)


if __name__ == "__main__":
    import sys

    seeds = range(int(sys.argv[2])) if len(sys.argv) > 2 else range(3)
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("single", "both"):
        run_single(seeds)
    if which in ("multi", "both"):
        run_multi(seeds)
