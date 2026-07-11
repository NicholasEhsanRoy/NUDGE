"""Run NUDGE's gLV attribution on REAL MDSINE2 vancomycin data (Task 3: push the boundary).

Tests the user's hypothesis: does denser (twice-daily) sampling that OBSERVES the antibiotic
window — MDSINE2's 7-day vancomycin pulse with ~7 in-pulse samples, vs Stein's 1-day pulse with
NONE observed during it — let NUDGE RESOLVE the ``ε`` (direct-kill) axis where Stein could only
abstain, and to a higher dimension ``k``?

For each ``k`` (genus panel size) and each group, reports NUDGE's verdict. There is no published
per-genus ε ground truth here, so the confident-WRONG guard is behavioral: a competitive-release
BLOOMER (Escherichia/Shigella — a vancomycin-resistant gram-negative that blooms from ≈0) must NOT
be confidently called a *direct* susceptibility effect; if NUDGE resolves a specific knob for it,
that is flagged LOUD.

Run: ``uv run python scripts/vv/mdsine2_attribution.py <raw_tables_dir> [k ...]``.
Reuses the SHIPPED ``attribute_glv`` unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mdsine2_glv import (  # noqa: E402
    VANCOMYCIN,
    build_mdsine2_dataset,
    load_mdsine2,
)

from nudge.inference.lotka_volterra import attribute_glv, fit_baseline_glv  # noqa: E402

KS = (3, 5, 8, 12)
BLOOMER = "Escherichia/Shigella"


def run_k(raw, k: int, steps: int, seed: int) -> dict:
    ds, labels, _memb = build_mdsine2_dataset(raw, k=k, perturbation=VANCOMYCIN, n_grid=14)
    n_sim = min(ds.reference.shape[0], ds.perturbed.shape[0])
    baseline, base_loss = fit_baseline_glv(ds, steps=max(steps, 300), n_sim=n_sim, seed=seed)

    rb = ds.reference.mean(axis=(0, 1))
    pb = ds.perturbed.mean(axis=(0, 1))
    rows = []
    for g, lab in enumerate(labels):
        res = attribute_glv(ds, baseline=baseline, target=g, steps=steps, n_sim=n_sim, seed=seed)
        f = res.fit
        d_null = f.bic["null"] - min(f.bic[m] for m in ("growth", "interaction", "susceptibility"))
        logfc = float(np.log((pb[g] + 1e-2) / (rb[g] + 1e-2)))
        cw = bool(lab == BLOOMER and res.call not in ("no-change", "unresolved"))
        rows.append({
            "group": lab, "call": res.call, "logFC": round(logfc, 2),
            "selected": f.selected, "d_null_best": round(float(d_null), 2),
            "delta": {m: round(float(v), 3) for m, v in f.delta.items()},
            "cond_number": round(float(f.cond_number), 1),
            "corr_alpha_beta": round(float(f.corr_alpha_beta), 3),
            "degenerate": bool(f.degenerate), "confident_wrong": cw,
            "reason": res.reason,
        })
    return {"k": k, "n_rep": ds.reference.shape[0], "n_timepoints": len(ds.t_obs),
            "pulse_frac": round(float(ds.u_grid.mean()), 2),
            "baseline_loss": round(float(base_loss), 4), "labels": labels, "rows": rows}


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: mdsine2_attribution.py <raw_tables_dir> [k ...]")
    raw_dir = sys.argv[1]
    ks = [int(x) for x in sys.argv[2:]] or list(KS)
    raw = load_mdsine2(raw_dir)
    out = {"dataset": "MDSINE2 Gibson healthy — vancomycin pulse, subjects 2-5", "results": []}
    any_cw = False
    for k in ks:
        print(f"\n{'=' * 70}\nk = {k}\n{'=' * 70}", flush=True)
        r = run_k(raw, k, steps=250, seed=0)
        out["results"].append(r)
        print(f"  baseline loss={r['baseline_loss']}  n_rep={r['n_rep']} n_t={r['n_timepoints']} "
              f"pulse_frac={r['pulse_frac']}", flush=True)
        for row in r["rows"]:
            tag = "  <<< CONFIDENT (check!)" if row["confident_wrong"] else ""
            print(f"  {row['group']:<24} logFC={row['logFC']:+5.2f} -> {row['call']:<12} "
                  f"(ΔBIC_null={row['d_null_best']:+.1f}, sel={row['selected']}, "
                  f"cond={row['cond_number']:.0f}){tag}", flush=True)
            any_cw = any_cw or row["confident_wrong"]
    outpath = Path(__file__).resolve().parent / "mdsine2_attribution_RESULTS.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outpath}")
    print(f"\nCONFIDENT (non-abstain) calls: "
          f"{sum(1 for r in out['results'] for row in r['rows'] if row['call'] in ('growth','interaction','susceptibility'))}")
    print(f"BLOOMER confident-wrong: {'YES (CRITICAL)' if any_cw else 'NONE'}")


if __name__ == "__main__":
    main()
