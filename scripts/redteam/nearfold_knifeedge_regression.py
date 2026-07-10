"""RED-TEAM (round 2) regression on the NUDGE-LIM-017 FIX: is the well-buffered margin
a knife-edge?

Round 1 found HOLE 1: a near-fold 3rd operating point (toggle basal=0.60) flipped a true
CEILING to a confident THRESHOLD in ``attribute_lyapunov_multi``. The fix (NUDGE-LIM-017)
gates the joint fit on ``bifurcation_proximity(p.circuit).proximity <= well_buffered_margin``
(default 0.15): it ABSTAINS if any operating point's deterministic proximity dial exceeds
0.15. basal=0.60 has proximity 0.231, so it is now gated out.

This regression probes JUST BELOW the gate. Proximity is continuous and monotone in the
toggle's basal (measured): it crosses 0.15 near basal≈0.41. A 3rd operating point at
basal=0.40 has proximity≈0.146 — it PASSES the gate. Question: is its covariance already
biased enough to still flip the true CEILING to THRESHOLD? If so, the fix is a knife-edge
and the confident-wrong reappears immediately below the margin.

For each 3rd-point basal we print every point's proximity + the gate decision, the joint
NLLs, the resolved gap, and the label. Clean 2-point subset ({0.05, 0.30}) resolves the
true CEILING (positive control). A THRESHOLD at a basal whose proximity <= 0.15 (gate
PASSES) is a knife-edge regression hole; an ``unresolved`` at basal > gate is the fix
working (positive control for the gate).

Run: uv run python scripts/redteam/nearfold_knifeedge_regression.py [nseeds]
Touches no src/ code and no fail-safe margins — diagnostic only.
"""

from __future__ import annotations

import sys

import numpy as np

from nudge.circuits import toggle
from nudge.data.stochastic import generate_toggle_perturbseq
from nudge.data.synthetic import PerturbationSpec
from nudge.inference.bifurcation import bifurcation_proximity
from nudge.inference.bridge import counts_to_activity
from nudge.inference.lyapunov import (
    OperatingPoint,
    attribute_lyapunov_multi,
    calibrate_from_wt,
    lna_reliable,
)
from nudge.mechanisms.readout import Readout

MARKERS = {"A": ["A"], "B": ["B"]}
DEEP = Readout.identity(2, scale=15.0)
CLEAN = [0.05, 0.30]  # well-buffered points (proximity 0.039, 0.112)
# 3rd-point basals to probe: 0.40 PASSES the 0.15 gate (prox≈0.146); 0.42/0.44 are gated.
THIRD = [0.40, 0.42, 0.44]
PARAM, FACTOR, TRUTH = "vmax", 0.6, "ceiling"  # a TRUE ceiling knockdown


def _activity(adata: object, condition: str, circ: object) -> np.ndarray:
    mask = np.asarray(adata.obs["condition"] == condition)
    return counts_to_activity(adata[mask], circ, MARKERS)


def _point(basal: float, seed: int) -> tuple[OperatingPoint, float]:
    circ = toggle(basal=basal)
    adata = generate_toggle_perturbseq(
        circ, [PerturbationSpec("cond", "edge", 0, PARAM, FACTOR)],
        readout=DEEP, n_cells_per_condition=3000, seed=seed,
    )
    wt = _activity(adata, "WT", circ)
    cond = _activity(adata, "cond", circ)
    scale, obs = calibrate_from_wt(wt, circ)
    score = bifurcation_proximity(circ)
    prox = float("nan") if score is None else score.proximity
    return OperatingPoint(data=cond, circuit=circ, scale=scale, obs_sd=obs), prox


def run(nseeds: int) -> int:
    print("=" * 80, flush=True)
    print("RED-TEAM r2: NUDGE-LIM-017 knife-edge regression "
          "(3rd point just below the 0.15 gate)", flush=True)
    print(f"TRUTH = {TRUTH} ({PARAM}x{FACTOR}); well_buffered_margin=0.15; "
          "resolve_margin=0.03", flush=True)
    print("=" * 80, flush=True)
    holes = 0
    for seed in range(nseeds):
        clean_pts = [_point(b, seed) for b in CLEAN]
        # positive control: the clean 2-point subset must resolve the true ceiling.
        pts2 = [p for p, _ in clean_pts]
        label2, _ = attribute_lyapunov_multi(pts2, target_edge=0, seed=seed)
        print(f"\nseed={seed}  clean 2-pt subset -> {label2!r} (expect {TRUTH})", flush=True)
        for third in THIRD:
            p3, prox3 = _point(third, seed)
            pts = pts2 + [p3]
            gate_ok = prox3 <= 0.15
            label, nlls = attribute_lyapunov_multi(pts, target_edge=0, seed=seed)
            if nlls:
                ordered = sorted(nlls.values())
                gap = ordered[1] - ordered[0]
                detail = "  ".join(f"{k}={nlls[k]:.3f}" for k in nlls)
            else:
                gap, detail = float("nan"), "(abstained at the well-buffered gate)"
            is_hole = (
                gate_ok and label in ("gain", "threshold", "ceiling") and label != TRUTH
            )
            holes += int(is_hole)
            flag = "  <== KNIFE-EDGE HOLE (gate passed, wrong call)" if is_hole else ""
            lr = [round(bifurcation_proximity(p.circuit).proximity, 3) for p in pts]
            lna = [lna_reliable(p.circuit, p.scale)[0] for p in pts]
            print(f"  3rd basal={third}  prox3={prox3:.3f}  gate_pass={gate_ok}  "
                  f"label={label!r}  gap={gap:.4f}{flag}", flush=True)
            print(f"      proximities={lr}  lna_reliable={lna}", flush=True)
            print(f"      NLLs: {detail}", flush=True)
    print(f"\n>>> knife-edge regression holes: {holes}", flush=True)
    return holes


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    run(ns)
