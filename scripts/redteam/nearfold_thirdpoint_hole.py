"""RED-TEAM repro: a NEAR-FOLD 3rd operating point flips a TRUE CEILING → threshold.

This reproduces + formalizes the hole first seen in
``scripts/analysis/toggle_gain_abstention_probe_RESULTS.txt`` (ceiling, pts=3, gap 0.30).

The claim under attack: ``attribute_lyapunov_multi`` "abstains loudly unless EVERY
operating point's LNA is trustworthy (one bad Gaussian corrupts the joint fit)" — the
guard is ``all(lna_reliable(p.circuit, p.scale))``.

The attack: build a TRUE CEILING perturbation seen at three basal operating points where
the 3rd (basal≈0.60) sits AGGRESSIVELY CLOSE to the toggle's fold but its LNA lobes have
NOT yet overlapped enough to trip ``lna_reliable`` (sep_ratio=1.0). That near-fold point's
LNA moments are corrupted (the Lyapunov covariance is swelling but still "reliable"),
which biases the shared-parameter joint fit so that a shared-K ("threshold") explanation
beats the true shared-vmax ("ceiling") by MORE than resolve_margin=0.03 → a CONFIDENT,
SPECIFIC, WRONG ``threshold`` call. The 2-point fit (clean, well-buffered points only)
gets it right (``ceiling``).

Prints, for each n_pts∈{1,2,3}: the lna_reliable verdict of EVERY point (to show the gate
PASSED — this is not an abstention), the joint NLLs, the resolved-channel gap, and the
label. A confident ``threshold`` at pts=3 with the gate passing is the hole.

Run: uv run python scripts/redteam/nearfold_thirdpoint_hole.py [nseeds]
Touches no src/ code and no fail-safe margins — diagnostic only.
"""

from __future__ import annotations

import sys

import numpy as np

from nudge.circuits import toggle
from nudge.data.stochastic import generate_toggle_perturbseq
from nudge.data.synthetic import PerturbationSpec
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
# basal 0.60 is the aggressive near-fold 3rd point; 0.05 / 0.30 are the clean points.
BASALS = [0.05, 0.30, 0.60]
# TRUE mechanism: a CEILING knockdown (vmax ×0.6) on target edge 0.
PARAM, FACTOR, TRUTH = "vmax", 0.6, "ceiling"


def _activity(adata: object, condition: str, circ: object) -> np.ndarray:
    mask = np.asarray(adata.obs["condition"] == condition)
    return counts_to_activity(adata[mask], circ, MARKERS)


def run(nseeds: int) -> int:
    print("=" * 78, flush=True)
    print("RED-TEAM: near-fold 3rd operating point corrupts the shared-K joint fit",
          flush=True)
    print(f"TRUTH = {TRUTH} ({PARAM}×{FACTOR}); resolve_margin=0.03", flush=True)
    print("=" * 78, flush=True)
    holes = 0
    for seed in range(nseeds):
        for n_pts in (1, 2, 3):
            points = []
            for basal in BASALS[:n_pts]:
                circ = toggle(basal=basal)
                adata = generate_toggle_perturbseq(
                    circ,
                    [PerturbationSpec("cond", "edge", 0, PARAM, FACTOR)],
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
            gates = [
                (b, *lna_reliable(p.circuit, p.scale))
                for b, p in zip(BASALS[:n_pts], points, strict=True)
            ]
            all_ok = all(ok for _b, ok, _r in gates)
            label, nlls = attribute_lyapunov_multi(points, target_edge=0, seed=seed)
            if nlls:
                ordered = sorted(nlls.values())
                gap = ordered[1] - ordered[0]
                detail = "  ".join(f"{k}={nlls[k]:.3f}" for k in nlls)
            else:
                gap, detail = float("nan"), "(abstained pre-fit)"
            is_hole = (
                all_ok
                and label not in ("unresolved", TRUTH)
                and label in ("gain", "threshold", "ceiling")
            )
            flag = "  <== CONFIDENT-WRONG HOLE" if is_hole else ""
            holes += int(is_hole)
            print(
                f"\nseed={seed} pts={n_pts}  gate_all_ok={all_ok}  "
                f"label={label!r}  gap={gap:.4f}{flag}",
                flush=True,
            )
            for b, ok, why in gates:
                print(f"    basal={b:<5} lna_reliable={ok}  ({why})", flush=True)
            print(f"    NLLs: {detail}", flush=True)
    print(f"\n>>> confident-wrong holes: {holes}", flush=True)
    return holes


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    run(ns)
