"""RED-TEAM (round 6, FINAL sweep): a perturbed-only BATCH/DEPTH scale drives the
multi-operating-point covariance breaker ``attribute_lyapunov_multi`` to a CONFIDENT-WRONG
``ceiling`` on a genuine THRESHOLD difference — defeating the LIM-017 best-buffered-pair
corroboration on WELL-BUFFERED points.

This is the "corroboration collusion on well-buffered points" surface that the round-5 final
sweep (``runs/000000017``) flagged plausible-but-UNRUN. It is DISTINCT from LIM-017 (a
NEAR-FOLD point corrupting the joint fit): here EVERY operating point is well-buffered
(``lna_reliable`` ok, proximity low) and the corruption is a *uniform* perturbed-only
multiplicative scale that corrupts ALL points identically — so the two MOST-BUFFERED points
read ``ceiling`` too, the corroboration AGREES, and the confident-wrong call is certified.

Mechanism (the systemic pattern: a guard keyed on the CONTROL, blind to the PERTURBED side):
``fit_lyapunov_multi`` PINS each point's depth ``scale`` from that point's CLEAN WT
(``calibrate_from_wt``) and holds it fixed for the perturbed condition — it ASSUMES WT and
perturbed share one capture/depth scale. A per-condition multiplicative batch/depth factor
``c`` on the PERTURBED cells only (WT clean) violates that: the pinned scale is now wrong by
``c``, and a free-``v_max`` fit absorbs ``c`` into the ceiling (both multiply the ON mode).
The multi-point path has NO OFF-cluster / off_scale analog of the differential module's gate
4c (P4/P5), so nothing catches it. The batch DOES scale the OFF mode too (a genuine ceiling
leaves it at basal), so the confound is DISTINGUISHABLE in principle — the info exists, the
gate does not — hence a genuine confident-wrong, NOT an observational degeneracy.

Ground truth: a genuine THRESHOLD (K x1.6) change (the multi-point breaker resolves it
cleanly at batch 1.0 — see the positive control). A confident bare ``ceiling`` under the
batch is unambiguously WRONG (a real threshold missed + a nonexistent v_max change asserted).

  HELD  = ``attribute_lyapunov_multi`` returns ``unresolved`` (or the true ``threshold``).
  HOLE  = it returns a confident bare ``ceiling`` (!= truth) with the gate passing.

Run: uv run python scripts/redteam/lyapunov_perturbed_batch_ceiling_hole.py [seed ...]
Default seeds 1 2 3 (the reproducing set from the 4-seed sweep; seed 0 HELD). Touches no
src/ code and no fail-safe margins — diagnostic only.
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
# Three WELL-BUFFERED basals (all far from the fold) — the corroboration path is active and
# every point passes lna_reliable, so this is NOT the near-fold LIM-017 case.
BASALS = [0.05, 0.15, 0.30]
# TRUE mechanism: a THRESHOLD (K) shift on target edge 0 (resolves cleanly at batch 1.0).
PARAM, FACTOR, TRUTH = "K", 1.6, "threshold"
# The confound: a perturbed-only batch/depth factor (WT clean). 1.0 = positive control.
CONTROL_BATCH, HOLE_BATCH = 1.0, 2.0


def _activity(adata: object, condition: str, circ: object) -> np.ndarray:
    mask = np.asarray(adata.obs["condition"] == condition)
    return counts_to_activity(adata[mask], circ, MARKERS)


def _points(seed: int, batch: float) -> tuple[list[OperatingPoint], bool]:
    points, all_ok = [], True
    for basal in BASALS:
        circ = toggle(basal=basal)
        adata = generate_toggle_perturbseq(
            circ,
            [PerturbationSpec("cond", "edge", 0, PARAM, FACTOR)],
            readout=DEEP,
            n_cells_per_condition=3000,
            seed=seed,
        )
        wt = _activity(adata, "WT", circ)               # CLEAN control
        cond = _activity(adata, "cond", circ) * batch   # perturbed-only batch scale
        scale, obs = calibrate_from_wt(wt, circ)        # scale pinned from CLEAN wt
        ok, _why = lna_reliable(circ, scale)
        all_ok = all_ok and ok
        points.append(OperatingPoint(data=cond, circuit=circ, scale=scale, obs_sd=obs))
    return points, all_ok


def _one(seed: int, batch: float) -> tuple[str, bool, float, str]:
    points, all_ok = _points(seed, batch)
    label, nlls = attribute_lyapunov_multi(points, target_edge=0, seed=seed)
    if nlls:
        ordered = sorted(nlls.values())
        gap = ordered[1] - ordered[0]
        detail = "  ".join(f"{k}={nlls[k]:.3f}" for k in nlls)
    else:
        gap, detail = float("nan"), "(abstained pre-fit)"
    return label, all_ok, gap, detail


def run(seeds: list[int]) -> int:
    print("=" * 80, flush=True)
    print("RED-TEAM r6: perturbed-only BATCH SCALE -> confident-wrong 'ceiling'")
    print(f"TRUTH = {TRUTH} ({PARAM}x{FACTOR}); a bare 'ceiling' under batch x{HOLE_BATCH} = HOLE")
    print("=" * 80, flush=True)
    holes = 0
    for seed in seeds:
        # positive control: no batch -> should NOT be ceiling.
        c_label, c_ok, c_gap, c_detail = _one(seed, CONTROL_BATCH)
        print(
            f"\nseed={seed} batch={CONTROL_BATCH:.1f} [control]  gate_all_ok={c_ok}  "
            f"label={c_label!r}  gap={c_gap:.4f}",
            flush=True,
        )
        print(f"    NLLs: {c_detail}", flush=True)
        # the confound.
        h_label, h_ok, h_gap, h_detail = _one(seed, HOLE_BATCH)
        is_hole = h_ok and h_label in ("gain", "threshold", "ceiling") and h_label != TRUTH
        holes += int(is_hole)
        flag = "  <== CONFIDENT-WRONG HOLE" if is_hole else ""
        print(
            f"seed={seed} batch={HOLE_BATCH:.1f}          gate_all_ok={h_ok}  "
            f"label={h_label!r}  gap={h_gap:.4f}{flag}",
            flush=True,
        )
        print(f"    NLLs: {h_detail}", flush=True)
    print(f"\n>>> confident-wrong holes: {holes} / {len(seeds)} seeds", flush=True)
    return 0 if holes == 0 else 2


if __name__ == "__main__":
    arg_seeds = [int(s) for s in sys.argv[1:]] or [1, 2, 3]
    raise SystemExit(run(arg_seeds))
