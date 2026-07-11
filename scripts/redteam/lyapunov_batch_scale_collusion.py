"""RED-TEAM (round 5, FINAL sweep): a perturbed-only BATCH SCALE collides the LIM-017
best-buffered-pair corroboration in ``attribute_lyapunov_multi``.

Priority-3 probe (round-3 flagged, never reached): engineer the corruption so it lands on
the TWO MOST-BUFFERED operating points as well, so the best-buffered-pair corroboration
cannot catch it.

The vehicle is the scale⇄v_max degeneracy the differential module gates (P4) but the
covariance multi-fit does NOT. ``fit_lyapunov_multi`` PINS each point's ``scale`` from that
point's own WT (``calibrate_from_wt``) and holds it fixed for the perturbed condition — it
ASSUMES WT and perturbed share one capture/depth scale. A per-condition multiplicative
batch/depth factor ``c`` on the PERTURBED cells only (WT clean) violates that: the pinned
scale is now wrong by ``c``, and a free-``v_max`` fit absorbs ``c`` into the ceiling (both
multiply the ON mode). Because EVERY operating point carries the same ``c``, the two
most-buffered points read ceiling too — so the corroboration AGREES and certifies it.

Ground truth here is a genuine **threshold** (``K``) change (which STATE.md/FINDINGS report
the multi-point breaker resolves cleanly, 3/3). A confident ``ceiling`` under the batch is
then unambiguously WRONG (a real threshold shift missed + a ceiling asserted from an
artifact), NOT the pure scale⇄v_max degeneracy (the modes are threshold-shifted, so the data
is not identical to a pure genuine ceiling). All points are well-buffered (low proximity),
so this is the CORROBORATION-COLLUSION case, not the near-fold case.

  HELD  = the shipped ``attribute_lyapunov_multi`` returns ``unresolved`` (or the true
          ``threshold``) under the batch.
  HOLE  = it returns a confident bare ``ceiling`` (!= truth) with the gate passing and the
          best-buffered pair corroborating.

Run: uv run python scripts/redteam/lyapunov_batch_scale_collusion.py [nseeds]
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
# Three WELL-BUFFERED basals (all far from the fold) — the corroboration path is active.
BASALS = [0.05, 0.15, 0.30]
# TRUE mechanism: a THRESHOLD (K) shift on target edge 0 (resolves cleanly clean).
PARAM, FACTOR, TRUTH = "K", 1.6, "threshold"
# Perturbed-only batch/depth factors (WT clean). 1.0 = positive control.
BATCH_FACTORS = [1.0, 1.5, 2.0]


def _activity(adata: object, condition: str, circ: object) -> np.ndarray:
    mask = np.asarray(adata.obs["condition"] == condition)
    return counts_to_activity(adata[mask], circ, MARKERS)


def _points(seed: int, batch: float) -> tuple[list[OperatingPoint], list[float]]:
    points, prox_info = [], []
    for basal in BASALS:
        circ = toggle(basal=basal)
        adata = generate_toggle_perturbseq(
            circ,
            [PerturbationSpec("cond", "edge", 0, PARAM, FACTOR)],
            readout=DEEP,
            n_cells_per_condition=3000,
            seed=seed,
        )
        wt = _activity(adata, "WT", circ)          # CLEAN control
        cond = _activity(adata, "cond", circ) * batch  # perturbed-only batch scale
        scale, obs = calibrate_from_wt(wt, circ)   # scale pinned from CLEAN wt
        ok, _why = lna_reliable(circ, scale)
        prox_info.append(1.0 if ok else 0.0)
        points.append(OperatingPoint(data=cond, circuit=circ, scale=scale, obs_sd=obs))
    return points, prox_info


def run(nseeds: int) -> int:
    print("=" * 80, flush=True)
    print("RED-TEAM r5: perturbed-only BATCH SCALE vs best-buffered-pair corroboration")
    print(f"TRUTH = {TRUTH} ({PARAM}x{FACTOR}); a confident 'ceiling' under batch = HOLE")
    print("=" * 80, flush=True)
    holes = 0
    for seed in range(nseeds):
        for batch in BATCH_FACTORS:
            points, oks = _points(seed, batch)
            all_ok = all(v == 1.0 for v in oks)
            label, nlls = attribute_lyapunov_multi(points, target_edge=0, seed=seed)
            if nlls:
                ordered = sorted(nlls.values())
                gap = ordered[1] - ordered[0]
                detail = "  ".join(f"{k}={nlls[k]:.3f}" for k in nlls)
            else:
                gap, detail = float("nan"), "(abstained pre-fit)"
            is_hole = (
                all_ok
                and label in ("gain", "threshold", "ceiling")
                and label != TRUTH
            )
            flag = "  <== CONFIDENT-WRONG HOLE" if is_hole else ""
            holes += int(is_hole)
            ctrl = "  [positive control]" if batch == 1.0 else ""
            print(
                f"\nseed={seed} batch={batch:.1f}{ctrl}  gate_all_ok={all_ok}  "
                f"label={label!r}  gap={gap:.4f}{flag}",
                flush=True,
            )
            print(f"    NLLs: {detail}", flush=True)
    print(f"\n>>> confident-wrong holes: {holes}", flush=True)
    return 0 if holes == 0 else 2


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    raise SystemExit(run(ns))
