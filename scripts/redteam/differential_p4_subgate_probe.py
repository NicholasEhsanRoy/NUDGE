"""RED-TEAM (round 5, FINAL sweep): does a SMALL multiplicative confound slip P4 gate 4c?

The P4 fix set gate 4c's upper band to ``_OFF_SCALE_INFLATION_MAX = 1.30`` from a measured
separator that tested confound factors {1.5, 2.0, 2.4} only (all giving ``off_scale >= 1.43``)
vs genuine ceilings (``off_scale <= 1.18``). The knife-edge probe found that a factor of
**1.30** already produces a CONFIDENT ceiling at ``off_scale ~ 1.33`` — i.e. confident
confounds exist with ``off_scale`` well below the measured 1.43, filling the supposed gap.

This probe sweeps SMALL factors (1.10 .. 1.30) on a ``mechanism="none"`` context pair and
asks the SHIPPED path whether any gives a confident ``ceiling-diff`` while ``off_scale`` is
INSIDE the blind band ``[0.80, 1.30]`` — a batch confound the gate lets through.

A batch factor ``c`` scales the WHOLE perturbed distribution (ON mode AND OFF-cluster spread)
by ``c``, so it is NOT observationally identical to a genuine ceiling (which leaves the OFF
spread anchored, ``off_scale ~ 1``). Truth is no-difference; a confident ``ceiling-diff`` with
``off_scale <= 1.30`` is a genuine confident-wrong slipping gate 4c.

Run: uv run python scripts/redteam/differential_p4_subgate_probe.py [nseeds]
Touches no src/ code and no fail-safe margins — diagnostic only.
"""

from __future__ import annotations

import sys

import numpy as np

from nudge.circuits import ras_switch_1node
from nudge.inference.differential import (
    Context,
    attribute_differential,
    simulate_context_pair,
)

FACTORS = [1.10, 1.15, 1.20, 1.25, 1.30]  # small confounds under the measured 1.5
SCALE = 20.0
N_CELLS = 3000


def _run(seed: int, factor: float):
    circuit = ras_switch_1node()
    ctx_a, ctx_b = simulate_context_pair(
        circuit, mechanism="none", n_cells=N_CELLS,
        scale_a=SCALE, scale_b=SCALE, obs_sd=0.5, seed=seed,
    )
    ctx_b = Context(name="B", data=np.asarray(ctx_b.data, float) * factor, control=ctx_b.control)
    return attribute_differential(
        ctx_a, ctx_b, circuit, target_edge=0, k_modes=2, steps=200, seed=seed, n_boot=0,
    )


def run(nseeds: int) -> int:
    print("=" * 84, flush=True)
    print("RED-TEAM r5: SMALL multiplicative confounds vs P4 gate 4c (band upper 1.30)")
    print("truth=no-difference; confident ceiling-diff with off_scale<=1.30 = HOLE")
    print("=" * 84, flush=True)
    holes = 0
    for seed in range(nseeds):
        for fac in FACTORS:
            r = _run(seed, fac)
            f = r.fit
            osc = max(f.off_scale_a, f.off_scale_b, key=lambda v: abs(np.log(max(v, 1e-9))))
            d_shared = f.bic["shared"] - f.bic[f.best_diff]
            others = sorted(m for m in ("n", "K", "vmax") if m != f.best_diff)
            runner = min(others, key=lambda m: f.bic[m])
            d_runner = f.bic[runner] - f.bic[f.best_diff]
            blind = 0.80 <= osc <= 1.30
            hole = r.call == "ceiling-diff" and blind
            holes += int(hole)
            tag = "  <== HOLE (confident ceiling INSIDE blind band)" if hole else (
                "  (gate fires: off_scale>1.30)" if osc > 1.30 else "")
            print(f"seed={seed} factor={fac:.2f} call={r.call!r} best={f.best_diff!r} "
                  f"off_scale={osc:.3f} dBIC_shared={d_shared:.1f} "
                  f"dBIC_runner({runner})={d_runner:.1f}{tag}", flush=True)
    print("\n" + "=" * 84, flush=True)
    print(f"confident ceiling-diff INSIDE the blind band (HOLES): {holes}", flush=True)
    return 0 if holes == 0 else 2


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    raise SystemExit(run(n))
