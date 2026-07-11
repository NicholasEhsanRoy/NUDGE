"""RED-TEAM (P5, TEST-regime companion to differential_small_mult_gain_hole.py): the SAME
small uniform multiplicative perturbed-only confound, but in the RESOLVABLE-OFF (basal=0.2)
regime the test suite uses — where the confound is BIC-assigned to the ceiling channel and
lands at off_scale INSIDE the genuine range (the P4 off_scale gate is blind). The P5 fix must
abstain here via the ceiling-MAGNITUDE gate. Truth = no-difference; any confident *-diff = HOLE.

Run: uv run python scripts/redteam/differential_small_mult_testregime.py [nseeds]
Touches no src/ code — diagnostic only.
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

# The resolvable-OFF (basal=0.2) regime the differential test suite validates in.
CIRC = ras_switch_1node(n=6.0, vmax=2.5, K=1.0, basal=0.2)
FACTORS = [1.15, 1.20, 1.25]
SCALE = 25.0
N_CELLS = 2000
CONF = {"threshold-diff", "gain-diff", "ceiling-diff"}


def _run(seed: int, factor: float):
    ctx_a, ctx_b = simulate_context_pair(
        CIRC, mechanism="none", n_cells=N_CELLS,
        scale_a=SCALE, scale_b=SCALE, obs_sd=0.5, seed=seed,
    )
    ctx_b = Context(name="B", data=np.asarray(ctx_b.data, float) * factor, control=ctx_b.control)
    return attribute_differential(
        ctx_a, ctx_b, CIRC, target_edge=0, k_modes=2, steps=250, seed=seed, n_boot=0,
    )


def run(nseeds: int) -> int:
    print("=" * 88, flush=True)
    print("RED-TEAM P5 (TEST regime, basal=0.2): small UNIFORM multiplicative confound")
    print("truth=no-difference; ANY confident *-diff through the shipped gates = HOLE")
    print("=" * 88, flush=True)
    holes = 0
    per_seed = {}
    for seed in range(nseeds):
        seed_holes = 0
        for fac in FACTORS:
            r = _run(seed, fac)
            f = r.fit
            osc = max(f.off_scale_a, f.off_scale_b, key=lambda v: abs(np.log(max(v, 1e-9))))
            resv = min(f.off_resolvability_a, f.off_resolvability_b)
            hole = r.call in CONF
            holes += int(hole)
            seed_holes += int(hole)
            tag = "  <== HOLE" if hole else ""
            print(f"seed={seed} factor={fac:.2f} call={r.call!r} best={f.best_diff!r} "
                  f"off_scale={osc:.3f} resolvability={resv:.3f} "
                  f"|log2|={abs(f.log2_ratio):.3f}{tag}", flush=True)
        per_seed[seed] = seed_holes
    seeds_with_holes = sum(1 for v in per_seed.values() if v > 0)
    print("\n" + "=" * 88, flush=True)
    print(f"confident *-diff on truth=no-difference (HOLES): {holes} "
          f"across {seeds_with_holes}/{nseeds} seeds", flush=True)
    return 0 if holes == 0 else 2


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    raise SystemExit(run(n))
