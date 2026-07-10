"""RED-TEAM (round 5, FINAL sweep): knife-edge probe of the P1/P4 measured separators.

Round 2 taught that a *measured* separator can be a **knife-edge** (LIM-017's useful vs
corrupting operating points sat only 0.007 apart). The P1 fix (gate 4b) picked
``_OFF_SHIFT_INFLATION_MAX = 2.5`` as the midpoint of a gap measured on an INTEGER offset
grid {0,1,2,3,5} (confident-wrong `off_shift` >= 2.99; genuine <= 1.96). The P4 fix
(gate 4c) picked the band upper ``_OFF_SCALE_INFLATION_MAX = 1.30`` from a grid of factors
{1.5,2.0,2.4} (confident-wrong `off_scale` >= 1.43; genuine <= 1.18).

Neither grid probed the *fractional* interior of the gap. This script sweeps a FINE grid of
additive offsets (P1) and multiplicative factors (P4) right at each gate boundary, on a
``mechanism="none"`` context pair (truth = no-difference), and asks the SHIPPED path:

  HOLE  = the shipped ``attribute_differential`` returns a confident ``*-diff`` while the
          triggering statistic is INSIDE the gate's blind band (``off_shift <= 2.5`` for
          gate 4b, ``off_scale <= 1.30`` for gate 4c) — a confound that slips the just-
          shipped measured separator.
  HELD  = the shipped path returns ``no-difference`` / ``unresolved`` (gate fired, or the
          confound is too weak to fake a confident diff).

Run: uv run python scripts/redteam/differential_gate_knifeedge.py [nseeds]
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

# Fine grids that straddle each gate boundary (the integer grids skipped these interiors).
ADDITIVE_OFFSETS = [2.2, 2.4, 2.6, 2.8]  # gate 4b blind band: off_shift <= 2.5
MULT_FACTORS = [1.30, 1.35, 1.40, 1.45]  # gate 4c blind band: off_scale <= 1.30
SCALE = 20.0
N_CELLS = 3000
CONF = {"threshold-diff", "gain-diff", "ceiling-diff"}


def _pair(seed: int) -> tuple[Context, Context]:
    circuit = ras_switch_1node()
    return simulate_context_pair(
        circuit,
        mechanism="none",
        n_cells=N_CELLS,
        scale_a=SCALE,
        scale_b=SCALE,
        obs_sd=0.5,
        seed=seed,
    )


def _attack_additive(ctx_b: Context, offset: float) -> Context:
    return Context(name="B", data=np.asarray(ctx_b.data, float) + offset, control=ctx_b.control)


def _attack_mult(ctx_b: Context, factor: float) -> Context:
    return Context(name="B", data=np.asarray(ctx_b.data, float) * factor, control=ctx_b.control)


def _fit(ctx_a: Context, ctx_b: Context, seed: int):
    return attribute_differential(
        ctx_a, ctx_b, ras_switch_1node(), target_edge=0, k_modes=2, steps=200,
        seed=seed, n_boot=0,
    )


def _margins(fit) -> tuple[float, float, str]:
    d_shared = fit.bic["shared"] - fit.bic[fit.best_diff]
    others = sorted(m for m in ("n", "K", "vmax") if m != fit.best_diff)
    runner = min(others, key=lambda m: fit.bic[m])
    d_runner = fit.bic[runner] - fit.bic[fit.best_diff]
    return d_shared, d_runner, runner


def run(nseeds: int) -> int:
    print("=" * 84, flush=True)
    print("RED-TEAM r5 knife-edge: fractional confounds at the P1/P4 gate boundaries")
    print("truth=no-difference; confident *-diff with statistic INSIDE the blind band = HOLE")
    print("=" * 84, flush=True)
    holes = 0
    for seed in range(nseeds):
        ctx_a, ctx_b = _pair(seed)
        # positive control: no confound must be no-difference/unresolved.
        r0 = _fit(ctx_a, ctx_b, seed)
        print(f"\nseed={seed} [clean control] call={r0.call!r} "
              f"off_shift=({r0.fit.off_shift_a:.2f},{r0.fit.off_shift_b:.2f}) "
              f"off_scale=({r0.fit.off_scale_a:.2f},{r0.fit.off_scale_b:.2f})", flush=True)

        print("  -- P1 additive (gate 4b blind band: max off_shift <= 2.5) --", flush=True)
        for off in ADDITIVE_OFFSETS:
            r = _fit(ctx_a, _attack_additive(ctx_b, off), seed)
            f = r.fit
            osh = max(f.off_shift_a, f.off_shift_b)
            d_s, d_r, runner = _margins(f)
            blind = osh <= 2.5
            hole = r.call in CONF and blind
            holes += int(hole)
            tag = "  <== HOLE (confident diff INSIDE blind band)" if hole else (
                "  (gate would fire: off_shift>2.5)" if not blind else "")
            print(f"    offset={off:>4.2f} call={r.call!r} best={f.best_diff!r} "
                  f"max_off_shift={osh:.2f} dBIC_shared={d_s:.1f} "
                  f"dBIC_runner({runner})={d_r:.1f}{tag}", flush=True)

        print("  -- P4 multiplicative (gate 4c blind band: off_scale in [0.80,1.30]) --",
              flush=True)
        for fac in MULT_FACTORS:
            r = _fit(ctx_a, _attack_mult(ctx_b, fac), seed)
            f = r.fit
            osc = max(f.off_scale_a, f.off_scale_b, key=lambda v: abs(np.log(max(v, 1e-9))))
            d_s, d_r, runner = _margins(f)
            blind = 0.80 <= osc <= 1.30
            hole = r.call in CONF and blind
            holes += int(hole)
            tag = "  <== HOLE (confident diff INSIDE blind band)" if hole else (
                "  (gate would fire: off_scale outside band)" if not blind else "")
            print(f"    factor={fac:>4.2f} call={r.call!r} best={f.best_diff!r} "
                  f"off_scale={osc:.2f} dBIC_shared={d_s:.1f} "
                  f"dBIC_runner({runner})={d_r:.1f}{tag}", flush=True)

    print("\n" + "=" * 84, flush=True)
    print(f"confident-wrong *-diff INSIDE a gate blind band (HOLES): {holes}", flush=True)
    return 0 if holes == 0 else 2


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    raise SystemExit(run(n))
