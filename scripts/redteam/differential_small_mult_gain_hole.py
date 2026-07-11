"""RED-TEAM (round 5, FINAL sweep): a SMALL uniform multiplicative confound fakes a
confident ``gain-diff`` / ``ceiling-diff`` that slips BOTH differential gate 4b and 4c.

**The systemic pattern, one more time.** A per-context multiplicative measurement scale on
ONE context's PERTURBED cells (control clean) is invisible to the control-keyed depth guard
(gate 2). The P4 fix added gate 4c for exactly this — BUT gate 4c is **scoped to the vmax
(ceiling) winner** and was calibrated on LARGE factors (c >= 1.5, which BIC reads as ceiling,
``off_scale >= 1.43``). It never tested SMALL factors.

At a small magnitude (c ~ 1.15..1.25) a uniform multiplicative scale is BIC-assigned to the
**gain (n)** channel (it slightly compresses the modes' relative separation) — a channel gate
4c NEVER checks (ceiling-scoped) — and it keeps the additive OFF-baseline shift ``off_shift``
~ 1 (gate 4b, keyed on TRANSLATION, is blind to a multiplicative scale that leaves the near-
zero OFF baseline near-zero). So a confident spurious ``gain-diff`` sails through both new
gates. Near c ~ 1.25 the winner flips to ``vmax`` but ``off_scale`` can still land <= 1.30
(gate 4c's blind band) → a confident spurious ``ceiling-diff`` too.

This is NOT the documented above-median-only evader (that construction is observationally
identical to a genuine ceiling). A UNIFORM scale multiplies the OFF cluster too
(``off_scale = c > 1``), so it is distinguishable IN PRINCIPLE from a genuine gain change
(which leaves ``off_scale ~ 1``) — but gate 4c only consults ``off_scale`` for a *ceiling*
winner, never for the gain winner. Truth = no-difference; any confident ``*-diff`` is the hole.

  HOLE = the SHIPPED ``attribute_differential`` returns ``gain-diff`` / ``threshold-diff`` /
         ``ceiling-diff`` (all gates passed) while truth is no-difference.

Run: uv run python scripts/redteam/differential_small_mult_gain_hole.py [nseeds]
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

FACTORS = [1.15, 1.20, 1.25]
SCALE = 20.0
N_CELLS = 3000
CONF = {"threshold-diff", "gain-diff", "ceiling-diff"}


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
    print("=" * 88, flush=True)
    print("RED-TEAM r5: small UNIFORM multiplicative confound -> confident gain-/ceiling-diff")
    print("truth=no-difference; ANY confident *-diff through the shipped gates = HOLE")
    print("=" * 88, flush=True)
    holes = 0
    per_seed = {}
    for seed in range(nseeds):
        seed_holes = 0
        for fac in FACTORS:
            r = _run(seed, fac)
            f = r.fit
            osh = max(f.off_shift_a, f.off_shift_b)
            osc = max(f.off_scale_a, f.off_scale_b, key=lambda v: abs(np.log(max(v, 1e-9))))
            others = sorted(m for m in ("n", "K", "vmax") if m != f.best_diff)
            runner = min(others, key=lambda m: f.bic[m])
            d_runner = f.bic[runner] - f.bic[f.best_diff]
            d_shared = f.bic["shared"] - f.bic[f.best_diff]
            hole = r.call in CONF
            holes += int(hole)
            seed_holes += int(hole)
            if hole:
                if f.best_diff == "vmax":
                    why = f"gate 4c blind: off_scale={osc:.3f}<=1.30 (band upper)"
                else:
                    why = (f"gate 4c ceiling-scoped, winner={f.best_diff!r}; "
                           f"gate 4b blind: off_shift={osh:.2f}<=2.5")
            else:
                why = ""
            tag = f"  <== HOLE ({why})" if hole else ""
            print(f"seed={seed} factor={fac:.2f} call={r.call!r} best={f.best_diff!r} "
                  f"off_shift={osh:.2f} off_scale={osc:.3f} "
                  f"dBIC_shared={d_shared:.1f} dBIC_runner({runner})={d_runner:.1f}{tag}",
                  flush=True)
        per_seed[seed] = seed_holes
    seeds_with_holes = sum(1 for v in per_seed.values() if v > 0)
    print("\n" + "=" * 88, flush=True)
    print(f"confident *-diff on truth=no-difference (HOLES): {holes} "
          f"across {seeds_with_holes}/{nseeds} seeds", flush=True)
    return 0 if holes == 0 else 2


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    raise SystemExit(run(n))
