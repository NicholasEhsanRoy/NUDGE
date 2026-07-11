"""P5 UQ-fixer measurement: (1) independently REPRODUCE the hole through the shipped
``attribute_differential``, and (2) MEASURE the free-affine earn-guard (the prototyped
principled fix, ``_proto_nuisance.guard_b_classify``) on the SAME interior + positive
controls. One run gives the validate + the fix-feasibility numbers for FINDINGS §P5.

Run: uv run python scripts/vv/p5_measure.py [nseeds]
"""

from __future__ import annotations

import sys

import numpy as np

from nudge.circuits import ras_switch_1node
from nudge.inference._proto_nuisance import guard_b_classify
from nudge.inference.differential import (
    Context,
    attribute_differential,
    simulate_context_pair,
)

CONF = {"threshold-diff", "gain-diff", "ceiling-diff"}
SCALE = 20.0
NCELLS = 3000
OBS = 0.5


def _osc(fit) -> float:
    return max(fit.off_scale_a, fit.off_scale_b, key=lambda v: abs(np.log(max(v, 1e-9))))


def _confound_pair(seed: int, factor: float):
    circ = ras_switch_1node()
    a, b = simulate_context_pair(
        circ, mechanism="none", n_cells=NCELLS,
        scale_a=SCALE, scale_b=SCALE, obs_sd=OBS, seed=seed,
    )
    b = Context(name="B", data=np.asarray(b.data, float) * factor, control=b.control)
    return circ, a, b


def _genuine_pair(seed: int, mechanism: str, factor: float):
    circ = ras_switch_1node()
    a, b = simulate_context_pair(
        circ, mechanism=mechanism, factor=factor, n_cells=NCELLS,
        scale_a=SCALE, scale_b=SCALE, obs_sd=OBS, seed=seed,
    )
    return circ, a, b


def run(nseeds: int) -> int:
    print("=" * 100, flush=True)
    print("P5 measurement — shipped attribute_differential (HOLE) vs earn-guard (FIX)")
    print("=" * 100, flush=True)

    n_hole_shipped = 0
    n_hole_earn = 0
    print("\n### CONFOUND (truth = no-difference); ANY *-diff is a confident-wrong ###")
    print(f"{'seed':>4} {'c':>5} | {'shipped':>13} {'best':>5} {'off_scale':>9} | "
          f"{'earn_call':>13} {'earn':>8} {'side':>4}")
    for seed in range(nseeds):
        for factor in (1.15, 1.20, 1.25):
            circ, a, b = _confound_pair(seed, factor)
            res = attribute_differential(a, b, circ, target_edge=0, k_modes=2,
                                         steps=200, seed=seed, n_boot=0)
            osc = _osc(res.fit)
            g = guard_b_classify(a, b, circ, winner=res.fit.best_diff, k_modes=2,
                                 steps=150, lr=0.05, earn_margin=6.0)
            hs = res.call in CONF
            he = g.call in CONF
            n_hole_shipped += int(hs)
            n_hole_earn += int(he)
            flag_s = " <HOLE" if hs else ""
            flag_e = " <HOLE" if he else " ok"
            side = g.extras and (list(g.extras.keys())[-1]) or "?"
            print(f"{seed:>4} {factor:>5.2f} | {res.call:>13} {res.fit.best_diff:>5} "
                  f"{osc:>9.3f}{flag_s:>0} | {g.call:>13} {g.earn_bic:>8.1f} {side:>4}{flag_e}",
                  flush=True)

    print("\n### POSITIVE CONTROLS (earn-guard must NOT over-abstain) ###")
    print(f"{'seed':>4} {'mech':>10} {'fac':>5} | {'shipped':>13} {'best':>5} | "
          f"{'earn_call':>13} {'earn':>8}")
    controls = [
        ("gain", 0.55), ("ceiling", 1.4), ("ceiling", 2.0), ("none", 1.0),
    ]
    over_abstain = 0
    for seed in range(nseeds):
        for mech, factor in controls:
            circ, a, b = _genuine_pair(seed, mech, factor)
            res = attribute_differential(a, b, circ, target_edge=0, k_modes=2,
                                         steps=200, seed=seed, n_boot=0)
            g = guard_b_classify(a, b, circ, winner=res.fit.best_diff, k_modes=2,
                                 steps=150, lr=0.05, earn_margin=6.0)
            # over-abstain = shipped resolves a genuine mech but earn-guard kills it
            expect_pos = mech in ("gain", "ceiling")
            if expect_pos and res.call in CONF and g.call not in CONF:
                over_abstain += 1
                tag = " <OVER-ABSTAIN"
            else:
                tag = ""
            print(f"{seed:>4} {mech:>10} {factor:>5.2f} | {res.call:>13} "
                  f"{res.fit.best_diff:>5} | {g.call:>13} {g.earn_bic:>8.1f}{tag}",
                  flush=True)

    print("\n" + "=" * 100, flush=True)
    print(f"SHIPPED confound confident-wrong (HOLE): {n_hole_shipped}", flush=True)
    print(f"EARN-GUARD confound confident-wrong:     {n_hole_earn}", flush=True)
    print(f"EARN-GUARD over-abstentions on genuine:  {over_abstain}", flush=True)
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    raise SystemExit(run(n))
