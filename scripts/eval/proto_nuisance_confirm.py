"""Confirmation run for the CORRECTED (earn-based) guard B — a representative reduced set
that exercises both ``check_both`` directions so the reported CALL comes from the code path,
not a hand-recompute. See ``scripts/eval/proto_nuisance_sweep.py`` for the full sweep.
"""

from __future__ import annotations

import time

import jax
import numpy as np

from nudge.circuits import ras_switch_1node
from nudge.inference._proto_nuisance import guard_b_classify
from nudge.inference.differential import (
    Context,
    attribute_differential,
    simulate_context_pair,
)

CIRC = ras_switch_1node(n=6.0, vmax=2.5, K=1.0, basal=0.2)
SCALE, OBS, NC = 25.0, 0.5, 2000
CONF = {"threshold-diff", "gain-diff", "ceiling-diff"}


def _pair(mech, factor, seed):
    return simulate_context_pair(CIRC, mechanism=mech, factor=factor, n_cells=NC,
                                 scale_a=SCALE, scale_b=SCALE, obs_sd=OBS, seed=seed)


def _report(tag, a, b, hole_expected):
    t0 = time.time()
    base = attribute_differential(a, b, CIRC, steps=250, seed=0)
    g = guard_b_classify(a, b, CIRC, winner=base.fit.best_diff, steps=180,
                         check_both=True)
    eb = g.extras.get("B", {})
    ea = g.extras.get("A", {})
    hole = g.call in CONF and hole_expected == "no"
    print(f"{tag:28s} base={base.call:14s} guardB={g.call:14s} knob={g.knob:4s} "
          f"earnB={eb.get('earn', float('nan')):7.1f} earnA={ea.get('earn', float('nan')):7.1f} "
          f"profB={eb.get('prof_ratio', float('nan')):.3f} "
          f"{'<== HOLE' if hole else ''} ({time.time()-t0:.0f}s)", flush=True)
    return g.call


def main() -> int:
    jax.config.update("jax_platform_name", "cpu")
    print("CORRECTED guard B (earn-based decision, check_both) — representative set\n")

    print("--- affine confounds (truth=no-difference; a *-diff = HOLE) ---")
    a, b = _pair("none", 1.0, 0)
    bd = np.asarray(b.data, float)
    holes = 0
    for label, newb in [("mult s=1.18", 1.18 * bd), ("mult s=1.25", 1.25 * bd),
                        ("mult s=1.40", 1.40 * bd), ("add o=3", bd + 3.0),
                        ("add o=5", bd + 5.0), ("mix s=1.3,o=3", 1.3 * bd + 3.0)]:
        call = _report(label, a, Context("B", newb, b.control), "no")
        holes += int(call in CONF)

    print("\n--- positive controls ---")
    pc = {}
    for mech, fac, seed in [("gain", 0.55, 1), ("ceiling", 1.4, 1),
                            ("ceiling", 1.4, 2), ("threshold", 1.4, 1),
                            ("none", 1.0, 1)]:
        a2, b2 = _pair(mech, fac, seed)
        pc[f"{mech}{seed}"] = _report(f"{mech} fac={fac} seed={seed}", a2, b2, "ok")

    print(f"\nconfound confident-wrong (guard B): {holes}/6")
    print(f"positive calls: {pc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
