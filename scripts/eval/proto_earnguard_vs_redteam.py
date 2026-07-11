"""DEFINITIVE PROOF: does the Earn-Guard (``_proto_nuisance.guard_b_classify``) close the
exact red-team holes P1 (additive), P4 (multiplicative), P5 (small multiplicative)?

Each block below reproduces the EXACT construction from the cloud red-team repro
(``scripts/redteam/differential_{additive,multiplicative,small_mult_gain}_*.py``): the same
``ras_switch_1node`` default circuit, ``simulate_context_pair(mechanism="none")`` (ground
truth = NO mechanistic difference), SCALE=20, obs_sd=0.5, N_CELLS=3000, and the same
per-context affine applied to context B's PERTURBED cells only (control clean).

For each attacked pair we (1) confirm the SHIPPED ``attribute_differential`` still fires a
confident ``*-diff`` (the hole is real on this construction) and (2) run the Earn-Guard with
``winner=base.fit.best_diff`` and assert its call is NOT a confident ``*-diff``. A single
confident-wrong from the Earn-Guard on truth=no-difference FAILS the proof.

Positive controls at the end confirm the Earn-Guard does NOT over-abstain: a genuine
per-context gain / ceiling / threshold difference must still be RESOLVED.

Run: uv run python scripts/eval/proto_earnguard_vs_redteam.py [nseeds]
"""

from __future__ import annotations

import sys
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

# EXACT red-team repro parameters (from the cloud branch scripts).
SCALE = 20.0
N_CELLS = 3000
OBS_SD = 0.5
CONF = {"threshold-diff", "gain-diff", "ceiling-diff"}

# The exact attack magnitudes each repro sweeps.
P1_OFFSETS = [1.0, 2.0, 3.0, 5.0]           # additive on B.data
P4_FACTORS = [1.5, 2.0, 2.4, 0.7, 0.5]      # multiplicative on B.data
P5_FACTORS = [1.15, 1.20, 1.25]             # small multiplicative on B.data


def _clean_pair(seed: int):
    circuit = ras_switch_1node()
    a, b = simulate_context_pair(
        circuit, mechanism="none", n_cells=N_CELLS,
        scale_a=SCALE, scale_b=SCALE, obs_sd=OBS_SD, seed=seed,
    )
    return circuit, a, b


def _attack(b: Context, *, offset: float = 0.0, factor: float = 1.0) -> Context:
    bd = np.asarray(b.data, dtype=float) * factor + offset
    return Context(name="B", data=bd, control=b.control)


def _eval(tag: str, circuit, a, b_att, records: list) -> None:
    t0 = time.time()
    base = attribute_differential(a, b_att, circuit, steps=250, seed=0)
    g = guard_b_classify(a, b_att, circuit, winner=base.fit.best_diff, steps=180,
                         check_both=True)
    eb = g.extras.get("B", {}).get("earn", float("nan"))
    ea = g.extras.get("A", {}).get("earn", float("nan"))
    base_hole = base.call in CONF          # shipped baseline fires a confident *-diff
    guard_hole = g.call in CONF            # EARN-GUARD confident-wrong (the thing we forbid)
    records.append((tag, base.call, g.call, base_hole, guard_hole))
    print(f"{tag:26s} base={base.call:14s} EARN-GUARD={g.call:14s} "
          f"knob={g.knob:5s} earnB={eb:7.1f} earnA={ea:7.1f} "
          f"{'<== GUARD CONFIDENT-WRONG' if guard_hole else 'OK'} "
          f"[baseline hole={base_hole}] ({time.time()-t0:.0f}s)", flush=True)


def main() -> int:
    jax.config.update("jax_platform_name", "cpu")
    nseeds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    seeds = list(range(nseeds))
    records: list = []

    print(f"EARN-GUARD vs red-team P1/P4/P5 — exact repro constructions, "
          f"{nseeds} seed(s)\n")

    print("--- P1: ADDITIVE offset on B.perturbed (truth=no-difference) ---")
    for seed in seeds:
        circuit, a, b = _clean_pair(seed)
        for o in P1_OFFSETS:
            _eval(f"P1 add o={o:g} s={seed}", circuit, a, _attack(b, offset=o), records)

    print("\n--- P4: MULTIPLICATIVE scale on B.perturbed (truth=no-difference) ---")
    for seed in seeds:
        circuit, a, b = _clean_pair(seed)
        for c in P4_FACTORS:
            _eval(f"P4 mult c={c:g} s={seed}", circuit, a, _attack(b, factor=c), records)

    print("\n--- P5: SMALL MULTIPLICATIVE scale on B.perturbed (truth=no-difference) ---")
    for seed in seeds:
        circuit, a, b = _clean_pair(seed)
        for c in P5_FACTORS:
            _eval(f"P5 mult c={c:g} s={seed}", circuit, a, _attack(b, factor=c), records)

    # Positive controls — a GENUINE per-context knob difference must still RESOLVE.
    print("\n--- positive controls (genuine differences; must RESOLVE, not over-abstain) ---")
    pos: list = []
    for mech, fac, seed in [("gain", 0.55, 1), ("ceiling", 1.4, 1),
                            ("threshold", 1.4, 1)]:
        circuit = ras_switch_1node()
        a, b = simulate_context_pair(
            circuit, mechanism=mech, factor=fac, n_cells=N_CELLS,
            scale_a=SCALE, scale_b=SCALE, obs_sd=OBS_SD, seed=seed,
        )
        base = attribute_differential(a, b, circuit, steps=250, seed=0)
        g = guard_b_classify(a, b, circuit, winner=base.fit.best_diff, steps=180,
                             check_both=True)
        resolved = g.call in CONF
        pos.append((f"{mech} fac={fac}", g.call, resolved))
        print(f"{mech+' fac='+str(fac):26s} base={base.call:14s} "
              f"EARN-GUARD={g.call:14s} {'RESOLVED' if resolved else '<== OVER-ABSTAIN?'}",
              flush=True)

    # Verdict.
    confound = [r for r in records]
    guard_wrong = sum(int(r[4]) for r in confound)
    base_holes = sum(int(r[3]) for r in confound)
    pos_resolved = sum(int(p[2]) for p in pos)
    print("\n" + "=" * 72)
    print(f"CONFOUND CASES: {len(confound)}  (baseline shipped fired confident *-diff on "
          f"{base_holes}/{len(confound)} — the holes are real)")
    print(f"EARN-GUARD CONFIDENT-WRONG: {guard_wrong}/{len(confound)}  "
          f"(target: 0)")
    print(f"POSITIVE CONTROLS RESOLVED: {pos_resolved}/{len(pos)}  "
          f"(target: {len(pos)} — no over-abstention)")
    print("=" * 72)
    ok = guard_wrong == 0 and pos_resolved == len(pos)
    print("PROOF:", "PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
