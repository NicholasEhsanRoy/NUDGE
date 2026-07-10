"""RED-TEAM (round 3): design()'s bifurcation SAFETY gate clears an intervention that
pushes a switch INTO the near-fold regime as "OK, stays away from the fold".

Target: ``nudge.design.invert.design`` (``NUDGE-METHOD-007`` / ``NUDGE-LIM-013``) — the
flagship inverse-design verb. Its safety gate (``_safety_report``) flags
``high_risk_of_instability`` ONLY when the INCREASE in fold proximity
(``delta = proximity_after - proximity_before``) exceeds ``margin`` (default 0.15). It
NEVER compares the ABSOLUTE ``proximity_after`` against the shipped near-fold threshold
``bifurcation.NEAR_FOLD = 0.55``. So an intervention that moves a robust switch
(proximity ~0.50) ACROSS 0.55 into the near-fold regime — but by a sub-margin increment —
is reported "safety: OK, stays away from the fold", directly contradicting
``classify_robustness`` on the very same intervened circuit (which calls it ``near-fold``).
That is a confident-wrong SAFETY flag on a PROPOSAL (the highest-harm output).

Construction: base = ras_switch_1node(n=2.0, vmax=3.0, K=1.5) → proximity 0.500 (robust).
Target = the HIGH state of the K=1.0 variant → proximity 0.589 (near-fold). Reducing K
1.5→1.0 (the reachable inversion) raises proximity by only ~0.089 < margin 0.15, so the
gate stays green while the switch lands in near-fold territory.

Run: uv run python scripts/redteam/design_safety_gate_absolute_proximity.py
Touches no src/ code and no fail-safe margins — diagnostic only.
"""

from __future__ import annotations

import numpy as np

from nudge.circuits import ras_switch_1node
from nudge.design.invert import CircuitFit, InterventionPlan, _rebuild, design, flip_target
from nudge.inference.bifurcation import (
    NEAR_FOLD,
    bifurcation_proximity,
    classify_robustness,
)


def _prox(circuit: object) -> float:
    s = bifurcation_proximity(circuit)
    return float("nan") if s is None else s.proximity


def run() -> int:
    print("=" * 80, flush=True)
    print("RED-TEAM r3: design() safety gate clears a push INTO the near-fold regime")
    print(f"(NEAR_FOLD={NEAR_FOLD}; gate flags only delta>margin, never absolute after)")
    print("=" * 80, flush=True)

    base = ras_switch_1node(n=2.0, vmax=3.0, K=1.5, basal=0.05)
    near = ras_switch_1node(n=2.0, vmax=3.0, K=1.0, basal=0.05)  # the near-fold target
    print(f"base circuit proximity   = {_prox(base):.3f}  "
          f"({classify_robustness(bifurcation_proximity(base))[0]})")
    print(f"target (K=1.0) proximity = {_prox(near):.3f}  "
          f"({classify_robustness(bifurcation_proximity(near))[0]})", flush=True)

    # A RELIABLE attribution (integrity gate passes) whose only addressable knob is K.
    fit = CircuitFit(
        circuit=base,
        free=[("edge", 0, "K")],
        is_reliable=True,
        reason="fitted threshold attribution (reliable) — target edge K addressable",
    )
    # Target: adjust the ON level to the near-fold variant's HIGH state (start from high).
    target = flip_target(near, to="high")

    plan = design(fit, target, free=[("edge", 0, "K")], start="high", steps=400,
                  l1=1e-3, tol=0.3, margin=0.15)
    if not isinstance(plan, InterventionPlan):
        print(f"\ndesign abstained: {plan.reason}")
        print("  (attack did not reach the target — no hole demonstrated this run)")
        return 0

    safety = plan.safety
    assert safety is not None
    # Independently recompute the intervened circuit's robustness verdict.
    vals = np.array(
        [float(np.exp(np.log(1.5) + plan.deltas[0][1]))] if plan.deltas else [1.5]
    )
    after_circuit = _rebuild(base, [("edge", 0, "K")], vals)
    after_score = bifurcation_proximity(after_circuit)
    after_call, after_reason = classify_robustness(after_score)

    print("\n--- design() output ---", flush=True)
    print(f"proposed deltas: {[(f, round(ld, 3), round(fac, 3)) for f, ld, fac in plan.deltas]}")
    print(f"achieved (rel gap remaining) = {plan.achieved_loss:.3f}")
    print(f"safety.proximity_before = {safety.proximity_before}")
    print(f"safety.proximity_after  = {safety.proximity_after}")
    print(f"safety.delta            = {safety.delta}")
    print(f"safety.high_risk_of_instability = {safety.high_risk_of_instability}")
    print(f"safety.crosses_fold             = {safety.crosses_fold}")
    print(f"design REASON: {plan.reason}", flush=True)

    print("\n--- independent classify_robustness on the SAME intervened circuit ---")
    print(f"proximity_after = {after_score.proximity:.3f}  -> classify: {after_call!r}")
    print(f"  {after_reason}", flush=True)

    prox_after = safety.proximity_after
    hole = (
        prox_after is not None
        and prox_after >= NEAR_FOLD
        and not safety.high_risk_of_instability
        and not safety.crosses_fold
        and "stays away from the fold" in plan.reason
        and after_call == "near-fold"
    )
    print("\n" + "=" * 80, flush=True)
    if hole:
        print("HOLE VERIFIED: design() reports 'OK, stays away from the fold' while the")
        print(f"intervened circuit sits at proximity {prox_after:.3f} >= NEAR_FOLD "
              f"{NEAR_FOLD} — NUDGE's own classify_robustness calls it 'near-fold'.")
        print("A confident-wrong SAFETY flag on the proposal (delta<margin masks the")
        print("absolute near-fold landing).")
        return 2
    print("no hole this run (target unreached or gate caught it).")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
