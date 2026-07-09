---
id: NUDGE-METHOD-007
name: inverse_design
role: attribution-method
registry_name: InverseDesign
vulnerable_to_decoys: []
documented_limitation: [NUDGE-LIM-012, NUDGE-LIM-013]
validated_in_regime: {min_stable_modes: 1, notes: "Inverts a RELIABLE attribution to PROPOSE an intervention, behind two honesty gates. Circuit mode: gradient inversion over a fitted differentiable Circuit (the fit_parameters loop run backwards) finds an additive log-delta on addressable kinetic knobs (edge K/n/vmax or species basal=dose) minimizing ||PredictedState - target||^2 + l1||delta||, then runs the Cap-5 bifurcation_proximity SAFETY gate on the intervened circuit. Curve mode: closed-form inversion of a DoseResponseFit to the dose achieving a target response y. Verified on synthetic ground truth: (a) known-intervention recovery — a monostable switch perturbed by a known x2 on v_max is recovered to factor≈2.0 with residual gap <1e-3; (b) integrity gate — an unresolved/no-effect attribution abstains immediately; (c) reachability — an impossible target abstains rather than extrapolate; (d) safety gate — a flip-ON intervention that raises basal over the fold sets crosses_fold=True / high_risk_of_instability=True while a modest ON-level nudge stays bistable (not high-risk); (e) curve mode round-trips (y=floor+amp/2 inverts to dose≈K) and abstains out of (floor, floor+amp). A real-data lock-in inverts the OCT4 dose-response fit to a knockdown dose (needs_data). Circuit mode is demoed on SYNTHETIC bistable switches (our real-data fits are dose-response curves, which have no circuit/fold); the safety gate lives only in circuit mode (stated)."}
references: [Strogatz2015, Scheffer2009, FerrellMachleder1998]
---

# Mechanism Card — Inverse / intervention design (the flagship)

> **ID:** `NUDGE-METHOD-007`  ·  **Role:** attribution-method
> **Stability:** experimental  ·  **Registry name:** `InverseDesign`

## Summary

NUDGE's other capabilities *diagnose* — which knob (threshold / gain / ceiling) a
perturbation turns. This one turns a diagnosis into a **prescription**: given a *reliable*
attribution, it runs the same differentiable machinery **backwards** to propose an
**untested intervention** — a change to a kinetic parameter, or a dose — that moves the
system toward a target. It is the headline of the brief ("NUDGE inverts the fit to
propose untested interventions"), delivered behind two honesty gates so the proposals are
trustworthy, not confident guesses.

Two modes, dispatched by what the attribution *carries* (a strictly-minimal structural
`AttributionResult` protocol — the integrity gate needs only `is_reliable` + `reason`):

- **Circuit-level** (flagship / deep-tech): gradient inversion over a fitted
  differentiable `Circuit`. It optimizes an additive log-delta over addressable kinetic
  knobs to reach a target readout state, then runs a **bifurcation safety gate**. Demoed
  on a synthetic bistable switch (a flip-ON with the Cap-5 safety dial).
- **Curve-level** (real-data surface): closed-form inversion of a `DoseResponseFit` to the
  dose achieving a target response `y`. Runs on the real OCT4/NANOG and Chure fits. No
  circuit ⇒ **no safety gate** (stated honestly).

## Governing equation

**Circuit mode** optimizes a log-delta `Δ` over the addressable knobs, applied
functionally (autodiff-clean, single-leaf params):

```
param'_j = exp( log(param_j) + Δ_j )
PredictedState(Δ) = readout.expression( circuit.steady_state(params'(Δ), x0) )
L(Δ) = ‖ PredictedState(Δ) − target_state ‖²  +  l1 · ‖Δ‖₁
```

optimized with Adam + `jax.value_and_grad` (the `fit_parameters` loop, run backwards).
`x0` seeds the solve at the circuit's *current* stable state (the resting basin, to flip
ON *from*). The `l1` term makes the proposal **sparse** — few-knob, actionable — and the
surviving Δ are ranked by magnitude. Here `param_j` is a kinetic knob (edge `K` threshold,
`n` gain, `vmax` ceiling, or species `basal` = the dose axis); `target_state` is a
readout-space vector (a convenience `flip_target` returns the high/low fixed-point state).

**Curve mode** inverts the fitted Hill in closed form. For a repressive curve
`y = floor + amp·Kⁿ/(Kⁿ + doseⁿ)`:

```
dose = K · ( amp/(y − floor) − 1 )^(1/n)          (repress)
dose = K · ( (y − floor)/(amp − (y − floor)) )^(1/n)   (activate)
```

## The gates (fail-safe, in order)

1. **Integrity gate.** If `attribution.is_reliable` is false — an `unresolved` /
   `no-effect` dose-response, a low-confidence fit — `design()` returns an
   `AbstentionResult` immediately. NUDGE never designs off a fit it does not trust.
2. **Reachability gate.** If no intervention closes the target gap within `tol` (circuit
   mode), or the target `y` is outside `(floor, floor+amp)` (curve mode), it abstains —
   "the target is unreachable within the fit's identifiable region" — rather than
   extrapolate a false proposal (`NUDGE-LIM-013`).
3. **Bifurcation safety gate** (circuit mode only). It scores `bifurcation_proximity`
   (Cap 5, `NUDGE-METHOD-006`) on the base vs the intervened circuit:
   - base not bistable → no switch to destabilize (`high_risk=False`);
   - intervened circuit no longer bistable → the intervention **crossed the fold**; the
     switch loses bistability (`crosses_fold=True`, `high_risk=True`) — the sharpest
     instability signal;
   - both bistable → `Δproximity = after − before`; `high_risk` if it exceeds a margin
     (pushed toward the fold). The proximity is a **one-sided LOWER bound** near the fold
     (`NUDGE-LIM-012`), so the risk is reported as "at least this close".

## Assumptions & simplifications

- **A proposal is valid only within the fit's identifiable region.** `design()` inverts a
  *model*; the intervention is a hypothesis to test, not a guaranteed outcome, and
  extrapolation beyond the fitted region is flagged (`NUDGE-LIM-013`). The one-sided
  safety bound near the fold is inherited from Cap 5 (`NUDGE-LIM-012`).
- **Gradient inversion sees only the basin it starts in.** From the resting (low) basin,
  a knob that continuously moves the low fixed point (e.g. `basal`) provides a gradient
  path across the fold; a knob whose effect on that fixed point is weak (e.g. `K` alone)
  can leave the optimizer stalled — reported honestly as a reachability abstention, never
  a forced call. The `start` basin is an explicit input.
- **The safety gate is circuit-only.** Curve mode has no circuit and no fold, so
  `safety=None` and this is stated in the plan's `reason` — a dose-response curve cannot
  be checked for a tipping point it does not represent.
- **Reliability is the caller's diagnosis.** The integrity gate trusts the attribution's
  own `is_reliable` flag (a positive dose-response / epistasis verdict; a circuit fit's
  self-reported diagnostics). `design()` adds no new fit; it inherits the honesty bounds
  of whatever produced the attribution (`NUDGE-LIM-006`/`007` for a dose-response fit).

## Known failure modes

| Failure mode | Guard / witness | Limitation |
|---|---|---|
| Designing off an untrustworthy fit | the integrity gate (`is_reliable` false → abstain); `test_integrity_gate_refuses_unreliable_attribution` | `NUDGE-LIM-013` |
| Extrapolating a proposal to an unreachable target | the reachability abstention (circuit `tol`; curve out-of-`(floor, floor+amp)`); `test_reachability_abstains_on_impossible_target` / `test_curve_inversion_abstains_out_of_range` | `NUDGE-LIM-013` |
| Silently proposing an intervention that destabilizes a switch | the Cap-5 safety gate (`crosses_fold` / `high_risk_of_instability`); `test_safety_gate_flags_a_fold_crossing_flip` | `NUDGE-LIM-012` |
| A false-precise safety number near the fold | inherited one-sided lower bound (`SafetyReport.one_sided`) | `NUDGE-LIM-012` |

There is **no dedicated design decoy battery yet** (`vulnerable_to_decoys: []`) — the
failure modes above are guarded by the two abstention gates, the safety gate, and the
known-intervention-recovery ground truth. A synthetic decoy (a target reachable only by a
fold-crossing intervention that a naive inverter would propose without warning) is future
work.

## Identifiability regime

- **Known-intervention recovery is the load-bearing ground truth.** A monostable switch
  perturbed by a known `×2` on `v_max` is recovered to `factor ≈ 2.0` with residual gap
  `< 1e-3` (`test_known_intervention_recovery_hits_zero_loss`). Because we know the true Δ,
  this is a clean recovery test, not a vibe check.
- **The safety gate is validated on the self-activation switch's known analytic fold.** A
  flip-ON intervention that raises `basal` over the fold sets `crosses_fold` / `high_risk`,
  while a modest ON-level nudge from the high basin stays bistable and is **not** high-risk
  — the two witnesses partition safe vs unsafe (`test_safety_gate_flags_a_fold_crossing_flip`,
  `test_safe_intervention_stays_away_from_the_fold`).
- **Curve mode round-trips exactly.** `y = floor + amp/2` inverts to `dose ≈ K`
  (`test_curve_inversion_round_trips_to_a_dose`) and an out-of-range target abstains.
- **Real-data lock-in.** The OCT4 dose-response switch fit inverts to a positive knockdown
  dose for a reachable self-renewal target, and abstains below the silenced floor
  (`test_design_inverts_real_oct4_fit`, `needs_data`).

## Implementation Mapping

| Step | Code |
|---|---|
| the minimal input contract (integrity-gate protocol) | `nudge.design.invert.AttributionResult` |
| the circuit-design substrate (circuit + knobs + reliability) | `nudge.design.invert.CircuitFit` |
| additive `is_reliable` on the dose-response verdict | `nudge.inference.dose_response.DoseResponseResult.is_reliable` |
| additive `is_reliable` on the epistasis verdict | `nudge.inference.epistasis.EpistasisResult.is_reliable` |
| gradient inversion over the fitted circuit (the fit loop, backwards) | `nudge.design.invert.design` |
| the flip-ON / flip-OFF target helper | `nudge.design.invert.flip_target` |
| the Cap-5 bifurcation safety gate (base vs intervened) | `nudge.inference.bifurcation.bifurcation_proximity` |
| the proposed intervention or the abstention | `nudge.design.invert.InterventionPlan` |
| the safety verdict (before→after, crosses_fold, one-sided) | `nudge.design.invert.SafetyReport` |
| CLI / MCP orchestration (circuit + curve modes) | `nudge.service.design_circuit` |

*(This table is machine-checked by `scripts/check_impl_mapping.py`: every `nudge.*`
reference must resolve to a real attribute.)*

## Verification evidence

- `tests/design/test_invert.py::test_known_intervention_recovery_hits_zero_loss` — the
  load-bearing ground truth: a known single-knob perturbation is recovered (factor ≈ true,
  loss ≈ 0).
- `tests/design/test_invert.py::test_integrity_gate_refuses_unreliable_attribution` — an
  `unresolved` attribution → `AbstentionResult` (never designs off an unreliable fit).
- `tests/design/test_invert.py::test_reachability_abstains_on_impossible_target` — an
  impossible target abstains (no false extrapolation).
- `tests/design/test_invert.py::test_safety_gate_flags_a_fold_crossing_flip` /
  `::test_safe_intervention_stays_away_from_the_fold` — the safety gate flags a
  fold-crossing flip and clears a safe nudge.
- `tests/design/test_invert.py::test_curve_inversion_round_trips_to_a_dose` /
  `::test_curve_inversion_abstains_out_of_range` — curve-mode round-trip + reachability.
- `tests/design/test_invert.py::test_design_inverts_real_oct4_fit` — the real OCT4 fit
  inverts to a knockdown dose (`needs_data`).
- `tests/test_service.py::test_design_circuit_wiring_flags_a_fold_crossing_flip` /
  `::test_design_file_curve_wiring_and_reachability` — the CLI/MCP service round-trip.

## References

- [@Strogatz2015] — *Nonlinear Dynamics and Chaos*: the saddle-node / fold bifurcation the
  safety gate guards against, and the fixed-point geometry the circuit inversion moves.
- [@Scheffer2009] — early-warning signals for critical transitions: why an intervention
  that pushes a switch toward its fold is dangerous (loss of resilience), grounding the
  safety gate.
- [@FerrellMachleder1998] — the ultrasensitive bistable switch (Xenopus MAPK) as the
  archetype of the cell-fate switch this capability flips ON / OFF.
