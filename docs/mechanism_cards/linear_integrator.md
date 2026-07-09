---
id: NUDGE-MECH-010
name: linear_integrator
role: integrator
registry_name: LinearIntegrator
vulnerable_to_decoys: []
documented_limitation: []
validated_in_regime: {min_cells_per_condition: null, notes: "Not exercised by a dedicated decoy. Its single-basin fixed-x0 solve is the structural reason NUDGE abstains rather than misclassify on emergent feedback bistability (FINDINGS Tier-0.5 T0.5-1)."}
references: [Yuan2021CellBox]
---

# Mechanism Card — Linear integrator

> **ID:** `NUDGE-MECH-010`  ·  **Role:** integrator
> **Stability:** stable  ·  **Registry name:** `LinearIntegrator`

## Summary

Simple relaxation dynamics for a species whose own production does not saturate —
`dx/dt = production − decay·x`, with algebraic steady state `production / decay`. The
default integrator for graded-production species (e.g. SOS / RasGRP1 activity).

## Governing equation

```
dx/dt = production − decay · x        steady state: x* = production / decay
```

- `production` — the summed regulatory drive into the species (from its incoming edges plus basal).
- `decay` — the first-order removal rate; sets the relaxation timescale `1/decay` and
  the steady-state gain `1/decay`.

## What it represents

A species that integrates its inputs linearly and relaxes to a drive-proportional
steady state — no intrinsic ceiling of its own (any ceiling comes from a regulatory
edge, not from the integrator). Cooperative/switch behaviour is supplied by the edges
feeding `production`, keeping the integrator itself interpretable.

## Assumptions & simplifications

- First-order (linear) decay; no saturation of removal.
- Production is treated as an external drive at each solve step; the steady state is
  taken as the observable (quasi-steady-state on the measurement timescale).
- The fit solves each cell from a fixed initial condition `x0 = 0`
  (`nudge.inference.fit`), so on a *feedback* switch it reaches the LOW basin only.

## Known failure modes

No dedicated decoy targets the integrator in isolation. Its one documented
consequence is structural and *fail-safe*: because the single-basin fit relaxes from
`x0 = 0`, on an emergent-bistable feedback circuit it cannot populate the HIGH mode,
so NUDGE **abstains rather than inventing a mechanism** (FINDINGS Tier-0.5 T0.5-1 —
zero wrong positives across seeds). The multi-basin representation
(`fit_multibasin`) exists to represent that HIGH mode; it is kept EXPERIMENTAL / not
fail-safe (FINDINGS T0.5-4).

## Identifiability regime

Not separately characterised as a parameter-recovery problem — `decay` sets the
scale but the identifiable quantities are the edge parameters feeding `production`
(see `hill_activation`). The relevant, measured fact is the single-basin abstention
above.

## Implementation Mapping

| Equation term | Code |
|---|---|
| rate `production − decay·x` | `nudge.mechanisms.integrators.linear.linear_rate` |
| kernel rate method | `nudge.mechanisms.integrators.linear.LinearIntegrator.rate` |
| algebraic steady state | `nudge.mechanisms.integrators.linear.LinearIntegrator.steady_state` |

*(Machine-checked by `scripts/check_impl_mapping.py`.)*

## Verification evidence

- `tests/mechanisms/test_integrators.py::test_linear_steady_state_is_rate_root` —
  the algebraic steady state is the root of the rate.
- `tests/verification/test_stochastic_inverse_crime.py::test_stochastic_fit_never_misclassifies`
  — the single-basin solve abstains (never wrong) on emergent bistability.

## References

- [@Yuan2021CellBox] — perturbation-biology ODE modelling (linear relaxation lineage).
