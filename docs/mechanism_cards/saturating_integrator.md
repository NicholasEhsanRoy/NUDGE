---
id: NUDGE-MECH-011
name: saturating_integrator
role: integrator
registry_name: SaturatingIntegrator
vulnerable_to_decoys: [NUDGE-DECOY-005]
documented_limitation: [NUDGE-LIM-005]
validated_in_regime: {min_cells_per_condition: 1000, notes: "Its production ceiling vmax/decay contributes to the K/vmax degeneracy that makes ceiling the most noise-fragile parameter (FINDINGS §2); a marginal saturating nonlinearity must not be over-called (parsimony gate)."}
references: [HuangFerrell1996, Yuan2021CellBox]
---

# Mechanism Card — Saturating integrator

> **ID:** `NUDGE-MECH-011`  ·  **Role:** integrator
> **Stability:** stable  ·  **Registry name:** `SaturatingIntegrator`

## Summary

Michaelis–Menten production with linear decay —
`dx/dt = vmax·drive/(km+drive) − decay·x`. Production saturates as the drive grows,
so the species carries its **own ceiling** `vmax/decay`, independent of any
regulatory edge.

## Governing equation

```
dx/dt = vmax · drive / (km + drive) − decay · x     steady state: x* = production(drive) / decay
```

- `drive` — the summed input into the species.
- `vmax` — maximal production rate; with `decay` it sets the species ceiling `vmax/decay`.
- `km` — the drive at half-maximal production (a Michaelis constant; a threshold on the drive).
- `decay` — first-order removal rate.

## What it represents

A species whose production machinery saturates — enzyme/promoter limitation giving an
intrinsic maximal output. This is a *graded* saturation (Hill exponent 1 on the
drive), not a cooperative switch: bistability still requires cooperative feedback on
an edge.

## Assumptions & simplifications

- Michaelis–Menten quasi-steady-state on production (single limiting step, no cooperativity).
- Linear decay; steady state taken as the observable.
- The ceiling is a species property (`vmax/decay`) distinct from an edge `vmax` —
  which is one route into the K/vmax attribution degeneracy.

## Known failure modes

| Failure mode | Decoy that exercises it | Limitation |
|---|---|---|
| Saturating production is a mild nonlinearity a greedy fit could over-call as a circuit switch | `NUDGE-DECOY-005` | `NUDGE-LIM-005` |

Guarded by the linear-baseline parsimony gate: a marginal saturating nonlinearity
within the loss floor does not beat linear and is returned `off-model`
(`margin_k = 1.7`).

## Identifiability regime

The species ceiling `vmax/decay` is one contributor to the **K/vmax partial
degeneracy** flagged in FINDINGS §2 — both a threshold move and a ceiling move shrink
the ON signal, so **ceiling is the most noise-fragile** parameter (0.92 → 0.17 as
technical noise rises at 1000 cells/condition) and **threshold needs the most cells**.
As with all mechanisms: below ~1000 cells/condition nothing is attributable and NUDGE
abstains.

## Implementation Mapping

| Equation term | Code |
|---|---|
| production `vmax·drive/(km+drive)` | `nudge.mechanisms.integrators.saturating.saturating_production` |
| rate `production − decay·x` | `nudge.mechanisms.integrators.saturating.saturating_rate` |
| kernel rate method | `nudge.mechanisms.integrators.saturating.SaturatingIntegrator.rate` |
| algebraic steady state | `nudge.mechanisms.integrators.saturating.SaturatingIntegrator.steady_state` |

*(Machine-checked by `scripts/check_impl_mapping.py`.)*

## Verification evidence

- `tests/mechanisms/test_integrators.py::test_saturating_production_half_max_at_km`
  — production is half-maximal at `drive = km`.
- `tests/mechanisms/test_integrators.py::test_saturating_steady_state_is_rate_root`
  — the algebraic steady state is the root of the rate.

## References

- [@HuangFerrell1996] — saturation vs. cooperative ultrasensitivity (distinguishing the two).
- [@Yuan2021CellBox] — saturating perturbation-response kinetics.
