# Mechanism Card — <Mechanism Name>

> **ID:** `NUDGE-MECH-NNN`  ·  **Role:** species | integrator | regulatory-edge | readout | perturbation
> **Stability:** experimental | stable

## Summary

One sentence: what this mechanism represents and when to reach for it.

## Governing equation

The exact functional form, with each parameter's biological meaning. For example,
Hill activation `v = V · xⁿ / (Kⁿ + xⁿ)`, where `K` is the half-max threshold,
`n` the cooperativity (gain), and `V` the ceiling.

## Assumptions & simplifications

- e.g. quasi-steady-state on the fast variable
- e.g. no explicit cooperative-binding intermediate

## Known failure modes

| Failure mode | Decoy that exercises it | Limitation |
|---|---|---|
| <how it mis-fits> | `NUDGE-DECOY-NNN` | `NUDGE-LIM-NNN` |

## Identifiability regime

From the synthetic power sweep: the data regime under which this mechanism's
parameters are recoverable (min cells/condition, min perturbations, max dropout).

## Implementation Mapping

| Equation term | Code |
|---|---|
| production term | `nudge.mechanisms.<module>.<Class>.update` |

*(This table is machine-checked by `scripts/check_impl_mapping.py`: every
`nudge.*` reference must resolve to a real attribute.)*

## Verification evidence

- `NUDGE-VER-NNN` — <the synthetic-recovery test that proves it>

## References

- [@Das2009] — <one-line human-readable description>
