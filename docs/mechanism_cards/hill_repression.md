---
id: NUDGE-MECH-003
name: hill_repression
role: regulatory-edge
registry_name: HillRepression
vulnerable_to_decoys: [NUDGE-DECOY-005]
documented_limitation: [NUDGE-LIM-005, NUDGE-LIM-006]
validated_in_regime: {min_cells_per_condition: 1000, notes: "Shares the K/n/vmax identifiability of hill_activation (FINDINGS §2). As the toggle's repression edge, single-snapshot toggle attribution abstains between gain and threshold (FIM corr(log m, log K) = -0.986)."}
references: [HuangFerrell1996, GardnerCollins2000]
---

# Mechanism Card — Hill repression

> **ID:** `NUDGE-MECH-003`  ·  **Role:** regulatory-edge
> **Stability:** stable  ·  **Registry name:** `HillRepression`

## Summary

A repressing regulatory edge whose response is a cooperative (Hill) decreasing
function of the regulator's activity — the mutual-repression edge of the toggle
switch, carrying the same threshold/gain/ceiling vocabulary as `hill_activation`.

## Governing equation

```
v(x) = vmax · Kⁿ / (Kⁿ + xⁿ)
```

- `x` — regulator activity.
- `K` — **half-max input**, the repression **threshold** (activity at which `v = vmax/2`).
- `n` — **Hill coefficient**, the **gain** / cooperativity of repression (`n > 1` is switch-like).
- `vmax` — the **ceiling**, the un-repressed maximal contribution (value as `x → 0`).

It is the exact complement of `hill_activation`: `hill_repression(x) = vmax − hill_activation(x)`
(same `K`, `n`, `vmax`).

## What it represents

Cooperative repression of a target — a repressor occupying an operator, a
sequestering interaction, or lumped cooperative inhibition. Two such edges closing a
2-node cycle give the mutual-repression **toggle** (Gardner–Collins): anti-correlated
bistable modes.

## Assumptions & simplifications

- Quasi-steady-state on the fast binding reaction (algebraic edge response).
- A single lumped cooperative-repression site set; no explicit intermediate species.
- The repressor's effect is captured entirely by this scalar edge.

## Known failure modes

| Failure mode | Decoy that exercises it | Limitation |
|---|---|---|
| A barely-nonlinear repression (`n` just above 1) over-called as a switch | `NUDGE-DECOY-005` | `NUDGE-LIM-005` |
| Repressive ultrasensitivity in the *readout* misattributed to the circuit edge | `nudge.data.decoy_generators.generate_readout_nonlinearity_decoy` (xfail witness) | `NUDGE-LIM-006` |

The marginal case is guarded by the same linear-baseline parsimony gate as
`hill_activation`. The readout confound is **not** a passing decoy: it is
`NUDGE-LIM-006`, a documented boundary where NUDGE can be a *confident false
positive* (the affine-readout assumption bounds the fail-safe guarantee), witnessed
by an **xfail** (`generate_readout_nonlinearity_decoy`), not a battery case NUDGE passes.

## Identifiability regime

Shares the K/n/vmax power law of `hill_activation` (FINDINGS §2: ≥ ~1000
cells/condition; gain > ceiling ≈ threshold; K/vmax partial degeneracy). As the
toggle's repression edge, the sharper statement is the toggle Fisher-information
result (FINDINGS "N-D saddle"): from a **single snapshot** the sloppy direction is
**gain (`n`) ⇄ threshold (`K`)** — `corr(log m, log K) = −0.986`, condition number
≈ 210 — so NUDGE **abstains between gain and threshold** rather than guess; ceiling
(`vmax`) is the *most* identifiable. A **second operating point** breaks the
degeneracy (condition number 210 → 22).

## Implementation Mapping

| Equation term | Code |
|---|---|
| pure response `vmax·Kⁿ/(Kⁿ+xⁿ)` | `nudge.mechanisms.regulatory.hill_repression` |
| kernel response method | `nudge.mechanisms.regulatory.HillRepressionEffect.response` |
| toggle motif built from two of these | `nudge.circuits.toggle` |

*(Machine-checked by `scripts/check_impl_mapping.py`.)*

## Verification evidence

- `tests/mechanisms/test_regulatory.py::test_repression_is_activation_complement`
  — repression is the exact complement of activation.
- `tests/core/test_fixed_points.py::test_symmetric_toggle_two_stable_one_saddle`
  — two mutual-repression edges yield a bistable toggle with an index-1 saddle.
- `tests/verification/test_toggle_nd_safety.py::test_toggle_transition_never_misclassifies`
  — NUDGE never emits a wrong positive on toggle threshold/gain/ceiling data.

## References

- [@HuangFerrell1996] — Hill ultrasensitivity (steepness/cooperativity).
- [@GardnerCollins2000] — the engineered mutual-repression toggle switch.
