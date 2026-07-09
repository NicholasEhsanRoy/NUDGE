---
id: NUDGE-MOTIF-002
name: ras_switch_2node
role: motif
registry_name: null
vulnerable_to_decoys: [NUDGE-DECOY-004]
documented_limitation: [NUDGE-LIM-004, NUDGE-LIM-006]
validated_in_regime: {min_cells_per_condition: null, notes: "Not yet characterised for the 2-node mutual-activation switch. The N-D fixed-point/saddle finder and multi-basin representation generalize to it, but the saddle gain gate is guarded to n_species==1, so NUDGE abstains rather than attribute on a 2-node switch (FINDINGS N-D saddle)."}
references: [Das2009, HuangFerrell1996]
---

# Mechanism Card — Ras switch, 2-node (motif)

> **ID:** `NUDGE-MOTIF-002`  ·  **Role:** motif
> **Stability:** experimental (attribution abstains; representation only)  ·  **Registry name:** — (a motif)

## Summary

The explicit **2-node mutual-activation** reading of the Ras switch: RasGTP and SOS
each `hill_activation`-drive the other (SOS ⇄ Ras-GTP positive feedback), so the
lobes are **correlated** (both-low / both-high). The second Gladstone T-cell
candidate topology; `model_select` must find it *earns* its extra parameter over the
1-node form.

## Governing equation

Two species (`RasGTP`, `SOS`), each `basal = 0.05`, `decay = 1.0`, with two
activating edges (defaults `n = 4`, `vmax = 2`, `K = 1`):

```
d[RasGTP]/dt = basal + vmax · SOSⁿ/(Kⁿ + SOSⁿ) − decay · RasGTP     (SOS → RasGTP)
d[SOS]/dt    = basal + vmax · RasGTPⁿ/(Kⁿ + RasGTPⁿ) − decay · SOS   (RasGTP → SOS)
```

The attributable gain/threshold/ceiling live on `target_edge = 0` (SOS → RasGTP).

## What it represents

The co-activation form of the Ras switch (Das 2009): the mutual positive feedback
between the digital GEF (SOS1, allosterically activated by Ras-GTP) and Ras-GTP
itself. Both species rise or fall together — the correlated bistability that
distinguishes it from the anti-correlated toggle.

## Assumptions & simplifications

- Two lumped species; the graded RASGRP1 input is folded into `basal`.
- Quasi-steady-state edge responses.
- Symmetric edge parameters by default (the fit relaxes this per edge).

## Known failure modes

| Failure mode | Decoy that exercises it | Limitation |
|---|---|---|
| A dead-guide (null) perturbation reported as a mechanism | `NUDGE-DECOY-004` | `NUDGE-LIM-004` |
| Readout ultrasensitivity misattributed to the circuit | `nudge.data.decoy_generators.generate_readout_nonlinearity_decoy` (xfail witness) | `NUDGE-LIM-006` |

The readout row is **not** a passing decoy: it is `NUDGE-LIM-006`, a documented
boundary where NUDGE can be a *confident false positive* rather than abstain,
witnessed by an **xfail** (`generate_readout_nonlinearity_decoy`). Beyond these, the
*dominant* honest limitation is that mechanism **attribution is not yet available** on
this 2-node topology (below).

## Identifiability regime

**Not yet characterised** for the 2-node mutual-activation switch. What is
established (FINDINGS "N-D saddle"): the **N-D fixed-point / saddle finder**
(`Circuit.fixed_points` / `transition_state`) and the **N-D multi-basin
representation** generalize to multi-species circuits, so a 2-node switch can be
*represented*. But the saddle transition-mode **gain gate is a measured NO-GO beyond
1 species** and is guarded to `n_species == 1`; on a 2-node switch NUDGE therefore
**abstains (off-model) rather than misclassify**. The Fisher/Lyapunov attribution
work in FINDINGS targets the mutual-**inhibition** toggle, not this mutual-activation
switch. Treat 2-node attribution as future work; representation is available today.

## Implementation Mapping

| Equation term | Code |
|---|---|
| the circuit builder | `nudge.circuits.ras_switch_2node` |
| edge response | `nudge.mechanisms.regulatory.HillActivationEffect.response` |
| N-D fixed points (basins) | `nudge.core.circuit.Circuit.fixed_points` |
| N-D saddle finder | `nudge.core.circuit.Circuit.transition_state` |

*(Machine-checked by `scripts/check_impl_mapping.py`.)*

## Verification evidence

- `tests/core/test_fixed_points.py::test_unsupported_topologies_return_gracefully`
  — the N-D finder degrades gracefully where a saddle is not resolved.
- `tests/verification/test_toggle_nd_safety.py::test_toggle_transition_never_misclassifies`
  — the N-D safety guard: with the 1-D-only gain gate inert, NUDGE abstains on
  multi-species switches rather than emit a wrong positive.

## References

- [@Das2009] — SOS ⇄ Ras-GTP mutual positive feedback (the 2-node biology).
- [@HuangFerrell1996] — ultrasensitivity from cooperative feedback.
