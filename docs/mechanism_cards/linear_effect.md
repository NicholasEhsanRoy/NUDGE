---
id: NUDGE-MECH-001
name: linear_effect
role: regulatory-edge
registry_name: LinearEffect
vulnerable_to_decoys: []
documented_limitation: [NUDGE-LIM-006]
validated_in_regime: {min_cells_per_condition: 1000, notes: "The parsimony baseline. Across 300 synthetic linear datasets the linear-baseline gate rejects a false switch at < 2% (margin_k = 1.7); NUDGE never misclassified a mechanism across 120 ground-truth switch datasets (FINDINGS §1)."}
references: [Yuan2021CellBox]
---

# Mechanism Card — Linear effect

> **ID:** `NUDGE-MECH-001`  ·  **Role:** regulatory-edge
> **Stability:** stable  ·  **Registry name:** `LinearEffect`

## Summary

The baseline linear regulatory edge `f(x) = weight · x` — the **null model** the
switch (Hill) edges must beat. It is the reference the parsimony gate ties against,
so a Hill only "wins" when it beats linear beyond the loss noise floor.

## Governing equation

```
v(x) = weight · x
```

- `x` — regulator activity.
- `weight` — the (signed) linear sensitivity of the target to the regulator. No
  threshold, gain, or ceiling — a linear edge has none of NUDGE's switch vocabulary,
  which is exactly why it is the discriminating baseline.

## What it represents

Graded, proportional regulation — the regime with no ultrasensitivity and no
saturation. It is the interpretable linear-response model (in the spirit of
CellBox-style perturbation-response modelling) against which NUDGE measures whether a
mechanistic switch is warranted at all.

## Assumptions & simplifications

- Effect is proportional to activity over the whole observed range (no saturation,
  no cooperativity).
- Single-edge, single-regulator contribution (additive with other edges).

## Known failure modes

| Failure mode | Decoy that exercises it | Limitation |
|---|---|---|
| A genuinely linear circuit read through a nonlinear (saturating/sigmoidal) reporter is misattributed to a circuit switch | `nudge.data.decoy_generators.generate_readout_nonlinearity_decoy` (xfail witness) | `NUDGE-LIM-006` |

A truly linear circuit *should* keep NUDGE at the linear baseline. The one place this
breaks is the readout confound: when the measurement map is strongly nonlinear, the
affine-readout switch model can explain the skew by bending the circuit, and a linear
circuit is confidently mis-called (FINDINGS "NUDGE-LIM-006"; a constitutive-reporter
control is the validated mitigation). This is `NUDGE-LIM-006`, a documented boundary
where NUDGE can be *confidently wrong* rather than abstain — captured by an **xfail
witness** (`generate_readout_nonlinearity_decoy`), not a decoy NUDGE passes. The decoys where a *monostable/linear*
population merely *looks* bimodal — telegraph noise (`NUDGE-DECOY-001`), cell-type
mixture (`NUDGE-DECOY-002`), technical dropout (`NUDGE-DECOY-003`) — are handled
correctly: the parsimony gate keeps NUDGE at this linear baseline (`off-model`).

## Identifiability regime

As the baseline it has no switch parameters to identify; its role is measured as the
**false-positive rate** of the parsimony gate. FINDINGS §1: across **300 synthetic
linear datasets** the linear-baseline gate rejects a false switch at **< 2%**
(`margin_k = 1.7`), and across 120 ground-truth switch datasets NUDGE **never
misclassified** a mechanism — it abstains when unsure. `margin_k` is a clean
specificity↔sensitivity dial with no wrong-answer risk on either end.

## Implementation Mapping

| Equation term | Code |
|---|---|
| pure response `weight·x` | `nudge.mechanisms.regulatory.linear_effect` |
| kernel response method | `nudge.mechanisms.regulatory.LinearEffect.response` |

*(Machine-checked by `scripts/check_impl_mapping.py`.)*

## Verification evidence

- `tests/mechanisms/test_regulatory.py::test_linear_effect_is_proportional` — the
  edge is exactly proportional to activity.
- `tests/inference/test_fit_end_to_end.py::test_fit_linear_data_returns_off_model`
  — linear ground-truth data is returned `off-model`, not a spurious switch.

## References

- [@Yuan2021CellBox] — interpretable linear perturbation-response modelling (the baseline lineage).
