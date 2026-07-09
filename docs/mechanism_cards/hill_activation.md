---
id: NUDGE-MECH-002
name: hill_activation
role: regulatory-edge
registry_name: HillActivation
vulnerable_to_decoys: [NUDGE-DECOY-005]
documented_limitation: [NUDGE-LIM-005, NUDGE-LIM-006]
validated_in_regime: {min_cells_per_condition: 1000, notes: "Gain (n) is the most robust of the three parameters; threshold (K) needs the most cells and partially degenerates with vmax (K/vmax). Numbers from FINDINGS §2."}
references: [HuangFerrell1996, Das2009, FerrellMachleder1998]
---

# Mechanism Card — Hill activation

> **ID:** `NUDGE-MECH-002`  ·  **Role:** regulatory-edge
> **Stability:** stable  ·  **Registry name:** `HillActivation`

## Summary

An activating regulatory edge whose response is a cooperative (Hill) saturating
function of the regulator's activity — NUDGE's canonical **switch edge**, carrying
the entire attribution vocabulary (threshold `K`, gain `n`, ceiling `vmax`).

## Governing equation

```
v(x) = vmax · xⁿ / (Kⁿ + xⁿ)
```

- `x` — regulator activity (the driving species).
- `K` — **half-max input**, i.e. the switch **threshold** (activity at which `v = vmax/2`).
- `n` — **Hill coefficient**, the **gain** / ultrasensitivity. `n = 1` is Michaelian
  (graded); `n > 1` is switch-like (sigmoidal). This is the parameter that
  distinguishes a threshold move from a steepness change.
- `vmax` — the **ceiling** (maximal regulatory contribution).

## What it represents

Cooperative activation of a target by a regulator — allosteric multi-site binding,
enzymatic zero-order ultrasensitivity, or an effective lumped cooperativity. On an
edge that closes a cycle it is the positive feedback that makes a switch bistable
(see `self_activation_switch`, `ras_switch_1node`).

## Assumptions & simplifications

- Quasi-steady-state on the fast binding reaction (the edge is an algebraic
  response `f(x)`, not a dynamical intermediate).
- A single lumped set of cooperative sites — no explicit cooperative-binding
  intermediate species is modelled.
- The regulator's effect on the target is captured entirely by this scalar edge
  (no combinatorial regulation within one edge).

## Known failure modes

| Failure mode | Decoy that exercises it | Limitation |
|---|---|---|
| A barely-nonlinear edge (`n` just above 1) fit as a confident switch | `NUDGE-DECOY-005` | `NUDGE-LIM-005` |
| Ultrasensitivity that actually lives in the *readout* attributed to this circuit edge | `nudge.data.decoy_generators.generate_readout_nonlinearity_decoy` (xfail witness) | `NUDGE-LIM-006` |

The marginal-Hill failure is guarded structurally by the linear-baseline parsimony
gate (calibrated margin `margin_k = 1.7`, FINDINGS §1): a within-floor nonlinearity
does not beat the linear edge and is returned `off-model`. The readout-confound
failure is a genuine identifiability degeneracy — only the composition
`readout ∘ circuit` is observed, so a nonlinear reporter can be misread as a
circuit switch (FINDINGS "NUDGE-LIM-006"). Note the honest asymmetry: the readout
confound is **not** a passing decoy — it is `NUDGE-LIM-006`, a documented boundary
where NUDGE can be a *confident false positive* rather than merely abstain, captured
by an **xfail witness** (`generate_readout_nonlinearity_decoy`), not a battery case
NUDGE passes.

## Identifiability regime

From the synthetic power sweep (FINDINGS §2), at the default `margin_k = 1.7`:

- **≥ ~1000 cells/condition** is required; below it essentially nothing is
  attributable and NUDGE correctly abstains.
- **Gain (`n`) is the most robust** parameter (identifiable across noise once cells
  suffice). **Threshold (`K`) needs the most cells** and partially degenerates with
  `vmax` (both shrink the ON signal — the K/vmax degeneracy). **Ceiling (`vmax`) is
  the most noise-fragile** (correct-attribution 0.92 → 0.17 as technical noise rises
  at 1000 cells).

## Implementation Mapping

| Equation term | Code |
|---|---|
| pure response `vmax·xⁿ/(Kⁿ+xⁿ)` | `nudge.mechanisms.regulatory.hill_activation` |
| kernel response method | `nudge.mechanisms.regulatory.HillActivationEffect.response` |

*(This table is machine-checked by `scripts/check_impl_mapping.py`: every
`nudge.*` reference must resolve to a real attribute.)*

## Verification evidence

- `tests/mechanisms/test_regulatory.py::test_threshold_is_half_max_input` — `K` is
  the half-max input.
- `tests/mechanisms/test_regulatory.py::test_gain_is_steepness_at_threshold` — `n`
  controls steepness at threshold.
- `tests/mechanisms/test_regulatory.py::test_ceiling_and_floor` — `vmax` is the ceiling.
- `tests/inference/test_fit_end_to_end.py::test_fit_attributes_threshold_gain_ceiling`
  — end-to-end recovery of all three at `n_cells=384, steps=400`.

## References

- [@HuangFerrell1996] — ultrasensitivity in the MAPK cascade (the Hill/steepness thesis).
- [@Das2009] — digital, hysteretic Ras activation in lymphocytes (the switch biology NUDGE targets).
- [@FerrellMachleder1998] — an all-or-none (ultrasensitive) cell-fate switch in Xenopus oocytes.
