---
id: NUDGE-MOTIF-004
name: self_activation_switch
role: motif
registry_name: null
vulnerable_to_decoys: [NUDGE-DECOY-001, NUDGE-DECOY-004, NUDGE-DECOY-005]
documented_limitation: [NUDGE-LIM-001, NUDGE-LIM-004, NUDGE-LIM-005, NUDGE-LIM-006]
validated_in_regime: {min_cells_per_condition: 1000, notes: "The one topology characterised on an independent stochastic simulator (Tier-0.5). Single-basin fit is fail-safe (zero wrong positives, seeds 0-3). The saddle transition-mode gain gate recovers gain at all four seeds (free-n w_trans 0.87-0.94 vs <=0.12 for other classes; tau=0.5). FINDINGS T0.5-1, T0.5-5."}
references: [Das2009, HuangFerrell1996, ToMaheshri2010, KeplerElston2001]
---

# Mechanism Card — Self-activation switch (motif)

> **ID:** `NUDGE-MOTIF-004`  ·  **Role:** motif
> **Stability:** stable (single-basin, fail-safe)  ·  **Registry name:** — (a motif, not a registered primitive)

## Summary

A one-species positive-feedback switch: a single species with a cooperative
`hill_activation` self-edge. Bistable (low/high) for cooperative `n`. This is the
motif NUDGE's fail-safe attribution is **most thoroughly characterised** on — the
canonical case for the Tier-0.5 independent-simulator test and the saddle
transition-mode gain gate. Its Ras instance is `ras_switch_1node`.

## Governing equation

One species `x` with a self-activating edge and linear relaxation:

```
dx/dt = basal + vmax · xⁿ / (Kⁿ + xⁿ) − decay · x
```

- `K` — switch **threshold** (self-edge half-max).
- `n` — **gain** / cooperativity of the positive feedback (`n > 1` gives bistability).
- `vmax` — **ceiling** of the self-drive.
- `basal`, `decay` — leak production and first-order removal.

For cooperative `n` the vector field has **three fixed points**: a stable LOW basin,
a stable HIGH basin, and an unstable **saddle** between them. The low/high lobes are
the resting/activated states read out in the snapshot.

## What it represents

An activation program that amplifies itself — the minimal bistable switch. In T-cell
Ras signalling it is the 1-node reading of the SOS positive-feedback loop (Das 2009).
Perturbations move the self-edge: gain = `n`, threshold = `K`, ceiling = `vmax`.

## Assumptions & simplifications

- One lumped activation species (the 2-node SOS⇄Ras form is `ras_switch_2node`).
- Quasi-steady-state edge response; steady state read as the observable.
- The single-basin fit relaxes from a fixed `x0 = 0`, so it natively reaches the LOW
  basin only — the HIGH mode requires the multi-basin representation.

## Known failure modes

| Failure mode | Decoy that exercises it | Limitation |
|---|---|---|
| Noise-induced bimodality (slow promoter switching, non-cooperative feedback) mistaken for this switch | `NUDGE-DECOY-001` | `NUDGE-LIM-001` |
| A dead-guide (null) perturbation reported as a mechanism because WT is a switch | `NUDGE-DECOY-004` | `NUDGE-LIM-004` |
| A marginal (`n` just above 1) self-edge over-called as a switch | `NUDGE-DECOY-005` | `NUDGE-LIM-005` |
| Ultrasensitivity in the readout misattributed to this circuit | `nudge.data.decoy_generators.generate_readout_nonlinearity_decoy` (xfail witness) | `NUDGE-LIM-006` |

The readout row is **not** a passing decoy: it is `NUDGE-LIM-006`, a documented
boundary where NUDGE can be a *confident false positive* rather than abstain,
witnessed by an **xfail** (`generate_readout_nonlinearity_decoy`). The other three
rows are genuine battery cases NUDGE resists (`off-model` / `no-effect`).

## Identifiability regime

The **only** topology characterised on an *independent* stochastic simulator
(Tier-0.5, tau-leaping SSA, emergent bimodality):

- **Single-basin fit is fail-safe** (FINDINGS T0.5-1): fitting the matched-topology
  switch to SSA data (3000 cells/condition; fit `n_cells=384, steps=400`,
  `margin_k=1.7`) across seeds 0–3 yields **zero wrong positive mechanisms** — it
  abstains or recovers only the most robust mechanism (gain). The cost: it does not
  reliably *recover* mechanism on emergent feedback bistability.
- **Saddle transition-mode gain gate** (FINDINGS T0.5-5) restores recovery,
  fail-safe: a gain reduction (`n`: 6→1.2) collapses bistability to a single
  intermediate fixed point at ~1.116 (near the WT saddle at 0.975) — graded cells
  fill the saddle region — while threshold and ceiling go monostable-low. A
  restricted free-`n` three-mode fit measures `w_trans` (seeds 0–3):
  **gain 0.87 / 0.89 / 0.87 / 0.94 (mean 0.89)** vs **≤ 0.12** for
  no-effect/threshold/ceiling. A `τ = 0.5` gate has a wide 0.12↔0.87 margin and
  `fit_multibasin(transition_mode=True)` **recovers gain at all four seeds** (incl.
  the notorious seed 2) with **zero wrong positives**; threshold/ceiling safely abstain.

## Implementation Mapping

| Equation term | Code |
|---|---|
| the Ras 1-node instance (self-activation builder) | `nudge.circuits.ras_switch_1node` |
| self-edge response | `nudge.mechanisms.regulatory.HillActivationEffect.response` |
| LOW/HIGH basins + saddle | `nudge.core.circuit.Circuit.fixed_points` |
| the saddle (transition-mode centre) | `nudge.core.circuit.Circuit.transition_state` |

*(Machine-checked by `scripts/check_impl_mapping.py`.)*

## Verification evidence

- `tests/verification/test_stochastic_inverse_crime.py::test_stochastic_wt_is_emergently_bimodal`
  — the SSA WT population is emergently bimodal (not designed-in).
- `tests/verification/test_stochastic_inverse_crime.py::test_stochastic_fit_never_misclassifies`
  — single-basin fit never emits a wrong positive.
- `tests/verification/test_stochastic_inverse_crime.py::test_saddle_transition_recovers_gain_never_wrong`
  — the saddle gain gate recovers gain, never wrong.
- `tests/core/test_fixed_points.py::test_bistable_switch_three_fixed_points` — three
  fixed points (LOW, saddle, HIGH) for cooperative `n`.

## References

- [@Das2009] — digital, hysteretic Ras activation via SOS positive feedback.
- [@HuangFerrell1996] — cooperativity/ultrasensitivity underlying the switch.
- [@ToMaheshri2010] — noise-induced bimodality without bistability (the DECOY-001 confound).
- [@KeplerElston2001] — stochastic transcription / noise-induced switching (the same confound).
