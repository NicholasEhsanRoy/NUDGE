---
id: NUDGE-MOTIF-001
name: ras_switch_1node
role: motif
registry_name: null
vulnerable_to_decoys: [NUDGE-DECOY-001, NUDGE-DECOY-004, NUDGE-DECOY-005]
documented_limitation: [NUDGE-LIM-001, NUDGE-LIM-004, NUDGE-LIM-005, NUDGE-LIM-006]
validated_in_regime: {min_cells_per_condition: 1000, notes: "The 1-node self-activation reading of Ras activation. Identifiability is characterised via the general self_activation_switch motif (Tier-0.5 fail-safe + saddle gain gate; FINDINGS T0.5-1, T0.5-5). Which of the 1-node vs 2-node topology the biology uses is left to model_select."}
references: [Das2009, HuangFerrell1996]
---

# Mechanism Card — Ras switch, 1-node (motif)

> **ID:** `NUDGE-MOTIF-001`  ·  **Role:** motif
> **Stability:** stable  ·  **Registry name:** — (a motif, not a registered primitive)

## Summary

The **1-node self-activation** reading of the Ras activation switch: a single
"Activation" species with a cooperative `hill_activation` self-edge (the SOS positive
feedback). One of the two **candidate topologies** for the Gladstone T-cell
validation — NUDGE does not presume which the biology uses; `model_select` fits both
1- and 2-node and lets a parsimony penalty choose.

## Governing equation

One species `Activation` (`basal = 0.05`, `decay = 1.0`) with a self edge:

```
d[Act]/dt = basal + vmax · Actⁿ / (Kⁿ + Actⁿ) − decay · Act
```

Defaults `n = 6`, `vmax = 2`, `K = 1`. Attribution: gain = `n`, threshold = `K`,
ceiling = `vmax`. Bistable for cooperative `n`; the low/high lobes are the
resting/activated T-cell states read out by the IEG panel.

## What it represents

The Ras activation switch (Das 2009): RASGRP1 (graded input GEF) primes Ras-GTP;
SOS1 (digital GEF) is allosterically activated **by** Ras-GTP → positive feedback →
bistability. Collapsed to one species, the activation program self-amplifies. This is
the mechanistic realisation of the general `self_activation_switch` motif.

## Assumptions & simplifications

- The 2-node SOS⇄Ras-GTP structure is lumped into one self-activating species (the
  explicit form is `ras_switch_2node`; model selection arbitrates).
- Quasi-steady-state edge response; single-basin fit relaxes from `x0 = 0` (LOW basin natively).

## Known failure modes

| Failure mode | Decoy that exercises it | Limitation |
|---|---|---|
| Noise-induced bimodality mistaken for the Ras switch | `NUDGE-DECOY-001` | `NUDGE-LIM-001` |
| A dead-guide (null) perturbation reported as a mechanism | `NUDGE-DECOY-004` | `NUDGE-LIM-004` |
| A marginal self-edge nonlinearity over-called | `NUDGE-DECOY-005` | `NUDGE-LIM-005` |
| Readout ultrasensitivity misattributed to the circuit | `nudge.data.decoy_generators.generate_readout_nonlinearity_decoy` (xfail witness) | `NUDGE-LIM-006` |

The readout row is **not** a passing decoy: it is `NUDGE-LIM-006`, a documented
boundary where NUDGE can be a *confident false positive* rather than abstain,
witnessed by an **xfail** (`generate_readout_nonlinearity_decoy`). The other three
rows are genuine battery cases NUDGE resists (`off-model` / `no-effect`).

## Identifiability regime

Characterised through the general `self_activation_switch` motif (same topology):
Tier-0.5 single-basin fit is **fail-safe** (zero wrong positives, seeds 0–3;
FINDINGS T0.5-1), and the **saddle transition-mode gain gate recovers gain at all
four seeds** (free-`n` `w_trans` 0.87–0.94 vs ≤ 0.12 for other classes; FINDINGS
T0.5-5). The general power rule applies: ≥ ~1000 cells/condition, gain > ceiling ≈
threshold. Whether the true topology is 1-node or 2-node is itself decided by
`model_select`, not assumed.

## Implementation Mapping

| Equation term | Code |
|---|---|
| the circuit builder | `nudge.circuits.ras_switch_1node` |
| self-edge response | `nudge.mechanisms.regulatory.HillActivationEffect.response` |
| LOW/HIGH basins + saddle | `nudge.core.circuit.Circuit.fixed_points` |

*(Machine-checked by `scripts/check_impl_mapping.py`.)*

## Verification evidence

- `tests/core/test_fixed_points.py::test_bistable_switch_three_fixed_points` — the
  1-node switch has LOW/saddle/HIGH fixed points.
- `tests/verification/test_stochastic_inverse_crime.py::test_saddle_transition_recovers_gain_never_wrong`
  — fail-safe gain recovery on emergent bistability of this topology.

## References

- [@Das2009] — digital, hysteretic Ras activation in lymphocytes (the source biology).
- [@HuangFerrell1996] — ultrasensitivity from cooperative feedback.
