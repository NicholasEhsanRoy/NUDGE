# Mechanism Cards

Every mechanism in the NUDGE library ships with a **Mechanism Card** — the
literature-grounded, machine-checkable half of its documentation (the other half is
the in-code `MechanismMeta` in `src/nudge/core/metadata.py`). Each card carries YAML
front-matter with machine-readable relations (decoys, limitations, identifiability
regime, references) plus a fixed body: governing equation, what it represents,
assumptions, known failure modes → the decoy that exercises each, identifiability
regime (real FINDINGS numbers), an Implementation Mapping to real code, and
verification evidence.

The cards are validated in CI by three gates:

- `scripts/check_citations.py` — every Pandoc citation key resolves in `docs/bibliography.bib`.
- `scripts/check_impl_mapping.py` — every `` `nudge.*` `` qualname resolves to real code.
- `scripts/check_mechanism_cards.py` — every registered mechanism has a card whose
  front-matter `registry_name` matches, and every card's front-matter parses with the
  required keys (also run as `tests/docs/test_mechanism_cards.py`).

New cards are added via the `new-mechanism` skill. See `_template.md` for the skeleton.

## Primitives (registered mechanisms)

| Card | Role | ID | Mechanism (registry name) |
|---|---|---|---|
| [linear_effect](linear_effect.md) | regulatory-edge | `NUDGE-MECH-001` | `LinearEffect` |
| [hill_activation](hill_activation.md) | regulatory-edge | `NUDGE-MECH-002` | `HillActivation` |
| [hill_repression](hill_repression.md) | regulatory-edge | `NUDGE-MECH-003` | `HillRepression` |
| [linear_integrator](linear_integrator.md) | integrator | `NUDGE-MECH-010` | `LinearIntegrator` |
| [saturating_integrator](saturating_integrator.md) | integrator | `NUDGE-MECH-011` | `SaturatingIntegrator` |
| [readout](readout.md) | readout | `NUDGE-MECH-020` | `Readout` |

## Motifs (named circuits)

| Card | Role | ID | Mechanism (topology) |
|---|---|---|---|
| [ras_switch_1node](ras_switch_1node.md) | motif | `NUDGE-MOTIF-001` | 1-node self-activation Ras switch |
| [ras_switch_2node](ras_switch_2node.md) | motif | `NUDGE-MOTIF-002` | 2-node mutual-activation Ras switch |
| [toggle](toggle.md) | motif | `NUDGE-MOTIF-003` | 2-node mutual-repression toggle |
| [self_activation_switch](self_activation_switch.md) | motif | `NUDGE-MOTIF-004` | general 1-species positive-feedback switch |

## Methods (attribution capabilities)

| Card | Role | ID | Capability |
|---|---|---|---|
| [dose_response_attribution](dose_response_attribution.md) | attribution-method | `NUDGE-METHOD-001` | K/n/v_max attribution from a dose-response curve (switch / graded / abstain) |
| [cross_modality_readout](cross_modality_readout.md) | attribution-method | `NUDGE-METHOD-002` | The same K/n/v_max attribution on a continuous single-channel readout (fluorescence / activity / fold-change) |
| [epistasis_attribution](epistasis_attribution.md) | attribution-method | `NUDGE-METHOD-003` | Synergy / epistasis of a two-perturbation combo (additive / synergistic / buffering / abstain) |
| [bifurcation_proximity](bifurcation_proximity.md) | attribution-method | `NUDGE-METHOD-006` | Robustness dial: how close a bistable switch is to a saddle-node fold (near-fold / robust / unresolved / not-bistable) |
| [inverse_design](inverse_design.md) | attribution-method | `NUDGE-METHOD-007` | Inverse / intervention design: invert a reliable attribution to propose an intervention (a kinetic Δ or a dose), behind an integrity gate + a bifurcation safety gate |
| [multi_reporter](multi_reporter.md) | attribution-method | `NUDGE-METHOD-008` | Multi-reporter joint attribution: fit several reporters of ONE latent switch jointly to break the K⇄v_max degeneracy — resolves threshold / gain / ceiling where a single reporter abstains, and abstains off-model on an inconsistent panel (`NUDGE-LIM-014`) |
| [hidden_node_abstention](hidden_node_abstention.md) | attribution-method | `NUDGE-METHOD-009` | Hidden-node **abstention** (the abstention half ONLY): turn a bare `off-model` verdict into a legible **differential** of candidate causes (not-a-switch / nonlinear readout / off-target / wrong topology / batch-depth confound / hidden node) — NEVER a positive hidden-node claim (`NUDGE-LIM-015`) |
| [differential_attribution](differential_attribution.md) | attribution-method | `NUDGE-METHOD-010` | Comparative / differential attribution: the SAME perturbation in TWO contexts (resistant vs sensitive; donor A vs B) — BIC-select which single knob differs (threshold / gain / ceiling), or abstain; a depth/batch shift aligned with the context axis is guarded as a ceiling confound (`NUDGE-LIM-016`) |
