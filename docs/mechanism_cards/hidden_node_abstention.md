---
id: NUDGE-METHOD-009
name: hidden_node_abstention
role: attribution-method
registry_name: HiddenNodeAbstention
vulnerable_to_decoys: []
documented_limitation: [NUDGE-LIM-015, NUDGE-LIM-009, NUDGE-LIM-006]
validated_in_regime: {min_dose_points: 0, notes: "A pure KNOWLEDGE / packaging layer — it consumes verdicts + diagnostic scalars (the off-model parsimony verdict, the off-axis / neomorphic ratio, a readout-linearity flag) and returns a rank-ordered DIFFERENTIAL of candidate causes for an inadequate switch model. It runs no fit (ZERO import of nudge.inference.fit). It ships ONLY the honest ABSTENTION half of the hidden-node problem: positive hidden-node identification is NOT identifiable from an off-model verdict (the six causes are observationally overlapping — NUDGE-LIM-015), so it NEVER asserts a hidden node. The strongest statement it makes is that an off-axis residual is CONSISTENT WITH — but does NOT prove — an unmeasured regulator. Validated by the honesty test (the report never emits a bare positive hidden-node claim), an off-model differential enumerating all six causes with LIM/decoy refs, and an adequate-case no-differential test (tests/inference/test_hidden_node.py)."}
references: [Norman2019, HuangFerrell1996]
---

# Mechanism Card — Hidden-node abstention

> **ID:** `NUDGE-METHOD-009`  ·  **Role:** attribution-method
> **Stability:** experimental  ·  **Registry name:** `HiddenNodeAbstention`

## Summary

When NUDGE's switch model is **inadequate** — the circuit-level parsimony gate returns
`off-model`, or a diagnostic residual fires (the off-axis / neomorphic ratio) — a user is
left with a one-word verdict and no idea *why*. This method packages that evidence into a
legible **differential diagnosis**: it **enumerates the candidate causes** of the
inadequacy, each with its evidence, the documented limitation / decoy it maps to, and the
experiment that would distinguish it — turning a bare "off-model" into "here is *why* the
model is inadequate and *what to measure next*."

**It ships ONLY the abstention half.** The mirror-image capability — positively identifying
a hidden node — is a documented trap and NUDGE does **not** attempt it (see the honesty
crux below).

## Why this exists (a bare abstention is not actionable)

NUDGE's fail-safe is its crown jewel: it returns `off-model` rather than manufacture a
switch. But `off-model` alone does not tell a screen-analysis user what to do next — and
the one thing they are most tempted to conclude (a leftover residual *is* a newly-discovered
hidden regulator) is exactly the thing the data cannot support. This method makes the
abstention **legible without letting it become an over-claim**: it lays out the full
differential of what *could* make the model inadequate, ranks the hypotheses by the
evidence actually present, and — most importantly — refuses to collapse the differential
into a positive hidden-node verdict.

## Governing "equation" (a differential, not a fit)

There is no fit here — the method is a deterministic map from **evidence** to a
**rank-ordered differential**. Given the off-model verdict and optional diagnostic signals,
it emits six candidate causes:

```
1. genuinely not-a-switch (linear circuit)   evidence: parsimony gate rejected the switch
2. nonlinear measurement readout             evidence: readout-linearity flag / affine assumption
3. off-target perturbation effect            evidence: large restricted-fit residual
4. wrong / misspecified topology             evidence: topology-adequacy uncertainty (T0.5-2)
5. batch / depth confound                    evidence: depth/batch aligned with condition
6. hidden node / unmeasured regulator        evidence: off-axis / neomorphic residual (HEDGED)
```

Each cause carries a coarse `qualitative_rank` (`leading` / `plausible` / `less-likely`) —
**not a probability**. The hidden-node cause's rank is **capped at `plausible`** so it is
never the lone leading answer; the parsimony-gate's own reading ("not a switch") and the
readout confound lead when the evidence warrants.

## The classifier (fail-safe, honesty-first)

1. **Adequate** — if there is no `off-model` verdict *and* no diagnostic residual fires,
   the method reports `is_adequate=True` with **no causes** and invents no differential.
2. **Inadequate** — the `off-model` verdict, or a fired residual (a `neomorphic_ratio` at/
   above threshold, a set readout flag, a finite restricted-fit residual, a flagged
   topology / depth confound), triggers the full six-cause differential.
3. **The hidden-node cause is always hedged** — its evidence string is phrased *consistent
   with — does not prove*, and points at the only thing that could establish a hidden node:
   a positive **measurement** of the candidate regulator, not a residual.

## Assumptions & simplifications

- **Positive hidden-node identification is NOT identifiable (NUDGE-LIM-015).** The six
  causes are observationally overlapping — they all present as "the affine switch model
  does not fit" — and there is essentially no real Perturb-seq data with a *known* hidden
  node to calibrate a positive detector against. So NUDGE ships only the differential; it
  **never** asserts a hidden node.
- **The off-axis residual can only UNDER-count, never discover (NUDGE-LIM-009).** A scalar
  along the additive axis cannot see a purely orthogonal emergent state; a large off-axis
  residual is a magnitude statement about non-additivity, not a hidden-node claim.
- **The readout confound is a live alternative (NUDGE-LIM-006).** A nonlinear reporter can
  manufacture apparent ultrasensitivity, so it sits in the differential alongside the
  circuit-side causes; a constitutive-reporter control is the distinguishing experiment.
- **It runs no fit.** The method consumes verdicts/evidence and has **zero import of**
  `nudge.inference.fit` — it never re-attributes, never touches the decoy battery, and is
  fully additive / opt-in.
- **The ranks are qualitative.** `leading` / `plausible` / `less-likely` order the
  hypotheses; they are not a posterior over causes (that would over-claim identifiability).

## Known failure modes

| Failure mode | Guard / witness | Limitation |
|---|---|---|
| Reading the differential as a positive "hidden node detected" claim | the hedged hidden-node wording (*consistent with, does not prove*) + the honesty test (`tests/inference/test_hidden_node.py`) | `NUDGE-LIM-015` |
| Over-reading an off-axis residual as a discovery | the rank cap on the hidden-node cause + the *under-count-only* framing | `NUDGE-LIM-009` |
| Attributing the inadequacy to the circuit when a nonlinear readout is the cause | the readout cause is always enumerated with a constitutive-control experiment | `NUDGE-LIM-006` |

There is **no dedicated hidden-node decoy battery** (`vulnerable_to_decoys: []`) — the
capability's whole point is that a positive detector would be un-decoyable (the causes are
observationally overlapping), so it ships only the abstention + differential, guarded by the
honesty test rather than a synthetic positive case.

## Identifiability regime

- **Consumes an existing verdict** (the `off-model` parsimony call, from
  `nudge.inference.classify.switch_detected`) plus any available diagnostic scalars (the
  off-axis / neomorphic ratio from `nudge.inference.epistasis.combo_geometry`, a readout-
  linearity flag, a restricted-fit residual). It adds no data requirement of its own.
- **Ships only the abstention half.** Positive hidden-node identification is out of regime
  by design (`NUDGE-LIM-015`): the six causes cannot be separated from an off-model verdict,
  and no synthetic-recovery or real-data validation of a positive detector is claimed
  (there is essentially no real data with a known hidden node). The *distinguishing
  experiments* named per cause (a second guide, a constitutive control, a topology refit, a
  depth-balanced re-run, adding the candidate regulator to the panel) are the honest path
  forward — each converts a differential hypothesis into a testable measurement.

## Implementation Mapping

| Step | Code |
|---|---|
| one candidate cause (hypothesis + evidence + distinguishing experiment) | `nudge.inference.hidden_node.CandidateCause` |
| the rank-ordered differential (adequate → no causes) | `nudge.inference.hidden_node.InadequacyReport` |
| map the evidence → the ranked differential (never a positive claim) | `nudge.inference.hidden_node.diagnose_inadequacy` |
| the off-model parsimony verdict this consumes | `nudge.inference.classify.switch_detected` |
| the off-axis / neomorphic residual it reads (flag, not claim) | `nudge.inference.epistasis.combo_geometry` |
| enrich each cause with its limitation title (read-only knowledge) | `nudge.knowledge.explain` |
| CLI / MCP orchestration | `nudge.service.diagnose_abstention` |

*(This table is machine-checked by `scripts/check_impl_mapping.py`: every `nudge.*`
reference must resolve to a real attribute.)*

## Verification evidence

- `tests/inference/test_hidden_node.py::test_off_model_enumerates_the_full_differential` —
  an off-model case returns a legible differential enumerating all six causes, each with its
  `NUDGE-LIM-*` / `NUDGE-DECOY-*` refs.
- `tests/inference/test_hidden_node.py::test_adequate_model_emits_no_differential` — an
  adequate case returns `is_adequate=True` with no causes.
- `tests/inference/test_hidden_node.py::test_never_emits_a_positive_hidden_node_claim` — the
  honesty test: the strongest hidden-node string is hedged (*consistent with, does not
  prove*); no surface asserts a hidden node exists (`NUDGE-LIM-015`).
- `tests/inference/test_hidden_node.py::test_service_round_trip` — the `service` /
  differential round-trip (the CLI / MCP path) preserves the abstention-half-only guarantee.

## References

- [@Norman2019] — the CRISPRa combination screen whose *neomorphic* dimension motivates the
  off-axis residual that this differential enumerates (as a flag, never a hidden-node claim).
- [@HuangFerrell1996] — the ultrasensitivity / `K`/`n`/`v_max` vocabulary the underlying
  switch model attributes, whose inadequacy this method makes legible.
