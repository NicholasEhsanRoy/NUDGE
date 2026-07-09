---
id: NUDGE-METHOD-003
name: epistasis_attribution
role: attribution-method
registry_name: SynergyAttribution
vulnerable_to_decoys: []
documented_limitation: [NUDGE-LIM-006, NUDGE-LIM-009]
validated_in_regime: {min_cells_per_condition: 30, notes: "Both single arms must be measurable (a combo inherits its weakest arm) and the readout approximately affine. Effects are measured in log-fold-change space, so the additive null is Bliss independence. Validated on Norman 2019 (GSE133344, CRISPRa, K562): CBL+CNN1 and CBL+UBASH3B call SYNERGISTIC (the paper's emergent erythroid synergy); DUSP9+ETS2 calls BUFFERING (the paper's DUSP9-dominant epistatic suppression of ETS2); FOXA1+FOXA3 calls ADDITIVE (interaction CI straddles 0)."}
references: [Norman2019, Bliss1939, HuangFerrell1996]
---

# Mechanism Card — Synergy / epistasis attribution

> **ID:** `NUDGE-METHOD-003`  ·  **Role:** attribution-method
> **Stability:** experimental  ·  **Registry name:** `SynergyAttribution`

## Summary

For a combination of two perturbations, attributes the **interaction**: is A+B's effect
**additive** (the two act on the same knob — just more of it) or **non-additive**
(**synergistic** super-additivity, or **buffering** / epistatic sub-additivity)? NUDGE
reads A, B and A+B as three **operating points** against a shared control, reduces each
to a scalar **effect** (a response-magnitude shift of a signature vs control), forms the
**additive null** `effect(A+B) = effect(A) + effect(B)`, and returns **additive** /
**synergistic** / **buffering** / **no-effect** / **unresolved** — abstaining rather than
over-calling an interaction whose components it cannot trust.

## Why this exists (a combination is three operating points)

Combination-therapy and genetic-interaction labs ask exactly this question, and — like
the switch-vs-threshold distinction — a linear screen analysis structurally cannot make
it: additivity is a statement about *how two effects compose*, which a per-gene
differential test never models. NUDGE already treats a dose axis as a set of operating
points (`NUDGE-METHOD-001`); a combination is the same idea with three points (A, B,
A+B) sharing one control. The interaction is the residual of the combo against the
additive prediction, carried with a bootstrap CI over cells.

## Governing equation

```
effect(X)      = mean(score | X) − mean(score | control)          (log-fold-change space)
additive null  = effect(A) + effect(B)                            (Bliss independence)
interaction    = effect(A+B) − [effect(A) + effect(B)]
```

- `score` — a per-cell scalar: the projection of a cell onto the **additive axis fixed by
  the two single arms** (`u = (vA+vB)/‖vA+vB‖`, computed from the singles only, never the
  combo), or the mean of a pre-specified signature gene set. Both are depth-normalized
  (size-factor to the median) and `log1p`-transformed.
- **Effect space is log-fold-change**, so the additive null `e(A)+e(B)` is **Bliss
  independence** — multiplicative in linear expression space [@Bliss1939]. The choice is
  reported with every call (`EpistasisFit.effect_space`) because a different space (HSA,
  raw counts) moves the additive baseline.
- Because `u` points along the singles' additive direction, a **positive** interaction is
  unambiguously **super-additive** and a **negative** one **sub-additive** — the
  synergistic/buffering labels are direction-safe by construction.

## The classifier (fail-safe, in order)

1. **unresolved** — a condition has fewer than `min_cells` cells (a combo inherits its
   weakest single arm — NUDGE-LIM-009), or the interaction CI is undefined.
2. **no-effect** — *both* single-arm CIs straddle 0 (neither perturbation moved the
   signature; there is no interaction to attribute).
3. **additive** — the interaction CI **straddles 0** *and* is tight (half-width ≤
   `rel_width ×` the single-arm effect scale) *and* a free A+B level does not beat the
   additive null by the BIC margin (same knob, more of it).
4. **synergistic** / **buffering** — the interaction CI is entirely **> 0** / **< 0**
   *and* a free A+B level beats the additive null by the **BIC margin** (both the CI *and*
   parsimony must agree — a conservative two-condition call).
5. **unresolved** — otherwise: a CI that straddles 0 but is too wide to *rule out* synergy
   (underpowered), or a clearly-nonzero CI the BIC cannot justify.

## Assumptions & simplifications

- **Both single arms must be correctly measured.** The additive null is only as good as
  `effect(A)` and `effect(B)`; a dead/underpowered arm makes the combo `unresolved`.
- The score is an (approximately) affine readout of the response; a nonlinear reporter
  can manufacture apparent non-additivity (NUDGE-LIM-006).
- The scalar interaction is measured **along the singles' additive axis**. A purely
  *orthogonal* emergent state (a combo moving in a direction neither single spans) is not
  captured by this scalar and is a documented bound (NUDGE-LIM-009).
- The additive-direction `u` is fixed from the point estimate (the singles' means); the
  bootstrap resamples cells with `u` held fixed, so the CI reflects sampling of the means,
  not of the axis.

## Known failure modes

| Failure mode | Guard / witness | Limitation |
|---|---|---|
| A super-additive residual read as a hidden-node / rewiring claim | reason string refuses the leap; the interaction is scalar-only | `NUDGE-LIM-009` |
| A depth/batch confound between the A+B condition and the singles | depth-normalization + the min-cells & CI-width abstention gates | `NUDGE-LIM-009` |
| An underpowered combo over-called | the min-cells gate + CI-width straddle gate (`unresolved`) | `NUDGE-LIM-009` |
| A nonlinear readout faking non-additivity | the affine-readout bound shared with all attribution | `NUDGE-LIM-006` |

There is **no dedicated synergy decoy battery yet** (`vulnerable_to_decoys: []`) — the
failure modes above are guarded by the classifier's gates plus the real-data Norman
lock-in; a synthetic epistasis decoy battery is future work.

## Identifiability regime

- **≥ `min_cells` cells per condition** (default 30) in all four of {control, A, B, A+B},
  or the call abstains — the interaction inherits its weakest single arm.
- **Verified on real data (GSE133344, Norman 2019 CRISPRa in K562).** With the
  additive-axis projection extractor: **CBL+CNN1** interaction `+0.95` (95% CI
  `[+0.48, +1.42]`, ΔBIC 19) and **CBL+UBASH3B** `+1.09` (CI `[+0.75, +1.45]`, ΔBIC 44)
  call **synergistic** — the paper's emergent erythroid synergy. **DUSP9+ETS2**
  interaction `−2.14` (CI `[−2.64, −1.60]`, ΔBIC 156) calls **buffering**: the combo
  lands near DUSP9-alone (`+4.31` vs DUSP9's `+4.79`), matching the paper's report that
  the DUSP9 phenotype dominates and antagonises ETS2. **FOXA1+FOXA3** interaction `−0.61`
  (CI `[−1.37, +0.25]`, ΔBIC `−2`) calls **additive** — the CI straddles 0. See
  `scripts/vv/FINDINGS.md` "Phase 4d".

## Implementation Mapping

| Step | Code |
|---|---|
| fit the additive null + interaction + bootstrap CI + parsimony BIC | `nudge.inference.epistasis.fit_synergy` |
| classify additive / synergistic / buffering / no-effect / unresolved | `nudge.inference.epistasis.classify_synergy` |
| fit + classify (CLI/MCP entry point) | `nudge.inference.epistasis.attribute_synergy` |
| combination → per-cell effect scores (projection or signature) | `nudge.inference.bridge.combo_effect_scores` |
| shared BIC parsimony discipline | `nudge.inference.model_select.select_topology` |
| CLI / MCP orchestration | `nudge.service.synergy_file` |

*(This table is machine-checked by `scripts/check_impl_mapping.py`: every `nudge.*`
reference must resolve to a real attribute.)*

## Verification evidence

- `tests/inference/test_epistasis.py::test_additive_combo_reads_additive` — a combo on
  the additive line reads `additive`.
- `tests/inference/test_epistasis.py::test_super_additive_combo_reads_synergistic` — a
  super-additive combo reads `synergistic`.
- `tests/inference/test_epistasis.py::test_sub_additive_combo_reads_buffering` — a
  sub-additive combo reads `buffering`.
- `tests/inference/test_epistasis.py::test_underpowered_arm_abstains_unresolved` — a
  combo with a dead/underpowered arm abstains (NUDGE-LIM-009).
- `tests/inference/test_epistasis.py::test_norman_synergy_lockin_real_data` — the
  Norman 2019 GSE133344 lock-in (synergistic / buffering / additive vs the paper).

## References

- [@Norman2019] — the genetic-interaction manifold and taxonomy (the validation source,
  GSE133344; CBL/CNN1/UBASH3B synergy, DUSP9-dominant ETS2 suppression).
- [@Bliss1939] — Bliss independence: the multiplicative (log-additive) null this method
  uses as its additive baseline.
- [@HuangFerrell1996] — ultrasensitivity / the response-magnitude vocabulary shared with
  the dose-response and single-cell attribution paths.
