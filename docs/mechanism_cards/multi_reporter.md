---
id: NUDGE-METHOD-008
name: multi_reporter
role: attribution-method
registry_name: MultiReporterAttribution
vulnerable_to_decoys: []
documented_limitation: [NUDGE-LIM-006, NUDGE-LIM-007, NUDGE-LIM-014]
validated_in_regime: {min_dose_points: 4, min_reporters: 2, notes: "Fits several downstream reporters of ONE latent switch JOINTLY (each an affine readout y_j = base_j + gain_j·A·f(dose; K, n) of the same latent), which OVER-DETERMINES the latent and breaks the K⇄v_max / gain⇄threshold degeneracy (FINDINGS §2 / Phase 4h) a single reporter abstains on. The affine gains are pinned from the CONTROL panel (a reporter is a fixed measurement device); with a single reporter (M=1) the gain cannot be pinned or checked, so ceiling stays degenerate with reporter gain and NUDGE returns unresolved. Validated on synthetic ground truth: a known threshold-only / gain-only / ceiling-only perturbation on one latent, seen through 4 heterogeneous-gain reporters — JOINT recovery 24/24 (100%) vs SINGLE-reporter 0/24, 0 confident-wrong calls. The consistency guard (NUDGE-LIM-014) abstains OFF-MODEL when a reporter reads a DIFFERENT latent. Reported n / K / A are APPARENT population parameters, not molecular constants; holds under an approximately-affine reporter (NUDGE-LIM-006)."}
references: [HuangFerrell1996, RazoMejia2018, ElfEhrenberg2003, Das2009]
---

# Mechanism Card — Multi-reporter joint attribution

> **ID:** `NUDGE-METHOD-008`  ·  **Role:** attribution-method
> **Stability:** experimental  ·  **Registry name:** `MultiReporterAttribution`

## Summary

Fits **several downstream reporters of ONE latent switch jointly** to break NUDGE's
dominant reason to abstain — the measured **K⇄v_max / gain⇄threshold degeneracy**
(`scripts/vv/FINDINGS.md` §2). A single reporter of one latent under-determines the
mechanism; a *panel* of reporters with heterogeneous gains over-determines the shared
latent, so the joint fit **resolves** whether a perturbation moved the switch's
**threshold** (`K`), **gain** (`n`), or **ceiling** (`v_max`) where a single reporter is
degenerate. It is the multi-*reporter* analogue of the second-operating-*point* ×16
degeneracy-break (`FINDINGS` "Covariance attribution" M3) — more observers of one latent,
instead of more conditions.

## Why this exists (the identifiability force-multiplier)

The reason one affine reporter `y = base + gain·activity` cannot separate a **threshold**
shift from a **ceiling** change is structural: a change in the *latent's* ceiling is the
same free parameter as a change in the *reporter's own gain*. NUDGE therefore abstains
(the honest `unresolved` in the single-cell and dose-response paths). The fix FINDINGS
keeps naming is to observe the same latent through **many** reporters: a threshold shift
moves the inflection **identically** across reporters, a gain change alters the shared
steepness, and a ceiling change scales **every** reporter's ON amplitude by the *same
fraction* (because the reporter gains are pinned from the control). These project
**differently** onto a panel of heterogeneous gains, so a joint fit with one shared latent
identifies the mechanism — higher Fisher information, the degeneracy broken.

## Governing equation

Each reporter `j` is an affine readout of one shared latent switch `f(dose; K, n)`
(the shipped Hill primitive, normalized to a unit ceiling so the latent's max lives in a
shared amplitude `A`):

```
y_j(dose) = base_j + gain_j · A · f(dose; K, n)      # activate: f rises with dose
```

The reporter affine `(base_j, gain_j)` is **calibrated once from the CONTROL panel** and
**pinned** across conditions (a reporter is a fixed measurement device). The perturbation
is localized by which single **shared** knob best explains the perturbed panel:

```
threshold : free K  (n, A at WT)   — the inflection shifts identically across reporters
gain      : free n  (K, A at WT)   — the shared switch steepness changes
ceiling   : free A  (K, n at WT)   — every reporter's ON amplitude scales by A_perturbed
```

against the WT-latent null (`loss_no_effect`) and a fully-free reference (`loss_full`).

## The classifier (fail-safe, in order)

1. **off-model — the consistency guard (`NUDGE-LIM-014`).** The panel is *not* one shared
   latent: the shared-latent `panel_r2` is poor, a reporter fits its OWN Hill cleanly yet
   the shared latent explains it badly (it reads a *different* latent — a hidden node or a
   wrong panel), or the shared-vs-independent residual ratio is large. NUDGE abstains
   rather than average an inconsistent panel into a mechanism.
2. **unresolved (identifiability).** A **single reporter** (`M = 1`) cannot pin or check
   its affine, so the ceiling stays degenerate with the reporter gain (the very
   abstention the panel is built to break); *or* the winning knob does not beat the
   runner-up by the loss margin; *or* the winner's bootstrap CI straddles 0.
3. **no-effect.** The WT-latent null is nearly as good as the best knob — the perturbation
   did not move the shared latent above noise.
4. **threshold / gain / ceiling.** The winning knob beats the runner-up by `knob_margin`
   **and** the WT null by `effect_margin` **and** its bootstrap CI excludes 0.

## Assumptions & simplifications

- **The panel genuinely reports ONE latent switch (`NUDGE-LIM-014`).** Reporter
  inconsistency is *flagged* (`off-model`), never silently averaged. A reporter reading a
  different latent — a hidden node, a co-regulated but distinct switch, a mislabeled panel
  — must be caught, not blended into a confident call.
- **The reporter affine is a fixed measurement device** — `(base_j, gain_j)` is the same
  in the control and perturbed conditions. This is what makes the latent ceiling `A`
  identifiable; it is *checkable* only with ≥ 2 reporters (the consistency guard), which is
  why a single reporter honestly abstains.
- The reporters are (approximately) **affine** readouts of the latent; a nonlinear
  reporter can manufacture apparent ultrasensitivity (`NUDGE-LIM-006`), so a mechanism
  call holds only under an approximately-affine panel.
- The doses must **span the latent's inflection** (shared with the dose-response regime,
  `NUDGE-LIM-007`); reported `K` / `n` / `A` are **apparent population** parameters (with
  bootstrap CIs), not molecular constants.

## Known failure modes

| Failure mode | Guard / witness | Limitation |
|---|---|---|
| A reporter secretly reads a DIFFERENT latent (hidden node / wrong panel) | the consistency guard → `off-model` (`tests/inference/test_multi_reporter.py::test_inconsistent_panel_is_off_model`) | `NUDGE-LIM-014` |
| A single reporter forced to a mechanism (ceiling ⇄ gain degeneracy) | the `M = 1` → `unresolved` gate (`test_joint_resolves_where_single_abstains`) | `NUDGE-LIM-014` |
| A nonlinear reporter faking ultrasensitivity | the affine-readout bound shared with all attribution | `NUDGE-LIM-006` |
| A dose series that does not span the inflection | inherits the dose-response identifiability regime | `NUDGE-LIM-007` |

There is **no dedicated multi-reporter decoy battery yet** (`vulnerable_to_decoys: []`) —
the failure modes above are guarded by the consistency guard, the single-reporter
abstention, and the synthetic degeneracy-break lock-in; a synthetic-panel decoy battery
is future work.

## Identifiability regime

- **≥ 2 reporters of one latent, each ≥ 4 dose points spanning the inflection**, with
  **heterogeneous gains** (the source of the identifiability); else the method abstains.
- **Validated on synthetic ground truth (the load-bearing result, `FINDINGS` Phase 4h).**
  A known **threshold-only / gain-only / ceiling-only** perturbation on ONE latent switch,
  observed through **4 reporters of heterogeneous gain**: the **JOINT** multi-reporter fit
  recovers the mechanism **24/24 (100%)** across seeds/mechanisms, while the **SINGLE**
  reporter resolves **0/24** (`unresolved`, the K⇄v_max degeneracy) — with **0
  confident-wrong calls** on either. The consistency guard abstains `off-model` on every
  inconsistent panel (a reporter with a shifted latent), across which reporter is the
  odd one out. A real-panel touch (e.g. an OCT4/NANOG self-renewal signature as a
  reporter panel of the pluripotency latent) is a deferred follow-up; the synthetic
  degeneracy-break is the load-bearing validation (we do **not** force a real-data claim).

## Implementation Mapping

| Step | Code |
|---|---|
| the shared-latent forward model (reuses the Hill primitive) | `nudge.mechanisms.regulatory.hill_activation` |
| each reporter as an affine readout of the latent activity | `nudge.mechanisms.readout.Readout` |
| calibrate + jointly fit the panel (control vs perturbed) | `nudge.inference.multi_reporter.fit_multi_reporter` |
| localize the perturbation to one shared knob (fail-safe classifier) | `nudge.inference.multi_reporter.classify_multi_reporter` |
| attribute a panel in one call | `nudge.inference.multi_reporter.attribute_multi_reporter` |
| synthetic ground-truth panel generator (the degeneracy-break demo) | `nudge.inference.multi_reporter.simulate_reporter_panel` |
| CLI / MCP orchestration | `nudge.service.multi_reporter_file` |

*(This table is machine-checked by `scripts/check_impl_mapping.py`: every `nudge.*`
reference must resolve to a real attribute.)*

## Verification evidence

- `tests/inference/test_multi_reporter.py::test_joint_resolves_where_single_abstains` —
  the headline: JOINT resolves threshold / gain / ceiling where SINGLE abstains.
- `tests/inference/test_multi_reporter.py::test_inconsistent_panel_is_off_model` — the
  consistency guard abstains `off-model` on a panel that is not one shared latent
  (`NUDGE-LIM-014`).
- `tests/inference/test_multi_reporter.py::test_fail_safe_never_a_confident_wrong_call` —
  0 confident-wrong mechanism calls across a mechanism / factor / noise / seed sweep.
- `tests/inference/test_multi_reporter.py::test_no_effect_reads_no_effect` — an unchanged
  latent reads `no-effect`, never a mechanism.
- `tests/test_service.py::test_multi_reporter_file_csv_wiring` — the tidy-CSV service
  round-trip the CLI / MCP share.

## References

- [@HuangFerrell1996] — ultrasensitivity / the K/n/v_max response-magnitude vocabulary
  shared with the single-cell, dose-response, and cross-modality attribution paths.
- [@RazoMejia2018] — a predictive multi-parameter switch (MWC induction) whose reporters
  are affine readouts of one latent allosteric state; the identifiability intuition for a
  shared-latent panel.
- [@ElfEhrenberg2003] — the linear-noise / Fisher-information framing behind "more
  observers of one latent lower the estimator variance" (the degeneracy-break mechanism).
- [@Das2009] — the bistable-switch motif whose threshold-vs-gain distinction this panel
  resolves.
