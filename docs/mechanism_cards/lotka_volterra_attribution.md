---
id: NUDGE-METHOD-012
name: lotka_volterra_attribution
role: attribution-method
registry_name: LotkaVolterraAttribution
vulnerable_to_decoys: []
documented_limitation: [NUDGE-LIM-020]
validated_in_regime: {min_replicates: 20, min_timepoints: 15, notes: "Synthetic gLV round-trip (3 taxa, dense-transient sampling). The antibiotic-susceptibility ε axis is the identifiable positive (recovered across seeds); a growth α change is recovered ONLY when the transient is densely sampled, else it abstains; a self-limitation βᵢᵢ change and a near-equilibrium growth change ABSTAIN (unresolved) because α⇄βᵢᵢ is degenerate (Kᵢ=−αᵢ/βᵢᵢ), with the degeneracy MEASURED by the Laplace curvature (condition number ≫ 100, |corr|→1). 0 confident-wrong across the battery. Real coda: Stein et al. 2013 clindamycin→C. difficile (11 taxa), structured as needs_data — the honest expectation is a direct-kill ε positive on the strongly-susceptible taxa and an ABSTENTION on C. difficile, whose bloom is interaction-mediated (published ε≈−0.31, near zero)."}
references: [Stein2013, Bucci2016MDSINE, Volterra1926, Buffie2015]
---

# Mechanism Card — Lotka–Volterra (temporal) attribution

> **ID:** `NUDGE-METHOD-012`  ·  **Role:** attribution-method
> **Stability:** experimental  ·  **Registry name:** `LotkaVolterraAttribution`

## Summary

The **temporal / trajectory-fit** attribution capability — the extensibility thesis made
concrete. Everything else in NUDGE observes a **steady-state snapshot** and attributes a
switch's `K` / `n` / `v_max`; this points the *same* attribution philosophy at a different
dynamical system, a **generalized Lotka–Volterra (gLV) microbial community**, whose
parameter information lives in **transients** rather than an equilibrium. Given a reference
community and a perturbed one (the same external antibiotic pulse `u(t)`, one knob of a
target taxon moved), it BIC-selects **which single knob moved** — **growth `α`**,
**interaction `β`**, or **antibiotic-susceptibility `ε`** — or **abstains** (`no-change` /
`unresolved`) when the perturbation is unidentifiable. It reuses NUDGE's mechanism-agnostic
scaffolding — the distributional energy distance, the BIC restricted-fit parsimony pattern,
and the Laplace/Fisher identifiability guard — while touching **neither `fit.py` nor
`core/circuit.py`** (`design/EXTENSIBILITY_SPIKE.md`, `design/MICROBIOME_DATA_GATE.md`).

## Why this exists (same engine, new field of biology)

The deferred **Capability 4 (temporal)** was shelved because scRNA-seq is destructive — a
fresh distribution per timepoint, never a tracked trajectory. **16S longitudinal microbiome
data provides real per-community trajectories with a designed perturbation contrast**
(antibiotic pulses), which unlocks it. The attribution NUDGE makes is deliberately narrow —
"for *this* time-localized perturbation, which of {α, β, ε} moved?" — not a full β-matrix
reconstruction (famously ill-posed). Fitting the trajectory (not a snapshot) is a **new
observable**, so the fit loop is re-instantiated in the new module rather than reused from
`fit()`, exactly as `lyapunov.py` / `dose_response.py` re-instantiate their own fits.

## Governing equation

```
dxᵢ/dt = xᵢ · (αᵢ + Σⱼ βᵢⱼ xⱼ + εᵢ · u(t))
```

- `xᵢ` — absolute abundance / density of taxon `i`.
- `αᵢ` — intrinsic **growth** rate.
- `βᵢⱼ` — **interaction** of taxon `j` on `i`; `βᵢᵢ < 0` is self-limitation, giving the
  carrying capacity `Kᵢ = −αᵢ/βᵢᵢ`.
- `εᵢ` — **susceptibility** of taxon `i` to a known external input `u(t)` (an antibiotic
  pulse). `εᵢ·u` acts **only while the drug is on** — the on/off contrast that makes `ε`
  the identifiable axis.

The field is integrated with a self-contained differentiable RK4 `lax.scan` (no `diffrax`
dependency); gradients flow to the kinetics for the trajectory fit.

## The classifier (fail-safe, in order)

1. **no-change** — no single-knob model beats the no-change null by the BIC margin (the
   perturbation is inert or not captured by an α/β/ε change).
2. **unresolved — the α⇄βᵢᵢ degeneracy (`NUDGE-LIM-020`).** The winning knob is `growth`
   or `interaction` AND the Laplace curvature on the pair `(αₜ, βₜₜ)` is near-singular
   (high condition number / a flat direction). `Kᵢ = −αᵢ/βᵢᵢ` means a growth change and a
   carrying-capacity change give the same steady state — separable only by the transient,
   which this sampling does not resolve. The abstention is **measured, not asserted**.
3. **unresolved — the two best knobs tie.** The winner does not beat the runner-up by the
   resolve margin — which knob moved is unidentifiable; abstain rather than guess.
4. **growth / interaction / susceptibility** — the winner earns its parameter over the
   null AND beats the runner-up AND (if in the confounded pair) is identifiable.

Scoring is done on the reference→perturbed **contrast** (`log(pert_mean) − log(ref_mean)`),
which cancels the baseline fit's mean-bias so a null cannot be beaten by a spurious knob.

## Assumptions & simplifications

- The community is (approximately) gLV with a known, time-localized external input `u(t)`;
  the reference pins the baseline kinetics (recovered from its own replicates), so the
  attribution is a **contrast** question relative to that reference.
- The perturbation moves **one** knob of **one** target taxon; multi-knob or
  multi-taxon perturbations are out of scope (the restricted-fit vocabulary is single-knob).
- gLV inference is ill-posed; abstention is on-thesis and expected, especially separating
  a diet-like α change from a βᵢᵢ change near equilibrium.

## Known failure modes

| Failure mode | Guard / witness | Limitation |
|---|---|---|
| A growth (α) change near equilibrium mis-called as an interaction (βᵢᵢ) change | `nudge.inference.lotka_volterra.generate_alpha_beta_confound_decoy` → `unresolved` (measured Laplace degeneracy) | `NUDGE-LIM-020` |
| Manufacturing a mechanism from a no-perturbation null | `nudge.inference.lotka_volterra.generate_no_perturbation_null` → `no-change` / `unresolved` | `NUDGE-LIM-020` |
| A self-limitation (βᵢᵢ) change reported as identifiable when it is confounded with α | the Laplace degeneracy gate abstains (`unresolved`) | `NUDGE-LIM-020` |
| A diverging gLV orbit NaN-ing the loss | abundance clipped `[0, 1e6]` inside the integrator (a bad fit, not a crash) | `NUDGE-LIM-020` |

There is **no Circuit-style decoy-battery entry** (`vulnerable_to_decoys: []`) because the
gLV decoys are trajectory-based, not the count/AnnData shape of `DECOY_BATTERY`; they live
as generator functions with dedicated `decoy`-marked tests (below).

## Identifiability regime

- **≥ ~20 replicate communities and ≥ ~15 timepoints**, with the transient densely sampled
  where an α-vs-β call is wanted.
- **The ε axis is the most identifiable** — the drug window is a time-localized on/off
  contrast, so a direct-kill signature is distinct from a constant growth/interaction shift.
  This is where a demoable positive appears (recovered across seeds on synthetic ground
  truth).
- **α ⇄ βᵢᵢ is degenerate near equilibrium** (`Kᵢ = −αᵢ/βᵢᵢ`) — separable only by the
  transient. The Laplace curvature on `(αₜ, βₜₜ)` measures it: near-singular (condition
  number ≫ 100, `|corr| → 1`) near equilibrium, better-conditioned when the transient is
  sampled. **0 confident-wrong** across the synthetic battery.
- **Real coda (Stein et al. 2013, structured `needs_data`).** The clindamycin→*C. difficile*
  time-series (11 taxa). The honest expectation: a direct-kill **ε positive** on the
  strongly-susceptible taxa, and an **abstention** on *C. difficile* itself, whose bloom is
  interaction-mediated (the paper's fitted ε ≈ −0.31 is near zero) — the confound surfaced,
  not hidden.

## Implementation Mapping

| Step | Code |
|---|---|
| gLV vector field | `nudge.inference.lotka_volterra.glv_vector_field` |
| differentiable RK4 trajectory integrator | `nudge.inference.lotka_volterra.simulate_glv` |
| synthetic generator (known single-knob perturbation) | `nudge.inference.lotka_volterra.simulate_glv_perturbseq` |
| baseline gLV fit (energy distance over trajectory ensembles) | `nudge.inference.lotka_volterra.fit_baseline_glv` |
| restricted-fit + BIC + identifiability | `nudge.inference.lotka_volterra.fit_glv_attribution` |
| the α⇄βᵢᵢ Laplace degeneracy measurement | `nudge.inference.lotka_volterra.alpha_beta_identifiability` |
| fail-safe classifier | `nudge.inference.lotka_volterra.classify_glv` |
| fit + classify (entry point) | `nudge.inference.lotka_volterra.attribute_glv` |
| α⇄βᵢᵢ confound decoy | `nudge.inference.lotka_volterra.generate_alpha_beta_confound_decoy` |
| no-perturbation null | `nudge.inference.lotka_volterra.generate_no_perturbation_null` |
| reused distributional loss | `nudge.inference.losses.energy_distance` |
| reused Laplace/Fisher guard | `nudge.inference.uncertainty.laplace_posterior` |

*(This table is machine-checked by `scripts/check_impl_mapping.py`: every `nudge.*`
reference must resolve to a real attribute.)*

## Verification evidence

- `tests/inference/test_lotka_volterra.py::test_susceptibility_is_the_identifiable_positive`
  — the ε axis recovers as a positive.
- `tests/inference/test_lotka_volterra.py::test_growth_resolves_when_the_transient_is_sampled`
  — a growth change is recovered when the transient is sampled (else abstains).
- `tests/inference/test_lotka_volterra.py::test_alpha_beta_confound_decoy_abstains` — the
  α⇄βᵢᵢ confound decoy abstains with a MEASURED degeneracy.
- `tests/inference/test_lotka_volterra.py::test_no_perturbation_null_makes_no_positive_call`
  — the null makes no positive call.
- `tests/inference/test_lotka_volterra.py::test_battery_has_zero_confident_wrong` — the
  headline fail-safe guarantee across the mixed battery.
- `tests/inference/test_lotka_volterra.py::test_laplace_curvature_measures_the_alpha_beta_degeneracy`
  — the abstention is grounded in the measured Laplace curvature.

## References

- [@Volterra1926] — the original Lotka–Volterra predator–prey dynamics.
- [@Stein2013] — gLV inference from intestinal-microbiota time-series with an explicit
  antibiotic (clindamycin) perturbation term (the real coda dataset).
- [@Bucci2016MDSINE] — MDSINE: gLV inference with an external-perturbation susceptibility
  term (the ε axis).
- [@Buffie2015] — the clindamycin → *C. difficile* biology the Stein series models.
