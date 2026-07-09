---
id: NUDGE-METHOD-001
name: dose_response_attribution
role: attribution-method
registry_name: DoseResponseAttribution
vulnerable_to_decoys: []
documented_limitation: [NUDGE-LIM-004, NUDGE-LIM-005, NUDGE-LIM-006, NUDGE-LIM-007]
validated_in_regime: {min_dose_points: 4, notes: "The dose series must SPAN the inflection (fitted K within the observed dose range) or the method abstains (NUDGE-LIM-007). n is reported as an APPARENT population gain + bootstrap CI, not molecular cooperativity. Validated on OCT4/NANOG (GSE283614): OCT4 resolves as a switch (n≈6.7, R²=0.99); NANOG correctly abstains (unresolved) because its knockdown reaches only ~75% and does not span its threshold."}
references: [HuangFerrell1996, Das2009, Niwa2000, Chambers2007, Yao2025]
---

# Mechanism Card — Dose-response attribution

> **ID:** `NUDGE-METHOD-001`  ·  **Role:** attribution-method
> **Stability:** experimental  ·  **Registry name:** `DoseResponseAttribution`

## Summary

Attributes a mechanism from a **dose-response curve** — the same `K` (threshold) /
`n` (gain) / `v_max` (ceiling) vocabulary as single-cell attribution, read from a
**dose axis** instead of a single-cell distribution. It fits a Hill curve to a readout's
response across a graded perturbation dose and returns **switch** (ultrasensitive),
**graded** (`n ≈ 1`), **no-effect**, or **unresolved** — abstaining rather than
over-calling an unidentifiable curve.

## Why this exists (two measurements of one circuit)

Single-cell bimodality and bulk / pseudobulk dose-response ultrasensitivity are two
*measurements of the same physical Hill circuit*. Where a single-cell distribution is not
bimodal — so the Lyapunov covariance path correctly abstains — the
ultrasensitivity can still live in the dose-response. In NUDGE's terms a **dose axis is a
set of operating points**: the Fisher-information result that a *second* operating point
breaks the gain⇄threshold degeneracy (`scripts/vv/FINDINGS.md`) is exactly why a dose
*series* can attribute a mechanism a single snapshot cannot. The method reuses the same
Hill primitive as the circuit vector field and the same BIC parsimony discipline as
topology model-selection — it is not a bolt-on curve-fitter but the same substrate read
through a different assay.

## Governing equation

```
response(dose) = floor + hill(dose, K, n, amp)
   repress:  hill = amp · Kⁿ / (Kⁿ + doseⁿ)      (readout falls with dose)
   activate: hill = amp · doseⁿ / (Kⁿ + doseⁿ)   (readout rises with dose)
```

- `dose` — the graded perturbation strength (e.g. fractional knockdown, compound dose).
- `K` — **half-max dose**, the switch **threshold** on the dose axis.
- `n` — **apparent population gain** (Hill coefficient). `n ≈ 1` is graded; `n > 1` is
  switch-like. **NOT molecular cooperativity** — see honesty note below.
- `amp` — the response **span** (ceiling minus floor); `floor` — the baseline asymptote.

## The classifier (fail-safe, in order)

1. **no-effect** — the fitted amplitude is within `noise_amp_ratio ×` the residual noise
   (an inert perturbation; nothing to fit).
2. **unresolved** — even the better model has `R² < min_r2`; *or* the bootstrap `n` CI is
   undefined; *or* the doses **do not span the inflection** (`K` outside the dose range →
   one arm of a sigmoid, on which gain is unidentifiable — NUDGE-LIM-007); *or* the `n`
   CI straddles the switch/graded line.
3. **switch** — free `n` beats the `n = 1` graded model by a **BIC margin** *and* the
   whole bootstrap `n` CI clears the ultrasensitive line (a conservative two-condition
   call).
4. **graded** — the `n` CI sits at/below the line, or free `n` is not justified over
   `n = 1` (a poor switch fit *is* the graded signature).

## Assumptions & simplifications

- The response is an (approximately) affine readout of one latent switch output; a
  nonlinear reporter can manufacture apparent ultrasensitivity (NUDGE-LIM-006).
- One dose point per condition/guide; per-cell depth is divided out by size-factor
  normalization before the pseudobulk mean (`nudge.inference.bridge.knockdown_dose_response`).
- `n` is an **apparent population gain**: pseudobulk conflates within-cell cooperativity
  with a *spread of single-cell thresholds*, so it is reported with a CI and never as a
  molecular Hill coefficient.

## Known failure modes

| Failure mode | Guard / witness | Limitation |
|---|---|---|
| A dose series that stops before the inflection over-called as switch/graded | OCT4/NANOG regression (`tests/inference/test_dose_response.py`) | `NUDGE-LIM-007` |
| A nonlinear reporter faking dose-response ultrasensitivity | `nudge.data.decoy_generators.generate_readout_nonlinearity_decoy` (xfail witness) | `NUDGE-LIM-006` |
| A marginal nonlinearity (`n` just above 1) over-called as a switch | BIC parsimony gate (free-n must beat `n = 1`) | `NUDGE-LIM-005` |
| An inert perturbation reported as a mechanism | the no-effect gate (amplitude within noise) | `NUDGE-LIM-004` |

There is **no dedicated dose-response decoy battery yet** (`vulnerable_to_decoys: []`) —
the failure modes above are currently guarded by the classifier's gates plus the
real-data OCT4/NANOG regression; a synthetic dose-response decoy battery is future work.

## Identifiability regime

- **≥ 4 dose points** (a Hill curve has 4 parameters); more importantly the series must
  **span the inflection** — a fit to one arm is unidentifiable and returns `unresolved`.
- **Verified on real data (GSE283614).** OCT4 self-renewal vs its own knockdown is a
  resolved switch (apparent `n ≈ 6.7`, `K ≈ 0.65`, R² = 0.99); its inflection is inside
  the knockdown range. **NANOG correctly abstains** — its knockdown reaches only ~75%, its
  fitted `K` sits past the maximum dose, and an independent `n`-profile shows R² flat
  within 0.075 across `n = 1…12` (a graded `n ≈ 1` and a high-threshold switch fit
  equally well). A naive bounded Hill fit reports a misleadingly graded `n ≈ 2.2`; NUDGE
  reports `unresolved`. This is the fail-safe catching the classic human over-reading of
  an under-determined curve (NUDGE-LIM-007).

## Implementation Mapping

| Step | Code |
|---|---|
| fit Hill + graded sibling (autodiff-Jacobian, multi-start) | `nudge.inference.dose_response.fit_dose_response` |
| classify switch / graded / no-effect / unresolved | `nudge.inference.dose_response.classify_dose_response` |
| fit + classify (CLI/MCP entry point) | `nudge.inference.dose_response.attribute_dose_response` |
| knockdown screen → `(dose, response)` points | `nudge.inference.bridge.knockdown_dose_response` |
| shared reused Hill primitive | `nudge.mechanisms.regulatory.hill_repression` |
| CLI / MCP orchestration | `nudge.service.dose_response_file` |

*(This table is machine-checked by `scripts/check_impl_mapping.py`: every `nudge.*`
reference must resolve to a real attribute.)*

## Verification evidence

- `tests/inference/test_dose_response.py::test_high_gain_curve_is_a_switch` — a genuine
  high-gain curve reads as a switch.
- `tests/inference/test_dose_response.py::test_autodiff_jacobian_lets_n_move_off_its_seed`
  — regression pinning the float32 finite-difference bug (n must not freeze at init).
- `tests/inference/test_dose_response.py::test_one_arm_curve_is_unresolved` — a curve that
  does not span the inflection abstains (NUDGE-LIM-007).
- `tests/inference/test_dose_response.py::test_oct4_switch_nanog_unresolved_real_data` —
  the OCT4=switch / NANOG=unresolved lock-in on real GSE283614 data.

## References

- [@HuangFerrell1996] — ultrasensitivity / Hill-gain as the switch signature.
- [@Das2009] — digital, hysteretic Ras activation (the switch biology NUDGE targets).
- [@Niwa2000] — quantitative OCT4 dose defines ES-cell fate (the threshold half of the demo).
- [@Chambers2007] — Nanog as a graded / heterogeneous pluripotency factor (the graded foil).
- [@Yao2025] — dose-dependent perturbation resolving threshold vs graded self-renewal control (the OCT4/NANOG source dataset, GSE283614).
