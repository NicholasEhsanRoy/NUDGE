---
id: NUDGE-METHOD-011
name: constitutive_control
role: attribution-method
registry_name: ConstitutiveControl
vulnerable_to_decoys: []
documented_limitation: [NUDGE-LIM-006, NUDGE-LIM-018]
validated_in_regime: {min_distinct_control_doses: 4, requires: "a constitutive-reporter control (the reporter driven at KNOWN activity doses, bypassing the circuit) + a known count model / floors / latent-input spread", notes: "Mitigates NUDGE-LIM-006 — a nonlinear measurement readout misattributed as a circuit switch. Only the composition readout∘circuit is observed, so from ONE population the circuit Hill n and the readout Hill h are unidentifiable (the profile over circuit n is FLAT: a graded n=1 fits as well as a real switch). A constitutive-reporter control drives the reporter at KNOWN activity doses, bypassing the circuit, and anchors the readout using READOUT parameters ONLY (the no-circuit-leak property: ∂(control loss)/∂(circuit params) ≡ 0, checked). A profile likelihood over circuit n WITHOUT vs WITH the control then breaks the degeneracy: WITH the control, 'no switch' (n=1) is REJECTED for a genuine circuit switch (Δloss ≫ the flat no-control span). Validated on synthetic ground truth: a TRUE switch (n=3) through a nonlinear reporter (h=6) → biological-switch (n=1 rejection ≈0.026 vs a flat no-control span ≈0.001, argmin off n=1); a LINEAR circuit (n=1), the LIM-006 false-positive hazard → unresolved (n=1 rejection ≈0, honest abstention); 0 confident-wrong across seeds. HONEST CAVEAT (NUDGE-LIM-018): the control REJECTS 'no switch' but does NOT point-identify the exact n (the circuit's internal K/n/vmax trade-off persists) — full point-ID needs a SECOND anchor (an input titration / circuit dose-response). Additive / opt-in: never touches fit()'s default; can only move a confident false positive toward a correct call or an abstention."}
references: [HuangFerrell1996, RazoMejia2018, Svensson2020]
---

# Mechanism Card — Constitutive-reporter calibration control

> **ID:** `NUDGE-METHOD-011`  ·  **Role:** attribution-method
> **Stability:** experimental  ·  **Registry name:** `ConstitutiveControl`

## Summary

Removes a known **confident-wrong** failure mode — `NUDGE-LIM-006`, *a nonlinear measurement
readout misattributed as a circuit switch*. NUDGE assumes an **affine** reporter
(`Λ = base + scale·activity`). A *linear* (non-switch) circuit read through a **nonlinear**
(saturating / sigmoidal Hill `h ≥ 1`) reporter produces a pseudo-bimodal count distribution
the affine-readout switch model can only explain by **bending the circuit** — a confident
false positive. This method adds an optional **constitutive-reporter control**: a calibration
population whose reporter is driven at **known activity doses**, *bypassing the circuit*. It
measures the reporter's own transfer function directly, **anchors the readout**, and lets a
**profile likelihood over the circuit Hill `n`** decide whether the observed ultrasensitivity
is **biological** (a real circuit switch) or lives in the **measurement**.

## Why this exists (an identifiability degeneracy, not a fitting bug)

Per cell, an input drives a circuit map `a = g(u; K, n, v_max)` (Hill), then a readout map
`Λ = R(a; K_m, h, V_max)` (Hill), then counts. **Only the composition `R∘g` is observed.**
From a single population you cannot factor the composition into its circuit and readout parts:
the profile likelihood over the circuit `n` is **FLAT** — a graded `n = 1` circuit (no switch;
*all* the nonlinearity in the reporter) fits as well as a true switch (measured span ≈ 0.001;
`design/CONSTITUTIVE_CONTROL.md` §1, `scripts/vv/FINDINGS.md`). **You cannot even tell a
circuit switch exists.** `NUDGE-LIM-006` is therefore a fundamental degeneracy, and the
sharpest bound on NUDGE's fail-safe guarantee.

## Governing equation

The composed forward model (self-contained; reuses the shipped Hill primitive
`nudge.mechanisms.regulatory.hill_activation`, matched by the energy distance
`nudge.inference.losses.energy_distance`):

```
u   ~ lognormal(mu_log, sd_log)                 latent input spread across cells (unobserved)
a   = basal + Hill(u; K, n, v_max)              circuit map   g(u)   (the biological switch)
Λ   = base  + Hill(a; K_m, h, V_max)            readout map   R(a)   (the measurement device)
y   ~ moment-matched NB(mean = Λ)               counts (observed for the circuit population)
```

The **constitutive control** observes `Λ = R(a_c)` at **known** driven activity `a_c` — it
depends on the **readout** parameters `{K_m, h, V_max}` **only**, never on `{K, n, v_max}`
(the load-bearing no-leak property, checked by `control_loss_circuit_gradient`: the gradient
of the control loss w.r.t. every circuit parameter is identically zero).

## The classifier (fail-safe, in order)

The profile fits, for each fixed circuit `n` on a grid, the remaining parameters
(`K, v_max, K_m, h, V_max`) minimizing the population energy distance — **WITHOUT** the
control (population term only) vs **WITH** it (population `+` a readout-only calibration term).

1. **no-confound.** The calibrated reporter is ~affine (`h` CI does not clear the affine
   line): there is no `NUDGE-LIM-006` confound; the default affine-readout attribution stands.
2. **biological-switch** — three conditions must **all** hold (a fail-safe conjunction): with
   the control anchoring the readout, the `n = 1` "no switch" loss clears an **absolute
   margin**; it is at least `structure_ratio ×` the **WITHOUT-control profile span** (the
   control *created* structure the degenerate profile lacked); and the with-control profile
   minimum sits **off** the `n = 1` end. The ultrasensitivity is then **biological**.
3. **unresolved.** A confound is present but the control does not decisively reject "no switch"
   (a narrow / weak / noisy control range, *or* the truth really is a graded / linear circuit
   whose apparent ultrasensitivity lives in the reporter). NUDGE **abstains** — turning the
   LIM-006 confident false positive into an **honest abstention**.

The method **never** emits a bare `threshold` / `gain` / `ceiling` (`is_confident_wrong` is
structurally always `False`): the strongest positive is *the switch is real*, not *which knob*.

## Assumptions & simplifications

- **The control drives the reporter independently of the circuit, at known/graded activity**
  (a constitutively-driven mCherry / synthetic-barcode reporter titration; a housekeeping
  proxy) — `design/CONSTITUTIVE_CONTROL.md` §3. Absent a control, NUDGE keeps the guarded
  affine default and `NUDGE-LIM-006` still bounds it; this method is **additive / opt-in**.
- **The calibration anchors the READOUT ONLY** — no circuit parameter can leak into it (the
  `∂/∂circuit ≡ 0` property). This is what makes the anchor legitimate.
- **A known count model + floors + latent-input spread** (`dispersion`, `basal`, `base`,
  `mu_log`, `sd_log`) are supplied; the method identifies whether a switch *exists*, not its
  exact kinetics, so it is robust to these being approximate.
- **The honest caveat (`NUDGE-LIM-018`, load-bearing).** The control **rejects "no switch"**
  but does **NOT point-identify** the circuit `n` (the profile argmin is *not* a reliable point
  estimate — recovered ≈ 5 vs true 3 in validation; the K/n/v_max trade-off persists). Full
  point-identification needs a **second anchor** (an input titration / circuit dose-response).
  Reported quantities are **apparent** population parameters, not molecular constants.

## Known failure modes

| Failure mode | Guard / witness | Limitation |
|---|---|---|
| A nonlinear readout faking a circuit switch (the confident false positive) | the constitutive control + the circuit-`n` profile → `biological-switch` or `unresolved`, never a bare knob (`tests/inference/test_constitutive.py`) | `NUDGE-LIM-006` |
| Over-reading the control as a POINT estimate of `n` | the `biological-switch` reason states it does NOT point-identify `n`; `test_true_switch_flips_to_biological_with_control` asserts the caveat | `NUDGE-LIM-018` |
| A narrow / weak / linear-circuit case where the control cannot reject "no switch" | the structure-ratio + absolute-margin conjunction → `unresolved` (`test_linear_circuit_lim006_hazard_abstains_not_confident_wrong`) | `NUDGE-LIM-018` |
| An ~affine reporter with no real confound | the calibration nonlinearity verdict → `no-confound` (`test_affine_reporter_is_no_confound`) | `NUDGE-LIM-006` |

No dedicated decoy battery entry yet (`vulnerable_to_decoys: []`); the LIM-006 hazard case
(a linear circuit through a nonlinear reporter, which must ABSTAIN) *is* the decoy-style
negative, and the existing `generate_readout_nonlinearity_decoy` remains the affine-readout
`fit()` witness for the un-mitigated bound.

## Identifiability regime

- **≥ 4 distinct constitutive control doses** (a Hill transfer function has 4 parameters),
  spanning the reporter's dynamic range; replicates per dose sharpen the calibration.
- **Validated on synthetic ground truth (the load-bearing result, `FINDINGS` "NUDGE-LIM-006
  mitigation — VALIDATED"; reproduced by `scripts/vv/constitutive_control.py`).** A TRUE
  switch (`n = 3`) through a nonlinear reporter (`h = 6`): WITHOUT the control the `n`-profile
  is FLAT (span ≈ 0.001 — degenerate); WITH it, `n = 1` is REJECTED (Δloss ≈ 0.026 ≫ the flat
  span, argmin off `n = 1`) → `biological-switch`. A LINEAR circuit (`n = 1`, the LIM-006
  hazard): WITH the control the profile does NOT reject `n = 1` (Δloss ≈ 0) → `unresolved`
  (the confident false positive becomes an honest abstention). **0 confident-wrong across
  seeds.** A real-data realization needs a fluorescent-reporter titration paired with a
  perturbation screen (uncommon in public data), which is why this is a stretch feature.

## Implementation Mapping

| Step | Code |
|---|---|
| the constitutive control (known driven activity → measured reporter) | `nudge.inference.constitutive.ConstitutiveControl` |
| the composed forward model's circuit + readout maps | `nudge.mechanisms.regulatory.hill_activation` |
| calibrate the readout from the control (READOUT params only) | `nudge.inference.constitutive.calibrate_readout` |
| the no-leak proof (∂ control-loss / ∂ circuit ≡ 0) | `nudge.inference.constitutive.control_loss_circuit_gradient` |
| profile circuit `n` WITHOUT vs WITH the control | `nudge.inference.constitutive.profile_circuit_n` |
| the fail-safe classifier (biological-switch / unresolved / no-confound) | `nudge.inference.constitutive.classify_constitutive` |
| calibrate + profile in one call | `nudge.inference.constitutive.constitutive_control_analysis` |
| synthetic ground-truth population + matched control generator | `nudge.inference.constitutive.generate_constitutive_dataset` |
| CLI / MCP orchestration (`.npz` + a zero-setup demo) | `nudge.service.constitutive_file` |

*(This table is machine-checked by `scripts/check_impl_mapping.py`: every `nudge.*`
reference must resolve to a real attribute.)*

## Verification evidence

- `tests/inference/test_constitutive.py::test_calibration_recovers_known_readout_params` —
  the calibration recovers the reporter's Hill parameters from the known doses.
- `tests/inference/test_constitutive.py::test_control_loss_has_zero_circuit_gradient` — the
  no-leak property: the control loss's gradient w.r.t. every circuit parameter is exactly 0.
- `tests/inference/test_constitutive.py::test_true_switch_flips_to_biological_with_control` —
  the LIM-006 flip: flat WITHOUT the control, `n = 1` rejected WITH it, and the honest caveat
  (does NOT point-identify `n`).
- `tests/inference/test_constitutive.py::test_linear_circuit_lim006_hazard_abstains_not_confident_wrong`
  — the LIM-006 hazard (a linear circuit) ABSTAINS `unresolved`, never a bare switch.
- `tests/inference/test_constitutive.py::test_never_confident_wrong_across_ground_truth` —
  0 confident-wrong across ground truths.
- `tests/inference/test_constitutive.py::test_affine_reporter_is_no_confound` — an ~affine
  reporter is correctly read as `no-confound`.
- `tests/mcp/test_server.py::test_server_registers_the_expected_tools` — the `constitutive`
  MCP tool is registered.

## References

- [@HuangFerrell1996] — ultrasensitivity and the `K` / `n` / `v_max` (threshold / gain /
  ceiling) response vocabulary the circuit and readout maps share.
- [@RazoMejia2018] — quantitative reporter induction / transfer functions: the measurement
  nonlinearity a constitutive calibration anchors.
- [@Svensson2020] — the single-cell UMI count model (negative binomial, no zero-inflation)
  the observation layer uses, and why the reporter→counts map matters.
