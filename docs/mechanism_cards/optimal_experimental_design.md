---
id: NUDGE-METHOD-014
name: optimal_experimental_design
role: analysis-method
registry_name: OptimalExperimentalDesign
vulnerable_to_decoys: []
documented_limitation: [NUDGE-LIM-024]
validated_in_regime: {min_replicates: 1, min_timepoints: 2, notes: "Synthetic differentiable ODE forward models (single-species logistic; 3-taxon gLV). The design parameter is the vector of measurement times φ; the target is the growth⇄self-limitation (α⇄βᵢᵢ) degeneracy (Kᵢ=−αᵢ/βᵢᵢ) that NUDGE-METHOD-012 abstains on near equilibrium. MEASURED at the nominal θ₀ (local OED): a naive near-equilibrium design is near-singular (high FIM condition number, tiny smallest eigenvalue); gradient-ascending a differentiable design criterion (D-optimality logdet FIM, E-optimality λ_min, or the targeted reciprocal-CRLB of α) resolves the sloppy parameter — the growth CRLB improves ~31x and the FIM smallest eigenvalue ~18x on the logistic, ~600x CRLB on the gLV. The white-box gradient reaches the exact continuous optimum; a 1-D design knob confirms the gradient optimum coincides with a fine grid's, while a structured grid guaranteeing the optimum costs r^m FIM evaluations (exponential in the design dimension). Local OED: the optimum is valid near θ₀ and the reported gains are measured there, not extrapolated (NUDGE-LIM-024)."}
references: [Transtrum2014, Chis2011, Fisher1935]
---

# Mechanism Card — Optimal Experimental Design (gradient-based)

> **ID:** `NUDGE-METHOD-014`  ·  **Role:** analysis-method
> **Stability:** experimental  ·  **Registry name:** `OptimalExperimentalDesign`

## Summary

The **differentiability moat** — the white-box advantage a legacy black-box ODE solver
cannot offer. Everywhere else NUDGE takes gradients of a *fit* loss w.r.t. the *parameters*;
here it takes gradients of an **identifiability criterion** w.r.t. the **experimental design**
itself. Because NUDGE fits a **differentiable** mechanistic model, the Fisher Information
Matrix `FIM(φ)` — and therefore any experimental-design criterion built from it
(D-/A-/E-optimality, or the reciprocal Cramér–Rao bound of a single target parameter) — is a
**differentiable function of an experimental-design parameter `φ`** (the measurement times, a
pulse window, a dose). So `∂criterion/∂φ` is available by autodiff — straight through the ODE
solve *and* the FIM assembly — and we can **gradient-ascend `φ` to the exact optimal
experiment `φ*`** that maximally resolves a sloppy / degenerate parameter. A black-box solver
has no `∂/∂φ`: it can only grid-search, whose cost is exponential in the number of design
knobs. Implemented in `nudge.inference.oed.optimize_design` (additive; touches neither
`fit.py` nor `core/`).

## Why this exists (it makes an abstention ACTIONABLE)

The temporal / Lotka–Volterra capability (`NUDGE-METHOD-012`) **abstains** when growth `α`
and self-limitation `βᵢᵢ` are degenerate near equilibrium (`Kᵢ = −αᵢ/βᵢᵢ`), with the
degeneracy *measured* by the Laplace curvature (`NUDGE-LIM-020`), and its directional
abstention says *"sample the transient to break the tie."* OED makes that **exact**: the
design gradient tells you **precisely which measurement times** break the tie, and by **what
measured factor** they resolve the previously-sloppy parameter. Abstention → prescription.

## What it computes

Given a `DesignProblem` — a differentiable forward model `observe(θ, φ)` at a nominal `θ₀`,
observation noise `σ`, and a valid range for `φ`:

1. **FIM assembly** — `FIM(φ) = J(φ)ᵀ J(φ) / σ²`, `J = ∂observe/∂θ` (autodiff through the ODE
   solve), assembled in float64 for a precision-safe information geometry.
2. **A differentiable design criterion** of the FIM:
   - **D-optimality** `log det FIM` — total information / overall identifiability;
   - **A-optimality** `−tr(FIM⁻¹)` — total parameter variance;
   - **E-optimality** `λ_min(FIM)` — the worst (sloppiest) direction directly;
   - **targeted reciprocal-CRLB** `−log([FIM⁻¹]_{ii})` — resolve *one named* sloppy parameter.
3. **Gradient ascent on `φ`** (projected Adam) to the optimum `φ*` within the physical window.
4. A **black-box grid-search baseline** (`grid_search_design`) so the "gradient beats grid"
   claim is *measured*: on a 1-D design knob the gradient optimum coincides with a fine
   grid's (validation), while a structured grid guaranteeing the optimum in the full `m`-time
   design costs `rᵐ` evaluations (infeasible), and the gradient reaches it in one pass.

## Honesty (the load-bearing points)

- **Local OED** (`NUDGE-LIM-024`): the FIM is the *local* curvature at `θ₀`; the optimum is
  optimal for parameters near `θ₀`, and the reported gains are the ones *measured at `θ₀`* —
  not extrapolated. A robust design would marginalize over a `θ` prior (not implemented).
- Near-singular FIMs are inverted with a **guarded ridge** — never a plain pseudo-inverse,
  which would *zero* a flat direction's variance and *understate* the CRLB (the opposite of
  safe) — mirroring `nudge.inference.uncertainty.laplace_posterior`.
- Every reported number (the CRLB / smallest-eigenvalue improvement factor) is **computed**
  from the float64 FIM, never asserted.

## Implementation mapping

- `nudge.inference.oed.DesignProblem` — the differentiable forward model + design knob `φ`.
- `nudge.inference.oed.fisher_information` — `FIM(φ) = JᵀJ/σ²` via autodiff.
- `nudge.inference.oed.d_optimality` / `nudge.inference.oed.a_optimality` /
  `nudge.inference.oed.e_optimality` / `nudge.inference.oed.neg_log_crlb` — the criteria.
- `nudge.inference.oed.design_gradient` — `∂criterion/∂φ` (the moat).
- `nudge.inference.oed.optimize_design` — projected gradient ascent to `φ*` + the measured gain.
- `nudge.inference.oed.grid_search_design` — the black-box evaluation baseline.
- `nudge.inference.oed.make_logistic_design_problem` /
  `nudge.inference.oed.make_glv_design_problem` — the showcase models.

## Validation

Synthetic, MEASURED (`scripts/vv/FINDINGS.md` "Optimal Experimental Design";
`tests/inference/test_oed.py`): the naive near-equilibrium design is near-singular; the
gradient-optimal design resolves the growth parameter (logistic ~31x CRLB, ~18x smallest
eigenvalue; gLV ~600x CRLB), across all three objectives; the 1-D design gradient lands on
the fine-grid optimum; the design gradient is finite and non-zero (the moat). Real-data
lock-in (a longitudinal microbiome series with a design choice) is deferred as a later
`needs_data` gate — the differentiable criterion + the synthetic round-trip are the
deliverable.

## Limitations

- `NUDGE-LIM-024` — **local OED**: the optimum + reported gains are valid near `θ₀`; no
  prior-marginalized robust design.
- Related: `NUDGE-METHOD-012` (the gLV abstention this makes actionable), `NUDGE-LIM-020`
  (the α⇄βᵢᵢ degeneracy targeted), `nudge.inference.sloppiness.fisher_information` (the
  Fisher/sloppiness grammar reused).
