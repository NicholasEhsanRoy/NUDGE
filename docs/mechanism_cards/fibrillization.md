---
id: NUDGE-METHOD-013
name: fibrillization_attribution
role: attribution-method
registry_name: FibrillizationAttribution
vulnerable_to_decoys: []
documented_limitation: [NUDGE-LIM-021]
validated_in_regime: {min_replicates: 8, min_timepoints: 40, notes: "Synthetic aggregation-curve round-trip. A SINGLE curve identifies the composites κ = √(2·k_+·k_2·m^{n2+1}) and λ = √(2·k_+·k_n·m^{nc}) (recovered κ≈1, λ≈0.01 in the secondary-dominated regime) and ABSTAINS on the three individual constants k_n, k_+, k_2 — a MEASURED exact gauge degeneracy (Fisher/Laplace condition number → ∞, null direction ≈ (+log k_n, −log k_+, +log k_2), numerical gauge check ~1e-16). A concentration series ALONE stays degenerate (the mass-fraction gauge is concentration-independent); a series + a seeded/elongation anchor (the Meisl discipline) resolves all three (k_n, k_+, k_2 within 25% in a balanced regime), 0 confident-wrong. Inhibitor attribution (control vs inhibited) recovers the microscopic target — primary nucleation k_n / elongation k_+ / secondary nucleation k_2 — from the composite log-ratios, or abstains; 0 confident-wrong across the battery. Efficiency demo: the honest answer a control LLM agent took 12.2 min / 28 turns / 6 scripts to hand-derive, returned in ONE deterministic call."}
references: [Knowles2009, Cohen2013, Meisl2016, Michaels2020]
---

# Mechanism Card — Protein aggregation / fibrillization attribution

> **ID:** `NUDGE-METHOD-013`  ·  **Role:** attribution-method
> **Stability:** experimental  ·  **Registry name:** `FibrillizationAttribution`

## Summary

The **efficiency demo**, and the extensibility thesis pointed at a third dynamical system.
NUDGE analyzes an **amyloid aggregation curve** (the sigmoidal ThT / polymer-mass trace of
a protein polymerizing into fibrils) and, **in one call**, reports the two identifiable
composite rate parameters and the **measured non-identifiability** of the three
microscopic rate constants — then attributes an inhibitor to the microscopic step it acts
on. It reuses NUDGE's Fisher/Laplace identifiability guard verbatim while touching
**neither `fit.py` nor `core/circuit.py`** (frozen). A control LLM agent, unaided, took
**12.2 minutes / 28 turns / six iterative scripts** to hand-derive the same answer
(`design/automated_scientist/runs/000000008`); NUDGE returns it deterministically in a
single call.

## Governing equation — the filament master equation, reduced to its moments

Following Knowles/Cohen/Meisl 2016 and Michaels 2020, the microscopic filament-assembly
master equation reduces to two moment ODEs in the **filament number** concentration `P`
and the **polymer mass** concentration `M`, with free monomer `m = m_tot − M`:

```
dP/dt = k_n · m^{n_c}   +   k_2 · m^{n_2} · M
dM/dt = 2 · k_+ · m · P
```

- `k_n` — **primary nucleation** rate (monomer order `n_c`);
- `k_+` — **elongation** rate (two ends);
- `k_2` — **secondary** surface-catalysed nucleation rate (order `n_2`).

Integrated by a self-contained differentiable RK4 `lax.scan` (mirroring
`lotka_volterra.simulate_glv`; **no `diffrax`**), differentiable through `(k_n, k_+, k_2)`.

## What it identifies — and where it abstains (the load-bearing honesty)

From a **single curve at a single monomer concentration** only the two composites

```
λ = √(2 · k_+ · k_n · m_tot^{n_c})       (primary-pathway rate / lag)
κ = √(2 · k_+ · k_2 · m_tot^{n_2+1})     (secondary / autocatalytic growth rate)
```

are identifiable. The three individual constants are **not**, because the moment model has
an **exact continuous gauge symmetry**: `(k_n, k_+, k_2) → (k_n/α, α·k_+, k_2/α)` leaves
`M(t)/m_tot` identical for any `α > 0` (verified numerically — a 100× `k_+` rescale changes
the curve by ~1e-16). So the Fisher/Laplace curvature on `(log k_n, log k_+, log k_2)` has a
**genuine zero eigenvalue** along the null direction ≈ `(+log k_n, −log k_+, +log k_2)`,
condition number → ∞. NUDGE **measures** this and returns the composites + the null
direction + *"need a concentration series and a seeded anchor"* — the earned abstention
(`NUDGE-LIM-021`).

## What a concentration series (and a seeded anchor) do resolve

The mass-fraction gauge is **concentration-independent**, so a concentration series ALONE
still cannot separate the individual constants (measured: the global-fit curvature stays
degenerate). A **seeded / elongation reference** (a heavily-seeded early-window curve where
`dM/dt ≈ 2·k_+·m·P_0` pins `k_+` directly) breaks the gauge; the global shared-parameter
fit then **resolves all three** — the Meisl discipline. In a strongly secondary-dominated
regime (`κ ≫ λ`) the primary rate `k_n` stays weakly determined even with the anchor (the
primary pathway is negligible) — a sloppy-but-identifiable direction reported with a wide
CI, never a false-precise value.

## What it attributes — an inhibitor's microscopic target

An inhibitor is a perturbation to specific `k`'s. From the composite log-ratios of a
control-vs-inhibited curve pair (`r_λ = log(λ_inhib/λ_ctrl)`, `r_κ = log(κ_inhib/κ_ctrl)`):

| Signature | Verdict | Biophysical reading |
|---|---|---|
| `r_λ ≈ r_κ < 0` (both drop equally) | `elongation` | fibril-**end** binder lowers `k_+` (λ, κ ∝ √k_+); a global monomer-sequestering binder gives the SAME signature (documented ambiguity) |
| `r_κ < 0`, `r_λ ≈ 0` | `secondary_nucleation` | fibril-**surface** binder lowers `k_2` |
| `r_λ < 0`, `r_κ ≈ 0` | `primary_nucleation` | primary-**nucleus** binder lowers `k_n` |
| both ≈ 0 | `no-effect` | no detectable kinetic effect |
| otherwise | `unresolved` | abstain rather than guess |

## The identifiability verdict (fail-safe)

- **The discriminator is the exact gauge, not the raw condition number.** The
  deep-research sloppiness caveat (Transtrum et al.; `design/DEEP_RESEARCH_drug_discovery_directions.md`)
  warns that a large condition number conflates *sloppy-but-identifiable* with
  *non-identifiable*. So NUDGE keys the verdict on a **flat (zero-curvature) direction**
  (a genuine gauge zero) and reports the condition number as a *sloppiness caveat*.
- A single curve / an anchor-less series → a genuine zero eigenvalue → **not identifiable**.
- A series **with** a seeded anchor → all eigenvalues positive (the gauge broken) → the
  individuals are **resolved** (though the spectrum may be sloppy).

## Assumptions & simplifications

- The observable is the **mass fraction** `M(t)/m_tot` (a ThT-like proxy), a deterministic
  sigmoid + measurement noise — so the fit is least-squares (the field-standard AmyloFit
  objective), not the distributional energy distance used on single-cell ensembles.
- Reaction orders `n_c`, `n_2` are treated as known for the single-curve composite fit (a
  single curve cannot determine them; a concentration series can).
- The perturbation moves **one** microscopic rate of a system whose baseline kinetics are
  otherwise shared with the control.

## Known failure modes

| Failure mode | Guard / witness | Limitation |
|---|---|---|
| Over-fitting three confident individual constants from one curve | `individual_k_identifiability` measures the gauge null (cond → ∞) → the three flagged unidentifiable | `NUDGE-LIM-021` |
| Claiming a concentration series alone resolves the individuals | `resolve_series(use_anchor=False)` stays degenerate (gauge is concentration-independent) | `NUDGE-LIM-021` |
| Naming the wrong microscopic step for an inhibitor | `attribute_inhibitor` abstains (`unresolved`) unless a single-target composite signature is clear | `NUDGE-LIM-021` |
| Elongation vs global monomer-sequestration ambiguity | reported explicitly on the equal-drop branch, never over-claimed | `NUDGE-LIM-021` |
| A diverging aggregation orbit NaN-ing the loss | `P`, `M` clipped `[0, 1e12]`, monomer clamped `≥ 0` inside the integrator | `NUDGE-LIM-021` |

There is **no Circuit-style decoy-battery entry** (`vulnerable_to_decoys: []`) because the
aggregation cases are curve-based, not the count/AnnData shape of `DECOY_BATTERY`; they
live as generator functions with dedicated `verification` / `decoy`-marked tests (below).

## Implementation Mapping

| Step | Code |
|---|---|
| moment vector field | `nudge.mechanisms.fibrillization.moment_vector_field` |
| differentiable RK4 integrator | `nudge.mechanisms.fibrillization.simulate_aggregation` |
| composites `λ`, `κ` | `nudge.mechanisms.fibrillization.composite_lambda_kappa` |
| single-curve generator (known kinetics) | `nudge.mechanisms.fibrillization.simulate_aggregation_curve` |
| seeded elongation anchor | `nudge.mechanisms.fibrillization.simulate_seeded_elongation` |
| concentration series | `nudge.mechanisms.fibrillization.simulate_concentration_series` |
| inhibitor curve pair | `nudge.mechanisms.fibrillization.simulate_inhibitor_pair` |
| single-curve composite fit | `nudge.mechanisms.fibrillization.fit_composites` |
| the MEASURED gauge degeneracy | `nudge.mechanisms.fibrillization.individual_k_identifiability` |
| one-call entry point | `nudge.mechanisms.fibrillization.attribute_aggregation` |
| inhibitor attribution | `nudge.mechanisms.fibrillization.attribute_inhibitor` |
| concentration-series global fit + verdict | `nudge.mechanisms.fibrillization.resolve_series` |
| reused Fisher/Laplace guard | `nudge.inference.uncertainty.laplace_posterior` |

*(This table is machine-checked by `scripts/check_impl_mapping.py`: every `nudge.*`
reference must resolve to a real attribute.)*

## Verification evidence

- `tests/mechanisms/test_fibrillization.py::test_single_curve_identifies_composites_and_abstains_on_individuals`
  — one call recovers κ, λ and ABSTAINS on the three individual constants (measured gauge).
- `tests/mechanisms/test_fibrillization.py::test_gauge_symmetry_is_exact` — the exact
  `(k_n/α, α·k_+, k_2/α)` symmetry, numerically confirmed.
- `tests/mechanisms/test_fibrillization.py::test_concentration_series_with_anchor_resolves_three_constants`
  — a series + seeded anchor resolves all three (0 confident-wrong).
- `tests/mechanisms/test_fibrillization.py::test_concentration_series_without_anchor_stays_degenerate`
  — the honesty half: a series alone cannot break the gauge.
- `tests/mechanisms/test_fibrillization.py::test_inhibitor_battery_zero_confident_wrong` —
  the inhibitor battery never names the wrong microscopic step.

## References

- [@Knowles2009] — the analytical solution to the kinetics of breakable filament assembly
  (the moment / master-equation framework).
- [@Cohen2013] — amyloid-β42 proliferation via a secondary-nucleation mechanism (the
  `k_2` pathway this module attributes).
- [@Meisl2016] — molecular mechanisms of protein aggregation from **global fitting** of
  kinetic models across a concentration series (the discipline; the composites κ, λ).
- [@Michaels2020] — thermodynamic and kinetic design principles for amyloid-aggregation
  **inhibitors** as perturbations to specific microscopic rate constants (the attribution
  target + the invertible design map).
