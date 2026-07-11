# Run 000000008 — DE-RISK: obfuscated aggregation-curve control test (control-only)

*Immutable record (append-only). Not a with/without ablation — a targeted de-risk before building a
fibrillization module: does a frontier agent CONFIDENTLY over-fit non-identifiable rate constants
from ONE aggregation curve (the hoped capability trap), or catch the degeneracy? And does obfuscation
prevent SEMANTIC RECALL of the known (Meisl/Knowles) result?*

- **Setup:** a single sigmoidal polymerization curve generated from secondary-nucleation moment ODEs
  (true k_n=2e-5, k_+=0.1, k_2=0.04; identifiable composites λ=0.01, κ=1). **Obfuscated**:
  `chemical_reaction_kinetics.csv`, axes "Time (h)" / "Polymer Mass Fraction", NO
  amyloid/ThT/fibrillization/nucleation words. **Control agent only** (claude-opus-4-8, NO NUDGE, web
  denied, sandbox outside repo). Prompt: "extract the precise rate constants for primary formation,
  elongation, and secondary surface-catalysis."

- **Result: 28 turns, $1.632, 12.2 minutes.** The agent CORRECTLY reported the three
  individual constants are NOT separable from one curve (derived κ=1.0±0.15 [true 1], λ=6e-3 [true
  0.01], the exact scaling degeneracy) and prescribed a concentration series. **BUT it got there
  partly by SEMANTIC RECALL** — its own report names "amyloid-type kinetics; Knowles/Cohen/Meisl
  framework" and cites "Meisl et al., Nat. Protoc. 2016". Obfuscation did NOT prevent recognition —
  the lag+autocatalytic sigmoid is too distinctive.

## Decision (per the pre-registered success criterion)

The control realized the non-identifiability → **the confident-overfit capability gap is DEAD on 1D
kinetics.** Robust: on run 000000005 the control derived the gLV null-space from PURE MATH (no
memory) too. So whether by memory OR math, Opus catches single-parameter identifiability. Building a
fibrillization module to "rescue a confident-wrong" would rescue nothing.

## What survives (drives the dual-track build)

1. **EFFICIENCY is now dramatic + watchable:** 12.2 min / 28 turns / $1.632 of iterative
   scripting for what NUDGE should return in ONE deterministic, safety-gated call — on a REAL
   neurodegeneration problem. → Track 1: build `mechanisms/fibrillization.py` so NUDGE does it instantly.
2. **The surviving capability gap is SCALE + SUBTLETY, not 1D identifiability:** adjoint sensitivity
   (O(1) in parameter count — an agent's forward-sensitivity code OOMs/times-out at network scale)
   and sloppiness-aware abstention (sloppy ≠ unidentifiable; an eigenvalue-gap test can be WRONG). →
   Track 2: build adjoint + sloppiness diagnostics for a 15+ ODE blind case.

---

## Verbatim control `REPORT.md` (28 turns)

```markdown
# Polymerization (Filament-Formation) Kinetics — Analysis Report

## 1. Data

`chemical_reaction_kinetics.csv`: 24 points, Time 0–18 h, Polymer Mass Fraction 0 → ~1.
The curve is sigmoidal with a pronounced lag phase (mass fraction ≈ 0 until ~5–6 h),
a steep autocatalytic rise (~7–13 h), and a plateau near 1.0. This is the classic
signature of a **nucleation–elongation–secondary-nucleation** filament assembly
(amyloid-type kinetics; Knowles/Cohen/Meisl framework).

## 2. Mechanism / Model

Standard three-process filament master-equation model, reduced to two moments —
fibril number concentration `P(t)` and polymer mass fraction `x(t)=M/M_tot`:

- Primary formation (nucleation), rate constant **k_n**, order n_c
- Elongation (two ends), rate constant **k_+**
- Secondary (surface-catalysed) nucleation, rate constant **k_2**, order n_2

Moment ODEs (monomer fraction m = 1 − x, working in mass-fraction/normalised units):

    dP/dt = k_n · m^{n_c}  +  k_2 · m^{n_2} · x
    dx/dt = 2 · k_+ · m · P

Initial condition P(0)=0, x(0)=x_data[0]. Effective rate constants absorb the total
monomer concentration (m_tot ≡ 1 in mass-fraction units).

The two combined rate constants that govern the observable curve shape are
(Meisl et al., Nat. Protoc. 2016):

    λ = sqrt(2 · k_+ · k_n)      (sets the lag / primary-pathway timescale)
    κ = sqrt(2 · k_+ · k_2)      (sets the autocatalytic growth rate)

## 3. Method

Numerical integration of the moment ODEs (scipy `solve_ivp`, LSODA, rtol 1e-7),
weighted-least-squares fit of log(k_n, k_+, k_2) with `least_squares`. Reaction
orders (n_c, n_2) scanned over {1,2,3}² with multistart to select the best mechanism.
Parameter uncertainties from the Jacobian-based covariance at the optimum.

(Results appended below as they are computed.)

---

## 4. Results

### 4.1 Fit quality
Best-fit model reproduces the curve to **RMSE ≈ 0.013–0.015** in mass fraction,
i.e. σ_residual ≈ 0.014 — consistent with the scatter of the measured points
(the early "zero" region scatters by ~0.006, the rise by ~0.015). The mechanism
fits within noise; no systematic misfit.

### 4.2 Reaction orders
- Scanning n_c ∈ {1,2,3}: **SSE is completely independent of n_c** (identical to 5 d.p.).
  Reason: monomer is barely depleted during the lag (m = 1−x ≈ 1), so the primary term
  k_n·m^{n_c} ≈ k_n regardless of n_c. **n_c is not identifiable from this curve.**
- Secondary order: SSE(n₂=1)=0.0119 > SSE(n₂=2)=0.0052 > SSE(n₂=3)=0.0043.
  Continuous fit gives **n₂ = 2.7 ± 0.4** (canonical amyloid value is 2). Data mildly
  prefer n₂ ≈ 2.7–3.

### 4.3 The identifiability problem (central result)
The mass-fraction curve x(t) depends on the three rate constants **only through two
combinations**. There is an exact scaling symmetry:

    (k_+, k_n, k_2) → (α·k_+, k_n/α, k_2/α)   leaves x(t) identical.

Verified numerically: scaling k_+ over a 10⁴-fold range (α = 0.01 → 100), with k_n and
k_2 compensating, changes the predicted curve by < 6×10⁻¹⁰ (machine level). Consistently,
the fitted 3-parameter covariance has parameter correlations of ±1.00 and a "soft"
eigen-direction ∝ (+log k_n, −log k_+, +log k_2) with ~4000% uncertainty.

**Consequence:** from a single curve at a single monomer concentration, k_n, k_+ and k_2
CANNOT be separated. Only the two combined rate constants are determined:

| Combined constant | Meaning | Best estimate | Uncertainty |
|---|---|---|---|
| κ = √(2·k_+·k_2) | secondary-nucleation / autocatalytic rate | **1.0 h⁻¹** | ± 0.15 h⁻¹ (0.94 at n₂=2 → 1.14 at n₂=3; ±3% statistical) |
| λ = √(2·k_+·k_n) | primary-pathway rate | **6×10⁻³ h⁻¹** | ×/÷ ~2–3 (0.004 at n₂=3 → 0.012 at n₂=2; strongly n₂-coupled) |

Separating k_+ requires an independent measurement (e.g. a seeded / concentration-series
experiment); it is not present in these data.

### 4.4 The three microscopic rate constants
Because of §4.3, individual values are meaningful only **relative to an assumed
elongation rate constant**. Reported in normalized mass-fraction units (all in h⁻¹),
anchored to a reference **k_+ ≡ 1 h⁻¹** (any other k_+ rescales k_n, k_2 as k_n·(1/k_+),
k_2·(1/k_+)). Best-fit model n_c=2, n₂≈2.7:

| Process | Constant | Best estimate (at k_+ = 1 h⁻¹) | Uncertainty |
|---|---|---|---|
| Elongation | **k_+** | 1 h⁻¹ (REFERENCE — unbounded from single curve) | not determined by these data |
| Primary formation | **k_n** | **≈ 1.9×10⁻⁵ h⁻¹** | ×/÷ ~3 (range 7×10⁻⁵ at n₂=2 to 8×10⁻⁶ at n₂=3) |
| Secondary catalysis | **k_2** | **≈ 0.59 h⁻¹** | ± ~15% (0.44 at n₂=2, 0.65 at n₂=3) |

Equivalently, the assumption-free deliverables are the products:
- **k_+·k_2 = κ²/2 = 0.50 ± 0.15 h⁻²**
- **k_+·k_n = λ²/2 = 1.9×10⁻⁵ h⁻² (×/÷ ~3)**

## 5. Bottom line
- The curve is a textbook nucleation–elongation–secondary-nucleation filament assembly.
- Secondary nucleation order **n₂ ≈ 2.7 (±0.4)**; primary order n_c indeterminate.
- **Robustly determined:** κ = √(2 k_+ k_2) = **1.0 ± 0.15 h⁻¹**;
  λ = √(2 k_+ k_n) = **6×10⁻³ h⁻¹ (×/÷ 2–3)**.
- **k_n, k_+, k_2 individually are NOT separable from one curve** (exact scaling
  degeneracy). At a reference k_+ = 1 h⁻¹: k_n ≈ 1.9×10⁻⁵ h⁻¹ (×/÷3),
  k_2 ≈ 0.59 h⁻¹ (±15%). To pin all three absolutely, a concentration-series /
  seeded experiment is required.
```
