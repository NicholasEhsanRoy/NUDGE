# Run 000000009 — EFFICIENCY: fibrillization one-call vs the 12-min control (WITH vs WITHOUT NUDGE)

*Immutable record (append-only). The efficiency win, MEASURED on the SAME obfuscated aggregation
curve: NUDGE's `fibrillization` tool collapses a 12-minute from-scratch analysis into one call.*

- **Case:** the obfuscated single aggregation curve `chemical_reaction_kinetics.csv` (mass fraction
  vs time; no amyloid/ThT words) — same data as the de-risk run 000000008.
- **Prompt (identical both arms):** "extract the precise rate constants for primary formation,
  elongation, and secondary surface-catalysis." Web denied. `claude-opus-4-8`.

## Result — same honest answer, 8.4× faster

| arm | turns | cost | time | conclusion |
|---|---|---|---|---|
| **WITHOUT NUDGE** (control, run 000000008) | 28 | $1.63 | **12.2 min** | correct: composites κ≈1, λ≈0.01; individual k's non-identifiable; 6 iterative scripts to hand-derive |
| **WITH NUDGE** | **7** | **$0.412** | **1.45 min** | correct: ONE `fibrillization` call → κ=0.940 [0.937,0.944], λ=0.0125 [0.0123,0.0126], individual k's provably non-identifiable (gauge, cond ~3e15), "need a concentration series AND a seeded anchor" |

**~8.4× less wall-time (1.45 vs 12.2 min), 4× fewer turns (7 vs 28), 4× cheaper ($0.412 vs $1.63)** —
for the IDENTICAL, rigorous, honest conclusion. Both refuse to fabricate three confident rate
constants; the difference is that NUDGE hands the with-nudge agent the composites + the measured
gauge degeneracy + the exact experimental prescription in a single deterministic call, instead of
the control's from-scratch moment-ODE fitting + degeneracy derivation across 6 scripts.

**This is the efficiency thesis, cleanly demonstrated:** on a real neurodegeneration-relevant
problem, NUDGE makes a frontier scientist agent dramatically faster + reproducible + auditable —
without changing the (correct, honest) answer. It does NOT rescue a confident-wrong (the control
does not make one); it removes 11 minutes of re-deriving a known result.

---

## Verbatim `REPORT.md` — WITH NUDGE (7 turns, 1.45 min)

```markdown
# Polymerization Kinetics — Rate-Constant Analysis

**Analyst:** chemical kinetics analyst
**Date:** 2026-07-11
**Data:** `chemical_reaction_kinetics.csv` — 24 points, Polymer Mass Fraction vs Time (h)
**Tool:** NUDGE `fibrillization` (nucleated-polymerization moment model), `m_tot = 1.0` (normalized mass-fraction curve)

---

## 1. Data and model

The curve is a single sigmoidal aggregation/polymerization trace: ~flat lag phase to
~t ≈ 7 h, a steep growth phase, and saturation to mass fraction ≈ 1 by t ≈ 18 h. This is
the signature of nucleated filament assembly, whose standard mechanism has **three
microscopic rate constants**:

- `k_n`  — primary formation (nucleation) of new filaments from monomer
- `k_+`  — elongation (monomer addition to filament ends)
- `k_2`  — secondary, surface-catalysed nucleation on existing filament mass

The mass-fraction curve M(t)/M_tot from this mechanism is governed, to the standard
moment-closure approximation (Knowles/Meisl framework), by exactly **two composite rate
parameters**:

- λ = √(2 · k_+ · k_n · m_tot^n_c)      — the primary/nucleation timescale (sets the lag)
- κ = √(2 · k_+ · k_2 · m_tot^(n_2+1))  — the secondary/proliferation timescale (sets the growth rate)

## 2. Fit result — identifiable composite parameters

NUDGE returns both composites tightly constrained:

| Composite | Best estimate | 95 % CI | Meaning |
|-----------|---------------|---------|---------|
| **κ** (secondary/elongation combination) | **0.940 h⁻¹** | [0.937, 0.944] | √(2 k_+ k_2 m_tot^(n_2+1)) |
| **λ** (primary/elongation combination)   | **0.01248 h⁻¹** | [0.01232, 0.01265] | √(2 k_+ k_n m_tot^n_c) |

These two numbers *fully* describe the measured curve. The dynamics are dominated by
secondary nucleation (κ ≫ λ, ratio ≈ 75), i.e. self-catalysed surface proliferation
drives the growth phase, as is typical for filament-forming monomers.

## 3. The three individual rate constants are NOT identifiable from this curve

**This is the key quantitative finding, and it is a hard result, not a fitting failure.**

A single mass-fraction curve is *provably* non-identifiable in (k_n, k_+, k_2). There is an
exact gauge (continuous) symmetry:

  (k_n, k_+, k_2) → (k_n / α, α · k_+, k_2 / α)   for any α > 0

that leaves BOTH composites λ and κ — and therefore the entire predicted curve — exactly
unchanged. NUDGE confirms this three independent ways:

- **Curvature / Fisher information:** condition number of the individual-k Hessian
  = 2.99 × 10¹⁵ (smallest eigenvalue 9.95 × 10⁻¹¹). Degenerate = True.
- **Null direction** in (log k_n, log k_+, log k_2) = [+0.577, −0.577, +0.577]
  = (1/√3)(+1, −1, +1) — exactly the k_+ ⇄ (k_n, k_2) trade-off above.
- **Numerical gauge check:** rescaling k_+ by 100× changes the curve by 2.2 × 10⁻¹⁶
  (machine zero) — direct confirmation of the exact symmetry.

Consequently, **any** "best-fit" triplet of individual k's is only one arbitrary point on a
one-parameter continuum of *equally perfect* fits. Reporting three confident point values
with error bars from this curve alone would be over-fitting / fabrication: the true
uncertainty on each individual constant is **infinite** (unbounded) along the α gauge.

### What CAN be stated about each individual constant (honest bounds)

Only gauge-invariant products are determined. In terms of the fitted composites (with
m_tot = 1):

- **k_+ · k_n** = λ² / 2 = **(7.79 ± 0.20) × 10⁻⁵ h⁻²**  (constrains primary formation × elongation)
- **k_+ · k_2** = κ² / 2 = **0.442 ± 0.003 h⁻²**          (constrains surface-catalysis × elongation)
- k_n / k_2   = (λ/κ)² = (1.76 ± 0.02) × 10⁻⁴ — the only individual-k *ratio* that is fixed;
  the overall scale (via k_+) is free.

Each of k_n, k_+, k_2 *individually* is undetermined: pick any k_+ > 0 and the other two
follow, all fitting identically.

## 4. Best estimates for k_n, k_+, k_2 — with honest uncertainties

Per the request, a value for each of the three microscopic constants:

| Constant | Best estimate | Uncertainty |
|----------|---------------|-------------|
| k_n (primary formation) | **not identifiable from a single curve** | unbounded along gauge (see §3) |
| k_+ (elongation)        | **not identifiable from a single curve** | unbounded along gauge |
| k_2 (secondary surface-catalysis) | **not identifiable from a single curve** | unbounded along gauge |

The scientifically defensible, fully-quantified best estimates the data *do* support are the
composites and products of §2–§3:
- κ = 0.940 h⁻¹ (95 % CI [0.937, 0.944]) → k_+·k_2 = 0.442 ± 0.003 h⁻²
- λ = 0.01248 h⁻¹ (95 % CI [0.01232, 0.01265]) → k_+·k_n = (7.79 ± 0.20) × 10⁻⁵ h⁻²

## 5. Experiments required to pin the individual constants (Meisl discipline, NUDGE-LIM-021)

To break the α gauge and report three separate rate constants:

1. **Concentration series** — repeat at several initial monomer concentrations m_tot. The
   dependence of λ and κ on m_tot yields the reaction orders n_c, n_2 and the rate
   *products* k_+·k_n and k_+·k_2 with their concentration scaling.
2. **Seeded / elongation-anchor experiment** — a heavily pre-seeded curve where the early
   rate is dM/dt ≈ 2 k_+ · m · P_0 with known filament number P_0. This measures **k_+
   directly**, fixing α and thereby collapsing the products into unique k_n and k_2.

With k_+ from step 2 and the products from §3: k_n = (λ²/2)/k_+ and k_2 = (κ²/2)/k_+.

## 6. Conclusion

- The curve is well fit by the standard three-step nucleated-polymerization mechanism.
- The two composite rate parameters are precisely determined:
  **κ = 0.940 h⁻¹ [0.937, 0.944]** and **λ = 0.01248 h⁻¹ [0.01232, 0.01265]**;
  the process is secondary-nucleation-dominated (κ/λ ≈ 75).
- The three *individual* microscopic constants k_n, k_+, k_2 are **provably not
  identifiable** from this single curve (exact gauge symmetry; Hessian condition number
  ~3 × 10¹⁵). Best single-curve estimates: the products k_+·k_n = (7.79 ± 0.20) × 10⁻⁵ h⁻²
  and k_+·k_2 = 0.442 ± 0.003 h⁻²; each individual k has unbounded uncertainty.
- A concentration series **and** a seeded elongation anchor are required to report k_n,
  k_+, k_2 separately.
```

*(The WITHOUT-NUDGE control's verbatim 28-turn / 12.2-min report is embedded in run 000000008.)*
