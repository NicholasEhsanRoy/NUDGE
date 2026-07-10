# NUDGE — Abstention analysis (where it declines, why, and what's fixable)

**Status: research + design analysis (read-only).** This maps every surface on which
NUDGE declines to call a mechanism, classifies each as a **fundamental identifiability
limit** (NUDGE *should* keep abstaining) or an **addressable gap** (a concrete lever
would resolve it), and prioritises the highest-payoff identifiability improvements. It
changes **no** `src/` inference code and **no** fail-safe margins. The toggle-gain
deep-dive is backed by a numerical probe (`scripts/analysis/toggle_gain_abstention_probe.py`,
`scripts/analysis/toggle_gain_mechanism.py`).

The project's thesis is *never claim more than you measured*. Abstention is the feature,
not the bug — so the value of this analysis is as much in the "keep abstaining, this is a
real limit" verdicts as in the "here is the lever" ones. Both are called out explicitly.

Sources it builds on (does not re-derive): `scripts/vv/FINDINGS.md` (§2 identifiability;
the independent-SSA toggle validation; the Laplace layer), `design/TOGGLE_ATTRIBUTION_RESEARCH.md`
(covariance/quasi-potential theory + the FIM measurement), `design/CAPABILITY_ROADMAP.md`
(Cap 1/4/6 levers), `src/nudge/inference/lyapunov.py`, `src/nudge/inference/uncertainty.py`.

---

## 1. The catalogue — every abstention surface, classified

Legend: **F** = fundamental identifiability limit (keep abstaining); **A** = addressable
(a concrete lever resolves it); **F/A** = a fundamental core with an addressable slice.

| # | Surface (verdict emitted) | Where | F / A | Root cause | Concrete lever | Expected payoff |
|---|---|---|---|---|---|---|
| 1 | **off-model** — mechanism doesn't beat the linear / no-switch null | core parsimony gate (`switch_detected`, BIC) | **F** | The data is not two populated attractor states (decoys 001–003, real Gladstone IEG: skew 12, one mode + tail). | none — this is the gate *working*. | Keep. This is the crown-jewel fail-safe; the honest verdict on real data. |
| 2 | **off-model** from a **nonlinear readout** (LIM-006) | core, when readout∘circuit is fit as affine | **A** | Circuit-vs-readout ultrasensitivity is degenerate from one population (profile over circuit `n` flat to 3e-4). | **Constitutive-reporter control** (validated: rejects "no switch", ridge collapses 0.07→1.00). | Turns a *misattribution hazard* into a resolved call **or** an honest circuit-vs-readout abstention. Screen-design ask. |
| 3 | **off-model / mis-call** under **topology misspecification** (T0.5-2) | core, wrong fitted topology | **F/A** | "The edge's K/n/vmax" means different things in different topologies; a confidently-wrong call can slip the gates. | Topology-adequacy check; multi-basin IC seeding (T0.5-3/4 feasible, attribution still degenerate). | A → an *adequacy* abstention; the positive-recovery half stays F (needs a topology prior). |
| 4 | **no-effect** — null / dead-guide perturbation | core, dose-response, epistasis | **F** | The perturbation genuinely does not move the signature above noise (decoy 004). | none. | Keep — a specificity guarantee. |
| 5 | **unresolved** — mechanism ambiguous at the margin | core `margin_k` gate | **F/A** | Below the parsimony margin the call isn't separable from the linear baseline. | More cells / fit budget (§2: resolves at ≥1000 cells, 384/400 budget). | A where underpowered (see #6); F at the genuine noise floor. |
| 6 | **unresolved** — **underpowering** (<~1000 cells/condition) | core, all capabilities | **F/A** | Dominant identifiability factor (§2 heatmap); real Gladstone guides 24–233 cells. | Mostly a **data-quantity** limit (state it). Partial lever: a **depth-aware / NB-heteroscedastic** covariance model recovers info the homoscedastic-Gaussian LNA discards (§"extrinsic noise" showed a heteroscedastic channel *raises* the floor ×1.5). | A modest lift near the boundary; does **not** rescue 24-cell guides — that stays F (focused screen needed). |
| 7 | **insufficient depth** — `lna_reliable` (scale·peak < 15) | Lyapunov attribution guard | **A** | LNA Gaussian relaxation of low counts is untrustworthy (real Gladstone Rest: scale·peak 0.3). | **Deeper sequencing** / a brighter reporter (design choice); the guard is doing its job. | A by experiment design, not by method. Keep the guard. |
| 8 | **near bifurcation** — `lna_reliable` (lobe CV > 1.5) | Lyapunov attribution guard | **F** | The LNA covariance **diverges at the fold** — least reliable exactly where a large perturbation pushes. | none within the LNA; a full non-Gaussian (SSA / large-deviation) likelihood (method change, §3). | Keep abstaining. This is also the toggle-gain crux (#11). |
| 9 | **gain_or_threshold / unresolved** — single toggle snapshot | `attribute_lyapunov_single` | **A** | One snapshot constrains only `n·ln(K/B)` — gain⇄threshold sloppy (FIM cond# 210; corr −0.986). | **A 2nd operating point** (basal/dose/2nd target). Measured: cond 210→22; recovers **threshold + ceiling** on independent SSA. | A for threshold + ceiling (measured, §2 below). **Not** for gain (#11). |
| 10 | **degenerate → confidence 0** (Laplace, cond# > 100) | `mechanism_confidence` | **A** | Same gain⇄threshold degeneracy seen through `H⁻¹` (cond ≈ 211, corr +0.99). | A 2nd operating point (cond 211→27; confidence →0.98). | Mirror of #9 — the UQ layer *quantifies* the lever's payoff. |
| 11 | **GAIN unresolved on a toggle — even with 2–3 operating points** | `attribute_lyapunov_multi` | **F** | The deep-dive (§2). Mild (LNA-trustworthy) gain leaves a near-null, non-unique covariance signature; the gain-visible regime is the LNA-*unreliable* regime. | Neither more operating points nor multi-reporter (measured no-go). Only **time-resolved / relaxation** data, or a **near-fold non-Gaussian** likelihood — a method change, unmeasured. | **Keep abstaining.** A genuine, near-fundamental limit of the stationary-covariance channel. |
| 12 | **N-D gain gate not shipped** (`w_trans` gate is 1-D-guarded) | saddle transition-mode gate | **F** | Occupancy/`w_trans` is a non-gradient quasi-potential channel; on a toggle gain doesn't collapse a lobe (measured 0.00–0.25 vs 1-D 0.87–0.94). | none (the refuted "gain zeroes a lobe" claim). | Keep — abstain rather than ship an unreliable N-D gate. |
| 13 | dose-response **unresolved** — inflection not spanned | `dose_response` | **F/A** | `K` outside the dose range → a Hill fit sees one arm, gain unidentifiable (NANOG: knockdown reaches ~75%, K past max dose). | **A wider / finer dose ladder** *if the biology allows it*; F when it can't (NANOG's K is genuinely past achievable knockdown → LIM-007). | A when the range is extendable; else keep abstaining. |
| 14 | dose-response **unresolved** — CI straddle | `dose_response` | **A** | `n`-profile flat within R² (NANOG bootstrap n-CI [1.2, 12]). | More dose points around the inflection; a 2nd operating axis. | A — the classifier already *prevents* the over-read; more doses would resolve. |
| 15 | cross-modality **inconclusive** — no knob clears the gate | `cross_modality` (Chure F164T, Q21M) | **A** | One operating point can't decompose a dynamic-**range** change into `Ka/Ki` vs `Δε_RA` (mild inducer / stronger-DNA mutants). | **A copy-number series** (2nd operating point) — the biophysics check confirmed this is the fix. | A — resolves the 3/7 Chure abstentions. Concrete screen-design ask. |
| 16 | cross-modality **non-responsive** — span collapses | `cross_modality` (Chure Q294R) | **F** | The variant is near-non-inducible (`Ka≈Ki`, amp≈0.02) — no measurable response to attribute. | none. | Keep — correct. |
| 17 | epistasis **unresolved / no-effect** | `epistasis` | **F/A** | A combo inherits its weakest arm (underpowered → #6); or both arms flat. | A = cells (#6); F = genuinely null combo. | As #6. |
| 18 | epistasis **off-axis / neomorphic under-count** (LIM-009) | `epistasis` scalar projection | **F** | A scalar along the additive axis cannot see a purely orthogonal emergent state (off-axis ≥ on-axis for every synergy pair). | A fuller multivariate interaction model (documented, not built); the flag already warns. | Keep the flag; can only *under-count*, never invert — a documented blind spot. |
| 19 | bifurcation **unresolved** — deep-basin far side | `bifurcation` (`classify_robustness`) | **F** | One-sided **by construction**: far from the fold the slowest rate saturates at decay and the noise lobes carry no fold info. | none — it is a one-sided estimator. | Keep — abstaining on the far side is the honest design. |
| 20 | bifurcation **one-sided lower bound** near the fold (LIM-012) | `bifurcation` | **F** | The LNA breaks *at* the fold (#8) — least reliable where it matters most. | none within the LNA. | Keep — report a lower bound, never a point estimate. |
| 21 | design **integrity / reachability abstention** | `design.invert` (LIM-013) | **F** | Won't prescribe from an `unresolved`/`no-effect` attribution; gradient inversion sees only its starting basin. | Upstream: resolve the attribution (then design proceeds). | Keep — every proposal is a model-bound hypothesis, never a forced call. |

**Reading the table.** The *addressable* surfaces cluster on one lever — **a second (or
carefully-chosen) operating point** (#2, #9, #10, #13, #14, #15) — which NUDGE already has
machinery for (`OperatingPoint`, `fit_lyapunov_multi`, dose axis, copy-number series). The
*fundamental* surfaces cluster on two honest walls: **"the data isn't a switch / isn't
responsive"** (#1, #4, #16, #19) and **"the LNA breaks at the fold, and that's where gain
lives"** (#8, #11, #12, #20). The single most load-bearing fundamental limit is #11.

---

## 2. Deep dive — the toggle-gain abstention: fundamental, not fixable

### The abstention, reproduced (independent SSA)

On the 2-node toggle, `attribute_lyapunov_multi` resolves **threshold (3/3)** and
**ceiling (3/3)** with two operating points but **abstains on gain (3/3)** — validated on
independent tau-leaping SSA data (`scripts/vv/toggle_lyapunov_ssa_RESULTS.txt`). The probe
(`toggle_gain_abstention_probe.py`, Part B) sweeps the number of operating points and
measures the **resolved-channel NLL gap** = NLL(2nd-best) − NLL(best); the resolve margin is
**0.03 nats**:

| true mechanism | pts=1 (call/gap) | pts=2 (call/gap) | pts=3 (call/gap) | verdict |
|---|---|---|---|---|
| **gain** (n×0.6) | `unresolved` 0.0192 | `unresolved` **0.0009** | `unresolved` **0.0001** | abstains at every point count — **the gap SHRINKS toward zero** |
| **threshold** (K×1.6) | `ceiling` 0.036 | **`threshold`** 0.344 | **`threshold`** 0.094 | resolves correctly at ≥2 points |
| **ceiling** (vmax×0.6) | `unresolved` 0.001 | **`ceiling`** 0.202 | `threshold` 0.304 ⚠ | resolves at 2 points; a **near-fold 3rd point flips it WRONG** (see note) |

The decisive line is the gain row: adding operating points does **not** open the gap — it
*closes* it (0.019 → 0.0009 → 0.0001), the exact opposite of threshold/ceiling. This is the
empirical proof that **more operating points is not the lever for toggle-gain.** For gain,
all three restricted fits (free-`n`, free-`K`, free-`vmax`) reach the *same* NLL (7.789 /
7.789 / 7.789 at 3 points): the true knob `n` has **no advantage** at reproducing the
gain-perturbed data over the two decoy knobs.

> **Probe caveat — more operating points is not monotonically safe.** With `pts=3` the probe
> added a 3rd point at **basal 0.60**, which sits near the toggle's saddle-node (the WT lobe
> std ≈ 1.94 barely clears the inter-mode separation ≈ 2.24, so `lna_reliable` passes it *at
> the edge*). That marginal point **corrupted the shared-parameter joint fit and flipped a
> true ceiling to a confident `threshold`** (gap 0.30). The **validated breaker is the
> 2-point basal 0.05 + 0.30 fit** (recover-or-abstain, 0 wrong on independent SSA); naively
> piling on a near-fold operating point can *hurt*. Lever #2 below carries this caveat: the
> 2nd axis must be LNA-reliable *with margin*, not just past the guard. (This does **not**
> touch the gain verdict — gain abstains at 1, 2 and 3 points regardless.)

### Why — the mechanistic root (deterministic probe)

`toggle_gain_mechanism.py` measures, with no fitting, how far each knob moves the LNA
observables the covariance loss actually reads — the per-mode **means** and **covariances** —
and the answer is a **near-disjointness** between the regime where gain is *visible* and the
regime where the LNA is *trustworthy*. Symmetric-`n` sweep at basal 0.05 (scale=15):

| n_factor | n_eff | rel. Δmean | rel. Δcov (Frobenius) | `lna_reliable`? |
|---|---|---|---|---|
| 0.90 | 3.60 | 0.017 | 0.040 | **OK** |
| 0.75 | 3.00 | 0.057 | 0.167 | **OK** |
| 0.60 | 2.40 | 0.182 | **0.979** | **near bifurcation (abstains)** |
| ≤0.50 | ≤2.00 | — | — | **monostable (bistability lost)** |

Gain's covariance signature is **small while the LNA is reliable** (Δcov ≤ 0.17) and only
becomes **large as `n` approaches the saddle-node** (Δcov jumps to ~1.0) — crossing exactly
into the `near bifurcation` regime where `lna_reliable` **abstains**, and just past that the
switch goes monostable. The physics: at a toggle's stable modes both species are **saturated**
against the repression threshold (repressor ≪K or ≫K), so the Hill slope — which sets the
drift Jacobian `A` and hence the Lyapunov covariance `AΣ+ΣAᵀ+D=0` — is **nearly independent
of the cooperativity `n`** until a mode drifts toward the saddle. The information about `n`
lives near the **fold / transition region** (a large-deviation property), not in the local
covariance of a deep basin. Threshold and ceiling, by contrast, move the mode **means**
first-order (K translates the nullclines; v_max sets the high plateau ≈ b+v), which is why a
basal shift resolves them.

And where gain *is* visible, it is **non-unique**. At the near-fold basal-0.3 point the
best single-knob mimics of a gain change converge — residual to the gain moments: gain (true)
0.219, threshold 0.281, **ceiling 0.235** — i.e. a v_max or K change reproduces the
gain-induced covariance change about as well as `n` itself. So the near-fold regime is
*both* LNA-untrustworthy *and* gain⇄ceiling⇄threshold degenerate.

### Verdict: **fundamental** (keep abstaining)

The mild-gain toggle abstention is a **genuine identifiability limit of the stationary
covariance channel**, not a tuning gap or a confound a second condition breaks:

- **Not fixable by more operating points** — measured: the gain gap shrinks with pts
  (0.019→0.0001). Operating points break threshold/ceiling because those move means; gain
  moves neither mean in the trustworthy regime.
- **Not fixable by a multi-reporter panel (Cap 6)** — the blindness is in the **latent
  dynamics** (the Jacobian at the mode), not the readout projection. Every reporter of the
  same latent sees the *same* `n`-insensitive covariance, so an over-determined readout
  cannot manufacture gain information the latent never carried. (Multi-reporter *does* help
  the K⇄v_max readout-side degeneracy — just not this one.)
- **The one observable that could carry it is time.** Gain shapes the **relaxation
  timescale / transition region**, not the deep-basin stationary shape. A **time-resolved /
  kinetic** readout (Capability 4) samples the approach to steady state through the region
  where `n` matters; a **near-fold non-Gaussian (SSA / large-deviation)** likelihood could
  read `n` from the transition mass. Both are **method changes**, both **unmeasured**, and
  the near-fold one fights the `lna_reliable` guard. Neither is a data-panel tweak.

This is exactly the tension `TOGGLE_ATTRIBUTION_RESEARCH.md` predicted ("covariance
attribution is weakest exactly in the regime a large gain reduction creates") and the FIM
foresaw ("a single-snapshot toggle fit *should* abstain on gain"). NUDGE ships that
abstention; the probe confirms it survives 2–3 operating points. **Keep abstaining on
toggle-gain — and say why.**

---

## 3. Prioritised short-list — the highest-payoff identifiability improvements

Honest ordering by (payoff × how-measured), tying each to a number. None require weakening
a margin; all are *additive*.

### #1 — Constitutive-reporter control channel (breaks LIM-006; addresses #2, #15, #16-adjacent)
- **What it buys, measured.** On the readout-nonlinearity confound the circuit-`n` profile
  is flat to **3e-4** (you cannot even tell a switch exists); adding a constitutive control
  makes `n=1` **rejected by 0.017 (≫ floor)** and collapses the near-optimal multistart
  ridge **0.07 → 1.00**. On the Chure cross-modality benchmark a **copy-number series** (the
  same "2nd anchor" idea) is the confirmed fix for the **3/7 `inconclusive` abstentions**
  (F164T, Q21M) — the biophysics check endorsed it.
- **Why #1.** It is the only lever that attacks a **misattribution hazard** (LIM-006 can go
  *confidently wrong*), not just an abstention; it doubles as a concrete **screen-design ask**
  to the field; and it is already validated in a standalone spike — the remaining work is
  wiring an optional calibration channel + "abstain on the circuit-vs-readout axis when it is
  absent." Payoff: converts a documented *bound* into a *capability + design recommendation*.

### #2 — Make the "2nd operating point" breaker a first-class, always-attempted path
- **What it buys, measured.** A 2nd operating point drops the gain⇄threshold FIM condition
  number **210 → 22** (smallest eigenvalue ×16.5), the Laplace cond# **211 → 27** (confidence
  0 → 0.98), and on **independent SSA** flips threshold + ceiling from `unresolved` to
  **resolved (3/3, gaps 0.16–0.34 nats ≫ 0.03 margin)** with **0 wrong**. The `OperatingPoint`
  API, dose axis, epistasis A/B/A+B, and multi-timepoint machinery already supply operating
  points — the lever is *plumbing them into the default attribution flow* wherever ≥2 clean
  conditions exist, and reporting the cond#-drop as the identifiability receipt.
- **Why #2.** It is the single most broadly-applicable resolved-call lift (it touches #9,
  #10, #13, #14, #15), it is *already measured end-to-end*, and it is the honest counter to
  "NUDGE abstains too much." **Two caveats that keep it honest:** (a) the 2nd axis must be
  *clean* — a batch/depth difference masquerading as an operating point re-introduces the
  confound (pin depth per context via `calibrate_from_wt`; abstain when the axis is
  confounded); (b) it must be **LNA-reliable with margin, not merely past the guard** — the
  probe showed a *near-fold* 3rd operating point (basal 0.60) flipped a true ceiling to a
  confident `threshold` (§2). So the recommendation is "add a clean, well-buffered 2nd point,"
  not "add as many points as possible." A cheap safeguard: weight each operating point's
  contribution by its `lna_reliable` margin, or require a minimum buffer, so a marginal point
  cannot dominate the shared-parameter fit.

### #3 — A depth-aware / NB-heteroscedastic covariance likelihood (attacks underpowering #6, guard #7)
- **What it buys, measured (indirectly).** The homoscedastic LNA Gaussian is the reason
  `lna_reliable` must abstain below scale·peak 15 and why underpowered guides (#6) resolve
  nothing. The extrinsic-noise study already showed a **heteroscedastic `dΣ/dθ` channel
  raises the identifiability floor ×1.5**; a count-faithful (NB / depth-aware) covariance
  would recover information the current Gaussian discards near the low-count boundary,
  lowering the depth guard threshold *without* lowering its safety.
- **Why #3, and honestly ranked last.** It is the most speculative of the three (the ×1.5 is
  a related-but-not-identical measurement) and it does **not** rescue 24-cell guides — that
  stays a genuine data-quantity limit. But it is the only lever that widens the *reliable*
  regime rather than abstaining at its edge, so it is the right long-horizon investment. It
  would **not** touch #11 (toggle-gain is a fold-locality limit, not a depth limit).

**Explicitly *not* on the list (keep abstaining):** toggle-gain (#11), the N-D gain gate
(#12), near-fold bifurcation proximity (#20), deep-basin robustness (#19), non-responsive
variants (#16), off-model-because-not-a-switch (#1). Each is a real identifiability wall;
NUDGE scores better by declining than by manufacturing a call. That honesty *is* the product.

---

## 4. One-line summary

The addressable abstentions almost all reduce to **"add a clean second operating point (or a
constitutive control)"** — a lever NUDGE has measured (cond# 210→22) and can plumb in more
widely. The fundamental ones reduce to **"the data isn't a switch"** and **"the linear-noise
Gaussian dies at the fold — which is exactly where gain lives on a toggle."** The toggle-gain
abstention is the sharpest of these: measured fundamental (the NLL gap *shrinks* with more
operating points), and the one observable that could carry it — relaxation time — is a future
method, not a data tweak. NUDGE should keep abstaining on it, loudly, and it does.
