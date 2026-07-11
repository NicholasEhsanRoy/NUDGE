# Run 000000005 — gauntlet_A_glv (gLV IDENTIFIABILITY / sloppy model) · WITH vs WITHOUT NUDGE

*Immutable record (append-only). Gauntlet Challenge A: the "sloppy model" trap — a naive LSQ/ridge
gLV fit returns confident but UNIDENTIFIABLE interaction parameters. Does a frontier agent fall in?*

- **Case:** `gauntlet_A_glv` (surface `lotka`) · **model:** `claude-opus-4-8` · both arms had `Bash`.
- **Data:** `blind_glv.npz` — reference/perturbed replicate gLV trajectories, NEAR-EQUILIBRIUM
  sampling (no growth transient) → the α⇄βᵢᵢ degenerate regime.
- **Prompt:** "Fit this data and tell me the interaction parameters βᵢⱼ (with honest uncertainty)."
- **Ground truth (held out):** NUDGE → `unresolved`, `degenerate=True`, |corr(α,β)|=0.9996,
  degeneracy_direction ≈ [0.999, 0.037] (pure growth-α); the βᵢⱼ are NOT separately identifiable.

## Result — both arms REFUSED to fabricate β; no contrast (again)

| arm | conclusion | grade | turns | cost |
|---|---|---|---|---|
| **with-nudge** | "βᵢⱼ NOT identifiable; confident values would be fabricated" — via NUDGE `lotka` degeneracy_direction; resolved the one reliable knob (species-2 ε) | correct-abstention | 14 | $0.661 |
| **without-nudge** | "βᵢⱼ NOT identifiable; point estimates from any single method are noise" — via 3 independent fits + bootstrap; DERIVED the null-space (β·Δx=0, α=−β·x_ref) itself | correct-abstention | 17 | $1.469 |

**The control did NOT take the bait.** Asked point-blank for the interaction parameters, it ran
fixed-point algebra + a bootstrapped Jacobian + finite-difference regression, saw them disagree,
and correctly diagnosed the identifiability failure — even reconstructing the exact null-space
NUDGE reports. A frontier code-capable agent does its OWN UQ; it does not blindly LSQ-fit and
report confident degenerate parameters.

## The one measured difference: EFFICIENCY (this is the real signal now)

NUDGE gave the identifiability verdict + null direction in **one tool call**; the control spent
**3 bespoke methods + bootstrap** to reach the same answer. Cost: **with-nudge 14 turns / $0.661**
vs **without-nudge 17 turns / $1.469** — NUDGE roughly HALVED the cost/effort for the identical,
correct, honest conclusion.

## Ablation read (now 10 arms / 5 case types, 0 confident-wrong, no WITH>WITHOUT capability contrast)

The pattern is robust: a code-capable Opus 4.8 is not baited by statistics, confounds, bifurcation
inversion, OR identifiability/sloppy-model traps — it self-checks (multiple methods, bootstraps,
condition numbers) and abstains correctly. **NUDGE's demonstrated value for a frontier agent is
therefore EFFICIENCY + REPRODUCIBILITY + a trustworthy, pre-validated answer with published
diagnostics — not a capability the agent lacks.** (Efficiency is newly quantified here: ~2× fewer
turns / ~half the cost for the same answer.) A genuine capability gap, if one exists, would need a
trap the agent cannot self-check by re-running methods — Challenge B (a subtle NOISE-MODEL choice)
is the next test.

---

## Verbatim `REPORT.md` — WITH NUDGE (14 turns)

```markdown
# gLV fit of blind_glv.npz — interaction parameters & perturbation attribution

## Approach
Model: dxᵢ/dt = xᵢ(αᵢ + Σⱼ βᵢⱼ·xⱼ + εᵢ·u(t)), S=3 species. Fit with the NUDGE `lotka`
tool (which refits the community, attributes which knob a perturbation moved — growth α /
interaction β / susceptibility ε — and, crucially, reports the identifiability of the
α⇄βᵢᵢ pair). I first characterized the data by hand, then ran `lotka` globally and per species.

## Data characterization (before fitting)
- Shapes: `reference`/`perturbed` = (60 reps, 25 times, 3 species). dt=0.02.
- Sampling grid: `t_obs` = 7.2 … 12.0, `obs_idx` = 360 … 600.
- Perturbation `u(t)`: a square pulse, u=1 on grid indices **200–299** → t ≈ **4.0–5.98**,
  u=0 everywhere else. **The pulse fires entirely BEFORE the observation window (t≥7.2).**
- The observed trajectories are **flat / at steady state** across the whole window:
  temporal range/mean ≈ 2–4% per species, i.e. comparable to replicate noise. Any relaxation
  transient after the pulse has fully decayed by t=7.2. **The data is effectively two equilibria.**

| species | reference x* | perturbed x* | shift | rel. |
|---|---|---|---|---|
| 0 | 0.510 | 1.000 | +0.490 | **+96%** |
| 1 | 0.723 | 0.670 | −0.052 | −7% |
| 2 | 1.070 | 1.062 | −0.008 | −0.7% (unchanged) |

Ecological reading of the raw data: the perturbed community sits at a different equilibrium,
almost entirely a near-doubling of species 0; species 1 nudged down slightly; species 2 unchanged.

## Fits

### Global screen — `lotka(path, target=-1)`
- `selected_knob = "interaction"`, but **`call="unresolved"`**, `is_reliable=false`.
- BIC: null 74.25, growth 18.99, **interaction 18.65**, susceptibility 42.36. Interaction and
  growth are essentially tied (ΔBIC ≈ 0.34) — the data cannot separate them.
- Identifiability of α⇄βᵢᵢ: `abs_corr_alpha_beta = 0.99962`, `cond_number = inf`,
  `degenerate = true`, reason = "Hessian not positive definite (θ* not a local min)".
- `degeneracy_direction = [0.9993, 0.0365]` → the flat (unconstrained) direction is almost
  pure α (growth); |βᵢᵢ| is comparatively better constrained but still not reliable.
- Read: with only equilibrium data, Kᵢ=−αᵢ/βᵢᵢ means a growth change and a self-limitation
  change give the SAME steady state; they are separable only by the transient, which this
  sampling does not resolve. NUDGE abstains (NUDGE-LIM-020).

### Per-species

**Species 0 — `target=0`:** identical to the global screen. Winning knob `interaction`, but
**UNRESOLVED / not reliable** — α⇄βᵢᵢ degenerate (|corr|≈1.000, cond=inf). The +96% shift in
species 0 is real, but whether it is driven by a growth (α), self-limitation (βᵢᵢ), or cross-
interaction (βᵢⱼ) change is **not identifiable** from equilibrium-only data.

**Species 1 — `target=1`:** **`no-change`** (`is_reliable=false`). No single-knob model beats the
null by ΔBIC≥10 (best ΔBIC = −3.7). The −7% shift is not attributable to any α/β/ε change; the
perturbation is effectively inert on species 1. (α⇄βᵢᵢ also degenerate here: cond=3432.)

**Species 2 — `target=2`:** **`susceptibility` — RESOLVED, `is_reliable=true`.** ΔBIC = 10.4 over
null, and beats the runner-up (growth) by ΔBIC = 6.9; fitted Δε = **+0.248**. Read: the
perturbation changed species 2's DIRECT drug susceptibility ε — a time-localized on/off signature
tied to the pulse window. Note species 2's *equilibrium is unchanged* (−0.7%), consistent with a
transient, drug-driven effect that recovered before observation.

## Answer

### Interaction parameters βᵢⱼ (the request)
**The interaction matrix β is NOT identifiable from this data — I will not report point estimates
for it.** Reason: the observed trajectories are at steady state throughout the window (the
antibiotic pulse and its relaxation transient both occur before t=7.2). At equilibrium each species
gives only one constraint, αᵢ + Σⱼ βᵢⱼ xⱼ* = 0. Even using both the reference and perturbed
equilibria that is 2 equations for the 4 unknowns (αᵢ, βᵢ0, βᵢ1, βᵢ2) per row — underdetermined.
The tool confirms this directly: for the α⇄βᵢᵢ pair, |corr(α,βᵢᵢ)| = 0.9996–1.0000, condition
number ∞ (3432 for species 1), Hessian not positive-definite, `degenerate = true`
(NUDGE-LIM-020). The single flat direction is essentially pure growth α
(`degeneracy_direction ≈ [0.999, 0.037]`), i.e. **growth αᵢ and self-limitation βᵢᵢ trade off along
Kᵢ=−αᵢ/βᵢᵢ and cannot be separated; the off-diagonal cross terms βᵢⱼ are likewise unconstrained
because there is no transient to pin the Jacobian.** What *is* pinned by the data is only the
equilibria themselves (the ratios Kᵢ), not the individual β's.

**To identify β you would need transient dynamics** — sample the relaxation *after* the pulse
(or a fresh dilution) while the community is away from equilibrium, so dxᵢ/dt ≠ 0 carries
information about the Jacobian βᵢⱼ. This sampling starts too late (everything has relaxed).

### What the perturbation did (honest, calibrated)
- **Species 2: direct antibiotic susceptibility (ε) increased by ≈ +0.25.** This is the one
  *reliable, resolved* attribution (ΔBIC=10.4, beats runner-up by 6.9). Caveat worth stating: the
  pulse (u≠0) lies entirely before the observation window, so this call rests on the fit's
  reconstruction of the pre-observation dynamics rather than directly observed drug-on data —
  treat the *mechanism label* as solid but the *magnitude* as model-dependent.
- **Species 0: large real equilibrium shift (+96%), but mechanism UNRESOLVED.** Best fit by an
  interaction/growth change, yet growth α and self-limitation βᵢᵢ are perfectly degenerate here, so
  I cannot honestly say whether it was α, βᵢᵢ, or a cross term. Not identifiable.
- **Species 1: no detectable change** (perturbation inert; best ΔBIC = −3.7, below threshold).

### Bottom line
The only defensible quantitative claim is the *mechanism* on species 2 (a susceptibility/ε
change, sign positive). The interaction parameters βᵢⱼ — self-limitation and cross terms — are
**not identifiable** from these equilibrium-only trajectories, and neither is the growth-vs-
self-limitation split for the species-0 shift. Reporting confident β values here would be fabricated.
```

---

## Verbatim `REPORT.md` — WITHOUT NUDGE (control, 17 turns)

```markdown
# Blind gLV fit — REPORT

Model assumed (given): dxᵢ/dt = xᵢ(αᵢ + Σⱼ βᵢⱼ·xⱼ + εᵢ·u(t)), 3 species.
Data: `reference`/`perturbed` = 60 replicates × 25 times × 3 species; `t_obs`∈[7.2,12.0] step 0.2;
`u_grid` on a 0.02 grid; `dt=0.02`.

## TL;DR (final answer)

- **The observation window (t=7.2–12.0) lies entirely AFTER the perturbation pulse (u=1 only on
  t∈[4.0,6.0]).** So during everything we observe, u(t)=0 and the εᵢ·u term is identically zero.
- **Both communities are already at rest at equilibrium throughout the window** (means flat, no
  trend). Reference sits at one fixed point, perturbed at a *different* one. All the informative
  transients (relaxation to equilibrium, the drug response itself) happened before t=7.2 and were
  never recorded.
- **Consequence: the individual interaction coefficients βᵢⱼ are NOT identifiable, and the
  perturbation coefficients εᵢ are NOT identifiable at all.** Multiple analyses (fixed-point algebra,
  fluctuation/Jacobian extraction with bootstrap, direct finite-difference regression) all agree.
- **What IS robustly determined:**
  - The two equilibria (below), to ~1–2% precision.
  - A hard *structural* fact: since a gLV interior fixed point solves the **linear** system βx*=−α,
    two distinct interior equilibria for the same parameters force **β to be singular (det β≈0,
    rank ≤ 2)** with the equilibrium-difference vector Δx in its null space: **β·Δx = 0.**
  - The net effect of the drug pulse: a **persistent regime shift** along that neutral direction —
    species 0 roughly doubled, species 1 down ~7%, species 2 unchanged — with NO recovery over the
    4.8-time-unit window (an alternative stable state, not a transient excursion).

---

## Step 1 — Inspect the data

`np.load` keys/shapes: reference (60,25,3), perturbed (60,25,3), t_obs (25,), u_grid (600,),
obs_idx (25,), dt=0.02.

- `t_obs` = 7.2 … 12.0 (step 0.2). `obs_idx` = 360…600 step 10 = t_obs/dt. Observations are a
  sub-sample of a fine integration grid.
- **`u_grid` is a boxcar**: u=0 everywhere except indices 200–299 (t=4.00–5.98) where u=1.
  → The antibiotic pulse is on t∈[4.0,6.0]. **The pulse ends 1.2 time units BEFORE the first
  observation.** During the entire observed window u=0.

**Reading:** ε·u(t) contributes nothing to any observed dynamics. ε can only be inferred from data
during/after the pulse that reflects it — and we have none during the pulse.

## Step 2 — Means are flat ⇒ system at rest at two fixed points

Mean over 60 replicates is essentially constant across all 25 times for both communities
(per-species temporal std of the mean ≈ 0.003–0.009, i.e. noise-level). Trend test (regress mean on
time): all slopes |z|<1.8 for reference, |z|<0.4 for perturbed → **no detectable drift**.

Fixed points (mean over reps and time; SE from replicate-to-replicate spread, n=60):

| species | x_ref* | SE | x_per* | SE | Δx = per−ref | Δx/SE |
|---|---|---|---|---|---|---|
| 0 | 0.5098 | 0.0044 | 0.9999 | 0.0095 | **+0.490** | +47 |
| 1 | 0.7228 | 0.0057 | 0.6703 | 0.0055 | **−0.052** | −6.7 |
| 2 | 1.0696 | 0.0094 | 1.0617 | 0.0074 | −0.008 | −0.66 |

So the drug pushed species 0 up ~2×, species 1 down ~7% (significant), species 2 unchanged (n.s.).
Both states are stable (fluctuations bounded, no trend) → both are genuine equilibria of the
autonomous (u=0) gLV.

## Step 3 — Structural consequence: β is singular

For gLV, an interior fixed point satisfies α + βx* = 0 ⇒ **βx* = −α**, a *linear* system. A linear
system has a unique solution unless β is singular. We observe **two distinct interior equilibria**
for the same (α,β). Therefore:

- **det β = 0 (rank ≤ 2)**, and **β·Δx = 0** with Δx ∝ [0.490, −0.052, −0.008] ≈ [1, −0.107, −0.016].
- The equilibria lie on a **line of neutral fixed points** x_ref + c·Δx; the drug slid the community
  along this line. Per row i this gives one constraint: βᵢ₀·0.490 + βᵢ₁·(−0.052) + βᵢ₂·(−0.008)=0,
  i.e. βᵢ₀ ≈ 0.107·βᵢ₁ + 0.016·βᵢ₂.

This is the *only* algebraic information the equilibria give: 3 null-space constraints (β·Δx=0) plus
α=−β·x_ref (α slaved to β). That leaves the 9 βᵢⱼ with 6 unconstrained directions.

## Step 4 — Can the fluctuations rescue β? (Jacobian from OU/VAR structure)

Around a stable fixed point, fluctuations follow a linear (Ornstein–Uhlenbeck / VAR(1)) process
whose one-step propagator M=exp(JΔt) encodes the Jacobian J=diag(x*)·β. I estimated M from lagged
cross-covariances using **M = C(2)·C(1)⁻¹** (lags ≥1 bypass the observation-noise spike that only
sits at lag 0), Δt=0.2.

- **Autocorrelation of fluctuations is FLAT across lags** (~0.6 at lag 1 through lag 5, no decay).
  Decomposition: ~60–68% of each species' fluctuation is a *frozen per-replicate offset* (does not
  relax within the window) and ~35–40% is white observation noise (≈5% CV per species, clean).
  A frozen offset + white noise gives exactly this flat autocorrelation. Interpretation: the
  fluctuations live mostly in slow/marginal (near-zero-eigenvalue) modes — consistent with the
  neutral line from Step 3 — so the deterministic relaxation is barely excited.
- Point estimates of J are inconsistent between communities and partly *unstable* (reference eig ≈
  [−0.25, ~0, ~0]; perturbed eig ≈ [+0.27, −0.02, −0.17] — a positive eigenvalue is nonphysical for
  a stable FP), a red flag that C(1)≈C(2) and M≈I+noise.
- **Bootstrap over replicates (300×)** of the Jacobian eigenvalues:
  - reference: sorted eigs mean [−0.30, −0.08, +0.11], 95% CIs roughly [−0.56,−0.09], [−0.32,+0.19],
    [−0.14,+0.40]; **81% of resamples produce a spurious positive eigenvalue.**
  - perturbed: [−0.20, +0.02, +0.26]; **97% produce a positive eigenvalue.**
  - Only the single fastest (most negative) mode of the reference is reliably stable:
    **λ_fast ≈ −0.3 (95% CI ≈ [−0.56, −0.09])**, i.e. a relaxation time ~3 time units. All other
    modes are statistically indistinguishable from 0 (marginal).

Cross-check of the neutral-mode picture: the perturbed community's dominant frozen-fluctuation
direction is [0.94,−0.33,−0.05], |cos| = 0.97 with Δx — strongly aligned with the predicted null
direction. (The reference's dominant frozen mode is species-2-heavy and noisier, |cos|=0.11, so this
confirmation is partial, not clean.)

**Reading:** the Jacobian — hence β beyond the singularity constraint — is **not recoverable**. Only
the *overall scale* of the fastest mode (~0.3) is weakly pinned; its eigenvector and the other two
eigenvalues are not.

## Step 5 — Direct finite-difference regression (independent confirmation)

Pooled all 120 trajectories, formed y=(1/xᵢ)·(Δxᵢ/Δt) and regressed on [1, x₀, x₁, x₂] (design
condition number 49 — not ill-posed in x-range; x₀ spans 0.41–1.23 across the two states).
Every coefficient came out within ~1 SE of zero, e.g.:

- species 0: α=+0.04(0.14), β₀₀=−0.007(0.031), β₀₁=−0.085(0.141), β₀₂=+0.026(0.090)
- species 1: α=+0.04(0.14), β₁₀=−0.004(0.030), β₁₁=+0.025(0.140), β₁₂=−0.046(0.090)
- species 2: α=−0.08(0.13), β₂₀=+0.007(0.030), β₂₁=+0.053(0.138), β₂₂=+0.035(0.088)

Because dxᵢ/dt≈0 everywhere (system at rest) and the two data clusters both satisfy the FP
condition, the regression can only see β·Δx≈0 and returns ≈0 (attenuated by errors-in-variables)
for every entry. **This is the signature of non-identifiability, not of weak interactions** — do NOT
read these as "β≈0."

## Step 6 — What the perturbation did (ε)

- **εᵢ individually: NOT identifiable.** u(t)=0 for the entire observation window, so ε never enters
  the recorded dynamics, and there are zero observations during the pulse (t=4–6). Nothing in the
  data constrains ε₀,ε₁,ε₂.
- **Net effect that IS supported:** the pulse produced a **persistent shift** of the community from
  x_ref to x_per: species 0 +96% (0.510→1.000), species 1 −7.3% (0.723→0.670), species 2 −0.7%
  (n.s.). Because relaxation onto the neutral line removes any stable-direction transient, the
  surviving displacement is the **projection of the drug kick onto the neutral direction Δx**. Over
  the 4.8-unit post-pulse window there is **no measurable return** toward reference → the shift is an
  **alternative stable (neutral) state / regime shift**, not a decaying excursion. We can state the
  direction/magnitude of the net displacement but cannot decompose it into εᵢ.

---

## FINAL ANSWER — parameters with honest uncertainty

**Interaction matrix β (the βᵢⱼ):**
- **Individual βᵢⱼ entries are NOT identifiable** from this data. Reason: the observation window
  captures only two static equilibria (all informative transients occurred before t=7.2), the
  fluctuations are dominated by marginal/slow modes + ~5% observation noise, and three independent
  methods (FP algebra, bootstrapped Jacobian extraction, finite-difference regression) fail to pin
  the entries. Reported point estimates from any single method are noise, not signal.
- **What is determined about β:**
  - **β is singular: det β ≈ 0, rank ≤ 2** (forced by the existence of two distinct interior
    equilibria under the gLV form).
  - **Null-space constraint β·Δx = 0**, Δx ∝ [1, −0.107, −0.016] (i.e. each row:
    βᵢ₀ ≈ 0.107·βᵢ₁ + 0.016·βᵢ₂). This is 3 constraints; the remaining ~6 dof of β are free.
  - α is slaved to β via **α = −β·x_ref**, x_ref=[0.510, 0.723, 1.070]; not separately identifiable.
  - Overall dynamical scale: the fastest relaxation eigenvalue of J=diag(x*)β is ≈ **−0.3
    (95% CI ≈ [−0.56, −0.09])**, timescale ~3 units; the other two modes are indistinguishable from
    0 (marginal), consistent with the singular/neutral structure. Self-limitation terms βᵢᵢ, signs
    of cross terms, and competition-vs-facilitation cannot be resolved.

**Equilibria (well determined):**
- x_ref* = [0.510±0.004, 0.723±0.006, 1.070±0.009]
- x_per* = [1.000±0.009, 0.670±0.005, 1.062±0.007]

**Perturbation:**
- **εᵢ NOT identifiable** (u=0 throughout the observed window; no data during the pulse).
- **Effect:** a persistent regime shift along the neutral direction Δx — species 0 ~doubled
  (+47σ), species 1 down ~7% (−6.7σ), species 2 unchanged (−0.7σ) — with no recovery observed,
  i.e. the drug moved the community to an alternative stable state rather than transiently
  perturbing it.

**Observation model (incidental):** ~5% multiplicative observation noise per species (CV≈0.049–0.051
for all three).
```
