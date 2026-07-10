# Overnight V&V — findings (gate calibration + identifiability)

Run: 300 synthetic *linear* datasets + 552 *switch* datasets (120 calibration +
432 power-sweep grid), ~85 min, **0 failures**. Reproduce: `python
scripts/vv/overnight_sweep.py all` then `analyze`. Figures in `results/`.

---

## 1. The fail-safe property is empirically proven

Across **300 linear** + **120 switch** ground-truth datasets, and **at every
`margin_k`**, the misclassification rate is **0%**. NUDGE never calls the wrong
mechanism — when it can't be sure, it **abstains** (`unresolved` / `off-model`).
This is the "fails safely and loudly" thesis, measured.

The `margin_k` knob is therefore a clean **specificity ↔ sensitivity dial with no
wrong-answer risk on either end**:

| `margin_k` | false-positive rate (linear→switch) | correct attribution | abstains |
|---|---|---|---|
| 0.5 | 24.7% | 98% | 0% |
| 1.0 | 7.7% | 88% | 10% |
| 1.5 | 3.0% | 71% | 28% |
| **1.7 (default)** | **1.7%** | **65%** | **34%** |
| 2.0 | 0.3% | 59% | 41% |
| 2.5 | 0.0% | 43% | 57% |

**Calibrated default = 1.7** — the linear-baseline parsimony gate rejects linear
data with a **< 2% false-positive rate**. (`fit()`'s default was updated to this.)

> Pitch line: *"We calibrated false-positive rejection against 300 synthetic
> linear datasets — under 2% false-switch rate — and across 120 ground-truth
> datasets NUDGE never misclassified a mechanism; it abstains when uncertain."*

## 2. Identifiability — the pre-flight power rule

Correct-attribution fraction vs cells/condition × technical-noise level, at the
default `margin_k=1.7` (`identifiability_cells_noise.png`):

- **Cells/condition is the dominant factor.** Below ~**1000 cells/condition**,
  essentially nothing is attributable — and NUDGE **correctly abstains** rather
  than guessing. At ≥ 1000 cells, mechanisms resolve.
- **Ranking: gain > ceiling ≈ threshold.** Gain is the most robust (identifiable
  across noise once cells suffice). **Ceiling is the most noise-fragile** (0.92 →
  0.17 as technical noise rises, at 1000 cells). Threshold needs the most cells,
  reflecting the **K / v_max partial degeneracy** (both shrink the ON signal).

> Pitch line: *"The identifiability heatmap tells a screener exactly when
> threshold-vs-gain becomes resolvable — a pre-flight power check: ≥ ~1000
> cells/condition, and ceiling attribution needs low technical noise."*

## 3. Caveats (honest)

- **All Tier-0** (inverse crime): generator and fitter share the model + noise.
  The next de-risking step is a Tier-0.5 independent stochastic simulator.
- The **K/v_max degeneracy** is real and quantified here — threshold is the
  hardest call and abstains most. A richer (multi-reporter) readout is the
  candidate fix.
- Numbers are at a moderate fit budget (`n_cells=256, steps=250`); more budget
  lifts the identifiable region (the end-to-end test resolves all three at
  `n_cells=384, steps=400`).

---

# Tier-0.5 — the independent stochastic simulator (inverse-crime break)

`nudge.data.stochastic` generates data from a **tau-leaping SSA of a self-activating
gene** — bimodality is **emergent** (intrinsic Poisson noise + cooperative feedback
populating two basins), not designed-in by a parameter distribution. This is the
honest robustness test Tier-0's inverse crime cannot provide. The observation layer
(`Readout` → NB `sample_counts`) is reused verbatim, so `fit()` consumes it
identically. Two findings — one reassuring, one a genuine boundary.

## T0.5-1. Fail-safe HOLDS on independent stochastic data (matched topology)

Fitting the matched-topology self-activation switch to the SSA data
(`n_cells_per_condition=3000`; fit `n_cells=384, steps=400`, `margin_k=1.7`), across
seeds 0–3 NUDGE emits **zero wrong positive mechanisms** — it either abstains
(`unresolved`/`off-model`) or recovers only the most robust mechanism (**gain**):

| seed | switch detected | thr (→threshold) | gai (→gain) | cei (→ceiling) | wrong? |
|---|---|---|---|---|---|
| 0 | no positives | off-model | unresolved | off-model | none |
| 1 | no positives | off-model | off-model | off-model | none |
| 2 | yes | off-model | **gain ✓** | off-model | none |
| 3 | no positives | off-model | off-model | off-model | none |

The reason is structural and honest: the fit solves each cell from a fixed `x0 = 0`
(`inference/fit.py::_simulate`), so on a *feedback* switch it only ever reaches the
LOW basin and cannot fully represent the emergent HIGH mode — so it **abstains rather
than inventing a mechanism**. Guarded by `tests/verification/test_stochastic_inverse_crime.py`.

> Pitch line: *"On data from a genuinely independent stochastic simulator — bimodality
> emergent, not designed-in — NUDGE emits zero wrong mechanism calls across seeds. It
> abstains on what its model can't represent, and when it does speak it's right."*

**Honest cost:** matched-topology abstention means NUDGE does not reliably *recover*
mechanism on emergent feedback bistability (only gain, only sometimes); it guarantees
it won't be *wrong*.

## T0.5-2. Fail-safe BREAKS under topology misspecification (the boundary)

Fitting a *plausible-but-wrong* feedforward hypothesis (IN→SW) to the same feedback
data, NUDGE partially represents the bimodality (via input spread) and then
**misclassifies**: the gain mover (n×0.2) was called `threshold` with confidence 1.00
at seed 0, at **every** `margin_k` (1.0 / 1.7 / 2.5) — a structural wrong call, not a
margin artifact. Seed-dependent (seed 1 → correct `gain`; seed 2 → safe `off-model`).

So the fail-safe guarantee is **conditional on the fitted topology being approximately
right**: "the edge's K/n/vmax" means different things in different topologies, and a
confidently-wrong mechanism can slip past the abstention gates under strong topology
mismatch. Candidate `NUDGE-LIM-*`: *mechanism attribution presumes approximately
correct topology; report attribution as topology-conditional.* Mitigations to explore:
a topology-adequacy check, a multi-reporter readout, and **multi-basin IC seeding** so
the fit can represent emergent feedback bistability directly (see the async R&D spike).

## T0.5-3. Async R&D spike (autonomous subagent) — multi-basin IC relaxation is FEASIBLE

An **autonomous background subagent** ran a standalone JAX/Optax investigation (no NUDGE
code imported; `scratchpad/spike_multibasin_fit.py` + `spike_multibasin_REPORT.md`)
into the root cause of finding T0.5-1: the fit solves from a fixed `x0 = 0`, so it
only reaches the LOW basin and can't represent emergent bistability. Could a
**multi-basin IC relaxation** — a latent basin-occupancy `p` fit by gradient descent,
seeding a fraction `p` of cells at the high basin and `1−p` at the low — recover a
bimodal population without the gradients going non-convex?

**Verdict: feasible, and recommended to prototype.** The decisive contrast:
- **Stage 1 (toy GMM, *free* mode means):** frequent **mode collapse** — ~half of
  inits collapse to a single wide Gaussian (`p→0/1`, σ inflates 2.3–2.8), e.g.
  `p_true=0.5 → p̂=0.053`. The non-convexity lives in the **free mode locations**.
- **Stage 2 (self-activating ODE; modes *pinned* to the fixed points low=0.050,
  high=2.021, via the exact semi-implicit solve):** recovering `p` is **easy, robust,
  init-insensitive, zero NaNs** — soft weighted mixture gives `p̂` within ≤0.03 of
  truth (loss ~0.003), `init_logit` −3 and +3 both converge, and a **joint `p+K` fit
  recovers both** (`p̂=0.611, K̂=0.911` vs 0.6/1.0). Gumbel-softmax also works but is
  noisier; the soft variant wins.

The key insight: **NUDGE pins mode locations to the ODE's fixed points, which removes
exactly the non-convexity that wrecks a free Gaussian mixture.** So a per-population
basin-occupancy latent `p` fit jointly with kinetics against the energy-distance loss
would let NUDGE *represent* emergent-bistable populations (instead of abstaining, T0.5-1)
and *attribute* basin-occupancy shifts vs. threshold/gain moves — a candidate fix for
the T0.5-2 boundary too.

**Honest caveat:** this is a simplified 1-species / 1-parameter proxy; the report itself
flags that fitting `p` alongside the full kinetic set (with the known K/v_max degeneracy,
multi-condition) could reintroduce coupling and needs a recovery test. So this is a
**promising, low-risk direction to prototype**, not a proven drop-in. Carry-over
guardrails: derive basin seeds from the fitted fixed points (never free); keep the
elevated-loss self-diagnostic (0.2+ vs 0.003) as an abstention check.

> Meta (creative-AI angle): the feasibility of a core fit-engine extension was scoped by
> an autonomous background agent running an isolated numerical spike in parallel with the
> main build — the "wrong place for the instability" diagnosis is the kind of result that
> reframes an architecture decision cheaply.

## T0.5-4. Multi-basin integrated into NUDGE — representation works, attribution degenerates

Acting on T0.5-3, the multi-basin model was built *inside* NUDGE (`inference/losses.py`
`energy_distance_weighted`; `inference/fit.py` `fit_multibasin_parameters` +
`fit_multibasin`, alongside the unchanged `fit`). Two clear results:

**Representation — a validated win.** On the Tier-0.5 WT data, the two-basin weighted
mixture cuts the fit loss from **0.166 → 0.016 (≈10×)** vs the single-basin `fit`, and
recovers occupancy `p̂ = 0.644` vs a true ON-fraction ≈ 0.62 (and `vmax̂ = 2.09` vs 2.0).
The spike's conclusion holds in the real codebase: because the modes are pinned
to the ODE fixed points, `p` is recoverable and the bimodality is representable.

**Attribution — a real degeneracy (fail-safe violation).** Pointing the full
orchestration at the Tier-0.5 movers, `fit_multibasin` **recovered gain at seed 0** where
single-basin `fit` abstained (a genuine gain) — **but at seed 2 it emitted `gain→ceiling`,
a confident WRONG call** where single-basin `fit` was correct. The error is consistent
(present both before and after the fix below), so it is structural, not noise. Root cause:
a **two-fixed-mode mixture cannot represent *graded* data**. A gain reduction (n↓) makes
the switch graded — intermediate cells the two modes can't hold — so the model fits it as
a *ceiling* reduction (vmax↓, which also lowers the high mode). Gain and ceiling become
degenerate.

Attempted fix — **decouple occupancy from kinetics** (estimate per-condition `p*` once,
then PIN it while the restricted kinetic fits compete on residual shape, `fixed_p`): this
*recovered gain at seed 0* but did **not** fix seed 2. The degeneracy is in the two-mode
representation itself, not just the `p` latent.

**Disposition.** `fit_multibasin` is kept as a **validated representation building block
and a documented negative**, marked EXPERIMENTAL / not-fail-safe; the Tier-0.5 guard test
stays on single-basin `fit` (never wrong). Single-basin trades attribution for
conservative abstention and keeps the fail-safe guarantee; multi-basin represents more but
can be confidently wrong — the classic sensitivity/safety trade, now measured.

**Next → resolved in T0.5-5.** Add a **third "transition" mode at the unstable saddle** —
a gain reduction piles intermediate cells near the saddle while a ceiling reduction does
not, a candidate signal to break the degeneracy. Prototyped by an autonomous subagent, then
integrated and verified (T0.5-5).

## T0.5-5. The saddle transition-mode gain gate — degeneracy BROKEN, fail-safe

The user's insight: a two-fixed-mode mixture has nowhere to put the *intermediate* cells
a gain reduction creates, but the ODE has a third relevant point — the **unstable saddle**
between the basins. Adding a transition mode there breaks the T0.5-4 degeneracy.

**Mechanistic root (why it must work).** Under the *true* perturbed kinetics the switch
changes regime: a gain reduction (n: 6→1.2) **collapses bistability to a single
intermediate fixed point at 1.116** — right where the WT saddle sits (0.975) — so gain
data is *graded*, centred on the saddle. Threshold (K·3) and ceiling (vmax·0.3) instead go
monostable-**low** (0.050). So only a gain reduction fills the saddle region.

**The discriminator (verified, fail-safe).** A restricted **free-`n`** three-mode fit is
*forced* to spend transition-mode weight to represent graded data. Measured `w_trans`
(free-n) across seeds 0–3, all four mechanism classes:

| condition | free-n `w_trans` (seeds 0–3) | mean |
|---|---|---|
| no-effect (WT) | 0.095 / 0.073 / 0.121 / 0.118 | 0.10 |
| threshold | 0.010 / 0.010 / 0.010 / 0.010 | 0.01 |
| ceiling | 0.009 / 0.010 / 0.010 / 0.010 | 0.01 |
| **gain** | **0.871 / 0.890 / 0.873 / 0.937** | **0.89** |

**Only gain exceeds ~0.12** — a τ=0.5 gate has a 0.12↔0.87 margin and misfires on
nothing (verified independently in-codebase, no NaNs across 80+ fits).

**End-to-end result.** `fit_multibasin(transition_mode=True)` **recovers `gai→gain` at all
four seeds — including the notorious seed 2** where single-basin `fit` abstains and the
2-basin model was confidently wrong (`gai→ceiling`) — with **zero wrong positives**;
threshold/ceiling safely abstain (`off-model`). This is fail-safe attribution on emergent
bistability. Guarded by `tests/verification/test_stochastic_inverse_crime.py`.

**Fail-safe engineering (the failure modes we designed against).** The user flagged two;
we added four more:
- **FM1 bifurcation-collapse NaN:** the transition sample is a scalar-centre + strictly-
  positive lognormal width (no covariance to collapse); when a fit wanders monostable the
  transition weight is *masked to 0* (`trans_valid`) so no fabricated saddle enters a
  gradient. Zero NaNs observed.
- **FM2 N-D saddle:** decoupled onto `Circuit.fixed_points`/`transition_state` (1-D
  root-find for a self-activation switch; `None` for N-species). The gate is guarded by
  `n_species == 1`; N-species defers to honest single-basin abstention — the fix ships for
  the 1-D case we proved, and crashes on nothing (verified: a 2-species fit runs, gate
  inert).
- **FM3 off-model / FM5 no-effect:** the gate runs *after* those gates (`decide_with_transition`),
  so a badly-fit or WT-like condition can never be promoted to GAIN.
- **FM4 which probe:** the gate reads the **free-`n`** fit's `w_trans` specifically — free-K
  overlaps (gain 0.45 vs ceiling 0.17), free-n is clean (0.89 vs 0.01).
- **FM6:** `gain_wtrans_tau` is a parameter (default 0.5) in a wide margin; the root-finder
  never raises (returns `None` on any failure). A subtle integration bug also surfaced and
  was fixed: the restricted fits must start from the **nominal** circuit, not the WT-
  recovered one (the 2-basin WT fit distorts `n`, shifting the saddle).

**Honest limits.** Verified for a 1-species self-activation switch at a strong gain factor
(n·0.2); the τ and the gain SIGNATURE are calibrated there, not yet swept over milder
factors or other 1-D circuits. Threshold/ceiling still abstain on this data (the fix is
specifically for the gain/ceiling *confusion*, not K/vmax recovery). N-D saddle finding
(mutual-inhibition toggle switches) is future work — a natural next subagent spike.

> The arc, all in the git history: single-basin abstains (safe) → 2-basin represents but
> misclassifies (T0.5-4) → a user hypothesis (saddle) scoped by an autonomous spike →
> integrated with six anticipated failure modes → fail-safe gain attribution on emergent
> bistability. The crown-jewel fail-safe guarantee was extended, not compromised.

---

# Decoy battery — bimodality that is NOT a switch (NUDGE-DECOY-001)

The "fails safely and loudly" claim is only real if NUDGE declines on adversarial
negatives. The first battery case is the scientifically-flagged one:

**NUDGE-DECOY-001 — telegraph / noise-induced bimodality (To & Maheshri 2010).** A
two-state promoter with **slow switching** and **non-cooperative (n=1) positive feedback**
is deterministically **monostable** (a single mean-field fixed point, verified — count
units ≈ 88.6) yet produces a **clearly bimodal** snapshot (a large OFF spike + a populated
ON mode) because the protein sees a quasi-static promoter. A naive bimodality detector
would call ultrasensitivity.

**Result: NUDGE correctly abstains, on both engines.** Fitting the bistable-switch
hypothesis, the circuit-level linear-baseline parsimony gate finds the mechanistic model
does not beat linear beyond the noise floor → `beats_linear_baseline = False` → every
condition is `off-model` ("no switch detected"). This holds for **both** the single-basin
`fit` and the powerful `fit_multibasin(transition_mode=True)` — the parsimony gate fires
first, so the saddle gain gate never even gets the chance to misread the telegraph's
intermediate cells. Guarded by `tests/decoys/test_battery.py` (slow) + a fast
generator check (`tests/data/test_telegraph.py`); limitation `NUDGE-LIM-001`.

> Takeaway: **bimodal ≠ switch**, and NUDGE encodes that distinction structurally (the
> parsimony gate), not by eyeballing a histogram. The same reasoning underwrites trusting
> a *positive* call on real bimodal data.

## Decoy battery — the gates it now covers

Five passing cases spanning three gates, plus one documented-limitation witness:

| Decoy | Failure mode faked | Gate exercised | Verdict |
|---|---|---|---|
| 001 telegraph | noise-induced bimodality | parsimony | off-model ✓ |
| 002 mixture | cell-type / doublet mixture | parsimony | off-model ✓ |
| 003 dropout | technical zero-peak | parsimony | off-model ✓ |
| 004 dead-guide | null perturbation on a real switch | no-effect | no-effect ✓ |
| 005 marginal Hill | within-floor nonlinearity | parsimony **margin** | off-model ✓ |
| **006 nonlinear readout** | readout ultrasensitivity | — | **fooled → NUDGE-LIM-006** |

All five passing cases hold on **both** fit paths, and — audited explicitly — at fit()'s
*default* budget (256/300) as well as the battery's 384/400. The abstention is **not** a
budget artifact; the only budget-sensitive case is the genuine readout confound (006).

## NUDGE-LIM-006 — a nonlinear readout is misattributed as a circuit switch (verified bound)

The sharpest bound we've found on the fail-safe guarantee, and a case study in verifying
an AI collaborator's claim. An autonomous spike proposed decoy 006 (a linear circuit read
through a Hill reporter) and reported NUDGE is *fooled* into a confident false positive.
**Independent reproduction changed the finding twice:**
1. At the spike's setup I first got *off-model* (NUDGE declines) — a contradiction.
2. Reconciling, the difference was **fit budget**: the spike used fit()'s default
   (n_cells=256, steps=300) → fooled; at 384/400 the *same data* correctly abstains. So
   the false positive is partly a **budget** effect the spike hadn't isolated.
3. But a seed sweep at the *higher* budget showed it is **not** purely budget: a steep
   readout (Hill h≈6) + a strong perturbation still fools NUDGE into **ceiling** (2/3
   seeds). So the limitation is **real and structural**, merely **budget-mitigated**.

**The mechanism (real):** NUDGE fixes the readout as affine, so it cannot tell whether
ultrasensitivity lives in the *circuit* or the *measurement* — only the composition
readout∘circuit is observed. A sigmoidal reporter on a broad input yields a skewed,
pseudo-bimodal distribution the affine-readout switch model explains by bending the
circuit; a perturbation that shifts the sigmoid on-fraction is then attributed to a
mechanism (variably threshold / gain / ceiling across seeds — it fails *unreliably*).

**Disposition (honest):** 006 is **not** a passing decoy (asserting NUDGE declines would
be false); it is `NUDGE-LIM-006` + an **xfail witness** (`generate_readout_nonlinearity_decoy`)
that asserts the *desired* abstention and currently xfails — so a future fix flips it to a
failure and forces a docs update. This makes the claim precise: **"fails safely" holds
under an approximately-affine readout.**

**Turning the bound into a feature (in progress):** an identifiability spike is testing
whether jointly fitting the circuit *and* readout nonlinearities can separate them, and
whether a **constitutive-reporter control** (a calibration population that drives the
reporter independent of the circuit) anchors the readout and breaks the degeneracy — which
would be both a candidate NUDGE capability and a concrete experimental-design suggestion to
the field.

## NUDGE-LIM-006 mitigation — VALIDATED: a constitutive control separates readout from circuit

Turning the bound into a contribution. A standalone JAX identifiability study (a latent
lognormal input → Hill circuit `a=g(u)` → saturating Hill readout `Λ=R(a)` → NB counts,
fit by energy distance) — **audited for leakage and independently reproduced** — tested
whether the readout/circuit confound can be broken:

- **Single population → genuinely degenerate (why LIM-006 exists).** Fitting the circuit
  Hill `n` and readout Hill `h` jointly to one population: the composition `R∘g` is pinned
  (observed-map rel-RMSE 6.6%) but the *split* is not. The profile over circuit `n` is
  **FLAT** (loss span 0.0003 across n∈[1,10]); a graded `n=1` circuit (no switch — all
  nonlinearity in the reporter) fits within 0.0001 of the true `n=3`. `corr(circuit n,
  readout h) = −0.905` among near-optimal fits. **You cannot even tell a circuit switch
  exists.** LIM-006 is thus a fundamental identifiability degeneracy, not a fitting weakness.

- **Add a constitutive-reporter control → the degeneracy breaks.** A calibration population
  that drives the reporter at known activity doses (bypassing the circuit) anchors the
  readout. Re-profiling: `n=1` is now **REJECTED** (Δloss 0.017 ≫ floor; the `n`-profile
  span grows ~50×), the ridge collapses (near-optimal multistart fraction 0.07 → 1.00), and
  the data now say the ultrasensitivity is **biological**. **Honest caveat (confirmed):**
  the control rejects "no switch" but does *not* point-identify the exact `n` (recovered
  ≈5 vs true 3 — the circuit's internal K/n/vmax trade-off persists). Full point-ID would
  need a second anchor (an input titration / circuit dose-response).

**Verification:** I read the forward model + control (the control uses only readout params
at known doses — no circuit-param leak) and independently reproduced the headline (no-ctrl
`n`-profile flat 0.0003; with-ctrl `n=1` rejected 0.017). Standalone artifacts in
`scratchpad/spike_ident_*` (not in the repo).

**Contribution.** (i) A concrete NUDGE feature candidate: an optional calibration-control
channel; absent one, abstain on the circuit-vs-readout axis rather than mis-attribute.
(ii) A concrete screen-design suggestion to the field: include a constitutively-driven
reporter titration when trustworthy mechanism attribution is needed. The limitation
(LIM-006) and its validated mitigation are, together, a publishable methodological result.

---

# N-D saddle: the finder + representation generalize; the gain gate is 1-D-specific

Generalizing the saddle transition-mode gain gate beyond the 1-species self-activation
switch to the canonical **2-node mutual-inhibition toggle** — done as bite-size milestones,
each ending with the full slow lane (5-case battery + Tier-0.5 + saddle) **green and
identical to baseline**, so nothing regressed the 1-species fail-safe guarantee.

**Landed (reusable, verified — M1/M2).**
- **N-D fixed-point / saddle finder** (`Circuit.fixed_points` / `transition_state`):
  multi-start Newton + Jacobian-eigenvalue index classification (index-1 saddle = exactly
  one +Re eigenvalue), vector field from `Circuit.production` (any topology). Reproduces the
  spike exactly on the toggle (symmetric saddle `[1.017, 1.017]`; asymmetric off-diagonal
  `[0.933, 1.061]`; monostable / feedforward → no saddle). Engineered against the numerical
  traps: local x64 context (f32 Newton cancels near the saddle-node), static padded `vmap`
  output + `jnp` masked dedupe (no XLA dynamic-shape cliff, no host sync), `stop_gradient`
  at the loss boundary.
- **N-D multi-basin representation**: the transition fit seeds basins at the **stable fixed
  points** (static slots + a monostable-excursion fallback + a deterministic root-sort for
  stable slot identity, so Optax momentum isn't thrashed). It *represents* a bistable toggle
  well — WT loss **0.015 vs 0.83** for the naive `0`/`high_ic` seeding (56×).

**The gain gate does NOT extend — a measured NO-GO (M3).** The `w_trans > τ → GAIN` signature
was 1-D-specific: it worked because a gain reduction on a *self-activation* loop collapses
the switch to a single intermediate fixed point *at the saddle* (graded cells → transition
mode). On a toggle, reducing cooperativity on **one** repression edge does not collapse the
system — the other edge keeps it bistable, so cells stay in the two basins. Measured free-`n`
`w_trans` for the gain condition across seeds: **0.00 / 0.25 / 0.22** — weakly elevated but
far below the calibrated τ=0.5 and seed-unreliable (vs the clean 1-D 0.87–0.94); the other
classes stay ≤0.04. So the gate stays guarded to `n_species == 1`.

**Fail-safe preserved.** With the gate 1-D-guarded, NUDGE **abstains** (off-model) on a toggle
rather than misclassifying — locked in by `tests/verification/test_toggle_nd_safety.py` (no
wrong positive on toggle threshold/gain/ceiling data). The N-D finder + representation are
reusable infrastructure (they make multi-species attribution *approachable*, and unblock the
T-cell circuits). We shipped the finder + representation and *declined* to ship an unreliable
N-D gate.

**Performance — the finder is jitted (byte-identical, ~300× cheaper per call).** The N-D
finder is recomputed every optimizer step (the saddle/basin seeds move as the kinetics do).
Profiling showed the per-step cost was *not* re-compilation or Newton solve-count but **Python
re-tracing** of the un-jitted `vmap(jacfwd-Newton)` plus per-root host syncs — ~0.3 s/step. The
Newton/dedupe/eigenvalue core is now a **jitted, per-topology-cached kernel** with the kinetics
as a *traced argument* (`_nd_kernel`): it traces once and only executes thereafter (~1 ms/call,
**333× per-call**; a full toggle transition fit **26 s → 4.1 s, 6.3× end-to-end** — the forward
simulation is now the floor). Results are **byte-identical** to the eager finder (same roots →
same fit trajectory: recovered `n`/`w_trans` unchanged), so this is a pure speedup, not a
behaviour change. (A warm-start/trust-region attempt was tried first and *rejected*: it gave
~1× — because tracing, not solve-count, was the cost — and introduced a reproducibility
divergence. Jitting subsumes it.)

**Why the gain gate failed, and the signature that should work (researched — `design/`).** An
adversarially-verified `/deep-research` sweep (non-equilibrium stat-mech, not just comp-bio)
explains the NO-GO and points past it, in **`design/TOGGLE_ATTRIBUTION_RESEARCH.md`**: mixture
*weights* (basin occupancy) are set by a **non-gradient quasi-potential barrier balance**, not
the deterministic saddle — so a saddle-centred `w_trans` was always the wrong channel for a
toggle (the "gain zeroes a lobe" claim was the one *refuted* finding). The gain signal for a
toggle lives in each lobe's **covariance** (the linear-noise **Lyapunov** solve `AΣ+ΣAᵀ+D=0`:
gain enters `A` via the repression elasticity ∝ `m`, ceiling via mean copy number — different
channels) and in **separatrix orientation**.

**Measured (Fisher-information spike) — the confound is gain⇄threshold, not gain⇄ceiling.** We
built the LNA Gaussian mixture (means from the fixed points via an IFT stop-grad step; covariances
from the autodiff-Jacobian Lyapunov solve) and computed the **FIM** over `(log m, log v, log K)`
of the perturbed edge (empirical Fisher, 6 seeds × N=20 000; sloppy-eig seed-std 3e-4;
`scripts/vv/fisher_sloppiness.py`). Result, correcting the medium-confidence synthesis:
(1) the sloppy direction is **gain(m)⇄threshold(K)** — `corr(log m, log K) = −0.986`, cond# ≈ 210;
(2) **ceiling(v) is the *most* identifiable** parameter (it sets the high-mode plateau ≈ b+v —
`dμ/d log v ≈ +2`), *not* confounded; (3) the confound is analytic — the high-repressor Hill term
is `(K/B)^m`, so the snapshot constrains only `m·ln(K/B)`. **What breaks it:** a **constitutive
control** does *not* (smallest FIM eigenvalue ×1.01 — it reads the already-identified `v`); a
**second operating point** (dose/basal shift) does (×16.5; cond# 210→22). So the toggle
degeneracy-breaker is a **second condition**, not the LIM-006 constitutive control (a different
axis). This also explains from information geometry *why* a single-snapshot toggle fit should
**abstain between gain and threshold** — consistent with the fail-safe behaviour we ship.
Researched + measured; the covariance attribution loss itself is not yet built. Full write-up:
`design/TOGGLE_ATTRIBUTION_RESEARCH.md`.

**The degeneracy is robust to extrinsic noise (measured; `scripts/vv/fisher_extrinsic.py`).** We
extended the FIM with the generator's **extrinsic** cell-to-cell spread — a per-cell log-normal
factor on species `basal` and `decay` (faithful to `data/synthetic.py::_per_cell_params`),
propagated to a per-mode covariance `Σ_ext = σ²(J_b J_bᵀ + J_d J_dᵀ)` (a Monte-Carlo re-solve
confirms the first-order form to ~18%). Modeled as a *known* nuisance, extrinsic noise is
**benign — mildly beneficial**: sweeping σ∈{0…0.5}, the gain⇄threshold confound does *not*
deepen (`corr(m,K)` −0.986→−0.980 at σ=0.3), ceiling never rotates into the null space (its
loading on the sloppy eigenvector stays −0.01 → stays identifiable), and the identifiability
floor actually *rises* ×1.5 at σ=0.3 (a heteroscedastic `dΣ_ext/dθ` information channel).
Independently reproduced (subagent → main-loop). **Caveat:** this assumes σ is *known*; a
misspecified/unknown extrinsic σ is the untested next check.

# Covariance attribution (the Lyapunov path): the confound reproduced, and broken

The Fisher-information analysis said the gain/threshold/ceiling signal lives in each toggle
lobe's **covariance**, not its weight. We built that attribution — a **covariance-structured
linear-noise Gaussian-mixture fit** (`nudge.inference.lyapunov`) — as an **additive, opt-in,
guarded** capability that never touches the energy-distance `fit()` default (risk isolation:
the decoy battery cannot be routed into the LNA). Milestones, each green:

- **M0 — the primitive** (`Circuit.mode_covariances`): per-stable-mode covariance from the
  Lyapunov equation `AΣ+ΣAᵀ+D=0` (autodiff Jacobian; `D=diag(2·decay·μ)`), reproducing the
  FIM reference *exactly* (toggle lobe cov diag `[0.199, 2.055]`, corr `−0.324`).
- **M1 — the differentiable fit** (`fit_lyapunov_parameters`): mode means made
  differentiable by an implicit-function-theorem step, covariances by the Lyapunov solve; an
  optax loop maximizes the mixture likelihood. Inverse-crime recovery: gain 1%, ceiling 3.7%,
  threshold 12%. **Design finding:** a free global `scale` is degenerate with `vmax` (both
  scale the mode means) — the LNA rediscovery of why single-cell pipelines **normalize by
  sequencing depth**. So depth is pinned from an *independent* reference (`calibrate_from_wt`):
  calibrating it from a perturbed condition's own magnitude would hide a ceiling change in
  depth and make ceiling unidentifiable.
- **M2 — the confound, honestly** (`attribute_lyapunov_single`): restricted free-K/n/vmax
  fits. Measured (3 seeds): true gain → NLL(gain)≈NLL(threshold), ceiling +0.05 worse; true
  ceiling → NLL(ceiling) best by +0.20. So it **identifies ceiling and abstains
  (`gain_or_threshold`) between gain and threshold** — never a bare gain/threshold from one
  snapshot. Correct-or-abstain, never confidently wrong.
- **M3 — the breaker** (`fit_lyapunov_multi` / `attribute_lyapunov_multi`): a **shared**
  kinetic value fit jointly across operating points (a clean `OperatingPoint` list API). A
  true gain perturbation observed at basal-B 0.05 + 0.30: the gain↔threshold NLL gap widens
  **0.005 → 0.098 (~20×)** — mirroring the FIM's ×16 — and attribution flips from *abstain*
  to **`gain` (resolved)**. Both gain and threshold truth resolve correctly. The synthetic
  operating point (a basal shift) is the stand-in for a **second Gladstone target**, which is
  exactly why the multi-target screen supplies the operating points the FIM proved we need.
- **M4 — the fail-safe guard** (`lna_reliable`): the LNA Gaussian is local and second-order,
  so attribution **abstains loudly** where it breaks — **low sequencing depth**
  (`scale·peak < 15` counts), **near a saddle-node** (a lobe's covariance swells, CV > 1.5),
  or **monostability** (<2 stable modes). Verified to pass a well-sampled bistable toggle and
  trip in each regime; wired into both attribution entry points before any fit runs.

**Honest bounds.** Validated on LNA/synthetic ground truth (inverse-crime + independent
operating points), *not yet on real data*; the low-count guard exists precisely because a
toggle's OFF state is where the Gaussian is weakest. Kept opt-in until proven on the Gladstone
T-cell screen. Full slow lane (5 decoys + LIM-006 + Tier-0.5 + saddle) stays green — the path
is additive. Tests: `tests/inference/test_lyapunov.py`, `tests/core/test_mode_covariance.py`.

## Independent-SSA validation — the single snapshot degenerates, the second operating point recovers (fail-safe)

M1–M3 above were measured on **inverse-crime** data (`sample_lna_mixture` — cells drawn from
the very LNA Gaussian the fitter maximizes). The honest question is whether the covariance
signature survives the **inverse-crime break**: data from the *independent* tau-leaping SSA
(`generate_toggle_perturbseq`, the true stochastic stationary distribution), bridged to
activity exactly as the real-data path does (`inference.bridge.counts_to_activity`), at a
depth that clears the `lna_reliable` guard (a deeper readout, `Readout.identity(2, scale=15)`
→ scale·peak ≈ 30 ≥ 15; at the generator's default scale=5 NUDGE correctly abstains before
any fit). Mild single-edge perturbations (gain n:4→2.4, threshold K:1→1.6, ceiling
vmax:2→1.2) keep both attractors populated. 3 seeds each. Reproduce:
`scripts/vv/toggle_lyapunov_ssa.py` (raw numbers in `toggle_lyapunov_ssa_RESULTS.txt`).

**Result — a documented single-snapshot NEGATIVE and a guarded two-point POSITIVE, 0 wrong:**

| true mechanism | single snapshot (basal 0.05) | two operating points (basal 0.05 + 0.30) |
|---|---|---|
| **gain** (n) | `unresolved` (3/3) | `unresolved` (3/3) — honest abstention |
| **threshold** (K) | `unresolved` (3/3) | **`threshold`** (3/3 ✓, wins by 0.34–0.35 nats) |
| **ceiling** (vmax) | `gain_or_threshold` (3/3) | **`ceiling`** (3/3 ✓, wins by 0.16–0.20 nats) |

1. **The single snapshot DEGENERATES on independent SSA.** The inverse-crime claim that
   "ceiling is the identifiable one" does **not** survive: on the true stochastic
   distribution the free-vmax fit becomes the **worst** explanation of a true ceiling (NLL
   6.47 / 6.47 / **6.55**), so the single-condition call mis-narrows a true ceiling to
   `gain_or_threshold` — a label that *excludes the correct answer*. Gain and threshold
   abstain (`unresolved`; vmax marginally best but inside the 0.05 ceiling margin). Crucially
   the single path **never emits a bare gain/threshold/ceiling** — only abstention-class
   labels — so it is never *confidently* wrong; but a single toggle snapshot must not be read
   as a positive. Why it breaks: the homoscedastic LNA Gaussian is misspecified against the
   NB/discrete/skewed true stationary distribution, and — as the deep-research synthesis
   warned — the Gaussian is weakest exactly where a perturbation pushes a lobe.
2. **The second operating point RECOVERS threshold + ceiling, and abstains on gain — 0
   confident-wrong.** The shared-parameter joint fit across two basal-B operating points
   resolves **threshold** and **ceiling** correctly at *every* seed (clear margins), and
   honestly **abstains on gain** (the residual gain⇄threshold confound the second point does
   not break *for gain* on this independent data — the inverse-crime M3 gain-resolution did
   not survive, threshold/ceiling did). Recovery **6/9 correct, 3/9 honest abstention (all
   gain), 0/9 wrong** — the non-negotiable fail-safe holds on genuinely independent
   stochastic data.

**Disposition (honest).** This is the first evidence the covariance signature separates
mechanism on **non-inverse-crime** toggle data: a **guarded positive** for the
two-operating-point breaker (recover-or-abstain, never wrong) plus a **documented negative**
for the single snapshot (it degenerates — inverse-crime-validated only). It remains
additive/opt-in and is **not** wired into `fit()`; NUDGE's production toggle path still
abstains (`tests/verification/test_toggle_nd_safety.py`). Still synthetic (SSA, not real
data), still depth-gated. Locked by `tests/inference/test_lyapunov_toggle_ssa.py`
(multi recover-or-abstain + single never-confidently-wrong).

# Phase 4 — real data (Gladstone CD4+ T-cell screen): NUDGE abstains, honestly

NUDGE ran end-to-end on the **real** genome-scale CRISPRi Perturb-seq screen
(`D1_Stim8hr.assigned_guide.h5ad`, 2.79M cells × 18,130 genes, 150 GB; GSE314342) — the
pointer-based loader read only the Ras-switch guides + IEG panel (6,367 cells: 6,000 NTC +
SOS1 110, RASGRP1 24, RASA2 233) without loading the matrix. `scripts/vv/gladstone_attribution.py`.

**Result: `no-switch` — NUDGE abstained.** The BIC topology gate scored the WT (NTC) IEG-
activation readout and preferred the no-switch single-Gaussian null over a 1-node bistable
switch (BIC **40,556 vs 40,599**). Grounding the call: the activation distribution is a single
sharp low mode + a sparse heavy tail — **5,884 / 6,000 cells in the lowest bin**, skew ≈ 12,
kurtosis ≈ 224; EGR1/FOS/NR4A1 are ~95% zero at 8 h; IL2/CD69 carry the (mostly-low) signal.
That is a graded/heavy-tailed *unimodal* population, **not two populated attractor states**.
(Sarle's bimodality coefficient reads 0.66, a false positive driven by the extreme skew — the
histogram is unambiguously one mode + a tail.)

**This is the fail-safe guarantee working on real biology.** Rather than force a
gain/threshold/ceiling call on data that doesn't support a switch, NUDGE declined — exactly
its defining property, now demonstrated on a real 150 GB screen, not a synthetic. Per-target
attribution never ran (no switch to attribute; and the targets are anyway underpowered — 24–233
cells vs the ~1,000/condition the FIM analysis showed is needed for identifiability, a real
limitation of genome-*wide* vs focused screens).

**Honest interpretation + follow-ups (we did NOT tune to manufacture a switch).** At 8 h post-
stimulation the *transcriptional* IEG output is a graded single population; the Das-2009 Ras
switch is bistable at the *signaling* (Ras-GTP) level, which a steady-state transcriptomic
snapshot need not resolve as two modes. Legitimate next steps, none of which change the honest
first-pass verdict: the other stim timepoints (Rest / Stim48hr — the operating-point axis, and
Stim48hr may show more commitment); a signaling-proximal or single-strong-marker readout
instead of the IEG mean; and a focused (not genome-wide) screen with enough cells/guide. The
value here is a *measured, honest* verdict on real data — the tool doing the hard thing.

# Phase 4b — dose-response attribution: OCT4 resolves as a switch, NANOG honestly abstains

A second real-data result, and the first *positive* mechanism call on real biology. Single-cell
bimodality and bulk dose-response ultrasensitivity are two measurements of one Hill circuit, so
`nudge.inference.dose_response` fits the *same* Hill primitive (`hill_repression`) to a readout's
response across a graded perturbation dose, with the *same* BIC parsimony discipline as topology
model-selection. Applied to the OCT4/NANOG pluripotency screen (GSE283614, Yao et al. 2025): a
self-renewal signature (SOX2/LIN28A/UTF1/DNMT3B/TDGF1/ZFP42/SALL4) vs each factor's own guide-
level knockdown (the guide axis *is* a dose axis).

**Result: OCT4 → `switch`; NANOG → `unresolved` (abstain).**
- **OCT4** (16 guide-dose points): apparent gain **n ≈ 6.7 (95% CI 4.5–12)**, K ≈ 0.65, **R² = 0.99**,
  ΔBIC(graded−switch) = +54 — an abrupt, ultrasensitive switch. Its inflection is *inside* the
  knockdown range. Matches the literature threshold behavior (Niwa 2000).
- **NANOG** (17 points): NUDGE **abstains**. Its knockdown reaches only ~75%, its fitted K sits
  *past* the maximum dose, and an independent n-profile shows **R² flat within 0.075 across
  n = 1…12** (a graded n≈1 and a high-threshold switch fit equally well). The gain is genuinely
  unidentifiable → `unresolved` (NUDGE-LIM-007).

**The classifier caught a classic human over-reading.** An exploratory bounded Hill fit (K ≤ 1.0)
reported NANOG as a graded `n ≈ 2.2` — but that fit *railed K against its bound* and had a
bootstrap n-CI of [1.2, 12]: an under-determined curve whose own uncertainty screamed
non-identifiability. The old script had no classifier and just printed the point estimate; a
human labeled it "graded." NUDGE's classifier reads the same fit and correctly returns
`unresolved`. Not a fail-safe break (no code ever "called graded"), and not a module bug — the
classifier *prevents* the overclaim. Two independent gates abstain (inflection-not-spanned AND
CI-straddle), so it is not a threshold artifact.

**Engineering note (verified root-cause, not guessed).** The fit reuses the JAX Hill primitive;
JAX defaults to float32, so `curve_fit`'s finite-difference Jacobian underflowed to *exactly zero*
in the `n` direction (scipy's default step is float64-sized) and `n` froze at its init — a
confident-wrong-`n` hazard. Fixed by handing `curve_fit` the **exact JAX-autodiff Jacobian** of the
primitive (local, no global x64 flag). Locked by
`tests/inference/test_dose_response.py::test_autodiff_jacobian_lets_n_move_off_its_seed` and the
OCT4/NANOG regression. Wired into the `nudge dose-response` CLI verb and the `dose_response` MCP
tool; carded as `NUDGE-METHOD-001`.

# Phase 4c — multi-timepoint capstone (Gladstone Rest/8h/48h): the abstention is robust

The 8h result teed up two follow-ups; we answered them across **all three** stimulation
timepoints as operating points (`scripts/vv/gladstone_multitimepoint.py`; D1_Rest + D1_Stim8hr +
D1_Stim48hr, ~440 GB loaded via the pointer reader).

**Q1 — does a later, more-committed timepoint push the readout into a genuine switch that
survives the BIC parsimony gate?** No — at *every* timepoint.
- **Rest:** `no-switch` (BIC **−86 vs 3240** — the 1-node switch is *decisively* rejected, the
  largest margin of the three; skew 9.4, 38% of cells near baseline — the resting state).
- **8h:** `no-switch` (BIC 40556 vs 40599; skew 12.2).
- **48h:** `no-switch` (BIC 36353 vs 36447; **skew 17.1, kurtosis 684**) — *more* skewed/unimodal,
  not bimodal.
The transcriptional IEG activation is a single heavy-tailed mode across the whole time course; the
switch never emerges. NUDGE's abstention is **consistent — and, at Rest, most decisive**.

**Q2 — do the timepoints as multiple `OperatingPoint`s break the gain/threshold degeneracy?**
Every target × operating point is **SKIPPED**, by *independent* guards (min_cells=100 pass):
- **SOS1** (143/110/118 cells at Rest/8h/48h) & **RASA2** (172/233/192): `LNA unreliable —
  insufficient depth` (scale·peak **0.3** at Rest, 11.3–13.6 at 8h/48h < the 15.0 threshold).
  Rest's depth is lowest of all — the guard declines hardest exactly where signal is weakest, the
  fail-safe UQ layer widening and abstaining as intended.
- **RASGRP1** (26/24/58): too few cells.
- **0 usable operating points → no breaker attempted, at any timepoint.**

**Honest conclusion.** On ~440 GB across all three real timepoints, NUDGE declines at *every* gate
(topology parsimony, LNA depth reliability, min-cells) — and Rest, the least-activated state,
abstains *most* decisively (BIC margin 3326; depth scale·peak 0.3). The multi-timepoint breaker's
premise — resolve gain/threshold *of a detected switch* across operating points — never engages,
because this **genome-wide** screen supplies neither a transcriptional switch nor the depth /
cell-counts identifiability needs (26–233 cells/guide and scale·peak ≈ 0.3–14, vs the ~1000 cells
and scale·peak ≥ 15 the guards require). The measured **×16 degeneracy-break stays a synthetic
ground-truth result (§2)**; on this real dataset the honest verdict is a robust, multi-gate
abstention — we did **not** manufacture a switch or force a call. The Ras switch is bistable at
the *signaling* (Ras-GTP) level (Das 2009); a steady-state transcriptomic snapshot of downstream
IEGs need not resolve two modes, and across Rest/8h/48h it doesn't. **This closes the T-cell
capstone:** the fail-safe holds across the full stimulation time course, not a single snapshot.


# Phase 4d — synergy / epistasis attribution (Norman 2019): agrees on the labeled pairs

Capability 2 (`nudge.inference.epistasis`, `NUDGE-METHOD-003`) reads a two-perturbation
combination A / B / A+B as three operating points against a shared control, reduces each to a
scalar **effect** in **log-fold-change space** (so the additive null `e(A)+e(B)` is **Bliss
independence**), and classifies the **interaction** `e(A+B) − [e(A)+e(B)]` — with a bootstrap CI
over cells — as `additive` / `synergistic` / `buffering`, or abstains (`no-effect` / `unresolved`).
The per-cell score projects each cell onto the **additive axis fixed by the two single arms**
(`nudge.inference.bridge.combo_effect_scores`; the axis comes from the singles only, never the
combo, so a positive interaction is unambiguously super-additive — no circularity, no manual sign
convention).

**Applied to Norman 2019 (GSE133344, CRISPRa in K562, ~111k cells).** Five pairs across the
interaction classes, called with `n_boot=500` (projection over the 2000 most-variable genes). An
**independent literature fact-check** (adversarial, sourced to the paper's PMC full text
[PMC6746554] + secondary sources) graded each call — and the honest score is **2/5 explicitly
confirmed against a per-pair statement in Norman 2019, 2/5 consistent with the paper's clusters but
without an explicit per-pair label, and 1/5 a paralog control the paper never analyses**:

| Pair | interaction (95% CI) | ΔBIC | NUDGE call | Norman 2019 grounding |
|---|---|---|---|---|
| **CBL+CNN1** | **+0.95** [+0.48, +1.42] | 19 | **synergistic** | ✅ explicit — flagship emergent-erythroid synergy (Fig 3), validated in HUDEP2 |
| **DUSP9+ETS2** | **−2.14** [−2.64, −1.60] | 156 | **buffering** | ✅ explicit — "DUSP9 phenotype dominated … antagonized ETS2" (Fig 5) |
| **CBL+UBASH3B** | **+1.09** [+0.75, +1.45] | 44 | **synergistic** | ⚠ erythroid RTK-regulator cluster (Fig 2B); no per-pair GI score |
| **CNN1+UBASH3B** | **+1.25** [+0.94, +1.58] | 67 | **synergistic** | ⚠ shared erythroid assoc. only (CNN1 not in the RTK group) — weakest grounding |
| **FOXA1+FOXA3** | **−0.61** [−1.37, +0.25] | −2 | **additive** | ❗ not in Norman; paralog control — additive expected from paralogy + CellCap (2024), not Norman |

The two **explicit** matches are the real result: **DUSP9+ETS2** is the sharpest — the observed
combo (+4.31) lands **at DUSP9-alone** (+4.79), far below the additive prediction (+6.45), and DUSP9
(a MAP-kinase phosphatase, ⊣ ERK→ETS2) suppressing ETS2 is textbook epistasis; **CBL+CNN1** is the
paper's flagship *unexpected* erythroid synergy. **FOXA1+FOXA3** is a paralog negative control NUDGE
correctly declines to over-call (CI straddles 0). Locked in by
`tests/inference/test_epistasis.py::test_norman_synergy_lockin_real_data`; demo in
`notebooks/Norman_Synergy.ipynb`.

**Two caveats the fact-check surfaced (do not drop these).** (1) **Sign-convention collision:**
Norman inherits the *fitness*-GI convention where "buffering = *positive* GI = antagonism"; NUDGE
uses "buffering" for a *negative* interaction coefficient (sub-additive). The two agree
*conceptually* (buffering = antagonism / sub-additive) but carry **opposite numeric signs** — a
reader must not think NUDGE inverted the paper. (2) **Null comparability:** NUDGE's Bliss
(log-additive) *scalar* null is a **coarse approximation** of Norman's fitness-map + full-
transcriptome-regression GI magnitude — agreement is at the level of interaction **type/direction**,
**not** a reproduction of Norman's GI scores, and the scalar-along-the-additive-axis structurally
**cannot see purely off-axis emergent states** (it can only under-count such synergy). The honest
claim is *"agrees with Norman 2019 on the two explicitly-labeled pairs and is consistent with the
paper's clusters on the rest,"* never *"recovers the published taxonomy."*

**Forensic deepening (`design/NORMAN_DISCREPANCY_ANALYSIS.md`).** A dedicated audit of the three
non-explicit pairs (mandate: diagnose, don't p-hack; no margin touched) found **no bug and all
three calls defensible** — and, notably, **FOXA1+FOXA3 is a real 216-cell measured combo**, which
*refutes* the fact-check's doubt that the pair even exists. The audit also makes the Bliss-vs-Norman
relationship **precise and quantified**: NUDGE's scalar interaction is *exactly the on-axis
projection* of Norman's full-transcriptome regression residual (`δab = c1·δa + c2·δb + ε`), and the
paper's **neomorphic** dimension is the *off-axis* component NUDGE discards. Measured in NUDGE's own
coordinates, the off-axis residual is **≥ the on-axis interaction for every synergy pair** (2.1–2.5
vs +1.1–1.3), and the two pairs that *explicitly* matched are exactly the **on-axis-dominated** ones
(DUSP9+ETS2 is a clean on-axis masking — the sharpest match). So NUDGE's equal-weight (`c1=c2=1`),
direction-safe, abstaining Bliss null is a **stricter, principled projection** of Norman's richer
free-coefficient regression: where both apply it agrees on type/direction, and it can only ever
**under-count** emergent synergy, **never invert** a call. The one structural blind spot — the
off-axis/neomorphic residual, and collapsing Norman's *epistasis* (asymmetric masking) vs
*suppression* into one "buffering" — is a **documented limitation (NUDGE-LIM-009), now measured in
the paper's terms**, not a defect.

**Off-axis diagnostic — now shipped (`NUDGE-METHOD-003`).** The audit's flagged enhancement is
built: `combo_effect_scores(..., return_geometry=True)` returns a `ComboGeometry`, so every fit now
carries `EpistasisFit.off_axis_residual` and `neomorphic_ratio = off_axis / max(|on_axis|, ε)`, and
a `synergistic`/`buffering` call with `neomorphic_ratio ≥ 1.0` (off-axis ≥ on-axis) gains an honest
*possible-neomorphic UNDER-count* warning in its `reason`. It is **additive and opt-in** — the pure
scalar-array fit, all five calls, and every fail-safe margin (`bic_margin`, `min_cells`, `rel_width`)
are unchanged; it is a flag, never a discovery or a hidden-node claim. Measured per pair: the three
synergy pairs are flagged (off-axis / on-axis / ratio: CBL+CNN1 2.54/0.95/**2.67**, CBL+UBASH3B
2.15/1.09/**1.98**, CNN1+UBASH3B 2.21/1.25/**1.76**), while the sharp DUSP9+ETS2 buffering — a clean
on-axis masking — is correctly *not* flagged (1.33/−2.14/**0.62**), and FOXA1+FOXA3 is `additive`
so is never flagged. This turns LIM-009 from prose into a number shown with every call (mechanism
card + `notebooks/Norman_Synergy.ipynb`, which now plots on-axis vs off-axis per pair).

**Honest bounds (NUDGE-LIM-009).** A combo inherits its weakest single arm (abstain when an arm is
underpowered); the additive null is effect-space-dependent (log-FC / Bliss, reported with every
call); and the interaction is a **scalar along the additive axis** — a purely orthogonal emergent
state is not captured by it, and a super-additive residual is **not** by itself a hidden-node claim.
CRISPRa combinations are genetic (on/off), so this illustrates combination *logic*, not a graded
drug-dose combination.

# Phase 4e — cross-modality readout (Chure 2019 LacI): NUDGE recovers the domain answer key

**Capability 1 (`NUDGE-METHOD-002`) validated on an author-labelled K-vs-ceiling ground truth.**
NUDGE's ingest hard-required raw integer counts; the cross-modality adapter runs the *same*
threshold/gain/ceiling attribution on a **continuous single channel** (fluorescence / activity /
fold-change) behind a modality-aware bouncer (`nudge.data.ingest.check_readout`, `NUDGE-LIM-008`)
that refuses log-normalized or raw counts masquerading as fluorescence, then feeds the fold-change
curve into the shipped dose-response fit (`NUDGE-METHOD-001`). The crown-jewel test is **Chure 2019**
(CaltechDATA D1.1241, `Chure2019_summarized_data.csv`), where the authors decompose LacI mutants by
*which* biophysical parameter each changes: **DNA-binding-domain** mutants alter only the DNA binding
energy `Δε_RA` (→ repression setpoint / leakiness = NUDGE's **ceiling**); **inducer-binding-domain**
mutants alter only `Ka`/`Ki`/`Δε_AI` (→ the inducer response = NUDGE's **threshold K** on the IPTG
axis). Two biophysical facts anchor the mapping (both independently confirmed against the MWC /
Chure / Razo-Mejia equations): **`K` is the induction EC50 — the half-max *inducer concentration*,
a function of `Ka`/`Ki`/`Δε_AI`, NOT the raw `Ka`** — so a `K` shift reports a changed inducer
response; and **the leakiness FLOOR (fold-change at zero inducer) is the clean pure-`Δε_RA` readout**
— it depends only on the DNA binding energy, so a raised floor is cleanly DNA-domain-attributable.
(Saturation / dynamic-RANGE depends on *both* `Ka`/`Ki` and `Δε_RA`, so a `ceiling` call is only
cleanly DNA-attributable when it is **driven by the floor** — see note (2).)

Fitting each single mutant's fold-change-vs-IPTG curve at the matched condition (operator O2, repressor
copy number 260) vs WT (`K≈71 µM, n≈1.4`), and localizing each to one knob:

| Mutant | Author domain | NUDGE knob | Evidence vs WT |
|---|---|---|---|
| **Q294K** | inducer-binding | **threshold** | K 71→626 µM (+3.1 oct), disjoint CIs; floor near WT |
| **Q294V** | inducer-binding | **threshold** | K 71→420 µM (+2.6 oct), disjoint CIs |
| **Y20I** | DNA-binding | **ceiling** | leakiness floor +0.46, span shrinks (K drifts *left*) |
| **Q21A** | DNA-binding | **ceiling** | leakiness floor +0.32, span shrinks |
| **Q294R** | inducer-binding | **non-responsive (abstain)** | span collapses (amp≈0.02) — near-non-inducible (`Ka≈Ki`) |
| **F164T** | inducer-binding | **inconclusive (abstain)** | mildest inducer mutant (Ka 139→201); no knob clears its gate |
| **Q21M** | DNA-binding | **inconclusive (abstain)** | *stronger*-binding DNA mutant (`ep_RA`≈−15.4) — no leakiness; mild rightward K |

**The honest score: 4/7 localized to the biophysically-correct knob, 3/7 honest abstentions, 0/7
mis-attributed, and no mutant reads gain(n).** The two dramatic inducer-weakening mutants (Q294K/V)
land on **threshold**, the two leaky DNA mutants (Y20I/Q21A) on **ceiling**, the non-inducible Q294R
abstains — recovering the authors' inducer-vs-DNA-domain decomposition wherever a single operating
point is identifiable. The **sign** of the EC50 shift is what separates the two classes: a *rightward*
shift is a weakened inducer response (threshold), whereas a raised leakiness floor drags the apparent
EC50 *left* (ceiling) — NUDGE's knob gate reads these apart rather than collapsing both to "K moved".

**Independently confirmed.** A biophysics literature check against the MWC / Chure / Razo-Mejia
equations **confirmed the mapping** (inducer→threshold/EC50, DNA→leakiness/floor, gain-abstain) and
confirmed NUDGE was **right to override the naive "DNA→K, inducer→n" prior** — that prior is
biophysically refuted (it inverts the domain roles). The Y20I floor +0.46 was reproduced exactly from
first principles, and the Q21M / Q294R abstentions were judged genuinely-correct hard cases.

**Three honesty notes (do not drop).** (1) **The naive "DNA→K, inducer→n" prior is biophysically
wrong, and NUDGE overrode it** — the inducer domain sets the induction EC50 (**K**, the half-max
inducer concentration) and the DNA domain sets leakiness (**floor / ceiling**). On the gain axis:
the *structural* cooperativity exponent (the exponent 2 = LacI's two inducer sites) is fixed by the
protein architecture, but the *effective* Hill coefficient depends *weakly* on the `Ka`/`Ki` ratio
(Razo-Mejia Eq. 10) — so `n` is **not** mathematically invariant. The honest framing is: the mutants'
dominant, cleanly-attributable effect is on the EC50/threshold; any effective-steepness change is
**second-order**, so **abstaining on the gain axis is the correct call** (not a claim that `n` is
fixed). (2) The knob call is **comparative** (vs WT) at a **single operating point (O2, R=260)**. The
clean guarantees it *can* deliver from one context are analytic and context-independent — the EC50 is
independent of `Δε_RA`, and the leakiness floor is independent of `Ka`/`Ki` — so **one context
suffices to separate a threshold shift from a floor shift**. What one context **cannot** do is
decompose a dynamic-RANGE change into its `Ka`/`Ki` vs `Δε_RA` parts; F164T (mildest inducer mutant)
and Q21M (a stronger-binding DNA mutant with no floor rise, only a mild EC50 drift) are honestly
*inconclusive* there — a copy-number series (a second operating point) would resolve them. (3)
**Residue numbering:** the CSV / mwc_mutants repo use the LacI convention *including* the N-terminal
Met (`Y20I`, `Q21A/M`, `F164T`, `Q294K/V/R`); the PNAS paper text uses the −3 convention (`Y17I`,
`Q18A/M`, `F161T`, `Q291K/V/R`) — the same mutants, a +3 numbering offset. Inherits the affine-readout
bound (`NUDGE-LIM-006`). Locked in by
`tests/inference/test_cross_modality.py::test_chure_laci_kn_ground_truth_real_data`; demo in
`notebooks/Chure_LacI_Benchmark.ipynb`. The per-mutant `Ka`/`Ki` (inducer) and `Δε_RA` (DNA) shifts
are in the repo's `Chure2019_KaKi_epAI_summary.csv` / `Chure2019_DNA_binding_energy_summary.csv`.

# Phase 4f — bifurcation / tipping-point proximity (the "robustness dial"): a one-sided lower bound

**What.** A new capability (`nudge.inference.bifurcation`, `NUDGE-METHOD-006`) answers *how close is
a bistable switch to LOSING bistability* — a saddle-node fold — as a scalar **0..1 robustness dial**
from three complementary channels, each with a known analytic limit at the fold:
**critical slowing** (`min|Re λ|` of the drift Jacobian at each stable mode → 0), **basin collapse**
(stable-node → index-1-saddle distance → 0), and **LNA lobe swell** (`√λmax(Σ) / min‖μᵢ−μⱼ‖` → 1). It
re-exposes a signal that was *already computed but buried* — the fixed-point eigenvalues that
`Circuit.fixed_points` labelled-then-dropped, and the lobe ratio used only as an abstention trigger
inside `lna_reliable`. The fused dial is `max(½·(p_slow + p_basin), p_lobe)`: the two deterministic,
depth-independent channels averaged, `max`'d with the LNA overlap so the noise channel can only
*raise* the alarm (fail-safe).

**The honesty crux (the capability lives or dies here; `NUDGE-LIM-012`).** The linear-noise Gaussian
that the third channel uses **breaks down PRECISELY at the fold** — a mode's variance diverges as its
Jacobian eigenvalue → 0 — so the estimate is *least* reliable exactly where it matters most.
Therefore the dial is reported as a **one-sided LOWER BOUND** near the fold (`one_sided` sets once the
lobes overlap, `lobe_ratio ≥ 1`), never a point estimate; and `classify_robustness` **ABSTAINS**
(`unresolved`) on the deep-basin far side — where the slowest relaxation rate has saturated at the
intrinsic decay rate and the noise lobes carry no fold information — rather than emit a false-precise
"far" number. `< 2 stable modes → not-bistable` (score `None`).

**Ground-truth result (the load-bearing validation — we control the fold).** The self-activation
switch (`ras_switch_1node`) has a KNOWN analytic saddle-node in its cooperativity `n` (and in its
threshold `K`). Sweeping `n` toward the fold, the measured channels (float64 Jacobian/Lyapunov):

| n | min\|Reλ\| (→0) | node→saddle (→0) | lobe ratio (→1) | dial | one_sided | call |
|---|---|---|---|---|---|---|
| 10 | 0.993 | 0.938 | 0.719 | 0.035 | False | **unresolved** (deep basin) |
| 6 | 0.915 | 0.925 | 0.754 | 0.073 | False | **robust** |
| 4 | 0.728 | 0.902 | 0.871 | 0.151 | False | **robust** |
| 3 | 0.524 | 0.812 | 1.085 | 0.253 | True | **robust** |
| 2.5 | 0.392 | 0.696 | 1.328 | 0.339 | True | **robust** |
| 2.2 | 0.324 | 0.643 | 1.536 | 0.536 | True | **robust** |
| 2.0 | 0.300 | 0.615 | 1.659 | 0.659 | True | **near-fold** |
| 1.5 | — monostable — | | | | | **not-bistable** |

On the clean ladder (n = 6 → 2.2) all three channels move **monotonically** toward their fold limits
and the fused dial **ranks proximity correctly** (0.073 → 0.536), with `one_sided` setting as the
lobes overlap. The K-sweep behaves the same (K = 1.0 → 1.3 ranks the dial; K ≥ 1.33 goes monostable).
At the very fold edge (n ≤ 2.0) the N-D Newton finder gets numerically noisy as an eigenvalue → 0, so
the monotonicity assertion is made on the clean rungs just short of it. Generalises to N-species
switches (toggle/2-node score without special-casing) — it is meant to be the future `design()`
**safety gate**. Locked in by `tests/inference/test_bifurcation.py` (near-fold → one-sided lower
bound; deep-basin → abstain; well-buffered → robust; monostable → not-bistable; the monotonic
parameter-sweep ground truth; the populated raw channels) + `tests/test_service.py`
(`test_robustness_circuit_wiring`, `test_bifurcation_file_npy_wiring`); demo in
`notebooks/Robustness_Dial.ipynb`. **A real-data dose-ladder lock-in is a deferred `needs_data`
follow-up** (toggle+hysteresis Zenodo 11817798 / morphogen top rung GSE233574); the synthetic
parameter sweep is the load-bearing validation.

# Phase 4g — inverse / intervention design (`design()`): from diagnosis to prescription

**What.** The flagship (`nudge.design.invert`, `NUDGE-METHOD-007`) delivers the brief's headline
thesis — NUDGE *inverts the fit to propose untested interventions*. Given a **reliable** attribution
it runs the same differentiable fit **backwards** to prescribe an intervention (a kinetic Δ, or a
dose), behind two honesty gates. **Circuit mode:** gradient inversion over a fitted `Circuit` (the
`fit_parameters` loop backwards — Adam over an additive log-Δ on addressable knobs, minimizing
`‖PredictedState − target‖² + l1‖Δ‖₁`), then the Cap-5 `bifurcation_proximity` **safety gate** on the
intervened circuit. **Curve mode:** closed-form inversion of a `DoseResponseFit` to the dose achieving
a target response `y`.

**Measured (synthetic ground truth, `tests/design/test_invert.py`).**
- **Known-intervention recovery — loss ≈ 0.** A monostable switch perturbed by a known `×2` on `v_max`
  is recovered to `factor ≈ 2.0` with residual gap `< 1e-3`. Because the true Δ is known, this is a
  clean recovery, not a vibe check.
- **Safety gate partitions safe vs unsafe.** A flip-ON intervention that raises `basal` from the
  resting basin **crosses the fold** (`crosses_fold=True`, `high_risk_of_instability=True`, proximity
  0.073 → None = bistability lost); a modest ON-level nudge from the high basin stays bistable
  (proximity 0.073 → 0.095, **not** high-risk). The near-fold number is a one-sided LOWER bound
  (inherited from Cap 5, `NUDGE-LIM-012`).
- **Both abstention gates fire.** An `unresolved` / `no-effect` attribution → integrity abstention; an
  unreachable target → reachability abstention (no false extrapolation, `NUDGE-LIM-013`).
- **Curve mode round-trips.** `y = floor + amp/2` inverts to `dose ≈ K`; an out-of-`(floor, floor+amp)`
  target abstains. **Curve mode carries NO safety gate** (no circuit/fold), stated in every dose plan.

**Real data (lock-in, `needs_data`).** The OCT4 self-renewal dose-response switch fit (`n≈6.7`,
`R²=0.99`) inverts to a positive knockdown dose (`≈0.61` fraction of POU5F1 silenced) for a reachable
target, and **abstains** below the fully-silenced floor. Demoed in `notebooks/Inverse_Design.ipynb`
(Part A synthetic flip-ON + safety dial; Part B real OCT4 inversion + reachability abstention).

**Honesty subtlety (stated, not hidden).** Gradient inversion sees only the basin it starts in — a
knob whose effect on the starting fixed point is weak (e.g. `K` alone, from the low basin) can leave
the optimizer stalled; that surfaces as a reachability abstention, never a forced call. Every proposal
is a **model-bound hypothesis to test**, valid only within the fit's identifiable region
(`NUDGE-LIM-013`) — never a guaranteed outcome. Wired into `nudge design` CLI + the `design` MCP tool
+ `nudge.service.design_circuit` / `design_file`; Mechanism Card `NUDGE-METHOD-007` (`inverse_design`).
# Laplace posterior — curvature CIs + the gain/threshold degeneracy reproduced

An **additive, opt-in, guarded** uncertainty layer (`nudge.inference.uncertainty`) turns the
fit's point estimate `θ*` (the log-space kinetics the fit recovers) into **curvature-based
error bars**: at the optimum the loss Hessian `H = ∇²L(θ*)` is the precision of a *local*
Gaussian posterior `θ ~ N(θ*, H⁻¹)` (Laplace's approximation). The Hessian target is the
**deterministic** Lyapunov Gaussian-mixture NLL (`lyapunov_nll_loss`), **not** the stochastic
energy distance — whose minibatch-noisy Hessian is not a likelihood curvature — so `H` is the
observed Fisher information and `H⁻¹/N` the covariance of the recovered kinetics. It touches
neither `fit()`'s default output contract nor the decoy battery.

**Curvature CIs cover the truth.** Fitting the identifiable knob (ceiling / `vmax`) on
inverse-crime toggle data across 20 seeds (N=1500) and building the log-space→lognormal
marginal CI at each `θ*` (the delta method for a log transform, done exactly), the interval
covers the true `vmax` **20/20** — ≥ the nominal 95%, and conservative, which is the fail-safe
direction (a wider honest interval is fine; a too-narrow one is not).

**The gain⇄threshold degeneracy reproduces as a near-singular Hessian.** On the 2-node toggle
over `(n, vmax, K)` at one operating point, the Laplace covariance has **condition number ≈ 211**
(N≈4000) — matching the FIM's ≈ 210 (§"N-D saddle") — and **corr(n, K) ≈ +0.99**. The condition
number is a *finite-sample* quantity: the flat direction is so barely curved that its empirical
curvature is noisy (≈ 150–250+ at N≈2000, occasionally singular), but it robustly far exceeds the
guard, which is the load-bearing point. Note the **sign**: the
covariance correlation is **+0.99** while the *Fisher* correlation is **−0.99** — inverting a 2×2
with a negative off-diagonal flips the sign; it is the *same* degeneracy, seen through `H⁻¹` rather
than `H`. The guard sets `degenerate=True` and marks **gain (`n`) and threshold (`K`)
unidentifiable / CI unbounded** (ceiling stays identifiable), so `mechanism_confidence` **abstains
to confidence 0.0** rather than report a false-precise interval.

**A second operating point breaks it (the ×16 mirror).** Summing the NLL over a second basal-B
operating point collapses the condition number **≈ 211 → ≈ 27** (mirroring the FIM's cond 210→22 /
smallest-eigenvalue ×16), `degenerate` flips to `False`, and every knob becomes identifiable
(confidence ≈ 0.98). Same result the covariance-attribution M3 breaker reports — now visible
directly in the posterior geometry.

**Fail-safe engineering (the load-bearing honesty point).** The inverse is a **guarded
ridge-regularized eigen-inverse**, never a plain pseudo-inverse — a plain `pinv` would *zero* a
flat direction's variance (false precision), the *opposite* of safe; the relative ridge instead
widens it to a **large-but-finite, PSD** variance (no NaN). A non-positive-definite Hessian (`θ*`
not a minimum) → cond = ∞ → abstain. The Laplace Gaussian is *local* and worst exactly at
degeneracies / near bifurcations, so the layer is engineered to **widen and abstain** there rather
than trust a bad Gaussian. Tests: `tests/inference/test_uncertainty.py` (analytic-Hessian CI;
singular / partial / non-PSD guards; the degeneracy reproduction + two-operating-point break; the
coverage calibration).

# Phase 4h — multi-reporter joint attribution: the K⇄v_max degeneracy, broken by a panel

Capability 6 (`nudge.inference.multi_reporter`, `NUDGE-METHOD-008`) attacks NUDGE's
*dominant* reason to abstain head-on: the measured **K⇄v_max / gain⇄threshold degeneracy**
(§2). The insight §2 already named as the fix — *a richer (multi-reporter) readout* — is
built: fit **several downstream reporters of ONE latent switch jointly**. Each reporter is
an affine readout `y_j = base_j + gain_j·A·f(dose; K, n)` of the *same* latent (genuinely a
`Readout` of a shared Hill activity). Pinning the reporter gains from the CONTROL panel and
sharing one latent makes the perturbed fit **over-determined**: a **threshold** shift moves
the inflection **identically** across reporters, while a **ceiling** change scales **every**
reporter's ON amplitude by the *same fraction* — so the two project DIFFERENTLY onto a panel
of heterogeneous gains, and the joint fit separates them. This is the multi-*reporter*
analogue of the second-operating-*point* ×16 break (the "Covariance attribution" M3 result):
more observers of one latent, instead of more conditions.

## The headline: JOINT resolves where a SINGLE reporter abstains (0 confident-wrong)

Synthetic ground truth — ONE latent switch carrying a KNOWN **threshold-only** /
**gain-only** / **ceiling-only** perturbation (`factor=3`), observed through **4 reporters
of heterogeneous gain** — run two ways with the *same* `attribute_multi_reporter`: the full
panel vs a single reporter.

| condition | JOINT (4 reporters) | SINGLE (1 reporter) |
|---|---|---|
| threshold (K×3) | **threshold** ✓ (8/8 seeds) | unresolved (0/8) |
| gain (n÷3) | **gain** ✓ (8/8 seeds) | unresolved (0/8) |
| ceiling (A÷3) | **ceiling** ✓ (8/8 seeds) | unresolved (0/8) |

**JOINT mechanism recovery: 24/24 (100%). SINGLE-reporter: 0/24. Confident-wrong calls: 0**
(on either path). The single reporter cannot separate a latent ceiling change from its own
gain drifting (the affine cannot be pinned or checked with one reporter), so it honestly
returns `unresolved` — the very abstention the panel is built to break. The joint winner
beats the runner-up knob by a large loss margin (threshold ×140–177, gain ×23–29, ceiling
×265–323) at panel R² ≈ 1.00. A wider sweep (mechanism × factor∈{2.5,3,4} × noise∈{.03,.06}
× seeds) holds the 0-wrong property (`test_fail_safe_never_a_confident_wrong_call`).

## The consistency guard — an inconsistent panel abstains off-model (`NUDGE-LIM-014`)

The identifiability gain rests entirely on the panel genuinely reporting ONE latent. When a
reporter secretly reads a **different** latent (a shifted `K` — a hidden co-regulated node
or a mislabeled panel), no shared `(K, n)` + per-reporter affine can fit all curves. NUDGE
must **not** average this into a call: the consistency guard abstains **off-model** — the
odd reporter fits its OWN Hill cleanly (R² ≈ 1.0) yet the shared latent explains it badly
(R² ≈ 0.5), and the shared-vs-independent residual ratio blows up (66–268×). Verified across
which reporter is the odd one out and across the ground-truth mechanism
(`test_inconsistent_panel_is_off_model`). Reporter inconsistency is **flagged, not silently
averaged away** — the honest new gate this capability adds.

> Takeaway: NUDGE's dominant abstention (threshold-vs-ceiling from one reporter) becomes a
> **resolved** call with a multi-reporter panel — a concrete experimental recipe — and the
> fail-safe is *extended*, not traded: a panel that isn't one latent switch abstains
> off-model. **JOINT 100% / SINGLE 0% recovery, 0 confident-wrong**, measured on synthetic
> ground truth. Wired into `nudge multi-reporter` CLI + the `multi_reporter` MCP tool +
> `nudge.service.multi_reporter_file`; Mechanism Card `NUDGE-METHOD-008` (`multi_reporter`);
> `notebooks/Multi_Reporter.ipynb`. A real-panel demonstration (e.g. OCT4/NANOG self-renewal
> reporters of the pluripotency latent) is a deferred follow-up — the synthetic
> degeneracy-break is the load-bearing validation (we do NOT force a real-data claim).
# Abstention catalogue + the toggle-gain deep dive: gain is a FUNDAMENTAL covariance-channel limit

A read-only analysis (`design/ABSTENTION_ANALYSIS.md`) maps every surface on which NUDGE declines a
mechanism call and classifies each **fundamental** (keep abstaining) vs **addressable** (a concrete
lever resolves it). Most addressable ones reduce to *one clean second operating point / a constitutive
control*; the fundamental ones to *"the data isn't a switch"* and *"the LNA dies at the fold — where
gain lives on a toggle."* Two numerical probes back the sharpest case (`scripts/analysis/`).

**The toggle-gain abstention is fundamental, and survives a 3rd operating point (measured).** The open
question was whether `attribute_lyapunov_multi`'s gain abstention (`unresolved` 3/3 on independent SSA,
while threshold + ceiling resolve) is fixable. Probe B (`toggle_gain_abstention_probe.py`, resolved-
channel NLL gap vs resolve_margin 0.03):

| true mechanism | pts=1 | pts=2 | pts=3 |
|---|---|---|---|
| **gain** (n×0.6) | `unresolved` gap 0.019 | `unresolved` gap **0.0009** | `unresolved` gap **0.0001** |
| threshold (K×1.6) | `ceiling` 0.036 | **`threshold`** 0.344 | **`threshold`** 0.094 |
| ceiling (vmax×0.6) | `unresolved` 0.001 | **`ceiling`** 0.202 | `threshold` 0.304 ⚠ |

Adding operating points **shrinks** the gain gap toward zero (0.019→0.0001) — the *opposite* of
threshold/ceiling — so more conditions is provably **not** the lever for gain. Mechanistic root
(`toggle_gain_mechanism.py`, deterministic): a mild gain change barely moves the LNA mode means/covs
while the LNA is trustworthy (rel. Δcov ≤ 0.17 for n_eff ≥ 3.0), and its covariance signature only
becomes large (Δcov ≈ 1.0) as `n` reaches the saddle-node — exactly where `lna_reliable` **abstains**,
just before bistability is lost. Gain (Hill `n`) shapes the *transition region / relaxation timescale*
(a large-deviation property), not the deep-basin stationary covariance the channel reads; deep in a
toggle basin both species are saturated against the repression threshold so the Jacobian — hence the
covariance — is ~independent of `n`. **Multi-reporter (Cap 6) cannot fix it either** (the blindness is
in the latent dynamics, not the readout projection). The only observable that could carry it is
**time-resolved data** (Cap 4) or a near-fold non-Gaussian likelihood — a method change, unmeasured.
Verdict: **keep abstaining on toggle-gain**; it is a genuine identifiability wall, consistent with the
FIM prediction and the `TOGGLE_ATTRIBUTION_RESEARCH.md` LNA-breaks-at-the-fold caveat.

**Honest side-finding — a near-fold operating point can HURT.** At `pts=3` the probe's aggressive 3rd
point (basal 0.60, which clears `lna_reliable` only *at the edge*: lobe std ≈ 1.94 vs separation ≈
2.24) corrupted the shared-parameter joint fit and flipped a true **ceiling → confident `threshold`**
(gap 0.30). So "add a second operating point" (the validated breaker is basal 0.05 + 0.30, 0 wrong)
means *a clean, well-buffered* point — not "pile on as many as possible." A cheap safeguard: weight
each operating point by its `lna_reliable` margin, or require a minimum buffer, so a marginal near-fold
point cannot dominate. (This does not affect the gain verdict — gain abstains at 1/2/3 points.) The
shipped production toggle path still abstains (`tests/verification/test_toggle_nd_safety.py`); this is
a probe-surfaced caveat on the opt-in multi-point breaker, not a shipped fail-safe break.

# Phase 4i — hidden-node ABSTENTION: a legible differential, never a positive claim

**What shipped.** `inference/hidden_node.py` (`NUDGE-METHOD-009`, `NUDGE-LIM-015`) — the
**abstention half ONLY** of the hidden-node problem. When NUDGE's switch model is inadequate
(the parsimony gate returns `off-model`, or the off-axis / neomorphic residual fires), it
packages the evidence into a rank-ordered **differential diagnosis** of six candidate causes
— genuinely not-a-switch (`NUDGE-LIM-005`/`DECOY-005`), a nonlinear readout (`NUDGE-LIM-006`),
an off-target perturbation (`NUDGE-LIM-004`/`DECOY-004`), a wrong/misspecified topology
(T0.5-2), a batch/depth confound (`NUDGE-LIM-003`/`DECOY-003`), and a hidden node / unmeasured
regulator (the off-axis residual, `NUDGE-LIM-009`) — each with its evidence, documented
limitation, and the experiment that would distinguish it.

**Why abstention-only (the measured design decision).** Positive hidden-node identification is
**not identifiable** from an off-model verdict: the six causes are *observationally
overlapping* (they all present as "the affine switch model does not fit"), and there is
essentially no real Perturb-seq data with a *known* hidden node to calibrate a detector
against. This is the same wall the abstention analysis (`design/ABSTENTION_ANALYSIS.md`, rows
#1/#3/#18) and `NUDGE-LIM-009` already documented. So NUDGE ships **only** the differential and
**never** asserts a hidden node.

**The honesty guarantee (enforced in CI).** `tests/inference/test_hidden_node.py` includes the
load-bearing test: even in the most tempting regime (off-axis ratio 12), the report emits **no
bare positive hidden-node claim** — every emitted string is scanned against a forbidden-phrase
battery — and the hidden-node cause is explicitly hedged (*consistent with … does NOT prove*),
with its rank **capped** so it is never the lone leading answer (on an `off-model` verdict the
"genuinely not-a-switch" reading — the gate working — leads). An **adequate** model yields
`is_adequate=True` with **no** causes (it does not manufacture a differential either).

**Wiring.** `nudge diagnose-abstention` CLI + the `diagnose_abstention` MCP tool +
`service.diagnose_abstention` (which enriches each cause with its limitation title via the
read-only `knowledge.explain` backbone) + a Mechanism Card + `notebooks/Hidden_Node_Abstention.ipynb`.
Pure packaging/knowledge layer with **zero import of `fit`** — it consumes verdicts, never
re-attributes, and never touches the decoy battery. Additive/opt-in.

# Phase 4j — comparative / differential attribution: WHICH knob differs between two contexts

**Capability (`nudge.inference.differential`, `NUDGE-METHOD-010`, `NUDGE-LIM-016`).** The same
perturbation, run in **two contexts** (a drug-resistant vs sensitive line; donor A vs B; disease
vs healthy), can differ mechanistically in the switch's **threshold** (`K`), **gain** (`n`), or
**ceiling** (`v_max`) — a distinction linear differential expression structurally cannot make (a
raised ceiling → *more of the same drug*; a rewired gain/threshold → *a different class*). NUDGE
fits the two contexts **jointly** with a **shared-vs-per-context** parameter structure and
**BIC-selects which single knob must differ** — enumerating `{shared, ΔK, Δn, Δv_max}` and picking
the min-BIC model (a Δ model must *earn* its extra per-context parameter over the shared null,
`BIC = k·ln N − 2·log L`). It **reuses the shipped LNA machinery verbatim**: the differentiable
Gaussian-mixture forward model (mode means + Lyapunov covariances) from `nudge.inference.lyapunov`,
per-context depth pinning via `calibrate_from_wt`, and the BIC parsimony pattern from
`model_select`. Additive / opt-in — it never touches `fit()` or the decoy battery.

**The confound guard (`NUDGE-LIM-016`, the load-bearing honesty point).** A sequencing-depth /
batch difference *aligned with the context axis* mimics a **ceiling** difference, because depth
(global `scale`) and `v_max` both multiply the mode means (the measured `scale⇄vmax` degeneracy).
NUDGE (1) pins depth/noise **per context from each context's OWN control** (a depth difference
captured by the controls is calibrated out — the per-sample library-size analogue; this requires
each context's control to be from the SAME library as its perturbed cells), and (2) **abstains
(`unresolved`) whenever the two contexts' pinned depths differ beyond a ratio** — a depth/batch
difference aligned with the context axis — because it cannot certify that an apparent ceiling /
no-clear difference is not a masked depth artifact. The **one exception** is a *cleanly-resolved
threshold or gain* difference, which reshapes the distribution (orthogonal to a global scale) and
survives a depth difference (measured: a Δn pair with the contexts sequenced 1.6× apart still
recovers `gain-diff`). An **OFF-baseline diagnostic** (the differential low-activity-quantile shift
in each context's data vs its own control) was prototyped and **measured too noisy to be
load-bearing** — on the 1-node switch the OFF mode's linear-noise CV is ≈ 1/√basal ≈ 2, so the
genuine-ceiling OFF shift (≈ 1.0–1.56 across seeds) *overlaps* the on-samples-batch shift
(≈ 1.46–2.25); it is reported for transparency only, and the guard turns on the **stable
per-context depth ratio** instead.

**Results (synthetic ground truth; `tests/inference/test_differential.py`, 7 slow tests green in
~234 s).** A KNOWN single-knob difference between two contexts, drawn from the LNA Gaussian mixture:

- **Δv_max → `ceiling-diff`** (a raised ceiling, factor ×1.4): recovered, `is_reliable`. The
  drug-resistance headline — *more of the same drug*.
- **Δn → `gain-diff`** (a rewired gain, n ÷ 1.8): recovered — *a different class*, not just more dose.
- **no-difference → `no-difference`**: the same mechanism in both contexts is correctly read as no
  attributable difference.
- **ΔK → recover-or-abstain**: threshold is the **hardest** to resolve from a bistable snapshot —
  the self-activation switch's *stable modes barely move with K* (K mainly moves the unstable
  saddle, not the modes), so a ΔK pair recovers `threshold-diff` when the signal is strong enough or
  else abstains (`no-difference` / `unresolved`), **never** a wrong mechanism. This is exactly the
  measured identifiability hierarchy **gain > ceiling > threshold** (§2).
- **Confounded pair → `unresolved`** (context B — control AND perturbed — sequenced ~1.6× deeper,
  a depth/batch difference aligned with the context axis, NO real mechanism difference): the
  depth-ratio guard fires across seeds 1–3 (depth ratio ≈ 1.5–1.6) — NUDGE abstains, **never** a
  spurious `ceiling-diff`.
- **Underpowered / untrustworthy context → `unresolved`** (too few cells, or `lna_reliable` False).
- **0 confident-wrong** mechanism-difference calls across a mechanism × seed sweep (the headline
  safety property).

Robustness engineered in: each per-context knob is **bounded** (a smooth `tanh` reparam around
nominal) so a free fit cannot run off to the LNA variance-collapse likelihood spike (n→∞ shrinks the
Lyapunov covariance → an unbounded likelihood), and a Δ model that wanders past the fold
(monostable → frozen roots → non-finite NLL) is scored **+inf** (unfittable, never a winner) so a
NaN can never poison the argmin. The classifier's fast gate logic is locked by 8 no-fit unit tests.

**Wiring.** `nudge differential PATH.npz` CLI verb + the `differential` MCP tool +
`service.differential_arrays` / `differential_file` (a `.npz` of four activity arrays:
`data_a`/`control_a`/`data_b`/`control_b`) + a Mechanism Card (`NUDGE-METHOD-010`) +
`notebooks/Differential.ipynb`. A real-data touch (sci-Plex A549 vs MCF7; Gladstone donors) is a
deferred best-effort follow-up; the synthetic ground truth is the load-bearing validation.
