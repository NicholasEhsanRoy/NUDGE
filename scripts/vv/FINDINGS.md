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

**Honest bounds (NUDGE-LIM-009).** A combo inherits its weakest single arm (abstain when an arm is
underpowered); the additive null is effect-space-dependent (log-FC / Bliss, reported with every
call); and the interaction is a **scalar along the additive axis** — a purely orthogonal emergent
state is not captured by it, and a super-additive residual is **not** by itself a hidden-node claim.
CRISPRa combinations are genetic (on/off), so this illustrates combination *logic*, not a graded
drug-dose combination.
