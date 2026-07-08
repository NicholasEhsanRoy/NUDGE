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
