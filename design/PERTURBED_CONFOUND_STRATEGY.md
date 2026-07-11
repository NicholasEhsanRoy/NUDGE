# Guarding the identifiability CLASS, not the confound INSTANCE

*A prototype + migration design for the whole family of per-condition **affine technical
nuisances** on a differential's perturbed cells. Status: **GUARD B (the nuisance-augmented
earn test) is now SHIPPED** as `differential.classify_differential` **gate 4d** (the P5 fix,
hardening loop; `NUDGE-LIM-016`, FINDINGS §P5) — the earn logic was ported inline from the
`_proto_nuisance` prototype (this doc's measured criterion), closing the whole uniform
per-condition affine class (P1/P4/P5) with 0 confident-wrong and no blind gap. Migration-plan
steps 1–3 (§4) are DONE; the OFF-cluster bands (4b/4c) are kept as a cheap first line + their
locked P1/P4 regressions rather than ripped out. **Guard A (the INERT-FEATURE anchor) remains
a proposal** — it is what buys back a genuine ceiling under a *non-uniform* perturbed-side
scale (the documented residual §3.4), still un-shipped pending an inert-gene ingestion path in
`inference/bridge.py` (§4 step 4).*

Prototype code: `src/nudge/inference/_proto_nuisance.py` (sibling; imports differential's
primitives read-only, touches no shipped code). Measurement: `scripts/eval/proto_nuisance_sweep.py`
(full sweep) + `scripts/eval/proto_nuisance_confirm.py` (corrected-rule confirmation). Baseline
to beat: this worktree's pre-4b/4c `nudge.inference.differential`.

---

## 1. The problem — one class, not three confounds

Every differential confound the red-team has found is the **same thing**: a per-condition
**affine technical nuisance** `y = s·x + o` — a scale `s` and offset `o` — applied to **one
context's perturbed cells only** (its control clean, so the control-keyed depth guard,
`NUDGE-LIM-016` gate 2, never engages). The instances:

| red-team hole | the affine | aliases onto |
|---|---|---|
| P1 additive | `s=1, o>0` | offset shifts modes → threshold / gain |
| P4 multiplicative (`c≥1.5`) | `s>1, o=0` | scale multiplies ON mode → ceiling `v_max` |
| P5 small multiplicative (`c≈1.15–1.25`) | `s∈(1.15,1.25], o=0` | slips gate 4c's `(1.18,1.30]` band + its ceiling-scoping |

The shipped fix is **per-confound hand-calibrated OFF-cluster bands** (gate 4b for additive,
gate 4c for multiplicative, a pending P5 gate). These have **measured blind gaps** — P5 slips
gate 4c's `(1.18,1.30]` interval and its ceiling-only scoping. This is whack-a-mole: the next
magnitude always slips the next band. **The fix must guard the affine CLASS continuously, not
each magnitude.**

The aliasing is real physics, not a bug: a per-condition scale multiplies the mode means, so
it is degenerate with `v_max`; an offset translates the modes, so it aliases threshold / gain.

---

## 2. Two complementary principled mechanisms

### (A) INERT-ANCHOR normalization — the fundamental fix
`estimate_affine_from_inert` + `anchor_normalize`. Estimate `(s, o)` from a perturbation-**inert**
feature block (housekeeping / non-signature genes / spike-ins co-measured on the perturbed
condition — on real Perturb-seq the perturbation moves a handful of the thousands of genes, so
the anchor is nearly free), then undo the affine on the perturbed readout *before* attribution:

```
s = sd(perturbed_inert) / sd(control_inert)          # a scale inflates the spread
o = mean(perturbed_inert) − s · mean(control_inert)  # residual mean shift
x_corrected = (y − o) / s
```

A per-cell technical affine (capture efficiency, ambient, batch) hits *all* of a cell's genes,
so the inert block carries the same `(s, o)` while its biology stays put — the anchor recovers
the nuisance and **buys back identifiability** (a genuine ceiling resolves even under a scale).

### (B) NUISANCE-AUGMENTED IDENTIFIABILITY ABSTENTION — the no-anchor fallback
`guard_b_classify`. Add the per-context affine `(s, o)` as **free nuisances** to the differential
fit and ask a **single, continuous, measured** question: with a free affine on the perturbed
context, does the biological knob still **earn its BIC parameter** over a pure-affine null?

```
model 0 (null):  mean_k = s·(depth·μ_k(θ_shared)) + o,   cov_k = (s·depth)²·Σ_k + obs²·I   # 2 params
model 1 (knob):  same, but the winning knob θ_w is also free                                # 3 params
earn = BIC(model 0) − BIC(model 1)     # >0 ⇔ the knob out-explains the free affine
```

Abstain when `earn < margin` (the difference is absorbable by an affine → not a certified
mechanism). Run in **both directions** (`check_both`) — is the apparent difference an affine on
A *or* on B — and abstain if *either* absorbs it, since we do not know which context is confounded.

**Why this covers the whole class with no bands.** The confound family *is, by construction,
inside the free-affine null's span.* So for any `(s, o)` confound the null fits it exactly and the
bio knob **provably cannot earn its parameter** over it (modulo estimation noise, which the BIC
margin covers). One guard, one continuous statistic, zero calibrated intervals.

---

## 3. Measured results — the honest "does it work" gate

Config: `ras_switch_1node(n=6, vmax=2.5, K=1, basal=0.2)`, scale 25, obs 0.5, 2000 cells (the
differential test-suite config). Guard B uses the **corrected earn-based decision** (see §4).

### 3.1 Affine-family coverage — **PASS (0 confident-wrong)**
Uniform affine `s·x + o` on context B's perturbed cells (control clean, **truth = no-difference**),
swept over a continuum incl. P5's interior `s∈(1.0,1.5)` and the `(1.18,1.30]` gap:

| family | cases | baseline confident-wrong | **guard B confident-wrong** | `earn` range |
|---|---|---|---|---|
| mult `s∈{1.05…1.50}` | 18 | many (ceiling-diff from s≥1.18) | **0** | −7.6 … −6.1 |
| add `o∈{1,2,3,5}` | 8 | 0 (baseline safe here) | **0** | −7.6 … −6.1 |
| mixed `(s,o)` | 4 | ceiling-diff | **0** | −7.6 … −6.1 |
| **full sweep (2 seeds)** | **30** | **13** | **0** | all `earn < 0` |

Six of these re-run **live through the corrected code path** (`proto_nuisance_confirm.py`):
`earn = −7.6` uniformly, every case → `no-difference`. **The entire uniform-affine family
abstains under one guard; the per-confound bands leave 13/30 confident-wrong.**

### 3.2 Positive controls preserved — **PASS**
| control | factor | baseline | **guard B** | `earn` |
|---|---|---|---|---|
| gain | 0.55 | gain-diff | **gain-diff** | +33.4 |
| ceiling | 1.4 (seed 1) | ceiling-diff | **ceiling-diff** | +54.9 |
| ceiling | 1.4 (seed 2) | ceiling-diff | **ceiling-diff** | +83.0 |
| threshold | 1.4 | no-difference | no-difference | −6.4 |
| none | 1.0 | no-difference | no-difference | −6.1 |

Guard B **preserves every positive the baseline resolves** and abstains exactly where the
baseline does (threshold is the hardest knob from a bistable snapshot; both abstain). `earn`
separates confound from mechanism with an enormous margin: **[−7.6, −6.1] vs [+33, +83]**.

> **Honest nuance vs the task's premise.** The task expected guard B to *abstain on a genuine
> ceiling* (anchor needed). Measured: with a **resolved OFF baseline** (`basal=0.2`), a genuine
> ceiling is *separable* under B — a uniform per-condition scale moves the OFF mode, a real
> ceiling does not, so the free affine cannot match both modes and the `v_max` knob earns +55/+83.
> The scale⇄`v_max` degeneracy that forces abstention is real only when **OFF ≈ 0** or the
> nuisance is **non-uniform** (§3.4). This is the same physics the shipped `off_scale` diagnostic
> exploits — re-expressed as a principled nuisance-augmented BIC instead of a calibrated band.

### 3.3 Anchor (A) recovers — **PASS**
Genuine ceiling (`×1.4`) under a technical scale `s_tech = 1.3`, 40 inert genes:

| seed | anchor `s_hat` | `o_hat` | raw (no anchor) | anchored |
|---|---|---|---|---|
| 1 | 1.302 | −0.004 | ceiling-diff | ceiling-diff |
| 2 | 1.309 | −0.050 | ceiling-diff | ceiling-diff |

The anchor recovers `s_tech` to <1% and preserves the call. Its load-bearing value is
**correcting the ceiling magnitude** (raw folds the 1.3× into the estimate) and — decisively —
resolving the **non-uniform** case that defeats (B) (§3.4).

### 3.4 The residual gap — **honestly bounded**
An **above-median-only** (non-uniform) nuisance — scale applied only to ON-cells — is *outside*
the uniform-affine family (B) guards, and it is **observationally identical to a genuine ceiling**:

| seed | c | baseline | guard B |
|---|---|---|---|
| 0 | 1.25 | ceiling-diff | **ceiling-diff (FOOLED)** |
| 0 | 1.50 | ceiling-diff | **ceiling-diff (FOOLED)** |
| 1 | 1.25 | ceiling-diff | no-difference |
| 1 | 1.50 | ceiling-diff | no-difference |

(B) is fooled on seed 0. This is **expected and acceptable**: it is the documented
"observationally-identical-to-ceiling" evader (the P5 repro explicitly excludes it), and *no*
readout-only method can separate it from a real ceiling. **This is exactly (A)'s domain** — the
inert block reveals the technical scale even when the signature scaling is non-uniform. So the
two mechanisms partition the class cleanly: **(B) covers the uniform-affine family completely;
(A) covers the non-uniform residual and the OFF≈0 degenerate regime.**

### 3.5 A measured negative-within-the-positive (important)
The *first* hypothesis for (B) was a **Fisher/Laplace condition-number** degeneracy test on the
joint `[knob, s, o]` Hessian. **It does not work:** the raw condition number saturates
(~900–2600) for *every* case — confound and genuine ceiling alike — because the linear offset
`o` (count units) and the log-scale params sit on wildly different unit scales, so the condition
number is dominated by that unit mismatch, not by identifiability. Wiring the guard on it
over-abstains catastrophically (abstains on genuine gain and ceiling too — safe but useless).
The **local profiled-curvature ratio** (Schur complement of the knob given the nuisance) is no
better: ~0.32 for a uniform-scale confound *and* ~0.30 for a genuine ceiling.

**The discrimination is GLOBAL, not local.** Whether one affine can match *both* modes at once
is a goodness-of-fit question, answered by the **integrated profiled ΔBIC (`earn`)**, not by any
local curvature. This is the honest refinement of the "measured Fisher degeneracy" hypothesis:
the degeneracy that matters is global, so the right measured statistic is the profiled ΔBIC.

### 3.6 Compute cost
Guard B ≈ **16–60 s/call** (median ~20 s) vs baseline `attribute_differential` ~ a few s —
roughly **5–10×** the shipped path (it fits a reference + two augmented models, up to two
directions). Acceptable for an opt-in per-call guard; halvable by running one direction when the
confounded side is known, and by fewer optimizer steps.

**Verdict: the prototype WORKS.** One continuous guard makes the entire uniform-affine family
abstain (0 confident-wrong where the bands leave 13/30), preserves every baseline-resolvable
positive, and the anchor recovers a ceiling under scale — with an honestly-stated residual
(non-uniform nuisances need the anchor) and an honestly-reported dead-end (the condition-number
test).

---

## 4. Migration plan — differential off the bands, onto the unified guard

Do **not** rip out the shipped gates; layer the earn-guard and retire the bands behind it.

1. **Add the affine-null fit to `differential`** as an opt-in refit: when a Δ model wins, refit
   it with a free per-context affine `(s, o)` and compute `earn` (profiled ΔBIC over the affine
   null). Reuse the exact NLL primitives already imported (`_mode_mean`/`_mode_cov`/`_mvn_logpdf`).
2. **Replace gates 4b/4c (and the pending P5 gate) with one gate:** *resolve a `*-diff` only if
   `earn ≥ margin` in both directions.* The `off_shift` / `off_scale` diagnostics stay as
   *reported* fields (transparency), no longer load-bearing thresholds. This removes every
   calibrated band and its blind gaps at once.
3. **Calibrate the single `earn_margin` once**, the way `margin_k` was calibrated (FINDINGS §1):
   sweep it against the affine family + the positive controls and pick the point with 0
   confident-wrong and maximal positive retention. Measured separation ([−7.6,−6.1] vs [+33,+83])
   says a wide plateau exists; **6.0 (a BIC "strong" margin) sits squarely in it.**
4. **Wire the inert anchor** into the counts→activity bridge (`inference/bridge.py`): when an
   inert gene set is supplied, estimate `(s, o)` per condition and normalize before attribution;
   when it is not, fall back to (B) and abstain on the non-uniform residual. Add a `NUDGE-LIM`
   entry stating the residual precisely (non-uniform / OFF≈0 nuisances need the anchor).
5. **Regression-lock** with the affine sweep as a decoy battery (a continuum, not sampled
   magnitudes) asserting 0 confident-wrong, plus the positive controls asserting retention.

---

## 5. Generalizing the pattern

Every attribution capability with a **perturbed-side technical nuisance** inherits the same
guard, because the mechanism is generic (a free affine null + a profiled-ΔBIC earn test):

- **`multi_reporter`** — each reporter is already an affine `y_j = base_j + gain_j·(…)`; a
  per-reporter technical affine is absorbed the same way. The heterogeneous-gain panel gives the
  earn test *more* leverage (a per-reporter nuisance must fit every reporter at once), so the
  guard is *stronger* here.
- **`epistasis`** — the additive-ambient synergy hole (`NUDGE-LIM-009`, locked xfail) is a
  per-condition additive offset on the A+B signature: the same free-`o` null absorbs it; the earn
  test asks whether the interaction term earns its parameter over the offset. A candidate route to
  *un*-lock that decoy with a runtime guard (currently there is none).
- **`attribute` (single-condition)** — an inert anchor normalizes the perturbed readout before the
  Lyapunov fit; the fallback is the existing depth pin plus the earn test against a free scale.

The unifying primitive to factor out: `absorbable_by_affine(ref, pert, knob) → (bool, earn)`.

---

## 6. Tradeoffs & preconditions

- **Compute** — 5–10× the shipped differential per call (§3.6); opt-in, so the default path is
  unaffected.
- **Over-abstention** — measured *none beyond the baseline* on the positive controls. The earn
  test only abstains when a free affine genuinely out-explains the knob; the huge margin keeps
  real mechanisms resolved.
- **The anchor precondition** — (A) needs an inert feature set. On real Perturb-seq this is nearly
  free (thousands of unmoved genes); on a bespoke low-plex readout it may not exist, and then only
  (B) applies (with its non-uniform residual). State this loudly wherever (A) is offered.
- **The residual** — (B) cannot separate a **non-uniform** perturbed-side nuisance from a genuine
  ceiling (§3.4); that is a true identifiability limit, not a tuning gap, and it is (A)'s job.

---

## 7. The principle (added to `design/STATE.md`)

> **Guard the identifiability, not the confound** — an orthogonal anchor or a measured
> degeneracy, never a calibrated band. A per-confound OFF-cluster band fixes one magnitude and
> leaves the next as a blind gap; a free-nuisance null *contains the whole confound family*, so a
> single measured statistic (the profiled ΔBIC / "does the knob earn its parameter over the free
> nuisance") covers the class continuously. When even that is degenerate (a non-uniform nuisance
> observationally identical to a mechanism), resolve it with an **orthogonal anchor** (inert
> features) or **abstain** — never widen a band.
