# Run 000000006 — gauntlet_B_noise (heavy over-dispersion / LNA trap) · WITH vs WITHOUT NUDGE

*Immutable record (append-only). Gauntlet Challenge B: heavily over-dispersed counts — does an
agent's ad-hoc noise model mis-attribute the observation noise to a kinetic (gain) knob?*

- **Case:** `gauntlet_B_noise` (surface `attribute`) · **model:** `claude-opus-4-8` · both arms had `Bash`.
- **Data:** `blind_test.h5ad` — 10k cells, a ceiling×2 perturbation under HEAVY NB over-dispersion
  (φ=0.6 → reporter Fano ≈ 12 both arms). NUDGE (pre-check): single-op `unresolved`.
- **Prompt:** determine the mechanism (threshold / gain / ceiling) or abstain honestly.

## Result — both arms ABSTAINED; the over-dispersion did NOT bait a gain call

| arm | conclusion | grade | turns | cost |
|---|---|---|---|---|
| **with-nudge** | ABSTAIN — no detectable effect (~0.85); single-op gain⇄threshold degenerate (NUDGE `unresolved`) | correct-abstention | 16 | $0.867 |
| **without-nudge** | ABSTAIN — modelled the noise explicitly (per-feature variance, zero-inflation, higher moments) → "abstain rather than over-interpret sampling noise" | correct-abstention | 12 | $0.63 |

**The trap did not land — twice over.** (1) The ceiling×2 effect was **swamped** at this operating
point (the 1-node readout barely moves; the shift was below detection under Fano ≈ 12, like run
000000001). (2) Even so, the control did NOT confuse over-dispersion for a gain change — it modelled
the count noise carefully (variance, zero-inflation, higher moments, both arms equally dispersed) and
correctly abstained. A frontier agent does not naively read excess variance as a kinetic knob.

**Efficiency signal REVERSED here:** on a clear-null case the control was *cheaper* (12 turns /
$0.63) than the NUDGE arm (16 turns / $0.867) — a fast null settles without NUDGE, and the tool
calls added overhead. So NUDGE's efficiency edge (run 000000005) is task-dependent, not universal.

## Ablation read (now 12 arms / 6 case types, 0 confident-wrong, no WITH>WITHOUT capability contrast)

The finding is now robust across statistics, confounds, bifurcation inversion, sloppy-model
identifiability, AND a heavy-noise LNA trap: a code-capable Opus 4.8 models its own noise and
abstains correctly; it is not baited into a confident-wrong. (Honest caveat on this specific case:
it was partly a WEAK case — the mechanism was swamped — so it under-tests the noise trap; a
detectable, distribution-shape-carrying mechanism in a bistable readout would test it harder. But
the control's explicit, correct noise modelling is itself the on-thesis signal.)

---

## Verbatim `REPORT.md` — WITH NUDGE (16 turns)

```markdown
# NUDGE Blind Mechanism Attribution — REPORT

Data: `blind_test.h5ad`. Task: determine the mechanism of a perturbation (condition
`Activation` vs `control`), or abstain honestly.

## 1. Data orientation

- Shape: 10000 cells × 101 features. `obs.condition`: `Activation` (5000) vs `control` (5000).
- Features: one reporter named **`Activation`** (idx 0) + 100 background genes `BG0..BG99`.
- Counts are integer (max 123), ~209 total counts/cell in BOTH conditions (no depth confound).

## 2. Descriptive differential (my own check, before NUDGE)

- Per-gene log2FC control vs Activation: **all |log2FC| < 0.12** (reporter included).
- The `Activation` reporter: mean 8.63 (ctrl) vs 8.81 (act), median 5 in both, near-identical
  heavy-tailed distributions. Welch t-test p > 0.1 — **reporter did not move**.
- Across all 101 genes: **0 pass Bonferroni** (α=0.05/101). 7 genes at raw p<0.05, vs ~5 expected
  by chance. This is the null-distribution signature — no real effect.
- Total library depth ctrl 208.5 vs act 209.1 (identical) → no depth/batch confound to explain.

**Reading:** at the descriptive level the "Activation" perturbation produced NO detectable
transcriptional effect on the reporter or any background gene. This points to a **no-effect**
outcome. Next: confirm with the NUDGE `attribute` circuit fit.

## 3. NUDGE `attribute` (target=Activation, control=control, 1node)

Result: `single.op.call = "unresolved"`. The three restricted-fit NLLs are essentially
identical — n=3.5737, K=3.5737, vmax=3.5611. The gain / threshold / ceiling hypotheses are
indistinguishable → NUDGE declines to name a knob.

`explain_abstention("unresolved")`: this is the **gain⇄threshold Fisher degeneracy** — at a
single operating point a gain change and a threshold shift give near-identical distributions.
The breaker is a **second operating point (a dose)**, not more cells. I have only two conditions
(binary Activation vs control), i.e. a single operating point — so this degeneracy is
structurally unbreakable with the data given.

`explain_abstention("no-effect")`: separate gate — perturbed distribution within the WT effect
margin (a "dead guide" that targets but does not knock down; NUDGE-DECOY-004 / NUDGE-LIM-004).

## 4. Is there any effect at all? (deciding no-effect vs a hidden subtle effect)

Full-distribution comparison of the `Activation` reporter, ctrl vs act:
- KS 2-sample: stat 0.015, **p = 0.63** → distributions statistically indistinguishable.
- Mean shift = 0.181 counts = **0.018 WT-std units** (1.8% of one SD) — far inside any effect margin.
- frac-zero 0.097 vs 0.098; skew 2.97 vs 2.93; variance 105 vs 118 (heavy-tail sampling noise,
  not significant given KS). Reporter is unimodal/geometric (Fano 12, overdispersed) — **no
  bimodal ON/OFF structure even in WT**, so no evidence the reporter reads a bistable switch.
- Reporter–background coupling unchanged (corr +0.015 vs −0.023).

So the data is consistent with **no effect** rather than a subtle-but-real perturbation.

## 5. Conclusion — ABSTAIN (no mechanism callable)

**I cannot attribute a mechanism, and I do not.** Calibrated read:

1. **Most likely: the "Activation" perturbation produced no measurable effect** (null / dead-guide
   archetype). Evidence: reporter KS p=0.63, mean shift 0.018 SD, 0/101 genes survive Bonferroni
   (7 at raw p<0.05 vs ~5 expected by chance), identical library depth (no confound masking a real
   effect). Confidence that there is **no detectable effect: high (~0.85)**.

2. **Even if a subtle effect exists, its mechanism is unresolvable here.** NUDGE returns
   `unresolved`: with a single operating point the gain⇄threshold (K⇄n) knobs are Fisher-degenerate
   (NLLs identical to 4 sig figs). Confidence a mechanism could be named from this data: **~0**.

**What would decide it:**
- A **positive control / second operating point** — a dose series of the activator (e.g. 0 →
  intermediate → saturating), which both confirms the perturbation does *anything* and breaks the
  gain⇄threshold degeneracy (`dose_response`). A single binary condition cannot.
- Or a **panel of multiple reporters of the same latent switch** (`multi_reporter`) to break
  K⇄v_max — not available here (only one `Activation` reporter; the 100 BG genes are inert
  background, not co-reporters).

**Final answer: ABSTAIN.** The data does not support any mechanism claim; the perturbation shows
no detectable effect, and even granting one, a single operating point cannot separate gain from
threshold. A confident mechanism call here would be wrong.
```

---

## Verbatim `REPORT.md` — WITHOUT NUDGE (control, 12 turns)

```markdown
# Blind Perturbation Mechanism Report

**Dataset:** `blind_test.h5ad`
**Analyst:** automated scientist (no NUDGE, generic Python only, web disabled)
**Date:** 2026-07-11

---

## 1. Data overview

- **Shape:** 10,000 cells × 101 features.
- **obs:** single column `condition` with two balanced groups — `control` (5000) and `Activation` (5000).
- **var:** 101 features named `Activation` (1) + `BG0`…`BG99` (100 "background" features).
- **X:** dense float32, but **integer-valued counts** (min 0, max 73, mean ≈ 2.07). No layers, no obsm, no uns, no var metadata.
- Counts are heavily overdispersed (see below), consistent with negative-binomial / bursty single-cell counts.

**Initial hypothesis:** The condition named `Activation` perturbs a pathway reported by the `Activation` feature; the 100 `BG` features are background/decoys. Expected mechanism signatures to test for, in order: (a) mean up/down-shift of `Activation` (transcriptional regulation); (b) variance/burst change; (c) rewiring of correlations between `Activation` and specific `BG` genes; (d) emergence/shift of a responding subpopulation; (e) any multivariate signature.

---

## 2. Tool calls and readings

### 2.1 Marginal mean/variance per feature (control vs Activation)
- `Activation`: ctrl mean 8.63 vs act mean 8.81 → **log2FC 0.03** (negligible). Variance 105 vs 118 (Fano ≈ 12–13, strongly overdispersed).
- **No** BG feature exceeded |log2FC| 0.3. Nothing stood out.
- **Read:** The perturbation is **not** a mean shift of the Activation reporter, nor of any feature.

### 2.2 Full distribution of the `Activation` feature per condition
- Percentiles (0/10/25/50/75/90/95/99/100), zero-fraction, and coarse histograms are **near-identical** between conditions (e.g. median 5 vs 5; frac==0 0.097 vs 0.098; 99th pct 48 vs 53).
- **Read:** The marginal distribution of the reporter is unchanged — rules out mean, variance-only, and gross shape/tail changes.

### 2.3 Correlation of `Activation` with each BG feature, per condition
- All correlations are at noise level in **both** conditions (mean |r| ≈ 0.011 ctrl, 0.012 act). Largest per-feature Δr ≈ 0.05 (BG47), consistent with sampling noise at n=5000.
- **Read:** `Activation` is statistically independent of every BG feature in control, and stays independent under perturbation. **No correlation rewiring.**

### 2.4 Per-feature hypothesis tests (all 101 features)
- Mann–Whitney (distribution), Welch t (mean), Levene (variance).
- Smallest p (MW) = **0.0070** (BG88). **Bonferroni ×101 = 0.71 → not significant.**
- Number of features with p<0.05 = **6**, exactly the ~5 expected by chance under the null.
- The `Activation` feature itself is **not** among the top-ranked — it shows no differential effect of any kind.
- **Read:** No feature is differentially expressed, dispersed, or distributed after multiple-testing correction.

### 2.5 Multivariate separability (global test)
- Logistic regression (log1p+scaled): 5-fold CV **AUC 0.505 ± 0.008**.
- Random forest (depth 6, 200 trees): 5-fold CV **AUC 0.504 ± 0.009**.
- **Read:** Neither a linear nor a nonlinear classifier can distinguish the conditions above chance (0.50). The label carries **no** learnable multivariate signature.

### 2.6 Confirmatory checks
- KS test on `Activation`: stat 0.015, **p = 0.63** (identical distributions).
- Skew/kurtosis of `Activation` near-identical (2.97/2.93; 14.8/13.0).
- KMeans (k=5) on log1p(X): cluster-by-condition contingency **chi² p = 0.82** — no subpopulation is enriched or depleted in the perturbed arm.
- **Read:** No hidden subpopulation response, no higher-moment change.

---

## 3. Abstentions hit and why

- **Mechanism class (transcriptional up/down):** cannot assign — no mean shift (2.1, 2.2).
- **Mechanism class (noise/bursting):** cannot assign — variance, Fano, higher moments unchanged (2.1, 2.6).
- **Mechanism class (regulatory rewiring):** cannot assign — no correlation change; reporter is independent of all BG in both arms (2.3).
- **Mechanism class (subpopulation / partial penetrance):** cannot assign — no cluster proportion shift, classifier at chance (2.5, 2.6).

Every mechanistic axis the provided readouts *could* resolve comes back null.

---

## 4. Conclusion

**The data does not support any specific perturbation mechanism. I abstain.**

Across every test I could construct — per-feature marginals (mean, variance, full quantile distribution, zero-inflation, higher moments), pairwise correlation structure with the putative `Activation` reporter, genome-wide differential testing with multiple-testing correction, unsupervised subpopulation structure, and both linear and nonlinear multivariate classifiers — the `Activation` condition is **statistically indistinguishable from control**. The strongest signal anywhere (raw p = 0.007) does not survive correction, and 5000 cells per arm give ample power to detect even modest effects (e.g. a ~10% mean shift in the reporter would be highly significant; none is present).

**Calibrated statement:** With **high confidence** I can say *what the mechanism is not*: it is not a detectable change in the level, variability, distributional shape, co-regulation, or subpopulation composition of any of the 101 measured features. I **cannot** name a positive mechanism, because there is no effect in the data to characterize.

The three explanations I cannot distinguish from this dataset alone are:
1. **The perturbation was ineffective / failed** (no biological effect to measure).
2. **The readout panel is blind to the affected axis** — the true target/effect lies outside these 101 features (the `Activation` reporter and `BG` decoys), so the mechanism is real but unprofiled.
3. **A labeling/control artifact** — e.g., the two arms were drawn from the same process (the "Activation" label is a decoy).

**What would be needed to decide (the honest "need X"):** a positive control or dose series showing the assay *can* register activation; an orthogonal readout of the intended target (the reporter here responds to nothing); replicate/batch metadata to rule out a swapped or null-labeled control; or the identity of the intended target gene so a directed, powered test can be run. Absent any of these, the correct scientific answer is to **abstain** rather than over-interpret sampling noise (the ~6 nominally p<0.05 features) as a mechanism.

**Final: ABSTAIN — the perturbation produces no measurable effect in this dataset; mechanism cannot be determined (high confidence in the null-effect finding).**
```
