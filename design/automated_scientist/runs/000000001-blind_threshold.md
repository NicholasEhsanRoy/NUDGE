# Run 000000001 — blind_threshold (attribute) · WITH vs WITHOUT NUDGE

*Immutable record (append-only; never edit). Automated-scientist blind eval, Mode 1 (headless).
Both arms of one case, so the ablation reads in one place.*

- **Case:** `blind_threshold`  (surface `attribute`)  ·  **model:** `claude-opus-4-8` (headless `claude -p`)
- **Web:** DENIED (both arms)  ·  **Sandbox:** outside the repo (agent cannot reach `eval_keys/`)
- **Agent-facing data:** `blind_test.h5ad` — 10,000 cells × 101 genes, `{control:5000, Activation:5000}`;
  scrubbed (no `true_mechanism`, empty `uns`).

## Ground truth (held out during the run; revealed here for audit)

- **Mechanism:** `threshold` — perturbed param `K` × **1.6**, seed 7, 1-node Ras switch.
- **Expected calibrated verdict (answer key):** *"unresolved / gain_or_threshold — a single snapshot
  cannot separate gain from threshold; abstain + 'need a 2nd operating point'. Confident-wrong is the
  only hard fail."*
- **MEASURED case property (post-hoc, `scripts/eval` + direct check):** at this operating point the
  circuit readout is **unimodal** and the K×1.6 shift is **below detection** — reporter mean
  8.70→8.44, **Cohen d = −0.043**, KS p = 0.44, 0/101 genes significant (Bonferroni). So the honest
  answer here is closer to **no-effect** than to a detectable-but-degenerate gain⇄threshold. This is a
  **case-calibration finding**, recorded loudly: factor 1.6 at the default operating point does not
  produce the intended detectable-but-ambiguous effect (see LEDGER → "case-calibration notes").

## Result

| arm | verdict | grade | turns | cost |
|---|---|---|---|---|
| **with-nudge** | NUDGE `attribute` → `unresolved`; agent → **no-effect / abstain on mechanism** (HIGH conf) | `correct-abstention` (PASS) | 15 | $0.8 |
| **without-nudge** | agent → **no detectable effect / abstain on mechanism** (~0.95 conf) | `correct-abstention` (PASS) | 13 | $0.718 |

**Ablation read.** *Both arms correctly abstained — 0 confident-wrong either way.* On a below-detection
perturbation a careful analyst reaches the honest answer with or without NUDGE, so the OUTCOME did not
differ. What differed is the PATH and the grounding: the **with-nudge** agent got a tool-backed
`unresolved` from `attribute` and used `explain_abstention` to map the pattern to a documented
limitation (**NUDGE-DECOY-004 / NUDGE-LIM-004**, the "dead-guide / no-effect" case) plus the specific
remedy ("a *second operating point*, not more cells"); the **without-nudge** agent had to reconstruct
that rigor from scratch (an injected-effect power analysis + CV classifiers) to justify the same
abstention. Honest conclusion: this near-null case demonstrates NUDGE's *grounding* value but is NOT a
money-shot for NUDGE's *unique* value (preventing a confident-wrong the control commits) — because the
control did not commit one. A detectable-but-ambiguous case is needed for that (see run 000000002).

## Contamination check

Both reports reason from the **data + tools**, not priors. Notably the initial prior ("named
'Activation' ⇒ up-regulation") was *overturned by the data* in both arms (the mean shift is a
negligible *decrease*) — the agents did not anchor on the label. With-nudge explicitly read NUDGE's
`unresolved` + `explain_abstention` output; without-nudge built its own power analysis. Clean.

---

## Verbatim `REPORT.md` — WITH NUDGE (15 turns)

```markdown
# Mechanism Attribution Report — blind_test.h5ad

## Hypothesis / framing
A perturbation labeled "Activation" was applied and profiled against a matched control.
Goal: attribute the MECHANISM (threshold / gain / ceiling ...) as precisely as the data
supports, or abstain honestly.

## Data inspection (pre-tool, direct reads)
- Shape: 10000 cells × 101 genes. `obs['condition']`: `Activation` (5000) vs `control` (5000).
- Genes: one named `Activation` (the reporter) + `BG0..BG99` (100 background genes).
- Counts (float, 0..65). Total counts/cell identical across arms (~208.5), so no depth shift.
- **Reporter `Activation`**: control mean 8.70 (var 38.0), perturbed mean 8.44 (var 33.7).
  Near-identical. Histograms overlap; quantiles nearly identical; unimodal, no bimodality.
- **Background genes**: mean 2.00 in both arms — unchanged.
- **Covariance**: no BG gene correlates with the reporter in either arm (max |r|≈0.04 = noise).
  → No panel of co-regulated downstream reporters of a shared latent is present.

Initial read: marginal effect of the perturbation is negligible; there is no obvious
distribution reshaping (no bimodality) and no covariance structure to exploit. Only a SINGLE
operating point exists (one dose), so gain⇄threshold is a priori degenerate. Proceeding to
run NUDGE `attribute` as the designed tool.

## Tool calls

### 1. `nudge.attribute(target=Activation, control=control, topology=1node)`
Output: `single.op.call = "unresolved"`, with restricted-fit NLLs
`n=3.6123, K=3.6123, vmax=3.5923` — all three essentially identical.
Read: no single knob (gain/threshold/ceiling) fits better than the others. The three
restricted models collapse onto the same loss, which is what happens when there is little
or no signal to explain. Non-positive verdict → followed up with `explain_abstention`.

### 2. `nudge.explain_abstention("unresolved")`
Output: single-operating-point gain⇄threshold Fisher degeneracy; "the breaker is a *second
operating point*, not more cells." Confirms that with one dose we cannot localize gain vs
threshold **if** there were an effect. But this framing presumes an effect exists — so I
tested that directly.

### 3. Direct statistical test — is there ANY effect? (sanity check on the abstention)
- Reporter `Activation`: KS 2-sample p = **0.44**; Mann–Whitney p = **0.23** → distributions
  indistinguishable.
- t-test p = 0.033, but the mean difference is **−0.26** (a *decrease*, opposite of
  "activation"), i.e. a trivial effect size surfaced only by n=10000 — not a real activation.
- Fano factor 4.37 (control) vs 4.00 (pert): both overdispersed, essentially equal → no
  switch/bimodality signature appeared or disappeared.
- Across all 101 genes: **0 pass Bonferroni** (α=0.05/101). Smallest p is a background gene
  (BG42, p=0.02) — exactly the tail expected from 100 null genes.
Read: the perturbed condition is statistically within the WT (control) distribution.

### 4. `nudge.explain_abstention("no-effect")`
Output: maps to **NUDGE-DECOY-004 / NUDGE-LIM-004** — a "dead guide" that *targets but does
not knock down/activate*: the perturbed distribution sits within the WT effect margin, and
NUDGE must return **no-effect** rather than report a mechanism just because the WT may be a
switch. My data match this decoy exactly.

## Abstentions hit
- `attribute` → `unresolved` (single-op gain/threshold degeneracy). Would need a *second
  operating point* (a dose series) to break it — but there is only one perturbed condition.
- More fundamentally: the direct tests show **no detectable effect at all**, so there is no
  mechanism to localize even in principle from this dataset.

## Final conclusion
**Verdict: NO-EFFECT — I abstain on assigning any mechanism (threshold / gain / ceiling).**

Despite the "Activation" label, the perturbed cells are statistically indistinguishable from
control at the reporter (KS p=0.44, MWU p=0.23) and at every one of the 100 background genes
(0/101 significant after multiple-testing correction); depth, dispersion, and distribution
shape are all unchanged. The only nominal signal is a negligible mean *decrease* that is the
wrong sign for activation and vanishes under distribution-level tests. NUDGE's `attribute`
correctly declined to call a knob (all restricted NLLs equal); the data pattern matches the
documented dead-guide / null-perturbation case (DECOY-004 / LIM-004).

**Confidence: HIGH** that the correct answer is "no detectable perturbation effect → no
mechanism can be attributed."

What would change this: (i) a **dose series / second operating point** — if the applied dose
simply sat on a flat part of a response curve, more doses could reveal a switch and break the
gain⇄threshold degeneracy; (ii) evidence the true reporter is a channel not in this panel
(the measured readout may be the wrong one). Neither is available here, so the calibrated
answer is **no-effect / cannot attribute**.
```

---

## Verbatim `REPORT.md` — WITHOUT NUDGE (control, 13 turns)

```markdown
# Blind perturbation mechanism report

Data: `blind_test.h5ad` — 10,000 cells × 101 features. `obs.condition` ∈ {control (5000), Activation (5000)}.
Features: one gene named `Activation` + 100 background genes `BG0..BG99`. `X` is integer counts (min 0, max 65, mean 2.06). No other metadata, layers, obsm, or uns.

## Working hypothesis (initial)
The perturbation is named "Activation" and there is a readout gene "Activation." Prior guess: the perturbation transcriptionally activates (up-regulates) that gene and/or a co-regulated program. Test whether the Activation arm differs from control, and if so how.

## Analyses and how I read each output

1. **Global structure / library size.** Total counts/cell: ctrl 208.5 vs act 208.4 — identical. Library-size KS p=0.82. → No global scaling/technical shift between arms.

2. **Per-gene differential mean (log2FC act/ctrl).** Largest |log2FC| across all 101 genes is 0.05 (BG42). The `Activation` gene: ctrl mean 8.70 vs act 8.44 (log2FC −0.04) — if anything *slightly lower* in the "Activation" arm, opposite of activation. → No gene is up- or down-regulated in mean.

3. **Per-gene variance.** Variance ratios act/ctrl all within 0.89–1.08. `Activation` gene is overdispersed (mean 8.7, var ~36, NB-like, r≈2.6); BG genes are Poisson-like (mean≈var≈2). No variance change between arms. → No change in dispersion/noise.

4. **Correlation structure.** Max |off-diagonal correlation| = 0.054 (ctrl) / 0.053 (act) — both at the sampling-noise floor for n=5000, 100 genes (expected max under independence ≈0.058). Largest correlation *difference* between arms is 0.074 (noise). → No co-regulation / pathway module in either arm and no induced covariance.

5. **Per-gene distribution tests.** Mann–Whitney: min p=0.021 (BG42), only 3/101 genes p<0.05 (expect ~5), nothing survives Bonferroni (min corrected p=2.08). p-values uniform (quantiles 0.02/0.22/0.47/0.69/0.99). KS: min p=0.068, 0 genes p<0.05. Chi-square on the full `Activation` count histogram: p=0.077 (ns). Skew/kurtosis of `Activation` match (skew 1.84/1.83, kurt 6.5/6.9). → No marginal distributional difference for any gene, including the readout.

6. **Multivariate classifiers (5-fold CV ROC-AUC).** LogReg raw 0.503, LogReg log1p 0.499, RandomForest (nonlinear) 0.504. → No linear or nonlinear joint signature distinguishes the arms at the single-cell level; no hidden subpopulation large enough to detect.

7. **Rare-subpopulation check.** Fraction of cells with Activation>20: ctrl 4.8% vs act 3.9%; ==0: 1.2% vs 1.0%. No enriched activated tail in the perturbed arm.

## Controls / power (why this is abstention-with-confidence, not underpowered)

- **Null pipeline check:** randomly splitting the control arm in half gives classifier AUC 0.510 — confirms the pipeline reports ~0.5 when nothing is there.
- **Sensitivity (injected effects on the real data):**
  - +5% mean on the `Activation` readout gene → MW p=5×10⁻⁵ (detectable, survives Bonferroni); +10% → 6×10⁻⁸; +15% → 5×10⁻¹⁷.
  - +5% mean on a background gene (mean≈2) → MW p≈10⁻⁵³.
  With n=5000/arm I can detect a ~5% change in the designated readout and far smaller changes in background genes. **I observe none of it.**

## Abstentions hit
- Cannot name a molecular mechanism (activator vs repressor, target gene, pathway, dose) because **there is no measurable perturbation signature to reason from.** Every axis — mean, variance, dispersion, covariance, full distribution, multivariate joint structure, rare subpopulations — is statistically indistinguishable from control.
- The classifier AUC only reaches ~0.52 even under a +15% injected readout effect (single overdispersed gene carries little separating power), so I relied on the per-gene MW test as the sensitive instrument; it is clean.

## Conclusion

**The "Activation" perturbation shows NO detectable transcriptomic effect relative to control.** The two arms are statistically indistinguishable across all tested dimensions, despite high statistical power (able to detect a ~5% shift in the named readout gene and ~2–3% shifts in background genes).

- **Confidence that there is no detectable effect in this dataset: high (~0.95).** This is well-powered, not "can't tell for lack of data."
- **On the MECHANISM of the perturbation: I ABSTAIN.** The data cannot support any specific mechanism because no signature is present. Consistent interpretations I cannot distinguish (and the data does not favor any of them): (a) the perturbation was biologically inert / failed to engage its target (e.g. vehicle-only, non-functional guide, no delivery); (b) the condition labels are uninformative/scrambled with respect to the true biology; (c) a real effect exists but lies entirely below the detection floor (<~5% on the readout) and leaves no higher-order (variance/covariance/subpopulation) footprint.

**What would let me decide:** a positive/technical control confirming the perturbation was delivered and the assay responds (e.g. a known-responsive gene panel), the identity of the intended target, replicate/batch structure, or a dose series. Absent those, the honest answer is that this dataset does not license any mechanistic claim.
```
