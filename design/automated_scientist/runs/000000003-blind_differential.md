# Run 000000003 — blind_differential (two-context CONFOUND) · WITH vs WITHOUT NUDGE

*Immutable record (append-only; never edit). Automated-scientist blind eval, Mode 1 (headless).
The intended money-shot: a confound a careful analyst was hoped to miss. The finding turned out
richer than that — read the ablation.*

- **Case:** `blind_differential` (surface `differential`) · **model:** `claude-opus-4-8` (headless `claude -p`)
- **Web:** DENIED (both arms) · **Sandbox:** outside the repo.
- **Agent-facing data:** `blind_diff.npz` — four activity arrays `data_a/control_a/data_b/control_b`
  (3000 cells each, 1 species), scrubbed.

## Ground truth (held out during the run; revealed here for audit)

- **Truth = NO mechanistic difference.** The SAME perturbation ran in both contexts
  (`mechanism="none"`), PLUS a per-condition **×2.0 multiplicative technical scale on context B's
  PERTURBED cells only** (B's control clean, seed 11). The scale aliases onto a spurious per-context
  ceiling/gain/threshold difference — the bait.
- **Expected verdict:** `no-difference` / abstain. A confident `*-diff` is the confident-WRONG hard fail.

## Result — 0 confident-wrong; BUT the money-shot did NOT land (and why that's the finding)

| arm | conclusion | grade | turns | cost |
|---|---|---|---|---|
| **with-nudge** | ABSTAIN — "per-condition technical scale on B's perturbed cells, not a mechanism difference" (~0.9) | **correct-abstention** (PASS) | 12 | $0.923 |
| **without-nudge** | ABSTAIN — "per-condition technical/readout gain artifact (~2× on data_b), not a mechanism difference" (~0.80–0.85) | **correct-abstention** (PASS) | 10 | $0.768 |

*(The heuristic grader flagged the without-nudge arm "LIKELY CONFIDENT-WRONG" — a FALSE ALARM: it
cannot parse negation ("NOT a real mechanistic difference") and the truth term is `no-difference`
rather than a mechanism word. Human confirm: it is a correct abstention.)*

**Both arms correctly abstained — the control was NOT baited.** It independently found the
confound's fingerprint: the ×2 scale also doubles the OFF-population's below-baseline *noise*, which
"is not something a biological switch does — it is the signature of an instrument/readout gain
applied to one sample." That is the *same* OFF-cluster-scale tell NUDGE's own gate 4c uses. So even
this confound — designed to bait — did not fool a careful Opus 4.8.

## Two things this run surfaced (both important)

**1. The with-nudge agent CAUGHT A NUDGE FALSE POSITIVE — using NUDGE's own diagnostics.** It called
BOTH tools: the banded `differential` **abstained** (`unresolved`), but `differential_robust`
returned a **spurious `threshold-diff` (earn ≈ 42, cond ≈ 9437)**. The agent did not blindly trust
it — it read the huge condition number as evidence of an aliased, non-identifiable fit, cross-checked
against the banded abstention + its own analysis, and correctly discounted the robust tool's call.
NUDGE's transparency (reporting `cond`/`earn`/`(s,o)`) is what let the agent detect NUDGE's own bad
call — an unplanned but on-thesis demonstration of honesty-by-design.

**2. The eval FOUND A REAL NUDGE BUG (now fixed).** `differential_robust` defaulted to `steps=150`,
which under-fits the affine null; at seed 11 the ×2.0 confound spuriously "earned" a `threshold-diff`.
MEASURED: `steps=150` → earn +42 (WRONG); `steps=180` → −4.3 (abstains); `steps=250` → abstains
across seeds 11–14. Additive confounds are exactly in the null span and abstain even at 150. **Fixed:
service/MCP default bumped 150→250; FINDINGS §EG amended.** An under-optimized fail-safe manufactured
the very confident-wrong it exists to prevent — the abstention guarantee is conditional on convergence.

## The honest ablation read (across runs 000000001–3)

Six arms, three case types (near-null, truncated dose, per-condition confound): **0 confident-wrong,
every arm a correct abstention — with AND without NUDGE.** Opus 4.8 is well-calibrated enough to
reach the honest answer unaided, even on a confound built to bait it. NUDGE's demonstrated value here
is therefore **not** "prevents a confident-wrong the model would commit" — it is **grounding** (a
quantified, documented-limitation-backed verdict), a **cross-checkable second opinion**, and
**transparency** (its diagnostics let the agent catch even a NUDGE misfire). Where NUDGE would change
the OUTCOME likely needs a **less-calibrated scientist** (a smaller/older model) or a subtler confound
— a candidate next case. Stated plainly rather than overclaimed.

---

## Verbatim `REPORT.md` — WITH NUDGE (12 turns)

```markdown
# Differential Mechanism Analysis — blind_diff.npz

## Task
Same perturbation applied in two contexts A and B, each with its own control. Decide:
does the perturbation act via a DIFFERENT MECHANISM (threshold K / gain n / ceiling v_max)
in B vs A, OR is any apparent difference NOT mechanistic (e.g. a per-condition technical /
batch effect)? Abstain if undecidable.

## Raw data inspection (before tools)
- control_a: mean 20.83, std 29.21
- control_b: mean 20.91, std 28.58  → the two OFF baselines / controls are nearly IDENTICAL.
- data_a (perturbed A): mean 21.52, std 29.28 → barely differs from control_a (small effect).
- data_b (perturbed B): mean 40.65, std 57.50 → mean ~1.89×, std ~1.96× larger than data_a.

Key diagnostic — sorted Q-Q of data_b vs data_a:
  data_b ≈ 1.962 · data_a − 1.571, residual std = 2.58 (vs data_b std 57.5).
An almost perfect affine (mostly pure-scale) map takes data_a → data_b. Meanwhile the two
controls match. So B's perturbed cells look like A's perturbed cells multiplied by ~1.96,
with the OFF baseline unmoved.

## Hypothesis
This is the classic per-condition affine technical confound: a SCALE on context B's
PERTURBED cells only (its control clean), which aliases onto an apparent raised ceiling
(v_max-diff). If real, this is NOT a mechanistic difference. Prediction:
- naive `differential` may confidently call ceiling-diff;
- `differential_robust` (Earn-Guard vs free affine) should abstain → no-difference.
Will confirm with both tools rather than assume.

## Tool calls

### Call 1 — `differential` (naive/banded)
Verdict: **unresolved** (abstains). `is_reliable=false`.
Reason: context B's perturbed OFF-cluster scale is inflated 1.82× vs its OWN control
(outside band [0.8,1.3]) — fingerprint of a per-context MULTIPLICATIVE scale on B's
PERTURBED cells. Control-derived depth ratio ≈1.00 and additive OFF shift ≈1, so those
gates are blind, but a multiplicative factor is exactly degenerate with a v_max/ceiling
difference. Best BIC model was `vmax` (57800 vs shared 58110), i.e. it WOULD have called
ceiling-diff, but correctly abstained citing NUDGE-LIM-016 P4.
Read: the tool itself flags the perturbed-condition scale confound and refuses to certify
the apparent ceiling difference as mechanistic.

### Call 2 — `differential_robust` (Earn-Guard)
Verdict: **threshold-diff (K)**. `is_reliable=true`, earn_bic=42.1, s_hat=1.68, o_hat=-2.11,
cond_number=9437.8, knob_identifiable=true.
Read: this DISAGREES with the naive tool. Two red flags:
  (a) reason text asserts "cond=9437.8 ≤ 100" — self-contradictory; a condition number of
      ~9400 is a classic near-singular / aliasing symptom, not identifiability.
  (b) it calls THRESHOLD, but the raw-data geometry is a SCALE (see adjudication below),
      not a threshold shape.

### Independent adjudication (raw data, decisive)
Mapped data_b -> data_a with a single global affine (s=0.509, o=0.84, i.e. data_b ≈ 1.96·data_a):
- KS(data_a, affine-corrected data_b) D=0.035, p=0.047 — effect size SMALLER than the
  within-data_a half-vs-half reference D=0.040. => the ENTIRE A->B difference is one affine.
- Intrinsic mixing fraction preserved: frac-ON ≈ 0.38 in data_a, affine-corrected data_b,
  control_a, control_b. A real THRESHOLD (K) shift would change the ON/OFF fraction — it does
  not. (Raw frac-ON 0.393->0.434 is an artifact of the fixed >20 cutoff under scaling.)
- OFF mode pinned at ~1.3 in every array; only the ON-mode amplitude scales (49.5 -> 89.5,
  ×1.8). OFF-preserved + ON-amplitude-scaled + fraction-preserved = ceiling/scale signature.
- control_b ON mode (~49.5) == control_a ON mode (~50): B's CONTROL is clean; only B's
  PERTURBED cells are scaled. Exactly "affine nuisance on one context's perturbed cells,
  control clean."

Therefore the robust tool's threshold-diff is a FALSE POSITIVE: a pure ON-mode multiplicative
scale, forced into a shared-switch fit at N=3000, leaks into a small apparent ΔK that clears
ΔBIC=42, while the huge condition number (9437) shows K is aliased with the affine nuisance
(not truly identifiable). The naive `differential` — which has an explicit guard for this exact
perturbed-condition scale — correctly abstained.

## FINAL CONCLUSION
The perturbation does **NOT** act via a different mechanism in context B than in context A.
The apparent A/B difference is a **per-condition multiplicative technical / batch scale
(~1.9×) on context B's PERTURBED cells only** (its control is clean). This nuisance aliases
onto an apparent ceiling (v_max) difference and, in the shared-switch fit, leaks a spurious
small threshold (K) difference. The underlying switch mechanism (threshold, gain, ceiling) is
shared between the two contexts within the resolution of the data.

- Mechanism-difference call: **ABSTAIN** — no mechanistic difference is supported.
- Positive characterization: technical/batch confound (perturbed-condition scale), NOT biology.
- Confidence that this is NOT a real mechanism difference: **HIGH (~0.9)**.
  Basis: (1) a single affine makes data_b ≡ data_a (KS within sampling noise); (2) mixing
  fraction and OFF mode are preserved, ruling out threshold and gain; (3) B's control is clean
  while only B's perturbed cells scale; (4) the naive tool independently detected and abstained
  on this exact confound. The dissenting robust-tool threshold-diff is not credible here (aliased
  fit, cond≈9437, geometry inconsistent with a threshold shift).
- Residual caveat: I cannot fully exclude that a genuine ceiling change and a technical scale
  are perfectly co-occurring (they are observationally degenerate); that degeneracy is itself a
  reason to abstain on any confident mechanism-difference claim rather than assert one.
```

---

## Verbatim `REPORT.md` — WITHOUT NUDGE (control, 10 turns)

```markdown
# Blind diff analysis — does the perturbation act via a DIFFERENT MECHANISM in B vs A?

## Setup
`blind_diff.npz`: four scalar activity arrays, each (3000,1):
`data_a`/`control_a` (context A), `data_b`/`control_b` (context B). Same perturbation, two contexts, each with its own control.

## Working hypothesis
Activity-space with a bistable "switch": an OFF population near 0 plus an ON shoulder.
A real mechanism difference = a change in the switch's **threshold** (fraction turned on),
**gain/steepness**, or **ceiling** (on-state height). The competing explanation is a
**per-condition technical/readout gain** that scales one sample's whole channel.

## Tool calls and readings

1. **Shapes/summary** (`np.load`, percentiles). All four arrays 3000×1.
   controls: mean≈20.8–20.9, std≈28.6–29.2. `data_a` mean 21.5 (≈ control). `data_b`
   mean 40.7, std 57.5 — roughly double. Read: A looks null, B strongly elevated.

2. **Histograms (30 bins).** All four are bimodal: a sharp OFF peak at ~0 and a broad ON
   shoulder. control_a ≈ control_b visually. data_a ≈ control_a. data_b: OFF peak wider,
   ON tail stretched out to ~270. Read: switch structure, effect only in B.

3. **Controls compared quantile-by-quantile.** control_a vs control_b agree within ~1 unit
   across 1st–99th pct. Read: **baseline is identical in both contexts** — no batch effect on
   controls, so any A/B difference lives in the perturbed samples.

4. **Transfer (Q–Q) curves, perturbed vs own control.**
   - A: dataA ≈ ctrlA at every quantile, ratio ≈ 1.0 across the whole range → **identity /
     no effect**.
   - B: dataB ≈ **2.0 × ctrlB at every quantile**, ratio ~1.8–2.05 including the negative and
     near-zero quantiles (pct2: −11.0→−24.7; pct99: 99.7→204). Read: a uniform multiplicative
     scaling of the *entire* distribution, not a reshaping.

5. **Model fit for B.** Global multiplicative gain (QQ through origin) g = **1.987**, residual
   std = 2.42 (4.4% of spread). Additive-offset model residual std = 27.9 (12× worse).
   `data_b / 1.987` matches `control_b` across quantiles (deviations only in the extreme 1% tail).
   Read: B is cleanly `data_b ≈ 1.99·control_b`.

6. **Region-wise gain (the decisive test).** In B the gain is ~2 in **every** band:
   negatives(<0)=2.36, 0–15=1.86, 15–60=1.93, >60=2.02. In A: ~1 everywhere (neg 0.94,
   mid 1.01, high 1.02). OFF-band mass frac(|x|<2): control_b 0.163 → data_b 0.095 (halved);
   A unchanged (0.170→0.179). OFF-mode spread (p40−p10) doubles in B (7.6→14.0, 1.83×),
   unchanged in A (1.01×).

7. **Bootstrap (500×).** gain_A = 1.016 (95% CI 0.97–1.06) → indistinguishable from 1.
   gain_B = 1.987 (95% CI 1.90–2.08). After de-gaining, `data_b/1.99` vs `control_b` max
   quantile diff = 6.8 (mostly the 1% tail; comparable to data_a's own 3.0 residual vs control_a).

## Interpretation
The apparent A-vs-B difference is a **uniform ~2× multiplicative scaling of the whole data_b
readout channel** — it amplifies the below-baseline (negative) noise and the OFF-state cells by
the same factor as the ON cells, and it *widens the OFF-mode noise* by 2×.

None of the three mechanistic knobs can produce this:
- **Threshold** change moves the fraction of ON cells; OFF cells stay near 0. ✗
- **Gain/steepness** reshapes the sigmoid (ratio→1 at both saturated ends); it does not
  amplify OFF-state noise. ✗
- **Ceiling** change raises only the ON height; OFF/negative cells stay ~0 (ratio→1 at bottom).
  Here ratio is ~2 even at the extreme negatives. ✗

Doubling the OFF-population's below-baseline noise is not something a biological switch does; it
is the signature of an **instrument/readout gain applied to one sample**. Combined with (a)
identical controls and (b) an essentially inert perturbation in A and — after removing the ~2×
factor — in B as well, the most parsimonious account is: the perturbation shows no detectable
switch response, and the only between-context difference is a **per-condition technical gain
artifact on `data_b`.**

### Abstentions / caveats honestly noted
- **The one alternative I cannot fully exclude:** if the negative activity values are genuine
  *signed* biological activity (real repression) rather than measurement noise about zero, then a
  symmetric doubling of dynamic range could in principle be a true systems-level "gain." I judge
  this unlikely because the OFF state sits at ~0 with a noise tail (no biological reason for
  genuine −50 activity, and its width scales too), but the data alone cannot rule it out with
  certainty. This is the reason my confidence is not higher.
- I did not attempt an explicit two-component mixture fit; the quantile/region analysis is
  distribution-free and sufficient, but a mixture fit could add precision on the ON-fraction.

## CONCLUSION
**The apparent difference between context B and context A is most consistent with a per-condition
technical/readout gain artifact (≈2× scaling of the entire `data_b` channel), NOT a genuine
mechanistic difference in the switch (no threshold/gain/ceiling change is supported).** After
removing the ~2× factor, the perturbation is essentially inert in both contexts, and the baselines
are identical.

**Confidence: ~80–85%** that this is a technical (non-mechanistic) difference rather than a real
change in switch mechanism. The residual uncertainty is the signed-activity interpretation of the
negative tail noted above. I therefore do **not** call a real mechanism difference; if forced to
choose between "different mechanism" and "not a real mechanistic difference," I select **not a real
mechanistic difference**, with the caveat stated.
```
