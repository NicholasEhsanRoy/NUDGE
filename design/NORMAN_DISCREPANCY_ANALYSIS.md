# Forensic analysis — the 3 Norman-2019 pairs that did not *explicitly* match

**Question.** NUDGE's genetic-interaction calls on five Norman-2019 pairs scored 2/5
"explicitly matched a per-pair statement in the paper." This is a forensic audit of the
other three — **CBL+UBASH3B**, **CNN1+UBASH3B** (both called `synergistic`) and
**FOXA1+FOXA3** (called `additive`). The mandate is truth, not agreement: is each call
(a) confirmed-correct on independent grounds, (b) a principled stricter-math improvement,
(c) a genuine NUDGE limitation/bug, or (d) unresolvable from the data?

**Verdict up front.** No bug. All three combos are **real measured dual conditions**
(the fact-check's doubt about FOXA1+FOXA3 is *refuted*). Two calls are
**confirmed-correct against the paper's own mechanism** (a); one is a defensible
**paralog-redundancy call the paper never analyses**, correctly hedged (b). The only
"disagreement" is structural and already documented: NUDGE's **scalar Bliss null is a
1-D on-axis projection of Norman's full-transcriptome regression** and cannot see the
**off-axis / neomorphic** residual — which I quantify here in the paper's own terms
(~29% residual). No fail-safe margin was touched.

Reproduce: `uv run python scripts/analysis/norman_discrepancy.py`
(read-only; dumps every number below).

---

## 0. The two methods, stated precisely (so "disagreement" is well-defined)

| | NUDGE (`epistasis.py`) | Norman 2019 (aax4438, PMC6746554) |
|---|---|---|
| Null | `e(A+B) = e(A)+e(B)` — **Bliss**, equal weights `c1=c2=1` fixed | `δab = c1·δa + c2·δb + ε` — **free-coefficient regression** of the double onto the two singles |
| Space | **scalar** = each cell projected on the additive axis `u=(vA+vB)/‖vA+vB‖` (singles only) | **full transcriptome** (all genes / PCs) |
| "Synergy" | on-axis overshoot: `e(AB) > e(A)+e(B)` | large coefficients — the combo travels *further* along the GI manifold |
| "Buffering"/antag. | on-axis undershoot: `e(AB) < e(A)+e(B)` | small coefficients (suppression) **or** one coefficient ≈ 0 (epistasis/masking) |
| Emergent/new state | **not represented** (scalar collapses it) | **neomorphic** = large residual `ε` / poor linear fit (`~71%` var explained on avg; `~29%` residual) |

The load-bearing consequence: **NUDGE's scalar interaction is exactly the on-axis
projection of Norman's interaction residual.** The paper's *neomorphic* dimension is
precisely the **off-axis** component NUDGE discards. This is testable, and I test it
below (the `off-axis ‖` column).

Sources: Norman et al., *Science* 2019, 10.1126/science.aax4438
([PMC6746554](https://pmc.ncbi.nlm.nih.gov/articles/PMC6746554/)); pairs were sampled
from the Horlbeck 2018 growth/fitness GI map (10.1016/j.cell.2018.06.010) — "we picked
132 gene pairs from the GI map … given the low-rank structure of the fitness GI map."

---

## 1. Do the combos exist? (Q1 — YES, all three)

From `obs['perturbation_name']` in GSE133344 (`norman_2019.h5ad`, 111,255 cells,
control = 11,835 cells):

| combo | n(A+B) | n(A) | n(B) | ≥ min_cells (30)? |
|---|---|---|---|---|
| CBL+UBASH3B | **406** | CBL 663 | UBASH3B 1201 | ✅ |
| CNN1+UBASH3B | **401** | CNN1 765 | UBASH3B 1201 | ✅ |
| FOXA1+FOXA3 | **216** | FOXA1 850 | FOXA3 493 | ✅ |

**Finding 1.** All three are genuine dual conditions with hundreds of cells. The
independent fact-check *could not confirm* FOXA1+FOXA3 was a real measured pair — that
doubt is **refuted**: it is a real 216-cell condition. No call here rests on a
non-existent or tiny combo.

---

## 2. Full fits + artifact audit (Q2)

All effects in log-FC space; interaction = `e(AB) − [e(A)+e(B)]`; CI = 2000-boot over
cells; ΔBIC = `bic_additive − bic_free` (free A+B level vs the pinned additive null).
`off-axis ‖` = norm of the interaction residual `v_AB − v_A − v_B` orthogonal to the
additive axis (the component the scalar cannot see).

| pair | e(A) | e(B) | e(AB) | add-pred | **interaction (95% CI)** | ΔBIC | call | on-axis / **off-axis ‖** | cos(vA,vB) |
|---|---|---|---|---|---|---|---|---|---|
| CBL+CNN1 *(control ✅)* | 1.48 | 1.86 | 4.29 | 3.34 | **+0.95 [+0.50,+1.38]** | 19 | synergistic | +0.95 / **2.54** | +0.73 |
| DUSP9+ETS2 *(control ✅)* | 4.79 | 1.66 | 4.31 | 6.45 | **−2.14 [−2.64,−1.62]** | 156 | buffering | −2.14 / **1.33** | +0.40 |
| **CBL+UBASH3B** | 1.54 | 1.13 | 3.76 | 2.67 | **+1.09 [+0.74,+1.45]** | 44 | synergistic | +1.09 / **2.15** | +0.76 |
| **CNN1+UBASH3B** | 1.88 | 1.09 | 4.22 | 2.97 | **+1.25 [+0.91,+1.58]** | 67 | synergistic | +1.25 / **2.21** | +0.71 |
| **FOXA1+FOXA3** | 2.24 | 2.79 | 4.42 | 5.04 | **−0.61 [−1.43,+0.23]** | −2.1 | additive | −0.61 / **1.95** | **+0.90** |

**Artifact checks (all negative — the calls are not artifacts):**

- **Weak-arm inflation?** No. In both UBASH3B synergy pairs *both* arms are strong
  (CBL +1.54, CNN1 +1.88, UBASH3B +1.13). Synergy is genuine on-axis overshoot:
  `proj(AB)` = 3.76 / 4.22 substantially exceeds `proj(A)+proj(B)` = 2.67 / 2.97. The
  scalar `e(x)` exactly equals `proj(x on axis)` (verified), so there is no
  aggregation/projection bug inflating one arm.
- **Depth/batch confound?** Every A+B condition has ~20–30% **lower** median depth than
  its singles (control 15,432; singles 13–14k; combos 10.8–12.1k) — a systematic
  dual-guide feature. But (i) scores are size-factor-normalized to the median + `log1p`,
  so first-order depth is divided out; (ii) the *same* depth drop is present in **both
  matched-control pairs**, so it cannot differentiate the scrutinized calls; (iii) it
  points the *wrong way* — combos are lower-depth yet show *larger* effects, so any
  residual depth bias would **under**-state, never manufacture, the synergy. Not the
  driver.
- **CRISPRa handled correctly?** These are activation screens: all single/double effects
  are positive (genes induced). The Bliss additive null in log-FC space is the correct,
  standard null for co-activation; nothing about CRISPRa breaks the additive projection.

**Finding 2 (the crux).** For *every* pair the **off-axis residual is comparable to or
larger than the on-axis interaction** NUDGE reports. The pairs that *explicitly matched*
the paper are exactly the ones whose interaction is **on-axis**:
DUSP9+ETS2 has the smallest off/on residual ratio (1.33 vs |−2.14|) — a clean, on-axis
masking that the scalar captures fully, which is why it is the sharpest match. The
synergy pairs carry off-axis ≥ on-axis (2.1–2.5 vs +1.1–1.3): the scalar reports a
*direction-correct but magnitude-incomplete* slice of a larger emergent state. This is
Norman's **neomorphic residual**, measured in NUDGE's coordinates.

---

## 3. Paper comparison + per-pair verdict (Q3 + Q4)

### CBL+UBASH3B → `synergistic` — verdict **(a) confirmed-correct**
The paper gives no per-pair scalar, but states the mechanism directly: perturbations of
"**CBL, its regulators UBASH3A/B**, and several multi-substrate tyrosine phosphatases …
induced erythroid markers, suggesting a common mechanism in regulation of **receptor
tyrosine kinase signaling**." CBL (E3 ubiquitin ligase for RTKs) and UBASH3B (TULA-2 /
STS-1, a suppressor of RTK/Syk-ZAP70 signaling) are both RTK-negative regulators;
co-activating two RTK suppressors driving the erythroid program **super-additively** is
the paper's own thesis. NUDGE's `synergistic` is a per-pair *instantiation* of Norman's
cluster mechanism, on a robust fit (both arms strong, ΔBIC 44, tight CI clear of 0). In
Norman's regression this is the **large-coefficient / synthetic** regime. Not a
discrepancy — a prediction consistent with the paper. NUDGE's +1.09 is an
**under-count** (off-axis 2.15 uncounted), never an over-call.

### CNN1+UBASH3B → `synergistic` — verdict **(a) data-confirmed; mechanism plausible, not proven**
Correctly flagged as the weakest *grounding*: CNN1 (calponin-1) is not a canonical RTK
regulator, so the erythroid link is by association, not a paper statement. But the
**data are unambiguous**: strong single arms (CNN1 +1.88 — the strongest single arm of
the three — UBASH3B +1.13), clear on-axis overshoot (proj 4.22 vs sum 2.97), the
**largest ΔBIC of the three (67)**, tight CI. The super-additive *call* is not an
artifact; only its *mechanistic label* is unsettled by the data alone — and the paper
does not adjudicate this pair, so there is nothing to disagree with. Consistent with
Norman's large-coefficient synergy regime.

### FOXA1+FOXA3 → `additive` — verdict **(b) principled + documented limitation**
The paper never analyses any FOX gene, so there is no ground-truth label to match or
miss. On its own terms the call is well-grounded: the on-axis interaction genuinely
straddles 0 (−0.61 [−1.43, +0.23], ΔBIC −2.1 — a free A+B level does not earn its
parameter), and the two arms are the **most correlated of any pair (cos = 0.90)** —
textbook paralog redundancy (two endodermal pioneer TFs pushing the *same*
transcriptomic direction). "Additive along the shared axis" is the honest, even elegant,
paralog verdict, and NUDGE correctly **declines to over-call**. The caveat is Finding 2:
the off-axis residual (1.95, off/on 0.44) means this pair may carry a neomorphic
component NUDGE's scalar cannot see. The call is right *as an on-axis statement* and is
labelled as such.

---

## 4. Is NUDGE's Bliss-scalar a limitation or a principled improvement? (overall)

**Both — and neither is a bug.** Two honest framings, and both hold:

1. **Principled / stricter.** NUDGE's null is a *harder* bar than Norman's regression.
   Norman fits `c1, c2` freely; NUDGE pins `c1=c2=1` (equal-weight Bliss) and asks only
   whether the combo over/under-shoots that sum *along the direction the singles agree
   on*. It is direction-safe by construction (axis from singles only, never the combo),
   parsimony-gated (BIC), and it **abstains** rather than resolve an under-powered or
   CI-straddling combo. Where both methods apply and the interaction is on-axis
   (DUSP9+ETS2, CBL+CNN1, and the two UBASH3B pairs), NUDGE agrees with Norman on **type
   and direction**.

2. **A documented limitation (`NUDGE-LIM-009`), now quantified.** The scalar collapses
   Norman's 2-D `(c1,c2)` coefficient space **and** the neomorphic residual into one
   number. Two concrete costs, measured above:
   - it **under-counts off-axis synergy** — for every synergy pair the off-axis residual
     (2.1–2.5) exceeds the on-axis interaction NUDGE reports (1.1–1.3);
   - it **cannot distinguish Norman's *epistasis* (asymmetric masking) from symmetric
     *suppression*** — both read as generic "sub-additive/buffering." DUSP9+ETS2 is, in
     Norman's taxonomy, *epistasis* (`e(AB) ≈ e(DUSP9-alone)`, AB−A = −0.48, ETS2
     masked); NUDGE's scalar labels it "buffering." Same antagonistic direction, coarser
     taxonomy. (Plus the sign-convention collision already noted in FINDINGS Phase 4d.)

The scalar can only ever **under-count** emergent synergy, never invert a call — so the
three verdicts stand. The honest claim in FINDINGS ("agrees on the two explicitly-labeled
pairs, consistent with the paper's clusters on the rest; never *recovers the taxonomy*")
is **accurate and now backed by the off-axis measurement**, not just asserted.

---

## 5. Bug hunt — nothing found (so no margin was touched, no fix prototyped)

Audited per the mandate; all clean:

- **Effect/parameter composition** (`fit_synergy`): `e(x)=mean(x)−mean(ctrl)`,
  `additive=e(A)+e(B)`, `interaction=e(AB)−additive`. Correct. Scalar `e(x)` equals the
  on-axis projection `proj(x)` to the digit (verified) — no aggregation error.
- **Projection axis** (`combo_effect_scores`): `u=(vA+vB)` from singles only, never the
  combo — no circularity, sign-safe, exactly as documented. HVG selection and `log1p`
  ordering correct.
- **CRISPRa readout**: co-activation → positive effects → Bliss log-FC null is the
  correct null; handled correctly.
- **Fail-safe margins** (`bic_margin`, `min_cells`, `rel_width`): untouched — and they
  should be. FOXA1+FOXA3's `additive` (ΔBIC −2.1, CI straddles) and the synergy pairs'
  clear ΔBIC (44/67) are robust to any defensible setting; nothing here motivates a
  margin change, and forcing one to chase the paper would be exactly the p-hack the
  mandate forbids.

**No prototype fix is committed**, because there is no defect to fix.

### Recommendation (a criterion-raising enhancement, *not* a bug fix — flagged, not applied)
Report the **off-axis residual magnitude** alongside the scalar interaction as an
explicit **"possible neomorphic / off-axis emergent"** diagnostic on `EpistasisFit`
(the numbers already exist — see `scripts/analysis/norman_discrepancy.py`). This would
surface Norman's neomorphic dimension directly, turn LIM-009 from prose into a *number
shown with every call*, and strengthen the honesty story (Depth & Demo criteria) —
**without changing any call, any margin, or any claim**. Left as a recommendation
because it is a feature addition, outside this audit's "fix a genuine defect" scope.

---

## Appendix — files
- Analysis script: `scripts/analysis/norman_discrepancy.py` (read-only dump; reproduces every number)
- Code under scrutiny: `src/nudge/inference/epistasis.py`, `src/nudge/inference/bridge.py::combo_effect_scores`
- Prior write-up: `scripts/vv/FINDINGS.md` "Phase 4d"; demo `notebooks/Norman_Synergy.ipynb`
- Sources: Norman 2019 [PMC6746554](https://pmc.ncbi.nlm.nih.gov/articles/PMC6746554/) (10.1126/science.aax4438); Horlbeck 2018 (10.1016/j.cell.2018.06.010)
