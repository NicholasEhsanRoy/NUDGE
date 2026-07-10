# NUDGE fail-safe red-team

**Mandate.** Adversarially try to make a NUDGE capability emit a *confident, specific,
WRONG* mechanism/interaction call (a positive call in an abstention-class situation) that
slips past the abstention gates. A found hole is a WIN (a documented limitation/decoy, as
NUDGE-LIM-006 was found); finding none hardens the claim. **This document reports; it does
NOT fix.** No `src/` capability code, decoy battery, or fail-safe margin was touched.

Repro scripts: `scripts/redteam/*.py` (lint-clean, ruff line-length 100; `uv run`).

## Score

| # | Capability | Attack | Verdict |
|---|-----------|--------|---------|
| 1 | `inference/lyapunov` (multi) | near-fold 3rd operating point corrupts the shared-param joint fit | **HOLE — verified** |
| 2 | `inference/epistasis` (+ bridge) | additive (non-depth) batch offset aligned with A+B fakes synergy end-to-end | **HOLE — verified** |
| 3 | `inference/cross_modality` (knob) | gain-down mutant via span-shrink ceiling gate; knob past a `unresolved` curve | HELD |
| 4 | `inference/dose_response` | graded (n≈1) curve + adversarial noise faking `switch` | HELD |
| 5 | `inference/bifurcation` | false `near-fold` on a robust/low-depth/monostable circuit | HELD |

**Confident-wrong holes found: 2** (lyapunov near-fold-3rd-point; epistasis additive-batch
confound). Capabilities that HELD under the attacks tried: cross_modality (knob localizer),
dose_response, bifurcation. Not adversarially exercised here (documented reasoning only, at
the end): multi_reporter, differential, design, classify/lyapunov-single.

---

## HOLE 1 — a near-fold 3rd operating point flips a TRUE CEILING → confident `threshold`

**Capability:** `nudge.inference.lyapunov.attribute_lyapunov_multi` (the M3 "second/third
operating point breaks the confound" path).
**Repro:** `scripts/redteam/nearfold_thirdpoint_hole.py` (`uv run python …`, ~1 min).
This reproduces + formalizes the hole first seen in
`scripts/analysis/toggle_gain_abstention_probe_RESULTS.txt` (ceiling, pts=3, gap 0.30).

**The claim under attack.** `attribute_lyapunov_multi` guards:
> "Abstain loudly unless EVERY operating point's LNA is trustworthy (one bad Gaussian
> corrupts the shared-parameter joint fit)" — `all(lna_reliable(p.circuit, p.scale))`.

**The attack.** A TRUE ceiling knockdown (`vmax ×0.6` on a mutual-repression toggle) seen
at three basal operating points `{0.05, 0.30, 0.60}`. The 3rd (basal=0.60) sits
aggressively close to the toggle's fold, but its LNA lobes have **not** overlapped enough
to trip `lna_reliable` (which fires only when a lobe's std exceeds `sep_ratio=1.0 ×` the
inter-mode separation). Its LNA moments are nonetheless already corrupted enough to bias
the shared-parameter joint fit.

**Confident-wrong output (verified, 2/2 seeds):**

```
seed=0 pts=2  gate_all_ok=True  label='ceiling'    gap=0.2017          # correct (clean pts)
seed=0 pts=3  gate_all_ok=True  label='threshold'  gap=0.3043  <== HOLE # + near-fold pt
    basal=0.05  lna_reliable=True (ok)
    basal=0.3   lna_reliable=True (ok)
    basal=0.6   lna_reliable=True (ok)   <-- the corrupting point PASSES the gate
    NLLs: gain=7.142  threshold=6.838  ceiling=7.257
seed=1 pts=2 -> 'ceiling'   gap=0.1609
seed=1 pts=3 -> 'threshold' gap=0.2377  <== HOLE
```

Truth = **ceiling**; with the clean 2 points NUDGE resolves it correctly; adding the
near-fold 3rd point makes it emit a **confident, specific, wrong `threshold`** by a margin
(gap 0.24–0.30) far exceeding `resolve_margin=0.03`.

**Which gate failed.** `lna_reliable` (the sole per-point trust gate) is the wrong
sufficient condition here. It only trips on **lobe overlap** (variance-collapse endgame)
and **insufficient depth**. A point that is *approaching* the fold — variance already
swelling, moments biased — but not yet overlapping is rated `ok`, and because
`attribute_lyapunov_multi` fits **one shared kinetic** across all points, that single
corrupted point poisons the joint argmin. Adding operating points is assumed to only ever
*help* (break the gain/threshold confound); here a third point that is individually
"reliable" but near-fold **actively flips a correct 2-point call to a wrong one**.

**Candidate decoy (described, not added).**
`NUDGE-DECOY-0xx — near-fold operating point corrupts a multi-point joint fit`: a true
`ceiling` toggle perturbation supplied at 3 operating points, the 3rd near the fold but
passing `lna_reliable`. Expected verdict: **`unresolved`** (abstain). NUDGE must NOT return
`threshold`. Generator: `toggle(basal∈{0.05,0.30,0.60})`, `vmax×0.6`, deep readout
(`Readout.identity(2, scale=15)`), SSA cells via `generate_toggle_perturbseq`. This is a
**positive-controlled** decoy: the 2-point subset (clean points only) must still resolve
`ceiling`, proving the decoy isolates the near-fold point as the culprit.

**Candidate limitation (described, not added).**
`NUDGE-LIM-0xx — a near-fold operating point corrupts the shared-parameter multi-point
fit; `lna_reliable` is necessary but not sufficient.` Severity: **major**, safety-relevant.
Body: the multi-operating-point breaker (M3) assumes each point that passes `lna_reliable`
contributes trustworthy moments to the shared-parameter joint fit. But `lna_reliable`
trips only at lobe *overlap* / low depth, so a point *approaching* the saddle-node fold —
whose Lyapunov covariance is already biased but whose lobes have not yet merged — passes
the gate and biases the joint argmin, flipping a true `ceiling` to a confident `threshold`
(verified: `scripts/redteam/nearfold_thirdpoint_hole.py`, 2/2 seeds, gap ≈0.24–0.30 ≫
resolve_margin 0.03). Mitigation direction (NOT applied here — main agent decides): a
**hard "well-buffered 2nd/3rd point" guard** that abstains when any operating point's fused
bifurcation proximity (`inference/bifurcation.bifurcation_proximity`, the two
*deterministic* channels, which `lna_reliable` ignores) exceeds a margin — i.e. gate the
multi fit on the *proximity dial*, not just on lobe overlap. The clean caveat NUDGE already
states ("the breaker is a *second, well-buffered* operating point") must become an enforced
precondition, not prose. Related: NUDGE-LIM-012 (the LNA breaks down at the fold).

---

## HOLE 2 — an additive (non-depth) batch offset aligned with A+B fakes `synergistic`

**Capability:** `nudge.inference.epistasis.attribute_synergy` fed by the shipped bridge
`nudge.inference.bridge.combo_effect_scores` (the CLI/MCP path).
**Repro (end-to-end):** `scripts/redteam/epistasis_pipeline_confound.py`.
**Repro (classifier-only):** `scripts/redteam/epistasis_dose_probe.py` (attack E1/E2).

**The claim under attack.** NUDGE-LIM-009 says the depth/batch confound is defended by
`combo_effect_scores`' **size-factor** normalization, and that NUDGE "abstains on an
underpowered or wide-CI combo," conceding only that it "cannot separate a confound that is
*perfectly aligned* with the A+B condition." The implied reader takeaway is that the
residual risk is exotic. This attack shows the residual risk is **routine and produces a
confident call with a *misdirecting* honesty flag**, not an abstention.

**The attack.** The confound is chosen to be **invisible to size-factor normalization**: an
**additive** count offset on the signature/HVG genes in the **A+B condition only** (ambient
RNA, or a batch-specific expression shift concentrated in the A+B 10x lane). Size-factor
normalization corrects a *multiplicative* library-size difference; an additive gene-level
offset survives it. The A+B condition is generated as the genuinely **Bliss-additive** sum
of A and B (truth = **additive**, interaction 0).

**Confident-wrong output (verified end-to-end through the shipped bridge, 4/4 seeds):**

```
seed=0: call='synergistic' interaction=+0.526 CI=(0.464, 0.602) neomorphic_ratio=1.69  <== HOLE
   reason: … a free A+B level beats the additive null by ΔBIC=849.2 — SUPER-ADDITIVE / synergistic …
seed=1: call='synergistic' … ΔBIC=1167.3 …
seed=2: call='synergistic' … ΔBIC=828.9 …
seed=3: call='synergistic' … ΔBIC=1023.7 …
```

Every abstention gate passes clean: 700 cells/condition (not underpowered), the interaction
CI is tight and entirely > 0 (not wide, does not straddle 0), and the BIC parsimony gate is
*emphatically* satisfied (ΔBIC ≈ 800–1200 ≫ margin 2.0). The classifier has **no internal
depth/batch defense** — the entire defense is the upstream size-factor step, which this
confound bypasses by construction.

**The honesty flag mis-fires.** `neomorphic_ratio` = 1.36–1.76 (≥ 1.0), so the off-axis
"possible-neomorphic / emergent structure" **under-count warning is appended** to the
reason. On a boring additive *batch artifact* this actively points the user toward "a
hidden regulator / emergent state" — the exact over-read NUDGE-LIM-015 exists to prevent.
The safety mechanism here amplifies the error instead of catching it.

**Which gate failed.** None of the epistasis gates (min-cells, CI-width `rel_width`, BIC
`bic_margin`, straddle-zero) can see a batch confound — that is upstream normalization's
job, and size-factor normalization is blind to a *non-multiplicative* (additive) offset.

**Candidate decoy (described, not added).**
`NUDGE-DECOY-0xx — additive batch offset on A+B faking synergy`: a synthetic AnnData whose
A+B condition is Bliss-additive but carries a fixed additive count offset on the signature
genes (perfectly aligned with the A+B lane). Expected verdict through
`combo_effect_scores → attribute_synergy`: **`unresolved`** (or `additive`). NUDGE must NOT
return `synergistic`. Generator sketch in `scripts/redteam/epistasis_pipeline_confound.py`
(`build_adata`). Pairs with a positive control (the same panel *without* the offset → the
true `additive`), so the decoy isolates the offset as the culprit.

**Candidate limitation (described, not added — SHARPENS NUDGE-LIM-009).**
NUDGE-LIM-009 already names "a confound perfectly aligned with A+B" as unresolvable, but it
frames the defense as size-factor normalization + the abstention gates. Sharpen it to:
*(a)* the residual confound need not be depth — an **additive, gene-level** batch offset
aligned with A+B is invisible to size-factor normalization and produces a **confident**
`synergistic` (tight CI, ΔBIC ~10³), tripping **no** abstention gate (verified,
`scripts/redteam/epistasis_pipeline_confound.py`, 4/4 seeds); and *(b)* the off-axis
`neomorphic_ratio` warning **mis-fires** on such a confound (it reads as emergent
structure), so it must not be read as corroboration of a real interaction. Mitigation
direction (NOT applied): require a batch/lane covariate to be *orthogonal to the A+B
condition* as a precondition, or add an A+B-vs-singles library/ambient-diagnostic gate that
abstains when a technical axis tracks the A+B condition. This is the epistasis analogue of
the differential per-context depth-ratio abstention (NUDGE-LIM-016) — which epistasis
currently lacks.

---

## HELD — cross_modality knob localizer (`inference/cross_modality`)

**Repro:** `scripts/redteam/cross_modality_knob_probe.py`.

- **Attack 1 (gain-down mis-called ceiling via the span-shrink gate).** A true gain-down
  mutant (Hill n 4 → 1.2, floor/amp/K held equal) over a dose grid that truncates the
  shallow curve's slow approach to plateau. Hypothesis: the `ceiling` gate fires on
  `span_shrunk` (no disjoint-CI requirement) before the `gain` gate. **HELD 0/6** — the
  robust multi-start Hill fit recovers `amp` well even for the shallow curve (no spurious
  span shrink), so it returns `gain` or `inconclusive`, never a wrong specific knob.
- **Attack 2 (knob emitted past a `unresolved` dose-response).** Architecturally the knob
  localizer is run regardless of the dose-response reliability verdict. **HELD 0/6** in the
  cases tried — the truncated variant curves collapsed in `amp`, so the `non-responsive`
  first gate caught them. *Note (latent, not a hole):* `classify_knob_shift` does not read
  `DoseResponseResult.call`, so a curve the shipped dose-response path abstains on for a
  reason *other than* a collapsed amp (e.g. an inflection-spanning failure with a large,
  rightward-shifted `K`) could still receive a knob. It was direction-correct in every case
  observed here; flagged for a future guard, not a verified confident-wrong.

## HELD — dose_response graded → switch (`inference/dose_response`)

**Repro:** `scripts/redteam/epistasis_dose_probe.py` (attack D1). A genuinely graded
(n=1 Michaelis–Menten) curve with adversarial noise + a planted "knee." **HELD 0/8** — the
bootstrap `n` CI always includes low `n` (lower bound never clears `n_switch=2`), so the
switch gate (which requires the *whole* CI > 2 *and* ΔBIC > margin) never fires. Calls were
`graded` (6/8) or `unresolved` (2/8) — correct abstention-class outcomes.

## HELD — bifurcation robustness dial (`inference/bifurcation`)

**Repro:** `scripts/redteam/bifurcation_probe.py`.

- **B1** deep well-buffered switches (n ∈ {4,6,8,12}) → `robust` / `unresolved`, never
  `near-fold`. **HELD.**
- **B2** low sequencing depth (scale ∈ {1,3,8,20}) → `robust` at all depths; the proximity
  is unchanged (the two deterministic channels are depth-independent) and `lna_reason`
  honestly caveats "insufficient depth" at low scale. The lobe channel can only *raise* the
  alarm (fail-safe `max`); it never manufactured a false `near-fold`. **HELD.**
- **B3** monostable circuits (n ∈ {1.0,1.5}, verified 1 stable mode) → `not-bistable`
  (score `None`), never a proximity. (n ≥ 2 IS bistable near the onset, so `near-fold` /
  `robust` there is *correct*, not a hole — positive control.) **HELD.**

---

## Not adversarially exercised here (documented reasoning)

- **`multi_reporter`** — the consistency guard + M=1 abstention (NUDGE-LIM-014) are the
  defense. The analogous attack to Hole 2 would be a **shared additive per-reporter batch
  offset** consistent across the panel (so the consistency guard sees a clean shared latent)
  but faking a ceiling change. Untested; a candidate follow-up.
- **`differential`** — has the per-context depth-ratio abstention (NUDGE-LIM-016) that
  epistasis lacks, so the *multiplicative* depth confound is guarded. The Hole-2 analogue —
  an **additive** batch on the perturbed cells but *not* the control (so per-context
  calibration doesn't absorb it) aligned with the context axis — is untested and is the most
  likely differential hole; it would mimic a `ceiling-diff`. Candidate follow-up.
- **`design/invert`** — inherits the reliability of whatever attribution it inverts; a
  confident-wrong upstream call (Hole 1 or 2) would propagate into a confident-wrong
  proposal. The safety gate (bifurcation proximity) is one-sided near the fold
  (NUDGE-LIM-013). Not separately exercised.
- **`classify` / `lyapunov`-single** — the single-snapshot path only ever returns an
  abstention-class label (`unresolved` / `gain_or_threshold`), so it is confident-wrong-safe
  by construction; the exposed positive surface is the *multi* path (Hole 1).

## Honest caveats on these findings

- Both holes are on **synthetic** data engineered to be adversarial. Hole 1 is a faithful
  reproduction of a previously-measured effect; Hole 2 is a plausible-but-constructed
  ambient/batch pattern. Neither is yet demonstrated on a real screen.
- Hole 2 lies *within the scope* NUDGE-LIM-009 already concedes ("cannot separate a
  perfectly-aligned confound"). Its value is **verifying** that the concession is a *confident
  `synergistic` with a misdirecting neomorphic flag*, not a graceful abstention — and that
  the trigger (an additive offset) defeats the specific defense (size-factor normalization)
  NUDGE-LIM-009 names. That sharpens an existing limitation; it is not a brand-new blind spot.
- The three HELD results are genuine wins for the fail-safe claim under the attacks tried;
  they do not prove no attack exists.
