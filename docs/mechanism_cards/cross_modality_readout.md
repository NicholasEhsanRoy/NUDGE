---
id: NUDGE-METHOD-002
name: cross_modality_readout
role: attribution-method
registry_name: CrossModalityReadout
vulnerable_to_decoys: []
documented_limitation: [NUDGE-LIM-006, NUDGE-LIM-008]
validated_in_regime: {min_dose_points: 4, notes: "Runs the shipped dose-response attribution (NUDGE-METHOD-001) on a CONTINUOUS single-channel readout (fluorescence / activity / fold-change), not UMI counts. The modality is DECLARED, never guessed: the bouncer refuses log-normalized or raw counts masquerading as continuous (NUDGE-LIM-008). Knob calls are COMPARATIVE (variant vs a control curve) and inherit the affine-readout bound (NUDGE-LIM-006). Validated on Chure 2019 (CaltechDATA D1.1241, LacI IPTG induction, operator O2, R=260 vs WT): inducer-binding mutants Q294K (K 71->626 uM) and Q294V (K 71->420 uM) localize to THRESHOLD; DNA-binding mutants Y20I (floor +0.46) and Q21A (floor +0.32) localize to CEILING/leakiness; the near-non-inducible Q294R ABSTAINS (span collapse). No mutant is mis-called and none is called gain(n); F164T (mildest inducer mutant) and Q21M (a stronger-binding DNA mutant with no added leakiness) are honestly INCONCLUSIVE at a single operating point."}
references: [Chure2019, RazoMejia2018, MonodWymanChangeux1965, HuangFerrell1996]
---

# Mechanism Card — Cross-modality readout attribution

> **ID:** `NUDGE-METHOD-002`  ·  **Role:** attribution-method
> **Stability:** experimental  ·  **Registry name:** `CrossModalityReadout`

## Summary

Runs the *identical* `K` (threshold) / `n` (gain) / `v_max` (ceiling) attribution NUDGE
does on single-cell counts, but when the readout is a **continuous single channel** —
flow-cytometry **fluorescence**, a live-cell **activity** reporter, or a **fold-change**
summary — instead of UMI counts. Nothing about the inference changes; only the observation
channel does. It fits each variant's response-vs-dose curve with the shipped dose-response
path and localizes its effect, relative to a control, to one **knob** — **threshold** (a
shift of the dose EC50), **gain** (a change of Hill steepness), or **ceiling / leakiness**
(a change of the response floor / span) — or abstains (**non-responsive** /
**inconclusive**).

## Why this exists (one vocabulary, any readout)

NUDGE's whole thesis — distinguishing *where* a switch trips (threshold) from *how sharply*
it commits (gain) and its ceiling — is measurement-agnostic, but its ingest hard-required
raw integer counts (`nudge.data.ingest.check_counts`), structurally locking out the entire
synthetic-biology and signaling world that measures switches by fluorescence. This adapter
removes that lock **without loosening the count guard**: a **modality-aware bouncer**
routes counts to the unchanged integer guard and a continuous readout to a new
continuous-readout guard, and the continuous curve then flows into the *same*
`NUDGE-METHOD-001` fit/classify. The decisive payoff is **Chure 2019's author-labelled
K-vs-ceiling ground truth** (below) — the single most direct external validation of the
threshold-vs-mechanism distinction available.

## Governing equation

The per-variant curve is the shipped Hill dose-response (`NUDGE-METHOD-001`), read on a
continuous response with an `"activate"` direction for an induction curve:

```
response(dose) = floor + hill(dose, K, n, amp)      # activate: rises with dose
```

Here **`K` is the induction EC50 — the half-max *inducer concentration*** — which for the
MWC induction curve is a function of *all* the allosteric parameters (`Ka`, `Ki`, `Δε_AI`),
**not** the raw inducer dissociation constant `Ka`. A shift in `K` therefore reports a
weakened/strengthened *inducer response*, the inducer-binding-domain axis.

The **knob call** is comparative — variant vs a control fit — in the direction each shift
is diagnostic:

```
threshold : log2(K_var / K_ctrl) >= k_octaves, disjoint K CIs   (dose-EC50 shift)
ceiling   : floor_var - floor_ctrl >= floor_rise, OR span shrinks (leakiness / range)
gain      : |n_var - n_ctrl| >= n_shift, disjoint n CIs          (Hill steepness)
non-resp. : amp_var < min_response_frac * amp_ctrl               (span collapse)
```

The **sign** of the EC50 shift is load-bearing: a *rightward* shift is a weakened dose
response (a threshold change), whereas a raised **leakiness floor** — which drags the
apparent EC50 the *other* way in fold-change space — is a changed repression setpoint (a
ceiling change). NUDGE reads these two apart rather than collapsing both to "K moved".

## The classifier (fail-safe, in order)

Each variant first passes the shipped dose-response classifier (**no-effect** / **graded**
/ **switch** / **unresolved**); then `classify_knob_shift` localizes the knob vs control:

1. **non-responsive** — the response span collapses (`amp` below `min_response_frac ×` the
   control's): no curve to attribute (e.g. a near-non-inducible mutant).
2. **threshold** — the dose EC50 shifts **right** by ≥ `k_octaves` with disjoint `K` CIs.
3. **ceiling** — the floor rises by ≥ `floor_rise` (leakier baseline) or the span shrinks:
   a changed setpoint / dynamic range.
4. **threshold** (leftward) — a pure leftward EC50 shift with no leakiness rise.
5. **gain** — Hill steepness changes by ≥ `n_shift` with disjoint `n` CIs.
6. **inconclusive** — no knob clears its gate at this single operating point (a raised
   floor can reposition the apparent EC50, so one curve cannot always separate the knobs —
   the honest answer is to abstain, not guess).

## Assumptions & simplifications

- **The modality is declared, never guessed.** The bouncer (`check_readout`) refuses
  ambiguous input — most sharply, **log-normalized or raw counts** dressed up as
  fluorescence (zero-inflation / all-integer / substantially-negative signatures) — rather
  than silently fit it (NUDGE-LIM-008). It does **not** convert modalities.
- The response is an (approximately) **affine** readout of one latent switch output; a
  nonlinear reporter (saturating FRET/YFP) can manufacture apparent ultrasensitivity
  (NUDGE-LIM-006), so the knob call holds only under a controlled/linear reporter.
- The knob call is **comparative** (variant vs control) and inherits the dose-response
  identifiability regime: the doses must span the inflection (`NUDGE-LIM-007`).
- **`K` is the induction EC50 (half-max inducer concentration), not the raw `Ka`.** A
  rightward `K` shift is a weakened inducer response — the inducer-binding axis.
- **The leakiness FLOOR (fold-change at zero inducer) is the clean pure-`Δε_RA` readout**
  — it depends only on the DNA binding energy, so a floor rise is cleanly
  DNA-binding-domain-attributable. **Saturation / dynamic-RANGE depend on *both* `Ka`/`Ki`
  and `Δε_RA`**, so a `ceiling` call is only cleanly DNA-attributable when it is **driven
  by the floor**; a range change alone cannot be decomposed from one operating point.
- **Gain (`n`) is abstained, not claimed invariant.** The *structural* cooperativity
  exponent (the exponent 2 = LacI's two inducer sites) is fixed by the protein
  architecture, but the *effective* Hill coefficient depends weakly on the `Ka`/`Ki` ratio
  (Razo-Mejia Eq. 10). Because the mutants' dominant, cleanly-attributable effect is on the
  EC50/threshold and any effective-steepness change is second-order, **abstaining on the
  gain axis is the honest call** — NUDGE does not manufacture a gain story.
- A single operating point cannot always separate threshold from ceiling (a leakiness
  change repositions the apparent EC50); those variants return **inconclusive**. The clean
  separations it *can* make from one context are analytic (EC50 ⟂ `Δε_RA`; leakiness floor
  ⟂ `Ka`/`Ki`), so one context suffices to tell a threshold from a floor shift — but **not**
  to decompose a dynamic-RANGE change into its `Ka`/`Ki` vs `Δε_RA` parts.

## Known failure modes

| Failure mode | Guard / witness | Limitation |
|---|---|---|
| Log-normalized / raw counts mislabeled as a continuous readout | the modality bouncer's zero-inflation / all-integer / negativity refusals (`tests/inference/test_cross_modality.py`) | `NUDGE-LIM-008` |
| A nonlinear reporter faking dose-response ultrasensitivity | the affine-readout bound shared with all attribution | `NUDGE-LIM-006` |
| A leakiness change read as an EC50/threshold shift | the sign-aware threshold-vs-ceiling gate (ceiling wins on a raised floor) | `NUDGE-LIM-008` |
| A knob forced on an underdetermined single-operating-point curve | the `inconclusive` / `non-responsive` abstentions (never a forced call) | `NUDGE-LIM-008` |

There is **no dedicated cross-modality decoy battery yet** (`vulnerable_to_decoys: []`) —
the failure modes above are guarded by the bouncer's refusals, the comparative knob gates,
and the real-data Chure lock-in; a synthetic continuous-readout decoy battery is future
work.

## Identifiability regime

- **≥ 4 dose points spanning the inflection**, an approximately-affine reporter, and a
  measurable control curve; else the method abstains rather than force a knob.
- **Verified on real data (Chure 2019, CaltechDATA D1.1241)** — LacI IPTG-induction
  fold-change, operator O2, repressor copy number 260, vs WT. The authors decompose LacI
  mutants into **DNA-binding-domain** (alter the DNA binding energy `Δε_RA` → repression
  setpoint / leakiness) vs **inducer-binding-domain** (alter `Ka`/`Ki`/`Δε_AI` → the
  inducer response). NUDGE recovers exactly this split onto its knobs: **inducer-binding
  Q294K** (induction EC50 `K` 71→626 µM, +3.1 octaves) and **Q294V** (`K` 71→420 µM, +2.6
  octaves) → **threshold**; **DNA-binding Y20I** (leakiness floor +0.46) and **Q21A**
  (floor +0.32) → **ceiling / leakiness** (a clean pure-`Δε_RA` floor rise); the
  near-non-inducible **Q294R** → **non-responsive** abstention (its span collapses). **No
  mutant is mis-called, and gain is abstained on every mutant** — the induction EC50 (`K`),
  not `n`, is where these single-site mutations act; the effective Hill slope's weak
  `Ka`/`Ki` dependence (Razo-Mejia Eq. 10) is second-order, so a gain call is never earned
  (this is an abstention, not a claim that `n` is invariant). `F164T` (the mildest inducer
  mutant) and `Q21M` (a stronger-binding DNA mutant with no added leakiness — only a mild
  EC50 drift, not cleanly separable at one context) are honestly **inconclusive** at one
  operating point — a copy-number series would resolve them. The recovered mapping
  (**inducer → threshold, DNA → leakiness/ceiling**) is the biophysically-correct reading
  of the authors' ΔF_inducer-vs-ΔF_DNA decomposition; the naive "DNA → K, inducer → n"
  prior is biophysically refuted (it inverts the roles) and NUDGE overrides it.
- **Residue numbering.** The CSV / mwc_mutants repo label residues with the LacI
  convention that **includes the N-terminal Met** — `Y20I`, `Q21A`, `Q21M` (DNA-binding),
  `F164T`, `Q294K/V/R` (inducer-binding). The PNAS paper text uses the −3 convention
  (`Y17I`, `Q18A`/`Q18M`, `F161T`, `Q291K/V/R`) — **the same mutants**, a LacI
  numbering-convention offset of +3, noted so a reader cross-referencing the paper is not
  confused.

## Implementation Mapping

| Step | Code |
|---|---|
| modality-aware ingestion bouncer (counts vs continuous; refuse ambiguous) | `nudge.data.ingest.check_readout` |
| continuous-readout table → `(dose, response)` per variant | `nudge.inference.bridge.fluorescence_dose_response` |
| fit + classify the continuous curve (reused verbatim) | `nudge.inference.dose_response.attribute_dose_response` |
| localize a variant's effect to one knob vs control | `nudge.inference.cross_modality.classify_knob_shift` |
| attribute a panel of variants vs a shared control | `nudge.inference.cross_modality.attribute_variant_panel` |
| CLI / MCP orchestration | `nudge.service.cross_modality_panel_file` |

*(This table is machine-checked by `scripts/check_impl_mapping.py`: every `nudge.*`
reference must resolve to a real attribute.)*

## Verification evidence

- `tests/inference/test_cross_modality.py::test_bouncer_refuses_log_normalized_counts` —
  the modality bouncer refuses zero-inflated log-normalized counts (NUDGE-LIM-008).
- `tests/inference/test_cross_modality.py::test_bouncer_refuses_raw_integer_counts` —
  raw integer counts mislabeled as fluorescence are refused.
- `tests/inference/test_cross_modality.py::test_counts_modality_delegates_to_integer_guard`
  — `modality="counts"` routes to the unchanged integer guard.
- `tests/inference/test_cross_modality.py::test_panel_localizes_threshold_and_ceiling` —
  a right-shifted-EC50 variant reads `threshold`, a leaky-floor variant reads `ceiling`,
  a flat variant abstains (synthetic ground truth).
- `tests/inference/test_cross_modality.py::test_chure_laci_kn_ground_truth_real_data` —
  the Chure 2019 lock-in: inducer-binding → threshold, DNA-binding → ceiling, Q294R
  abstains, no gain call, vs the author labels.

## References

- [@Chure2019] — the LacI mutant benchmark (CaltechDATA D1.1241): DNA-binding vs
  inducer-binding domain mutations decomposed by their free-energy shift (the K-vs-ceiling
  ground truth this recovers).
- [@RazoMejia2018] — the predictive MWC induction theory + the WT LacI reference
  parameters (Ka, Ki) the mutant shifts are read against (fallback dataset D1.743).
- [@MonodWymanChangeux1965] — the Monod-Wyman-Changeux allosteric model underlying the
  IPTG-induction fold-change curve (why inducer-domain mutations move the inducer axis).
- [@HuangFerrell1996] — ultrasensitivity / the K/n/v_max response-magnitude vocabulary
  shared with the single-cell and dose-response attribution paths.
