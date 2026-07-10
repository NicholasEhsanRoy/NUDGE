# Readout–circuit identifiability & the constitutive-control mitigation

**Status: validated in simulation; a STRETCH feature (not core).** This document
preserves a result we do not want to lose: *why* NUDGE misattributes a nonlinear
measurement readout as a circuit switch (the identifiability degeneracy behind
`NUDGE-LIM-006`), and a **validated mitigation** — a constitutive-reporter control — that
turns the limitation into a concrete capability and a suggestion to the field. It is a
stretch feature only because controlled public data is hard to find (see "Real-world
realization"); the method itself is validated.

Cross-refs: `scripts/vv/FINDINGS.md` ("NUDGE-LIM-006 mitigation"),
`docs/known_limitations.yaml` `NUDGE-LIM-006`, `JUDGES_GUIDE.md`. Reproducible validation
code: `scripts/vv/readout_identifiability/` (standalone JAX).

## 1. The problem — why LIM-006 is fundamental, not a fitting bug

NUDGE assumes an **affine** reporter (`Λ = base + scale · activity`). If the true reporter
is **nonlinear** (saturating / sigmoidal), a *linear* (non-switch) circuit produces a
skewed, pseudo-bimodal count distribution that NUDGE can only explain by bending the
*circuit* — a confident false positive (`NUDGE-LIM-006`, verified: at a steep readout +
strong perturbation NUDGE emits `threshold`/`gain`/`ceiling` even at high fit budget).

The root cause is **identifiability**: per cell, an input `u` drives a circuit map
`a = g(u; θ)` (Hill, `θ = {K, n, vmax}`), then a readout map `Λ = R(a; φ)` (Hill,
`φ = {Km, h, Vmax}`), then counts. **Only the composition `R∘g` is observed.** From a
single population you cannot factor the composition into its circuit and readout parts.

Measured (standalone JAX study; true circuit `n=3`, true readout `h=2`; audited +
independently reproduced):

- The profile likelihood over the circuit Hill `n` is **FLAT** — loss span **0.0003**
  across `n ∈ [1, 10]`; a graded **`n = 1`** circuit (no switch — *all* nonlinearity in
  the reporter) fits within **0.0001** of the true `n = 3`. **You cannot even tell a
  circuit switch exists.**
- The circuit `n` and readout `h` trade off along a ridge: `corr(n, h) = −0.905` among
  near-optimal fits, which reach `h ≈ 5–7` (true 2) at loss *below* the truth's.
- The composition `R∘g` is nonetheless pinned (observed-map rel-RMSE **6.6%**) — the fit
  is good; the *split* is unidentifiable.

## 2. The mitigation — a constitutive-reporter control

**Idea.** Add a **calibration population** in which the reporter is driven **directly at
known activity levels**, *bypassing the circuit*. This population observes `Λ = R(a)` at
known `a` — i.e. it measures the reporter's transfer function directly — so it **anchors
the readout parameters `φ`**. The circuit population, with `φ` anchored, then identifies
the circuit nonlinearity.

**Result (validated + reproduced).** Adding the control (reporter at known doses):

- **`n = 1` is REJECTED** — Δloss **0.017 ≫ floor** (the `n`-profile span grows ~50×). The
  data now say the ultrasensitivity is **biological**, not a measurement artifact.
- The readout is pinned — `h`-profile develops a deep well at the true `h = 2` (span ~800×
  the no-control curvature).
- The ridge collapses — near-optimal multistart fraction **0.07 → 1.00**; `corr(n, h)`
  goes from **−0.905 → +0.38**.
- **Honest caveat (confirmed):** the control lets you *reject "no switch"* but does **not
  point-identify** the exact `n` (recovered `≈ 5` vs true `3`) — the circuit's *internal*
  `K/n/vmax` trade-off persists. Full point-identification would need a **second anchor**:
  an input titration / circuit dose-response (anchoring the circuit's input axis).

**Verification.** The forward model + control were audited (the control uses **only**
readout params at **known** doses — no circuit-parameter leakage) and the headline was
independently reproduced (no-control `n`-profile flat 0.0003; with-control `n = 1` rejected
0.017). See `scripts/vv/readout_identifiability/` (`run.py`, `model.py`, `results.json`,
`profiles.png`).

## 3. Real-world realization — how a lab would run the control

The control needs a cell population whose reporter is driven **independently of the
circuit**, at graded/known levels, so its activity→signal transfer function is observed
directly. Plausible ways a well-resourced lab could do this:

- A **constitutive fluorescent reporter** (e.g. **mCherry**) expressed at graded levels —
  directly traces the reporter's transfer function.
- An **engineered constitutively-expressed synthetic barcode transcript** in a cell line,
  titrated across a known range.
- **Highly-stable housekeeping genes** used as a proxy for a known, approximately-linear
  reporter axis.

This is why the feature is *stretch, not core*: such a control is uncommon in existing
public Perturb-seq datasets. **If time permits we will search for a suitable real dataset**
(a fluorescent-reporter titration or dose-response paired with a perturbation screen).

## 4. Proposed NUDGE feature (stretch)

An optional **calibration-control channel**:

1. Accept a second data population (constitutive reporter at known/titrated activity).
2. In the forward model, replace the fixed affine `Readout` with a **fitted (non-affine)
   readout** whose parameters are anchored by the control's calibration curve.
3. Fit circuit + readout jointly across both populations.
4. **Absent a control, abstain on the circuit-vs-readout axis** (report that ultrasensitivity
   is present but not localized to circuit vs measurement) rather than mis-attribute — the
   fail-safe default consistent with `NUDGE-LIM-006`.

## 5. Contribution

Together, `NUDGE-LIM-006` and this validated mitigation are a self-contained methodological
result: mechanism attribution from single-cell snapshots is **fundamentally confounded by
reporter nonlinearity**, and a **constitutive-reporter calibration** is a concrete,
simulation-validated experimental design that restores identifiability enough to reject
"no switch." That is both a NUDGE roadmap item and a suggestion any lab could act on.

---

## The capture-scale confound (`NUDGE-LIM-019`) and its future robustness fix (Option B)

**The hole (red-team round 2, verified).** The constitutive control is a **separate
population**, and single-cell samples routinely differ in **capture / sequencing efficiency**.
The shipped module compares the control's reporter counts to the circuit population's directly
(`log1p` counts + energy distance) with **no relative-depth normalization between the two
populations**. Reading the control at ~0.5× the population's efficiency mis-anchors the reporter
`Vmax` low, so the with-control `n`-profile develops a spurious well away from `n=1` and asserts
`biological-switch` on a **truly linear (n=1) circuit** — re-opening exactly the `NUDGE-LIM-006`
artifact this capability exists to reject (`scripts/redteam/constitutive_control_batch_confound.py`,
3/3 seeds). It slipped past the module's own `is_confident_wrong` guard because that predicate
was scoped to bare knob calls only.

**What shipped (the honest, safe response — "Option A").** No runtime gate: a capture-scale
difference between the two populations is, *without a switch-independent shared reference*,
indistinguishable from a real switch at the count-magnitude level (a real switch also raises the
population's counts), so a naïve magnitude gate would false-abstain on genuine switches. Instead:
the shared-capture **precondition** is now stated; the verdict is reframed as **adversarially
bounded** (not "structurally fail-safe"); the falsifiable positive is surfaced
(`ConstitutiveResult.asserts_biological_switch`); and the confound is **locked** as a strict-xfail
decoy. Matched capture between the control and the population is a reasonable *experimental-design*
requirement to state — like asking for the constitutive control itself.

**Option B — the fundamental fix (DESIGNED, deferred; do not lose this idea).** Anchor both
populations to a **switch-independent shared reference** so a capture-efficiency difference
normalizes out *without* cancelling the switch signal:

1. **The reporter FLOOR as the shared reference.** The reporter's OFF-state level
   (`readout_base`) is the *same molecule at its basal level in both populations* and is
   **independent of whether the circuit switches** (an OFF cell reads the floor whether the
   circuit is linear or bistable). Estimate the floor in each population — in the **control**
   from its lowest-dose point(s); in the **circuit population** from its low-activity quantile
   (the OFF sub-population; robust because even a switch keeps an OFF mode populated at the
   depths where the LNA is trustworthy). The ratio *floor(control) / floor(population)* is a
   **capture-scale estimator that carries no switch information**, so rescaling the control to
   match the population's floor corrects the confound while leaving the switch signal intact.
   Unlike a total-count size factor (degenerate with `Vmax`, the classic `scale⇄vmax` problem),
   the floor anchor is orthogonal to the ON amplitude.
2. **A spike-in alternative.** If an exogenous spike-in / housekeeping panel is co-measured in
   both populations, use it as the depth reference directly (the standard normalization) — the
   most robust option when the experimental design provides it.
3. **Fail-safe fallback.** When neither the floor nor a spike-in can be estimated reliably
   (e.g. the population has no resolvable OFF mode, or the control lacks a near-zero dose),
   **abstain** (`unresolved`) rather than assume matched capture — converting the current silent
   confident-wrong into an honest abstention.

Option B would let the module **recover** the correct call under a capture mismatch (not merely
abstain), turning `NUDGE-LIM-019` from a locked bound into a solved case. It is deferred (not
attempted under time pressure) because the floor-estimation-from-a-mixed-population step needs
its own validation to avoid trading one confound for another. Tracked here so it is not lost.
