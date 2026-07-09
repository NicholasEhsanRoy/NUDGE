# NUDGE — Capability Roadmap (mechanism-attribution as a reusable substrate)

**Status: strategy / design doc.** Written 2026-07-09. This proposes and rigorously
assesses *additional attribution capabilities* that extend NUDGE from "one clever trick
on single-cell bistable Perturb-seq" into a **general mechanism-attribution engine for
perturbation biology** — each expressed as a natural extension of the existing primitives,
not a rewrite. It is a menu with honest ratings, not a commitment. The honesty rule
(CLAUDE.md) governs every claim here: a capability with no real data or a shaky abstention
story is flagged as such, loudly.

Companion docs: `design/NOTEBOOK_LIBRARY.md` (the real-dataset catalogue this cross-
references), `scripts/vv/FINDINGS.md` (the measured identifiability / Fisher results the
ratings lean on), `design/WORKING_BACKWARDS.md` (the thesis), `design/ONTOLOGY.md` (the
mechanism-vocabulary graph these all emit into).

---

## 0. The substrate — what every capability below reuses

The insight driving this roadmap: NUDGE already proved (with the in-progress
`nudge.inference.dose_response` module) that the **same K / n / v_max vocabulary that
attributes mechanism from single-cell bistable distributions also attributes it from bulk
dose-response ultrasensitivity** — because single-cell bimodality and dose-response
ultrasensitivity are two *measurements of the same physical circuit*. Generalize that: many
biological screen types are different **measurements** or different **operating-point
structures** over the same Hill circuit. NUDGE's primitives are the reusable parts:

| Primitive | File | What it gives a new capability |
|---|---|---|
| **Hill circuit** K / n / v_max, differentiable JAX vector field | `core/circuit.py` (`production`, `vector_field`, `steady_state`) | the shared mechanism vocabulary + forward model |
| **`vmap` population model** | `Circuit.solve_population` | a single-cell simulator from a deterministic solve (the elegant core) |
| **N-D fixed-point / saddle finder** | `Circuit.fixed_points` / `transition_state` (jitted `_nd_kernel`) | attractors, barriers, Jacobian eigenvalues (timescales) for *any* topology |
| **LNA covariance (Lyapunov solve)** | `Circuit.mode_covariances` | per-mode covariance `AΣ+ΣAᵀ+D=0` — the channel where gain/ceiling separate |
| **Multi-operating-point fit** | `inference/lyapunov.py` `OperatingPoint`, `fit_lyapunov_multi`, `attribute_lyapunov_multi` | the **measured degeneracy-breaker** (a 2nd operating point resolves gain⇄threshold, ×16–20) |
| **Parsimony / BIC model-selection** | `inference/classify.py::switch_detected`, `inference/model_select.py` | the fail-safe "does mechanism beat the simpler null" gate |
| **Uncertainty + reliability guards** | `inference/lyapunov.py::lna_reliable`, self-distance floors, (Laplace planned) | abstention when a call is unresolved / near a bifurcation / underpowered |
| **Counts→activity bridge + readout** | `inference/bridge.py`, `mechanisms/readout` | the swap point where a new observation modality plugs in |
| **Mechanism Cards + knowledge graph + decoy battery** | `docs/mechanism_cards/`, `nudge.knowledge`, `tests/decoys/` | the machine-readable honesty layer every verdict emits into |

Two load-bearing empirical facts from `FINDINGS.md` that shape every rating:

1. **The degeneracy-breaker is a second operating point** (dose / combo / condition / time),
   measured (FIM cond# 210→22, NLL gap ×16–20). Any capability that supplies a second
   operating point inherits the breaker *for free* via the `OperatingPoint` API.
2. **Fail-safe is conditional on approximately-correct topology and an approximately-affine
   readout** (Tier-0.5 T0.5-2; NUDGE-LIM-006). Any new capability must keep an honest
   abstention path for when those conditions fail.

---

## 1. Prioritized table

Ratings: Effort (S/M/L build size on top of the substrate) · Impact (breadth × unmet-need) ·
Data (is there real, ingestible public data *today*) · Fail-safe (how clean the abstention
story is). "Anchor" = already in progress. ★ = recommended to build next.

| # | Capability | Question it answers | Effort | Impact | Real data | Fail-safe story |
|---|---|---|---|---|---|---|
| 0 | **Dose-response ultrasensitivity** (anchor) | competitive (K) vs allosteric (n) vs resistance (v_max) inhibitor | — (building) | High | **Ready** (sci-Plex GSE139944, morphogen GSE233574) | strong (dose axis = operating points) |
| 1 | **Cross-modality readout adapter** (flow / activity / protein) ★ | same K/n/v_max fit on fluorescence or reporter-activity instead of counts | **S–M** | High | **Ready, best ground truth** (Chure 2019 D1.1241: DNA-binding=K, inducer-binding=n) | good (needs a modality-aware ingest bouncer) |
| 2 | **Synergy / epistasis mechanism** (A / B / A+B) ★ | is combo synergy a threshold-, gain-, or ceiling-change; is it super-additive (hidden feedback)? | **S–M** | High | **Ready** (Norman 2019 GSE133344, one-line pertpy) | good (abstain if parts abstain / underpowered) |
| 3 | **Comparative / differential attribution** ★ | resistant-vs-sensitive / donor / cell-type: which of K, n, v_max *differs*? | **S–M** | High (clinical) | Medium (MIX-seq figshare 10298696; Gladstone 4 donors) | good but confound-prone (must pin depth/batch per context) |
| 4 | **Temporal / kinetics attribution** | did a perturbation change a **rate constant** vs a **steady-state setpoint**? | **M** | High | Medium, modality-bound (ERK-KTR S-BIAD2275 / idr0064 — activity, not counts) | good (abstain when sampling can't resolve the timescale) |
| 5 | **Bifurcation / tipping-point proximity** | how close is the switch to losing bistability (a robustness score)? | **S** (re-expose) | Medium–High | Medium (toggle+hysteresis Zenodo 11817798; dose ladders) | subtle (LNA is *weakest* exactly at the bifurcation — one-sided estimate) |
| 6 | **Multi-reporter joint attribution** (identifiability force-multiplier) | break the K⇄v_max / gain⇄threshold degeneracy with several reporters of one latent | **M** | High (lifts *all* others) | Reuses existing panels (IEG genes already in Gladstone) | strengthens the fail-safe (over-determined latent) |
| 7 | **Feedback / hidden-node detection** (positive claim) | is model rejection caused by an *unmeasured regulator*? | **M–L** | High if it worked | **Weak** (GSE114071 19–40 cells; mostly synthetic) | **asymmetric — safe to abstain, risky to positively claim → partial trap** |

---

## 2. Capability details

### Capability 0 — Dose-response ultrasensitivity attribution (the anchor / proof of the thesis)

Documented here as the *template* the rest imitate; it is already being built
(`nudge.inference.dose_response`), so it is not a "build next" item — it is the existence
proof that the substrate generalizes across screen types.

- **(a) Question / who.** Across a compound (or agonist) **dose series**, does a perturbation
  right-shift the response (competitive inhibitor → **threshold K**), flatten its slope
  (allosteric → **gain n**), or cap it (resistance / partial agonism → **ceiling v_max**)?
  Pharmacologists, chemical-genetics and MAPK/morphogen screeners.
- **(b) Reuse.** The dose axis *is* a set of `OperatingPoint`s — the exact structure
  `fit_lyapunov_multi` was built for and the FIM proved breaks gain⇄threshold. Same Hill
  fit, same parsimony gate, same cards.
- **(c) New piece.** A dose-curve ingest + a shared-kinetics-across-doses fit head. Small;
  in progress.
- **(d) Data.** sci-Plex **GSE139944** (188 compounds × 4 doses + vehicle, one-line
  `pertpy`, raw UMI — **ready**); Pașca morphogen **GSE233574** (SAG 50/250/1000/2000 nM —
  **ready**); Treutlein/Camp **E-MTAB-15667** (6 morphogens × 5 conc).
- **(e) Fail-safe.** Abstain on inert compounds (no-effect / off-model — a built-in
  specificity battery); abstain when doses don't span the inflection (a Hill fit to one arm
  of the curve is unidentifiable).
- **(f) Effort/Impact.** Anchor / High.

---

### Capability 1 — Cross-modality readout adapter (flow, reporter-activity, proteomics)  ★

**The widest-aperture, cheapest, best-data extension. Recommended #1.**

- **(a) Question / who.** Run the *identical* K/n/v_max attribution when the readout is not
  UMI counts but **single-channel fluorescence** (flow cytometry), a **live-cell activity
  reporter** (ERK-KTR, KTR-family), or **protein / phospho intensity**. Unlocks the entire
  synthetic-biology and signaling world that measures switches by fluorescence, not
  transcriptome — a population NUDGE structurally cannot serve today because its ingest
  hard-requires integer counts.
- **(b) Reuse.** *Everything* except the observation channel. The energy-distance / MMD loss
  is distribution-shape-based and already modality-agnostic; the fit, the fixed-point/saddle
  finder, the LNA, the parsimony gate, the multi-operating-point breaker, and every
  Mechanism Card are unchanged. This is the "heterogeneity is the signal" thesis (WORKING_
  BACKWARDS Part 5) with the signal carried by fluorescence instead of counts.
- **(c) New piece.** A **continuous-readout observation model** to sit beside the NB count
  model — fluorescence as a log-normal / gamma emission (with autofluorescence offset), and
  a **modality-aware ingest bouncer** so `check_counts` stops demanding integers when the
  user declares `modality="fluorescence"`. Small–Medium: the NB layer (`data/noise.py`) and
  the raw-counts guard (`data/ingest.py`) are the only count-specific code; the readout is
  already a latent→observed link (`inference/bridge.py`).
- **(d) Data shape + accessions.** Per-cell single-channel intensity + condition + control.
  **This modality has the single best ground-truth in the whole library:**
  - **Chure 2019 LacI mutants — CaltechDATA D1.1241.** DNA-binding-domain mutants perturb
    *only* DNA affinity (pure **threshold K**); inducer-binding-domain mutants perturb *only*
    allosteric sensitivity (**gain n**) — **author-decomposed ground-truth labels for exactly
    NUDGE's distinction**, over an IPTG dose × copy-number × operator matrix.
  - **Razo-Mejia 2018 — CaltechDATA D1.743** (12 IPTG doses × 3 operators × 6 copy numbers,
    tidy per-cell CSVs).
  - **Bistable toggle + hysteresis — Zenodo 11817798** (inducer dose + initial-state arms).
  - **ERK-KTR activity — BioStudies S-BIAD2275** (OptoFGFR1, 7 calibrated light doses) /
    **IDR idr0064** (429 kinase inhibitors) — feeds *both* this and Capability 4.
- **(e) Fail-safe.** The one new risk: the raw-counts bouncer must not silently accept
  *log-normalized counts* (which would break everything) while accepting genuine
  fluorescence. Solution: an explicit `modality` flag; abstain/refuse on ambiguous input.
  NUDGE-LIM-006 (readout nonlinearity) applies *more* here — a nonlinear fluorescent reporter
  (e.g. a saturating FRET sensor) can manufacture apparent ultrasensitivity — so the
  constitutive-control mitigation (`design/CONSTITUTIVE_CONTROL.md`) and abstention on the
  circuit-vs-readout axis carry over directly.
- **(f) Effort/Impact.** Small–Medium / High. **Because Chure 2019 provides labelled K-vs-n
  ground truth, this is also the single most convincing possible validation demo** (Demo is
  the weakest, highest-weight judging criterion) — NUDGE recovering "DNA-binding mutant →
  threshold, inducer-binding mutant → gain" against an author-provided answer key.

---

### Capability 2 — Synergy / epistasis mechanism (A / B / A+B)  ★

**Reuses the measured degeneracy-breaker almost verbatim; one-line-ready data. Recommended #2.**

- **(a) Question / who.** For a combination (two drugs, two genetic perturbations, drug×gene),
  classify the *interaction* mechanism: does A+B shift the switch **additively** (same knob,
  more of it), or does it move a *different* knob than A and B predict — a **super-additive
  gain/ceiling change** that signals rewiring or hidden feedback? Combination-therapy teams,
  genetic-interaction (GI) mappers, chemical-genetics.
- **(b) Reuse.** A, B, A+B are three **conditions / operating points**. `fit_lyapunov_multi`
  already fits shared kinetics across operating points; here the shared-vs-free structure
  encodes the **additive null** (A+B's edge-parameter deltas = compose(A, B)) versus a
  **synergy alternative** (A+B needs its own K/n/v_max). BIC model-selection
  (`model_select.py`) already does exactly this nested comparison. The parsimony gate ensures
  we only *call* synergy when it beats additive by a margin surviving uncertainty. Super-
  additivity that *no* single-edge parameterization explains is the **bridge to hidden-node
  detection** (Capability 7).
- **(c) New piece.** An **additivity baseline model** (compose two fitted perturbations in
  parameter space) + a combo-attribution head that BIC-selects additive vs synergistic and,
  if synergistic, attributes the *interaction* to threshold/gain/ceiling. Small–Medium — the
  perturbation composition already exists (perturbations modify edge params); the null is a
  new assembled circuit, the selection is existing machinery.
- **(d) Data.** **Norman 2019 — GSE133344**: 105 single + 131 two-gene (A/B/A+B) CRISPRa
  activations + non-targeting controls, ~91k K562 cells, **one-line `pertpy.data.norman_2019`,
  raw counts — ready and small.** The cleanest combination sandbox in existence. Scale-up:
  sci-Plex-GxE **GSE225775** (gene×drug, >1M cells — heavy/stretch).
- **(e) Fail-safe.** Abstain on the combo whenever *either* single arm abstains (you cannot
  attribute an interaction whose components you cannot call); non-additivity with overlapping
  posteriors → **unresolved**, not a synergy claim; underpowered combos (few cells) widen and
  abstain. All three land on the existing vocabulary.
- **(f) Effort/Impact.** Small–Medium / High. Makes the abstract "multi-operating-point
  breaker" concrete for a question thousands of labs ask (is my combo synergistic, and how).

---

### Capability 3 — Comparative / differential attribution (resistant vs sensitive, donor, cell type)  ★

**Clinically the largest audience; reuses OperatingPoint + BIC. Recommended #3.**

- **(a) Question / who.** Given the *same* perturbation in two **contexts** (drug-resistant vs
  sensitive line; donor A vs B; cell type X vs Y; disease vs healthy), is the mechanistic
  difference located in **K, n, or v_max**? A resistant line with a *raised ceiling* needs
  more dose of the same drug; one with a *rewired gain/threshold* needs a different drug
  class — a decision linear differential expression structurally cannot make. Translational /
  drug-resistance / precision-oncology teams; anyone doing donor-variability or cross-tissue
  comparison.
- **(b) Reuse.** Fit both contexts jointly with a **shared-vs-per-context** parameter
  structure (the `OperatingPoint` API, but the *difference* is the target, not a nuisance),
  then **BIC-select which single parameter must differ** to explain the two contexts. The
  reliability guards (`lna_reliable`) and depth-pinning (`calibrate_from_wt`) apply per
  context. Uncertainty on the difference is the "is it real" test.
- **(c) New piece.** A **two-context differential head**: enumerate {shared, ΔK-only, Δn-only,
  Δv_max-only} nested models and BIC-select. Small–Medium — it is model-selection over the
  existing restricted fits, one axis wider.
- **(d) Data.** **MIX-seq — figshare 10298696** (trametinib / idasanutlin across pools of
  24–100+ cancer lines demultiplexed by SNP; sensitive vs resistant backgrounds, dose × time
  — needs metadata wrangling). Gladstone **GSE314342** (4 donors — donor as the context axis).
  sci-Plex cell lines (A549/MCF7/K562). Medium accessibility.
- **(e) Fail-safe.** The sharp risk is **confounding**: a depth or batch difference between
  contexts mimics a mechanism difference (this is literally the "batch aligned with
  perturbation" decoy). Mitigation: pin sequencing depth *per context* from each context's own
  control, and abstain when depth/batch cannot be separated from the context axis. Abstain
  when either context is underpowered.
- **(f) Effort/Impact.** Small–Medium / High. The "raised ceiling vs rewired gain" resistance
  call is a genuinely clinical, linear-analysis-proof distinction.

---

### Capability 4 — Temporal / kinetics attribution (rate constant vs steady-state setpoint)

**High-impact, but data is modality-bound — strongest coupled to Capability 1.**

- **(a) Question / who.** From **time-resolved** perturbation data, did a perturbation change
  a **rate constant** (how fast the system relaxes) or the **steady-state setpoint** (where it
  settles)? These are *invisible to each other at steady state*: a snapshot constrains only
  the setpoint (ratios of rates), so a pure kinetic change is unidentifiable from a snapshot —
  the FIM/Jacobian-timescale degeneracy that the analysis flagged. Signaling biologists
  (adaptation vs gain), anyone with live-imaging reporters or stimulation time courses.
- **(b) Reuse.** The **Jacobian eigenvalues at the fixed point are the relaxation timescales**
  — and `Circuit.mode_covariances` / `_lna_covariance` already build that Jacobian `A`. The
  setpoint is `Circuit.fixed_points`. Time is another **operating-point axis** the
  `OperatingPoint` machinery can hold.
- **(c) New piece.** A **transient forward solve** returning a trajectory (the current fit uses
  `steady_state` only) + a **decay/rate free-parameter** in the restricted-fit menu (today's
  free params are K/n/v_max, all setpoint-side). Medium: the vector field exists; add a short
  fixed-step or diffrax rollout and a rate-vs-setpoint model-selection head.
- **(d) Data shape + accessions.** True single-cell **trajectories** (not destructive
  snapshots): **ERK-KTR live imaging — BioStudies S-BIAD2275** (OptoFGFR1, 7 light doses, time
  series) / **IDR idr0064** (429 kinase inhibitors) — **but the readout is activity, not
  counts**, so this *requires Capability 1's adapter first*. Population-moment time courses from
  destructive snapshots (Gladstone Rest/Stim8/Stim48; Waddington-OT **GSE122662**) are a
  weaker, pseudo-trajectory fallback (moments over time, not per-cell traces).
- **(e) Fail-safe.** Abstain when the **sampling interval cannot resolve the timescale** (a
  Nyquist-like guard: if Δt ≫ 1/|Re λ|, the rate is unidentifiable — report it, don't guess);
  abstain on steady-state-only data (no kinetic information present at all). Clean, principled.
- **(f) Effort/Impact.** Medium / High — but honestly gated by (1) the activity-readout adapter
  and (2) the scarcity of true single-cell transcriptomic trajectories (destructive assay).
  Build *after* Capability 1; the live-imaging ERK data is where it pays off.

---

### Capability 5 — Bifurcation / tipping-point proximity (a robustness readout)

**Cheapest of all (re-expose an existing internal), novel — but the UQ caveat is real.**

- **(a) Question / who.** How close is a bistable switch to **losing bistability** (a
  saddle-node / fold)? A scalar **robustness / "flippability" score**: is this switch a hair-
  trigger cliff or a well-buffered dial? Resilience and critical-transition biology (aging,
  disease progression, cell-fate commitment), engineered-circuit robustness QA.
- **(b) Reuse.** `Circuit.fixed_points` already returns the stable nodes *and* the saddle with
  Jacobian-index classification. Proximity to a saddle-node is directly readable as (i) the
  **smallest-magnitude real part of the Jacobian eigenvalue** (→ 0 at the fold), (ii) the
  **state-space distance from a stable node to the saddle** (basin depth → 0 at the fold), or
  (iii) the LNA variance swell. **`lna_reliable` already computes exactly this** ("near a
  saddle-node → a lobe's CV > 1.5") as an *abstention trigger* — this capability re-exposes
  the same quantity as a *result*.
- **(c) New piece.** A `bifurcation_proximity(circuit) -> (score, one-sided interval)`
  reporter + calibration of what the number means. Small — it re-uses internals; the work is
  honest UQ, not new math.
- **(d) Data.** Any bistable system with a dose axis approaching the fold: toggle + hysteresis
  **Zenodo 11817798**, morphogen ladders **GSE233574**, OCT4-exit **GSE283614**. Grounding:
  critical-slowing-down / early-warning-signal literature (rising variance + autocorrelation
  near tipping points).
- **(e) Fail-safe.** The honest catch: **the LNA Gaussian breaks down *precisely* at the
  bifurcation** (variance diverges, the linear-noise approximation fails) — the estimate is
  least reliable exactly where it matters most. Must be reported as a **one-sided /
  lower-bound** estimate ("at least this close") and abstain on the far (deep-basin) side
  where a Gaussian lobe carries no fold information. Subtle but statable.
- **(f) Effort/Impact.** Small / Medium–High. A demoable, evocative "robustness dial," but
  keep the UQ honesty front-and-center or it becomes an overclaim.

---

### Capability 6 — Multi-reporter joint attribution (identifiability force-multiplier)

**Not a new *question* — a lever that strengthens every capability above and the fail-safe itself.**

- **(a) Question / who.** The K⇄v_max and gain⇄threshold degeneracies (measured, `FINDINGS.md`
  §2, Fisher analysis) are the dominant reason NUDGE abstains. `FINDINGS.md` repeatedly names
  "a richer multi-reporter readout" as the candidate fix. Whoever wants *more resolved* calls
  (fewer abstentions) rather than a new screen type.
- **(b) Reuse.** Fit **several downstream reporter genes jointly** as multiple `Readout`
  instances of the *same* latent switch (each with its own gain/offset). The shared latent is
  then **over-determined** — a threshold shift and a ceiling change, degenerate through one
  reporter, project differently onto a *panel* of reporters. The vmap population model and the
  energy-distance loss extend to a joint (multi-gene) distribution unchanged.
- **(c) New piece.** A **multi-readout observation model** (shared latent → several emissions)
  + a joint distributional loss over the gene panel. Medium.
- **(d) Data.** Reuses panels that already exist — the Gladstone IEG panel (IL2/CD69/EGR1/FOS/
  NR4A1) is five reporters of one activation latent; every dataset in the library with a
  multi-gene readout qualifies. No new download.
- **(e) Fail-safe.** Strictly *strengthens* it — an over-determined latent means a spurious
  mechanism must now be consistent across all reporters, which is harder to fake; inconsistency
  across reporters is itself an off-model signal (a reporter reading a *different* latent flags
  a hidden node / wrong panel).
- **(f) Effort/Impact.** Medium / High — because it lifts the resolved-call rate of *all* the
  attribution capabilities, it is the highest-leverage internal investment even though it
  ships no new "screen type."

---

### Capability 7 — Feedback / hidden-node detection (the positive claim)  ⚠ partial trap

**A great abstention story that already ships; a *risky positive claim* with poor real data.**

- **(a) Question / who.** When a simple model is consistently rejected, is the cause an
  **unmeasured regulator or feedback loop** — computational evidence of a hidden node whose
  identity a biologist can then chase? NUDGE's most differentiated *aspirational* claim.
- **(b) Reuse.** The pieces exist: the **off-model parsimony tripwire** (`switch_detected`),
  **BIC topology selection** (1-node vs 2-node vs no-switch, `model_select.py`), and the
  **LNA covariance residual** — a hidden node leaves a covariance structure ("comet-tail") a
  closed low-D ODE cannot reproduce (`mode_covariances` predicted vs empirical). The N-D
  saddle finder makes fitting a *candidate* extra node approachable.
- **(c) New piece.** A **hidden-node test statistic** that discriminates "off-model because of
  a hidden node" from the *other* causes of off-model (bad readout / off-target / batch). This
  discrimination is the hard, risky part — Medium–Large.
- **(d) Data — the honest gap.** GSE114071 (miRNA-mRNA co-seq) co-measures the hidden node as
  an answer key **but has only 19–40 cells** → validation, not attribution. The canonical
  hidden-node synthetic-biology systems (Mukherji 2011, Bleris 2011, ceRNA sponges) have **no
  ingestible single-cell public data** (`NOTEBOOK_LIBRARY.md` C1). This capability is almost
  entirely **synthetic-data-bound** today.
- **(e) Fail-safe — the trap.** The abstention direction is *safe and already shipped*: NUDGE
  correctly returns off-model when its model is inadequate. But the **positive** claim ("there
  is a hidden node *here*") is **weakly identifiable** — off-model is a degenerate verdict
  (NUDGE-LIM-006 nonlinear readout, off-target effects, batch confounds, and a genuine hidden
  node *all look alike*). Claiming a hidden node from an off-model verdict risks exactly the
  confident-wrong-answer failure the whole project exists to avoid.
- **(f) Effort/Impact / verdict.** Medium–Large / High-if-it-worked. **Recommendation: ship
  the abstention half (done — off-model *is* the honest output) and treat positive hidden-node
  *identification* as a research direction, not a product claim, until (i) a discriminating
  statistic is validated and (ii) real co-measured data exists.** Flagged as a partial trap so
  a future session does not overclaim it.

---

## 3. The three to build next (and why)

1. **Cross-modality readout adapter (Capability 1).** Cheapest structural change (a readout
   swap + a modality-aware bouncer), widest new audience (all of flow-cytometry synthetic
   biology + activity-reporter signaling), and — decisively — it unlocks **Chure 2019's
   author-labelled K-vs-n ground truth**, the most convincing validation/demo available and a
   direct hit on the weakest, highest-weight judging criterion (Demo). It also *enables*
   Capability 4.
2. **Synergy / epistasis (Capability 2).** Reuses the *measured* degeneracy-breaker
   (`OperatingPoint` + BIC) almost verbatim, runs on **one-line-ready Norman 2019 data**, and
   answers a question thousands of combination-therapy and GI labs ask.
3. **Comparative / differential attribution (Capability 3).** Largest clinical audience
   (drug resistance: "raised ceiling vs rewired gain" is a real treatment decision linear
   analysis cannot make), and it is model-selection one axis wider than what already ships.

Runner-up: **Bifurcation proximity (Capability 5)** is the cheapest single item (re-expose
`lna_reliable`'s internal) and demoable, but its UQ-near-the-fold caveat needs care — worth a
quick spike, not a headline. **Multi-reporter (Capability 6)** is the highest-leverage
*internal* investment (it raises the resolved-call rate of everything) but ships no new screen
type, so it competes for the same effort budget.

## 4. Traps / honest cautions

- **Hidden-node *positive* detection (Capability 7)** — sounds like NUDGE's coolest claim, but
  the positive call is weakly identifiable (off-model has many causes) and there is essentially
  **no real single-cell data with a known hidden node** (19–40 cells is not enough). Safe to
  abstain, risky to assert. Ship the abstain half only.
- **Temporal attribution (Capability 4)** — genuinely valuable, but true single-cell
  *trajectories* barely exist in transcriptomics (the assay is destructive); it is really a
  *live-imaging / activity-reporter* capability and should be built only *after* the
  cross-modality adapter, on ERK-KTR data.
- **Any capability whose "second operating point" is actually a confound** (Capability 3's
  batch/depth, Capability 5's near-fold LNA breakdown) — the breaker only helps if the second
  axis is *clean*. Pin depth per context and keep the reliability guards; abstain otherwise.
- **The two standing conditions on the whole fail-safe guarantee** (approximately-correct
  topology, approximately-affine readout) apply to every capability here — each must carry the
  BIC topology gate and the constitutive-control / abstain-on-readout-axis mitigation forward,
  not silently assume them away.
