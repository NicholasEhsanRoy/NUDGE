# NUDGE — Capability Implementation Plans + Day 3–7 Schedule

**Status: planning doc.** Written 2026-07-09 (Day 2 of the 7-day hackathon), the day after
Capability 0 (dose-response) shipped. This turns `design/CAPABILITY_ROADMAP.md`'s 8-capability
menu into concrete, codebase-grounded build plans — each mirroring the exact pattern the
dose-response anchor established — plus a prioritized 5-day execution schedule (Days 3–7).

The honesty rule (CLAUDE.md) governs every line: a plan never proposes overclaiming. Where a
capability's data is weak or its positive claim is unidentifiable, that is flagged loudly and
folded into the abstention design — most sharply for Capability 7, where **only the abstention
half is planned** (the positive "there is a hidden node here" claim stays a research direction).

Companion docs: `design/CAPABILITY_ROADMAP.md` (WHAT each capability is), `design/NOTEBOOK_
LIBRARY.md` (real datasets + accessions), `scripts/vv/FINDINGS.md` (the measured identifiability
facts the ratings lean on — esp. that a *second operating point* breaks gain⇄threshold ×16).

---

## 0. The template every plan follows (Capability 0, dose-response)

Capability 0 defines the **definition of done** and the exact code shape each capability copies.
Studying it once fixes the pattern:

**Module** (`src/nudge/inference/dose_response.py`) exposes three functions in a fixed shape,
which every capability below mirrors:
- `fit_*(...) -> *Fit` — a differentiable fit reusing NUDGE primitives (here the exact circuit
  Hill primitive + an autodiff Jacobian), returning a frozen dataclass carrying **everything the
  classifier needs** (point estimate, bootstrap CI, BIC of the null sibling, reliability flags).
- `classify_*(fit, ...) -> (call, reason)` — a **fail-safe classifier**: gates in order, most
  conservative first (no-effect → unresolved/abstain → the positive call only when it *earns* its
  parameter by a BIC margin **and** the CI clears the line). Returns a human-readable `reason`.
- `attribute_*(...) -> *Result` — fit + classify in one call; the CLI/MCP entry point.

**Data→(x, y) bridge** (`inference/bridge.py::knockdown_dose_response`) — extracts the
`(dose, response)` pairs from a real `.h5ad`, depth-normalized (size-factor to median) so per-cell
sequencing depth is divided out. Reuses `_norm_counts` / `counts_to_activity`.

**Shared service layer** (`service.py::dose_response_file` + `_dose_points` +
`dose_response_to_dict`) — the one place CLI and MCP share, so both give byte-identical output.
Wired into the CLI verb (`cli.py::dose_response`) and the MCP tool (`mcp/server.py::dose_response`).

**Mechanism Card** (`docs/mechanism_cards/dose_response_attribution.md`, `NUDGE-METHOD-001`) —
machine-checkable front-matter (`id`, `name`, `role: attribution-method`, `registry_name`,
`vulnerable_to_decoys`, `documented_limitation`, `validated_in_regime`, `references`) + fixed body
(governing equation, classifier, assumptions, failure-mode table, identifiability regime,
Implementation Mapping, verification evidence). Validated by `check_mechanism_cards.py`,
`check_impl_mapping.py` (every `nudge.*` qualname must resolve), `check_citations.py` (every bib
key must resolve). Registered via the `mechanism-card` skill; needs a registry entry so
`nudge mechanisms` lists it.

**Limitation** (`docs/known_limitations.yaml`, `NUDGE-LIM-007`) — a NUDGE-LIM-NNN entry validated
by `check_anomalies.py`, cross-referenced from the card's `documented_limitation`.

**Tests** (`tests/inference/test_dose_response.py`) — synthetic unit tests with known ground truth
(a genuine switch reads switch; an n≈1 curve reads graded; a one-arm curve abstains; a flat curve
is no-effect) + a `@pytest.mark.needs_data` **real-data lock-in** + a **regression** pinning a
specific bug (the float32 finite-difference Jacobian that froze `n`).

**Demo notebook** (`notebooks/OCT4_NANOG_Flagship.ipynb`) — built with `nbformat`, executed
headless with `nbconvert` to embed outputs; tells one story with one honest positive + one honest
abstention.

**Reusable substrate every capability draws on** (skim, don't rebuild):
- `core/circuit.py`: `Circuit.fixed_points` (stable nodes + index-1 saddle + Jacobian labels),
  `transition_state`, `vector_field`, `mode_covariances` (`AΣ+ΣAᵀ+D=0`), `solve_population`
  (vmap), `steady_state`.
- `inference/lyapunov.py`: `OperatingPoint`, `attribute_lyapunov_single`, `fit_lyapunov_multi` /
  `attribute_lyapunov_multi` (the **measured degeneracy-breaker**), `calibrate_from_wt`,
  `lna_reliable` (the abstention guard: low depth / near-fold CV>1.5 / monostable).
- `inference/model_select.py::select_topology` (BIC parsimony over circuit candidates + no-switch
  null), `inference/classify.py::switch_detected` / `decide` / `decide_with_transition`.
- `inference/pipeline.py::attribute_across_operating_points` (label→AnnData orchestration + honest
  `skipped`).
- `inference/bridge.py` (counts→activity + `adata_to_operating_point`), `mechanisms/regulatory.py`
  (Hill primitives), `circuits.py` (`ras_switch_1node/2node`, `toggle`), `data/ingest.py`
  (`check_counts`), `data/decoy_generators.py`.

**Two load-bearing empirical facts** (FINDINGS.md) shape every fail-safe: (1) a **second operating
point** breaks the gain⇄threshold degeneracy (FIM cond# 210→22, NLL gap ×16–20) — any capability
supplying one inherits the breaker free via the `OperatingPoint` API; (2) the fail-safe is
**conditional on approximately-correct topology + an approximately-affine readout** (T0.5-2,
NUDGE-LIM-006) — every capability must carry the BIC topology gate + the constitutive-control /
abstain-on-readout-axis mitigation forward, never assume them away.

> **LIM ids.** Next free id is **NUDGE-LIM-008**. I assign 008–014 to Capabilities 1–7 below for
> readability; the actual number a capability lands with follows build order, so treat these as
> provisional and allocate at merge time.

---

## Capability 1 — Cross-modality readout adapter (flow / activity / protein) ★

**(a) Bio question + who.** Run the *identical* K/n/v_max attribution when the readout is
single-channel **fluorescence** (flow), a live-cell **activity reporter** (ERK-KTR), or **protein
intensity** — not UMI counts. Unlocks the entire synthetic-biology + signaling world that measures
switches by fluorescence (a population NUDGE structurally cannot serve today because ingest
hard-requires integer counts). Decisive lever: it unlocks **Chure 2019's author-labelled K-vs-n
ground truth** — the single most convincing validation demo available.

**(b) Primitives reused.** *Everything except the observation channel.* The Chure benchmark is a
dose × mutant fluorescence matrix, so the core path reuses **`dose_response.fit_dose_response` /
`classify_dose_response` / `attribute_dose_response` verbatim** — a mutant's fluorescence-vs-IPTG
curve is a dose-response, and DNA-binding→K vs inducer-binding→n is exactly its switch-vs-threshold
axis. Also reuses `bridge._norm_counts` conceptually (a fold-change normalization), and — for the
single-cell-fluorescence-distribution stretch — `Circuit.mode_covariances` + the energy-distance
loss (already distribution-shape-based, modality-agnostic).

**(c) New code.**
- `src/nudge/data/ingest.py` — extend the guard: `check_readout(adata_or_df, *, modality="counts",
  readout_col=None)` routing `modality="counts"` to the existing `check_counts`, and
  `"fluorescence"/"activity"` to a **continuous-readout bouncer** (finite, non-negative; **refuse
  ambiguous input** — the sharp risk is silently accepting *log-normalized counts*, which would
  break everything; refuse anything integer-quantized-then-logged, require an explicit `modality`
  flag). ~45 LOC.
- `src/nudge/inference/bridge.py` — add `fluorescence_dose_response(df, *, dose_col, intensity_col,
  variant, control_label=None, autofluor=None) -> tuple[np.ndarray, np.ndarray]`: per-dose
  summary (median fold-change over the control, minus an autofluorescence offset) for one variant.
  Mirrors `knockdown_dose_response`'s shape. ~55 LOC.
- `service.py` — `dose_response_file` gains a `modality` + `intensity_col` path (CSV of per-cell or
  per-dose fluorescence → `(dose, response)` via the new extractor, then the existing
  `attribute_dose_response`). CLI `--modality` flag + MCP param. ~35 LOC.
- Card `docs/mechanism_cards/cross_modality_readout.md` (`NUDGE-METHOD-002`, registry
  `CrossModalityReadout`) + registry entry. LIM-008.

  Public API (mirrors the anchor; the fit/classify are *reused*, so the new surface is the
  extractor + bouncer):
  ```python
  def check_readout(x, *, modality="counts", readout_col=None) -> None            # ingest.py
  def fluorescence_dose_response(df, *, dose_col, intensity_col, variant,
                                 control_label=None, autofluor=None
                                 ) -> tuple[np.ndarray, np.ndarray]                # bridge.py
  # then reuse: attribute_dose_response(dose, response, direction="activate")
  ```

**(d) Data.** **Chure 2019 LacI mutants — CaltechDATA D1.1241** (DNA-binding mutants = pure K,
inducer-binding mutants = gain n; author-decomposed labels; IPTG dose × copy-number × operator).
Tidy per-condition CSVs → **light download, no FCS parsing needed** (the "adapter" is the
continuous bouncer + the fold-change extractor, not a flow-cytometry reader). Secondary: **Razo-
Mejia 2018 D1.743** (12 doses × 3 operators × 6 copies, `flow_master.csv`). Ingest: **needs-work
but small** (a tidy-CSV adapter, not a one-line pertpy fetch).

**(e) Fail-safe / abstention.** Inherits the whole dose-response classifier (no-effect / unresolved
/ graded / switch). **NUDGE-LIM-008 (new):** the modality bouncer must refuse *ambiguous /
log-normalized* input rather than silently fit it — the one new failure mode. Inherits
**NUDGE-LIM-006** *more strongly*: a nonlinear fluorescent reporter (saturating FRET/YFP) can
manufacture apparent ultrasensitivity, so the constitutive-control mitigation
(`design/CONSTITUTIVE_CONTROL.md`) and abstain-on-circuit-vs-readout-axis carry over — abstain when
the reporter's linearity is uncontrolled.

**(f) Tests.** Synthetic: a known fluorescence Hill curve (autofluor offset + log-normal
measurement noise) reads `switch`; a log-normalized-counts array is **refused** by
`check_readout(modality="fluorescence")`. Real-data lock-in (`needs_data`): Chure D1.1241 —
DNA-binding mutants → threshold (K shifts, n≈const), inducer-binding mutants → gain (n changes),
matching the author labels; abstain where copy-number/operator degeneracy bites.

**(g) Demo notebook.** `notebooks/Chure_LacI_Benchmark.ipynb` — "NUDGE recovers *DNA-binding mutant
→ threshold, inducer-binding mutant → gain* against an author-provided answer key" (a K-vs-n
scatter coloured by mutant class vs the published decomposition).

**(h) Effort.** **S–M, ~8–10 h.** No hard dependency (reuses the shipped dose-response module).
**Enables Capability 4** (activity readout). Highest Demo leverage in the menu.

**(i) Honesty flags.** The Chure path is dose-response over fluorescence — strong and honest. The
**single-cell-fluorescence-distribution LNA path** (log-normal/gamma emission beside the NB model)
is a genuine new observation model and is scoped as a **stretch**, not core — do not claim it until
built. The modality bouncer is a *refusal*, not a converter: NUDGE never guesses a modality.

---

## Capability 2 — Synergy / epistasis mechanism (A / B / A+B) ★

**(a) Bio question + who.** For a combination (two genes, two drugs, drug×gene), is the interaction
**additive** (same knob, more of it) or **super-additive** — A+B moving a *different* knob
(threshold/gain/ceiling) than A and B predict, signalling rewiring or hidden feedback?
Combination-therapy teams, genetic-interaction (GI) mappers, chemical-genetics.

**(b) Primitives reused.** A, B, A+B are three **operating points** — the exact structure
`fit_lyapunov_multi` was built for. `adata_to_operating_point` builds each condition's
`OperatingPoint`; `attribute_lyapunov_single` calls each single arm; **`select_topology` /
`_bic`-style BIC** does the nested additive-vs-synergistic comparison; `lna_reliable` guards each
arm. The perturbation composition already exists (perturbations modify edge params), so the
additive null is an assembled circuit, not new math.

**(c) New code.**
- `src/nudge/inference/epistasis.py` (~180 LOC):
  ```python
  @dataclass(frozen=True)
  class EpistasisFit:      # single-arm calls, additive-null NLL/BIC, combo free-param NLL/BIC, CIs
  @dataclass(frozen=True)
  class EpistasisResult:   # fit + call + reason

  def fit_additive_null(fit_a, fit_b, circuit, *, target_edge=0) -> AdditiveModel
      # compose A's and B's fitted edge-param deltas → predicted A+B kinetics (the null)
  def classify_synergy(combo_fit, additive_null, *, bic_margin=2.0
                       ) -> tuple[str, str]
      # additive | synergistic-threshold | synergistic-gain | synergistic-ceiling
      # | no-effect | unresolved  (BIC-select additive vs a free combo K/n/vmax)
  def attribute_epistasis(op_a, op_b, op_ab, circuit, *, target_edge=0,
                          ...) -> EpistasisResult
  ```
- `bridge.py` — small helper `combo_operating_points(adata, gene_a, gene_b, ...)` returning the A,
  B, A+B, WT `OperatingPoint`s from Norman-style condition labels. ~40 LOC.
- `service.py::epistasis_file` + `epistasis_to_dict`; CLI `nudge epistasis` verb; MCP `epistasis`
  tool. ~60 LOC.
- Card `epistasis_attribution.md` (`NUDGE-METHOD-003`, `SynergyAttribution`) + registry entry;
  LIM-009.

**(d) Data.** **Norman 2019 — GSE133344**, one-line `pertpy.data.norman_2019`, raw counts, ~91k
K562 cells, 105 single + 131 two-gene CRISPRa + NTC controls. **Ready and small** — the cleanest
combination sandbox in existence. Scale-up (stretch): sci-Plex-GxE GSE225775 (>1M cells, heavy).

**(e) Fail-safe / abstention.** **Abstain on the combo whenever *either* single arm abstains** (you
cannot attribute an interaction whose components you cannot call); non-additivity with overlapping
posteriors → `unresolved`, never a synergy claim; underpowered combos (few cells) widen and abstain
via `lna_reliable` + the min-cells gate. **NUDGE-LIM-009 (new):** combo attribution inherits the
weakest single arm; a super-additive residual is *not* by itself a hidden-node claim (that bridges
to Capability 7's trap — keep them separate). Inherits NUDGE-LIM-006 (readout) and the topology gate.

**(f) Tests.** Synthetic (known ground truth via the generator composing two edge-param deltas): an
additive pair reads `additive`; a pair whose A+B needs a distinct gain reads `synergistic-gain`; a
pair where one arm is a dead guide reads `unresolved` (arm abstains). Real-data lock-in
(`needs_data`): a Norman 2019 coactivated pair vs a non-interacting pair.

**(g) Demo notebook.** `notebooks/Norman_Synergy.ipynb` — "Is this CRISPRa combo just more of the
same knob, or a rewired gain? NUDGE calls additive vs synergistic-{K,n,vmax} — or abstains."

**(h) Effort.** **S–M, ~8–10 h.** No dependency (reuses the multi-operating-point breaker
verbatim). Data is the most turnkey in the menu.

**(i) Honesty flags.** The additive null presumes the two single arms are correctly attributed and
the topology is approximately right — surface both. Genetic (CRISPRa on/off) perturbations, not
graded drug doses, so it illustrates combination *logic*, not a literal dose combo — say so.

---

## Capability 3 — Comparative / differential attribution (resistant vs sensitive, donor, cell type) ★

**(a) Bio question + who.** Same perturbation in two **contexts** (resistant vs sensitive line;
donor A vs B; cell type X vs Y): is the mechanistic difference in **K, n, or v_max**? A resistant
line with a *raised ceiling* needs more dose of the same drug; one with *rewired gain/threshold*
needs a different drug class — a clinical decision linear DE structurally cannot make. Translational
/ drug-resistance / precision-oncology teams.

**(b) Primitives reused.** Fit both contexts with a **shared-vs-per-context** parameter structure
via `fit_lyapunov_multi` (but the *difference* is the target, not a nuisance); **BIC-select which
single parameter must differ** using `model_select`/`_bic`. `calibrate_from_wt` pins depth **per
context** (from each context's own control); `lna_reliable` guards each.

**(c) New code.**
- `src/nudge/inference/differential.py` (~170 LOC):
  ```python
  @dataclass(frozen=True)
  class DifferentialFit:    # per-model BIC for {shared, ΔK, Δn, Δvmax}, CIs on the winning delta
  @dataclass(frozen=True)
  class DifferentialResult: # fit + call + reason

  def fit_differential(ctx1: OperatingPoint, ctx2: OperatingPoint, circuit,
                       *, target_edge=0, steps=200, seed=0) -> DifferentialFit
      # nested fits: all-shared null vs one-parameter-free-per-context alternatives
  def classify_differential(fit, *, bic_margin=2.0) -> tuple[str, str]
      # same-mechanism | differs-threshold | differs-gain | differs-ceiling | unresolved
  def attribute_differential(...) -> DifferentialResult
  ```
- `bridge.py` — `context_operating_points(adata, target, context_col, ...)` → two `OperatingPoint`s
  with per-context WT calibration. ~40 LOC.
- `service.py::differential_file` + dict; CLI `nudge differential`; MCP tool. ~60 LOC.
- Card `differential_attribution.md` (`NUDGE-METHOD-004`, `DifferentialAttribution`) + registry;
  LIM-010.

**(d) Data.** **MIX-seq — figshare 10298696** (trametinib/idasanutlin across sensitive vs resistant
lines, dose × time; needs demultiplex-metadata wrangling). Alternatives: **Gladstone GSE314342** (4
donors = context axis), **sci-Plex GSE139944** cell lines (A549 vs MCF7 trametinib — MCF7 is a
built-in true negative). **Medium** accessibility.

**(e) Fail-safe / abstention.** The sharp risk is **confounding**: a depth/batch difference between
contexts mimics a mechanism difference (the "batch aligned with perturbation" decoy).
**NUDGE-LIM-010 (new):** pin sequencing depth *per context* from each context's own control, and
**abstain when depth/batch cannot be separated from the context axis** or either context is
underpowered. Inherits the topology + readout gates.

**(f) Tests.** Synthetic two-context (only K differs; only n differs; nothing differs → `same-
mechanism`; a depth-confounded pair → `unresolved`). Real-data lock-in (`needs_data`): A549 vs MCF7
trametinib in sci-Plex (or two Gladstone donors).

**(g) Demo notebook.** `notebooks/Differential_Resistance.ipynb` — "Resistant vs sensitive: is it a
raised ceiling (more dose) or a rewired gain (different drug class)? NUDGE localizes the difference
to one knob — or abstains on a confounded axis."

**(h) Effort.** **S–M, ~9–11 h.** No hard dependency (model-selection one axis wider than what
ships). Data wrangling (MIX-seq metadata) is the main cost — prefer sci-Plex A549/MCF7 to stay
one-line.

**(i) Honesty flags.** Confound is the whole ballgame — the demo must *show* the depth-pinning and
an abstention on a deliberately confounded axis, or it overclaims. Two contexts = two operating
points, so gain⇄threshold can resolve, but only if the axis is clean.

---

## Capability 4 — Temporal / kinetics attribution (rate constant vs steady-state setpoint)

**(a) Bio question + who.** From **time-resolved** data, did a perturbation change a **rate
constant** (how fast the system relaxes) or the **setpoint** (where it settles)? These are invisible
to each other at steady state — a snapshot constrains only the setpoint, so a pure kinetic change is
unidentifiable from one snapshot (the Jacobian-timescale degeneracy). Signaling biologists
(adaptation vs gain), live-imaging / stimulation-time-course users.

**(b) Primitives reused.** The **Jacobian eigenvalues at the fixed point are the relaxation
timescales** — `Circuit.mode_covariances` / `_lna_covariance` already build the drift Jacobian `A`;
the setpoint is `Circuit.fixed_points`. Time is another **operating-point axis** the `OperatingPoint`
machinery holds. New: a **transient forward solve** (the current fit uses `steady_state` only).

**(c) New code.**
- `src/nudge/inference/kinetics.py` (~200 LOC):
  ```python
  def fit_kinetics(times, trajectory, circuit, *, target_edge=0, ...) -> KineticsFit
      # short fixed-step (or diffrax) rollout of vector_field; fit a rate free-param
      # + a setpoint free-param; return per-model BIC + the Nyquist ratio Δt·|Re λ|
  def classify_kinetics(fit, *, min_nyquist=...) -> tuple[str, str]
      # rate-change | setpoint-change | unresolved | no-effect
  def attribute_kinetics(...) -> KineticsResult
  ```
- New forward primitive: a `Circuit.solve_trajectory(x0, params, times)` short rollout (or reuse
  `solve_population` per timepoint). ~50 LOC in `core/circuit.py`.
- service/CLI (`nudge kinetics`)/MCP + card `kinetics_attribution.md` (`NUDGE-METHOD-005`) +
  registry; LIM-011. ~90 LOC.

**(d) Data.** True single-cell **trajectories**: **ERK-KTR — BioStudies S-BIAD2275** (OptoFGFR1, 7
light doses, time series) / **IDR idr0064** (429 kinase inhibitors) — but the readout is **activity,
not counts**, so this **requires Capability 1's adapter first**. Weaker pseudo-trajectory fallback:
population-moment time courses (Gladstone Rest/Stim8/Stim48; Waddington-OT GSE122662).

**(e) Fail-safe / abstention.** Abstain when the **sampling interval cannot resolve the timescale**
(a Nyquist-like guard: Δt ≫ 1/|Re λ| → the rate is unidentifiable — *report* it, don't guess);
abstain on steady-state-only data (no kinetic information present). **NUDGE-LIM-011 (new):**
sampling-below-Nyquist rate unidentifiability + destructive-snapshot pseudo-trajectory caveat.

**(f) Tests.** Synthetic: a known rate change reads `rate-change`; a known setpoint change reads
`setpoint-change`; an under-sampled trajectory (Δt too coarse) abstains. Real-data lock-in
(`needs_data`): an ERK-KTR light-dose time series (via Cap 1).

**(g) Demo notebook.** `notebooks/ERK_KTR_Kinetics.ipynb` — "Did this perturbation change how *fast*
ERK relaxes, or *where* it settles? NUDGE separates rate from setpoint — or abstains when the
sampling can't resolve the timescale."

**(h) Effort.** **M, ~12–14 h.** **Depends on Capability 1** (activity readout) and on true
single-cell trajectories being scarce (destructive transcriptomic assays barely provide them). Build
*after* Cap 1, on ERK-KTR data.

**(i) Honesty flags.** Genuinely valuable but honestly gated by data scarcity + the Cap 1
dependency. Do not build on pseudo-trajectory transcriptomic moments and claim single-cell kinetics
— the fallback is population-moment-only and must be labelled as such. **Stretch/deferred this week.**

---

## Capability 5 — Bifurcation / tipping-point proximity (a robustness readout)

**(a) Bio question + who.** How close is a bistable switch to **losing bistability** (a saddle-node
/ fold)? A scalar **robustness / "flippability" score**: hair-trigger cliff vs well-buffered dial.
Resilience / critical-transition biology (aging, disease progression, cell-fate commitment),
engineered-circuit robustness QA.

**(b) Primitives reused.** `Circuit.fixed_points` already returns stable nodes + the index-1 saddle
with Jacobian-eigenvalue labels. Proximity to a fold is directly readable as (i) the
**smallest-magnitude real part of the Jacobian eigenvalue** (→ 0 at the fold), (ii) the
**state-space distance from a stable node to the saddle** (basin depth → 0 at the fold), or (iii)
the **LNA variance swell**. **`lna_reliable` already computes exactly this** ("near a saddle-node →
a lobe's CV > 1.5") as an *abstention trigger* — this capability **re-exposes the same internal as a
result**.

**(c) New code.**
- `src/nudge/inference/bifurcation.py` (~120 LOC — the smallest module):
  ```python
  @dataclass(frozen=True)
  class BifurcationScore:  # min|Re λ|, node→saddle distance, max lobe CV, one_sided flag
  def bifurcation_proximity(circuit) -> BifurcationScore
      # read the three proximity channels off fixed_points + _lna_covariance
  def classify_robustness(score, *, ...) -> tuple[str, str]
      # robust | near-fold | unresolved (deep-basin: LNA carries no fold info → abstain)
  def attribute_bifurcation(data, circuit, ...) -> BifurcationResult  # fit then score
  ```
- service/CLI (`nudge robustness`)/MCP + card `bifurcation_proximity.md` (`NUDGE-METHOD-006`) +
  registry; LIM-012. ~70 LOC.

**(d) Data.** Any bistable system with a dose axis approaching the fold: toggle+hysteresis **Zenodo
11817798**, morphogen ladders **GSE233574**, OCT4-exit **GSE283614**. Grounding: critical-slowing-
down / early-warning-signal literature (rising variance + autocorrelation near tipping points).

**(e) Fail-safe / abstention.** The honest catch: **the LNA Gaussian breaks down *precisely* at the
bifurcation** (variance diverges, the linear-noise approximation fails) — least reliable exactly
where it matters most. Must be reported as a **one-sided / lower-bound** estimate ("at least this
close") and **abstain on the far, deep-basin side** where a Gaussian lobe carries no fold
information. **NUDGE-LIM-012 (new):** one-sided-only near-fold estimate.

**(f) Tests.** Synthetic: a circuit tuned near the fold scores `near-fold` with the `one_sided`
flag set; a deep-basin circuit abstains rather than returning a precise "far" number; a well-
buffered switch scores `robust`. Real-data lock-in (`needs_data`): the toggle-hysteresis Zenodo
set or a morphogen top rung.

**(g) Demo notebook.** `notebooks/Robustness_Dial.ipynb` — "How close is this switch to the cliff?
NUDGE reports a robustness dial — as a one-sided lower bound, because the noise model is weakest
exactly at the fold."

**(h) Effort.** **S, ~6–8 h** (re-expose an existing internal). Cheapest single item. No
dependency. The UQ-near-the-fold honesty must stay front-and-center or it becomes an overclaim.

**(i) Honesty flags.** The number is a **lower bound**, not a point estimate — the whole capability
lives or dies on saying so. A demoable, evocative dial, but keep the one-sided caveat loud.

---

## Capability 6 — Multi-reporter joint attribution (identifiability force-multiplier)

**(a) Bio question + who.** Not a new *question* — a lever that **raises the resolved-call rate of
every capability above** by breaking the K⇄v_max / gain⇄threshold degeneracies (the dominant reason
NUDGE abstains; FINDINGS §2 repeatedly names "a richer multi-reporter readout" as the fix). Whoever
wants *fewer abstentions* rather than a new screen type.

**(b) Primitives reused.** Fit **several downstream reporter genes jointly** as multiple emissions
of the *same* latent switch — the `solve_population` vmap and the energy-distance loss extend to a
joint (multi-gene) distribution unchanged; each reporter is a `Readout` with its own gain/offset.
The shared latent becomes **over-determined**, so a threshold shift and a ceiling change (degenerate
through one reporter) project differently onto a *panel*.

**(c) New code.**
- `src/nudge/inference/multireadout.py` (~180 LOC):
  ```python
  def fit_multireadout(panel_data, circuit, readouts, *, ...) -> MultiReadoutFit
      # shared latent → several affine emissions; joint distributional loss over the panel
  def classify_multireadout(fit, *, ...) -> tuple[str, str]
      # resolved-{threshold,gain,ceiling} | unresolved | off-model(panel-inconsistent)
  def attribute_multireadout(...) -> MultiReadoutResult
  ```
- `bridge.py` — a panel extractor (per-reporter activity columns, per-reporter WT calibration). ~50
  LOC.
- service/CLI/MCP + card `multireadout_attribution.md` (`NUDGE-METHOD-007`) + registry. Possibly
  LIM-013 (panel-inconsistency semantics). ~90 LOC.

**(d) Data.** **Reuses panels that already exist — no new download.** The Gladstone IEG panel
(IL2/CD69/EGR1/FOS/NR4A1) is five reporters of one activation latent; every library dataset with a
multi-gene readout qualifies.

**(e) Fail-safe / abstention.** **Strictly strengthens** the fail-safe: an over-determined latent
means a spurious mechanism must be consistent across all reporters (harder to fake), and
**inconsistency across reporters is itself an off-model signal** — a reporter reading a *different*
latent flags a hidden node / wrong panel (the honest bridge toward Capability 7's abstention half,
kept as a flag, not a positive claim). **NUDGE-LIM-013 (optional, new):** panel disagreement is an
off-model witness, not a hidden-node identification.

**(f) Tests.** Synthetic: a panel of 4 affine reporters of one latent **resolves** a mechanism the
single-reporter path abstains on (the force-multiplier, measured against ground truth); a reporter
secretly reading a second latent triggers `off-model(panel-inconsistent)`. Real-data lock-in
(`needs_data`): the Gladstone IEG panel jointly vs single-reporter.

**(g) Demo notebook.** `notebooks/MultiReporter_Resolves.ipynb` — "One reporter abstains; five
reporters of the same switch resolve the call — and disagreement across reporters flags a wrong
panel."

**(h) Effort.** **M, ~11–13 h.** No new data, but a genuine new observation model (shared latent →
several emissions + a joint loss). Highest-leverage *internal* investment — lifts every attribution
capability — but ships no new screen type, so it competes for the same budget.

**(i) Honesty flags.** The "resolves more" claim must be *measured* on synthetic ground truth before
it is stated. Panel inconsistency flags off-model; it does **not** identify the hidden node
(respect the Cap 7 boundary).

---

## Capability 7 — Feedback / hidden-node detection — **abstention half only** ⚠ partial trap

**(a) Bio question + who.** When a simple model is consistently rejected, is the cause an
**unmeasured regulator / feedback loop**? NUDGE's most differentiated *aspirational* claim.
**Per the roadmap's explicit flag, this plan covers ONLY the abstention half** — NUDGE correctly and
loudly returning **off-model** when its model is inadequate. The **positive** claim ("there is a
hidden node *here*") stays a research direction, not a product claim.

**(b) Primitives reused.** The pieces already exist and largely ship: the **off-model parsimony
tripwire** (`switch_detected` / `select_topology` preferring the no-switch null), **BIC topology
selection** (1-node vs 2-node vs no-switch), and the **LNA covariance residual** (a hidden node
leaves a covariance structure a closed low-D ODE cannot reproduce — `mode_covariances` predicted vs
empirical).

**(c) New code.** Deliberately thin — the abstention already ships; this **packages it as an
explicit, legible verb** rather than building a new detector:
- `src/nudge/inference/hidden_node.py` (~90 LOC):
  ```python
  def offmodel_diagnostics(data, circuit, ...) -> OffModelReport
      # assemble: BIC(no-switch) win margin, covariance-residual ("comet-tail") magnitude,
      # N-D-saddle-needed flag — the EVIDENCE that the model is inadequate
  def classify_offmodel(report, ...) -> tuple[str, str]
      # returns "off-model" + a reason that ENUMERATES the indistinguishable causes
      # (nonlinear readout / off-target / batch / genuine hidden node) and REFUSES to
      # name one — the abstention, made legible
  ```
- CLI/MCP: fold into the existing `explain` verb + a `nudge diagnose-offmodel` reporter; card
  `hidden_node_abstention.md` (`NUDGE-METHOD-008`) + registry; LIM-014. ~50 LOC.

**(d) Data — the honest gap.** GSE114071 (miRNA-mRNA co-seq) co-measures the hidden node **but has
only 19–40 cells** → validation, not attribution. Canonical hidden-node systems (Mukherji 2011,
Bleris 2011, ceRNA sponges) have **no ingestible single-cell public data** (NOTEBOOK_LIBRARY C1).
**Almost entirely synthetic-data-bound today** — build the abstention against a synthetic IFFL/ceRNA
generator + the existing decoys.

**(e) Fail-safe / abstention.** The abstention direction is *safe and already shipped*. **NUDGE-LIM-
014 (new):** off-model is a **degenerate verdict** — nonlinear readout (LIM-006), off-target
effects, batch confounds, and a genuine hidden node **all look alike** — so claiming a hidden node
from an off-model verdict risks exactly the confident-wrong failure the project exists to avoid.
Inherits every existing decoy (001–005) whose correct answer is off-model.

**(f) Tests.** Synthetic: a hidden-node IFFL generator reads `off-model` with the covariance-
residual diagnostic populated; the classifier's reason **must enumerate** the alternative causes
(assert the string names readout/off-target/batch, not a hidden node). No positive-claim test — by
design. Real-data lock-in (`needs_data`, weak): GSE114071 naive 1-D fit rejects.

**(g) Demo notebook.** `notebooks/OffModel_Legible.ipynb` — "NUDGE rejects the simple model *and
tells you why it can't name the cause* — a hidden node is one of four indistinguishable
explanations. The honest output is the abstention, not a guess."

**(h) Effort.** **S–M for the abstention half, ~7–9 h** (mostly packaging + card + LIM + a synthetic
generator). The positive detector (a validated discriminating statistic) is **M–L and explicitly
out of scope this week.** No dependency.

**(i) Honesty flags.** This is the sharpest honesty case in the menu. **Ship abstention only.** The
positive hidden-node identification is weakly identifiable and has no real data — do not build or
imply it. The deliverable's whole value is making the *refusal* legible.

---

## Day 3–7 execution schedule

**Weighing** (per the brief): dependencies, data-readiness (one-line pertpy > needs-an-adapter),
**Demo leverage** (30%, weakest — Chure's author-labelled K-vs-n is the single best lever), effort,
and the four criteria. The T-cell multi-timepoint breaker capstone runs **in parallel** (not
scheduled here). At the anchor's definition-of-done, **8 in 5 days is not honest** — the realistic,
ambitious target is **3 capabilities fully complete + 1 partial**, with clear stretch items.

**Build order (highest-value-first, front-loading the Demo lever while there's slack to absorb
slippage):** Capability 2 → Capability 1 → Capability 5 → Capability 3 → (stretch) 6 / 7-abstain →
(deferred) 4.

Rationale for opening with **Capability 2, not 1**: it is the *only* fully one-line-pertpy dataset
(Norman 2019) and reuses the shipped `fit_lyapunov_multi` breaker almost verbatim, so it is the
fastest *guaranteed* second capability and a second real-data positive — it de-risks Day 3 and
builds the "second capability" muscle. Capability 1 (the Demo crown jewel, but carrying a data
adapter + a CSV download) then gets **two full days** so its Chure validation lands solid.

| Day | Build | "Done" means |
|---|---|---|
| **3** | **Capability 2 — Synergy/epistasis.** `epistasis.py` (additive null + `classify_synergy`), `combo_operating_points`, service/CLI/MCP, card `NUDGE-METHOD-003`, LIM-009, synthetic tests + Norman lock-in, `Norman_Synergy.ipynb` executed. | Module + fail-safe classifier + CLI verb + MCP tool + card + LIM + synthetic & real-data (`needs_data`) tests + executed notebook — all checks green (ruff 88-char, `uv run pyright src`, the 4 doc checkers, fast lane). |
| **4** | **Capability 1 (core) — Cross-modality adapter.** `check_readout` modality bouncer, `fluorescence_dose_response` extractor, service/CLI `--modality`/MCP, card `NUDGE-METHOD-002`, LIM-008, synthetic fluorescence tests + the log-normalized-counts refusal test. | The adapter code + fail-safe (bouncer refusal) + wiring + card + LIM + synthetic tests, all green. (Reuses the shipped dose-response fit/classify, so no new inference math.) |
| **5** | **Capability 1 (demo) — Chure 2019 validation.** Download D1.1241 tidy CSVs, real-data lock-in test (DNA-binding→K, inducer-binding→n vs author labels), `Chure_LacI_Benchmark.ipynb` executed. | The single biggest Demo lever, standing on its own day: a `needs_data` lock-in + an executed flagship notebook recovering the author's answer key — all green. |
| **6** | **Capability 5 — Bifurcation proximity** (half day; cheapest, re-expose `lna_reliable`'s internal): `bifurcation.py`, service/CLI/MCP, card `NUDGE-METHOD-006`, LIM-012, synthetic + real-data tests, `Robustness_Dial.ipynb`. **+ start Capability 3** (`differential.py` nested-BIC head + synthetic tests). | Cap 5 complete + green; Cap 3 module + synthetic tests landed (not yet wired/carded). |
| **7** | **Finish Capability 3 — Comparative differential:** service/CLI/MCP, card `NUDGE-METHOD-004`, LIM-010, real-data lock-in (sci-Plex A549 vs MCF7), `Differential_Resistance.ipynb`. **+ integration polish + refresh living docs** (STATE.md, FINDINGS.md, JUDGES_GUIDE.md, README/CHANGELOG). | Cap 3 complete + green; all four living docs updated in-change; a clean top-of-tree. |

**One-liner per day:** D3 ship synergy (turnkey data) · D4 build the modality adapter · D5 land the
Chure ground-truth demo (the Demo crown jewel) · D6 ship the cheap robustness dial + start
differential · D7 finish differential + polish/refresh docs.

### Core vs stretch — the explicit cut line

- **CORE (commit to these):** Capability 2 (Day 3), Capability 1 incl. the Chure demo (Days 4–5),
  Capability 5 (Day 6). These are the three fully-complete targets — data-ready or cheapest, and
  including the single highest Demo lever.
- **STRETCH (do only if ahead):** Capability 3 (Days 6–7 — first to drop if behind); then
  Capability 6 (multi-reporter, no new data but a real new observation model); then Capability 7's
  **abstention-packaging only** (thin, mostly card + LIM + a synthetic generator).
- **DEFERRED this week (out of scope, stated honestly):** Capability 4 (temporal — blocked on
  Capability 1 *and* on scarce true single-cell trajectories) and Capability 7's **positive**
  hidden-node claim (weakly identifiable, no real data — the roadmap's partial trap).

**If behind after Day 5:** drop Capabilities 3 and 6 entirely and spend Days 6–7 **hardening the Cap
1 and Cap 2 demos** — the Demo criterion (30%) rewards two *trustworthy, watchable* validations far
more than a third half-finished capability. Never trade the honesty rule for a green checkmark: a
polished-but-false claim scores worse than an honest gap.

### The 2–3 biggest risks / dependencies

1. **Capability 1's Chure data ingest (needs-work, not one-line).** The Demo crown jewel depends on
   downloading + wrangling D1.1241 tidy CSVs and a fold-change/autofluorescence extractor. If the
   CSVs fight back, the fallback is the synthetic-fluorescence path (still a complete capability,
   weaker demo) — but the *ground-truth* validation is what moves Demo, so protect Day 5.
2. **Capability 4 is genuinely blocked** on Capability 1 (activity readout) *and* on the scarcity of
   real single-cell trajectories — correctly deferred, not scheduled. Do not let it tempt mid-week
   scope creep.
3. **The fail-safe must survive every new capability.** Each is additive/opt-in and must never touch
   the energy-distance `fit()` default or the decoy battery; each carries the topology + affine-
   readout gates (NUDGE-LIM-006) and its own new LIM. The recurring trap: a capability whose "second
   operating point" is actually a *confound* (Cap 3's batch/depth, Cap 5's near-fold LNA breakdown)
   — pin depth per context, keep the reliability guards, and abstain otherwise.
