# Changelog

All notable changes to NUDGE are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); the project follows
[Semantic Versioning](https://semver.org/). The `fit` / `design` public surface
is the stability contract (see `docs/architecture/verification_vs_validation.md`).

## [Unreleased]

### Added

- **Multi-reporter joint attribution — the identifiability force-multiplier
  (`nudge.inference.multi_reporter`, `NUDGE-METHOD-008`):** breaks NUDGE's dominant reason
  to abstain — the measured **K⇄v_max / gain⇄threshold degeneracy** (FINDINGS §2) — by
  fitting **several downstream reporters of ONE latent switch jointly**. Each reporter is
  an affine readout `y_j = base_j + gain_j·A·f(dose; K, n)` of the *same* latent
  (genuinely a `Readout` of a shared Hill activity); pinning the reporter gains from the
  control and sharing one latent over-determines the fit, so a **threshold** shift (moves
  the inflection identically across reporters) and a **ceiling** change (scales every
  reporter's ON amplitude by the same fraction) project **differently** onto a panel of
  heterogeneous gains — the multi-*reporter* analogue of the second-operating-*point* ×16
  degeneracy-break. **Headline (synthetic ground truth, FINDINGS "Phase 4h"):** a known
  threshold-only / gain-only / ceiling-only perturbation on one latent, seen through 4
  heterogeneous-gain reporters — the **JOINT** panel recovers the mechanism **24/24 (100%)**
  where a **SINGLE** reporter resolves **0/24** (`unresolved`, the degeneracy), with **0
  confident-wrong calls**. **Fail-safe, strengthened:** the **consistency guard**
  (`NUDGE-LIM-014`) abstains **off-model** when the panel cannot be explained by one shared
  latent (a reporter reads a *different* latent — a hidden node / wrong panel), never
  averaging it into a call; a single reporter honestly returns `unresolved`. Wired into the
  `nudge multi-reporter` CLI verb + the `multi_reporter` MCP tool +
  `service.multi_reporter_file` + a Mechanism Card + `notebooks/Multi_Reporter.ipynb`.
  Additive/opt-in — touches neither the energy-distance `fit()` default nor the decoy
  battery.

- **Inverse / intervention design — the flagship `design()` (`nudge.design.invert`,
  `NUDGE-METHOD-007`):** delivers the brief's headline thesis — NUDGE *inverts the fit to
  propose untested interventions*. Given a **reliable** attribution it runs the same
  differentiable fit **backwards** to prescribe an intervention, behind two honesty gates.
  **Circuit mode** (flagship): gradient inversion over a fitted `Circuit` (the
  `fit_parameters` loop backwards — Adam over an additive log-Δ on addressable kinetic
  knobs, minimizing `‖PredictedState − target‖² + l1‖Δ‖₁`), then a **bifurcation safety
  gate** reusing the Cap-5 `bifurcation_proximity` dial on the intervened circuit — it
  flags an intervention that pushes a bistable switch toward its fold
  (`high_risk_of_instability`) or crosses it and destroys bistability (`crosses_fold`),
  inheriting the one-sided lower bound near the fold (`NUDGE-LIM-012`). **Curve mode**
  (real-data surface): closed-form inversion of a `DoseResponseFit` to the dose achieving a
  target response `y` (no circuit ⇒ **no safety gate**, stated honestly). **Two honesty
  gates:** an **integrity gate** (refuses to invert an unreliable attribution — a
  strictly-minimal `AttributionResult` protocol; `DoseResponseResult` / `EpistasisResult`
  gain an additive `is_reliable` property) and a **reachability abstention** (abstains
  rather than extrapolate to an unreachable target — `NUDGE-LIM-013`). **Validated on
  synthetic ground truth:** known-intervention recovery to residual gap `<1e-3` (a known
  `×2` on `v_max` recovered to `factor≈2.0`); the fold-crossing flip flagged HIGH RISK
  while a safe nudge clears; curve round-trip (`y=floor+amp/2` → `dose≈K`) + out-of-range
  abstention (FINDINGS "Phase 4g"). **Real-data lock-in** (`needs_data`): inverts the OCT4
  self-renewal switch fit to a knockdown dose. Wired into the `nudge design` CLI verb + the
  `design` MCP tool + `service.design_circuit`/`design_file` + a Mechanism Card +
  `notebooks/Inverse_Design.ipynb`. Additive/opt-in — it touches neither the energy-distance
  `fit()` default nor the decoy battery.
- **Laplace posterior uncertainty — curvature error bars on the recovered kinetics
  (`nudge.inference.uncertainty`):** turns the fit's point estimate `θ*` (log-space kinetics)
  into a *local* Gaussian posterior `θ ~ N(θ*, H⁻¹)` from the loss Hessian `H = ∇²L(θ*)`
  (Laplace's approximation). The Hessian target is the **deterministic** Lyapunov
  Gaussian-mixture NLL (`lyapunov_nll_loss`) — *not* the stochastic energy distance — so `H`
  is the observed Fisher information and `H⁻¹/N` the covariance. Gives **(a)** natural-unit
  marginal CIs on `K` / `n` / `v_max` (log-space Gaussian → exact lognormal interval),
  **(b)** the parameter correlation structure, and **(c)** a `mechanism_confidence` that
  **abstains**. **Fail-safe first (the load-bearing honesty point):** the inverse is a
  **guarded ridge-regularized eigen-inverse** — never a plain pseudo-inverse, which would
  *zero* a flat direction's variance (false precision) — so a flat / degenerate direction
  widens to a large-but-finite, PSD variance (no NaN), sets `LaplacePosterior.degenerate`,
  and marks the affected knob **unidentifiable / CI unbounded**; a non-positive-definite
  Hessian → cond ∞ → abstain. **Validated (FINDINGS "Laplace posterior"):** the marginal CI
  covers the true ceiling **20/20** across seeds; the measured **gain⇄threshold degeneracy
  reproduces as a near-singular Hessian** (condition number ≈ 210, `|corr(n, K)| ≈ 0.99` —
  the *inverse* of the FIM's −0.99, same degeneracy) with `n` + `K` flagged unidentifiable;
  and a **second operating point breaks it** (condition number ≈ 210 → ≈ 27, resolving),
  mirroring the covariance-attribution ×16 Fisher result. **Additive / opt-in:** it computes
  over a caller-supplied loss and touches neither the energy-distance `fit()` default output
  contract nor the decoy battery. Tests: `tests/inference/test_uncertainty.py`.
- **Toggle covariance attribution validated on INDEPENDENT stochastic data — the single
  snapshot degenerates, the second operating point recovers (fail-safe).** The Lyapunov
  covariance path (`nudge.inference.lyapunov`) was previously validated only under the
  *inverse crime* (cells drawn from the LNA Gaussian the fitter maximizes). New measurement
  on data from the **independent tau-leaping SSA** (`generate_toggle_perturbseq`), bridged to
  activity as the real-data path does, at a depth that clears the `lna_reliable` guard
  (3 seeds; `scripts/vv/toggle_lyapunov_ssa.py`): **(a) the single toggle snapshot
  DEGENERATES** — the inverse-crime "ceiling is identifiable" result does *not* survive; on
  the true stochastic distribution the free-vmax fit is the *worst* explanation of a true
  ceiling, so a true ceiling mis-narrows to `gain_or_threshold` and gain/threshold abstain
  (`unresolved`). It still only ever returns an abstention-class label (never a bare
  gain/threshold/ceiling), so it is never confidently wrong — but a single snapshot is not a
  positive on real data (docstring corrected). **(b) The two-operating-point breaker
  RECOVERS** — a shared-parameter joint fit across two basal-B operating points resolves
  **threshold** (3/3) and **ceiling** (3/3) correctly and honestly **abstains on gain**
  (residual gain⇄threshold confound), with **0 confident-wrong calls** across all mechanisms
  × seeds. This is the first evidence the covariance-difference signature separates mechanism
  on **non-inverse-crime** toggle ground truth — a guarded positive for the multi-condition
  path plus a documented single-snapshot negative. Additive/opt-in, **not** wired into
  `fit()`; NUDGE's production toggle path still abstains (`test_toggle_nd_safety`). Locked by
  `tests/inference/test_lyapunov_toggle_ssa.py`; FINDINGS "independent-SSA validation".

- **Bifurcation / tipping-point proximity — the "robustness dial" (`nudge.inference.bifurcation`,
  `NUDGE-METHOD-006`):** answers a new question — **how close is a bistable switch to
  *losing* bistability** (a saddle-node fold)? — as a scalar 0..1 dial from **three
  complementary channels**, each with a known analytic limit at the fold: **critical
  slowing** (`min|Re λ|` of the drift Jacobian at each stable mode → 0), **basin collapse**
  (stable-node → index-1-saddle distance → 0), and **LNA lobe swell** (`√λ_max(Σ)/sep` →
  1). It re-exposes a signal that was *already computed but buried* (the fixed-point
  eigenvalues that `Circuit.fixed_points` dropped; the lobe ratio inside `lna_reliable`) as
  a public, honestly-bounded score. **The load-bearing honesty point (`NUDGE-LIM-012`):**
  the linear-noise Gaussian **breaks down precisely at the fold** (variance diverges), so
  the dial is a **one-sided LOWER BOUND** near the fold (`BifurcationScore.one_sided`) —
  never a point estimate — and `classify_robustness` **abstains** (`unresolved`) on the
  deep-basin far side rather than emit a false-precise "far" number; `not-bistable` when < 2
  stable modes; `robust` only for a well-buffered switch. **Validated on the self-activation
  switch's known analytic saddle-node in `n` and `K`:** sweeping toward the fold, all three
  channels move **monotonically** and the fused dial ranks proximity correctly, with
  `one_sided` setting near the fold (FINDINGS "Phase 4f"). `BifurcationScore.channels`
  retains the raw per-mode values for the demo. Wired into the `nudge robustness` CLI verb +
  the `robustness` MCP tool + `service.robustness_circuit`/`bifurcation_file` + a Mechanism
  Card + `notebooks/Robustness_Dial.ipynb`. It generalises to N-species switches (it is the
  hard dependency for the future `design()` **safety gate**). Additive/opt-in — it touches
  neither the energy-distance `fit()` default nor the decoy battery. A real-data dose-ladder
  lock-in is a deferred `needs_data` follow-up.

- **Possible-neomorphic off-axis diagnostic** for synergy/epistasis
  (`NUDGE-METHOD-003`): every combination fit now carries the magnitude of its
  interaction residual `r = v_AB − v_A − v_B` **orthogonal** to the additive axis —
  the emergent component the scalar interaction structurally cannot see. Computed in
  `nudge.inference.bridge.combo_effect_scores` (new `return_geometry=True` →
  `ComboGeometry`), surfaced as `EpistasisFit.off_axis_residual` / `.neomorphic_ratio`,
  and threaded into `service.synergy_to_dict`. For a `synergistic`/`buffering` call with
  `neomorphic_ratio ≥ 1.0` (off-axis ≥ on-axis) the reason gains an honest **UNDER-count
  warning** — the scalar is direction-correct but may under-count an emergent piece
  (NUDGE-LIM-009). It is **additive and opt-in**: the pure scalar-array fit, every call,
  and every fail-safe margin (`bic_margin`, `min_cells`, `rel_width`) are unchanged — it
  is a flag, never a discovery or a hidden-node claim. On Norman 2019 the three synergy
  pairs are flagged (off-axis 2.2–2.5 vs on-axis 0.9–1.3, ratio 1.8–2.7) while the sharp
  DUSP9+ETS2 buffering match — a clean on-axis masking — is correctly **not** flagged
  (ratio 0.62). Turns `NUDGE-LIM-009` from prose into a number shown with every call; see
  `design/NORMAN_DISCREPANCY_ANALYSIS.md`, FINDINGS "Phase 4d", `notebooks/Norman_Synergy.ipynb`.

- **Cross-modality readout adapter (`nudge.inference.cross_modality`, `NUDGE-METHOD-002`):**
  runs the *same* threshold (K) / gain (n) / ceiling (v_max) attribution on a **continuous
  single channel** — flow-cytometry fluorescence, an activity reporter, or a fold-change
  summary — instead of raw UMI counts, reusing the shipped dose-response fit/classify
  (`NUDGE-METHOD-001`) verbatim. Two new pieces make it modality-aware: a **bouncer**
  (`nudge.data.ingest.check_readout`) that routes `modality="counts"` to the unchanged
  integer guard and refuses ambiguous continuous input — most sharply **log-normalized or
  raw counts masquerading as fluorescence** (all-integer / zero-inflated / centered
  fingerprints; new `NUDGE-LIM-008`), never guessing a modality — and a **fold-change
  extractor** (`nudge.inference.bridge.fluorescence_dose_response`). A panel
  (`attribute_variant_panel`) localizes each variant's effect vs a control to **threshold**
  (dose-EC50 shift) / **gain** (Hill steepness) / **ceiling** (leakiness / dynamic range) —
  or abstains (**non-responsive** / **inconclusive**). Wired into the `nudge cross-modality`
  CLI verb + the `cross_modality` MCP tool + a Mechanism Card. **Validated on Chure 2019
  (CaltechDATA D1.1241, LacI IPTG induction):** against the authors' domain answer key,
  inducer-binding mutants **Q294K / Q294V** localize to **threshold** (K 71 → 420–626 µM),
  DNA-binding mutants **Y20I / Q21A** to **ceiling / leakiness** (floor +0.3–0.5), the
  near-non-inducible **Q294R** abstains — 4/7 cleanly correct, 3/7 honest abstentions,
  **0 mis-calls, no gain(n) overclaim**; `F164T` / `Q21M` inconclusive at one operating
  point. The dose-response fit additionally records bootstrap CIs on the response span
  (`ci_amp`) and baseline (`ci_floor`) for the ceiling axis (additive; the count path and
  the `check_counts` integer guard are untouched). Demo: `notebooks/Chure_LacI_Benchmark.ipynb`.
- **Synergy / epistasis attribution (`nudge.inference.epistasis`, `NUDGE-METHOD-003`):**
  for a two-perturbation combination, calls the interaction **additive** / **synergistic**
  / **buffering** — or abstains (**no-effect** / **unresolved**). Reads A, B and A+B as
  three operating points against a shared control, reduces each to a scalar **effect**
  (log-fold-change space, so the additive null is **Bliss independence**), and reports the
  **interaction** `effect(A+B) − [effect(A)+effect(B)]` with a **bootstrap CI over cells**.
  A combo **inherits its weakest single arm**: the classifier abstains when an arm is
  underpowered or the CI is too wide, and a super-additive residual is **not** a hidden-node
  claim (new `NUDGE-LIM-009`). The per-cell score projects onto the additive axis fixed by
  the two single arms (direction-safe; `nudge.inference.bridge.combo_effect_scores`). Wired
  into the `nudge synergy` CLI verb + the `synergy` MCP tool + a Mechanism Card. Applied to
  Norman 2019 (GSE133344); an independent literature fact-check graded **2/5 pairs
  explicitly confirmed** against the paper — DUSP9+ETS2 → **buffering** (Fig 5
  DUSP9-dominant suppression of ETS2) and CBL+CNN1 → **synergistic** (Fig 3 emergent
  erythroid synergy) — with CBL+UBASH3B / CNN1+UBASH3B → synergistic cluster-consistent but
  unlabeled, and FOXA1+FOXA3 → additive a paralog control Norman does not analyse. Agreement
  is at interaction type/direction (a Bliss scalar null vs Norman's regression GI), not a
  reproduction; note "buffering" here = a negative interaction = the same antagonism as
  Norman's fitness-GI "buffering" but the opposite sign. See FINDINGS "Phase 4d"; demo in
  `notebooks/Norman_Synergy.ipynb`.
- **Dose-response attribution (`nudge.inference.dose_response`, `NUDGE-METHOD-001`):**
  a second measurement of the same circuit — fits the *same* Hill primitive
  (`hill_repression`/`hill_activation`) to a readout's response across a graded dose and
  classifies **switch / graded / no-effect / unresolved** with the *same* BIC parsimony
  discipline. Reports `n` as an **apparent population gain + bootstrap CI** (not molecular
  cooperativity) and abstains when the doses don't span the inflection (new `NUDGE-LIM-007`).
  Wired into the `nudge dose-response` CLI verb + the `dose_response` MCP tool + a Mechanism
  Card; the fit hands `curve_fit` an exact JAX-autodiff Jacobian (the float32
  finite-difference Jacobian froze `n` at its init — verified, regression-locked). Validated
  on OCT4/NANOG (GSE283614): OCT4 resolves as a switch (n≈6.7, R²=0.99); NANOG correctly
  abstains — see FINDINGS "Phase 4b".
- **Phase 0 — skeleton:** `src/nudge` package, the two-layer circuit API
  (`Circuit`, `CircuitBuilder`, `CircuitSpec`), the `MechanismRegistry`, the
  attribution vocabulary (`MechanismClass` with first-class abstention classes),
  and the `MechanismMap` output schema.
- **Phase 1 — generative backbone:** the differentiable `Circuit` (self-contained
  JAX vector field + semi-implicit steady-state solve), the mechanism library
  (species, integrators, regulatory Hill/linear effects, readout), the negative-
  binomial observation model (`data/noise.py`, no zero-inflation), the Tier-0
  synthetic generator (`data/synthetic.py`), and the raw-count ingestion guardrail
  (`data/ingest.py`).
- **Phase 2 — fit engine:** distributional losses (energy distance / MMD,
  `inference/losses.py`), the optax population fit (`inference/fit.py`), and the
  two-level abstention gates (`inference/classify.py`) — a circuit-level
  linear-baseline parsimony gate plus per-perturbation resolution. Proof of concept
  closed end to end.
- **Tier-0.5 — independent stochastic simulator** (`data/stochastic.py`): a
  tau-leaping SSA of a self-activating gene with *emergent* bimodality, breaking the
  inverse crime of self-benchmarking.
- **Multi-basin + saddle transition-mode gain gate:** `energy_distance_weighted`,
  `fit_multibasin` / `fit_transition_parameters`, `Circuit.fixed_points` /
  `transition_state`, and `classify.decide_with_transition` — fail-safe mechanism
  attribution on emergent-bistable stochastic data.
- **Decoy battery (started):** `NUDGE-DECOY-001` — the telegraph / noise-induced
  bimodality decoy (`generate_telegraph_perturbseq`, To & Maheshri 2010): bimodal but
  deterministically monostable data that NUDGE must decline as not-a-switch (it does,
  on both fit paths). Registry `data/decoys.py`; limitation `NUDGE-LIM-001`.
- **N-D saddle finder + multi-basin representation:** `Circuit.fixed_points` /
  `transition_state` generalized to N species (multi-start Newton + Jacobian-index
  classification; verified on a 2-node toggle), and the transition fit seeds basins at the
  stable fixed points (`generate_toggle_perturbseq` for toggle data). The `w_trans` gain
  gate is 1-D-specific and does *not* extend to the toggle (measured), so it stays guarded
  to `n_species == 1`; NUDGE abstains (never misclassifies) on toggles. See FINDINGS
  "N-D saddle".
- **N-D finder jitted (performance):** the per-optimizer-step Newton/dedupe/eigenvalue core
  is now a jitted, per-topology-cached kernel (`_nd_kernel`; kinetics as a traced argument)
  — **byte-identical** roots, ~1 ms/call (333× per-call; a toggle transition fit 26 s → 4.1 s).
- **Toggle attribution — researched (`design/TOGGLE_ATTRIBUTION_RESEARCH.md`):** an
  adversarially-verified `/deep-research` synthesis of why the saddle gain gate does not
  extend to a toggle (mixture weights are non-gradient-quasi-potential-set, not saddle-set)
  and the signature that would (linear-noise Lyapunov mode covariance). A Fisher-information
  analysis (`scripts/vv/fisher_sloppiness.py`, `fisher_extrinsic.py`) then *measured* the
  degeneracy: it is **gain⇄threshold** (analytically `n·ln(K/B)`; ceiling is the most
  identifiable), robust to extrinsic noise, broken by a **second operating point**.
- **Covariance attribution — the Lyapunov path (`nudge.inference.lyapunov`):** an additive,
  opt-in, guarded capability (never wired into `fit()`). `Circuit.mode_covariances`
  (per-stable-mode linear-noise covariance); `fit_lyapunov_parameters` (differentiable LNA
  Gaussian-mixture fit); `calibrate_from_wt` (pins the scale⇄vmax sequencing-depth nuisance
  from WT); `attribute_lyapunov_single` (identifies ceiling, abstains between gain/threshold);
  `OperatingPoint` + `fit_lyapunov_multi` / `attribute_lyapunov_multi` (a shared-parameter
  joint fit across operating points that **resolves** gain vs threshold — the breaker); and
  `lna_reliable` (abstains loudly at low depth / near a bifurcation / when monostable).
  Validated on LNA/synthetic ground truth; not yet real data. See FINDINGS "Covariance
  attribution".
- **Phase 4 — real-data infrastructure (the Gladstone T-cell screen):** a generic,
  backed-mode Perturb-seq loader (`data/loaders/perturbseq.py` — config-driven, subsets
  ~150 GB files on disk without loading the matrix; Gladstone config in `tier2.py`); named
  Ras-switch circuits (`circuits.py` — `ras_switch_1node` / `ras_switch_2node`); the
  counts→activity bridge (`inference/bridge.py` — depth-normalize + reduce to activity
  space); **topology model-selection** (`inference/model_select.py` — a BIC parsimony gate
  over {no-switch, 1-node, 2-node} so the circuit is inferred, not assumed); and the
  end-to-end attribution pipeline + CLI (`inference/pipeline.py`,
  `scripts/vv/gladstone_attribution.py`). The real-data attribution run is pending the data
  download.
- **`nudge` CLI (typer):** a thin, tested command layer over the existing API —
  `nudge load` (backed-load + summarise a Perturb-seq file), `nudge check-data`
  (the raw-count ingestion guardrail, exits loudly), `nudge attribute` (covariance
  attribution at an operating point, printing the call + honest skip/abstention
  reasons), `nudge mechanisms` (the registered library), and `nudge explain` (the
  "why did it abstain?" verb). `src/nudge/cli.py`, `src/nudge/service.py`
  (the CLI/MCP-shared orchestration), `[project.scripts] nudge`.
- **Claude integration — MCP server** (`src/nudge/mcp/server.py`, `nudge-mcp`
  entry point + `.mcp.json`): a FastMCP stdio server exposing `attribute`,
  `explain_abstention`, `list_mechanisms`, and `get_mechanism_card` so Claude
  (Claude Code / Desktop / the Claude Science workbench) drives NUDGE in plain
  language and gets the *same* honest, abstaining output. Feasibility verified
  and the exact connection recipes recorded in `design/INTEGRATION_FEASIBILITY.md`.
  Guarded behind the optional `nudge-bio[mcp]` extra.
- **Shared knowledge base** (`src/nudge/knowledge.py`): read-only lookups over the
  mechanism registry, decoy battery, `known_limitations.yaml`, and Mechanism Cards
  — the one tested source the CLI, MCP server, and skills all use, so an
  abstention always resolves to *which* decoy / limitation / card explains it.
- **Mechanism-Card knowledge base** (`docs/mechanism_cards/`): 10 cards (6
  primitives + 4 motifs) with machine-readable YAML front-matter
  (`vulnerable_to_decoys`, `documented_limitation`, `validated_in_regime`,
  `references`), a README index, `scripts/check_mechanism_cards.py` + a test
  asserting every registered mechanism has a card, and the primary-literature bib
  entries. Registry population fixed so it is complete (`LinearIntegrator` was
  silently dropped) — `src/nudge/mechanisms/__init__.py`.
- **Agent Skills** (`.claude/skills/`): `nudge-attribute`, `nudge-explain`, and
  `mechanism-card` — compose the CLI/MCP into NUDGE workflows.
- **Ontology design** (`design/ONTOLOGY.md`): the SPARQL/RDF vision + a costed
  `rdflib` prototype sketch (not on the critical path; the knowledge layer already
  answers the "why abstain?" traversal in Python).
- Traceability inherited from `maddening.compliance` (`NUDGE-*` ID prefixes) and CI
  validators (`check_anomalies`, `check_citations`, `check_impl_mapping`,
  `check_mechanism_cards`); PEP 561.

### Performance

- **Loader ~5× faster** (`data/loaders/perturbseq.py`): the pointer-read hot path
  (`_read_h5ad_rows`, ~99% of load time) now coalesces adjacent selected rows into
  contiguous h5py slice reads (`_coalesced_gather`) instead of one big fancy-index —
  **byte-identical** output, ~4.6–5.4× uncompressed / ~1.7–2× gzip, still O(selection)
  (holds at 150 GB). Profiling report + benchmarks: `design/PERFORMANCE.md`, `scripts/perf/`.
- **Demo-latency warmup** (`nudge.warmup`, `nudge warmup`): pre-compiles the cached hot
  JAX paths (the dose-response model + the circuit fixed-point kernel) on tiny dummy data,
  so the first real fit in a long-lived process is fast (dose-response first fit ~405→55 ms;
  `_nd_kernel` 512→2 ms). Wired into the MCP server startup + the demo notebooks; idempotent,
  no numerics change. (GPU verdict in `design/PERFORMANCE.md`: stay on CPU for these sizes.)

### Verification

- V&V calibration sweep (`scripts/vv/`): **0% misclassification** across 300 linear
  + 552 switch datasets; calibrated `margin_k = 1.7`. Tier-0.5 inverse-crime guard
  and the seed-2 saddle gain-recovery test. Findings in `scripts/vv/FINDINGS.md`.

### Known Limitations

- See `docs/known_limitations.yaml` (`NUDGE-LIM-*`); the full decoy battery, Laplace
  uncertainty, and real-data validation are not yet built. `design()` is a stub.
