# Changelog

All notable changes to NUDGE are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); the project follows
[Semantic Versioning](https://semver.org/). The `fit` / `design` public surface
is the stability contract (see `docs/architecture/verification_vs_validation.md`).

## [Unreleased]

### Added

### Changed

### Fixed

## [0.4.1] — 2026-07-13

### Fixed

- **`oed` no longer emits a false-precise improvement factor on a rank-deficient naive design**
  (`NUDGE-LIM-029`). When the naive baseline's Fisher information is singular in the target
  direction (smallest eigenvalue at the guarded-ridge floor — e.g. a literal baseline+end
  `[0, 12]` plaque schedule, where plaque ≈ 0 at t=0 zeroes both sensitivities), the true CRLB is
  infinite, so any finite `crlb_improvement` was a Tikhonov-ridge artifact (~10⁷). The tool now
  flags `naive_rank_deficient` / `naive_target_identifiable=False` and reports the improvement as
  a LOWER BOUND — the honest abstention a competent analyst otherwise makes by hand. The finite
  gain on informative designs is unchanged: the default `ad_qsp` schedule still reports
  `crlb_improvement ≈ 259` byte-for-byte (the guard is a curvature-grounded ridge-floor test, not
  a magic threshold, so it does not over-abstain).
- Removed the redundant PyPI-version badge from `PYPI_README.md`: it permanently lagged one
  release on the PyPI project page (PyPI snapshots the proxied shields image at upload, before its
  own index reflects the just-published version). PyPI's header already shows the true version;
  the GitHub `README.md` keeps the live badge.

## [0.4.0] — 2026-07-13

### Added

- **Hierarchical / nonlinear-mixed-effects AD QSP population model (`ad_qsp_nlme`) with a
  genuinely COUPLED (arrowhead) joint Fisher-Information Matrix** (`NUDGE-LIM-028`). Subjects
  share population hyperparameters (geometric-mean `μ`, fixed effects `φ`, optional log-spread
  `ω`), so the joint FIM over `[μ | ω | φ | rᵢ]` is a bordered/**arrowhead** matrix — cross-subject
  blocks exactly zero, a dense border coupling every subject — and is **not** block-decomposable
  like the independent-subjects cohort. Registered for the `identifiability` tool over the model
  registry. This makes the matrix-free-vs-dense scale contrast honest (the block-diagonal cohort
  was a strawman we refused): MEASURED (`scripts/demo_nlme_scale.py`), dense `jacfwd` OOM-kills at
  N≥300 under a 2.5 GB cap while the matrix-free `jvp∘vjp` path stays ~0.56→0.89 GB across
  N=100→2500 (≈linear in cohort state, **not** flat) and returns the same `unidentifiable`
  verdict; the `NUDGE-LIM-023` rank-deficiency fail-safe is preserved. Arrowhead is in-principle
  Schur-decomposable, so the shipped claim is the measured dense-OOM-vs-matrix-free contrast, not
  impossibility.
- **A self-contained NumPy WITHOUT-arm** for the coupled model (`scripts/demo_ab/
  ad_qsp_nlme_forward.py` + `cohort_nlme.npz`) so a raw agent can attempt the same joint
  identifiability question (no `nudge` import) for a fair A/B.
- **3-minute demo/pitch video shooting script** (`design/DEMO_VIDEO_SCRIPT.md`).

### Changed

- **Corrected the AD QSP confound attribution to be design-dependent** (honesty; caught by a
  Claude Science reviewer). On the 2-biomarker / 8-visit cohort the confounded pair is the
  microglial clearance ⇄ activation pair `k_gl` ⇄ `k_ga` — antibody binding `k_on` is *well*
  -identified there; the antibody-binding ⇄ microglial-clearance `k_on` ⇄ `k_gl` confound belongs
  to the sparser single-biomarker (amyloid-PET, baseline+end) OED design. Fixes the
  `scripts/demo_ab/ad_qsp_forward.py` docstring + README, which had mis-attributed the OED
  design's confound to the cohort.

## [0.3.0] — 2026-07-13

### Added

- **Two GENERAL-PURPOSE MCP tools — `identifiability` and `oed` — over a model registry
  (`nudge.inference.model_registry`; `NUDGE-METHOD-015`, `NUDGE-LIM-027`).** Expose NUDGE's
  matrix-free identifiability (`NUDGE-LIM-023`) and gradient OED (`NUDGE-METHOD-014`,
  `NUDGE-LIM-024`) as *model-agnostic* tools driven **by name** over a general registry — not a
  demo-specific cheat. The registry ships **≥3 models per tool across domains** (`glv` microbiome
  ecology · `linear_pathway` reaction kinetics · `ad_qsp` clinical pharmacology · `logistic`
  population dynamics · the canonical `sum_of_exponentials` / `redundant_exponential` /
  `well_conditioned` toys), so both tools demonstrably "work on any differentiable ODE." Each
  takes a *model reference* (never a hardcoded answer), runs the REAL analysis, and returns
  whatever it MEASURES — including honest abstentions. `identifiability(model, free, n_free,
  method, sigma)` returns the verdict (`well-constrained` / `sloppy-but-predictive` /
  `unidentifiable`), the named null directions, the fail-safe bound (`NUDGE-LIM-023`), and the
  **FIM-spectrum figure inline** (base64 + the regenerating `fig.py` + data sidecar under
  `NUDGE_ENV=cloud`). `oed(model, target, objective, steps)` returns the gradient-optimal design,
  the **MEASURED** CRLB / smallest-eigenvalue lift (never asserted; local OED `NUDGE-LIM-024`),
  and the **95%-ellipse-collapse GIF inline** with provenance. Both run through
  `job_submit`/`job_status`. Ships like a real capability: two Mechanism Cards
  (`matrix_free_identifiability` `NUDGE-METHOD-015`; the `oed` tool folded into
  `optimal_experimental_design` `NUDGE-METHOD-014`), a standing decoy battery (a well-constrained
  model must NOT be flagged sloppy; a rank-deficient one must abstain), tests
  (`tests/mcp/test_identifiability_oed.py`), and `NUDGE-LIM-027` (registry scope; arbitrary
  models remain a plain `import nudge` library path). Plus a self-contained **A/B "without-NUDGE"
  package** (`scripts/demo_ab/`): the same AD-QSP forward model + synthetic cohort as a NumPy
  module with no `nudge` dependency, for a fair raw-agent comparison (`NUDGE-LIM-026`). Additive;
  frozen `fit.py`/`core/` untouched. MEASURED via a local MCP client: `identifiability` returns
  the verdict + inline figure + `fig.py` on multiple models (glv sloppy-but-predictive,
  linear_pathway/logistic well-constrained, decoys correct); `oed` returns the design + measured
  gain + inline GIF (logistic ×49 CRLB, ad_qsp ×259 CRLB / corr 1.000→cond 22).

- **Flagship AD amyloid-β QSP clinical demo — matrix-free population identifiability +
  gradient OED on a REAL, published, open model (`nudge.mechanisms.ad_qsp`, `NUDGE-LIM-026`).**
  Points NUDGE's white-box machinery — matrix-free population identifiability (`NUDGE-LIM-023`)
  and gradient-based OED (`NUDGE-METHOD-014`) — at the **Proctor et al. 2013** Alzheimer's-disease
  amyloid-β QSP model (PLoS ONE 8(9):e73631, PMID 24098635; BioModels `BIOMD0000000488`, **CC0**),
  calibrated against a **synthetic ground-truth cohort** (no patient data). A deep-research
  subagent downloaded + machine-parsed the SBML (GO; CC0 confirmed); the **full 64-state**
  network (Aβ + tau + p53/Mdm2 + ubiquitin–proteasome + microglia — 73 parameters, 112
  reactions) is transcribed faithfully into `nudge.mechanisms._proctor2013` by an offline
  MathML→JAX generator (`scripts/vv/gen_proctor2013.py`). **Action 2** (`scripts/demo_matrix_free_scale.py`):
  population-scale calibration (N subjects × 12 subject-specific parameters → thousands of free
  params at fixed state) — MEASURED, dense `jacfwd` **OOM-killed at n_free≥1000** under a 2.5 GB
  cap while the **matrix-free FIM matvec stays flat at ~0.57 GB** and returns the same verdict
  (the honest rank-deficient regime → certified `unidentifiable`, `NUDGE-LIM-023`); single-subject
  identifiability flags the plaque-growth **gain `k_pg` / threshold `K_pg`** and disaggregation
  `k_dis` as the **sloppiest** (cond ≈7e8, span 8.8 decades). **Action 3**
  (`scripts/demo_gradient_oed.py`): the antibody-binding `k_on` ⇄ microglial-clearance `k_gl`
  pair is **perfectly confounded** by a naive baseline+end amyloid-PET schedule (corr ≈1.000),
  and gradient OED (D-optimality) resolves it — MEASURED **×220 CRLB / ×205 smallest-eigenvalue**
  improvement (local OED at θ₀, `NUDGE-LIM-024`) — with the 95%-confidence-ellipse-collapse GIF
  (`nudge.viz.oed`). **Honesty (`NUDGE-LIM-026`):** synthetic cohort (never real patients) +
  demo-scaled dimensionless constants (the published stiff seconds-to-years parameterization
  cannot be integrated by explicit RK4 — the published topology + rate-law forms are preserved).
  Additive; frozen `fit.py`/`core/` untouched. Tests: `tests/mechanisms/test_ad_qsp.py`;
  FINDINGS "AD amyloid-β QSP clinical demo".

### Changed

### Fixed

- **Fail-safe red-team P7: the multi-operating-point breaker
  (`attribute_lyapunov_multi`) resolved a threshold-DOMINATED large-gain perturbation to a
  CONFIDENT WRONG `threshold` (`NUDGE-LIM-025`).** A true gain knockdown (Hill n 4→1.5) whose
  perturbed condition slides through the saddle-node fold (monostable at one operating point,
  at the fold at the other) is threshold-dominated in the LNA moments, so the second operating
  point never breaks the gain⇄threshold degeneracy — yet the pure NLL-gap resolution (guarded
  only on the WT/control side, `NUDGE-LIM-017`) resolved `threshold` with a gap ≈1.7 ≫ 0.03
  (2/2 seeds; `scripts/redteam/lyapunov_multi_gain_threshold_hole.py`). Fixed (additive in
  `inference/lyapunov.py`; frozen `fit.py`/`core/` untouched) by a MEASURED **identifiability
  gate**: after a bare mechanism resolves, NUDGE fits the joint (winner, runner-up)
  two-mechanism model and reads its Laplace posterior (`uncertainty.laplace_posterior`),
  abstaining if a runner-up is identifiable and displaced from its no-change value beyond
  `_CONTAM_MARGIN=0.5` log-units (measured separator: genuine resolutions ≤0.12 vs the hole
  ≈1.0). Plus a graceful "bistability lost" degradation — a monostable operating point returns
  `("unresolved", {})` instead of a `ValueError`. **CLOSED** (n=1.5 → `unresolved`, 0
  confident-wrong; K=2.0 still resolves `threshold`; n=2.4 still abstains); residual **BOUND**
  (locked strict-xfail): a genuinely-degenerate large-gain perturbation can only be abstained,
  never positively resolved to `gain`. Regression-locked by
  `tests/inference/test_lyapunov_multi_uq_gate.py`; FINDINGS §P7.
## [0.2.0] — 2026-07-13

### Added

- **MCP figure transport — inline base64 for the Claude Science reality
  (`mcp render_figure` + `service.render_result` + `nudge.viz.inline`).** Live-testing against
  the Claude Science connector proved that **file-path delivery is structurally impossible**
  there (the shared dir is mounted read-only → `EROFS`; the connector's own temp is invisible
  to the client) and that **MCP `resources/read` is not bridged** — so a figure can only come
  back as **inline base64**. `render_figure` now honours a `NUDGE_ENV` toggle: `=cloud` →
  inline (`image_base64` + `mime_type` + the regenerating `fig.py` `code` and the sidecar
  `data` inline as text, the artifact-provenance grain); otherwise → write to
  `NUDGE_ARTIFACT_DIR` (fallback tempdir) and return `image_path`/`code_path`/`data_path`
  (`png_path` kept as a back-compat alias). **GIF size discipline** (`viz/inline.py`):
  downscale + frame-limit + tight-palette re-encode with a **never-inflate guard** (keep the
  compressed bytes only if smaller than the original), a ~1.5 MB base64 hard cap, and — above
  the cap — a **static final-frame PNG preview** (with `image_base64_omitted_reason`), never a
  silent truncation. Verified end-to-end through a local MCP client in both env modes (static
  PNG 122,752 chars; the constitutive-flip GIF 343,180 chars, never inflated; a synthetic
  13.6 MB GIF → PNG-preview fallback; an abstaining animation keeps `abstained=True` through
  the transport). Frozen core (`fit.py`/`core/`) untouched.
- **Async job pattern — the ~60 s connector-cap breaker (`mcp job_submit` / `job_status`).**
  The Claude Science connector kills any single tool call over ~60 s, but a covariance
  `attribute`, an OED optimization, or the `constitutive` demo (~64 s) legitimately exceeds
  that. `job_submit(tool, args_json)` runs any NUDGE tool in a `ThreadPoolExecutor` and returns
  a `{job_id}` in <1 s; `job_status(job_id)` polls → `running` (with `elapsed_s`) / `done`
  (with the tool's real `result`) / `error`. JAX releases the GIL during XLA, so the worker
  doesn't block the event loop, and each call stays well under the cap while the real compute
  spans however long it takes. Reuses the real FastMCP dispatch (no impl duplication). Verified
  with a local MCP client: a slow demo (~27 s) submits in 0.00 s, polls at ~1 ms/call, and
  delivers the real result. The server also advertises a connector usage note via FastMCP
  `instructions`. Heavy tools documented; fast tools (`list_mechanisms`, `dose_response`,
  `explain_abstention`, `get_mechanism_card`, `diagnose_abstention`) stay synchronous.
- **The animation battery — `nudge.viz.animate` generic `build_animation` dispatch.** The
  animation engine grows from the single constitutive-flip GIF to a **battery of ten
  animators**, each an additive `build_animation` in its own renderer module (mirroring the
  static `build`) + an `_ANIMATORS` registry entry + a render + abstention-overlay-fires test,
  each authored small (~24–28 frames, downscaled, tight palette) to ride the inline transport:
  `constitutive` (the LIM-006 flip), `oed` (measurement times sliding into the transient + the
  (α,β) 95% confidence ellipse collapsing over gradient steps — `optimize_design(...,
  capture_phi=True)` captures the design φ per step), `robustness` (the dial climbing 0→1 + the
  three channels rising + the 1-node potential well flattening two basins → one), `aggregation`
  (the **gauge orbit** — the three microscopic constants swing while the mass-fraction curve +
  the composites κ, λ stay identical, measured gauge Δ ≈ 1e-7; the sharpest honesty visual),
  `temporal` (the gLV community diverging from its reference under the antibiotic pulse),
  `multi_reporter` (SINGLE abstains → JOINT resolves as reporters are added), `identifiability`
  (perturb along the sloppy vs the stiff eigenvector — predictions barely move vs swing),
  `design` (the intervention trajectory to target + the safety dial flagging HIGH RISK at the
  fold), and `dose_response` (the Hill curve traced as the dose sweeps; `cross_modality`
  reuses it). Kinds with no natural frame variable (`differential` / `diagnose` /
  `attribution` / `epistasis`) are deliberately NOT animated. Every animator stamps the
  **abstention overlay per-frame off the result's own verdict** (a positive call is never
  drawn where the fit abstained) and ships the `fig.py` + sidecar provenance so the GIF
  replays with no re-fit. New enriched-demo helpers (`service.oed_animation_demo` /
  `robustness_animation_demo` / `fibrillization_animation_demo` / `temporal_animation_demo`)
  compute the sweep/orbit/trajectory frame sequences in the analysis layer; `viz` only READS
  them. Additive / opt-in — `fit.py`/`core/` untouched.
- **Claude Science connector guide — the inline-transport + job-pattern findings
  (`docs/user_guide/claude_science.md`).** Documents the EROFS/inline-only reality, the
  `/usr/bin/env NUDGE_ENV=cloud …` single-command-field trick, the `job_submit`/`job_status`
  pattern for the ~60 s cap, an agent-notes checklist (figures arrive inline → decode + don't
  echo the blob → attach the code/data provenance → wrap heavy calls), and matching
  troubleshooting rows. `design/VISUALIZATION_DESIGN.md` refreshed (the animation battery +
  the two-transport `render_result` schema).

### Changed

- **`service.render_result` return schema (0.2.0).** Now transport-aware: always
  `{transport, kind, caption, abstained, mime_type, code, data}`; inline adds `image_base64`
  (+ `image_base64_omitted_reason`, `compression_note`), path adds
  `image_path`/`code_path`/`data_path` (+ the `png_path` back-compat alias). The CLI figure
  paths pin `transport="path"` so local file output is unchanged.

### Fixed

## [0.1.0] — 2026-07-12

### Added

- **Fail-safe red-team FINAL-SWEEP — P5: a SMALL multiplicative perturbed-scale confound in
  `differential` FIXED; the free-affine EARN guard closes the whole affine confound class
  (`NUDGE-LIM-016` sharpened).** The final full sweep found the P4 gate 4c's measured blind gap: a
  **SMALL** uniform multiplicative factor `c ≈ 1.15–1.25` on ONE context's PERTURBED cells (its
  control clean) fakes a confident **`gain-diff` / `ceiling-diff`** where the truth is
  `no-difference`, slipping gate 4c two ways — at small `c` the joint BIC assigns the scale to the
  **gain (n)** channel (which the ceiling-scoped gate 4c never checks), and when it wins the
  ceiling channel its `off_scale` ≈ 1.14–1.29 lands in the **`(1.18, 1.30]` blind gap** between the
  genuine-ceiling maximum (1.18) and gate 4c's upper cut (1.30). Verified 8 confident-wrong / 4
  seeds (`scripts/redteam/differential_small_mult_gain_hole.py`), 3 confident-wrong reproduced
  through the shipped path at 2 seeds (`scripts/vv/p5_measure.py`). **Root cause + fix (measured,
  the STATE.md principle "guard the identifiability, not the confound — never a calibrated band"):**
  P1 (additive), P4 (large multiplicative) and P5 (small multiplicative) are ONE class — a
  per-condition **affine** `y = s·x + o` on one context's perturbed cells — and per-magnitude
  OFF-cluster bands each leave the next magnitude a blind gap. Rather than a fourth band, a new
  classifier gate (**4d**) adds the affine `(s, o)` as a **FREE nuisance** on the perturbed context
  and, before ANY positive `*-diff`, abstains unless the BIC-winning knob still **EARNS** its
  parameter over a pure-affine null (the profiled ΔBIC `earn`, min over both directions, ALL
  winners). The confound family is by construction inside the free-affine null's span, so no
  `(s, o)` lets the knob earn: **measured earn ≤ −2.1** across the uniform-affine sweep (incl. P5's
  interior AND the `(1.18, 1.30]` gap) vs **+59 … +616** for a genuine gain/ceiling; margin 6.0
  (`FINDINGS` §P5). The refit runs only for a candidate-positive winner (abstentions stay cheap);
  purely additive-abstention (can only turn a positive into an abstention). **CLOSED the whole
  UNIFORM per-condition affine class (P1/P4/P5)** — 0 confident-wrong, every positive control
  preserved (genuine `ceiling-diff` ×1.4/×2.0 earn +116/+616, `gain-diff` ×0.55 earn +59), no blind
  gap. **Honest residual BOUNDs:** a **non-uniform** (above-median-only) perturbed-side scale is
  observationally identical to a genuine ceiling and needs an independent inert-feature anchor
  (`design/PERTURBED_CONFOUND_STRATEGY.md`); the gate-4c deflation ceiling-reduction sacrifice is
  unchanged. Ported from the measured prototype `_proto_nuisance` (guard B). Regression-locked by
  the P5 decoy (`test_decoy_small_multiplicative_perturbed_scale_abstains`, the 3 exact
  confident-wrong cases + the factor-1 control), the gate-4d contract tests (which **replaced** the
  now-falsified `test_classify_off_scale_guard_is_ceiling_scoped`), and the no-over-abstention locks
  (`test_genuine_ceiling_earns_over_the_affine_nuisance` / `..._gain_...`). The P4 "INFLATION is
  CLOSED" wording is corrected everywhere (it was measured only on `c ≥ 1.5`) in the module
  docstring, `FINDINGS` §P4, `NUDGE-LIM-016`, `design/STATE.md`, and below. Frozen core untouched
  (`fit.py` / `core/` unchanged).
- **`nudge.viz` figure battery — a renderer per mechanism result type, honest by
  construction (`feat/viz-renderers`).** The opt-in figure layer grows from the single
  flagship dose-response slice to the full surface: `attribution` (the core
  `AttributionReport` across operating points — per-op verdict chips, first-class SKIP cells,
  and the joint breaker's restricted-NLL profile), `identifiability` (the Fisher / sloppiness
  eigenvalue spectrum + the *naive-vs-measured* verdict, so **sloppy-but-predictive** reads as
  usable rather than an over-abstention — off `inference.sloppiness`), `epistasis`,
  `differential`, `multi_reporter`, `temporal`/gLV, `aggregation`/fibrillization,
  `constitutive`, `diagnose`, `design`, `oed`, `cross_modality` (which REUSES the
  dose-response Hill panel), and `robustness` (the 0–1 bifurcation-proximity dial). Each
  renderer reads only the frozen
  result dataclass / its `*_to_dict()` / demo dict (never re-fits, never imports the fit
  engine), ships a standalone `fig.py` + data sidecar (provenance replay), and — the
  load-bearing guarantee — inherits the **automatic abstention overlay**: the render pipeline
  stamps the hatch + "I CAN'T TELL" banner off each panel's OWN verdict, so a positive call
  can never be drawn where the fit abstained (a per-kind test locks this). Added a shared
  **collision-aware placement layer** (`viz/layout.py`) so the abstain banner, the K-line
  label, and the legend land in data-free regions in both light and dark themes (fixing the
  overlap complaint). Added an **animation engine** (`viz/animate.py`, `render(..., animate=
  True)` → Pillow GIF, no ffmpeg): the flagship **constitutive-flip** shows the circuit-`n`
  profile going FLAT → `n=1` rejected as the constitutive control switches on
  (`NUDGE-LIM-006`), staying flat + hatched when the fit abstains. Cross-modality variant
  abstentions (`non-responsive`, `inconclusive`) were added to the shared `ABSTAIN_CALLS` so
  their overlay fires; the single-condition `gain_or_threshold` confound and the
  `unidentifiable` sloppiness verdict were added too, so the attribution + identifiability
  overlays fire. **CLI + MCP surface:** a `nudge viz KIND [--demo | --json FILE] [--out]
  [--theme light|dark] [--animate] [--no-fig-code]` verb and a mirrored MCP `render_figure`
  tool, both over the shared `service.render_result` seam (no duplicated render logic), plus a
  zero-setup demo per kind (`nudge.viz.demo.demo_result`, reusing the genuine `service.*_demo()`
  synthetic analyses where they exist — `attribution`/`identifiability`/`constitutive`/
  `temporal`/`oed`/`aggregation`/`dose_response` — and illustrative layout examples for the
  rest). Additive/opt-in — `fit.py`/`core/` untouched. `tests/viz/`, `tests/cli/test_cli.py`,
  `tests/mcp/test_server.py`.

- **Matrix-free identifiability / sloppiness diagnostic — scales past the dense-`jacfwd` OOM
  (`NUDGE-LIM-023`).** The sloppiness diagnostic (FIM = `JᵀJ/σ²` eigenspectrum →
  `well-constrained` / `sloppy-but-predictive` / `unidentifiable`) gains a **matrix-free** path
  that computes the spectrum + verdict using ONLY matvec products `JᵀJ·v` — one `jax.jvp`
  (`J·v`) composed with one `jax.vjp` (`Jᵀ·w`) — **never materializing** the sensitivity matrix
  `J = ∂(observables)/∂θ`, so its memory is O(n_params + n_obs), not O(n_obs·n_params). The
  dense `sloppiness_diagnostic(jac_log, …)` API is UNCHANGED; the additions are
  `sloppiness_diagnostic_matrixfree` / `analyze_model_matrixfree` / `fim_matvec`
  (`src/nudge/inference/sloppiness.py`) and `ode_identifiability` / `ode_trajectory_predict_fn`
  (`src/nudge/inference/adjoint.py`, large-ODE end-to-end). **Verified equal to the dense
  diagnostic bit-for-bit** on the validated cases (same label / eigenvalues / null direction:
  the sum-of-exponentials sloppy model and the `A·e^{-(k₁+k₂)t}` unidentifiable model);
  `fim_matvec` reproduces `JᵀJ/σ²·v` to rtol 1e-8. **MEASURED scaling**
  (`scripts/vv/sloppiness_scaling.py`, 77-state gLV): dense `jacfwd` OOMs at ~4000–6000 free
  params (systemd `MemoryMax=2.5 GB` cap; peak RSS grows ∝ n_params) while matrix-free stays
  **flat at ~0.42 GB, ~3 s to 6000 params** with the same verdict. **Honest bound
  (`NUDGE-LIM-023`, fail-safe; hardened by P6 — see Fixed below):** a matrix-free Krylov solver
  cannot certify the SMALLEST FIM eigenvalue (the sloppy/near-null direction) of an
  ill-conditioned FIM — so the iterative path certifies `unidentifiable` via shape rank
  deficiency (`n_params > n_obs`), `auto` DEFERS to the exact dense-via-matvec reconstruction up
  to `dense_below=2048`, and above that an inverse-iteration null probe catches an isolated null
  while the path **abstains rather than assert identifiability it cannot verify**; the exact
  dense-via-matvec route (`method="dense"`) is the definitive verdict for moderate `n_params`.
  Additive/opt-in — never touches frozen `fit.py`/`core/`.
  `tests/inference/test_sloppiness_matrixfree.py`; FINDINGS "Matrix-free identifiability".

- **Gradient-based Optimal Experimental Design — the differentiability moat
  (`NUDGE-METHOD-014`, `NUDGE-LIM-024`).** The white-box advantage a black-box ODE solver
  cannot offer: because NUDGE's forward model is **differentiable**, the Fisher-information
  design criterion is itself a differentiable function of the *experiment*, so
  `∂criterion/∂φ` is available by autodiff (straight through the ODE solve *and* the FIM
  assembly) and we **gradient-ascend the measurement schedule φ to the exact optimal
  experiment φ\***. A black-box solver has no `∂/∂φ` — it can only grid-search (exponential in
  the design size). New `nudge.inference.oed` (additive; self-contained RK4 `lax.scan`;
  touches neither `fit.py` nor `core/`) with D-/A-/E-optimality and a **targeted
  reciprocal-CRLB** objective. Makes the gLV growth⇄self-limitation (α⇄βᵢᵢ) directional
  abstention (`NUDGE-METHOD-012`) **actionable** — "sample the transient to break the tie"
  becomes a computed schedule. **MEASURED** (logistic growth, target α): a naive
  near-equilibrium design is near-singular (cond 136); the gradient-optimal design puts
  samples in the transient → growth **CRLB 31.5× better, FIM smallest eigenvalue 17.8×
  better**; all three objectives agree; a 3-taxon gLV community resolves **600×**. Honest
  white-box-vs-grid check (the gradient lands on a fine grid's optimum from either start;
  guaranteeing the optimum over `m` free times costs `rᵐ` evals — the moat is the scaling) +
  the **local-OED** caveat (gains measured at θ₀, not extrapolated; `NUDGE-LIM-024`).
  `nudge oed` CLI + `service.oed_demo` + Mechanism Card + `tests/inference/test_oed.py`
  (7 fast + 5 slow) + `notebooks/Optimal_Experimental_Design.ipynb`.
- **Protein aggregation / fibrillization attribution (`NUDGE-METHOD-013`, `NUDGE-LIM-021`)
  — the efficiency demo + a third dynamical-systems domain.** NUDGE analyzes an amyloid
  aggregation curve (the sigmoidal ThT / polymer-mass trace) by fitting the filament master
  equation's principal moments (`dP/dt = k_n·m^{n_c} + k_2·m^{n_2}·M`, `dM/dt = 2·k_+·m·P`;
  Knowles 2009 / Cohen 2013 / Meisl 2016 / Michaels 2020) with a self-contained
  differentiable RK4 `lax.scan` (no `diffrax`; touches neither `fit.py` nor `core/`). In
  ONE deterministic call it returns the two IDENTIFIABLE composites **κ = √(2·k_+·k_2·
  m^{n_2+1}) ≈ 1** and **λ = √(2·k_+·k_n·m^{n_c}) ≈ 0.01**, and the MEASURED
  non-identifiability of the three microscopic constants — an **exact continuous gauge
  symmetry** `(k_n, k_+, k_2) → (k_n/α, α·k_+, k_2/α)` (Fisher/Laplace condition number →
  ∞, null direction `[+0.577, −0.577, +0.577]`, numerical gauge check ~1e-16). A control
  LLM agent took 12.2 min / 28 turns / 6 scripts to hand-derive the same answer. A
  concentration series ALONE stays degenerate; a series + a seeded/elongation anchor (the
  Meisl discipline) resolves all three (0 confident-wrong). An inhibitor is attributed to
  the microscopic step it lowers (primary / elongation / secondary nucleation) from the
  composite log-ratios, or abstained on — 0 confident-wrong across the battery.
  `src/nudge/mechanisms/fibrillization.py`, `tests/mechanisms/test_fibrillization.py`
  (13 tests), `nudge fibrillization` CLI + `service.fibrillization_demo` + Mechanism Card +
  `notebooks/Aggregation_Kinetics.ipynb`; FINDINGS "Protein aggregation / fibrillization".

- **Fail-safe red-team ROUND 3 (design safety gate) + fix (`design/FAILSAFE_REDTEAM_3.md`,
  HOLE 3 → `NUDGE-LIM-013`).** A third adversarial pass found that `design()`'s bifurcation
  **safety gate** flagged `high_risk_of_instability` on a *relative* proximity rise only
  (`delta > margin`, default 0.15) and **never** checked the **absolute** `proximity_after`
  against the shipped near-fold cut `bifurcation.NEAR_FOLD = 0.55`. So an intervention that
  pushed a robust switch (proximity 0.500) **across** 0.55 into the near-fold regime by a
  **sub-margin** increment (→ 0.589) was cleared "safety: OK, stays away from the fold" — a
  confident-wrong SAFETY label on a **proposal** (the highest-harm output), contradicting
  `classify_robustness` on the identical circuit. **FIXED (additive, `design/invert.py`;
  frozen core untouched):** the gate now fires on `delta > margin` **OR** `proximity_after
  >= NEAR_FOLD` (an absolute check reusing the *existing* `NEAR_FOLD` constant, so the safety
  gate and `classify_robustness` never disagree; recorded as `SafetyReport.near_fold`), routes
  the near-fold case through wording that agrees with `classify_robustness`, and — closing the
  aggravating factor — carries the one-sided-LOWER-bound caveat (`NUDGE-LIM-012`) on the SAFE
  ("OK") reason branch too. Measured: the hole case now flags near-fold high-risk (0
  confident-wrong); a positive control (a genuinely-robust intervention below `NEAR_FOLD`) is
  still cleared "OK" (no over-abstention). Regression-locked by three `tests/design/
  test_invert.py` cases; the deterministic repro
  (`scripts/redteam/design_safety_gate_absolute_proximity.py`) now exits "no hole".
- **Fail-safe red-team P1-RESCAN — P4: the MULTIPLICATIVE perturbed-condition scale confound in
  `differential` FIXED (`NUDGE-LIM-016` sharpened).** A re-scan of the P1 fix found the sharpest
  differential confound yet: a constant **multiplicative factor `c` on ONE context's PERTURBED
  cells only** (its control clean) aliases a genuine **ceiling (`v_max`) difference 1:1** (both
  multiply the ON mode) and slips past **both** earlier guards — the control-keyed `depth_ratio`
  stays ≈ 1 (gate 2 blind) **and** a factor scales the near-zero OFF *baseline* to near-zero, so the
  additive `off_shift` stays ≈ 1 (gate 4b blind). Result: a confident spurious **`ceiling-diff`**
  where the truth is `no-difference`, both inflating and deflating (verified,
  `scripts/redteam/differential_multiplicative_confound.py`, **9 confident-wrong across 2 seeds**).
  **Fix (measured, ceiling-scoped):** a new classifier gate (4c) abstains (`unresolved`) before a
  `ceiling-diff` when either context's perturbed **OFF-cluster SCALE** (`off_scale` = the
  median-absolute-deviation of the below-median-activity cells, perturbed vs its own control) leaves
  the measured band `[0.80, 1.30]` — a multiplicative factor dilates the OFF-cluster spread by `c`
  (`off_scale` ≈ `c`) while a genuine `v_max` difference leaves it anchored at basal. The guard is
  **ceiling-scoped** (a global scale is degenerate with `v_max` specifically), so a genuine
  gain/threshold difference reshapes the distribution and is **untouched** (no over-abstention).
  **INFLATION is CLOSED for LARGE factors** (`c ≥ 1.5`; the small-factor interior slips gate 4c —
  that is P5, closed by the free-affine gate 4d, see the P5 entry above) — a clean measured gap
  (genuine ceiling ×1.4–×4 ≤ 1.18; every inflating confound `c` ≥ 1.5 ≥ 1.43; midpoint 1.30 is the
  upper guard; `FINDINGS` §P4). **DEFLATION is
  BOUNDED** — a genuine ceiling *reduction* collapses the switch toward monostable and shrinks the
  OFF cluster into the same band as a deflating scale (indistinguishable), so the lower guard abstains
  on both, killing the deflating confound at the honest cost of no longer resolving a strong genuine
  ceiling reduction (a per-context multiplicative scale without an independent depth anchor cannot be
  separated from a ceiling change). Re-validated through the shipped path: **0 confident-wrong across
  2 seeds** (inflating + deflating), and every positive control still resolves (genuine
  `ceiling-diff` ×1.4/×2.0, `gain-diff`, `no-difference`, the additive P1 confound still caught by
  gate 4b). Regression-locked by a decoy (`test_decoy_multiplicative_perturbed_scale_abstains`, 8
  cases + the factor-1 positive control + a genuine-ceiling positive control + a **strict-xfail bound
  lock** for the sacrificed genuine ceiling reduction) and 4 fast unit tests; the module docstring,
  the Mechanism Card, `design/STATE.md`, `README.md`, and `NUDGE-LIM-016` all name the multiplicative
  channel so nothing implies differential's perturbed-side confounds are all closed. Frozen core
  untouched (`fit.py` / `core/` unchanged).
- **Fail-safe red-team ROUND 3 hardening — P1: the additive perturbed-condition offset
  confound in `differential` FIXED (`NUDGE-LIM-016` sharpened; `design/FAILSAFE_REDTEAM_3.md`
  HOLE 1).** The differential depth guard keys `depth_ratio` on the two **controls**, so it was
  structurally blind to a constant **additive / ambient offset on ONE context's PERTURBED cells
  only** (its control left clean): `depth_ratio` stays ≈ 1 and the depth gate never engages, yet
  the offset shifts that context's perturbed modes and *compresses* their separation, which the
  joint LNA-BIC misreads as reduced cooperativity — a confident spurious **`gain-diff`** where the
  truth is `no-difference` (verified, `scripts/redteam/differential_additive_confound.py`, **3
  confident-wrong across 2 seeds**). **Fix (measured, one-sided):** a new classifier gate (4b)
  abstains (`unresolved`) before any positive call when either context's perturbed OFF baseline is
  inflated above its own control beyond `off_shift_max = 2.5` — the fingerprint of an additive
  offset (it TRANSLATES the OFF baseline up), a **measured separator** (every confident-wrong offset
  had `off_shift ≥ 2.99`, the strongest genuine gain/ceiling/threshold difference only `≤ 1.96`;
  `FINDINGS` §P1). This promotes the previously non-load-bearing `off_shift` diagnostic to
  **one-sided load-bearing** on its inflation side. **CLOSED for the demonstrated (inflating) offset,
  BOUNDED in general** — a *deflating* perturbed-only offset (dropout-like) aliases with a genuine
  knob reduction and stays an unguarded, documented residual. Regression-locked by a decoy
  (`tests/inference/test_differential.py::test_decoy_additive_perturbed_offset_abstains` + the
  offset-0 positive control + 3 one-sided-guard unit tests); the module docstring, the Mechanism
  Card, and `NUDGE-LIM-016` are corrected (the OFF-baseline is no longer described as
  non-load-bearing). Frozen core untouched (`fit.py` / `core/` unchanged).
- **Fail-safe red-team ROUND 3, P2 fix — the multi-reporter per-condition batch/depth scale
  (`NUDGE-LIM-014`; `scripts/redteam/multi_reporter_batch_confound.py`).** A uniform
  multiplicative scale `c` on the whole PERTURBED reporter panel (a batch / sequencing-depth /
  instrument-gain difference between the control-condition and perturbed-condition measurement)
  is invisible to the control-only consistency guard and the module applied no per-condition
  depth normalization, so it aliased 1:1 onto a shared latent-ceiling change `A = c` — a
  confident `ceiling` (CI excludes 0, margins ≫ thresholds) where the truth is **no-effect**
  (reproduced 6/6 across 2 seeds × 3 factors, robust to realistic floors).
  - **CLOSED (measurable floors) by a ceiling-scoped FLOOR-CONSISTENCY gate.** A *genuine*
    ceiling scales only the ON term `gain·A·f` and leaves each reporter's OFF baseline (floor,
    dose→0) **fixed**; a *batch* scales the whole perturbed signal so every reporter's floor is
    rescaled by `c`. NUDGE now measures `off_on_coupling = log(perturbed/control OFF baseline) /
    log(A)` — ≈ 0 for a genuine ceiling, ≈ 1 for a batch — and abstains `unresolved` before a
    `ceiling` call when `off_on_coupling > 0.5` (the physical midpoint). Measured median
    separation: genuine ceiling 0.00–0.10 vs batch 0.67–1.00, at BOTH tiny (0.0,0.02) and
    realistic (0.2,0.6) floors. Additive (`multi_reporter.py` only; frozen core untouched); the
    genuine `ceiling`/`threshold`/`gain` positive controls, `no-effect`, `off-model`, and
    single-reporter abstentions are all preserved (0 confident-wrong, no over-abstention on the
    positive controls).
  - **Residual BOUND (locked, not closed).** On a (near-)ZERO-floor panel the perturbed OFF
    doses are pure ON-leakage, which a batch and a genuine ceiling scale identically — genuinely
    inseparable. NUDGE detects this (`floor_measurability < 0.6`) and abstains `unresolved` on
    BOTH (never a confident-wrong, at the cost of over-abstaining a real floorless ceiling);
    separating them needs an independent depth anchor (spike-in / housekeeping / no-response
    reporter) NUDGE does not yet ingest. `NUDGE-LIM-014` sharpened; measured in
    `scripts/vv/FINDINGS.md` (P2); regression-locked by a batch-scale decoy + the
    genuine-ceiling positive control + a strict-xfail floorless-bound marker
    (`tests/inference/test_multi_reporter.py`).
- **Fail-safe red-team ROUND 2 (core engine) + two fixes (`design/FAILSAFE_REDTEAM_2.md`).**
  A second adversarial pass targeting the core engine found **2 more verified confident-wrong
  holes** — both in work shipped the same day — and both are now fixed:
  - **Constitutive capture-scale confound → FIXED honesty contract + LOCKED (`NUDGE-LIM-019`).**
    The constitutive control is a *separate* population; a control-vs-population
    **capture-efficiency mismatch** (~0.5×, a routine single-cell batch difference — the module
    applies no relative-depth normalization between the two populations) mis-anchors the reporter
    `Vmax` and makes NUDGE assert `biological-switch` on a **linear** circuit (the `NUDGE-LIM-006`
    artifact resurrected; 3/3 seeds). It slipped past the module's own `is_confident_wrong`
    (scoped to bare-knob calls only). Fixes: the honesty contract is broadened
    (`ConstitutiveResult.asserts_biological_switch` surfaces the falsifiable positive), the
    framing is corrected from "structurally fail-safe / 0 confident-wrong" to **adversarially
    bounded** (`biological-switch` is valid only when the control shares the population's capture
    scale — a stated precondition), the confound is **locked as a strict-xfail decoy**, and the
    principled robustness fix (anchor to the switch-independent reporter floor / a spike-in) is
    designed as future work (`design/CONSTITUTIVE_CONTROL.md`, Option B).
  - **Near-fold multi-fit — the round-1 `NUDGE-LIM-017` hard margin was a KNIFE-EDGE → replaced.**
    A 3rd operating point at proximity 0.146 (just under the 0.15 margin) still flipped a true
    ceiling → confident `threshold` (gap 0.53). Measuring the corruption onset showed *why* a hard
    margin is scientifically invalid: the useful 2nd point (proximity 0.112) and a corrupting one
    (0.119) sit **0.007 apart**, and the onset is **non-monotonic** in proximity. The hard gate is
    replaced by **graded near-fold down-weighting** (a far near-fold point is weighted ~0 in the
    joint loss so it cannot corrupt the fit) **+ best-buffered-pair corroboration** (a bare
    mechanism is accepted only if the two most-buffered points confirm it, else abstain —
    threshold-free, closing the knife-edge). Measured after the fix: the {0.05,0.30} control still
    resolves the true ceiling, and every near-fold 3rd point (proximity 0.119 / 0.146 / 0.231)
    honestly ABSTAINS — 0 confident-wrong. Regression-locked by the knife-edge + near-fold decoys.
- **Temporal / Lotka–Volterra attribution — *same engine, new dynamical-systems domain*
  (`nudge.inference.lotka_volterra`, `NUDGE-METHOD-012`, `NUDGE-LIM-020`):** the first
  **temporal / trajectory-fit** capability, and the extensibility thesis made concrete.
  Everything else in NUDGE observes a steady-state *snapshot*; this points the *same*
  abstain-and-attribute philosophy at a **generalized Lotka–Volterra (gLV) microbial
  community** — `dxᵢ/dt = xᵢ(αᵢ + Σⱼ βᵢⱼxⱼ + εᵢ·u(t))` — whose parameter information lives
  in **trajectories**. Given a reference vs perturbed community under an antibiotic pulse,
  it BIC-selects **which single knob moved** — **growth (α) / interaction (β) /
  susceptibility (ε)** — or **abstains**. A self-contained differentiable RK4 `lax.scan`
  integrator (no `diffrax`); the fit loop is **re-instantiated in the module** (reusing
  `inference.losses.energy_distance` over per-timepoint replicate ensembles) so it touches
  **neither `fit.py` nor `core/circuit.py`** — the frozen core stays frozen. **The
  ε axis is the identifiable positive** (the drug pulse is a time-localized on/off
  contrast). **The α⇄βᵢᵢ pair is degenerate near equilibrium** (`Kᵢ=−αᵢ/βᵢᵢ`) and NUDGE
  **abstains** (`unresolved`), with the degeneracy **MEASURED** — a near-singular Laplace
  curvature on `(αₜ, βₜₜ)` (reusing `inference.uncertainty.laplace_posterior`, condition
  number ≫ 100, `|corr|→1`), exactly how NUDGE measures the gain⇄threshold degeneracy
  elsewhere. Attribution scores the reference→perturbed **contrast**, which cancels the
  baseline mean-bias so a null cannot be beaten by a spurious knob. **Fail-safe:
  recover-or-abstain, 0 confident-wrong** across the synthetic battery (ε recovers, a
  dense-transient growth change recovers, a self-interaction change + a near-equilibrium
  growth change abstain, a null makes no positive call —
  `tests/inference/test_lotka_volterra.py`). Two gLV decoys — the **α⇄βᵢᵢ confound**
  (a growth change near equilibrium that looks like an interaction change → must abstain)
  and a **no-perturbation null**. Real coda: **Stein et al. 2013** clindamycin→*C.
  difficile* (structured `needs_data`), which surfaces the honest abstention — *C.
  difficile*'s bloom is **interaction-mediated** (the paper's fitted ε≈−0.31 is near zero),
  the very α/β confound. Wired into `nudge lotka` CLI + `service.lotka_demo` + a Mechanism
  Card (`docs/mechanism_cards/lotka_volterra_attribution.md`) + `NUDGE-LIM-020` +
  `notebooks/Temporal_Ecology.ipynb`. Additive / opt-in.
- **Visualization module — `nudge.viz` (opt-in `[viz]` extra), first slice: the flagship
  dose-response dual panel.** An additive, provenance-carrying figure layer that turns
  NUDGE's honest *result dataclasses* into honest *pictures* from one `render(result,
  out=…)` surface — keyed off the existing frozen results (and their `*_to_dict()` dicts;
  dual-input). It only **reads** results (never re-attributes; never imports `fit`/`core`),
  lazily imports matplotlib (`Agg`, headless-safe), and — the load-bearing part — applies
  the **abstention overlay in `render()` itself, off the result's own verdict**, so a
  renderer *cannot* draw an abstention as a confident call; one-sided bounds
  (`spans_inflection = False`) draw as **open-ended arrows**, never closed error bars. The
  flagship `plot_dose_response` renders the real ESC-screen **OCT4 → `switch`** (n≈6.7,
  R²=0.99) beside the honest **NANOG → `unresolved`** abstention **in one frame** — the
  fail-safe thesis, visualized. Every figure ships the **Claude Science provenance grain**:
  `fig.png` + `fig.data.json` + a standalone, deterministic `fig.py` that regenerates the
  *exact* figure from the fit's output (no re-fit; verified **pixel-identical**), with a
  `self_contained=True` inline-data mode for Artifacts. Wired through a shared
  `service.render_result()` seam + an opt-in `--fig-out/--fig-code/--fig-theme/
  --fig-self-contained` flag on `nudge dose-response` (default text output **unchanged**).
  Embedded in `notebooks/OCT4_NANOG_Flagship.ipynb` via the one-call API (re-executed
  headless, 0 errors). matplotlib moved from `[dev]` into a `[viz]` extra (core stays
  matplotlib-free); `[dev]` depends on `[viz]` so existing V&V figures + notebooks keep
  working. Tests: `tests/viz/test_render.py` (PNG emitted; `fig.py` re-runs pixel-identical;
  the **honesty test** — a known abstention yields `FigureResult.abstained == True` with the
  overlay drawn; dual-input dataclass ⇄ dict; the real flagship marked `needs_data`). The
  remaining ~11 renderers + the LIM-006 constitutive-flip animation + the MCP `render_figure`
  tool are designed (`design/VISUALIZATION_DESIGN.md`) and land in later slices.

- **Constitutive-reporter calibration control — the `NUDGE-LIM-006` mitigation
  (`nudge.inference.constitutive`, `NUDGE-METHOD-011`, `NUDGE-LIM-018`):** removes a known
  **confident-wrong** failure mode. NUDGE assumes an *affine* reporter; a **nonlinear**
  (saturating / sigmoidal Hill) reporter over a *linear* circuit produces a pseudo-bimodal
  count distribution the affine-readout switch model can only explain by bending the circuit
  — a **confident false positive** (`NUDGE-LIM-006`, the sharpest bound on the fail-safe
  guarantee). Only the composition readout∘circuit is observed, so from one population the
  circuit Hill `n` and the reporter Hill `h` are unidentifiable — the profile over circuit
  `n` is **FLAT** (a graded `n=1` fits as well as a real switch). A **constitutive-reporter
  control** — the reporter driven at KNOWN activity doses, *bypassing the circuit* — anchors
  the readout using **readout parameters ONLY** (the load-bearing no-leak property:
  ∂(control loss)/∂(circuit params) ≡ 0, proven by `control_loss_circuit_gradient`). A
  profile likelihood over circuit `n` WITHOUT vs WITH the control then breaks the degeneracy:
  WITH the control, "no switch" (`n=1`) is **REJECTED** for a genuine switch (Δloss ≫ the
  flat span) → the ultrasensitivity is **biological**. **Fail-safe:** it NEVER emits a bare
  threshold/gain/ceiling — it turns the confident false positive into a correct
  `biological-switch` call **or** an honest `unresolved` abstention — and it does NOT
  point-identify the exact `n` (that needs a second anchor: an input titration / circuit
  dose-response; `NUDGE-LIM-018`). Validated on synthetic ground truth
  (`scripts/vv/constitutive_control.py`; FINDINGS "NUDGE-LIM-006 mitigation"): a true switch
  (`n=3`) through a nonlinear reporter (`h=6`) → `biological-switch` (n=1 rejection ≈0.026 vs
  a flat no-control span ≈0.001), a linear circuit (`n=1`, the LIM-006 hazard) → `unresolved`
  (n=1 rejection ≈0), 0 confident-wrong across seeds **on the clean-control validation** (the
  `biological-switch` verdict is adversarially bounded by the shared-capture precondition —
  `NUDGE-LIM-019`, see the red-team round-2 entry above). Additive / opt-in (never touches
  `fit()`'s default, the decoy battery, or the Lyapunov / epistasis paths); reuses the
  shipped Hill primitive + energy distance. Wired into `nudge constitutive` CLI + the
  `constitutive` MCP tool + `service.constitutive_file` + a Mechanism Card +
  `notebooks/Constitutive_Control.ipynb`.

- **Fail-safe red-team + two hardenings (`design/FAILSAFE_REDTEAM.md`).** An adversarial
  pass tried to make each capability emit a *confident, specific, WRONG* call past its
  abstention gates. It found **2 verified holes** (3 attacks HELD); both are now closed or
  locked:
  - **Hole 1 → FIXED (`NUDGE-LIM-017`).** A near-fold 3rd operating point corrupted the
    multi-point covariance breaker (`attribute_lyapunov_multi`) and flipped a true
    **ceiling → confident `threshold`** (gap ≈0.24–0.30 ≫ `resolve_margin`), because the
    only per-point trust gate (`lna_reliable`) trips solely at lobe *overlap* while a point
    *approaching* the fold is already biasing the joint fit. Fixed by gating the joint fit
    on the **bifurcation-proximity dial** (its two deterministic channels `lna_reliable`
    ignores): it abstains unless every operating point is well-buffered
    (`proximity ≤ well_buffered_margin`, default `0.15`) — the "well-buffered second point"
    caveat becomes an enforced precondition. `proximity = max(det, lobe) ≥ det`, so the gate
    can only *add* abstentions. Regression-locked by a near-fold decoy
    (`tests/inference/test_lyapunov_toggle_ssa.py`). Repro now reports 0 confident-wrong.
  - **Hole 2 → LOCKED + sharpened (`NUDGE-LIM-009`).** An **additive** (ambient / batch)
    count offset on the A+B signature genes is invisible to size-factor (multiplicative)
    normalization and — perfectly aligned with A+B, with no orthogonal batch covariate —
    fakes a confident `synergistic` (ΔBIC ~10³, 4/4 seeds), *and* mis-fires the off-axis
    neomorphic flag toward "emergent biology". No runtime gate is added (any gate sensitive
    enough would false-abstain on real synergy — unsafe); instead the confound is **locked
    as a strict-xfail decoy**, `NUDGE-LIM-009` is **sharpened** (additive-offset failure
    mode + the required orthogonal-covariate mitigation, bumped to *major / safety_relevant*),
    and the neomorphic note is **re-worded** so an off-axis residual can never be read as
    corroboration of the interaction (it is equally consistent with a batch artifact).

- **Comparative / differential attribution — WHICH knob differs between two contexts
  (`nudge.inference.differential`, `NUDGE-METHOD-010`, `NUDGE-LIM-016`):** given the SAME
  perturbation in two **contexts** (a drug-resistant vs sensitive line; donor A vs B;
  disease vs healthy), isolates whether the mechanistic difference is in the switch's
  **threshold** (`K`), **gain** (`n`), or **ceiling** (`v_max`) — a distinction linear
  differential expression structurally **cannot** make (a raised *ceiling* → more dose of
  the SAME drug; a rewired *gain / threshold* → a DIFFERENT class). Fits the two contexts
  **jointly** with a **shared-vs-per-context** parameter structure and **BIC-selects**
  which single knob must differ (`shared` / `ΔK` / `Δn` / `Δv_max`), reusing the shipped
  LNA Gaussian-mixture forward model + the BIC parsimony pattern; or abstains
  (`no-difference` / `unresolved`). **Confound guard (the load-bearing honesty point,
  `NUDGE-LIM-016`):** a sequencing-depth / batch difference aligned with the context axis
  is degenerate with a **ceiling** difference (`scale ⇄ v_max`), so depth is pinned PER
  CONTEXT from each context's OWN control (`calibrate_from_wt`), and when the two contexts'
  pinned depths **differ beyond a ratio** (a depth/batch difference aligned with the context
  axis) NUDGE **abstains** rather than risk a spurious `ceiling-diff` — *unless* the winner
  is a cleanly-resolved threshold / gain difference, which reshapes the distribution
  (orthogonal to a global scale) and survives.
  **Validated on synthetic ground truth (`FINDINGS` Phase 4j):** a Δv_max pair recovers
  `ceiling-diff` and a Δn pair recovers `gain-diff`; a no-difference pair reads
  `no-difference`; a ΔK pair recovers-or-abstains (threshold is hardest from a bistable
  snapshot — the measured hierarchy gain > ceiling > threshold, `FINDINGS` §2); and the
  depth-aligned-with-context confound **abstains `unresolved`**, **0 confident-wrong across
  seeds**. Additive / opt-in — it never touches `fit()` or the decoy battery. Wired into
  the `nudge differential` CLI verb + the `differential` MCP tool + `service.differential_*`
  + a Mechanism Card + `notebooks/Differential.ipynb`.
- **Hidden-node abstention — the honest differential (`nudge.inference.hidden_node`,
  `NUDGE-METHOD-009`, `NUDGE-LIM-015`):** turns a bare **`off-model`** verdict (or a fired
  diagnostic residual) into a legible **differential diagnosis** — it **enumerates** the
  candidate causes of an inadequate switch model, each with its evidence, the documented
  limitation / decoy it maps to, and the experiment that would distinguish it: **(1)**
  genuinely not-a-switch (the parsimony gate working, `NUDGE-LIM-005`), **(2)** a nonlinear
  measurement readout (`NUDGE-LIM-006`), **(3)** an off-target perturbation, **(4)** a
  wrong / misspecified topology (T0.5-2), **(5)** a batch / depth confound (`NUDGE-LIM-003`
  / `NUDGE-LIM-009`), and **(6)** a hidden node / unmeasured regulator (the off-axis /
  neomorphic residual, `NUDGE-LIM-009`). **The abstention half ONLY (the crux,
  `NUDGE-LIM-015`):** positive hidden-node identification is **not identifiable** from an
  off-model verdict — the six causes are observationally overlapping — so NUDGE **never**
  asserts a hidden node; the strongest it says is that an off-axis residual is *consistent
  with — but does not prove —* an unmeasured regulator, and the hidden-node hypothesis's
  rank is capped so it is never the lone leading answer. It is a pure **packaging /
  knowledge** layer built on `knowledge.explain` with **zero import of `fit`** — it
  consumes verdicts, never re-attributes, and never touches the decoy battery. Wired into
  the `nudge diagnose-abstention` CLI verb + the `diagnose_abstention` MCP tool +
  `service.diagnose_abstention` + a Mechanism Card + `notebooks/Hidden_Node_Abstention.ipynb`.
  The honesty guarantee (the report NEVER emits a bare positive hidden-node claim) is
  enforced in CI (`tests/inference/test_hidden_node.py`).

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

### Fixed

- **Matrix-free identifiability no longer mislabels an isolated structural null
  `well-constrained` (P6; `NUDGE-LIM-023`).** On a model with an isolated exact Fisher-null in
  an otherwise well-conditioned spectrum (`n_params ≤ n_obs`), `eigsh(which='SA')` converged
  *past* the null to the well-conditioned cluster; its pairs passed the Rayleigh residual (which
  verifies eigenpair-ness, not smallest-ness), so the iterative / `method="auto"`
  (`n_params > dense_below`) path returned `well-constrained` (`n_null=0`) on a
  provably-`unidentifiable` model — the single most dangerous mislabel (verified 6/6 across
  seeds, float64). Fixed additively in `src/nudge/inference/sloppiness.py`: (1) `auto` now
  DEFERS to the exact dense-via-matvec reconstruction up to `dense_below=2048` (raised from 256;
  measured affordable — ~18 s / 0.7 GB at n=2048, recovering the exact null at every size
  256–4096); (2) above that, an INVERSE-ITERATION null probe (shift-invert `(FIM+εI)⁻¹` via CG)
  reliably CATCHES the isolated null `eigsh('SA')` misses (measured: Rayleigh quotient ~1e-12–1e-10,
  ≤ the rank floor, where `eigsh` returned ~1e3; 0 false nulls on well-conditioned + sloppy controls), else the
  path ABSTAINS (`unidentifiable`, "cannot certify the smallest eigenvalue"). **CLOSED:** 0/6
  confident-wrong; positive controls (`well-constrained`, `sloppy-but-predictive`) still resolve
  via the dense path (no over-abstention); dense==auto agreement preserved. **BOUNDED:** above
  `dense_below` a genuine well-constrained model over-abstains (fail-safe; use `method="dense"`
  for the exact verdict). Frozen `fit.py`/`core/` untouched.
  `tests/inference/test_sloppiness_p6_structural_null.py`;
  `scripts/redteam/sloppiness_matrixfree_iterative_mislabel.py`; FINDINGS §P6.

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

- See `docs/known_limitations.yaml` for the tracked `NUDGE-LIM-*` bounds (each honest
  gap is a documented, often decoy-locked limitation — not a hidden one). Genuinely
  deferred for 0.1.0: real-data lock-ins for the newest capabilities (constitutive,
  multi-reporter, differential, fibrillization, OED, the gLV temporal path) and
  per-result provenance tracking (`provenance.py` is a Phase-0 stub schema).
