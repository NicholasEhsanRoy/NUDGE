# NUDGE — project state, roadmap & next-step plan (context-handoff)

**This is the single source of truth for "where are we and what's next."** Read it
first each session. Last updated 2026-07-08, after Phase 2 (proof of concept) +
overnight V&V calibration.

---

## 0. How to work here (operational essentials)

- **Toolchain: `uv`** (there is NO system `pip`). Run everything via `uv run`:
  - Tests: `uv run pytest -q` (fast lane). Slow lane: `uv run pytest -m slow`.
  - Lint/type: `uv run ruff check src tests scripts` · `uv run pyright src`.
  - Env is a local `.venv` (git-ignored); `uv pip install -e ".[dev]"` to set up.
- **Checks that must pass before every commit:** ruff, pyright, `pytest` (fast lane),
  and the three `scripts/check_*.py` validators.
- **Commits: always credit Claude.** Append `Co-Authored-By: Claude Opus 4.8
  <noreply@anthropic.com>`, write a real body saying what Claude did. See `CLAUDE.md`
  and the `/commit` skill. Commit + push on `main` (the user is fine with this).
- **Ruff/pyright are pinned** (`ruff==0.15.20`, `pyright==1.1.411`) so local == CI.
  `scripts/vv/*` has relaxed lint (throwaway analysis code).
- **Test lanes** (pyproject markers): default CI runs everything except
  `slow`/`validation`/`needs_llm`; `verification` + `decoy` run in default;
  heavy things are `slow` (scheduled lane). Real-data checks are `validation`+`needs_data`.
- **The approved structural plan** lives at
  `/home/nick/.claude/plans/great-let-s-now-work-ticklish-hamming.md` (outside the
  repo). The design docs (`design/WORKING_BACKWARDS.md`, `PITCH.md`,
  `GENERATOR_DESIGN.md`) hold the reasoning; `GENERATOR_DESIGN.md` especially — it
  has the two literature-review syntheses (count model + the bistability crux).
  `design/CONSTITUTIVE_CONTROL.md` documents the `NUDGE-LIM-006` readout-nonlinearity
  limitation + its validated constitutive-control mitigation (a stretch feature).

## 1. What NUDGE is (one paragraph)

Fits a compositional, differentiable gene-regulatory **circuit** ODE to single-cell
Perturb-seq counts and attributes each perturbation to a **mechanism** — does it
move a switch's **threshold** (K), **gain** (Hill n), or **ceiling** (v_max) — and
**abstains loudly** when it can't tell. Built on **MADDENING** (a differentiable
JAX graph-physics engine) — reuses its `ift_linear_solve` primitive and
`maddening.compliance` traceability, but NOT its `GraphManager` (see §4).

## 2. Current state — Phases 0–2 DONE (proof of concept closed + calibrated)

| Phase | Status | Key files |
|---|---|---|
| 0 Bootstrap | ✅ | `pyproject.toml`, CI `.github/workflows/`, `docs/known_limitations.yaml`, `scripts/check_*.py` |
| 1 Generative backbone | ✅ | `core/circuit.py`, `mechanisms/` (species, integrators, regulatory, readout), `data/synthetic.py`, `data/noise.py`, `data/ingest.py` |
| 2 Fit | ✅ (PoC) | `inference/losses.py`, `inference/fit.py`, `inference/classify.py` |
| V&V calibration | ✅ | `scripts/vv/` (harness + results + `FINDINGS.md`) |
| 3 Fail-loud | ◑ ~50% | gate logic + Tier-0.5 simulator + saddle gain gate + decoy battery started (`NUDGE-DECOY-001` telegraph, `NUDGE-LIM-001`) + **Laplace posterior uncertainty layer landed** (`inference/uncertainty.py`: curvature CIs + a degeneracy guard that reproduces the gain/threshold near-singular Hessian and abstains; FINDINGS "Laplace posterior"); verification suite + more decoys NOT built |
| 4 Validation + provenance | ◑ | **infra landed + two real-data results.** Generic backed-mode Perturb-seq loader `data/loaders/perturbseq.py` + Gladstone `tier2.py`; Ras circuits `circuits.py`; counts→activity `inference/bridge.py`; BIC topology model-selection `inference/model_select.py`; attribution pipeline `inference/pipeline.py`. **(a) Gladstone Rest + 8h + 48h (~440 GB, GSE314342): NUDGE abstains at every gate, across the whole time course** — `no-switch` at *all three* timepoints (the switch never emerges; Rest rejects it most decisively, BIC −86 vs 3240), and the multi-timepoint breaker finds 0 usable operating points (LNA depth-guard + min-cells skips; Rest depth is lowest, scale·peak 0.3). A robust, multi-gate fail-safe on real data — consistent across time, no switch manufactured. **Capstone closed** (`scripts/vv/gladstone_multitimepoint.py`; FINDINGS "Phase 4" + "Phase 4c"). **(b) Dose-response attribution `inference/dose_response.py` (`NUDGE-METHOD-001`): the first *positive* real-data call — OCT4 → switch (n≈6.7, R²=0.99), NANOG → honest abstain (unresolved, `NUDGE-LIM-007`)** on OCT4/NANOG GSE283614 (FINDINGS "Phase 4b"); reuses the Hill primitive + BIC parsimony, wired into the `nudge dose-response` CLI + `dose_response` MCP tool + Mechanism Card. **(c) Synergy/epistasis `inference/epistasis.py` (`NUDGE-METHOD-003`): Norman 2019 — 2/5 pairs explicitly confirmed** (DUSP9+ETS2 buffering, CBL+CNN1 synergistic; FINDINGS "Phase 4d"), plus a possible-neomorphic off-axis diagnostic (`ComboGeometry`; the forensic audit in `design/NORMAN_DISCREPANCY_ANALYSIS.md`). **(d) Cross-modality readout adapter `inference/cross_modality.py` (`NUDGE-METHOD-002`, `NUDGE-LIM-008`): the same threshold/gain/ceiling attribution on a CONTINUOUS readout (fluorescence/fold-change), behind a modality bouncer that refuses log-normalized/raw counts. Chure 2019 LacI (D1.1241): inducer-binding mutants → threshold (induction EC50), DNA-binding → leakiness/ceiling (floor), non-inducible Q294R abstains — 4/7 correct, 3/7 honest abstentions, 0 mis-calls; gain abstained (effective-steepness changes are second-order), and the naive DNA→K/inducer→n prior was biophysically refuted and overridden** (FINDINGS "Phase 4e"); `nudge cross-modality` CLI + `cross_modality` MCP tool + Mechanism Card + `notebooks/Chure_LacI_Benchmark.ipynb`. **(e) Bifurcation / tipping-point proximity — the "robustness dial" `inference/bifurcation.py` (`NUDGE-METHOD-006`, `NUDGE-LIM-012`): how close is a bistable switch to LOSING bistability (a saddle-node fold)? A 0..1 dial from three channels (critical slowing `min|Reλ|→0`, basin collapse node→saddle→0, LNA lobe swell →1), re-exposing the fixed-point eigenvalues `Circuit.fixed_points` dropped + the `lna_reliable` lobe ratio. Honesty crux: the LNA breaks down PRECISELY at the fold, so the dial is a ONE-SIDED LOWER BOUND near the fold and ABSTAINS (`unresolved`) on the deep-basin side — never a false-precise "far" number; `not-bistable` when <2 modes. Validated on the self-activation switch's KNOWN analytic fold in n and K: all three channels move monotonically toward it and the dial ranks proximity (FINDINGS "Phase 4f").** Generalises to N-species (the `design()` safety gate); `nudge robustness` CLI + `robustness` MCP tool + Mechanism Card + `notebooks/Robustness_Dial.ipynb`; real-data dose-ladder lock-in deferred (`needs_data`). **(f) Inverse / intervention design — the flagship `design()` `design/invert.py` (`NUDGE-METHOD-007`, `NUDGE-LIM-013`): the brief's headline thesis — NUDGE *inverts the fit to propose untested interventions*. Given a RELIABLE attribution (a minimal `AttributionResult` protocol; `DoseResponseResult`/`EpistasisResult` gain an additive `is_reliable`), it runs the fit BACKWARDS to prescribe an intervention behind two honesty gates. Circuit mode: gradient inversion over a fitted `Circuit` (`fit_parameters` backwards — Adam over an additive log-Δ on addressable knobs) + the Cap-5 `bifurcation_proximity` SAFETY gate on the intervened circuit (flags `crosses_fold`/`high_risk_of_instability`, one-sided near the fold). Curve mode: closed-form `DoseResponseFit` inversion to a dose (NO safety gate — no circuit/fold, stated). Integrity gate refuses an unreliable fit; reachability gate abstains on an unreachable target (never extrapolate, `NUDGE-LIM-013`). Validated: known-intervention recovery to loss ~0 (a ×2 vmax → factor≈2.0); fold-crossing flip flagged HIGH RISK vs a safe nudge cleared; curve round-trip + out-of-range abstain; real OCT4 fit inverted to a knockdown dose (FINDINGS "Phase 4g").** `nudge design` CLI + `design` MCP tool + `service.design_circuit`/`design_file` + Mechanism Card + `notebooks/Inverse_Design.ipynb`. **(g) Multi-reporter joint attribution `inference/multi_reporter.py` (`NUDGE-METHOD-008`, `NUDGE-LIM-014`): the identifiability force-multiplier — breaks the K⇄v_max / gain⇄threshold degeneracy (§2, NUDGE's dominant abstention) by fitting SEVERAL downstream reporters of ONE latent switch JOINTLY. Each reporter is an affine `Readout` `y_j = base_j + gain_j·A·f(dose; K,n)` of the same latent; pinning the gains from the control + sharing one latent over-determines the fit, so a threshold shift (moves the inflection identically across reporters) and a ceiling change (scales every reporter's ON amplitude by the same fraction) project DIFFERENTLY onto a heterogeneous-gain panel. Headline (synthetic ground truth, FINDINGS "Phase 4h"): JOINT recovery 24/24 (100%) vs SINGLE-reporter 0/24 (`unresolved`), 0 confident-wrong. Fail-safe STRENGTHENED — the consistency guard abstains OFF-MODEL when the panel is not one shared latent (a reporter reads a different latent, `NUDGE-LIM-014`), never averaging it into a call; a single reporter honestly abstains. **P2 (red-team round 3) sharpened `NUDGE-LIM-014`:** the consistency guard is control-only and was BLIND to a per-condition batch/depth scale on the perturbed panel, which aliases 1:1 onto a confident `ceiling` (verified 6/6). CLOSED (measurable floors) by a ceiling-scoped FLOOR-CONSISTENCY gate — a genuine ceiling leaves each reporter's OFF baseline fixed, a batch rescales it with the ON scale (`off_on_coupling` ≈0 vs ≈1); abstains `unresolved` when the floor moves with the ceiling. Residual BOUND (locked strict-xfail): on a (near-)ZERO-floor panel a batch and a real ceiling are inseparable without an independent depth anchor, so NUDGE abstains on both (measured separation + bound in FINDINGS "P2").** `nudge multi-reporter` CLI + `multi_reporter` MCP tool + `service.multi_reporter_file` + Mechanism Card + `notebooks/Multi_Reporter.ipynb`; real-panel demo deferred. **(h) Hidden-node ABSTENTION `inference/hidden_node.py` (`NUDGE-METHOD-009`, `NUDGE-LIM-015`) — the abstention HALF ONLY (positive hidden-node ID is a documented TRAP, NOT built): turns a bare `off-model` verdict (or a fired off-axis/neomorphic residual) into a legible six-cause DIFFERENTIAL — not-a-switch (`NUDGE-LIM-005`), nonlinear readout (`NUDGE-LIM-006`), off-target, wrong topology (T0.5-2), batch/depth confound (`NUDGE-LIM-003`/`009`), hidden node (the off-axis residual, `NUDGE-LIM-009`) — each with its evidence, documented limitation, and distinguishing experiment. The crux (`NUDGE-LIM-015`): the six causes are OBSERVATIONALLY OVERLAPPING, so NUDGE NEVER asserts a hidden node — the strongest it says is an off-axis residual is *consistent with, does not prove* an unmeasured regulator (rank capped so it's never the lone leading answer). Pure packaging/knowledge layer on `knowledge.explain`, ZERO import of `fit` — consumes verdicts, never re-attributes.** `nudge diagnose-abstention` CLI + `diagnose_abstention` MCP tool + `service.diagnose_abstention` + Mechanism Card + `notebooks/Hidden_Node_Abstention.ipynb`; honesty guarantee (never a bare positive hidden-node claim) enforced in CI. **(i) Comparative / differential attribution `inference/differential.py` (`NUDGE-METHOD-010`, `NUDGE-LIM-016`): the SAME perturbation in TWO contexts (drug-resistant vs sensitive line; donor A vs B; disease vs healthy) — isolate whether the difference is in threshold (K) / gain (n) / ceiling (v_max), a call linear DE structurally can't make (a raised ceiling → more of the SAME drug; a rewired gain/threshold → a DIFFERENT class). Fits the two contexts JOINTLY with a shared-vs-per-context parameter structure and BIC-selects which SINGLE knob differs {shared / ΔK / Δn / Δv_max}, reusing the LNA Gaussian-mixture forward model + BIC parsimony; abstains no-difference / unresolved. CONFOUND GUARD (`NUDGE-LIM-016`): a depth/batch shift aligned with the context axis is degenerate with a ceiling difference (scale⇄vmax), so depth is pinned PER CONTEXT from each control (`calibrate_from_wt`) AND when the two contexts' pinned depths differ beyond a ratio (a depth/batch difference aligned with the context axis) NUDGE abstains — never a spurious ceiling-diff — unless the winner is a cleanly-resolved threshold/gain difference, which reshapes the distribution (orthogonal to a global scale) and survives. Validated on synthetic ground truth (FINDINGS "Phase 4j"): Δv_max→ceiling-diff, Δn→gain-diff, none→no-difference, ΔK→recover-or-abstain (threshold hardest from a bistable snapshot, the gain>ceiling>threshold hierarchy §2), and a depth-aligned-with-context confound ABSTAINS unresolved, 0 confident-wrong across seeds. Additive/opt-in (never touches `fit()` / the decoy battery); `nudge differential` CLI + `differential` MCP tool + `service.differential_*` + Mechanism Card + `notebooks/Differential.ipynb`.** **(j) FAIL-SAFE RED-TEAM (`design/FAILSAFE_REDTEAM.md`): 2 verified confident-wrong holes found, both closed/locked.** Hole 1 — a near-fold 3rd operating point flipped a true ceiling → confident `threshold` in `attribute_lyapunov_multi`; FIXED (`NUDGE-LIM-017`) by gating the joint fit on the bifurcation-proximity dial's two deterministic channels (abstain unless every point is well-buffered, `proximity ≤ 0.15`), regression-locked by a near-fold decoy. Hole 2 — an additive (ambient/batch) offset on the A+B signature genes bypasses size-factor normalization and fakes a confident `synergistic`; no safe runtime gate exists (would false-abstain on real synergy), so LOCKED as a strict-xfail decoy + `NUDGE-LIM-009` sharpened (→ major/safety_relevant) + the neomorphic flag re-worded so it can't be read as corroboration. 3 further attacks (cross-modality, dose-response, bifurcation) HELD. **RED-TEAM ROUND 2 (core engine, `design/FAILSAFE_REDTEAM_2.md`): 2 more verified confident-wrong holes, both in same-day work, both fixed.** (1) The constitutive capture-scale confound (`NUDGE-LIM-019`): a control-vs-population capture-efficiency mismatch fakes `biological-switch` on a linear circuit → honesty contract broadened (`asserts_biological_switch`; "adversarially bounded" replaces "structurally fail-safe"), locked as a strict-xfail decoy, Option B robustness fix designed. (2) The round-1 LIM-017 hard margin was itself a KNIFE-EDGE (a 3rd point at proximity 0.146 still corrupted; onset non-monotonic, only 0.007 above the useful point) → replaced by graded near-fold down-weighting + threshold-free best-buffered-pair CORROBORATION (measured 0 confident-wrong: control resolves ceiling, every near-fold 3rd point abstains). **RED-TEAM ROUND 3 (design safety gate, `design/FAILSAFE_REDTEAM_3.md`): the `design()` bifurcation safety gate flagged only a RELATIVE proximity rise (`delta > margin`) and never checked the ABSOLUTE `proximity_after` against `NEAR_FOLD=0.55`, so a sub-margin push ACROSS the near-fold line (0.500→0.589) was cleared "OK, stays away from the fold" — contradicting `classify_robustness` on the identical circuit (a confident-wrong SAFETY label on a proposal, the highest-harm output). FIXED (`NUDGE-LIM-013` sharpened; additive in `design/invert.py`, frozen core untouched): `high_risk` now fires on `delta > margin` OR `proximity_after ≥ NEAR_FOLD` (reusing the EXISTING `NEAR_FOLD` constant so the gate and `classify_robustness` agree, recorded as `SafetyReport.near_fold`), the near-fold wording matches `classify_robustness`, and the one-sided-lower-bound caveat is carried on the SAFE branch too. Measured: the hole flags near-fold high-risk (0 confident-wrong); a robust intervention below `NEAR_FOLD` still clears "OK" (no over-abstention). Regression-locked by 3 `tests/design/test_invert.py` cases; deterministic repro `scripts/redteam/design_safety_gate_absolute_proximity.py` now exits "no hole" (FINDINGS "P3").** **P1 (differential, `NUDGE-LIM-016`) FIXED —** an ADDITIVE offset on ONE context's PERTURBED cells only (control clean) is invisible to the control-keyed `depth_ratio` (gate 2 never engages) and fakes a confident `gain-diff` (3 confident-wrong / 2 seeds, `scripts/redteam/differential_additive_confound.py`). Fixed by a MEASURED one-sided gate 4b: abstain when either context's perturbed OFF baseline is inflated above its own control beyond `off_shift_max=2.5` (measured separator — confident-wrong offsets `off_shift≥2.99` vs genuine `≤1.96`, FINDINGS §P1); CLOSED for the inflating offset, BOUNDED for a deflating one, decoy-locked. The previously non-load-bearing `off_shift` is now one-sided load-bearing.  **P4 (differential, `NUDGE-LIM-016`) FIXED —** the P1 re-scan found the MULTIPLICATIVE sibling of P1: a per-context multiplicative scale on ONE context's PERTURBED cells (control clean) fakes a confident `ceiling-diff` (truth no-difference), slipping under BOTH gate 2 (control-keyed `depth_ratio`≈1) and the P1 gate 4b (`off_shift` on the near-zero OFF baseline≈1). Fixed by a MEASURED ceiling-scoped gate 4c: abstain before a `ceiling-diff` when a context's perturbed OFF-cluster SCALE (raw MAD of below-median cells, perturbed÷control) leaves the band [0.80, 1.30] (genuine ceiling ×1.4–4 → ≤1.18; inflating confound c≥1.5 → ≥1.43; FINDINGS §P4). INFLATION CLOSED; DEFLATION BOUNDED (a genuine strong ceiling *reduction* deflates the OFF cluster into the same band → indistinguishable, so both abstain — a strict-xfail-locked honest bound). Genuine ceiling increases + gain/threshold still resolve. `provenance.py` still a stub |
| Temporal (Cap-4) | ✅ (synthetic + real coda) | **Temporal / Lotka–Volterra trajectory-fit attribution `inference/lotka_volterra.py` (`NUDGE-METHOD-012`, `NUDGE-LIM-020`) — the deferred Capability 4, unlocked, and the extensibility thesis made concrete (a NEW dynamical-systems domain: microbiome ecology).** Everything else observes a steady-state *snapshot*; this points the same abstain-and-attribute philosophy at a generalized Lotka–Volterra community `dxᵢ/dt = xᵢ(αᵢ + Σⱼ βᵢⱼxⱼ + εᵢ·u(t))`, whose info lives in **trajectories**. A self-contained differentiable RK4 `lax.scan` integrator (no `diffrax`); the trajectory fit is **re-instantiated in-module** (reusing `losses.energy_distance` over per-timepoint replicate ensembles) so it touches **neither `fit.py` nor `core/`** (frozen). Given a reference vs perturbed community under an antibiotic pulse, BIC-selects which single knob moved — **growth (α) / interaction (β) / susceptibility (ε)** — scored on the reference→perturbed **contrast** (cancels the baseline mean-bias → no spurious null wins). **ε is the identifiable positive** (the pulse is a time-localized on/off contrast); **α⇄βᵢᵢ is degenerate near equilibrium** (`Kᵢ=−αᵢ/βᵢᵢ`) → NUDGE **abstains** with the degeneracy **MEASURED** — a near-singular Laplace curvature on `(αₜ,βₜₜ)` (reusing `uncertainty.laplace_posterior`, cond ≫100, `|corr|→1`), exactly how NUDGE measures gain⇄threshold. **0 confident-wrong** across the synthetic battery (ε recovers, dense-transient growth recovers, self-interaction + near-eq growth abstain, null → no positive; FINDINGS "Temporal / Lotka–Volterra"). Two gLV decoys (`generate_alpha_beta_confound_decoy`, `generate_no_perturbation_null`). **Real coda now GENUINELY RUN (not just argued): Stein 2013 + MDSINE2 real-data adapters (`scripts/vv/stein_glv.py`, `scripts/vv/mdsine2_glv.py`) → the shipped `attribute_glv`.** Measured verdict (FINDINGS "Real-data gLV attribution on Stein 2013 + MDSINE2"): on BOTH datasets, TWO antibiotics (clindamycin, vancomycin), at EVERY dimension k, NUDGE returns **0 confident calls and 0 confident-WRONG** — heavy honest abstention. It **genuinely abstains on *C. difficile*** (`no-change` at every k; its ~0-baseline, non-pulse-locked, interaction-mediated bloom, published ε≈−0.31) and on MDSINE2's *Escherichia/Shigella* competitive-release bloom. Boundary finding: the synthetic "ε positive" does NOT transfer to real data — the strongly-suppressed commensals DETECT an effect and pick ε with the correct sign but ABSTAIN on the exact knob (ε⇄α confound, unobserved 1-day pulse); the limit is temporal resolution, not k. Denser MDSINE2 in-pulse sampling improved detection but did not cross the resolution threshold (k≥8 the baseline fit even NaNs). `nudge lotka` CLI + `service.lotka_demo` + Mechanism Card + `notebooks/Temporal_Ecology.ipynb` (§7 runs the genuine Stein fit). Data gates in `design/EXTENSIBILITY_SPIKE.md` / `design/MICROBIOME_DATA_GATE.md`. Additive/opt-in. |
| Stretch | ◑ | **N-D saddle finder + toggle representation DONE** (attribution is 1-D only — see below); **constitutive-reporter calibration control SHIPPED** (`inference/constitutive.py`, `NUDGE-METHOD-011`, `NUDGE-LIM-018`) — the `NUDGE-LIM-006` mitigation: a control drives the reporter at KNOWN doses (bypassing the circuit) and anchors the readout using READOUT params ONLY (gradient-proven no leak), then a profile over circuit `n` WITHOUT vs WITH the control breaks the degeneracy — WITHOUT it the `n`-profile is FLAT (span ≈0.001, can't tell a switch exists); WITH it "no switch" (n=1) is REJECTED for a true switch (Δloss ≈0.026 ≫ span) → `biological-switch`, while the LIM-006 hazard (a linear circuit) ABSTAINS `unresolved` — 0 confident-wrong on the clean-control validation, never a bare knob, never point-IDs `n` (needs a 2nd anchor; `NUDGE-LIM-018`). **ADVERSARIALLY BOUNDED (`NUDGE-LIM-019`, red-team round 2):** the `biological-switch` verdict is valid ONLY when the control shares the circuit population's capture scale — a control-vs-population capture-efficiency mismatch re-anchors the reporter and re-opens LIM-006 (locked as a strict-xfail decoy; Option B robustness fix designed in `design/CONSTITUTIVE_CONTROL.md`). `nudge constitutive` CLI + `constitutive` MCP tool + `service.constitutive_file` + Mechanism Card + `notebooks/Constitutive_Control.ipynb` (design in `design/CONSTITUTIVE_CONTROL.md`); **Claude integration layer BUILT** — `nudge` typer CLI (`cli.py`) + shared `service.py`/`knowledge.py`, FastMCP stdio server (`mcp/server.py`, `nudge-mcp`, `.mcp.json`) exposing `attribute`/`explain_abstention`/`list_mechanisms`/`get_mechanism_card`, the Mechanism-Card knowledge base (`docs/mechanism_cards/`, 17 cards incl. `dose_response_attribution` + `multi_reporter` + `hidden_node_abstention` + validator), and Agent Skills (`nudge-attribute`/`nudge-explain`/`mechanism-card`); feasibility + recipes in `design/INTEGRATION_FEASIBILITY.md`, ontology vision in `design/ONTOLOGY.md`. **`design/invert.py` — the flagship `design()` — is now BUILT (see Phase 4 (f) above; `NUDGE-METHOD-007`).** Still homes-reserved: `zero_order.py`, `data/loaders/tier{1,2}.py` (tier2 landed), docs site, `scripts/ai/` |

**Visualization layer — `nudge.viz` (opt-in `[viz]` extra), FIRST SLICE landed (Demo, 30%).**
An additive, provenance-carrying figure module (`src/nudge/viz/`: `theme`/`base`/`provenance`/
`dose_response` + a `render()` dispatcher) that turns the frozen result dataclasses (and their
`*_to_dict()` dicts — dual-input) into honest figures. It only READS results — never touches
`fit.py`/`core`, never re-attributes — lazily imports matplotlib (`Agg`, headless). The
load-bearing honesty is STRUCTURAL: `render()` applies the **abstention overlay itself off the
result's own verdict**, so a renderer can't draw an abstention as a confident call; one-sided
bounds (`spans_inflection=False`) draw as **open-ended arrows**, never closed error bars. Slice 1
is the flagship `plot_dose_response` **dual panel** — real ESC-screen **OCT4 → `switch`** (n≈6.7,
R²=0.99) beside the honest **NANOG → `unresolved`** in one frame. Every figure ships `fig.png` +
`fig.data.json` + a standalone deterministic `fig.py` that regenerates it from the fit's output
(no re-fit; verified **pixel-identical**), plus a `self_contained` inline mode — the Claude Science
provenance grain. Wired via `service.render_result()` + an opt-in `--fig-out/--fig-code/--fig-theme`
flag on `nudge dose-response` (default text output UNCHANGED); embedded in
`notebooks/OCT4_NANOG_Flagship.ipynb` (re-executed headless, 0 errors). Tests in `tests/viz/`
(incl. the honesty test: a known abstention → `FigureResult.abstained==True` + overlay drawn).
matplotlib moved to `[viz]`; `[dev]` depends on it. **Deferred** (designed in
`design/VISUALIZATION_DESIGN.md`, later slices): the ~11 other renderers, the LIM-006
constitutive-flip **animation**, the MCP `render_figure` tool.

**The PoC works end to end** (`tests/inference/test_fit_end_to_end.py`, slow lane):
generate ground-truth data → `fit()` → recover kinetics → attribute
threshold/gain/ceiling → and `off-model` when a linear model suffices.

## 3. Key empirical findings (overnight V&V — `scripts/vv/FINDINGS.md`)

- **0% misclassification at every `margin_k`** across 300 linear + 120 switch
  datasets. NUDGE never calls the wrong mechanism; it abstains. Fail-safe, measured.
- **Calibrated default `margin_k = 1.7`** → <2% false-positive rate on linear data.
  It's a specificity/sensitivity dial (1.0 → 88% correct/7.7% FP; 1.7 → 65%/1.7%).
- **Identifiability:** needs **≥~1000 cells/condition**; **gain > ceiling ≈
  threshold**; **ceiling is the most noise-fragile**; threshold hardest (K/v_max
  partial degeneracy — both shrink the ON signal).
- **Tier-0.5 (independent stochastic data) — fail-safe survives, with a boundary.**
  On data from the new tau-leaping SSA (`data/stochastic.py`, emergent bimodality),
  a matched-topology fit emits **0 wrong positives across seeds 0–3** (abstains, or
  recovers only gain) — the fail-safe property holds off the inverse crime. BUT
  fitting a *wrong* (feedforward) topology to the feedback data produced a confident
  wrong call (`gain→threshold`, every `margin_k`): **the guarantee is conditional on
  approximately-correct topology.** Full write-up in `scripts/vv/FINDINGS.md` §Tier-0.5.

## 4. Architecture decisions & gotchas (a fresh context MUST know these)

- **Circuit = self-contained differentiable JAX vector field, NOT `GraphManager`.**
  `GraphManager` bakes params as compile-time constants; we need params as a traced
  pytree (`{"species": {...}, "edges": {...}}`) to `vmap` over per-cell draws and take
  gradients. `Circuit.solve_population` = `jax.vmap` of a semi-implicit steady-state
  solve. (Documented as a MADDENING case study in
  `../plans/NUDGE_deterministic_solve_vs_graphmanager.md`.)
- **The population model = `vmap` over per-cell parameter draws** (extrinsic noise on
  basal/decay), NOT stochastic dynamics. This is the validated deterministic
  transfer-function route (Ochab-Marcinek & Tabaka 2010) — bimodality is *designed-in*
  (a Tier-0 feature). **This is exactly what Tier-0.5 must break** (see §6).
- **Count model = negative binomial, NO zero-inflation** (`data/noise.py`), via
  Poisson-Gamma. UMI droplet data is not zero-inflated (Svensson 2020 etc.).
- **Fit forward model** (`inference/fit.py`): the NB count *sample* is discrete/
  non-differentiable, so the loss uses a **reparameterized moment-matched Gaussian
  observation** `μ + √(μ+φμ²)·ζ`, **clamped ≥0** (counts can't be negative; also
  avoids negative-tail NaNs). A **`log1p` transform** makes the energy distance
  shape-sensitive (bimodality) — this is what separates the mechanisms.
- **classify.py has two levels:** (1) circuit-level **`switch_detected`** — the
  linear-baseline parsimony gate (mechanistic must beat linear on WT beyond the loss
  noise floor, else no switch → all `off-model`); (2) per-perturbation **`decide`** —
  no-effect / off-model(absolute) / unresolved / threshold-gain-ceiling. The gate is
  at the CIRCUIT level deliberately: per-perturbation it misfires because a strong
  gain/ceiling reduction genuinely linearizes a perturbed condition.
- **Attribution = three restricted fits** (free K / n / vmax of the target edge from
  the WT baseline); the winner is the mechanism. Noise floor = WT self-distance
  bootstrap (`_self_distance`).
- **Compile-cache gotcha:** the persistent on-disk JAX compile cache (MIME's conftest
  pattern) served corrupted `random.poisson` executables (silent -1). It is
  **disabled** in `tests/conftest.py` — do NOT re-enable it.
- **maddening pin:** `maddening[ift]>=0.3.1` (the `[ift]` extra pulls `lineax` for
  `ift_linear_solve`). 0.3.0 lacked it.
- **PRINCIPLE — guard the identifiability, not the confound** (an orthogonal anchor or a
  measured degeneracy, **never a calibrated band**). A per-confound OFF-cluster band (e.g.
  differential gates 4b/4c) fixes one magnitude and leaves the next as a blind gap (P5 slips
  `(1.18,1.30]`). Instead add the nuisance itself as a **free parameter** — a free-affine null
  *contains the entire confound family*, so ONE measured statistic ("does the bio knob earn its
  BIC parameter over the free nuisance", the profiled ΔBIC) covers the class continuously. When
  even that is degenerate (a non-uniform nuisance observationally identical to a mechanism),
  resolve with an **orthogonal anchor** (perturbation-inert genes) or **abstain**. **Prototyped
  and measured** for the differential's per-condition affine class: `_proto_nuisance.py` — 0
  confident-wrong across the whole uniform-affine sweep (bands leave 13/30) with every positive
  control preserved; the anchor recovers a ceiling under a scale; residual = non-uniform
  nuisances need the anchor. Full design + migration plan: `design/PERTURBED_CONFOUND_STRATEGY.md`.
  (Honest dead-end recorded there: the raw Laplace **condition number** is NOT the discriminator
  — it saturates on the offset's unit scale; the degeneracy that matters is GLOBAL, so the
  integrated profiled ΔBIC is the right statistic.)

## 5. Public API surface (import-light where it matters)

`nudge.fit(adata, circuit) -> MechanismMap` · `nudge.Circuit` / `CircuitBuilder` ·
`nudge.generate_synthetic_perturbseq` · `nudge.PerturbationSpec` ·
`nudge.MechanismClass` (threshold/gain/ceiling + no-effect/unresolved/
technical-artifact/off-model) · `nudge.MechanismMap`/`MechanismCall`.
Lower-level: `inference.fit.fit_parameters` (the optax recovery engine),
`inference.classify.{decide, switch_detected}`, `inference.losses.{energy_distance,
rbf_mmd}`, `data.ingest.check_counts` (the raw-counts bouncer).

---

## 6. Tier-0.5 independent stochastic simulator — ✅ LANDED (simulator + guard)

**Status (updated after build).** Built: `data/stochastic.py`
(`generate_stochastic_perturbseq`, a tau-leaping SSA of a self-activating gene,
emergent bimodality, reusing the `Readout`+NB observation layer verbatim; re-exported
via `data/loaders/tier05.py` and `nudge.__init__`) and its guard test
`tests/verification/test_stochastic_inverse_crime.py` (a fast bimodality check +
the slow never-wrong fit assertion). **Result:** matched-topology fit → **0 wrong
positives across seeds 0–3** (fail-safe holds off the inverse crime); wrong-topology
fit → can misclassify (the fail-safe boundary is topology). See `FINDINGS.md`
§Tier-0.5. **Deferred follow-ons:** the To & Maheshri bimodality-without-bistability
*decoy* (needs a telegraph/promoter mechanism + a short lit search — user-confirmed
fast-follow), and a **multi-basin IC seeding** extension to the fit so it can
*represent* emergent feedback bistability, now **built into NUDGE** and taken all the way
to a fail-safe fix. The arc (full detail in `scripts/vv/FINDINGS.md` §T0.5-3→5):
1. An autonomous R&D subagent found multi-basin representation feasible (`p` recovers, ≈10× lower loss).
2. Built `energy_distance_weighted` + `fit_multibasin_parameters` + `fit_multibasin`: it
   *represents* bistability but plain 2-basin **attribution degenerates** (conflates gain
   with ceiling → a confident wrong call) — so `fit_multibasin(transition_mode=False)` is
   EXPERIMENTAL / not-fail-safe.
3. **RESOLVED** via the user's saddle idea (a 2nd autonomous spike → integrated): a **third
   transition mode at the unstable saddle** (`Circuit.fixed_points`/`transition_state`,
   `fit_transition_parameters`, `classify.decide_with_transition`). The free-`n` transition
   weight is a fail-safe gain detector (0.89 gain vs 0.01 else). **`fit_multibasin(
   transition_mode=True)` recovers `gai→gain` at all seeds incl. the seed-2 bug, zero wrong
   positives**, N-species safely abstains. Six failure modes engineered against (FM1 NaN
   masking, FM2 N-D decoupling+`n_species==1` guard, FM3-6). Guarded by the Tier-0.5 test.

**N-D saddle (DONE — finder + representation; attribution NO-GO).** `Circuit.fixed_points`
/`transition_state` now find the index-1 saddle of an N-species circuit (multi-start Newton
+ Jacobian index; verified on a 2-node toggle), and the transition fit represents a toggle
(FP-seeded basins, 56× lower loss than naive seeding). But the `w_trans` gain gate is
**1-D-specific** and does NOT extend to the toggle (a single-edge gain reduction doesn't
collapse it to the saddle) — so the gate stays guarded to `n_species == 1` and NUDGE
**abstains** (never misclassifies) on toggles (`tests/verification/test_toggle_nd_safety.py`).
Finder + representation are reusable infra. **Performance:** the N-D finder (recomputed every
optimizer step) is now a **jitted, per-topology-cached kernel** (`_nd_kernel`; kinetics as a
traced arg) — byte-identical roots, ~1 ms/call (**333×** per-call; a toggle transition fit
**26 s → 4.1 s**). A warm-start/trust-region attempt was tried and *rejected* (tracing, not
solve-count, was the cost → ~1× + a reproducibility divergence; jit subsumes it).
**The toggle attribution signature is researched + BUILT.**
(`design/TOGGLE_ATTRIBUTION_RESEARCH.md`): `w_trans`/occupancy was the wrong channel (mixture
weights are set by a non-gradient quasi-potential, not the saddle); the gain signal lives in
each lobe's **covariance** (linear-noise Lyapunov `AΣ+ΣAᵀ+D=0`). A Fisher-information analysis
*measured* the degeneracy — it is **gain⇄threshold** (not gain⇄ceiling; ceiling is the *most*
identifiable), analytically `n·ln(K/B)`, robust to extrinsic noise
(`scripts/vv/fisher_sloppiness.py`, `fisher_extrinsic.py`) — and the breaker is a **second
operating point**, not a constitutive control.

That whole chain is now a working, **additive/opt-in/guarded** capability
(`nudge.inference.lyapunov`, milestones M0–M4; **not** wired into `fit()`, for risk isolation):
`Circuit.mode_covariances` (M0); the differentiable LNA Gaussian-mixture fit
`fit_lyapunov_parameters` (M1; found the `scale⇄vmax` = sequencing-depth degeneracy →
`calibrate_from_wt` pins depth from WT); single-condition `attribute_lyapunov_single` that
identifies ceiling and **abstains between gain/threshold** (M2); the multi-operating-point
breaker `fit_lyapunov_multi`/`attribute_lyapunov_multi` that **resolves** gain vs threshold
(M3; NLL gap 0.005→0.098, ×20); and `lna_reliable`, which **abstains loudly** at low depth /
near a bifurcation / when monostable (M4). Full write-up: `scripts/vv/FINDINGS.md`
"Covariance attribution".

**Broken past the inverse crime (measured on INDEPENDENT SSA).** M1–M3 above were inverse-crime
(cells drawn from the LNA Gaussian the fitter maximizes). New: fed data from the *independent*
tau-leaping SSA (`generate_toggle_perturbseq`, bridged to activity, at a guard-clearing depth;
`scripts/vv/toggle_lyapunov_ssa.py`, 3 seeds). **The single snapshot DEGENERATES** — the
inverse-crime "ceiling is identifiable" claim does not survive (the free-vmax fit becomes the
*worst* fit of a true ceiling → mis-narrows to `gain_or_threshold`; gain/threshold abstain),
though it still only ever returns an abstention-class label (never a bare mechanism, so never
confidently wrong). **The second operating point RECOVERS** — the two-basal-B joint fit resolves
**threshold (3/3)** and **ceiling (3/3)** and honestly **abstains on gain**, with **0
confident-wrong calls**. First evidence the covariance signature separates mechanism on
non-inverse-crime toggle ground truth: a guarded positive for the multi-condition breaker + a
documented single-snapshot negative. Still additive/opt-in, not in `fit()`; production toggle
path still abstains. Locked by `tests/inference/test_lyapunov_toggle_ssa.py`; FINDINGS
"independent-SSA validation".

**Phase 4 — real-data validation (infra landed; run pending the download).** The pipeline to
point this at the Gladstone CD4+ T-cell screen is built and green on synthetic ground truth:
a **generic backed-mode Perturb-seq loader** (`data/loaders/perturbseq.py`, config-driven,
subsets the ~150 GB donor files *on disk*; Gladstone config in `tier2.py`), the two candidate
**Ras circuits** (`circuits.py`: `ras_switch_1node` / `ras_switch_2node`), the
**counts→activity bridge** (`inference/bridge.py`), a **BIC topology model-selection** gate
(`inference/model_select.py` — infers linear/1-node/2-node rather than assuming it), and the
**attribution pipeline** (`inference/pipeline.py` + `scripts/vv/gladstone_attribution.py`).
**RUN DONE — NUDGE abstained (fail-safe on real data).** On the real `D1_Stim8hr` screen
(2.79M cells, 150 GB; pointer-read loader → 6,367 cells), the BIC topology gate selected
**no-switch** (40,556 vs 40,599 for 1-node): the 8-h IEG-activation readout is a single sharp
low mode + a heavy tail (5,884/6,000 cells in the low bin, skew ≈ 12), *not* two attractors, so
NUDGE declined to attribute rather than fabricate a switch. Full write-up: `scripts/vv/FINDINGS.md`
"Phase 4 — real data". Follow-ups (don't change the verdict): other stim timepoints, a
signaling-proximal readout, a focused screen (the targets are underpowered, 24–233 cells). The
gain-vs-threshold *breaker* would need ≥2 stim-condition files. `provenance.py` still a stub.

The original plan is retained below for reference.

### (original plan) the Tier-0.5 independent stochastic simulator

**Why (the one caveat the whole PoC can't escape):** everything so far is **Tier-0
inverse crime** — the generator and the fitter share the same deterministic model +
noise params. The literature (`GENERATOR_DESIGN.md` "THE CRUX, RESOLVED"; Kepler &
Elston 2001; To & Maheshri 2010) says the honest robustness test is data from an
**independent, genuinely stochastic** process where bimodality is **emergent** (noise-
induced switching, mode occupancy set by the landscape/basin depths), NOT designed-in
by a parameter distribution. If NUDGE's deterministic fit still attributes mechanism
on such data, the approach generalizes. If it breaks, we learn the failure mode cheaply.

**What to build:**

1. **A self-contained stochastic simulator** — `src/nudge/data/stochastic.py` (or fill
   `data/loaders/tier05.py`). NO new heavy deps; a **tau-leaping SSA** of a
   self-activating gene is ~100 lines of numpy:
   - State = molecule count `X` per cell. Reactions per step `dt`:
     `production ~ Poisson((basal + vmax·X^n/(K^n+X^n))·dt)`,
     `degradation ~ Poisson(decay·X·dt)`; `X += prod − deg`, clip ≥ 0.
   - Run N independent cells from random ICs to time T (steady state) → snapshot `X`.
     Bimodality is **emergent** (Poisson noise + positive feedback → noise-induced
     switching), occupancy set by the landscape — the key difference from Tier-0.
   - Add mild per-cell extrinsic variation (rate constants) on top of intrinsic noise.
   - Map `X` → counts via the SAME observation layer (`data/noise.py`) so `fit()`
     consumes the AnnData identically. Emit the same schema as
     `generate_synthetic_perturbseq` (`.X` counts, `.obs` condition/true_mechanism,
     `.uns['ground_truth']`).
   - Perturbations move `K` / `n` / `vmax` in the propensity (ground-truth mechanism).
   - Give it a `generate_stochastic_perturbseq(...)` mirroring the Tier-0 signature.
   - Sanity: JAX is awkward for variable-length Gillespie; **tau-leaping is
     vectorizable over cells in numpy/jnp and fine** (generation isn't differentiated).

2. **The inverse-crime-guard test** — `tests/verification/` (mark `slow`+`verification`):
   generate Tier-0.5 switch data with strong K/n/vmax movers → `fit()` → assert
   **correct-or-unresolved, NEVER the wrong mechanism** (the fail-safe property must
   survive the model mismatch), and ideally correct for the clear (gain) case. This is
   the headline Tier-0.5 result.

3. **A "bimodality-without-bistability" decoy** (To & Maheshri 2010 route) — generate
   noise-induced bimodal data from a NON-cooperative (n≈1) stochastic feedback loop
   (bimodal but no deterministic switch) → `fit()` must return `off-model`/not-a-switch
   (its deterministic Hill fit must not beat linear beyond the floor). This is a
   scientifically-validated decoy the literature specifically flags.

**Success criterion:** on independent stochastic data, misclassification stays 0% and
the clear mechanisms (esp. gain) are still recovered. Expect threshold/ceiling to
abstain more (harder under mismatch) — that's acceptable and honest.

**After Tier-0.5, the rest of Phase 3:** the full decoy battery (`data/decoys.py` +
`tests/decoys/` — several buildable now: two-cell mixture, dropout zero-peak, dead
guide), the verification suite (confusion matrix as a test, calibration coverage).
**Laplace uncertainty is DONE** (`inference/uncertainty.py`, not `uncertainty/laplace.py`):
curvature-based CIs from the loss Hessian at θ*, a guarded ridge-inverse, and a degeneracy
guard that reproduces the gain/threshold near-singular Hessian (cond ≈ 210, |corr| ≈ 0.99)
and abstains — additive/opt-in, does NOT alter `fit()`'s output contract or wire onto
`MechanismCall` (that stays for later, to avoid colliding with in-flight work). Then
Phase 4 (T-cell SOS/RasGRP1 + wire `provenance.py`) and stretch (`design()`, MCP).

## 7. Other de-risking V&V still open (lower priority than Tier-0.5)

- Misspecification robustness (fit with wrong dispersion/library σ than the data).
- Bistable-FEEDBACK circuits (self-activation loop, basin-spanning ICs) vs the current
  feedforward switch.
- ~~A richer multi-reporter readout to break the K/v_max degeneracy.~~ **DONE** —
  `inference/multi_reporter.py` (`NUDGE-METHOD-008`); JOINT 100% / SINGLE 0% recovery on
  synthetic ground truth, 0 confident-wrong (FINDINGS "Phase 4h").
- Replicate the identifiability sweep at higher fit budget.
