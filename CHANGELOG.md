# Changelog

All notable changes to NUDGE are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); the project follows
[Semantic Versioning](https://semver.org/). The `fit` / `design` public surface
is the stability contract (see `docs/architecture/verification_vs_validation.md`).

## [Unreleased]

### Added
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
- Traceability inherited from `maddening.compliance` (`NUDGE-*` ID prefixes) and CI
  validators (`check_anomalies`, `check_citations`, `check_impl_mapping`); PEP 561.

### Verification
- V&V calibration sweep (`scripts/vv/`): **0% misclassification** across 300 linear
  + 552 switch datasets; calibrated `margin_k = 1.7`. Tier-0.5 inverse-crime guard
  and the seed-2 saddle gain-recovery test. Findings in `scripts/vv/FINDINGS.md`.

### Known Limitations
- See `docs/known_limitations.yaml` (`NUDGE-LIM-*`); the full decoy battery, Laplace
  uncertainty, and real-data validation are not yet built. `design()` is a stub.
