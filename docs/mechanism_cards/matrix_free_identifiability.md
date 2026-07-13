---
id: NUDGE-METHOD-015
name: matrix_free_identifiability
role: analysis-method
registry_name: MatrixFreeIdentifiability
vulnerable_to_decoys: []
documented_limitation: [NUDGE-LIM-023, NUDGE-LIM-027]
validated_in_regime: {min_replicates: 1, min_timepoints: 1, notes: "Any differentiable forward model predict_fn(theta) -> observations, driven BY NAME through the general model registry (nudge.inference.model_registry) — populated with several genuinely different models across domains (glv microbiome ecology; linear_pathway reaction kinetics; ad_qsp clinical pharmacology; logistic population dynamics; plus the canonical sum_of_exponentials / redundant_exponential / well_conditioned toys). The verdict is one of well-constrained / sloppy-but-predictive / unidentifiable, MEASURED from the Fisher-information (FIM = JᵀJ/σ²) eigenspectrum computed matrix-free (one jvp + one vjp per matvec, never forming J). Honesty is load-bearing on both ends: sloppy != unidentifiable (a wide Fisher spectrum with tight predictions is USABLE — do not over-abstain, do not Fisher-greedily optimize the sloppy directions); and the matrix-free smallest-eigenvalue is one-sided (eigsh(which=SA) can miss an isolated near-null), so the iterative path ABSTAINS rather than assert an identifiability it cannot certify (NUDGE-LIM-023). Standing decoys: the well_conditioned model must NOT be flagged sloppy; the redundant_exponential structural null (A·e^{-(k1+k2)t}) must abstain unidentifiable and name the (k1,k2) null direction. Registry scope is a convenience surface, not a capability limit (NUDGE-LIM-027): arbitrary models remain a plain import nudge library path."}
references: [Transtrum2014, Chis2011, Svensson2020]
---

# Mechanism Card — Matrix-free Identifiability (the `identifiability` tool)

> **ID:** `NUDGE-METHOD-015`  ·  **Role:** analysis-method
> **Stability:** experimental  ·  **Registry name:** `MatrixFreeIdentifiability`

## Summary

A **general** "which parameters of this differentiable ODE model are identifiable, sloppy, or
unrecoverable from data?" tool. It takes a model **by reference** — a name from the general
model registry (`nudge.inference.model_registry.list_models`) — runs NUDGE's real matrix-free
Fisher-information diagnostic, and returns the verdict it MEASURES plus the FIM spectrum, the
named null directions, and the honest fail-safe bound. It is exposed as the `identifiability`
MCP tool and the `nudge.service.identifiability_tool` service function; the underlying analysis
is `nudge.inference.sloppiness.sloppiness_diagnostic_matrixfree`.

## Why this exists (sloppy ≠ unidentifiable, measured never asserted)

The obvious identifiability test — declare a model unidentifiable when the Fisher condition
number is huge — is **wrong for sloppy models**. A *sloppy* model has a Fisher spectrum
spanning many decades (its individual parameters are loose) yet is often perfectly
**predictive** and structurally identifiable; calling it "unidentifiable" would make NUDGE
over-abstain on a usable model and tempt a Fisher-greedy experiment that *destroys*
predictivity. So the tool separates the two questions the naive test conflates:

- **Is a direction genuinely unrecoverable?** — a *structural null* (an exact functional
  redundancy) detected from the rank of the sensitivity matrix, categorically different from a
  sloppy small-but-finite eigenvalue.
- **Are the predictions constrained?** — propagate the parameter covariance through the
  prediction map; sloppy directions blow up the covariance but map to ~0 prediction change.

The verdict is one of `well-constrained` / `sloppy-but-predictive` / `unidentifiable`.

## Governing equation

The Fisher information of the (log-)parameters for iid Gaussian observation noise `σ` is
`FIM = Jᵀ J / σ²`, where `J = ∂(observations)/∂(log θ)`. The **matrix-free** path never forms
`J`: it touches the FIM only through matrix-vector products `JᵀJ·v` — one forward tangent
(`jvp`, `J·v`) composed with one reverse cotangent (`vjp`, `Jᵀ·w`) — so peak memory is
`O(n_params + n_obs + tape)`, independent of the product `n_obs·n_params`. The verdict reads
the FIM eigenspectrum (span, condition number, smallest eigenvalue) plus the propagated
prediction uncertainty.

## Assumptions & simplifications

- The forward model is differentiable and given in RAW parameters, so `θ·∂/∂θ` recovers the
  log-sensitivity (the sloppiness convention).
- The FIM is the **local** curvature at the nominal `θ₀`; the verdict is a local-identifiability
  statement (not global).
- iid Gaussian observation noise `σ` (a per-model default, overridable).

## Known failure modes

| Failure mode | Guard | Limitation |
|---|---|---|
| An ill-conditioned FIM's smallest eigenvalue is not certifiable matrix-free (`eigsh(which='SA')` can miss an isolated near-null) | shape rank-deficiency ⇒ `unidentifiable`; else an inverse-iteration null probe; else **abstain** — never assert un-certified identifiability | `NUDGE-LIM-023` |
| An unregistered model name | explicit error listing the registered models (never a fabricated verdict); arbitrary models remain the `import nudge` library path | `NUDGE-LIM-027` |

## Decoys (standing honesty checks)

- **`well_conditioned`** (a well-posed linear model) must return `well-constrained` — it must
  **NOT** be flagged sloppy.
- **`redundant_exponential`** (`A·e^{-(k₁+k₂)t}`, an exact structural null) must **abstain**
  `unidentifiable` and name the `(k₁, k₂)` null direction.

Both are registered models, exercised by `tests/mcp/test_identifiability_oed.py`.

## Identifiability regime

Any differentiable `predict_fn(theta) -> observations`. Validated across the registry: `glv`
(sloppy-but-predictive), `linear_pathway` (well-constrained), `logistic` (well-constrained on a
rich schedule), `ad_qsp` (sloppy-but-predictive at moderate scale; rank-deficient
`unidentifiable` at population scale). Float64 is required to resolve the smallest FIM
eigenvalues (float32 truncates the sloppy end); the tool enables it locally.

## Implementation Mapping

| Equation term | Code |
|---|---|
| matrix-free FIM diagnostic | `nudge.inference.sloppiness.sloppiness_diagnostic_matrixfree` |
| end-to-end analyze from a model | `nudge.inference.sloppiness.analyze_model_matrixfree` |
| the model registry | `nudge.inference.model_registry.build_identifiability_problem` |
| register your own model | `nudge.inference.model_registry.register_model` |
| the tool service | `nudge.service.identifiability_tool` |

*(This table is machine-checked by `scripts/check_impl_mapping.py`.)*

## Verification evidence

- `tests/mcp/test_identifiability_oed.py` — the tool's result shape on ≥2 models + the two
  decoys (well-constrained must not be flagged sloppy; the structural null must abstain).
- `tests/inference/test_sloppiness_matrixfree.py` — the matrix-free path matches the dense
  diagnostic bit-for-bit on the validated small cases.

## References

- Transtrum et al. 2014 — sloppy models: wide Fisher spectra, tight predictions.
- Chis et al. 2011 — structural identifiability via the sensitivity-matrix rank.
- Svensson 2020 — the count/observation-noise model context.
