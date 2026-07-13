# A/B: the same model + data, with vs without NUDGE

This directory is a **fair A/B harness**. It ships the *same* differentiable Alzheimer's
amyloid-β QSP model and synthetic cohort that NUDGE's `identifiability` and `oed` MCP tools
analyse — but as a **self-contained NumPy module with no `nudge` dependency** — so you can hand
a raw agent (a bare LLM / coding agent) the identical model + data and compare its answer to
NUDGE's.

The point NUDGE is making: on this problem the honest answer is *"you can't fit these
constants from one sparse schedule — here is the identifiable combination and the experiment
that would resolve it."* A tool that returns confident per-subject rate constants, or a naive
baseline+end measurement schedule, is **guessing**. NUDGE refuses to.

## Contents

| File | What it is |
|---|---|
| `ad_qsp_forward.py` | The Aβ-cascade ODE vector field + the 12 kinetic parameters + an RK4 integrator + a synthetic-cohort generator. **No `nudge` import.** Same rate-law forms as `nudge.mechanisms.ad_qsp`. |
| `make_dataset.py` | Regenerates the dataset from the forward model. |
| `cohort.npz` | The synthetic cohort: `true_params` (40×12 ground truth), `observations` (40×8×2 log-biomarkers), `obs_times`, `dose`/`dose_window`. |
| `cohort.csv` | The same observations as a tidy long table (subject, time, biomarker, log_value). |
| `ad_qsp_nlme_forward.py` | The **coupled hierarchical / NLME** version — subjects share population hyperparameters (`μ`, `φ`), so the joint FIM is a genuine **arrowhead** (not block-diagonal). **No `nudge` import.** Includes a finite-difference Jacobian (the only route a NumPy model has). |
| `make_nlme_dataset.py` | Regenerates the coupled cohort from the NLME model. |
| `cohort_nlme.npz` | The coupled cohort: `theta0` (joint `[μ | φ | r_i]` ground truth), `mu`/`phi` (shared border), `subject_multipliers`, `observations`, `border_size`. |

## The COUPLED NLME arm (the non-strawman scale wall)

The `cohort.npz` above gives every subject **independent** kinetics, so its population FIM is
**block-diagonal** — a competent agent decomposes it into per-subject 12×12 blocks and never
needs a big matrix. So "a dense FIM OOMs" would be a **strawman** on that cohort.

`ad_qsp_nlme_forward.py` + `cohort_nlme.npz` fix that. Subjects share population
hyperparameters: each subject's random-effect kinetic value is drawn around a **shared**
geometric-mean `μ`, and the non-random kinetics are a **shared** fixed-effect vector `φ`. The
joint parameter vector is `θ = [ μ | φ | r₀ … r_{N-1} ]` (`r_i` = subject `i`'s multiplicative
random effect). Because `μ`/`φ` enter **every** subject's observations, the joint FIM is a
bordered / **arrowhead** matrix — a dense border coupling every subject, plus per-subject blocks;
cross-subject blocks are exactly zero. **Block-summing no longer applies**: you cannot analyze a
subject in isolation.

- *Without NUDGE:* asked for the full joint identifiability spectrum, a raw NumPy agent most
  naturally forms the dense Jacobian `J = ∂(obs)/∂θ` (here by finite differences — `2·n_params`
  cohort re-simulations, no autodiff) and the dense FIM `JᵀJ`, an `(n_params × n_params)` array
  that reaches ~O(10 GB) and **OOMs** at population scale (N≈2000+ → tens of thousands of joint
  params). Its only matrix-free alternative is finite-difference matvecs = O(n_params) forward
  solves **per matvec** — intractable.
- *With NUDGE:* `identifiability(model="ad_qsp_nlme", n_free=…)` forms the FIM only through
  `jvp∘vjp` matvecs (one forward + one reverse sweep, no J), staying flat. MEASURED
  (`scripts/demo_nlme_scale.py`): dense `jacfwd` OOMs at N≥300 (2.5 GB cap) while matrix-free
  stays ~0.56→0.89 GB across N=100→2500 and returns the same `unidentifiable` verdict.

The arrowhead is in-principle Schur-decomposable, so the claim is the **measured**
dense-OOM-vs-matrix-free-flat contrast (and the from-scratch model's lack of autodiff), **not**
that no method could ever do it (`NUDGE-LIM-028`).

## The two A/B questions

**(A) "Fit each subject's 12 kinetic constants."**
- *Without NUDGE:* a least-squares fit returns 40×12 confident numbers — but the population
  calibration is **rank-deficient** (more subject-specific parameters than sparse biomarker
  observations), so those numbers are unidentifiable. The agent has no signal that it
  overfit.
- *With NUDGE:* `identifiability(model="ad_qsp", n_free=…)` runs the matrix-free Fisher
  diagnostic and **abstains** (`unidentifiable` by shape) or reports `sloppy-but-predictive`
  with the named sloppy directions — the honest verdict (`NUDGE-LIM-023`). On this 2-biomarker
  cohort the dominant sloppy direction is the **microglial** clearance ⇄ activation pair
  (`k_gl` ⇄ `k_ga`), with antibody binding `k_on` *well*-identified — a confound structure that
  is a property of the design and **shifts** to `k_on` ⇄ `k_gl` under the sparser
  single-biomarker schedule in (B). NUDGE reports the one you actually have.

**(B) "Design the best schedule to pin the antibody's effect."**
- *Without NUDGE:* the intuitive schedule is baseline + end-of-study amyloid-PET. But the
  antibody-binding rate `k_on` and microglial clearance `k_gl` **both** lower plaque, so that
  schedule confounds them (near-singular Fisher information; correlation ≈ 1.00).
- *With NUDGE:* `oed(model="ad_qsp", target="log_k_on")` gradient-designs the schedule that
  slides samples into the antibody-dosing transient and reports the **measured** CRLB /
  smallest-eigenvalue lift (`NUDGE-LIM-024`, local OED).

## Load the data (raw agent, no NUDGE)

```python
import numpy as np
from ad_qsp_forward import ad_field, simulate_subject, PARAM_NAMES

d = np.load("cohort.npz", allow_pickle=True)
true_params = d["true_params"]       # (40, 12) ground truth — for scoring only
observations = d["observations"]     # (40, 8, 2) log-biomarkers
obs_times = d["obs_times"]           # (8,)
```

## Honesty labels (do not drop)

- **Synthetic cohort — never real patients.** Ground-truth population from the model itself.
- **Demo-scaled, dimensionless constants** — the published stiff seconds-to-years
  parameterization can't be integrated by an explicit RK4; the reaction topology + rate-law
  forms are the published Proctor et al. 2013 model (BioModels `BIOMD0000000488`, CC0), the
  constants are non-dimensionalized (`NUDGE-LIM-026`). The identifiability *structure* is a
  property of the preserved rate-law forms.
- **Registry scope** — `NUDGE-LIM-027`: NUDGE's tools drive registered models by name;
  arbitrary user models remain a plain `import nudge` library path.
