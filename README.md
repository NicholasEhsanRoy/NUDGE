# NUDGE

**N**ode/edge **U**ltrasensitivity **D**iagnostic for **G**ene-regulatory **E**ffects

NUDGE fits a compositional, differentiable gene-regulatory **circuit** model to
single-cell perturbation (Perturb-seq) data and classifies each perturbation by
*mechanism* — does it move a switch's **threshold**, change its **gain**, or
shift its **ceiling**? — a distinction the field's default linear models cannot
make. It then inverts the fit to propose untested interventions. Built on
[MADDENING](https://github.com/Microrobotics-Simulation-Framework/MADDENING), a
differentiable JAX graph-physics engine.

This repository is my project for the **Built with Claude: Life Sciences** remote
hackathon, 7–13 July 2026.

> **Design docs (read first):** [`design/PITCH.md`](design/PITCH.md) —
> plain-language; [`design/WORKING_BACKWARDS.md`](design/WORKING_BACKWARDS.md) —
> full PR/FAQ + engineering reasoning.

## Status

**Working proof of concept, calibrated.** The generative backbone and the fit
engine are built end to end: generate ground-truth Perturb-seq data → `fit()` →
recover the circuit → attribute threshold / gain / ceiling → **abstain when the
data can't say**. The fail-safe property is *measured* — **0% misclassification**
across hundreds of synthetic datasets (it abstains, never guesses wrong).

Highlights so far:
- **Phases 0–2 done:** circuit model, differentiable population fit, distributional
  losses, and the two-level abstention gates (calibrated `margin_k = 1.7`).
- **Tier-0.5 independent stochastic simulator** — a tau-leaping SSA with *emergent*
  bimodality that breaks the "inverse crime" of self-benchmarking; the fail-safe
  guarantee survives it.
- **Saddle transition-mode gain gate** — fail-safe mechanism attribution on
  genuinely bistable stochastic data (recovers *gain* where a naive fit is
  confidently wrong). See [`scripts/vv/FINDINGS.md`](scripts/vv/FINDINGS.md).

Not yet: the full decoy battery, Laplace uncertainty, real-data (T-cell) validation,
and `design()` inversion. See [`design/STATE.md`](design/STATE.md) for the live roadmap
and [`JUDGES_GUIDE.md`](JUDGES_GUIDE.md) for a guided tour.

## Install

```bash
uv venv && uv pip install -e ".[dev]"     # local development
pip install nudge-bio                      # (once published)
```

Requires `maddening[ift]>=0.3.1`, `jax==0.5.1` (pinned), Python ≥ 3.10.

## The two verbs

```python
import nudge

result = nudge.fit(adata, circuit)     # → MechanismMap (per-perturbation calls + uncertainty)
plan = nudge.design(target_outcome)    # → ranked interventions (stretch)
```

`fit` wants **raw integer counts** — NUDGE owns the observation model. See the
data contract in `docs/user_guide/data_contract.md`.

## Capabilities NOT provided

Scope discipline, stated up front:

- **Not** a general Perturb-seq hit-caller — it answers a sharper question than
  "is this gene a hit?"
- **Not** a black-box response predictor — the deliverable is the *mechanism*,
  not just the number.
- **Not** a clinical or diagnostic tool.
- **Not** a substitute for a wet-lab screen — it tells you which experiment is
  worth running next.
- **Not** a medical device; makes no clinical claims.

## License

MIT. See [LICENSE](LICENSE).
