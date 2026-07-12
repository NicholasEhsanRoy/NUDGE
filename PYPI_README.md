# NUDGE (`nudge-bio`)

**N**ode/edge **U**ltrasensitivity **D**iagnostic for **G**ene-regulatory **E**ffects.

NUDGE is a mechanism-attribution tool for perturbation (Perturb-seq) screens. It fits a
compositional, differentiable gene-regulatory **circuit** model to single-cell count data
and classifies each perturbation by *mechanism* — does a knockdown move a switch's
**threshold** (`K`), change its **gain** (`n`), or shift its **ceiling** (`v_max`)? — a
distinction the field's default linear models cannot make. Its defining property is
**honesty: when the data cannot identify the mechanism, NUDGE abstains** (`unresolved` /
`off-model`) rather than emit a confident guess. Built on
[MADDENING](https://github.com/Microrobotics-Simulation-Framework/MADDENING), a
differentiable JAX graph-physics engine.

Gene-regulatory circuits are what NUDGE was **initially built for** — and what it's named
for — but that was only the first target. Its core is domain-general: a *compositional,
differentiable ODE model plus a calibrated abstention gate*. Anywhere a mechanism hides in
the **shape of a dynamical response** and a naive fit would over-claim, the same engine
applies — and it already reaches well beyond gene circuits to **microbial community
dynamics** (generalized Lotka–Volterra trajectory fits, `NUDGE-METHOD-012`), **protein
aggregation kinetics** (amyloid fibrillization, `NUDGE-METHOD-013`), and **differentiable
experimental design** (gradient-optimal experiments + matrix-free identifiability at scale,
`NUDGE-METHOD-014`). Each new domain is a new module over a frozen core — never a rewrite.

NUDGE originated at the **Built with Claude: Life Sciences** hackathon (July 2026), and is
itself an experiment in Claude-assisted development — the git history is written to make
that involvement auditable.

## Install

```bash
pip install nudge-bio
```

Core install pulls `maddening[ift]>=0.3.1`, `jax==0.5.1` (pinned), `numpy`, `optax`,
`pydantic`, `anndata`, `typer`, `pyyaml`. **Python ≥ 3.10.**

Optional extras:

| Extra | `pip install "nudge-bio[…]"` | What it adds |
|---|---|---|
| `bio` | real-data loaders | `scanpy` / `pertpy` for Tier-1/2 Perturb-seq loading + E-distance |
| `viz` | honest figures | `matplotlib` — the opt-in `nudge.viz` figure battery (core stays matplotlib-free) |
| `mcp` | Claude server | the `mcp` SDK for the `nudge-mcp` Model Context Protocol server |

## Quickstart

### Attribute a mechanism from a dose-response curve

The flagship positive: give NUDGE a knockdown dose-response of a readout signature and it
calls **switch vs graded** (or abstains). Here a genuinely ultrasensitive curve resolves to
`switch`:

```python
import numpy as np
from nudge.mechanisms.regulatory import hill_repression
from nudge.inference.dose_response import fit_dose_response, classify_dose_response

# A knockdown dose-response of a self-renewal signature (an ultrasensitive switch, n=6).
dose = np.linspace(0.0, 1.0, 22)
response = 0.2 + np.asarray(hill_repression(dose, 0.5, 6.0, 0.8))
response += np.random.default_rng(0).normal(0.0, 0.02, dose.shape)  # measurement noise

fit = fit_dose_response(dose, response, direction="repress", n_boot=200)
call, reason = classify_dose_response(fit)
print(f"call = {call!r}")
print(f"apparent gain n = {fit.n:.1f}  (95% CI {fit.ci_n[0]:.1f}-{fit.ci_n[1]:.1f})   "
      f"K = {fit.k_threshold:.2f}   R2 = {fit.r2:.2f}")
```

```text
call = 'switch'
apparent gain n = 6.5  (95% CI 6.0-7.5)   K = 0.49   R2 = 1.00
```

(`n` is an *apparent population gain*, not molecular cooperativity — NUDGE says so in the
`reason` string.) A curve whose doses don't span the inflection, or whose gain CI straddles
the ultrasensitive line, returns `unresolved` / `no-effect` instead.

### The two verbs — `fit` and `design`

```python
result = nudge.fit(adata, circuit)     # → MechanismMap (per-perturbation calls + uncertainty)
plan   = nudge.design(target)          # → ranked interventions, behind safety gates
```

`fit` wants **raw integer counts** — NUDGE owns the observation model (a negative-binomial +
dropout count model; the mechanism signal lives in the *shape* of the single-cell
distribution, which standard log/normalize pipelines destroy). Pass an `AnnData` of raw
counts with `obs["condition"]` labels (a `"WT"` control plus one label per perturbation).

**Honesty, by design:** a *single* under-powered snapshot at *one* operating point genuinely
tends to abstain — the gain⇄threshold degeneracy is real, and forcing a call would be
guessing. That is why the resolving capabilities read a **dose axis** (dose-response, above),
**several reporters** of one latent (`multi_reporter`), or **two operating points**. On a
single synthetic snapshot, `nudge.fit` honestly abstains:

```python
import nudge
from nudge.circuits import ras_switch_1node
from nudge.data.synthetic import PerturbationSpec

circuit = ras_switch_1node()
adata = nudge.generate_synthetic_perturbseq(
    circuit,
    perturbations=[PerturbationSpec("KD", scope="edge", index=0, param="K", factor=3.0)],
    n_cells_per_condition=1000, seed=0,
)
result = nudge.fit(adata, circuit)     # one operating point; raw counts checked at the boundary
for c in result.calls:
    print(c.perturbation, "->", c.mechanism.value, f"(confidence {c.confidence:.2f})")
```

```text
KD -> no-effect  (confidence 0.00)
```

That abstention is the tool working, not failing — NUDGE won't over-call a single snapshot.

### Command line

```bash
nudge check-data screen.h5ad                 # raw-count guardrail — fails loudly on normalized input
nudge attribute screen.h5ad --target SOS1    # mechanism call + honest abstentions/skips
nudge explain unresolved                     # why an abstention was the honest answer
nudge mechanisms                             # the registered mechanism library + cards
```

## What it does (capability map)

Each capability is fail-safe by construction (0 confident-wrong on its synthetic battery) and
ships a Mechanism Card, tests, and a decoy it must correctly resist.

| Capability | ID | One line |
|---|---|---|
| Dose-response attribution | `NUDGE-METHOD-001` | switch vs graded from a dose axis, or abstain |
| Cross-modality readout | `NUDGE-METHOD-002` | same K/n/v_max attribution on a continuous channel (fluorescence/activity) |
| Synergy / epistasis | `NUDGE-METHOD-003` | additive vs synergistic/buffering for a two-perturbation combo |
| Robustness dial | `NUDGE-METHOD-006` | 0..1 proximity of a bistable switch to losing bistability (one-sided near the fold) |
| Inverse design — `design()` | `NUDGE-METHOD-007` | invert a reliable fit to propose an intervention, behind a bifurcation safety gate |
| Multi-reporter joint fit | `NUDGE-METHOD-008` | several reporters of one latent switch break the K⇄v_max degeneracy |
| Hidden-node abstention | `NUDGE-METHOD-009` | turn a bare `off-model` verdict into a legible differential (never asserts a hidden node) |
| Differential attribution | `NUDGE-METHOD-010` | which knob differs for the SAME perturbation across two contexts |
| Constitutive-reporter control | `NUDGE-METHOD-011` | separate circuit ultrasensitivity from a nonlinear readout (the `NUDGE-LIM-006` fix) |
| Temporal / Lotka–Volterra | `NUDGE-METHOD-012` | trajectory-fit attribution for a microbial community (growth/interaction/susceptibility) |
| Fibrillization kinetics | `NUDGE-METHOD-013` | amyloid aggregation curve → identifiable composites + a measured gauge degeneracy |
| Optimal experimental design | `NUDGE-METHOD-014` | gradient-optimize *when to measure* to resolve a sloppy parameter |
| Honest figures — `nudge.viz` | (opt-in `[viz]`) | render any frozen result to a figure; abstentions draw as abstentions |

## The honesty differentiator

NUDGE's whole thesis is *never claim more than you measured*. A **confident-wrong** call — a
specific mechanism where the truth is "can't tell" — is the only hard failure; an abstention
or a one-sided bound is a feature, not a bug. The fail-safe guarantee is **measured** (0%
misclassification across hundreds of synthetic datasets) and **adversarially red-teamed across
seven rounds** — every found hole was independently reproduced, then closed or locked as a
regression decoy, with a final full re-scan finding 0 holes.

## Learn more

- Full README, capability essay, and status:
  [github.com/NicholasEhsanRoy/NUDGE](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/README.md)
- Live roadmap + architecture notes:
  [`design/STATE.md`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/design/STATE.md)
- The data contract (raw counts only, and why):
  [`docs/user_guide/data_contract.md`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/docs/user_guide/data_contract.md)
- Guided notebooks:
  [dose-response flagship](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/notebooks/OCT4_NANOG_Flagship.ipynb),
  [inverse design](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/notebooks/Inverse_Design.ipynb),
  [the full index](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/notebooks/README.md)
- Measured results (V&V + calibration):
  [`scripts/vv/FINDINGS.md`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/scripts/vv/FINDINGS.md)

## License

MIT. See [LICENSE](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/LICENSE).
