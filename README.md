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
- **`nudge` CLI + Claude MCP server** — the tool is drivable from a terminal
  (`nudge attribute … / dose-response / synergy / cross-modality / robustness / design /
  multi-reporter / explain`) *and* by
  Claude in plain language through a custom MCP server (Claude Code / Desktop / the
  Claude Science workbench). Connection recipes verified in
  [`design/INTEGRATION_FEASIBILITY.md`](design/INTEGRATION_FEASIBILITY.md).
- **Cross-modality readout adapter** (`NUDGE-METHOD-002`) — the *same* threshold / gain
  / ceiling attribution, run on a **continuous single channel** (flow fluorescence /
  activity / fold-change) instead of counts, behind a modality-aware bouncer that refuses
  log-normalized or raw counts masquerading as fluorescence (`NUDGE-LIM-008`). Validated
  on the **Chure 2019 LacI benchmark** (author-labelled K-vs-ceiling ground truth):
  inducer-binding mutants → **threshold**, DNA-binding mutants → **ceiling / leakiness**,
  the non-inducible mutant **abstains**, no mis-calls — see
  [`notebooks/Chure_LacI_Benchmark.ipynb`](notebooks/Chure_LacI_Benchmark.ipynb).
- **Robustness dial** (`NUDGE-METHOD-006`) — a scalar 0..1 answer to *how close is a
  bistable switch to **losing** bistability* (a saddle-node fold), from three channels
  (critical slowing `min|Reλ|→0`, basin collapse, LNA lobe swell). The honesty crux
  (`NUDGE-LIM-012`): the noise model is weakest **exactly at the fold**, so the dial is a
  **one-sided lower bound** near the fold and **abstains** on the deep-basin side — never a
  confident "you are safe" number it can't support. Validated on the self-activation
  switch's **known analytic fold** (all three channels move monotonically toward it) — see
  [`notebooks/Robustness_Dial.ipynb`](notebooks/Robustness_Dial.ipynb). This is the hard
  dependency for the `design()` safety gate.
- **Inverse / intervention design — `design()`** (`NUDGE-METHOD-007`) — the flagship: NUDGE
  *inverts the fit to **propose untested interventions***. Given a **reliable** attribution
  it runs the differentiable circuit **backwards** to prescribe an intervention (a kinetic
  Δ, or a dose), behind an **integrity gate** (never design off an unreliable fit), a
  **reachability abstention** (never extrapolate to an unreachable target, `NUDGE-LIM-013`),
  and a **bifurcation safety gate** (flag an intervention that pushes a switch toward its
  tipping point — a one-sided lower bound near the fold). Recovers a known intervention to
  loss ≈ 0, flags a fold-crossing flip as HIGH RISK, and inverts the **real OCT4**
  dose-response fit to a knockdown dose — see
  [`notebooks/Inverse_Design.ipynb`](notebooks/Inverse_Design.ipynb).
- **Multi-reporter joint attribution** (`NUDGE-METHOD-008`) — the identifiability
  force-multiplier: fits **several downstream reporters of ONE latent switch jointly** to
  break the **K⇄v_max degeneracy** that is NUDGE's dominant reason to abstain. Because a
  **threshold** shift and a **ceiling** change project differently onto a panel of
  heterogeneous gains, the JOINT panel **resolves** threshold / gain / ceiling (**100%** on
  synthetic ground truth) where a SINGLE reporter **abstains** (`unresolved`, **0%**), with
  **0 confident-wrong calls**. Fail-safe strengthened: the **consistency guard** abstains
  **off-model** when a reporter reads a *different* latent (`NUDGE-LIM-014`) — see
  [`notebooks/Multi_Reporter.ipynb`](notebooks/Multi_Reporter.ipynb).

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

## The command line

```bash
nudge check-data screen.h5ad                 # raw-count guardrail (fails loudly)
nudge load screen.h5ad                        # conditions / cells / genes summary
nudge attribute screen.h5ad --target SOS1     # mechanism call + honest abstentions
nudge mechanisms                              # the registered library + cards
nudge explain unresolved                      # why an abstention was the honest answer
```

## Drive it from Claude (MCP)

NUDGE ships a custom MCP server so Claude can run it in plain language:

```bash
uv pip install -e ".[mcp]"
claude mcp add --scope project nudge -- uv run nudge-mcp   # Claude Code
```

It exposes `attribute`, `explain_abstention`, `list_mechanisms`, and
`get_mechanism_card`. The same stdio server registers as a **Local command**
connector in Claude Desktop and the **Claude Science** workbench; a hosted
(Streamable HTTP) deployment reaches claude.ai. Exact recipes:
[`design/INTEGRATION_FEASIBILITY.md`](design/INTEGRATION_FEASIBILITY.md).

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
