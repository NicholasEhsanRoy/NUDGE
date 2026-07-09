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
  (`nudge attribute … / explain / mechanisms`) *and* by Claude in plain language
  through a custom MCP server (Claude Code / Desktop / the Claude Science
  workbench). Connection recipes verified in
  [`design/INTEGRATION_FEASIBILITY.md`](design/INTEGRATION_FEASIBILITY.md).

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
