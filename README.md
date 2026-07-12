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

> **Design docs (read first):** [`design/PITCH.md`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/design/PITCH.md) —
> plain-language; [`design/WORKING_BACKWARDS.md`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/design/WORKING_BACKWARDS.md) —
> full PR/FAQ + engineering reasoning.

## Status

**Working, calibrated, and adversarially red-teamed — this is the `0.1.0` first
release.** The generative backbone and the fit engine are built end to end:
generate ground-truth Perturb-seq data → `fit()` → recover the circuit →
attribute threshold / gain / ceiling → **abstain when the data can't say**. The
fail-safe property is *measured* — **0% misclassification** across hundreds of
synthetic datasets (it abstains, never guesses wrong).

Highlights so far:
- **Phases 0–2 done:** circuit model, differentiable population fit, distributional
  losses, and the two-level abstention gates (calibrated `margin_k = 1.7`).
- **Tier-0.5 independent stochastic simulator** — a tau-leaping SSA with *emergent*
  bimodality that breaks the "inverse crime" of self-benchmarking; the fail-safe
  guarantee survives it.
- **Saddle transition-mode gain gate** — fail-safe mechanism attribution on
  genuinely bistable stochastic data (recovers *gain* where a naive fit is
  confidently wrong). See [`scripts/vv/FINDINGS.md`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/scripts/vv/FINDINGS.md).
- **`nudge` CLI + Claude MCP server** — the tool is drivable from a terminal
  (`nudge attribute … / dose-response / synergy / cross-modality / robustness / design /
  multi-reporter / diagnose-abstention / differential / constitutive / lotka / fibrillization /
  oed / viz / explain`, plus the utility verbs `load` / `check-data` / `mechanisms`) *and* by
  Claude in plain language through a custom MCP server (Claude Code / Desktop / the
  Claude Science workbench). Connection recipes verified in
  [`design/INTEGRATION_FEASIBILITY.md`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/design/INTEGRATION_FEASIBILITY.md).
- **Cross-modality readout adapter** (`NUDGE-METHOD-002`) — the *same* threshold / gain
  / ceiling attribution, run on a **continuous single channel** (flow fluorescence /
  activity / fold-change) instead of counts, behind a modality-aware bouncer that refuses
  log-normalized or raw counts masquerading as fluorescence (`NUDGE-LIM-008`). Validated
  on the **Chure 2019 LacI benchmark** (author-labelled K-vs-ceiling ground truth):
  inducer-binding mutants → **threshold**, DNA-binding mutants → **ceiling / leakiness**,
  the non-inducible mutant **abstains**, no mis-calls — see
  [`notebooks/Chure_LacI_Benchmark.ipynb`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/notebooks/Chure_LacI_Benchmark.ipynb).
- **Robustness dial** (`NUDGE-METHOD-006`) — a scalar 0..1 answer to *how close is a
  bistable switch to **losing** bistability* (a saddle-node fold), from three channels
  (critical slowing `min|Reλ|→0`, basin collapse, LNA lobe swell). The honesty crux
  (`NUDGE-LIM-012`): the noise model is weakest **exactly at the fold**, so the dial is a
  **one-sided lower bound** near the fold and **abstains** on the deep-basin side — never a
  confident "you are safe" number it can't support. Validated on the self-activation
  switch's **known analytic fold** (all three channels move monotonically toward it) — see
  [`notebooks/Robustness_Dial.ipynb`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/notebooks/Robustness_Dial.ipynb). This is the hard
  dependency for the `design()` safety gate.
- **Inverse / intervention design — `design()`** (`NUDGE-METHOD-007`) — the flagship: NUDGE
  *inverts the fit to **propose untested interventions***. Given a **reliable** attribution
  it runs the differentiable circuit **backwards** to prescribe an intervention (a kinetic
  Δ, or a dose), behind an **integrity gate** (never design off an unreliable fit), a
  **reachability abstention** (never extrapolate to an unreachable target, `NUDGE-LIM-013`),
  and a **bifurcation safety gate** (flag an intervention that pushes a switch toward its
  tipping point — firing on a relative proximity rise **or** an absolute landing at/above
  the near-fold cut, so it agrees with `classify_robustness` on the same circuit; a
  one-sided lower bound near the fold). Recovers a known intervention to
  loss ≈ 0, flags a fold-crossing flip as HIGH RISK, and inverts the **real OCT4**
  dose-response fit to a knockdown dose — see
  [`notebooks/Inverse_Design.ipynb`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/notebooks/Inverse_Design.ipynb).
- **Multi-reporter joint attribution** (`NUDGE-METHOD-008`) — the identifiability
  force-multiplier: fits **several downstream reporters of ONE latent switch jointly** to
  break the **K⇄v_max degeneracy** that is NUDGE's dominant reason to abstain. Because a
  **threshold** shift and a **ceiling** change project differently onto a panel of
  heterogeneous gains, the JOINT panel **resolves** threshold / gain / ceiling (**100%** on
  synthetic ground truth) where a SINGLE reporter **abstains** (`unresolved`, **0%**), with
  **0 confident-wrong calls**. Fail-safe strengthened: the **consistency guard** abstains
  **off-model** when a reporter reads a *different* latent, and a **floor-consistency gate**
  abstains **`unresolved`** when a per-condition batch/depth scale on the perturbed panel
  aliases onto a `ceiling` — a genuine ceiling leaves each reporter's OFF baseline fixed, a
  batch rescales it (`NUDGE-LIM-014`; near-zero-floor panels are a documented bound) — see
  [`notebooks/Multi_Reporter.ipynb`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/notebooks/Multi_Reporter.ipynb).
- **Hidden-node abstention** (`NUDGE-METHOD-009`) — the **abstention half only**: turns a
  bare **`off-model`** verdict into a legible **six-cause differential** (genuinely
  not-a-switch / nonlinear readout / off-target / wrong topology / batch-depth confound /
  hidden node), each with its documented limitation and the experiment that would
  distinguish it. The honesty crux (`NUDGE-LIM-015`): positive hidden-node identification is
  *not identifiable* from an off-model verdict (the causes are observationally overlapping),
  so NUDGE **never** asserts a hidden node — the strongest it says is that an off-axis
  residual is *consistent with, does not prove* an unmeasured regulator (`nudge
  diagnose-abstention`) — see
  [`notebooks/Hidden_Node_Abstention.ipynb`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/notebooks/Hidden_Node_Abstention.ipynb).
- **Comparative / differential attribution** (`NUDGE-METHOD-010`) — the SAME perturbation
  in **two contexts** (drug-resistant vs sensitive line; donor A vs B; disease vs healthy):
  isolate whether the difference is in **threshold** (`K`), **gain** (`n`), or **ceiling**
  (`v_max`), a call linear differential expression structurally cannot make (a raised
  ceiling → **more of the same drug**; a rewired gain/threshold → a **different class**).
  Fits the two contexts **jointly** and **BIC-selects** which single knob differs, or
  abstains. The honesty crux (`NUDGE-LIM-016`): a depth/batch shift aligned with the context
  axis mimics a ceiling difference, so depth is pinned **per context** from each control and
  NUDGE **abstains** when the two contexts' depths differ beyond a ratio; a red-team-found
  **per-condition affine confound on one context's *perturbed* cells** — an additive
  offset (P1), a large multiplicative scale (P4), or a **small** multiplicative scale (P5,
  `c ≈ 1.15–1.25`, which per-magnitude OFF-cluster bands miss) — is one class, and all of it is
  caught by the load-bearing **free-affine "earn" guard**: before any positive call NUDGE refits the
  perturbed context with the affine `(s, o)` as a **free nuisance** and abstains unless the winning
  knob still **earns** its BIC parameter over that null (the whole confound family is inside the
  null's span, so it cannot; a genuine mechanism reshapes the distribution and earns ≫ margin). This
  **closes the whole uniform affine class** — 0 confident-wrong, positives preserved; the honest
  residual is a *non-uniform* perturbed-side scale (identical to a genuine ceiling), which needs an
  independent inert-feature anchor. Never a spurious mechanism
  from an artifact (`nudge differential`) — see
  [`notebooks/Differential.ipynb`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/notebooks/Differential.ipynb).
- **Constitutive-reporter calibration control** (`NUDGE-METHOD-011`) — removes a known
  **confident-wrong** failure mode (`NUDGE-LIM-006`: a **nonlinear measurement readout**
  misattributed as a **circuit switch**). Only the composition readout∘circuit is observed,
  so from one population the circuit gain and the reporter nonlinearity are unidentifiable —
  the profile over circuit `n` is **FLAT** (you cannot even tell a switch exists). A
  **constitutive-reporter control** — the reporter driven at **known** activity doses,
  *bypassing the circuit* — anchors the readout (using **readout parameters only**, no
  circuit leak), and a profile over circuit `n` then **rejects "no switch"** for a genuine
  switch → **biological**, or **abstains** for a linear circuit whose apparent
  ultrasensitivity lives in the reporter. **Adversarially bounded (`NUDGE-LIM-019`):** it
  turns the confident false positive into a correct call **or** an honest abstention and never
  a bare knob — but `biological-switch` is a falsifiable positive claim, valid **only when the
  control and the circuit population share a capture/depth scale** (co-measured); an unmodeled
  capture mismatch between the two populations re-opens `NUDGE-LIM-006` (a red-team round-2
  finding, now locked as a strict-xfail decoy). It also does **not** point-identify `n` (needs
  a second anchor; `NUDGE-LIM-018`) (`nudge constitutive --demo`) — see
  [`notebooks/Constitutive_Control.ipynb`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/notebooks/Constitutive_Control.ipynb).
- **Temporal / Lotka–Volterra attribution** (`NUDGE-METHOD-012`) — *same engine, new field
  of biology*. The first **trajectory-fit** capability: it points the same
  abstain-and-attribute philosophy at a **microbial community** (generalized Lotka–Volterra,
  `dxᵢ/dt = xᵢ(αᵢ + Σⱼ βᵢⱼxⱼ + εᵢ·u(t))`) and attributes a perturbation to a change in
  **growth (α) / interaction (β) / antibiotic-susceptibility (ε)** — from **trajectories**,
  not a snapshot — in a module that touches **neither `fit.py` nor `core/`**. The **ε** axis
  is the identifiable positive; **α⇄βᵢᵢ** (growth vs self-limitation, `Kᵢ=−αᵢ/βᵢᵢ`) is
  **degenerate near equilibrium** and NUDGE **abstains**, with the degeneracy **MEASURED** by
  a near-singular Laplace curvature (`NUDGE-LIM-020`), never asserted. **0 confident-wrong**
  across the battery; real coda on the **Stein 2013** clindamycin→*C. difficile* series
  surfaces the honest abstention (*C. difficile*'s bloom is interaction-mediated, ε≈−0.31)
  (`nudge lotka`) — see
  [`notebooks/Temporal_Ecology.ipynb`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/notebooks/Temporal_Ecology.ipynb).
- **Protein aggregation / fibrillization attribution** (`NUDGE-METHOD-013`) — *the
  efficiency demo, and a THIRD dynamical system*. NUDGE analyzes an **amyloid aggregation
  curve** (the sigmoidal ThT trace) by fitting the filament master equation's principal
  moments (`dP/dt = k_n·m^{n_c} + k_2·m^{n_2}·M`, `dM/dt = 2·k_+·m·P`) and, **in one
  deterministic call**, returns the two identifiable composites **κ ≈ 1, λ ≈ 0.01** and the
  **measured** non-identifiability of the three microscopic rate constants — an *exact gauge
  symmetry* `(k_n, k_+, k_2) → (k_n/α, α·k_+, k_2/α)` (Fisher condition number → ∞, null
  `[+0.577, −0.577, +0.577]`, gauge check ~1e-16; `NUDGE-LIM-021`). A **control LLM agent
  took 12.2 min / 28 turns / 6 scripts** to hand-derive the same answer. A concentration
  series + a seeded anchor (the Meisl discipline) resolves all three (0 confident-wrong); an
  inhibitor is attributed to the microscopic step it lowers — primary / elongation /
  secondary nucleation — or abstained on (`nudge fibrillization`) — see
  [`notebooks/Aggregation_Kinetics.ipynb`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/notebooks/Aggregation_Kinetics.ipynb).
- **Optimal experimental design — the differentiability moat** (`NUDGE-METHOD-014`) — *the
  white-box advantage a black-box ODE solver can't offer*. Because NUDGE's forward model is
  **differentiable**, the Fisher-information design criterion is itself a differentiable
  function of the *experiment*, so `∂criterion/∂φ` is available by autodiff and NUDGE
  **gradient-optimizes *when to measure*** to the exact schedule that resolves a sloppy,
  degenerate parameter — a black box has no `∂/∂φ` and can only grid-search (exponential in
  the design size). This makes the gLV growth⇄self-limitation abstention **actionable**:
  from a naive **near-equilibrium** schedule (where α⇄βᵢᵢ is degenerate, cond 136) the
  optimal design puts samples in the growth **transient** and **measurably** resolves α —
  **CRLB 31× better, FIM smallest eigenvalue 18× better** (600× on a gLV community); all
  D-/E-/CRLB objectives agree. Local OED (`NUDGE-LIM-024`): the gains are measured at the
  nominal θ₀, not extrapolated (`nudge oed`) — see
  [`notebooks/Optimal_Experimental_Design.ipynb`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/notebooks/Optimal_Experimental_Design.ipynb).
- **Honest figures — `nudge.viz`** (opt-in `[viz]` extra) — an additive,
  provenance-carrying figure layer that renders NUDGE's frozen result objects from one
  `render(result, out=…)` call, with a renderer for every result type (dose-response,
  attribution, identifiability/sloppiness, epistasis, differential, multi-reporter,
  temporal/gLV, aggregation, constitutive, diagnose, design, OED, cross-modality,
  robustness). It only *reads* results (never re-fits; never touches `fit`/`core`), and the
  honesty is **structural**: `render()` applies the abstention overlay itself off each
  result's own verdict, so a figure can never draw an abstention as a confident call, and
  one-sided bounds draw as **open-ended arrows**, never error bars. The flagship
  **dose-response dual panel** shows the real ESC-screen **OCT4 → switch** (n≈6.7) beside the
  honest **NANOG → unresolved** in one frame. Every figure also ships a standalone `fig.py`
  that regenerates it from the fit's output (no re-fit; **pixel-identical**) — the Claude
  Science provenance grain. Reachable from the command line — `nudge viz KIND --demo --out
  fig.png` (any kind; `--json FILE` replays a saved run) — the MCP `render_figure` tool, the
  per-result CLI flag (`nudge dose-response … --fig-out fig.png`), and
  [`notebooks/OCT4_NANOG_Flagship.ipynb`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/notebooks/OCT4_NANOG_Flagship.ipynb).

**The fail-safe guarantee is adversarially red-teamed across seven rounds**
([`design/FAILSAFE_REDTEAM.md`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/design/FAILSAFE_REDTEAM.md)
→ [`_7.md`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/design/FAILSAFE_REDTEAM_7.md),
plus the auditable red-team → fix → independent-audit
[`design/hardening/`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/design/hardening/LEDGER.md)
loop): dedicated passes tried to force *any* capability into a confident, specific, wrong
call past its abstention gates, and every found hole was independently reproduced,
fixed-or-locked, and re-audited. Round 1 surfaced **2 holes** — a near-fold operating point
corrupting the multi-point covariance fit (**closed**, `NUDGE-LIM-017`) and an additive
ambient/batch offset faking synergy where no safe runtime gate exists (**locked** as a
strict-`xfail` decoy + a sharpened `NUDGE-LIM-009`). Round 2 hardened the constitutive
capture-scale bound (`NUDGE-LIM-019`) and replaced the near-fold knife-edge with graded
down-weighting; round 3 closed the **`design()` safety gate** so it fires on an **absolute**
near-fold check reusing the same threshold `classify_robustness` uses (`NUDGE-LIM-013`). The
hardening loop then closed the differential per-condition **affine confound class** — P1
(additive) / P4 (large multiplicative) / P5 (small multiplicative), all folded into one
**free-affine "earn" gate (4d)** that abstains unless the winning knob out-earns a free
affine null (`NUDGE-LIM-016`) — and P6, where the matrix-free identifiability path mislabeled
an **isolated structural Fisher-null** as `well-constrained` (it verified eigenpair-*ness*,
not smallest-*ness*) → fixed with an exact dense-via-matvec deferral + a one-sided
inverse-iteration null probe (`NUDGE-LIM-023`, sharpened to *major*). The final full re-scan
found **0 holes**
([`design/FAILSAFE_REDTEAM_7.md`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/design/FAILSAFE_REDTEAM_7.md)).
A found hole is a *win*: closed or locked, never hidden.

**Genuinely deferred for 0.1.0** (honest gaps, stated up front, not hidden): **real-data
lock-ins for the newest capabilities** — constitutive-control (needs a constitutively-driven
reporter titration, uncommon in public data), multi-reporter, differential, fibrillization,
gradient OED, and the gLV temporal path are synthetic-validated (the gLV path with a real-data
*coda* on Stein 2013) and their full real-data demos are `needs_data` follow-ups; and
**per-result provenance tracking** (`provenance.py`) is a Phase-0 stub schema. (Shipped, so no
longer "not yet": **temporal / Lotka–Volterra attribution** (`NUDGE-METHOD-012`),
**fibrillization** (`013`), **gradient OED** (`014`), the **`nudge.viz` figure battery**,
`design()` inversion, the Laplace uncertainty layer, and a real-data fail-safe validation on
the Gladstone T-cell screen, where NUDGE honestly abstained.)
See [`design/STATE.md`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/design/STATE.md)
for the live roadmap and
[`JUDGES_GUIDE.md`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/JUDGES_GUIDE.md) for a
guided tour.

## Install

```bash
uv venv && uv pip install -e ".[dev]"     # local development
pip install nudge-bio                      # from PyPI
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

It exposes the full modelling surface as tools — `attribute`, `dose_response`, `synergy`,
`cross_modality`, `robustness`, `design`, `multi_reporter`, `diagnose_abstention`,
`differential` (+ `differential_robust`), `lotka`, `fibrillization`, `constitutive`, and
`render_figure` — alongside the knowledge tools `explain_abstention`, `list_mechanisms`, and
`get_mechanism_card`. The same stdio server registers as a **Local command**
connector in Claude Desktop and the **Claude Science** workbench; a hosted
(Streamable HTTP) deployment reaches claude.ai. A step-by-step **Claude Science**
walkthrough (connect + an α-synuclein / Parkinson's aggregation-kinetics case) is in
[`docs/user_guide/claude_science.md`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/docs/user_guide/claude_science.md);
the verified connection recipes are in
[`design/INTEGRATION_FEASIBILITY.md`](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/design/INTEGRATION_FEASIBILITY.md).

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

MIT. See [LICENSE](https://github.com/NicholasEhsanRoy/NUDGE/blob/main/LICENSE).
