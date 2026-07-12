# Connecting NUDGE to Claude Science (MCP)

NUDGE ships a custom **MCP server** (`nudge-mcp`) so Claude can drive it in plain language.
This guide connects that server to the **Claude Science** workbench and walks through a
concrete case — α‑synuclein aggregation kinetics — where NUDGE's honest refusal to over‑fit
is the whole point.

> **The payoff, up front.** Ask Claude Science to fit α‑synuclein's three microscopic
> aggregation rate constants, and instead of emitting three confident‑but‑meaningless
> numbers, NUDGE returns the two *identifiable composites* (κ, λ), **proves** the three
> individual constants are gauge‑degenerate, and prescribes the experiment that would resolve
> them. That is the tool working as intended.

---

## What you need to know about the runtime

Claude Science runs a **Local command** connector inside a *managed MCP runtime*. Two
constraints shape the setup:

- The runtime provides **`node`, `npx`, and `python3`** — but **not** `uv`/`uvx` or `pipx`.
- You may instead point `command` at the **absolute path of a binary installed on the
  machine**.

Because NUDGE is a *Python* server (there is no `pipx`/`uvx` in the runtime), the most
reliable recipe is to install it once and point the connector at the resulting console
script's **absolute path** — its shebang self‑selects the correct Python and dependencies, so
you never depend on *which* `python3` the runtime happens to resolve.

NUDGE is published on PyPI as **`nudge-bio`**; the MCP server needs the `[mcp]` extra.

---

## Option B (recommended): install once, point at the absolute path

Isolated install with `pipx` (or `pip install --user`), then use the absolute path:

```bash
pipx install "nudge-bio[mcp]"      # or:  pip install --user "nudge-bio[mcp]"
which nudge-mcp                     # -> e.g. /home/<you>/.local/bin/nudge-mcp
```

In Claude Science → **Settings › Connectors › Add connector › Local command**:

| Field | Value |
|---|---|
| **Name** | `nudge` (lowercase / digits / hyphens) |
| **Command** | the absolute path from `which nudge-mcp` (e.g. `/home/<you>/.local/bin/nudge-mcp`) |
| **Args** | *(none)* |
| **Env** | *(none)* |

Equivalent connector JSON:

```json
{ "mcpServers": { "nudge": {
    "type": "stdio",
    "command": "/home/<you>/.local/bin/nudge-mcp"
} } }
```

Why the absolute path: the console script begins with a shebang like
`#!/home/<you>/.local/pipx/venvs/nudge-bio/bin/python`, so running it always uses the Python
that has `nudge` + `mcp` installed — regardless of the runtime's own `python3`.

### Option A (try first if you like): plain `python3 -m`

If you installed into the *same* interpreter the runtime's `python3` resolves to
(`pip install "nudge-bio[mcp]"`), you can skip the absolute path:

| Field | Value |
|---|---|
| **Command** | `python3` |
| **Args** | `-m`  ·  `nudge.mcp.server` |

This is cleaner, but only works when the runtime's `python3` can import `nudge`. If tool
loading fails with **`No module named nudge`**, the runtime's `python3` is a different
interpreter than your install target — fall back to Option B.

### Option C (zero‑install fallback): self‑bootstrapping one‑liner

No prior install; the server pip‑installs itself on first launch. Needs the runtime's
`python3` to have **pip + network access**.

| Field | Value |
|---|---|
| **Command** | `python3` |
| **Args** | `-c` · `import importlib.util,subprocess,sys; importlib.util.find_spec('nudge') or subprocess.check_call([sys.executable,'-m','pip','install','-q','nudge-bio[mcp]']); import nudge.mcp.server as s; s.main()` |

---

## Pre‑flight (prove the server launches, before wiring anything)

```bash
python -c "from nudge.mcp.server import build_server; build_server(); print('MCP builds OK')"
nudge-mcp        # starts the stdio server (warms JAX caches, then serves); Ctrl-C to exit
```

A clean start with no traceback means the server is good; the connector just needs to spawn
this same command.

## Smoke‑test the connection (no data file)

Confirm the wire independently of any file access. In Claude Science:

> "Using the **nudge** connector, call `list_mechanisms`, then show me the `fibrillization`
> mechanism card."

Approve the tool call when prompted (set **Always allow** once you trust it). Getting the
mechanism list + card back confirms the connection.

The server exposes these tools: `list_mechanisms`, `get_mechanism_card`, `explain_abstention`,
`attribute`, `dose_response`, `synergy`, `cross_modality`, `robustness`, `design`,
`multi_reporter`, `diagnose_abstention`, `differential`, `differential_robust`, `lotka`,
`fibrillization`, `constitutive`, `render_figure`.

---

## The case: α‑synuclein aggregation kinetics (Parkinson's)

α‑synuclein is the amyloid protein whose aggregation underlies Parkinson's disease. Its
fibrillization is **secondary‑nucleation‑dominated** (κ ≫ λ; Buell et al., *PNAS* 2014) — the
regime NUDGE's fibrillization model (`NUDGE-METHOD-013`) is built for.

### 1 · Make a curve

Any single aggregation trace works as a two‑column CSV — `time` and `mass_fraction` (∈ [0, 1]).
This snippet writes a **synthetic** α‑synuclein‑realistic ThT curve (secondary‑dominated
kinetics; not real experimental data). Swap in your own CSV if you have one.

```python
import numpy as np, pandas as pd
from nudge.mechanisms.fibrillization import AggregationParams, simulate_aggregation_curve

# α-synuclein-like: secondary nucleation dominant  ->  κ = √(2·k₊·k₂) = 1.0,  λ = √(2·k₊·kₙ) = 0.01
asyn  = AggregationParams(k_n=1e-4, k_plus=0.5, k_2=1.0)
curve = simulate_aggregation_curve(params=asyn, m_tot=1.0, t_max=40.0,
                                   n_obs=60, obs_noise=0.015, seed=1)
y = np.clip(curve.signal.mean(axis=0), 0.0, 1.0)          # mean ThT trace across replicates
pd.DataFrame({"time_h": curve.t_obs, "mass_fraction": y}).to_csv("alpha_synuclein_ThT.csv",
                                                                 index=False)
```

**File visibility caveat.** A Local‑command connector runs sandboxed with a *per‑connector
writable directory*, so the `fibrillization` tool can only read a path its process can reach.
Put `alpha_synuclein_ThT.csv` in that writable directory (or have Claude Science write the CSV
there first, then call the tool on that path). If a path fails, that is almost always why —
the smoke test above needs no file and isolates connection problems from file‑access ones.

### 2 · Ask the naive question

Phrase it as the *over‑ask* — the honest answer is the surprise:

> "I have a single ThT aggregation curve for **α‑synuclein** (mass fraction vs time) at
> `alpha_synuclein_ThT.csv`. Fit its microscopic aggregation rate constants — primary
> nucleation kₙ, elongation k₊, and secondary nucleation k₂ — and give me their values."

Claude Science should call `fibrillization` and return something equivalent to:

```
call: composites-identified
κ (kappa)  ≈ 1.00    95% CI [0.998, 1.005]      # identifiable
λ (lambda) ≈ 0.0099  95% CI [0.0098, 0.010]     # identifiable
individual_k_identifiable: False                 # the three constants are NOT identifiable
null_direction: [+0.577, -0.577, +0.577]         # the exact gauge symmetry
guidance: run a concentration series + a seeded / elongation reference to resolve kₙ, k₊, k₂
```

### 3 · What "it works" looks like

Success is **not** three rate constants — it is the abstention: NUDGE reports the two
identifiable composites, states that the three microscopic constants are gauge‑degenerate
(a single curve cannot separate them; `NUDGE-LIM-021`), and prescribes the experiment that
would. A tool that answered with three confident kₙ / k₊ / k₂ values from one curve would be
*guessing* — NUDGE refuses to. (Ask Claude Science to `render_figure` with `kind="aggregation"`
to get the curve + the gauge‑null picture inline.)

To see NUDGE's prescription actually *resolve* all three constants, supply a concentration
series **plus** a seeded/elongation reference (the Meisl discipline) — the fibrillization
capability handles that path too.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Command 'uvx' is not available…` | runtime has no `uvx`/`pipx` | use Option B (absolute path) or A (`python3 -m`) |
| `No module named nudge` | the runtime's `python3` isn't your install target | Option B — point at the absolute `nudge-mcp` path (shebang self‑selects Python) |
| Tools load, but `fibrillization` can't read the file | sandbox file visibility | put the CSV in the connector's writable directory |
| First tool call is slow | JAX compile‑cache warm‑up on server start | expected once; subsequent calls are fast (the server is long‑lived) |

## The remote (hosted) alternative

Only the browser‑only `claude.ai` app cannot spawn a local process; it needs a **Remote**
connector (an HTTPS URL). The MCP SDK supports `transport="streamable-http"` over the same
`build_server()`, so a hosted deployment is a transport switch, not a rewrite. NUDGE ships the
`stdio` server today; hosting is left as a follow‑on (see
[`design/INTEGRATION_FEASIBILITY.md`](../../design/INTEGRATION_FEASIBILITY.md)).
