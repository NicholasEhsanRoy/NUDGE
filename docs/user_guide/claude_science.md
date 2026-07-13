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

Claude Science runs a **Local command** connector inside a *managed, filesystem‑restricted
MCP sandbox*. Three constraints shape the setup:

- The runtime provides **`node`, `npx`, and `python3`** — but **not** `uv`/`uvx` or `pipx`.
- **Your home directory is not visible inside the sandbox.** A binary under `~` (e.g. a
  `pipx` or `pip --user` install at `~/.local/bin/nudge-mcp`) will fail to load with
  *"Connection closed … command not found inside the MCP sandbox"* — the script, its Python
  interpreter, and its site‑packages all live under `~`, which the sandbox can't read.
- The sandbox **can** see system paths (`/usr/local`, `/opt`), and you can grant extra
  directories read‑only via `[sandbox] user_read_paths` in the connector `config.toml`.
- **The connector command is a single field** — there is no separate *args* or *env* box.
  Pass arguments and environment variables inline via `/usr/bin/env`:
  `/usr/bin/env VAR=val /opt/nudge/bin/nudge-mcp`.
- **Figures can only come back inline, and long calls are killed at ~60 s** — the two
  facts that shape how you *use* the server (not just install it). We verified both against
  the live Claude Science harness; see **[Figures and long jobs](#figures-come-back-inline-long-jobs-run-async)**
  below. In short: set `NUDGE_ENV=cloud` so `render_figure` returns the image as **inline
  base64**, and run heavy tools through **`job_submit` / `job_status`**.

So the reliable recipe is to install NUDGE **on a system path** and point the connector at
the console script there — its shebang self‑selects the correct Python and dependencies, and
nothing points back into `~`. NUDGE is published on PyPI as **`nudge-bio`**; the MCP server
needs the `[mcp]` extra, and the `[viz]` extra so the `render_figure` tool can draw the figure
server‑side (install `nudge-bio[mcp,viz]`).

---

## Option B (recommended): install into `/opt`, point at the absolute path

Install NUDGE into a venv on a **system path** the sandbox can see (`/opt`). The script, its
Python, and all dependencies then live under `/opt` — no part of it reaches into `~`.

```bash
sudo python3 -m venv /opt/nudge
sudo /opt/nudge/bin/pip install "nudge-bio[mcp,viz]"
# sanity (does not block):
/opt/nudge/bin/python -c "from nudge.mcp.server import build_server; build_server(); print('MCP OK')"
```

In Claude Science → **Settings › Connectors › Add connector › Local command**:

| Field | Value |
|---|---|
| **Name** | `nudge` (lowercase / digits / hyphens) |
| **Command** | `/usr/bin/env NUDGE_ENV=cloud /opt/nudge/bin/nudge-mcp` |
| **Args** | *(none)* |
| **Env** | *(none)* |

Equivalent connector JSON:

```json
{ "mcpServers": { "nudge": {
    "type": "stdio",
    "command": "/usr/bin/env NUDGE_ENV=cloud /opt/nudge/bin/nudge-mcp"
} } }
```

Why `/opt`: the console script's shebang is `#!/opt/nudge/bin/python`, so running it always
uses the venv that has `nudge` + `mcp` installed — and every path (script, interpreter,
site‑packages) is on a sandbox‑visible system path.

Why the `/usr/bin/env NUDGE_ENV=cloud …` prefix: the connector command is a **single
field** (no separate env box), and `NUDGE_ENV=cloud` switches `render_figure` into the
**inline‑base64** transport that is the only one that works in this sandbox (next section).
`/usr/bin/env VAR=val PROG` is the portable one‑liner for "set an env var, then exec". Drop
the prefix (`command = /opt/nudge/bin/nudge-mcp`) only on a host where the client *can* read
a file path the server writes — then figures come back as paths instead.

### No‑sudo alternative: keep a home install, grant read access

If you'd rather install with `pipx` (`pipx install "nudge-bio[mcp,viz]"`), the binary lands under
`~/.local` — invisible by default. Grant the sandbox read‑only access to it in the connector
`config.toml`, then point the command at `~/.local/bin/nudge-mcp`:

```toml
[sandbox]
user_read_paths = [
  "/home/<you>/.local/bin",
  "/home/<you>/.local/pipx/venvs/nudge-bio",   # the interpreter + site-packages the shebang needs
]
```

(The venv path is what `pipx list` reports — grant the whole `nudge-bio` venv, not just `bin`.)

### Option A (try first if you like): plain `python3 -m`

If you installed into the *same* interpreter the runtime's `python3` resolves to
(`pip install "nudge-bio[mcp,viz]"`), you can skip the absolute path:

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
| **Args** | `-c` · `import importlib.util,subprocess,sys; importlib.util.find_spec('nudge') or subprocess.check_call([sys.executable,'-m','pip','install','-q','nudge-bio[mcp,viz]']); import nudge.mcp.server as s; s.main()` |

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

---

## Figures come back inline; long jobs run async

Two properties of the Claude Science connector shape how you *use* NUDGE (we verified both
end‑to‑end against the live harness — don't fight them):

### 1 · A figure can only be delivered as inline base64

The connector mounts its shared data directory (e.g. `/opt/nudge/data`) **read‑only** —
writing there raises `EROFS` — and the connector's own writable temp directory is **not
visible to the client**. `render_figure` runs *on the connector*, so it has nowhere to write
a file the client could then read: **file‑path delivery is structurally impossible here.**
Custom `nudge://…` **MCP resource URIs are also a dead end** — the harness bridges
`tools/call` but not `resources/read`, so a resource the server exposes can't be
dereferenced.

The transport that works is **inline base64**, which `NUDGE_ENV=cloud` selects. `render_figure`
then returns:

```jsonc
{
  "transport": "inline",
  "image_base64": "iVBORw0KGgo…",   // decode + display this
  "mime_type": "image/png",          // or image/gif for an animation
  "code": "# fig.py … viz.render(…)",// the standalone regenerator (provenance)
  "data": "{ …fig.data.json… }",     // the figure-data sidecar (provenance), capped
  "caption": "OCT4 → switch …",
  "abstained": false,
  "kind": "dose_response"
}
```

Decode `image_base64` into the notebook/kernel and display it **immediately**; do **not**
echo the raw base64 string back into the chat. Animated GIFs are size‑disciplined before
encoding (downscaled, frame‑limited, a tight palette, and a never‑inflate guard), capped at
~1.5 MB of base64; above the cap `render_figure` falls back to a **static final‑frame PNG
preview** (with `image_base64_omitted_reason` set) rather than truncating. The `code` and
`data` fields are the **provenance grain** — attach them to the artifact so the figure is
reproducible from the fit's own output (no re‑fit).

> On a host where the client *can* read a server‑written path (a local Claude Desktop /
> Claude Code, `NUDGE_ENV` unset), the same tool writes to `NUDGE_ARTIFACT_DIR` (fallback:
> the system temp dir) and returns `image_path` / `code_path` / `data_path` instead. One
> tool, two transports, chosen by `NUDGE_ENV`.

### 2 · Heavy calls exceed the ~60 s per‑call cap — use `job_submit` / `job_status`

The connector kills any single tool call that runs longer than ~60 s. Several NUDGE tools
legitimately exceed that — a covariance `attribute`, an OED optimization, the `constitutive`
demo (~64 s on CPU). Run them as a **background job**:

```
job_submit(tool="constitutive", args_json="{\"demo\": true}")   →  { "job_id": "…", "status": "running" }   (returns in <1 s)
job_status(job_id="…")   →  { "status": "running", "elapsed_s": 21.0 }        (poll…)
job_status(job_id="…")   →  { "status": "done", "elapsed_s": 64.3, "result": { … } }
```

`job_submit(tool, args_json)` starts the real tool in a background thread and returns a
`job_id` immediately; poll `job_status(job_id)` until `status` is `done` (carrying the tool's
real `result`) or `error`. JAX releases the GIL during compilation/execution, so the worker
doesn't block the server. Each individual call stays well under the cap while the real
compute takes however long it takes. The fast, always‑synchronous tools
(`list_mechanisms`, `dose_response`, `get_mechanism_card`, `explain_abstention`,
`diagnose_abstention`) can just be called directly. A slow `render_figure` `demo=True`
build (e.g. `constitutive` / `oed` / `temporal` / `aggregation`) is itself a good
`job_submit` candidate.

### Agent notes (driving the connector well)

A short checklist for an agent using the `nudge` connector inside Claude Science:

- **Figures arrive as inline base64, not files.** Resource URIs aren't dereferenceable
  here. Decode `image_base64` into the kernel and display it right away; **never echo the
  base64 blob** into the conversation.
- **Attach the provenance.** The `code` (a standalone `fig.py`) and `data`
  (`fig.data.json`) fields regenerate the figure from the fit's output with no re‑fit —
  attach them as the artifact's provenance.
- **Trust the honesty overlay.** Every figure stamps its own abstention off the result's
  verdict; when `abstained` is `true`, present it *as* an abstention (an "I can't tell"
  figure), never as a confident call.
- **Wrap heavy calls.** A fit / OED / `constitutive` demo can exceed the 60 s cap — submit
  it with `job_submit` and poll `job_status` instead of calling it directly.
- **Animations too.** `render_figure(..., animate=True)` returns a size‑disciplined GIF (or
  a final‑frame PNG preview above the cap) inline; the `animate`‑able kinds are
  `constitutive`, `oed`, `robustness`, `aggregation`, `temporal`, `multi_reporter`,
  `identifiability`, `design`, `dose_response`, `cross_modality`.

The server exposes these tools: `list_mechanisms`, `get_mechanism_card`, `explain_abstention`,
`attribute`, `dose_response`, `synergy`, `cross_modality`, `robustness`, `design`,
`multi_reporter`, `diagnose_abstention`, `differential`, `differential_robust`, `lotka`,
`fibrillization`, `constitutive`, `render_figure`, `identifiability`, `oed`, `list_models`,
and the async‑job pair `job_submit` / `job_status` (see
[Figures and long jobs](#figures-come-back-inline-long-jobs-run-async)).

**General model-analysis tools (`identifiability` / `oed`).** Beyond the perturbation-attribution
tools, the server exposes two *model-agnostic* tools that work on any differentiable ODE model in
NUDGE's registry (`list_models` — `glv`, `linear_pathway`, `ad_qsp`, `logistic`, plus canonical
sloppiness toys, across domains): `identifiability(model, …)` reports which parameters are
identifiable / sloppy / unrecoverable from the matrix-free Fisher spectrum (and **abstains** when
it can't certify — `NUDGE-LIM-023`), returning the FIM-spectrum figure inline; `oed(model, target,
…)` gradient-designs the measurement schedule that best resolves a confounded parameter, returning
the **measured** CRLB / eigenvalue lift (local OED, `NUDGE-LIM-024`) and the ellipse-collapse GIF
inline. Both are heavy → run them via `job_submit`.

**Analyse your OWN model file (`model_path` / `model_code`).** Beyond the registry names, both
tools accept a user model file directly: `identifiability(model_path="/opt/nudge/data/my_model.py")`
or inline `identifiability(model_code="…")` (precedence `model_code` > `model_path` > `model`; supply
exactly one). The file needs **no `nudge` import** — it defines `nudge_identifiability(n_free=0,
seed=0, sigma=None) -> {"predict_fn", "theta0", "param_names", "sigma"}` (with `predict_fn(theta)`
JAX-autodiff-differentiable in RAW positive params) and/or `nudge_oed(target=None, sigma=None,
seed=0) -> {"observe", "theta0", "param_names", "phi_bounds", "sigma"}`. NUDGE runs its real
matrix-free Fisher / gradient-OED analysis on whatever differentiable model the file returns. The
shipped `scripts/demo_ab/ad_qsp_model.py` / `ad_qsp_nlme_model.py` are worked examples (they mirror
the registered `ad_qsp` models to machine precision). **Security + host staging (`NUDGE-LIM-030`):**
`model_path` / `model_code` **execute arbitrary Python in the server process** — a local,
trusted-input convenience (like `python your_model.py`), NOT sandboxed and NOT safe for untrusted
callers. Because the connector's shared directory is mounted read-only and the home dir isn't
visible, stage the model file onto a server-visible path exactly like a data file —
`sudo mkdir -p /opt/nudge/data && sudo cp my_model.py /opt/nudge/data/` (or grant its directory via
`user_read_paths`) — and pass that **absolute** path. Arbitrary models are also always reachable via
the plain `import nudge` library path (`NUDGE-LIM-027`).

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

**File visibility caveat.** The sandbox can't see your home directory, so a CSV under `~`
won't be readable either. Put it on a **visible system path** — the same `/opt` tree works:

```bash
sudo mkdir -p /opt/nudge/data && sudo cp alpha_synuclein_ThT.csv /opt/nudge/data/
```

and reference `/opt/nudge/data/alpha_synuclein_ThT.csv` in your prompt. (Or grant its
directory via `[sandbox] user_read_paths`, or have Claude Science write the CSV into the
connector's own writable directory first, then call the tool on that path.) If a path fails,
that is almost always why — the smoke test above needs no file, so use it first to isolate
connection problems from file‑access ones.

### 2 · Ask the naive question

Phrase it as the *over‑ask* — the honest answer is the surprise:

> "I have a single ThT aggregation curve for **α‑synuclein** (mass fraction vs time) at
> `/opt/nudge/data/alpha_synuclein_ThT.csv`. Fit its microscopic aggregation rate constants —
> primary nucleation kₙ, elongation k₊, and secondary nucleation k₂ — and give me their values."

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
| `Command 'uvx' is not available…` | runtime has no `uvx`/`pipx` | use Option B (`/opt` install) or A (`python3 -m`) |
| `Connection closed … command not found inside the MCP sandbox` / `exec: …/.local/bin/nudge-mcp: not found` | the binary is under `~`, which the sandbox can't see | Option B — install into `/opt` (system path), **or** grant the home dir via `[sandbox] user_read_paths` |
| `No module named nudge` | the runtime's `python3` isn't your install target | Option B — point at the absolute `/opt/nudge/bin/nudge-mcp` path (shebang self‑selects Python) |
| Tools load, but `fibrillization` can't read the file | sandbox file visibility (home not visible) | put the CSV on a visible path (`/opt/nudge/data/…`), grant it via `user_read_paths`, or use the connector's writable directory |
| `render_figure` returns a *string* (an install hint), not an image | the `[viz]` extra (matplotlib) isn't installed, so the figure can't be drawn server‑side | `sudo /opt/nudge/bin/pip install --upgrade "nudge-bio[mcp,viz]"` and restart the connector |
| `render_figure` returns an `image_path` but the client can't open it | `NUDGE_ENV` isn't `cloud`, so the server wrote a file the sandbox client can't read | set `NUDGE_ENV=cloud` (via the `/usr/bin/env NUDGE_ENV=cloud …` command) so the image comes back as inline `image_base64` |
| A tool call fails with a timeout / "connection closed" after ~1 min | the call exceeded the connector's ~60 s cap (a fit / OED / the constitutive demo) | run it via `job_submit(tool, args_json)` + poll `job_status(job_id)` instead of calling it directly |
| `image_base64_omitted_reason` is set and you got a static PNG, not the GIF | the animation exceeded the ~1.5 MB inline cap | expected — it's the final‑frame preview; fetch the full GIF via the `code` regenerator, or re‑render with fewer `anim_frames` |
| First tool call is slow | JAX compile‑cache warm‑up on server start | expected once; subsequent calls are fast (the server is long‑lived) |

## The remote (hosted) alternative

Only the browser‑only `claude.ai` app cannot spawn a local process; it needs a **Remote**
connector (an HTTPS URL). The MCP SDK supports `transport="streamable-http"` over the same
`build_server()`, so a hosted deployment is a transport switch, not a rewrite. NUDGE ships the
`stdio` server today; hosting is left as a follow‑on (see
[`design/INTEGRATION_FEASIBILITY.md`](../../design/INTEGRATION_FEASIBILITY.md)).
