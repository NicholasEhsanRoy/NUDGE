# NUDGE visualization module — design & recommendation

**Status:** LARGELY BUILT on `feat/viz-renderers` (Author: Claude Opus 4.8). Built so far:
the shared spine (`__init__`/`theme`/`base`/`provenance`), a **collision-aware placement
layer** (`layout.py` — banner/K-label/legend never land on data or each other), the flagship
`dose_response`, and a renderer per existing result type — `epistasis`, `differential`,
`multi_reporter`, `temporal`/gLV, `aggregation`/fibrillization, `constitutive`, `diagnose`,
`design`, `oed`, `cross_modality` (reuses the dose-response Hill panel), and `robustness`
(the 0–1 dial). The **animation engine** (`animate.py`) is wired (`render(..., animate=True)`)
with the constitutive-flip GIF. Every renderer inherits the automatic abstention overlay and
ships the provenance sidecar; each has a render + overlay-fires test in `tests/viz/`. Not yet
built: `attribution` (the `AttributionReport` chips) and `identifiability` (the cells×noise /
Fisher spectrum), and the CLI `--fig-out`-on-every-verb / `nudge viz` verb / MCP tool surface
(§3) beyond the existing `render_result` seam. Original design pass below.
**Scope:** a generalized, reusable, provenance-carrying figure layer for NUDGE's
result types — additive/opt-in, never touching `fit.py` or `core`. Targets the
**Demo (30%)** judging criterion, currently the weakest.

> One-line thesis: NUDGE already produces honest *dataclasses* and honest *text*. It
> does not yet produce honest *pictures* from one consistent surface. This module makes
> the pictures — and makes an abstention look like an abstention, on purpose.

---

## 1. The seam — what plotting exists today, and what's reusable

### 1.1 Where plotting lives now (all ad-hoc matplotlib)

| Surface | Plotting today | Reusable as… |
|---|---|---|
| `notebooks/OCT4_NANOG_Flagship.ipynb` | `plot_fit(ax, label, res, color)` (Hill curve + scatter + K/maxdose guides), an `n_profile` R²-vs-n plot (OCT4 peaked / NANOG flat), an exploratory-vs-honest overlay | The canonical **dose-response** renderer + the **identifiability n-profile** panel |
| `notebooks/Robustness_Dial.ipynb` | proximity gauge + the three channels vs a swept n/K | The **bifurcation "robustness dial"** renderer + the **swept-parameter** animation source |
| `notebooks/Norman_Synergy.ipynb` | additive-null bar + interaction CI + off-axis residual | The **epistasis** renderer |
| `notebooks/Multi_Reporter.ipynb` | per-reporter curves + the shared-latent restricted-loss bars | The **multi-reporter** renderer |
| `notebooks/Inverse_Design.ipynb` | flip-ON trajectory + safety before/after dial | The **design()** renderer + a **trajectory animation** |
| `notebooks/Constitutive_Control.ipynb` | WITHOUT-vs-WITH-control n-profile (the LIM-006 flip) | The **constitutive-flip** renderer + its **animation** |
| `notebooks/Chure_LacI_Benchmark.ipynb`, `Differential.ipynb`, `Hidden_Node_Abstention.ipynb` | per-variant knob tables, BIC bars, differential cards | **cross-modality / differential / diagnose** renderers |
| `scripts/vv/overnight_sweep.py` → `scripts/vv/results/*.png` | `calibration.png`, `identifiability_cells_noise.png`, `identifiability_effect_cells.png` | The **calibration** + **identifiability heatmap / sloppiness** renderers |
| `scripts/vv/fisher_sloppiness.py`, `fisher_extrinsic.py` | Fisher-information eigen-spectrum / the gain⇄threshold degeneracy | The **sloppiness / covariance-mode** renderer |

Every one of these is a *pattern* that recurs (scatter + fit + K guide + verdict title;
bar + CI + null line; profile + argmin), re-implemented per notebook with copy-pasted
helpers. That duplication is the reuse target.

### 1.2 The clean architectural seam already exists

`service.py` already contains a **`*_to_dict()` serializer for every result type**
(`dose_response_to_dict`, `bifurcation_to_dict`, `synergy_to_dict`,
`multi_reporter_to_dict`, `differential_to_dict`, `constitutive_to_dict`,
`design_to_dict`, `variant_attribution_to_dict`, `inadequacy_to_dict`,
`report_to_dict`). These are the **CLI↔MCP boundary** and are already JSON-safe. The viz
module plugs into *exactly this seam*: it renders from the frozen dataclasses when it has
them, and from the serialized dicts when it doesn't (the MCP/preserved-code path). No new
serialization contract is invented; the existing one is reused.

### 1.3 Confirmation: purely additive

- **Nothing in `fit.py` / `core` / any `inference/*` module changes.** The viz module
  only *reads* result dataclasses and dicts; it never re-attributes and never imports the
  fit engine.
- matplotlib is currently a **`dev`-only** dependency (`pyproject.toml`, "V&V sweep
  figures … analysis-only"). The module lives behind a new **`[viz]` extra** so the core
  install stays matplotlib-free. Every matplotlib import is **lazy** (inside functions),
  matching how the codebase already defers `anndata` / `scanpy` / `mcp`.
- The CLI/MCP additions are new **optional flags** and one new **optional tool** — no
  existing verb's default behavior or text output changes.

---

## 2. Proposed architecture

### 2.1 Module layout — `src/nudge/viz/` (new, opt-in)

```
src/nudge/viz/
  __init__.py         # public API: render() dispatcher + re-exports; friendly ImportError if [viz] missing
  theme.py            # NUDGE house style: mechanism colors, light/dark, headless Agg default, fonts
  base.py             # FigureResult dataclass; save(); the ABSTENTION OVERLAY (load-bearing); axis helpers
  provenance.py       # figure-code preservation: emit standalone .py + data sidecar (Claude Science grain)
  dose_response.py    # plot_dose_response(DoseResponseResult|dict)  ← flagship OCT4/NANOG
  attribution.py      # plot_attribution(AttributionReport|dict): single/multi calls + skips + abstentions
  bifurcation.py      # plot_robustness(BifurcationResult|dict): the 0..1 dial + 3 channels + one-sided band
  epistasis.py        # plot_epistasis(EpistasisResult|dict): additive null + interaction CI + off-axis residual
  multi_reporter.py   # plot_multi_reporter(MultiReporterResult|dict): panel curves + restricted-loss bars
  differential.py     # plot_differential(DifferentialResult|dict): per-model BIC + winning-knob Δ + confound flags
  constitutive.py     # plot_constitutive(ConstitutiveResult|dict): WITHOUT vs WITH control n-profile (LIM-006)
  design.py           # plot_design(InterventionPlan|AbstentionResult|dict): flip trajectory + safety dial
  identifiability.py  # plot_identifiability(...): the cells×noise heatmaps + Fisher sloppiness spectrum
  animate.py          # animation engine: FuncAnimation → PillowWriter (GIF); sweep + optimizer-trajectory drivers
```

Rationale for one file per result type: each mirrors an existing `inference/*` result and
its notebook, so a contributor edits one obvious place, and the `[viz]` surface maps 1:1
to the analysis surface.

### 2.2 The API — keyed off the existing result dataclasses

**Primary entry (dispatcher):**

```python
nudge.viz.render(result, out=None, *, emit_code=True, theme="auto",
                 self_contained=False, animate=False) -> FigureResult
```

`render` **dispatches on type** and always applies the abstention overlay itself (see
§2.5) — honesty is not caller-optional. It accepts either the **frozen dataclass** or its
**`to_dict()` dict** (dual-input; the dict path is what the preserved `.py` and the MCP
tool use). Dispatch table (the named types this module targets):

| Result dataclass (module) | Renderer | Figure |
|---|---|---|
| `DoseResponseResult` / `DoseResponseFit` (`inference.dose_response`) | `plot_dose_response` | Hill fit + points + K guide; **abstention band** when `not is_reliable` / `not spans_inflection` |
| `AttributionReport` (`inference.pipeline`) | `plot_attribution` | per-op call chips, restricted-NLL bars, **skips/abstentions rendered as first-class cells** |
| `BifurcationResult` / `BifurcationScore` (`inference.bifurcation`) | `plot_robustness` | 0..1 proximity dial + 3 channels; **one-sided lower-bound arrow** near the fold |
| `EpistasisResult` / `EpistasisFit` / `ComboGeometry` (`inference.epistasis`) | `plot_epistasis` | additive-null vs observed A+B, interaction CI, off-axis residual (labelled *not a hidden-node claim*) |
| `MultiReporterResult` / `MultiReporterFit` (`inference.multi_reporter`) | `plot_multi_reporter` | per-reporter curves + restricted-loss bars (threshold/gain/ceiling); off-model overlay |
| `DifferentialResult` / `DifferentialFit` (`inference.differential`) | `plot_differential` | ΔBIC per model + winning-knob Δ CI + per-context depth/confound flags |
| `ConstitutiveResult` (`inference.constitutive`) | `plot_constitutive` | WITHOUT (flat) vs WITH (n=1 rejected) n-profile — **the LIM-006 flip** |
| `InterventionPlan` / `SafetyReport` / `AbstentionResult` (`design.invert`) | `plot_design` | flip-ON trajectory (`predicted_trajectory`) + safety before→after dial; abstention card |
| `VariantAttribution` panel (`inference.cross_modality`) | `plot_cross_modality` | per-variant knob localization strip (Chure LacI) |
| `InadequacyReport` (`inference.hidden_node`) | `plot_diagnose` | ranked differential-diagnosis card (never a positive hidden-node claim) |
| identifiability sweep / Fisher (`scripts/vv`, `identifiability`) | `plot_identifiability` | cells×noise heatmap + sloppiness eigen-spectrum |

Each `plot_*` also has the signature `(...) -> FigureResult` and accepts an optional
`ax=`/`fig=` so notebooks can compose panels (the current `plot_fit(ax, …)` idiom is
preserved, just centralized).

**`FigureResult` (in `base.py`):**

```python
@dataclass(frozen=True)
class FigureResult:
    path: str | None            # PNG (or GIF) written, if out= given
    code_path: str | None       # the standalone regenerating .py (Claude Science grain)
    data_path: str | None       # the input-data sidecar (.json/.npz), unless self_contained
    png_base64: str | None      # inline image for MCP (size-capped; None for large/animation)
    caption: str                # the honest one-line caption (verdict + any abstention)
    abstained: bool             # did the overlay fire? (verdict is non-positive / one-sided)
    kind: str                   # e.g. "dose_response"
```

### 2.3 Rendering backend

- **matplotlib (default and only hard dep).** Everything static is matplotlib with the
  **`Agg`** backend forced in `theme.py` (headless-safe; no `$DISPLAY` needed).
- **Animation = `matplotlib.animation.FuncAnimation` → `PillowWriter` (GIF).** Pillow is
  already a transitive matplotlib dependency, so **animated GIFs need no new dep.** We do
  **not** add `ffmpeg`/mp4 (a heavy system dep) — GIF embeds in notebooks, Artifacts, and
  chat, and is the right medium for a 2–5 s loop. This is the only "needs more than a
  static plot" case, and it is satisfied within the existing dep closure.
- No plotly/d3/bokeh — they would fracture the "one consistent surface" goal and add
  weight for no Demo gain that a GIF doesn't already give.

### 2.4 Theming

`theme.py` defines a **NUDGE house style** so every figure reads as one system:

- **Mechanism palette** (consistent everywhere): threshold `K`, gain `n`, ceiling
  `v_max`, no-effect, and a dedicated **abstain** color (muted grey + hatch) that is
  *visually distinct from any positive call*. This is a semantic palette, not decoration:
  the same color always means the same mechanism across dose-response, attribution,
  multi-reporter, and differential.
- **Light/dark aware** (`theme="auto"|"light"|"dark"`) so figures look right embedded in a
  notebook, an Artifact, or a dark chat client.
- One `apply_theme()` called at the top of every renderer; rcParams centralized.

*(Where the `dataviz` skill is available, load it before finalizing the palette so the
color choices pass its contrast/accessibility validator — the mechanism palette must be
colorblind-safe because it is load-bearing semantics, not styling.)*

### 2.5 Headless + abstention-honest behavior (the load-bearing part)

This is a visualization of a **fail-safe** tool; the viz must not undermine the thesis. So
honesty is enforced structurally, not left to each renderer:

- **`base.abstain_overlay(ax, verdict, reason, one_sided=False)`** is applied **by
  `render()` itself**, driven off the result's own `call`/`verdict` field. A renderer
  cannot forget it. It:
  - greys + **hatches** the plot region and stamps a large `I CAN'T TELL — <verdict>`
    banner for the abstention classes (`unresolved`, `no-effect`, `off-model`,
    `technical-artifact`, `no-difference`, `not-bistable`);
  - renders a **one-sided lower bound** (bifurcation/safety near the fold, `spans_inflection
    = False` in dose-response) as an **arrow + open-ended band**, never a point estimate or
    a closed error bar — the "at least this close / unidentifiable past here" grammar;
  - always writes the honest `caption` (verdict + the abstention reason) so a figure lifted
    out of context still carries its caveat.
- **A positive call is never drawn where the dataclass abstained.** Because the overlay is
  keyed off the same field the CLI/MCP print, the picture and the text can never disagree.
- Headless: `Agg` + `savefig`; `render()` never calls `plt.show()` (the notebook does).
  Animations render frame-by-frame under `Agg` with no display.

---

## 3. CLI + MCP + service integration

### 3.1 Service layer (the shared seam)

Add **one** thin service function so CLI and MCP share the figure path exactly as they
share attribution today:

```python
# service.py (additive)
def render_result(kind: str, result_or_dict, *, out: str | None,
                  emit_code: bool = True, theme: str = "auto",
                  self_contained: bool = False, animate: bool = False) -> dict:
    """Import nudge.viz lazily, dispatch, return FigureResult as a dict."""
```

It lazy-imports `nudge.viz`, raising the friendly `[viz]`-extra install message if absent.
This keeps `viz` optional and keeps CLI/MCP byte-identical, matching the module's stated
purpose ("the one place the CLI and MCP server share").

### 3.2 CLI — an opt-in flag on the existing verbs (preferred) + a `nudge viz` verb

**Per-verb flag (additive, no behavior change when unset):** add to `attribute`,
`dose-response`, `synergy`, `cross-modality`, `robustness`, `design`, `multi-reporter`,
`differential`, `constitutive`, `diagnose-abstention`:

```
--fig-out PATH          # write a PNG (or .gif) of this result; text output unchanged
--fig-code / --no-fig-code   (default: --fig-code)   # also emit the regenerating .py
--fig-theme auto|light|dark
--animate               # where meaningful (robustness sweep, design trajectory, constitutive flip)
```

Flow: the verb runs exactly as today, prints its honest text report, then — **only if
`--fig-out` is set** — calls `service.render_result(...)` and prints the written paths
(`wrote fig.png + fig.py`). The result object it already has in hand is passed straight in.

**Standalone verb** for re-plotting a saved run (decouples compute from figure):

```
nudge viz RESULT.json --kind dose-response --out fig.png [--animate] [--no-fig-code]
```

`RESULT.json` is any prior `*_to_dict` output (the CLI can gain a `--json-out` on each verb
to save it, or the MCP dict can be dropped to disk). This makes every past run
re-visualizable without re-fitting — useful for a live demo and for Claude.

### 3.3 MCP — one `render_figure` tool + return-type design

Add a single tool (keeps the tool list legible; the per-result plotters are reachable
through its `kind` arg):

```python
@mcp.tool()
def render_figure(kind: str,
                  result_json: str = "",         # a prior *_to_dict() JSON string, OR
                  out_dir: str = "",             # where to write (server-side path)
                  self_contained: bool = False,
                  animate: bool = False,
                  theme: str = "auto") -> dict:
    """Render a NUDGE result to a figure. Returns paths + (size-capped) inline PNG +
    the regenerating code path. Abstentions are rendered as abstentions."""
```

**What it returns (both a path and an inline image):**

```json
{
  "png_path": "/abs/out/fig.png",
  "code_path": "/abs/out/fig.py",
  "data_path": "/abs/out/fig.data.json",
  "png_base64": "iVBORw0KGgo…",   // present only if under the size cap
  "png_base64_omitted_reason": null,  // else: "exceeds 1.5MB inline cap; read png_path"
  "caption": "OCT4 → SWITCH (n≈6.7, R²=0.99); NANOG → ABSTAIN (gain unidentifiable past max dose)",
  "abstained": true,
  "kind": "dose_response"
}
```

Design decisions:
- **Return both path and base64** because the MCP client and server may or may not share a
  filesystem. Path is authoritative; base64 lets a chat/Artifacts client show the image
  inline immediately.
- **Size cap the inline PNG (~1.5 MB base64).** Over the cap → return the path only with an
  explicit `png_base64_omitted_reason` (never silently drop it). **Animations (GIF) are
  always path-only** (or a single-frame PNG preview inlined) — GIFs blow the payload.
- Consistent with the existing layering: `render_figure` calls `service.render_result`,
  same as every other tool calls its `*_file` service function. No modelling logic in the
  tool.

An alternative — a `plot: bool` + `fig_out` param on *every* existing tool — was
considered and **rejected**: it bloats ten tool signatures and duplicates the base64/size
logic ten times. One `render_figure` that consumes any `*_to_dict` JSON is cleaner and
means a Claude workflow is "run `dose_response` → feed its JSON to `render_figure`."

---

## 4. Figure-code preservation (the Claude Science grain — a hard requirement)

**Requirement:** every figure NUDGE emits must carry (or be regenerable from) its exact
generating code, so a scientist can reproduce or modify it. Claude Science / Artifacts
treat a figure and its source as inseparable; this module matches that grain.

### 4.1 Mechanism — emit a standalone, runnable `.py` + a data sidecar

For `out="fig.png"`, `render(emit_code=True)` writes **three** files:

1. **`fig.png`** — the image.
2. **`fig.data.json`** (or `fig.data.npz` for array-heavy results like `design`'s
   trajectory / identifiability grids) — the input the figure was drawn from, produced by
   the **existing `*_to_dict()` serializer**. No new format.
3. **`fig.py`** — a standalone script, byte-stable, with a provenance header:

```python
# Generated by NUDGE viz — regenerates fig.png exactly.
# nudge <version> · git <sha|unknown> · <UTC timestamp>
# call: nudge dose-response ESC.h5ad --target OCT4 … --fig-out fig.png
import json, nudge.viz as viz
data = json.load(open("fig.data.json"))
viz.render(data, out="fig.png", kind="dose_response", emit_code=False)
```

**Why this is exact and cheap:** the renderers accept the **dict form** (dual-input,
§2.2), so the preserved script replays *the same plotting code over the same numbers* — it
does **not** re-fit (no JAX, no data download, deterministic, fast). The figure is
reproduced from the fit's *output*, which is what a reviewer wants to trust and tweak.

### 4.2 Optional single-file mode (`self_contained=True`)

For Artifacts / chat where sidecar files are awkward, inline the data as a base64 blob
inside `fig.py` (a `_DATA = "…"; data = json.loads(base64.b64decode(_DATA))` preamble), so
**one file** regenerates the figure. Costs file size; buys portability. This is the mode to
hand to a Claude Artifact.

### 4.3 Integration with the three surfaces

- **Notebooks** are *already* a preserved-code surface — the code cell IS the source. Here
  the module simply *reduces* the preserved code to one honest call
  (`viz.plot_dose_response(res, ax=ax)`) instead of 30 lines of copy-pasted matplotlib, so
  the notebook stays the reproducible artifact it already is (per the `demo-notebook`
  skill), just cleaner. `emit_code` defaults off in-notebook.
- **CLI** `--fig-code` writes the `.py`+sidecar next to the PNG.
- **MCP** `render_figure` returns `code_path` (and, with `self_contained`, a one-file
  script) so Claude can hand the scientist the regenerating code alongside the picture —
  which is precisely the Claude Science contract.

---

## 5. Prioritized build plan

### 5.1 Build FIRST — the flagship dose-response dual panel (`plot_dose_response`)

**One figure: OCT4 → SWITCH beside NANOG → HONEST ABSTAIN**, from the real ESC screen,
each as a Hill fit + guide-dose scatter + K marker, with NANOG's panel carrying the
**first-class abstention rendering** ("K past max dose → gain unidentifiable", hatched
band, not a fake curve).

Why this one, tied to **Demo (30%)**:
- It is the project's **first positive real-data call** *and* an honest abstention **in a
  single frame** — the entire fail-safe thesis ("attribute when you can, abstain loudly
  when you can't") made visual and trustworthy. Judges see the honesty claim *demonstrated*,
  not asserted.
- **Lowest risk / highest reuse:** the data, the fit, and 90% of the plotting already
  exist in `OCT4_NANOG_Flagship.ipynb`; this lifts them into `plot_dose_response` and adds
  the overlay. It becomes the keystone reusable primitive (cross-modality and design curve
  mode reuse the same Hill renderer).
- It exercises the **whole spine** end-to-end (dataclass → renderer → overlay → provenance
  `.py` → `FigureResult`), de-risking every later renderer.

### 5.2 Then — the "genuinely cool to watch" animation (highest Demo delta after #1)

`animate.py` + **the constitutive LIM-006 flip animation**: animate the circuit-`n`
profile going from **FLAT (WITHOUT control — "you can't even tell a switch exists")** to
**the n=1 point getting REJECTED (WITH control)** as the constitutive control is switched
on. This is NUDGE's sharpest fail-safe story (turning a documented *confident false
positive* into a correct biological call) and it is *dynamic* — it earns the "cool to
watch" half of Demo that a static panel can't. Data is already produced by
`constitutive_demo`/`profile_circuit_n`; the animation is a `FuncAnimation` over the two
loss profiles → GIF. Its sibling driver (a **parameter sweep of the robustness dial
crossing the fold**, and the **design() flip-ON trajectory** from `predicted_trajectory`)
reuse the same `animate.py` engine.

### 5.3 Then — the remaining renderers, in Demo-leverage order

1. `plot_robustness` (the dial is inherently demo-friendly; the one-sided-bound grammar).
2. `plot_attribution` (the core `AttributionReport`; skips/abstentions as first-class cells).
3. `plot_epistasis` (Norman; additive null + off-axis-residual honesty label).
4. `plot_multi_reporter` (the "single reporter abstains → joint panel resolves" contrast).
5. `plot_constitutive` static + `plot_design` static + `plot_differential` +
   `plot_cross_modality` + `plot_identifiability` + `plot_diagnose`.

### 5.4 Then — the integration surface

CLI `--fig-out/--fig-code/--animate` on every verb + the `nudge viz` verb; the MCP
`render_figure` tool; a headless smoke test per renderer (assert a PNG is written and the
abstention overlay fires on a known abstention) + a 3-frame GIF smoke test.

### 5.5 Honest effort estimate (calibrated to this team's velocity)

This team repeatedly ships in ~8 h what looked like a week; but a figure battery is
breadth, not one hard problem, so it parallelizes rather than compresses:

- **Session 1 (~6–8 h): the demo-visible core.** Scaffold (`__init__`/`theme`/`base` with
  the overlay/`provenance`) + `plot_dose_response` (§5.1) + `animate.py` and the LIM-006
  flip GIF (§5.2) + CLI `--fig-out` on `dose-response` + `constitutive` and the MCP
  `render_figure` tool. **Outcome: the single highest-leverage figure and the single
  coolest animation are live end-to-end, with preserved code.** This is the Demo lever.
- **Session 2 (~6–8 h): breadth.** The remaining ~9 renderers (each ~30–60 min, lifted
  from its notebook) + `--fig-out` across all verbs + `nudge viz` verb + the smoke tests +
  refresh the notebooks to call the module.

So: **~1 session to a materially stronger Demo, ~2 to a complete, tested viz surface.** Not
a week; not an afternoon either — the overlay/provenance/animation plumbing is real work
and is where the honesty guarantees live.

---

## 6. Honest risks & scope limits

- **Animation in headless CI.** `FuncAnimation` + `Agg` + `PillowWriter` is headless-safe
  but slow and occasionally flaky under xdist. Mitigation: keep animation *smoke* tests to
  ≤3 frames and mark the full renders `slow` (scheduled lane, not PR CI) — consistent with
  the existing test-lane policy. No mp4/ffmpeg (avoid the system dep); GIF only.
- **MCP image payload size.** A base64 PNG can be large and a GIF far larger than any sane
  tool-result. Mitigation: hard size cap on inline PNG with an explicit
  `png_base64_omitted_reason`; GIFs are path-only (or a one-frame PNG preview). Never
  silently truncate.
- **Keeping it opt-in / not bloating core.** matplotlib moves to a `[viz]` extra (core
  stays matplotlib-free); every import is lazy; `import nudge.viz` without the extra raises
  a friendly install message; `core`/`fit.py` never import `viz`. A CI guard can assert
  `nudge` imports with no matplotlib installed.
- **The honesty risk (the important one).** A prettier picture is a *stronger* vehicle for
  a false-confident claim — the exact failure this project exists to avoid. Mitigations are
  structural, not stylistic: the abstention overlay is applied by `render()` off the
  dataclass's own verdict (a renderer cannot omit it); one-sided bounds are drawn as
  open-ended arrows, never closed error bars or point estimates; the caption always carries
  the verdict + reason; and a test asserts that feeding a known-abstention result produces
  `FigureResult.abstained == True` with the overlay present. **The picture can never claim
  more than the dataclass.**
- **Scope limit.** This module *visualizes* existing results; it performs **no** fitting,
  model selection, or attribution. If a result type gains a field, its renderer updates —
  but the module never becomes a second, divergent analysis path. That boundary is what
  keeps it safely additive.

---

## 7. Summary

A new, opt-in `src/nudge/viz/` renders every NUDGE result type from **one `render()`
surface keyed off the existing frozen dataclasses** (and their `*_to_dict()` dicts),
reusing the plotting patterns already scattered across the notebooks and `scripts/vv/`.
matplotlib (already present, moved to a `[viz]` extra) is the only backend; animated GIFs
come free via its bundled Pillow writer. CLI verbs gain an additive `--fig-out/--fig-code`
flag plus a `nudge viz` verb; the MCP server gains one `render_figure` tool returning
path + size-capped base64 + the regenerating code path. **Every figure ships a standalone
`.py` + a data sidecar that replays the exact plot from the fit's output** (no re-fit) —
the Claude Science provenance grain. Abstentions are rendered *as* abstentions by
construction, so the visuals reinforce rather than undermine the fail-safe thesis. Build
the **OCT4/NANOG dose-response dual panel first** (real-data positive + honest abstain in
one frame — the thesis, visualized, at lowest risk), then the **LIM-006 constitutive-flip
animation** (the fail-safe story, made cool to watch). Estimated ~1 session to a
materially stronger Demo, ~2 to the full tested surface. Top risk: a polished figure is a
stronger vehicle for a false claim — mitigated by making the abstention overlay
non-optional and keyed off the same verdict the text prints.
