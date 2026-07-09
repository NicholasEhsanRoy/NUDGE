---
name: demo-notebook
description: Use when building or updating a guided demo notebook in notebooks/*.ipynb — the visual, outputs-embedded walkthrough that shows one NUDGE capability working (the Demo judging criterion, 30%). Enforces the build-with-nbformat + execute-headless-to-embed-outputs pattern, the narrative→code cell rhythm, in-notebook ground-truth asserts, the loud honesty caveats (abstentions / one-sided bounds), and the notebooks/README.md index row.
---

# demo-notebook

A demo notebook is NUDGE's answer to the **Demo criterion (30% — the highest weight and
historically the weakest)**: a guided, **outputs-embedded** walkthrough that a judge can
read top-to-bottom without running anything and come away trusting that the capability
works. Each existing notebook tells one story: `OCT4_NANOG_Flagship` (dose-response),
`Norman_Synergy` (epistasis), `Chure_LacI_Benchmark` (cross-modality),
`Robustness_Dial` (bifurcation proximity). Copy their shape.

## The pattern (how these are actually built)

**Do NOT hand-edit `.ipynb` JSON.** Build the notebook from a small Python script with
`nbformat`, then execute it headless so the outputs (plots + printed verdicts) are
**embedded in the committed file**. That is what lets the notebook be read as a static
story.

1. **Write a build script** (in your scratchpad, not committed) that assembles cells:
   ```python
   import nbformat as nbf
   from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
   nb = new_notebook()
   cells = [new_markdown_cell(r"""# Title — the one-line thesis ..."""), ...]
   nb.cells = cells
   nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                 "name": "python3"},
                  "language_info": {"name": "python"}}
   nbf.write(nb, "notebooks/My_Demo.ipynb")
   ```
   Use `r"""..."""` raw strings for cells (they carry LaTeX / unicode / backslashes).
2. **Execute headless to embed outputs** (the load-bearing step):
   ```bash
   uv run jupyter nbconvert --to notebook --execute --inplace notebooks/My_Demo.ipynb
   ```
   Re-run this every time you change the build script. Verify outputs embedded:
   ```bash
   uv run python -c "import nbformat; nb=nbformat.read('notebooks/My_Demo.ipynb',as_version=4); \
     print('images:', any(o.get('output_type')=='display_data' for c in nb.cells \
     if c.cell_type=='code' for o in c.get('outputs',[])))"
   ```
3. **Add a row to `notebooks/README.md`** (Notebook | Dataset | What it shows), naming the
   `NUDGE-METHOD-*` / `NUDGE-LIM-*` ids and the headline result.

## Cell rhythm

Alternate **markdown (narrative) → code (produces an output)**. A good spine:

- **Title markdown** — the capability + the *one-line thesis*, in bold, up front. State
  the honesty caveat here too (it is a feature, not fine print).
- **Imports code** — `import warnings; warnings.filterwarnings("ignore")` first (JAX/CUDA
  warnings are noise in a demo), then numpy / matplotlib / the `nudge.*` entry points.
- **Section pairs** — each a markdown cell that says what the next cell shows, then a code
  cell that prints a table / verdict and draws a plot. Keep plots self-contained
  (matplotlib inline; no external network — the CSP-style rule: everything embedded).
- **Fail-safe / honesty cell** — show the abstention, the one-sided bound, the
  `unresolved` verdict *by printing the reason strings*, not just numbers. This is where a
  NUDGE demo earns trust.
- **Closing markdown** — "what this buys / where it goes next", and how to reproduce it
  (`nudge <verb>` CLI + the MCP tool).

## Prove the story in the notebook (honesty is load-bearing)

The demo must not *assert in prose* something the code doesn't show. Put a real `assert`
in a code cell for the ground-truth claim the narrative makes (e.g. "all three channels
move monotonically toward the fold" → assert monotonicity on the clean rungs). If it
can't pass, the narrative is wrong — fix the narrative, never loosen the honesty. Mirror
the numbers to the capability's `tests/` and `scripts/vv/FINDINGS.md` entry so the
notebook, the tests, and the findings agree.

## Data sources

- **Prefer self-contained / synthetic** where the point is a mechanism, not a dataset (the
  `Robustness_Dial` sweep is fully synthetic — it executes anywhere, including CI-free
  reviewers' machines).
- **Real-data notebooks** read from the Seagate drive with an **env-var override + an
  early `assert path.exists()`** (see `Chure_LacI_Benchmark`), so a reader can point it at
  their copy and the missing-data failure is loud, not silent.

## Gotchas

- **Numerical noise at extremes.** If a finder/fit gets noisy at a boundary (e.g. an
  eigenvalue → 0 right at a fold), assert monotonicity on the clean rungs just short of it,
  and say so in a comment — don't overclaim a monotonicity the numerics don't deliver.
- **Unicode is fine in cells** (they are strings, not source ruff-checks), but the **build
  script** is ruff-checked — keep its lines ≤ 88 (a `→`/`λ` is one column).
- **Keep the committed `.ipynb` small-ish.** One or two figures per notebook; embedded PNGs
  are large. Prefer a couple of decisive plots over many.
- The `[dev]` extra provides `jupyter` / `matplotlib` / `nbconvert`
  (`uv pip install -e ".[dev]"`).
