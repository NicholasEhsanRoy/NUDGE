# Installation

NUDGE targets Python ≥ 3.10 and pins JAX exactly (`jax==0.5.1`) to avoid the
version churn that has bitten the wider stack.

## With uv (recommended)

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"     # full dev environment (bio, mcp, ai extras + ruff/pyright)
```

Light lane (no heavy scanpy/pertpy stack), enough to run the fast suite:

```bash
uv pip install -e "." pytest pytest-xdist pyyaml ruff pyright
```

## Extras

| Extra | Adds | For |
|---|---|---|
| `bio` | scanpy, pertpy, scikit-misc | Tier 1/2 real-data loaders |
| `mcp` | mcp | the Claude Science MCP server (stretch) |
| `ai` | anthropic | the AI-in-the-loop dev/test harnesses |
| `ci` | pytest, pytest-xdist, pyyaml | headless CI |
| `dev` | all of the above + ruff, pyright, pre-commit, build | local development |

`maddening>=0.3.0` installs from PyPI.
