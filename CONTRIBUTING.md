# Contributing to NUDGE

## Development setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"      # or the light lane: uv pip install -e "." + pytest ruff pyright
```

## Checks (all must pass before commit)

```bash
uv run ruff check src tests scripts
uv run pyright src
uv run pytest                                   # CI-fast: excludes slow / validation / needs_llm
uv run python scripts/check_anomalies.py
uv run python scripts/check_citations.py
uv run python scripts/check_impl_mapping.py
```

## ID prefix convention

NUDGE inherits MADDENING's traceability machinery (`maddening.compliance`) and
uses its own ID prefixes:

| Kind | Prefix | Lives in |
|---|---|---|
| Mechanism | `NUDGE-MECH-` | `MechanismMeta.algorithm_id` on each mechanism |
| Verification benchmark | `NUDGE-VER-` | `@verification_benchmark` in `tests/verification/` |
| Known limitation | `NUDGE-LIM-` | `docs/known_limitations.yaml` |
| Decoy | `NUDGE-DECOY-` | `DecoyCase.decoy_id` in `src/nudge/data/decoys.py` |

## Adding a mechanism

Use the `/new-mechanism` skill (`.claude/skills/new-mechanism/`). Every mechanism
ships with: a `MechanismMeta`, a Mechanism Card (`docs/mechanism_cards/`, from
`_template.md`), a synthetic-recovery verification test, and at least one decoy it
must correctly resist.

## Commits — always credit Claude explicitly

This project is deliberately an experiment in Claude-assisted development. Every
commit that involved Claude appends a `Co-Authored-By: Claude <model> ...` trailer
and a real body saying what Claude did. Never fabricate co-authorship. See
`CLAUDE.md` and the `/commit` skill.

Commit-message subject prefixes: `feat` / `fix` / `refactor` / `docs` / `test` /
`perf` / `ci` / `chore`.
