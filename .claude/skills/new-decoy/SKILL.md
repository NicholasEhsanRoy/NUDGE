---
name: new-decoy
description: Use when adding a case to the decoy battery — a synthetic negative engineered to look positive, where NUDGE must return the correct negative/abstention verdict. This is the CI that earns the "fails safely and loudly" claim.
---

# new-decoy

A decoy is a dataset a naive method would call a confident hit; the pass
condition is that NUDGE returns the correct *negative* verdict, loudly.

1. **Write the generator** — a `() -> AnnData` function producing the adversarial
   data (e.g. a linear circuit read through a saturating readout; a two-cell-type
   mixture faking bimodality; a dropout-driven zero peak). Human-authored, or
   AI-authored via `scripts/ai/generate_decoy.py` (creative-AI idea 1).
2. **Register a `DecoyCase`** in `src/nudge/data/decoys.py`: `decoy_id`
   (`NUDGE-DECOY-NNN`), `summary`, `generate`, `expected_verdict` (the correct
   `MechanismClass` abstention, e.g. `TECHNICAL_ARTIFACT` / `UNRESOLVED` /
   `NO_EFFECT` / `OFF_MODEL`), `limitation_ref`, and — for AI-authored decoys —
   `authored_by="ai"` + `prompt_ref`.
3. **Add the matching `NUDGE-LIM-` entry** in `docs/known_limitations.yaml`.
4. The parametrized battery test in `tests/decoys/` picks it up automatically.
   Confirm it passes: `pytest -m decoy`.

Each decoy has exactly one correct answer, so each is a green/red CI test.
