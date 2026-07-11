# Automated-scientist blind eval (`scripts/eval/`)

A blind, un-memorizable test of whether an AI scientist reaches a **calibrated** mechanistic
answer with NUDGE — and the material for the demo ablation. Design: `design/AUTOMATED_SCIENTIST.md`.

**Governing rule: honesty.** Grade **calibration**, not point-accuracy — a correct call *or* an
honest abstention is a PASS; a **confident-wrong is the only hard fail**. The answer is labeled +
preserved for us but **completely inaccessible to the test agent** (held-out keys dir, gitignored).

## The three tools

| script | role |
|---|---|
| `blind_harness.py` | Generates a **blinded** case from NUDGE's synthetic generators — a registry keyed by *surface* (capability). Writes scrubbed data to an agent dir + a **held-out** answer key. |
| `run_blind_eval.py` | Sets up an isolated sandbox (scrubbed data + task prompt + web denied + nudge-mcp wired or withheld) and runs headless `claude`. **Dry-run by default** — pass `--run` on a stable connection. Supports the `--mode with-nudge` / `--mode without-nudge` **ablation**. |
| `grade_blind_eval.py` | Reads the agent's `REPORT.md` + the held-out key and **proposes** a calibration grade for the human in the loop to confirm. |

## Surfaces (add one = one `@register` builder)

`attribute` (gene-circuit snapshot → honest abstain), `dose-response` (the resolving flagship:
`switch` / `graded` / `truncated`→abstain), `lotka` (gLV / microbiome trajectories). Pending:
`fibrillization` (a `nudge-jax-physics` build). `python blind_harness.py --list`.

## Workflow (generate now; RUN later when the connection is stable)

```bash
# 1) Generate blind cases (data + held-out keys). Repeat per surface/case.
uv run python scripts/eval/blind_harness.py --surface dose-response --case switch \
    --seed 0 --agent-dir eval_cases/dose_switch --keys-dir eval_keys
uv run python scripts/eval/blind_harness.py --surface dose-response --case truncated \
    --seed 0 --agent-dir eval_cases/dose_truncated --keys-dir eval_keys

# 2) Set up the run sandbox (DRY RUN — builds the sandbox + prints the command, runs nothing):
uv run python scripts/eval/run_blind_eval.py --agent-dir eval_cases/dose_switch \
    --surface dose-response --mode with-nudge
#    ... and the control arm for the ablation:
uv run python scripts/eval/run_blind_eval.py --agent-dir eval_cases/dose_switch \
    --surface dose-response --mode without-nudge

# 3) WHEN ON A STABLE CONNECTION — actually run it (add --run):
uv run python scripts/eval/run_blind_eval.py --agent-dir eval_cases/dose_switch \
    --surface dose-response --mode with-nudge --run

# 4) Grade (proposes a grade; human confirms):
uv run python scripts/eval/grade_blind_eval.py \
    --report eval_cases/dose_switch/_run_with-nudge/REPORT.md \
    --key eval_keys/dose_switch_answer_key.json
```

## Integrity rules (make-or-break for a blind test)

- The agent's sandbox gets **only** the scrubbed data + the task prompt — never the generator,
  the seed, or the answer key.
- `eval_cases/`, `eval_keys/`, `blind_test.h5ad`, and `*answer_key*.json` are **gitignored** — the
  answer never enters the repo, so an agent with repo access still cannot find it.
- Web is **denied** in both ablation arms (the test is the data + NUDGE, not the model's memory).
- The cover story tells the agent it is a **synthetic/fictional** test — not a lie, and the exact
  signal to use the tools, not memory. It must never hint at the mechanism.

## Prerequisites for `--run`

- `claude` CLI on PATH; `uv run nudge-mcp` working in the repo (the with-nudge arm wires it via a
  sandbox `.mcp.json`). Web tools are denied via a sandbox `.claude/settings.json`.
- Confirm your Claude Code version supports `--print --output-format json` + `--allowedTools`.
