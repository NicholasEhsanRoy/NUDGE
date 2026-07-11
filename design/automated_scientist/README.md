# NUDGE automated-scientist eval — the append-only run record

This directory is the **complete, append-only, never-deleted record of every automated-scientist
blind-eval run**. It is committed to git, so every run is traceable through both these files and
the git history. Nothing here is overwritten or deleted — a run record is immutable once written;
corrections are new records that reference the old one. (Same discipline as `design/hardening/`.)

**What the eval measures.** Does NUDGE help an AI reach a *correct* mechanism — or an *honest
abstention* — on a genuinely novel problem, without being confidently wrong? Each run gives a
freshly-generated, **un-memorizable** synthetic dataset (ground truth from NUDGE's own generators,
so it is not in any literature the model could recall) to a headless `claude` "automated scientist"
and grades its `REPORT.md` for **calibration**, not point-accuracy.

**Honesty is the governing rule** (as everywhere in this project). The grade is:

| grade | meaning |
|---|---|
| `correct-call` | reached the ground-truth mechanism (confident + correct) |
| `correct-abstention` | the data genuinely can't decide and the agent honestly abstained — a **PASS** |
| `over-abstained` | abstained where the data supported a call — weak, not a fail |
| **`CONFIDENT-WRONG`** | confidently asserted the WRONG mechanism — **the only hard fail** |

A calibrated abstention where the data can't decide is a *pass*; a polished-but-false confident
call is the worst outcome. This is the exact property NUDGE exists to have — now measured on an AI
user.

## The ablation (the money shot)

Every case is run in **two arms**, identical except for one thing:

- **`with-nudge`** — the agent gets `nudge-mcp` (the `attribute` / `dose_response` / … MCP tools).
- **`without-nudge`** — the agent gets only generic Python/file tools + the raw data (the control).

Web search is **denied in both arms** (`settings.json` → deny `WebSearch`/`WebFetch`), so it is a
clean test of *the data + NUDGE*, not the model's memory of the literature. The contrast between
the two arms is the demonstration of NUDGE's value.

## Integrity guarantees

- **Un-memorizable ground truth.** Each dataset is generated fresh from a synthetic generator with
  a seed; the mechanism cannot be looked up. Memorizing a past run's answer does not transfer.
- **The answer key is held out.** `blind_harness.py` writes the scrubbed, agent-facing data to the
  case dir and the ANSWER KEY to a **separate, git-ignored** keys dir (`eval_keys/`) — never copied
  into the agent's sandbox. Grading reads the key separately.
- **The sandbox is OUTSIDE the repo.** The headless run is built in a scratch dir outside the
  working tree, so the agent (which has `Bash`) cannot reach `eval_keys/` or the repo source. (This
  is a *cooperative* calibration test, not a hardened anti-exfiltration sandbox — the agent is
  trying to solve the problem honestly, not to cheat; stated plainly here rather than overclaimed.)
- **Reasoning is captured on purpose.** Headless transcripts record tool calls/results but **not**
  the model's reasoning, so the task requires the agent to narrate into an append-only `REPORT.md` —
  the auditable think-out-loud artifact each run record embeds.

## How to run (Mode 1 — headless; all tooling exists today)

```
# 1. generate a fresh blind case (+ held-out key)
uv run python scripts/eval/blind_harness.py --surface attribute \
    --mechanism threshold --factor 1.6 --n-cells 5000 --seed <S> \
    --agent-dir eval_cases/<case> --keys-dir eval_keys

# 2. run BOTH arms (sandbox OUTSIDE the repo). --run actually invokes claude; omit for a dry run.
uv run python scripts/eval/run_blind_eval.py --agent-dir eval_cases/<case> \
    --surface attribute --mode with-nudge    --sandbox <SCRATCH>/with_nudge    --run
uv run python scripts/eval/run_blind_eval.py --agent-dir eval_cases/<case> \
    --surface attribute --mode without-nudge --sandbox <SCRATCH>/without_nudge --run

# 3. grade each arm's REPORT.md against the held-out key (a PROPOSAL for the human in the loop)
uv run python scripts/eval/grade_blind_eval.py \
    --report <SCRATCH>/with_nudge/REPORT.md --key eval_keys/<case>_answer_key.json
```

Then **append an immutable run record** to `runs/NNNNNNNNN-<case>-<mode>.md` (never overwrite one),
embedding the task, the REPORT.md, the (now-revealable) ground truth, and the proposed + confirmed
grade — and update the run index + ablation table in `LEDGER.md`.

## Files

- `LEDGER.md` — live index: the run table, the WITH-vs-WITHOUT ablation summary, the resume pointer.
- `runs/` — one immutable record per (case, arm). Newest at the highest number; never edit a past one.
- Harness: `scripts/eval/blind_harness.py` (cases), `run_blind_eval.py` (the headless runner),
  `grade_blind_eval.py` (the calibration grader). Design + feasibility: `design/AUTOMATED_SCIENTIST.md`.
