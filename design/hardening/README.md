# NUDGE hardening loop — the agent-team audit trail

This directory is the **complete, append-only, never-deleted record of everything the
hardening-loop agents do**. It is committed to git, so every action is traceable through
both these files and the git history. Nothing here is ever overwritten or deleted — a run
record is immutable once written; corrections are new records that reference the old one.

**Honesty is the governing rule of this entire system** (it is the #1 rule in every agent's
definition). Never claim more than has been *measured* — about NUDGE, about a found hole,
about a fix, or about an audit. Abstaining / "bounded + documented" is a valid, valued
outcome; a confident-wrong (in the tool, a report, a fix claim, or an audit) is the one
unacceptable outcome. A polished-but-false claim scores worse than an honest gap.

## The four agents (roles)

| # | Agent (`.claude/agents/`) | Role |
|---|---------------------------|------|
| 1 | `nudge-red-team` | Searches for **confident-wrong holes**; reports + reproduces, never fixes. Its verdict (`HOLES_FOUND: n`) drives the loop; `0` after a genuine sweep = **STOP**. |
| 2 | `nudge-jax-physics` | Builds a **new differentiable model** (additive, frozen-core) + finds datasets via a deep-research subagent of itself. The loop's seed/builder. |
| 3 | `nudge-uq-fixer` | Takes one red-team hole → independently **validates** → **measured, targeted fix** → writes the honesty record (a NUDGE-LIM + a FINDINGS entry). |
| 4 | `nudge-audit` | **Independently re-verifies** the fixer's claim (hole gone? positive control still resolves? honesty record accurate? full gate green?). Returns `AUDIT: PASS/FAIL`. |

## The protocol (the sequence)

- **Build loop:** `2` → then repeat `[ 1 → 3 → 4 ]` → **STOP** when `1` reports `HOLES_FOUND: 0`.
- **Fix loop** (harden existing problems): repeat `[ 3 → 4 → 1 ]` seeded from the LEDGER
  problem queue → **STOP** when `1` reports `HOLES_FOUND: 0`.
- After each `3 → 4` (one problem fixed + audited), return to `1` to **re-scan** — this
  catches fix-induced regressions.
- `4` returns **FAIL** → the problem goes **back to `3`** (do not advance to `1`).
- Every agent runs **worktree-isolated**, commits to its own branch, and **does not merge**.
  The orchestrator verifies, merges to `main`, and writes the immutable run record.

## How to RESUME (read this first each session)

1. Read `LEDGER.md` → the **Resume pointer** (the next agent + its target) and the **problem
   queue** (open / in-progress / fixed / audited).
2. Dispatch that agent (via the Agent tool, `subagent_type` = the agent name).
3. When it returns: **append an immutable run record** to `runs/NNNN-<role>-<target>.md`
   (never overwrite an existing one), verify + merge its branch if applicable, and update the
   Resume pointer + queue in `LEDGER.md`.
4. Repeat per the protocol. Because the state lives in these committed files, the loop
   survives a network drop / new session — just re-read `LEDGER.md`.

## The resume pointer & the queue — deletion-proof by design

The pointer (what runs next) is the only piece of *live* state, so it is engineered so a
deletion or a bad edit can never lose it:

- **The authoritative pointer is the `NEXT →` block at the top of the highest-numbered run
  record in `runs/`.** Those records are immutable and append-only (CI-enforced), so the
  pointer's ground truth can never be deleted or silently rewritten. **Every run record MUST
  begin with a `NEXT →` block** naming the next agent + target; the current pointer is simply
  the newest record's `NEXT →`.
- `LEDGER.md`'s "Resume pointer" is a **mirror** of that latest `NEXT →`, kept for
  convenience. If `LEDGER.md` is ever lost or corrupted, the entire live state (pointer +
  what happened) is reconstructable from `runs/` alone — the LEDGER is a view, not the source
  of truth.
- **The problem queue is never row-deleted.** A problem's status transitions *in place*
  (`OPEN → IN-PROGRESS → FIXED → CLOSED`), and a closed problem is moved to the LEDGER's
  "Closed" section — its row is preserved, and every transition also produces an immutable
  `runs/` record. So even though `LEDGER.md` is editable, no problem, fix, or audit verdict is
  ever erased from the trail.
- **CI (`scripts/check_hardening_append_only.py`) enforces the file-level invariant:** no file
  under `design/hardening/` may be deleted, and no `runs/` record may be modified or renamed —
  on every change, not just doc changes. Content-level history is protected by the immutable
  `runs/` records themselves (the LEDGER can be edited, but its ground truth cannot be lost).

## How to AUDIT (what to check)

- `LEDGER.md` is the index; `runs/` holds one immutable record per agent invocation
  (who ran, on what, its verbatim conclusion, the commit SHA it produced, the outcome).
- Cross-check any fix against its `nudge-audit` run record (the independent PASS/FAIL + the
  commands re-run), the merged commit, and the NUDGE-LIM / FINDINGS the fix wrote.
- Nothing is deleted: a superseded finding/fix stays, with a later record explaining it.

## Run-record naming (and what to do if we run out)

`runs/NNNNNNNNN-<role>-<short-target>.md` — a strictly-monotonic integer **zero-padded to 9
digits** (`000000001` …), `<role>` ∈ {`redteam`, `jax-physics`, `uq-fixer`, `audit`,
`orchestrator`}. Immutable once committed.

- **The integer is the source of truth; the zero-padding is cosmetic** (it just makes a plain
  lexical `ls` sort correctly up to `999999999` ≈ one billion records).
- **If we ever exceed 9 digits, nothing breaks:** records simply continue with more digits
  (`1000000000-…`). The append-only invariant is unaffected. The only rule: **sort run records
  numerically by the leading integer, never lexically** — then mixed-width numbers still order
  correctly. (`scripts/check_hardening_append_only.py` does not care about the width.)
- **Optional large-scale ergonomics (only if a single `runs/` dir gets unwieldy):** begin
  bucketing *new* records into range subdirectories — `runs/0000000000-0000009999/`, then
  `…0010000-…0019999/`, etc. — and note the active bucket in `LEDGER.md`. **Existing records
  are never moved** (that would violate append-only); a new bucket only ever starts from the
  next unused number. Bucketing is purely forward-looking reorganization.
