# 000000021 · orchestrator · P5 merged (audit PASSED) + independent verification + precision fix

*Immutable run record (write-once). This is currently the highest-numbered record, so its
`NEXT →` block below is the AUTHORITATIVE resume pointer (`../LEDGER.md` mirrors it).*

## ▶ NEXT →

```
NEXT  →  nudge-red-team  (FINAL FULL sweep — P3/P1/P4/P2/P5 all addressed)
STOP   when nudge-red-team reports HOLES_FOUND: 0 after a genuine full sweep.
(If it finds a NEW hole, that hole is queued and the fix loop continues.)
```

## What the orchestrator did

`nudge-audit` returned **PASS** on the P5 re-fix (run `000000020`, after the first attempt
FAILED in `runs/000000018`). The orchestrator independently re-verified and merged.

- **Merge:** `git merge --no-edit 1904b46` into `claude/nudge-hardening-loop-jeel0r` →
  **fast-forward** `1f14ce3..1904b46` (no conflicts — the fixer's branch was based on the full
  hardening trail incl. the P5-FAIL commit `1f14ce3`, so it is a strict superset). The
  fast-forward's full delta is 11 files: 10 are the P5 fix; the 11th is `design/AUTOMATED_
  SCIENTIST.md` — a **legitimate user commit** (`0bcd2a6`, authored by the repo owner, the tip
  of `origin/main`) the fixer's lineage merged in. The only `src/` change is `differential.py`.
  Verified P3 (`near_fold` in `invert.py`) and P2 (`off_on_coupling` in `multi_reporter.py`)
  intact.
- **Orchestrator honesty-precision fix (the audit's advisory, actioned):** tightened the
  escaping-confound magnitude claim "≤ 0.48" → "≤ ~0.50 (0.498 at c=1.35, audit `runs/000000020`)"
  in `differential.py` (docstring), `scripts/vv/FINDINGS.md` §P5, and `docs/known_limitations.yaml`
  LIM-016 — a doc-only change (no logic, gate constants unchanged: `_OFF_RESOLVABILITY_MIN=0.25`,
  `_CEILING_RESOLVE_MAG_MIN=0.60`).
- **Frozen core:** the P5 `src/` delta is only `differential.py`. No `fit.py`, no `core/`. ✔
- **Independent gate on the merged tree (pinned toolchain — note: my venv reverted to pyright
  1.1.408 after an environment reset, which false-positives 2 pre-existing `bridge.py` errors
  that also appear on the base; `uv sync` restored the pin 1.1.411 → clean):**
  - ruff: All checks passed · pyright src: **0 errors**
  - check_anomalies / citations / impl_mapping / mechanism_cards / hardening_append_only: OK
  - RT repro `differential_small_mult_gain_hole.py 4` (independent): **HOLES: 0** *(see below)*
  - `pytest -q` (fast lane): **277 passed / 5 skipped / 2 xfailed** *(see below)*
  - the TEST repro (HOLES: 0, 5 seeds), the interior-factor probe (HOLES: 0), and the
    differential slow suite (34 passed + 2 xfailed) were run by the P5 audit on the
    byte-identical `differential.py` logic (my only edit was docstrings) — not re-run here.

## Outcome

**P5 BOUNDED (honest) and merged.** A small perturbed-only multiplicative scale no longer fakes
a confident `gain-diff`/`ceiling-diff`: the near-zero-basal regime is caught by the measured
resolvability gate; the resolvable regime is caught by the measured ceiling-magnitude gate; both
separators are genuine ~0.10-margin gaps. The residual is an honest over-abstention BOUND (a
small genuine ceiling difference / the near-zero-basal channel are sacrificed, strict-xfail-
locked and documented) — never a residual confident-wrong (the first attempt's failure). The
first attempt's overclaims are retracted with a SUPERSEDED banner. Moved to LEDGER "Closed".

- Merge (fast-forward) HEAD: `1904b46` + the orchestrator precision commit.
- Fix branch: `worktree-agent-a7c7ce9742440090b` · Audit: run `000000020` (PASS).
