# 000000008 · orchestrator · P1 merged (audit PASSED) + independent verification

*Immutable run record (write-once). This is currently the highest-numbered record, so its
`NEXT →` block below is the AUTHORITATIVE resume pointer (`../LEDGER.md` mirrors it).*

## ▶ NEXT →

```
NEXT  →  nudge-red-team  (re-scan: P1-fix regression check + fresh confident-wrong sweep)
then  →  nudge-uq-fixer  on  P2  (multi_reporter per-condition batch-scale confound, LIM-014)
then  →  nudge-audit → merge → red-team ; STOP at HOLES_FOUND: 0 (full sweep, all fixed)
```

## What the orchestrator did

`nudge-audit` returned **PASS** on P1 (run `000000007`). The orchestrator independently
re-verified on the merged working branch and merged.

- **Merge:** `git merge --no-edit b562da9d3a0a98ab7f39b0624f56188e9a643b8a` into
  `claude/nudge-hardening-loop-jeel0r` → merge commit **`99d73b8`**. Two additive doc
  conflicts (CHANGELOG, STATE — both P3 and P1 appended a round-3 note to the same section)
  resolved by KEEPING BOTH notes (no content lost). `differential.py` and `design/invert.py`
  verified **byte-identical** to their audited commits (`b562da9` / `a263789`), so the
  audited heavy-gate results transfer to the merged tree.
- **Frozen core:** the P1 commit touches only `src/nudge/inference/differential.py` under
  `src/`. No `fit.py`, no `core/`. ✔
- **Independent gate on the merged tree (pinned toolchain):**
  - ruff: All checks passed
  - pyright src: **0 errors**
  - check_anomalies / check_citations / check_impl_mapping / check_mechanism_cards /
    check_hardening_append_only: OK
  - `pytest -q` (fast lane): **257 passed, 5 skipped, 61 deselected, 1 xfailed** (257 =
    254 + the 3 new gate-4b unit tests; the xfail is the pre-existing synergy lock)
  - the heavy differential repro (~10–13 min) + slow suite (~8 min) were run by the P1
    audit on the byte-identical `differential.py`: red-team HOLES: 0 (seeds 0,1,2), slow
    suite 11 passed (genuine ceiling/gain still resolve). Not re-run here (identical content,
    OOM-aware — no concurrent JAX).

## Outcome

**P1 CLOSED (inflating) / BOUNDED (deflating) and merged.** The confident `gain-diff` from
an additive perturbed-only offset now abstains (measured one-sided OFF-baseline-inflation
gate at 2.5); every genuine positive control still resolves (no over-abstention); the
deflating residual is documented loudly. Moved to LEDGER "Closed problems".

- Merge commit: `99d73b8`  ·  Fix commit: `b562da9d3a0a98ab7f39b0624f56188e9a643b8a`
- Audit: run `000000007` (PASS)
