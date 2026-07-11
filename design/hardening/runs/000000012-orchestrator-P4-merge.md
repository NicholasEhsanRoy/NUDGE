# 000000012 · orchestrator · P4 merged (audit PASSED) + independent verification

*Immutable run record (write-once). This is currently the highest-numbered record, so its
`NEXT →` block below is the AUTHORITATIVE resume pointer (`../LEDGER.md` mirrors it).*

## ▶ NEXT →

```
NEXT  →  nudge-red-team  (re-scan: P4-fix regression check + fresh confident-wrong sweep)
then  →  nudge-uq-fixer  on  P2  (multi_reporter per-condition batch-scale confound, LIM-014)
then  →  nudge-audit → merge → red-team ; STOP at HOLES_FOUND: 0 (full sweep, all fixed)
```

## What the orchestrator did

`nudge-audit` returned **PASS** on P4 (run `000000011`). The orchestrator independently
re-verified on the merged working branch and merged.

- **Merge:** `git merge --no-edit f5d0b870b56f008480290fb194258f360638e5d7` into
  `claude/nudge-hardening-loop-jeel0r` → merge commit **`ebda9c6`**. Three conflicts resolved:
  CHANGELOG + STATE (additive — combined the P3 + P1 + P4 round-3 notes, no content lost);
  `scripts/redteam/differential_multiplicative_confound.py` (add/add — the red-team discovery
  version vs the fixer's comprehensive regression version) resolved to **theirs** (the fixer's
  5-factor inflating+deflating version the audit validated; the red-team's discovery original
  is preserved in git `63516d4` + run `000000009`).
- **Source integrity:** `src/nudge/inference/differential.py`, `tests/inference/test_differential.py`,
  the `differential_attribution` Mechanism Card verified **byte-identical** to the audited
  P4 commit `f5d0b87`; `docs/known_limitations.yaml`'s `NUDGE-LIM-016` entry byte-identical
  to f5d0b87's (P4 sharpening) while retaining P3's `NUDGE-LIM-013` — so the audited gate
  results transfer.
- **Frozen core:** the P4 commit touches only `src/nudge/inference/differential.py`. No
  `fit.py`, no `core/`. ✔
- **Independent gate on the merged tree (pinned toolchain):**
  - ruff: All checks passed · pyright src: **0 errors**
  - check_anomalies / citations / impl_mapping / mechanism_cards / hardening_append_only: OK
  - `pytest -q` (fast lane): **261 passed, 5 skipped, 1 xfailed** (261 = the audit's 258
    on the P4-on-`3fafbda` base + P3's 3 `tests/design` cases my branch also carries)
  - the heavy repro (~10 min, both directions → HOLES: 0) + differential slow suite (~10.5 min,
    22 passed + 1 xfailed) were run by the P4 audit on the byte-identical `differential.py` /
    `test_differential.py`; not re-run here (identical content, OOM-aware).

## Outcome

**P4 CLOSED (inflating) / BOUNDED (deflating) and merged.** The multiplicative perturbed-only
scale no longer fakes a confident `ceiling-diff` (measured ceiling-scoped OFF-cluster-scale
gate 4c, band [0.80, 1.30]); genuine ceiling increases + gain + threshold still resolve; the
narrow, honest sacrifice of strong genuine ceiling reductions is strict-xfail-locked and
documented loudly. Moved to LEDGER "Closed problems".

- Merge commit: `ebda9c6`  ·  Fix commit: `f5d0b870b56f008480290fb194258f360638e5d7`
- Audit: run `000000011` (PASS)
- Recorded future red-team candidate (audit's out-of-scope find): a pre-existing
  gain⇄ceiling-*reduction* mis-attribution degeneracy in `differential`, unaffected by P4.
