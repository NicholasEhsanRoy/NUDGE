# 000000005 · red-team · P3-fix regression re-scan → HOLES_FOUND: 0

*Immutable run record (write-once). This is currently the highest-numbered record, so its
`NEXT →` block below is the AUTHORITATIVE resume pointer (`../LEDGER.md` mirrors it).*

## ▶ NEXT →

```
NEXT  →  nudge-uq-fixer  on  P1  (differential additive-perturbed-offset confound, LIM-016)
then  →  nudge-audit → orchestrator merge → nudge-red-team (re-scan) ; then P2
STOP when nudge-red-team reports HOLES_FOUND: 0 after a genuine FULL sweep (P1+P2 fixed).
```

## Role / target

- **Role:** `nudge-red-team` (role 1), worktree-isolated. A **regression re-scan** after
  the P3 fix merged (`017bd58`) — scoped to catch a P3-fix-induced confident-wrong /
  over-abstention / gameable boundary, plus a light fresh sweep.
- **Target:** the merged P3 fix in `src/nudge/design/invert.py`.

## Verdict: **HOLES_FOUND: 0** (P3 fix HELD)

Five probes, all HELD (deterministic repro
`scripts/redteam/design_p3_regression_check.py`, rc 0):

| # | probe | verdict |
|---|-------|---------|
| I1 | absolute-check completeness — any reachable plan lands `proximity ≥ NEAR_FOLD` yet NOT high-risk? (38 plans × 1node/2node/toggle × both basins) | HELD |
| I2 | base-not-bistable → creates a near-fold switch | HELD (flags near_fold/high_risk; agrees with classify_robustness) |
| I3 | over-abstention — high-risk on a genuinely-robust (<0.55) intervention? | HELD (no spurious high-risk) |
| I4 | gameable via a huge `margin` (suppress the relative alarm) | HELD (absolute check is margin-independent; original P3 flags even at margin=100) |
| I5 | SAFE-branch wording reassuring under a one-sided lower bound? | HELD (safe branch carries the "may be higher" caveat) |

**Why it holds structurally:** `_safety_report` reuses the SAME `NEAR_FOLD` constant that
`classify_robustness` uses, so the two can never disagree on the reported proximity of the
identical circuit; a reported `proximity_after ≥ 0.55` always sets `near_fold → high_risk`.

## Residual observations (NOT holes — honest, recorded)

1. Minor SAFE-branch wording asymmetry: a monostable base intervened into a *robust*
   bistable switch reports "base circuit is not bistable" without echoing the created
   switch's proximity / one-sided caveat. NOT a confident-wrong (`classify_robustness`
   independently calls that created switch **robust**, so "safe" is honest; no independent
   more-accurate reference exists). Candidate wording refinement only.
2. LIM-012 residual unchanged (a reported sub-0.55 value is a lower bound); the fix now
   caveats it on the SAFE branch — the honest treatment.

## Independent verification (orchestrator)

Merged the additive artifacts (`design/FAILSAFE_REDTEAM_4.md` + the repro) into the
hardening branch; ran the repro against the fixed in-tree code → **rc 0, "ALL INVARIANTS
HELD"**. No `src/` change in the red-team commit (`git show --stat a561bdd`: 2 files, both
new).

## P1 / P2 status

Not re-reproduced (per budget) — both remain **OPEN / structurally unchanged** (the P3 fix
touched only `src/nudge/design/invert.py`). Next fix target: **P1**.

## Branch + commit (merged additively by the orchestrator)

- Branch: `worktree-agent-a0871af4c5ccd9fea` · Commit: `a561bdd94cf17fdd2e47971e6e98a547f50a2ad9`
- Env note: the red-team's worktree based off `3fafbda` (session-start HEAD), not the
  fixed tip; it verified against the fixed shared checkout. **General gotcha for the loop:
  worktree agents base off session-start HEAD, so the orchestrator's post-merge in-tree
  verification is the authoritative gate.**
