# 000000001 · orchestrator · starting brief — round-3 problems (loop seed)

*Immutable run record (write-once). The live pointer is in `../LEDGER.md`; this is the seed
that starts the fix loop.*

## ▶ NEXT-AGENT KEY

```
NEXT  →  nudge-uq-fixer   on  P3  (design/invert safety gate — highest harm)
then  →  nudge-audit      on  the P3 fix
then  →  nudge-red-team   (re-scan)  →  P1  →  P2  →  …  until HOLES_FOUND: 0
STATUS:  AWAITING USER LAUNCH — do NOT dispatch any agent yet.
```

**Agent legend** (the sequence key): `1 = nudge-red-team` (find) · `2 = nudge-jax-physics`
(build new) · `3 = nudge-uq-fixer` (validate + fix) · `4 = nudge-audit` (independently
confirm). **Fix loop = `[ 3 → 4 → 1 ]`** repeated until red-team finds nothing; a `4`→FAIL
sends the problem back to `3`. Full protocol: `../README.md`.

## The problems to harden (most recently identified)

**Latest red-team brief (READ THIS for the full write-ups, repros, and verdicts):
[`design/FAILSAFE_REDTEAM_3.md`](../../FAILSAFE_REDTEAM_3.md).** Repros under
`scripts/redteam/`. These are red-team *claims* — role 3 independently re-reproduces each
before fixing (honesty rule: a hole is real only when reproduced).

| id | capability | LIM | one-line | repro |
|----|-----------|-----|----------|-------|
| **P3** | `design/invert` | LIM-013 | safety gate checks only the proximity *rise* > margin, never absolute `proximity_after ≥ NEAR_FOLD` → clears an intervention to proximity 0.589 as "safe" while `classify_robustness` calls it `near-fold`. **A confident-wrong safety flag on a *proposed intervention* — highest harm.** | `scripts/redteam/design_safety_gate_absolute_proximity.py` |
| **P1** | `differential` | LIM-016 | additive offset on ONE context's **perturbed** cells (control clean) fakes a confident `gain-diff` (truth: no-difference); invisible to the control-keyed depth guard, and lands on the `gain` channel the guard *exempts* as depth-robust. | `scripts/redteam/differential_additive_confound.py` |
| **P2** | `multi_reporter` | LIM-014 | multiplicative batch factor on the **perturbed** panel aliases 1:1 to a ceiling change → confident `ceiling` (truth: no-effect); the consistency guard checks only the control curves. | `scripts/redteam/multi_reporter_batch_confound.py` |

## The through-line (use this to design each fix)

Every confident-wrong hole across rounds 1–3 is a **confound applied to the perturbed / one
condition, invisible to a guard keyed on the control.** The fix direction: guards must inspect
the *perturbed* side too, not just the control. Prefer a **measured** guard (a degeneracy /
curvature) or a threshold-free consistency check over a hard threshold — hard thresholds near
a fold are scientifically invalid (round-2 lesson: the corruption onset is non-monotonic).

## Honesty (the governing rule)

Fix over confident-wrong is paramount, but never trade one confident-wrong for another and
never overclaim a fix: if a hole can only be **bounded** (abstain / lock + document), do that
honestly (like `NUDGE-LIM-019`) and say so. Every fix writes its NUDGE-LIM + FINDINGS record.
