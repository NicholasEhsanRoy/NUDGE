# 000000017 · red-team · FINAL FULL sweep → HOLES_FOUND: 1 (NEW hole P5) — NOT a STOP

*Immutable run record (write-once). This is currently the highest-numbered record, so its
`NEXT →` block below is the AUTHORITATIVE resume pointer (`../LEDGER.md` mirrors it).*

## ▶ NEXT →

```
NEXT  →  nudge-uq-fixer  on  P5  (differential SMALL multiplicative confound — completes the P4 gate)
then  →  nudge-audit → orchestrator merge → nudge-red-team (re-scan)
STOP when nudge-red-team reports HOLES_FOUND: 0 after a genuine FULL sweep.
```

## Role / target

- **Role:** `nudge-red-team` (role 1), worktree-isolated. The **final full sweep** — the loop's
  STOP gate, run with all four round-3 holes (P3/P1/P4/P2) fixed + merged.
- **Result:** **HOLES_FOUND: 1** → NOT a STOP. The four merged fixes HELD jointly (no
  cross-fix regression), but a genuine sweep surfaced a NEW confident-wrong (**P5**).

## The NEW hole — P5 (verified 4/4 seeds; orchestrator independently re-confirming)

A **SMALL uniform** multiplicative scale on ONE context's PERTURBED cells (`c ≈ 1.15–1.25`,
control clean), `mechanism="none"` (truth = no-difference) → a confident `gain-diff` /
`ceiling-diff`. **8 confident-wrong across 4 seeds.** Repro:
`scripts/redteam/differential_small_mult_gain_hole.py`.

```
seed=0 factor=1.15 gain-diff    off_shift=0.99 off_scale=1.231 dBIC_runner(vmax)=14.0
seed=1 factor=1.25 ceiling-diff               off_scale=1.279 dBIC_runner(n)=149.9
seed=2 factor=1.15 ceiling-diff               off_scale=1.131 dBIC_runner(K)=28.5
seed=3 factor=1.15 gain-diff    off_shift=0.97 off_scale=1.138 dBIC_runner(vmax)=10.9
```

**Why the P4 gate 4c misses it (the P4 fix was calibrated only on `c ≥ 1.5`):**
1. **Gain channel:** at small `c` the BIC winner is the **gain (n)** knob, and gate 4c is
   **ceiling-scoped** — it never consults `off_scale` for a gain winner (a locked test,
   `test_classify_off_scale_guard_is_ceiling_scoped`, explicitly enforces that). Gate 4b
   (`off_shift ≈ 0.99`) is blind to a multiplicative scale.
2. **Ceiling channel:** `off_scale` lands in the **(1.18, 1.30] blind gap** — above NUDGE's
   own measured genuine-ceiling maximum (1.18, even at ×4) yet below gate 4c's band upper
   (1.30 = the 1.18↔1.43 midpoint, chosen without probing the `c ∈ (1.0, 1.5)` interior).

**Not a degeneracy / not a documented bound (honest):** a uniform scale inflates the
OFF-cluster spread (`off_scale > 1`), a fingerprint distinguishing it from genuine gain
(`≈1`) and genuine ceiling (`≤1.18`) — the distinguishing statistic EXISTS, but the gate
ignores it (gain winner) or has too loose a cut (ceiling winner). Distinct from the P4
above-median-only evader (which WAS observationally-identical-to-genuine).

## HONESTY FOLLOW-THROUGH (important — the loop caught two overclaims)

1. P5 shows the P4 "INFLATION is CLOSED" claim — even after `runs/000000013`'s tightening to
   "closed against a **uniform** or smoothly content-dependent inflating scale" — is **still
   an overclaim**: a *small* uniform inflating scale is NOT closed. **The P5 fix MUST correct
   this claim** (in `differential.py` docstring + gate-4c comment + FINDINGS §P4) so it names
   only what is actually closed, and add the P5 channel to `NUDGE-LIM-016`.
2. This also means the P4 audit's PASS, while valid for what it *tested* (large factors +
   genuine ×1.4–4), did not probe the `c ∈ (1.0, 1.5)` interior — a coverage gap the final
   sweep closed. No fix was confident-wrong *at merge*; the claim was broader than the test.

## Mitigation direction (red-team's pointer — the P5 fixer MEASURES + decides)

Extend the `off_scale` out-of-band abstention to **all** `*-diff` winners (not just ceiling —
a small multiplicative confound inflates `off_scale` whichever channel BIC picks; genuine
gain/threshold measured `off_scale ≈ 1`, so this should not over-abstain), AND tighten the
ceiling band upper from 1.30 toward the measured genuine-ceiling max (1.18) + a defensible
margin. Re-measure the separator on the `c ∈ [1.05, 1.5]` interior the P4 fix skipped. Revise
the now-falsified locked test `test_classify_off_scale_guard_is_ceiling_scoped`. Expect: some
CLOSED (off_scale above the tightened cut) and possibly a residual BOUND for very small `c`
whose `off_scale` overlaps genuine ceiling (≤1.18) — decide honestly by measurement.

## Score table (final sweep)

| Probe | Verdict |
|-------|---------|
| P5: small mult confound → confident gain-/ceiling-diff | **HOLE — verified 4/4 seeds** |
| P1/P4 boundary knife-edge (fractional confounds at each gate edge) | HELD |
| P2 batch confound re-confirm | HELD (6/6 `unresolved`) |
| P3 near-fold + margin-gaming re-confirm | HELD |
| LIM-017 lyapunov corroboration collusion | **NOT REACHED** (budget spent on P5; plausible, UNRUN probe at `scripts/redteam/lyapunov_batch_scale_collusion.py` — NOT a claimed result) |

## Coverage vs skipped (budget-honest — recorded so nothing reads as "fully swept")

- Covered (heavy, sequential): the P5 sweep; P1/P4 boundary knife-edge; P2 batch; P3 regression.
- **Not reached:** the LIM-017 best-buffered-pair corroboration collusion in
  `attribute_lyapunov_multi` (analysed plausible — that path has no gate-4b/4c analog — but
  UNRUN, so NOT claimed); partial-panel P2 gaming; end-to-end composition into `design()`
  (narrow — both upstream feeders HELD in prior rounds). **These remain open probe surface for
  a future sweep** once P5 is fixed.

## Independent verification (orchestrator)

Merged the additive artifacts (`design/FAILSAFE_REDTEAM_5.md` + 4 repro scripts, incl. the
UNRUN lyapunov probe) — no `src/` touched. Re-running the P5 repro to confirm before queueing.

## Branch + commit (merged additively by the orchestrator)

- Branch: `worktree-agent-aa11b76233c1561bb` · Commit: `6a1c774ec3defa00922a75528b9bde825b004cdc`
