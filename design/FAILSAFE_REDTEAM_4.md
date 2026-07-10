# NUDGE fail-safe red-team — ROUND 4 (P3-fix regression re-scan)

**Mandate.** Same as rounds 1–3: adversarially force a NUDGE capability to emit a
*confident, specific, WRONG* call where the honest answer is abstention. This round is a
**re-scan after the P3 fix merged** — its purpose is to catch **fix-induced regressions**
and any NEW confident-wrong in `design/invert` or an adjacent capability the P3 change
touched. This document reports and reproduces; it does **not** fix. No `src/` capability
code, `fit.py`, `core/`, the decoy battery, or any fail-safe margin was touched — only this
report + one new `scripts/redteam/*.py`.

**The P3 fix under test** (`src/nudge/design/invert.py`, merge `017bd58`): the bifurcation
safety gate now fires `high_risk_of_instability` on `(delta > margin)` OR
`(proximity_after >= NEAR_FOLD)` — an absolute near-fold alarm reusing the shipped
`nudge.inference.bifurcation.NEAR_FOLD = 0.55`. It added `SafetyReport.near_fold`, reworded
the near-fold reason to AGREE with `classify_robustness`, and carries the
one-sided-lower-bound caveat on the SAFE branch.

## Result: **HOLES_FOUND: 0** — the P3 fix HELD; no regression, no new confident-wrong.

Every attack drove the SHIPPED `design()` path end-to-end against the fixed code (run from
the fixed checkout; the fixer's / audit's numbers reproduced independently).

## Score

| # | Probe (the way the fix could have broken) | Verdict |
|---|-------------------------------------------|---------|
| 1 | **Absolute-check completeness** — can any reachable plan land `proximity_after >= NEAR_FOLD` yet NOT flag `high_risk`? (38 plans across `1node`/`2node`/`toggle`, both start basins; + the bistable→bistable path) | **HELD** |
| 2 | **base-not-bistable → creates-near-fold** — the P3-claimed branch: create a switch at `proximity >= 0.55` from a monostable base | **HELD** (flags `near_fold`/`high_risk`; agrees with `classify_robustness`) |
| 3 | **Over-abstention** — does the new absolute alarm fire "HIGH RISK" on a genuinely-robust intervention (proximity below 0.55, sub-margin delta)? | **HELD** (no spurious high-risk; robust moves cleared "OK") |
| 4 | **Gameable boundary via `margin`** — a user routing around the relative alarm by passing a huge `margin` | **HELD** (absolute check is independent of `margin`; original P3 construction still flags high-risk at `margin=100`) |
| 5 | **SAFE-branch wording** — is the "stays away from the fold" reason still reassuring when `proximity_after` is a one-sided lower bound? | **HELD** (the SAFE branch now carries the "one-sided LOWER bound — the true proximity may be higher" caveat) |

**Repro (single, fast, deterministic):**
`scripts/redteam/design_p3_regression_check.py` (`uv run python …`, ~1–2 min). Asserts
invariants I1–I5 (above) through the shipped `design()` path; exits 0 when every invariant
HOLDS, 2 on any violation. No seed dependence (a gradient inversion to a fixed target).

## Why each probe HELD

- **The fix is internally consistent by construction.** Because `_safety_report` reuses the
  *same* `NEAR_FOLD` constant `classify_robustness` uses, the two can never disagree on the
  *reported* proximity of the identical circuit. Across all three bistable code paths
  (bistable→bistable, monostable→creates-switch, crosses-fold) a reported
  `proximity_after >= 0.55` **always** sets `near_fold=True → high_risk=True`. The original
  P3 hole (`design_safety_gate_absolute_proximity.py`) now returns rc 0 / "no hole": at
  `proximity 0.50→0.59`, `delta 0.089 < margin 0.15` yet `high_risk=True`, reason names the
  near-fold regime and cites `classify_robustness`.
- **No over-abstention.** The absolute alarm fires only at/above `NEAR_FOLD`, which is
  exactly where `classify_robustness` itself calls a circuit `near-fold`; below it, robust
  interventions are cleared "OK, stays away from the fold". So the gate never abstains where
  NUDGE's own robustness verdict would call the circuit robust — there is no daylight a real
  user would route around.
- **`margin` cannot disable the absolute check.** `high_risk = (delta > margin) OR
  near_fold`. Even `margin=100` (relative alarm fully suppressed) still flags the original
  near-fold landing via `near_fold`.
- **The one-sided caveat is present on the SAFE branch.** Verified end-to-end: a robust
  ON-level adjustment (`proximity 0.50→0.43`, `one_sided=True`) is cleared with the reason
  "… OK, stays away from the fold (proximity 0.50->0.43 (a one-sided LOWER bound — the true
  proximity may be higher))."

## Residual observations (NOT holes — recorded honestly, DESCRIBED only)

1. **A minor SAFE-branch wording asymmetry (quality note, not a confident-wrong).** When a
   monostable base is intervened into a *robust* bistable switch (`proximity < 0.55`), the
   reason reads "safety: base circuit is not bistable (no switch to destabilize)." That
   branch describes the *base* correctly, but (a) does not mention that the intervention
   **created** a bistable switch, and (b) unlike the sibling else-branch, carries **no**
   one-sided caveat when the created switch's proximity is a lower bound. Reproduced through
   the shipped path (monostable `vmax∈{2.5..2.75}` → `vmax≈×1.1–1.4` creating switches at
   reported `proximity 0.37–0.50`, `one_sided=True`). This is **not** scored as a hole: the
   safety *verdict* is correct — `classify_robustness` independently calls the created
   switch **robust**, so "safe" is the honest answer, and I have no independent, more-accurate
   reference to show the created switch is truly near-fold. It is a candidate wording
   refinement (mirror the created-switch's `proximity_after` + the one-sided caveat into that
   branch), not a fail-safe defect. Severity: cosmetic.
2. **The LIM-012 residual is unchanged.** A reported `proximity_after` below `0.55` that is a
   one-sided *lower bound* could correspond to a truly-nearer circuit; the fix now caveats
   this on the SAFE branch (else-branch), which is the honest treatment. Establishing a
   *confident-wrong* here would require an independent more-accurate proximity reference,
   which does not exist; so this stays a documented bound (`NUDGE-LIM-012`/`-013`), not a
   verified hole.

## P1 / P2 status (already-known OPEN holes — not re-reproduced, per budget)

The P3 fix touched **only** `src/nudge/design/invert.py` (verified: `git show --stat`
`017bd58` under `src/` = `invert.py` alone). It did not touch `inference/differential.py`
or `inference/multi_reporter.py`, so **P1** (differential additive-perturbed-offset →
confident `gain-diff`) and **P2** (multi_reporter per-condition batch scale → confident
`ceiling`) are **structurally unchanged** and remain OPEN in the queue. Not re-run this pass.

## Honest caveats

- All checks are on **synthetic** circuits with engineered near-fold / robust interventions;
  none is demonstrated on a real screen.
- HELD results are genuine outcomes under the attacks tried (I1–I5); they do not prove no
  other attack exists. The strongest structural reason the fix holds — it reuses the *same*
  `NEAR_FOLD` constant as `classify_robustness` — makes a *reported*-proximity disagreement
  impossible by construction, which is why the sweep found zero violations.
