# NUDGE hardening loop — LEDGER (live state · resume here)

**Governing rule: honesty above all** — never claim more than measured; abstain over
confident-wrong; document residual bounds loudly. See `README.md` for the protocol.

---

## ▶ RESUME POINTER

*(Mirror of the `NEXT →` block in the highest-numbered `runs/` record — currently
`runs/000000001-orchestrator-round3-brief.md`. That immutable record is the source of truth;
this is a convenience copy. See `README.md` → "The resume pointer & the queue".)*

**Status: NOT LAUNCHED — awaiting user go.** The user asked to set up the machinery and the
audit trail but **stop before running the fix loop** (limited network while traveling).

- **Next action when launched:** Fix loop `[ 3 → 4 → 1 ]` seeded from the problem queue.
- **Next agent:** `nudge-uq-fixer` on **P3** (design/invert safety gate — highest harm),
  then `nudge-audit` on its fix, then `nudge-red-team` to re-scan. Then P1, then P2.
  (Order is P3 → P1 → P2 by harm; the orchestrator may re-order.)
- **Do NOT dispatch any agent until the user says go.**

---

## Problem queue (found, not yet fixed)

Reported by red-team round 3 (`design/FAILSAFE_REDTEAM_3.md`); **pending independent UQ
validation** (role 3 re-reproduces before fixing — status reflects that they are red-team
claims, not yet main-loop-verified).

| id | capability | LIM | summary | repro | status |
|----|-----------|-----|---------|-------|--------|
| **P3** | `design/invert` | LIM-013 | Safety gate flags risk only on a proximity *increase* > margin, never absolute `proximity_after ≥ NEAR_FOLD` — an intervention to proximity 0.589 is cleared as "safe" while `classify_robustness` calls it `near-fold`. **Highest harm (a confident-wrong safety flag on a proposed intervention).** | `scripts/redteam/design_safety_gate_absolute_proximity.py` | OPEN |
| **P1** | `differential` | LIM-016 | Additive offset on ONE context's **perturbed** cells (control clean) fakes a confident `gain-diff` where truth is no-difference — invisible to the control-keyed depth guard, and lands on the `gain` channel the guard *exempts*. | `scripts/redteam/differential_additive_confound.py` | OPEN |
| **P2** | `multi_reporter` | LIM-014 | Multiplicative batch factor on the **perturbed** panel aliases 1:1 to a ceiling change → confident `ceiling` where truth is no-effect — consistency guard checks only the control curves. | `scripts/redteam/multi_reporter_batch_confound.py` | OPEN |

**Systemic pattern (the through-line for fixes):** every hole across rounds 1–3 is a confound
applied to the **perturbed / one** condition, invisible to a guard keyed on the **control**.
The fix direction is: guards must inspect the perturbed side too, not just the control.

**HELD attacks (no action — recorded as fail-safe wins):** hidden_node lone-leading probe
(rank-capped), constitutive dispersion route (structure-ratio scales with noise). Earlier
rounds also HELD: cross_modality, dose_response, bifurcation, the core parsimony gate, the
transition-mode gain gate.

---

## Run index (append-only — newest at the bottom; never edit a past row)

Pre-loop history (the found+fixed problems that motivated this system; kept for audit):

| # | when | role | target | outcome | commit / report |
|---|------|------|--------|---------|-----------------|
| H1 | round 1 | redteam | all attribution caps | 2 holes (near-fold multi-fit; additive-ambient synergy) + 3 HELD | `design/FAILSAFE_REDTEAM.md` |
| H2 | round 1 fix | uq+audit (main loop) | LIM-017 (v1) + LIM-009 | near-fold gated; synergy locked (xfail) | `0ff21bc`, `ab66f7f` |
| H3 | round 2 | redteam | core engine | 2 holes (constitutive capture-scale; the LIM-017 v1 margin was a knife-edge) + HELD | `design/FAILSAFE_REDTEAM_2.md` |
| H4 | round 2 fix | uq+audit (main loop) | LIM-019 lock + LIM-017 v2 | constitutive locked "adversarially bounded"; near-fold → graded down-weighting + best-buffered-pair CORROBORATION (measured 0 confident-wrong) | `c8c200b` |
| H5 | round 3 | redteam | differential / multi_reporter / design / hidden_node | 3 holes (P1/P2/P3) + 2 HELD | `design/FAILSAFE_REDTEAM_3.md` |

Loop runs (append-only — one immutable `runs/NNNNNNNNN-*.md` per agent invocation; newest
last; never edit a past row):

| # | run record | role | target | outcome |
|---|-----------|------|--------|---------|
| 000000001 | `runs/000000001-orchestrator-round3-brief.md` | orchestrator | round-3 seed | starting brief written; NEXT = uq-fixer on P3; **awaiting launch** |

---

## Closed problems (no-delete: fixed problems move here, rows never deleted)

*(none yet — P1/P2/P3 above are OPEN. When a problem passes audit, its row moves here with the
fix commit + the audit run record, so the queue above only holds OPEN work while the full
history is preserved.)*
