# NUDGE hardening loop — LEDGER (live state · resume here)

**Governing rule: honesty above all** — never claim more than measured; abstain over
confident-wrong; document residual bounds loudly. See `README.md` for the protocol.

---

## ▶ RESUME POINTER

*(Mirror of the `NEXT →` block in the highest-numbered `runs/` record — currently
`runs/000000008-orchestrator-P1-merge.md`. That immutable record is the source of truth;
this is a convenience copy. See `README.md` → "The resume pointer & the queue".)*

**Status: RUNNING.** P3 **CLOSED**; P1 **CLOSED (inflating) / BOUNDED (deflating) + merged**
(audit PASS, independently re-verified). Now: P1-fix red-team re-scan, then P2.

- **Next agent:** `nudge-red-team` — re-scan (P1-fix regression check + fresh sweep). Then
  `nudge-uq-fixer` on **P2** (multi_reporter per-condition batch-scale confound, LIM-014),
  then `nudge-audit` → merge → `nudge-red-team`.
- **STOP** when `nudge-red-team` reports `HOLES_FOUND: 0` after a genuine FULL sweep with
  P1 + P2 both fixed.

---

## Problem queue (found, not yet fixed)

Reported by red-team round 3 (`design/FAILSAFE_REDTEAM_3.md`); **pending independent UQ
validation** (role 3 re-reproduces before fixing — status reflects that they are red-team
claims, not yet main-loop-verified).

| id | capability | LIM | summary | repro | status |
|----|-----------|-----|---------|-------|--------|
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
| 000000002 | `runs/000000002-uq-fixer-P3.md` | uq-fixer | P3 (LIM-013) | fix claim: two-alarm safety gate (`delta>margin` OR `proximity≥NEAR_FOLD`); CLOSED; commit `a263789` |
| 000000003 | `runs/000000003-audit-P3.md` | audit | P3 fix | **AUDIT: PASS** — hole flags, no over-abstention, frozen core untouched, full gate green |
| 000000004 | `runs/000000004-orchestrator-P3-merge.md` | orchestrator | P3 merge | independently re-verified + merged → `017bd58`; P3 CLOSED |
| 000000005 | `runs/000000005-redteam-P3rescan.md` | redteam | P3-fix re-scan | **HOLES_FOUND: 0** — P3 fix held (5 probes); report `FAILSAFE_REDTEAM_4.md`; commit `a561bdd` |
| 000000006 | `runs/000000006-uq-fixer-P1.md` | uq-fixer | P1 (LIM-016) | fix claim: measured one-sided OFF-baseline-inflation gate 4b (>2.5); CLOSED-inflating/BOUNDED-deflating; commit `b562da9` |
| 000000007 | `runs/000000007-audit-P1.md` | audit | P1 fix | **AUDIT: PASS** — hole abstains (seeds 0,1,2), genuine diffs still resolve, frozen core untouched, honesty accurate |
| 000000008 | `runs/000000008-orchestrator-P1-merge.md` | orchestrator | P1 merge | independently re-verified + merged → `99d73b8` (2 additive doc conflicts resolved); P1 CLOSED/BOUNDED |

---

## Closed problems (no-delete: fixed problems move here, rows never deleted)

| id | capability | LIM | resolution | fix commit | audit |
|----|-----------|-----|------------|-----------|-------|
| **P3** | `design/invert` | LIM-013 | **CLOSED** — safety gate now fires on `delta > margin` OR absolute `proximity_after ≥ NEAR_FOLD` (reuses the shipped constant → agrees with `classify_robustness` on the identical circuit); one-sided caveat carried on the SAFE branch. Passing decoy + 3 regression tests. 0 confident-wrong; positive control still "OK". | `a263789` (merge `017bd58`) | `runs/000000003-audit-P3.md` (PASS) |
| **P1** | `differential` | LIM-016 | **CLOSED (inflating) / BOUNDED (deflating)** — measured one-sided gate 4b abstains when a context's perturbed OFF baseline is inflated >2.5× its own control (separator: confident-wrong `off_shift≥2.99` vs genuine `≤1.96`). Deflating perturbed-only offset aliases with a genuine reduction → documented residual, not guarded. Decoy + tests; genuine ceiling/gain still resolve (no over-abstention). | `b562da9` (merge `99d73b8`) | `runs/000000007-audit-P1.md` (PASS) |

*(P1/P2 remain OPEN in the queue above. When each passes audit its row moves here with the
fix commit + the audit run record, so the queue only holds OPEN work while the full history
is preserved.)*
