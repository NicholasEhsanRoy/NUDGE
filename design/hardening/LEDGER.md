# NUDGE hardening loop — LEDGER (live state · resume here)

**Governing rule: honesty above all** — never claim more than measured; abstain over
confident-wrong; document residual bounds loudly. See `README.md` for the protocol.

---

## ▶ RESUME POINTER

*(Mirror of the `NEXT →` block in the highest-numbered `runs/` record — currently
`runs/000000018-audit-P5-FAIL.md`. That immutable record is the source of truth;
this is a convenience copy. See `README.md` → "The resume pointer & the queue".)*

**Status: RUNNING — P5 fix FAILED audit, back to the fixer.** P3/P1/P4/P2 CLOSED/BOUNDED +
merged. P5 (differential small multiplicative confound) is still **OPEN**: the first fix
attempt (`779fc3a`) **FAILED audit** (`runs/000000018`) — it validated on only 2 seeds while
the repro's default is 4, at which the hole still yields 3 confident-wrong; its "gap
[1.184, 1.231]" claim was independently falsified (confident confounds at off_scale
1.13–1.18), and the 1.20 cut is too loose for the default red-team regime (genuine-ceiling
max ~1.104 there). The failed fix is **NOT merged**.

- **Next agent:** `nudge-uq-fixer` on **P5 again** — tighten the cut to a measured margin
  above the genuine-increase max (regime-aware: 1.104 RT vs 1.184 TEST), ABSTAIN across the
  confounded band rather than resolve, retract the falsified numbers, re-validate ≥4 seeds.
- **STOP** when `nudge-red-team` reports `HOLES_FOUND: 0` after a genuine FULL sweep.
- **Recorded future candidate** (P4 audit, out-of-scope): a pre-existing gain⇄ceiling-
  *reduction* mis-attribution degeneracy in `differential`, unaffected by P4 — a possible
  later red-team target, not yet a queued hole.

---

## Problem queue (found, not yet fixed)

| id | capability | LIM | summary | repro | status |
|----|-----------|-----|---------|-------|--------|
| **P5** | `differential` | LIM-016 | **NEW (final sweep, `runs/000000017`).** A SMALL uniform multiplicative perturbed-only scale (`c≈1.15–1.25`, control clean) fakes a confident `gain-diff` / `ceiling-diff` (truth no-difference). Slips the P4 gate 4c two ways: (1) at small `c` the BIC winner is often **gain (n)**, and gate 4c is ceiling-scoped (never checks `off_scale` for a gain winner); (2) on the ceiling channel `off_scale` lands in the **(1.18, 1.30] blind gap** — above the measured genuine-ceiling max (1.18) yet below gate 4c's upper cut (1.30). The P4 fix was calibrated only on `c≥1.5`; the interior was unprobed. Verified 8 confident-wrong / 4 seeds. | `scripts/redteam/differential_small_mult_gain_hole.py` | OPEN |

Reported by red-team round 3 (`design/FAILSAFE_REDTEAM_3.md`); **pending independent UQ
validation** (role 3 re-reproduces before fixing — status reflects that they are red-team
claims, not yet main-loop-verified).

| id | capability | LIM | summary | repro | status |
|----|-----------|-----|---------|-------|--------|
*(no OPEN rows — P3/P1/P4/P2 are all in "Closed problems" below.)*

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
| 000000009 | `runs/000000009-redteam-P1rescan.md` | redteam | P1-fix re-scan | **HOLES_FOUND: 1** — NEW hole **P4** (differential multiplicative confound); P1 additive fix held; commit `63516d4` |
| 000000010 | `runs/000000010-uq-fixer-P4.md` | uq-fixer | P4 (LIM-016) | fix claim: measured ceiling-scoped OFF-cluster-scale gate 4c band [0.80,1.30]; inflation CLOSED / deflation BOUNDED; commit `f5d0b87` |
| 000000011 | `runs/000000011-audit-P4.md` | audit | P4 fix | **AUDIT: PASS** — hole gone both directions, genuine ceiling increases resolve, reduction-sacrifice narrow+honest, frozen core untouched |
| 000000012 | `runs/000000012-orchestrator-P4-merge.md` | orchestrator | P4 merge | independently re-verified + merged → `ebda9c6` (3 conflicts resolved); P4 CLOSED/BOUNDED |
| 000000013 | `runs/000000013-redteam-P4rescan.md` | redteam | P4-fix re-scan | **HOLES_FOUND: 0** — both differential gates held (above-median evader = degenerate-with-genuine, not a hole); orchestrator tightened "INFLATION CLOSED" wording; commit `47a8400` |
| 000000014 | `runs/000000014-uq-fixer-P2.md` | uq-fixer | P2 (LIM-014) | fix claim: ceiling-scoped floor/OFF-consistency gate (`off_on_coupling` ≈0 genuine vs ≈1 batch); CLOSED measurable / BOUNDED near-zero floor; commit `b870354` |
| 000000015 | `runs/000000015-audit-P2.md` | audit | P2 fix | **AUDIT: PASS** — hole gone (seeds 0,1,2), genuine ceiling resolves 4/4, near-zero-floor bound narrow+honest, frozen core untouched |
| 000000016 | `runs/000000016-orchestrator-P2-merge.md` | orchestrator | P2 merge | independently re-verified + merged → `1d091c1` (2 additive doc conflicts resolved); P2 CLOSED/BOUNDED; queue now EMPTY |
| 000000017 | `runs/000000017-redteam-finalsweep.md` | redteam | FINAL full sweep | **HOLES_FOUND: 1** — NOT a STOP: NEW hole **P5** (differential small mult confound slips P4 gate 4c); 4 fixes held jointly; report `FAILSAFE_REDTEAM_5.md`; commit `6a1c774` |
| 000000018 | `runs/000000018-audit-P5-FAIL.md` | audit | P5 fix (`779fc3a`) | **AUDIT: FAIL** — hole NOT closed (HOLES: 3 at the repro's default 4 seeds; fixer validated only 2); "gap [1.184,1.231]" claim falsified; cut 1.20 too loose (RT genuine max ~1.104). NOT merged → back to fixer |

---

## Closed problems (no-delete: fixed problems move here, rows never deleted)

| id | capability | LIM | resolution | fix commit | audit |
|----|-----------|-----|------------|-----------|-------|
| **P3** | `design/invert` | LIM-013 | **CLOSED** — safety gate now fires on `delta > margin` OR absolute `proximity_after ≥ NEAR_FOLD` (reuses the shipped constant → agrees with `classify_robustness` on the identical circuit); one-sided caveat carried on the SAFE branch. Passing decoy + 3 regression tests. 0 confident-wrong; positive control still "OK". | `a263789` (merge `017bd58`) | `runs/000000003-audit-P3.md` (PASS) |
| **P1** | `differential` | LIM-016 | **CLOSED (inflating) / BOUNDED (deflating)** — measured one-sided gate 4b abstains when a context's perturbed OFF baseline is inflated >2.5× its own control (separator: confident-wrong `off_shift≥2.99` vs genuine `≤1.96`). Deflating perturbed-only offset aliases with a genuine reduction → documented residual, not guarded. Decoy + tests; genuine ceiling/gain still resolve (no over-abstention). | `b562da9` (merge `99d73b8`) | `runs/000000007-audit-P1.md` (PASS) |
| **P4** | `differential` | LIM-016 | **CLOSED (inflating) / BOUNDED (deflating)** — the MULTIPLICATIVE sibling of P1: a per-context multiplicative perturbed-only scale fakes a confident `ceiling-diff`, slipping under gate 2 AND the P1 gate 4b (`off_shift`≈1 for a multiplicative factor). Measured ceiling-scoped gate 4c on the OFF-cluster SCALE (raw MAD, perturbed÷control), band [0.80,1.30] (genuine ceiling ×1.4–4 → ≤1.18; inflating confound c≥1.5 → ≥1.43). Deflating confound is degenerate with a genuine strong ceiling reduction → both abstain (a strict-xfail-locked honest bound; the narrow sacrifice verified by the audit). Genuine ceiling increases + gain/threshold still resolve. | `f5d0b87` (merge `ebda9c6`) | `runs/000000011-audit-P4.md` (PASS) |
| **P2** | `multi_reporter` | LIM-014 | **CLOSED (measurable floors) / BOUNDED (near-zero floors)** — a per-condition multiplicative batch scale on the perturbed panel fakes a confident `ceiling` (control-only consistency guard blind; no per-condition depth normalization). Measured ceiling-scoped floor/OFF-consistency gate: a genuine ceiling leaves each reporter's floor fixed (`off_on_coupling`≈0) while a batch rescales every floor with the ON amplitude (≈1); abstain when the floor moves with the ceiling (cut 0.5 = physical midpoint) or floors are unmeasurable. On a (near-)zero-floor panel a batch ≡ a real ceiling without an independent anchor → both abstain (strict-xfail-locked honest bound). Genuine ceiling/threshold/gain still resolve. | `b870354` (merge `1d091c1`) | `runs/000000015-audit-P2.md` (PASS) |

*(P1/P2 remain OPEN in the queue above. When each passes audit its row moves here with the
fix commit + the audit run record, so the queue only holds OPEN work while the full history
is preserved.)*
