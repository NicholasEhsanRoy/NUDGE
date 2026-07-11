# NUDGE hardening loop — LEDGER (live state · resume here)

**Governing rule: honesty above all** — never claim more than measured; abstain over
confident-wrong; document residual bounds loudly. See `README.md` for the protocol.

---

## ▶ RESUME POINTER

*(Mirror of the `NEXT →` block in the highest-numbered `runs/` record — currently
`runs/000000024-orchestrator-P6-merge.md`. That immutable record is the source of truth;
this is a convenience copy. See `README.md` → "The resume pointer & the queue".)*

**Status: RUNNING (POST-MOAT hardening loop) — queue EMPTY, at the STOP gate.** The moat-first
full sweep (`runs/000000018`) found **P6** (matrix-free `sloppiness` mislabel, `LIM-023`); the
pre-existing **P5** (differential small mult confound, `LIM-016`) was also queued. **Both are now
CLOSED + merged:** P5 (gate-4d free-affine earn guard; audit PASS `runs/000000020`; merge
`480468c`); P6 (`dense_below=2048` deferral + inverse-iteration null probe; audit PASS
`runs/000000023`; merge `3cd1f41` + honesty correction `ed3c381`).

- **Next:** `nudge-red-team` **re-scan** — the loop's STOP gate. It MUST cover:
  (a) a P5+P6 fix-induced-regression check; and (b) the two surfaces the moat sweep left UNREACHED
  (`runs/000000018`): `adjoint.ode_identifiability` reached through a real large ODE (same P6 root
  cause), and OED multi-knob design-space / end-to-end composition.
- **STOP** when `nudge-red-team` reports `HOLES_FOUND: 0` after a genuine FULL sweep.
- **HELD this sweep (recorded as fail-safe wins):** OED structural-null (`min_eig` honest
  `0.0→0.0`), OED guarded ridge (over-cautious absolute CRLB), OED demo (no merge regression),
  and all four round-3 fixes (P1/P2/P3/P4) — no merge-induced regression.
- **Recorded future candidate** (P4 audit, out-of-scope): a pre-existing gain⇄ceiling-
  *reduction* mis-attribution degeneracy in `differential`, unaffected by P4 — a possible
  later red-team target, not yet a queued hole.

---

## Problem queue (found, not yet fixed)

| id | capability | LIM | summary | repro | status |
|----|-----------|-----|---------|-------|--------|
*(no OPEN or IN-PROGRESS rows — **P5 and P6 are both CLOSED + merged**; see "Closed problems"
below. The queue is EMPTY. The loop is at the STOP gate: a `nudge-red-team` re-scan must return
`HOLES_FOUND: 0` after a genuine full sweep — covering the P5+P6 regression AND the two moat
surfaces left UNREACHED by `runs/000000018` — before the loop STOPS.)*

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
| 000000018 | `runs/000000018-redteam-moat-fullsweep.md` | redteam | POST-MOAT full sweep (moat-first) | **HOLES_FOUND: 1** — NOT a STOP: NEW hole **P6** (matrix-free `sloppiness` iterative/`auto` path mislabels an unidentifiable model `well-constrained`, LIM-023); OED targets + 4 prior fixes HELD; report `FAILSAFE_REDTEAM_6.md`; commit `749954f` |
| 000000019 | `runs/000000019-uq-fixer-P5.md` | uq-fixer | P5 (LIM-016) | fix claim: gate-4d free per-condition affine "earn" guard closes the whole uniform P1/P4/P5 affine class (margin 6.0 on a global ΔBIC); CLOSED-uniform / BOUNDED-nonuniform; corrects P4 overclaim; commit `416c17e` (fixer stalled pre-commit; orchestrator finalized the commit) |
| 000000020 | `runs/000000020-audit-P5.md` | audit | P5 fix | **AUDIT: PASS** — hole abstains (fixer seeds 0,1 + independent 5,7 → HOLES 0; earn −7.5…−6.9), positive controls resolve (slow 29 passed/1 xfailed), honesty accurate (minor genuine-earn label nit → orchestrator corrected), frozen core untouched, full gate green (301 passed) |
| 000000021 | `runs/000000021-orchestrator-P5-merge.md` | orchestrator | P5 merge | independently re-verified + merged → `480468c` (clean ort); corrected FINDINGS §P5 genuine-earn label additively; P5 CLOSED/BOUNDED |
| 000000022 | `runs/000000022-uq-fixer-P6.md` | uq-fixer | P6 (LIM-023) | fix claim: `auto` defers to exact reconstruction up to measured `dense_below=2048` + inverse-iteration one-sided null probe (catches the isolated zero `eigsh` misses; else abstain); trace-completeness rejected; CLOSED-mislabel / BOUNDED-over-abstention; commit `2d54b57` (1st dispatch was wrong-base via harness isolation → orchestrator caught + re-dispatched) |
| 000000023 | `runs/000000023-audit-P6.md` | audit | P6 fix | **AUDIT: PASS** — hole closed (0/6 exit=1; independent seeds 7,13,42; slow probe test), BOUNDED residual structurally fail-safe (n=2100 auto→abstain, dense→well-constrained; decoy xfails), no over-abstention, threshold measured (n=2000 11.8s/0.75GB), pinned-pyright 0, fast-lane +9 tests (0 pre-existing fail); flagged the ~1e-19 probe-RQ overclaim (non-blocking) |
| 000000024 | `runs/000000024-orchestrator-P6-merge.md` | orchestrator | P6 merge | independently re-measured probe RQ (~1e-12…1e-10, confirming ~1e-19 non-reproducing) + merged → `3cd1f41` (clean ort) + honesty correction `ed3c381`; P6 CLOSED/BOUNDED; queue now EMPTY |

---

## Closed problems (no-delete: fixed problems move here, rows never deleted)

| id | capability | LIM | resolution | fix commit | audit |
|----|-----------|-----|------------|-----------|-------|
| **P3** | `design/invert` | LIM-013 | **CLOSED** — safety gate now fires on `delta > margin` OR absolute `proximity_after ≥ NEAR_FOLD` (reuses the shipped constant → agrees with `classify_robustness` on the identical circuit); one-sided caveat carried on the SAFE branch. Passing decoy + 3 regression tests. 0 confident-wrong; positive control still "OK". | `a263789` (merge `017bd58`) | `runs/000000003-audit-P3.md` (PASS) |
| **P1** | `differential` | LIM-016 | **CLOSED (inflating) / BOUNDED (deflating)** — measured one-sided gate 4b abstains when a context's perturbed OFF baseline is inflated >2.5× its own control (separator: confident-wrong `off_shift≥2.99` vs genuine `≤1.96`). Deflating perturbed-only offset aliases with a genuine reduction → documented residual, not guarded. Decoy + tests; genuine ceiling/gain still resolve (no over-abstention). | `b562da9` (merge `99d73b8`) | `runs/000000007-audit-P1.md` (PASS) |
| **P4** | `differential` | LIM-016 | **CLOSED (inflating) / BOUNDED (deflating)** — the MULTIPLICATIVE sibling of P1: a per-context multiplicative perturbed-only scale fakes a confident `ceiling-diff`, slipping under gate 2 AND the P1 gate 4b (`off_shift`≈1 for a multiplicative factor). Measured ceiling-scoped gate 4c on the OFF-cluster SCALE (raw MAD, perturbed÷control), band [0.80,1.30] (genuine ceiling ×1.4–4 → ≤1.18; inflating confound c≥1.5 → ≥1.43). Deflating confound is degenerate with a genuine strong ceiling reduction → both abstain (a strict-xfail-locked honest bound; the narrow sacrifice verified by the audit). Genuine ceiling increases + gain/threshold still resolve. | `f5d0b87` (merge `ebda9c6`) | `runs/000000011-audit-P4.md` (PASS) |
| **P2** | `multi_reporter` | LIM-014 | **CLOSED (measurable floors) / BOUNDED (near-zero floors)** — a per-condition multiplicative batch scale on the perturbed panel fakes a confident `ceiling` (control-only consistency guard blind; no per-condition depth normalization). Measured ceiling-scoped floor/OFF-consistency gate: a genuine ceiling leaves each reporter's floor fixed (`off_on_coupling`≈0) while a batch rescales every floor with the ON amplitude (≈1); abstain when the floor moves with the ceiling (cut 0.5 = physical midpoint) or floors are unmeasurable. On a (near-)zero-floor panel a batch ≡ a real ceiling without an independent anchor → both abstain (strict-xfail-locked honest bound). Genuine ceiling/threshold/gain still resolve. | `b870354` (merge `1d091c1`) | `runs/000000015-audit-P2.md` (PASS) |
| **P5** | `differential` | LIM-016 | **CLOSED (uniform affine class) / BOUNDED (non-uniform + deflation)** — a SMALL uniform multiplicative perturbed-only scale (`c≈1.15–1.25`) faked a confident `gain-diff`/`ceiling-diff` slipping gate 4c (ceiling-scoped → silent on the `n` winner; `off_scale` in the `(1.18,1.30]` gap). Fixed by **gate 4d**: instead of a fourth per-magnitude band, add the per-condition affine `(s,o)` as a FREE nuisance on the perturbed context and abstain unless the BIC-winning knob **earns** its parameter over a pure-affine null by ≥ 6.0 (profiled ΔBIC, min over both directions). The whole confound family (P1 additive / P4 large-mult / P5 small-mult) is by construction inside the null's span → measured `earn` −7.5…−2.1 (confound) vs +59…+616 (genuine); margin 6.0 is a single cut on a GLOBAL fit statistic, no blind gap. Corrects the falsified P4 "INFLATION CLOSED" overclaim. Residual BOUND: a NON-uniform above-median-only scale is observationally identical to a genuine ceiling (needs an inert-feature anchor); gate-4c deflation sacrifice unchanged. | `416c17e` (merge `480468c`) | `runs/000000020-audit-P5.md` (PASS) |
| **P6** | `sloppiness` (matrix-free) | LIM-023 | **CLOSED (isolated-null mislabel) / BOUNDED (huge-regime over-abstention)** — the iterative/`method="auto"` Krylov path certified a structurally-UNIDENTIFIABLE model (two params via their sum ⇒ a provable Fisher zero) **`well-constrained`**, because `_verified_smallest_eigsh` Rayleigh-checks eigenpair-*ness* not smallest-*ness* and `eigsh('SA')` misses the isolated zero. Fixed by (1) `auto` deferring to the exact dense-via-matvec reconstruction up to a MEASURED `dense_below=2048` (n=2000 = 11.8 s / 0.75 GB, recovers the null at every size 256–4096), and (2) an inverse-iteration null probe (shift-invert `(FIM+εI)⁻¹` via CG) that amplifies the smallest eigenvalue → catches the isolated null (measured probe RQ ~1e-12…1e-10 ≪ the rank floor ~3e-10; 0 false nulls on controls), used ONE-SIDED (a low RQ proves a null; when none is found the path ABSTAINS). Trace-completeness rejected (blind to a missed zero). Residual BOUND: above `dense_below` a genuine well-constrained model OVER-ABSTAINS (fail-safe, never confident-wrong; exact verdict via `method="dense"`), locked by a strict-xfail decoy. `NUDGE-LIM-023` → severity major; the ~1e-19 probe-RQ overclaim re-measured + corrected to ~1e-12…1e-10. | `2d54b57` (merge `3cd1f41`, honesty fix `ed3c381`) | `runs/000000023-audit-P6.md` (PASS) |

*(P1/P2 remain OPEN in the queue above. When each passes audit its row moves here with the
fix commit + the audit run record, so the queue only holds OPEN work while the full history
is preserved.)*
