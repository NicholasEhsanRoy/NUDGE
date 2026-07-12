# 000000025 · red-team · post-fix re-scan (STOP gate) → HOLES_FOUND: 0 → **STOP**

*Immutable run record (write-once). This is currently the highest-numbered record, so its
`NEXT →` block below is the AUTHORITATIVE resume pointer (`../LEDGER.md` mirrors it).*

## ▶ NEXT →

```
NEXT  →  orchestrator: final full gate. Fast lane GREEN (310 passed / 5 skipped / 2 xfailed);
         the COMPLETE slow lane (`pytest -m slow`) is being verified as the release gate.
         If GREEN → push claude/nudge-differentiability-hardening-ftdi3m + open PR to main
         ("hardening: post-moat red-team loop → release candidate"), recorded in a follow-on
         orchestrator record. If the slow lane surfaces a REGRESSION, it re-opens the loop
         (fix → audit → re-scan) before any release.
The red-team STOP signal is genuine (HOLES_FOUND: 0, orchestrator-verified: both unreached moat
surfaces re-run HELD). `main` untouched. No further agent dispatch unless a hole is reported.
```

## Role / result

- **Role:** `nudge-red-team` (role 1), the post-fix re-scan / STOP gate, in the main checkout on the
  integration branch (P5+P6 merged). A genuine FULL sweep covering the two surfaces the moat sweep
  (`runs/000000018`) left UNREACHED + a P5/P6 regression check + a general sweep.
- **Result:** **HOLES_FOUND: 0** → **STOP.** No new confident-wrong.

## Score (attack → verdict; all HELD)

| # | Capability | Attack | Verdict / gate that held |
|---|-----------|--------|--------------------------|
| 1 | `adjoint.ode_identifiability` (dense/auto) | isolated exact sum-null ODE, float32-default + float64 | HELD — `unidentifiable` (proportional columns cancel ~1e-16 even at float32) |
| 2 | `ode_identifiability` (genuine iterative) | same null, n_theta=17 | HELD — P6 inverse-iteration probe catches it; path can ONLY return `unidentifiable` |
| 3 | `ode_identifiability` | diffuse rank-deficient gLV | HELD — `unidentifiable` (17–29 nulls) |
| 4 | `ode_identifiability` | non-proportional 3-column null; x64-off default | HELD — cancels ~3e-16; x64-off warns visibly |
| 5 | `service.oed_demo` / `nudge oed` | 144-config regression / masking sweep | HELD — 0/144 (no `crlb_improvement<1`, no masked CRLB) |
| 6 | `oed.optimize_design` | last-iterate honesty (LR 0.5–20 overshoot) | HELD — reported == measured ratio, no false >1 |
| 7 | `oed.crlb` / `OEDResult` | guarded-ridge masking a structural singularity | HELD — raw `min_eig≈0` + absolute CRLB≈1e4 self-flag it |
| 8 | `differential` gate 4d | fresh NON-affine confound (perturbed-only overdispersion) | HELD — `no-difference` (shared model absorbs it; gate 4d not even reached) |
| 9 | P5/P6 + prior-fix regression | merge-induced regression | HELD — fast lane 310p/2xf/0fail; P6 repro 0/6; slow differential 29p/1xf; slow P6 3p/1xf |

## Independent verification (orchestrator — did not trust the red-team's report)

- **Commit `54204e0` additive:** `design/FAILSAFE_REDTEAM_7.md` + 2 repro scripts only; no `src/`,
  `fit.py`, `core/`, decoy, or margin. Frozen core `git diff origin/main..HEAD -- fit.py core/` = empty.
- **Re-ran `scripts/redteam/ode_identifiability_float32_null_rescan.py`:** exit 0 — **0/12
  confident-wrong**; every isolated/diffuse/iterative null ODE → `unidentifiable` at float32-DEFAULT
  and float64. The P6 fix holds END-TO-END through `adjoint.ode_identifiability`.
- **Re-ran `scripts/redteam/oed_shipped_demo_and_masking_rescan.py`:** exit 0 — [1] 144-config
  `oed_demo` 0 confident-wrong; [2] `optimize_design` last-iterate honesty 0 false-improvement;
  [3] guarded-ridge masking 0 deceptive (raw `min_eig` + absolute CRLB self-flag any degeneracy).

## Coverage honesty (recorded — nothing reads as "fully swept" that wasn't)

- **Covered:** both UNREACHED moat surfaces end-to-end; P5/P6 regression; a fresh non-affine
  differential attack; the global fast lane (310 passed).
- **NOT fully swept (explicit):** the full slow lane per-capability (only P5/P6-relevant slow suites
  run by the agent — the orchestrator runs the COMPLETE slow lane as the release gate, see the merge
  step); cross-capability composition into `design()` was *analyzed* (bounded by upstream fail-safes),
  not exhaustively driven — NOT a claimed no-hole result; a maximally-adversarial
  different-computation-path float32 null was *analyzed*, not constructed. HELD results do not prove
  no other attack exists.

## Branch + commit

- Report: `design/FAILSAFE_REDTEAM_7.md` · Repros: `scripts/redteam/ode_identifiability_float32_null_rescan.py`,
  `scripts/redteam/oed_shipped_demo_and_masking_rescan.py`
- Committed additively to the integration branch: **`54204e0`** (3 files, +440; no `src/` touched).
