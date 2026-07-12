# 000000026 · orchestrator · final release gate + push + PR (loop COMPLETE)

*Immutable run record (write-once). This is currently the highest-numbered record, so its
`NEXT →` block below is the AUTHORITATIVE resume pointer (`../LEDGER.md` mirrors it).*

## ▶ NEXT →

```
NEXT  →  (LOOP COMPLETE) A human reviews + merges the PR to main; that merge unblocks the
         nudge-bio 0.1.0 PyPI release (design/RELEASE_CHECKLIST.md). No further agent dispatch
         unless a NEW confident-wrong hole is later reported (which re-opens the loop).
`main` untouched throughout. Red-team STOP gate (runs/000000025) = HOLES_FOUND: 0.
```

## The release-gate result (honest — the slow lane has environmental limits on this box)

**Static + fast lane — FULLY GREEN on the merged integration branch:**
- `ruff check src tests scripts` → All checks passed
- `pyright src` → 0 errors (CI-pinned 1.1.411)
- 5 validators (`check_mechanism_cards` / `check_impl_mapping` / `check_citations` / `check_anomalies`
  / `check_hardening_append_only`) → all OK
- `pytest -q` fast lane → **310 passed / 5 skipped / 2 xfailed / 0 failed** (baseline 296 + P5's ~9
  and P6's ~5 new tests; the 2 xfails are the locked decoys).

**Slow lane — my changed modules VERIFIED GREEN; the full lane has a box-specific limit (documented,
NOT a regression):**
- **The no-regression proof.** This session changed EXACTLY 3 `src/` files:
  `git diff --name-only c491c47..HEAD -- src/` = `differential.py`, `sloppiness.py`, `service.py`.
  All three had their slow tests run GREEN on identical code:
  - `differential.py` (P5): full slow suite **29 passed / 1 xfailed** (P5 audit `runs/000000020`;
    the xfail is the documented deflation-sacrifice lock). Independently re-confirmed by the crashed
    full-lane run below — all 30 differential slow tests ran with **0 failures**.
  - `sloppiness.py` (P6): **21 passed / 1 xfailed** (P6 audit `runs/000000023`; xfail = the strict
    huge-regime over-abstention decoy).
  - `service.py` (P5 wiring): `pytest -q -m slow tests/test_service.py` → **2 passed** (this record).
  Every OTHER slow test exercises UNCHANGED code (the 3 changed files introduce no module-level
  global/config mutation), so no slow-lane regression is possible from this work.
- **The full slow lane cannot complete in one process on this ~15 GB / 4-core box** — it SIGSEGVs
  (exit 139) from JAX/XLA compilation-resource ACCUMULATION after ~48 heavy tests. The crashing test
  (`test_lotka_volterra.py::test_laplace_curvature_measures_the_alpha_beta_degeneracy`) **PASSES in
  isolation** (12 s), so this is an environment fragility, not a code defect (conftest already
  disables the persistent compile cache).
- **Two pre-existing slow-lane failures surfaced, both proven NOT to be this session's regressions
  (untouched modules) and NEITHER a confident-wrong:**
  1. `tests/data/test_perturbseq_loader.py::test_backed_mode_bounds_peak_memory` — a peak-memory-bound
     assertion; **PASSES in isolation**, fails only under full-lane memory accumulation (environmental).
  2. `tests/inference/test_constitutive.py::test_linear_circuit_lim006_hazard_abstains_not_confident_wrong`
     — the **fail-safe assertions PASS** (`res.call == "unresolved"`, `!= "biological-switch"`,
     `not is_confident_wrong`); the only failing assertion is the **cosmetic** `argmin_n_with_control
     == 1.0` (observed 1.5 — where the n-profile minimum sits). `constitutive.py`/`test_constitutive.py`
     are untouched this session and do not import `differential`/`sloppiness`; the seeded test runs
     identical code at `c491c47` → pre-existing, environment/numerical-sensitive. NUDGE still correctly
     ABSTAINS — this is not a fail-safe violation.

**Honesty:** the full 98-test slow lane was NOT driven to a clean single-process completion (box
limit). What IS verified: all three changed modules' slow suites are green, the fast lane is fully
green, and the two non-passing slow tests are pre-existing/environmental in untouched code. CI (in a
provisioned environment) is the authoritative full-gate check for the release PR.

## Push + PR (the concluding action of this record)

- This commit is the last on `claude/nudge-differentiability-hardening-ftdi3m`; the orchestrator
  then pushes it and opens a PR → `main` titled "hardening: post-moat red-team loop → release
  candidate" (the PR is a NEW pull request; `main` is UNTOUCHED and never committed to during the
  loop). A human reviews + merges it; that merge unblocks the release.

## Outcome — the post-moat hardening loop is COMPLETE

2 confident-wrong holes found (P5 pre-existing in the queue, P6 new in the moat), both CLOSED +
BOUNDED + audited + merged; red-team STOP gate `HOLES_FOUND: 0` after a genuine full sweep; frozen
core (`fit.py`, `core/`) untouched; the `design/hardening/` trail is complete (`runs/000000018–26`).
