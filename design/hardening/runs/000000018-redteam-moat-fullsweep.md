# 000000018 · red-team · post-moat FULL sweep (moat-first) → HOLES_FOUND: 1 (NEW hole P6)

*Immutable run record (write-once). This is currently the highest-numbered record, so its
`NEXT →` block below is the AUTHORITATIVE resume pointer (`../LEDGER.md` mirrors it).*

## ▶ NEXT →

```
NEXT  →  nudge-uq-fixer  on  P6  (matrix-free sloppiness iterative-path mislabel, LIM-023)
  ‖   →  nudge-uq-fixer  on  P5  (differential small mult confound, LIM-016) — already in flight
then  →  nudge-audit on each → orchestrator merge → nudge-red-team (re-scan)
STOP when nudge-red-team reports HOLES_FOUND: 0 after a genuine FULL sweep.
```

## Role / target

- **Role:** `nudge-red-team` (role 1). The **post-moat full sweep**, moat-first — the freshly-
  merged differentiability moat (`oed.py`, `sloppiness.py` matrix-free, `adjoint.ode_identifiability`)
  had never been red-teamed. Also re-confirmed the four merged round-3 fixes (P1/P2/P3/P4) for
  merge-induced regression.
- **Result:** **HOLES_FOUND: 1** → NOT a STOP. One NEW confident-wrong (**P6**) in the matrix-free
  identifiability path. OED targets + the four prior fixes HELD.
- Note: this agent ran in-place on the integration branch and committed its additive artifacts
  directly (`749954f`, report + repro only — verified no `src/`/`core/`/`fit.py`/decoy/margin).

## The NEW hole — P6 (independently re-reproduced by the orchestrator, 6/6)

`nudge.inference.sloppiness.sloppiness_diagnostic_matrixfree` / `analyze_model_matrixfree`
(`NUDGE-LIM-023`). On a model with an EXACT structural redundancy (two params enter only via
their sum ⇒ a provable Fisher zero in the `(1,…,−1)` direction ⇒ non-recoverable from ANY
data), the **iterative** path — and `method="auto"` whenever `n_params > dense_below=256`, its
raison-d'être large-network regime — returns:

```
label = 'well-constrained'   reason = '…every parameter is individually identifiable…'   n_null_dims = 0
```

while the exact dense oracle (`method="dense"` and the `jacfwd`-SVD `analyze_model`) both
return `unidentifiable (n_null=1)` and name the redundant pair. **This is the single most
dangerous mislabel: unidentifiable → well-constrained.**

Orchestrator independent repro (`uv run python scripts/redteam/sloppiness_matrixfree_iterative_mislabel.py`,
x64 ON — NOT the float32 caveat): 6/6 confident-wrong.
```
CASE 1 method='auto' (DEFAULT) n=300>256:  auto='well-constrained' cond≈1.9e2 lam_min≈184  | dense(exact)='unidentifiable' n_null=1   ×3 seeds
CASE 2 method='iterative'      n=40:        iter='well-constrained' lam_min≈1.5e3            | jacfwd-SVD oracle='unidentifiable' n_null=1 ×3 seeds
```

**Failing gate + why:** `_verified_smallest_eigsh` Rayleigh-verifies that a returned pair IS an
eigenpair — **not that it is the SMALLEST**. `eigsh(which="SA")` misses the isolated exact-zero
eigenvalue and converges to the well-conditioned cluster; those pairs are genuine eigenpairs, so
they PASS the residual check → `smallest_certified=True`, `lam_min` set to a large (wrong) value,
`cond`/`span` understated, `computed_null=0`, verdict tree → `well-constrained`. The shape-null
certificate does not apply (`n_params ≤ n_obs`). This contradicts the module's stated fail-safe
("abstains rather than assert identifiability it cannot verify").

**Not the documented bound:** the module's `NUDGE-LIM-023` caveat is about float32 downcast and
about the iterative path being unreliable for the smallest eigenvalues — but it claims the
Rayleigh verification makes it *fail-safe* (abstain, not assert). P6 shows it ASSERTS
`well-constrained`. Distinct from the honest float32 caveat (x64 was ON).

**Measured mitigation direction (fixer MEASURES + decides):** `method="dense"` (exact spectrum
via `_dense_spectrum_from_matvec`, O(n²) memory but never forms `J`/the `jacfwd` tangent
fan-out) is correct 6/6 and is feasible to far larger `n_params` than the `jacfwd` OOM point —
so `auto` can defer to the exact reconstruction well past 256, and the genuinely-huge regime
(where even O(n²) is infeasible) must **abstain** when it cannot certify the smallest eigenvalue,
per the module's own promise. Alternatively add an independent smallest-eigenvalue /
trace-completeness certificate. Expect: CLOSED where dense-reconstruction is affordable, and a
documented ABSTAIN bound in the truly-huge regime.

## Score table

| # | Target | Capability | Attack | Verdict |
|---|--------|-----------|--------|---------|
| 1 | **P6** moat | `inference/sloppiness` (`NUDGE-LIM-023`) | isolated EXACT structural null in an otherwise well-conditioned FIM → `well-constrained` | **HOLE — verified 6/6** |
| 2 | OED structural-null | `inference/oed` | target unidentifiable at every design → fake resolution? | HELD (`min_eig` honest `0.0→0.0`; abs CRLB stays large) |
| 3 | OED guarded ridge | `inference/oed` | relative-ridge understates CRLB | HELD (over-cautious in absolute terms) |
| 4 | OED demo | `service.oed_demo` / `nudge oed` | merge regression | HELD (`crlb 31.5× / min_eig 17.8×`) |
| 5 | differential P1/P4/P5 + multi_reporter P2 + design P3 | four merged fixes | merge-induced regression | HELD (54 passed / 1 skipped / 1 xfailed) |
| 6 | `ode_identifiability` isolated-null ODE | `inference/adjoint` | reach P6 through an ODE | **NOT REACHED** (analyzed, not run — not a claimed result) |

## Coverage vs skipped (budget-honest)

- Covered: the moat's `sloppiness` matrix-free path (the verified hole); OED structural-null +
  guarded-ridge + demo regression; the four merged differential/multi_reporter/design fixes.
- **Not reached (recorded so nothing reads as "fully swept"):** `adjoint.ode_identifiability`
  reached through a genuine large ODE (analyzed as the same P6 root cause, but UNRUN — not a
  claimed result); OED multi-knob design-space adversary; end-to-end OED→attribution composition.
  These remain open probe surface once P6 is fixed.

## Branch + commit (additive artifacts)

- Report: `design/FAILSAFE_REDTEAM_6.md` · Repro: `scripts/redteam/sloppiness_matrixfree_iterative_mislabel.py`
- Committed additively to the integration branch: **`749954f06149fb0923d55e268645149cdaf3552c`**
  (2 files, +376; verified no `src/`/`core/`/`fit.py`/decoy touched).
