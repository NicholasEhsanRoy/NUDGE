# NUDGE fail-safe red-team — ROUND 6 (moat-first full sweep; run `000000018`)

**Mandate.** Same as rounds 1–5: adversarially force a NUDGE capability to emit a *confident,
specific, WRONG* call where the honest answer is abstention, past the shipped gates. A verified
hole is a WIN. **This document reports and reproduces; it does NOT fix.** No `src/` capability
code, `fit.py`, `core/`, the decoy battery, or any fail-safe margin was touched — only this
report + one new `scripts/redteam/*.py` were added.

This round spent the HEAVY budget on the **freshly-merged differentiability MOAT** (never
red-teamed, least-hardened): gradient OED (`inference/oed.py`), matrix-free identifiability
(`inference/sloppiness.py`, `inference/adjoint.py::ode_identifiability`), their wiring, then a
regression pass. **Result: `HOLES_FOUND: 1`** — a NEW confident-wrong in the matrix-free
identifiability path (`NUDGE-LIM-023`) that lands squarely on the "one dangerous mislabel" the
mandate flagged. P5 (differential small-mult) was NOT re-hunted (owned by the concurrent fix
loop); the regression pass re-confirms the four merged differential/multi-reporter/design fixes
still hold jointly.

Repro (lint-clean, ruff line-length 100; `uv run`) reproduces the confident-wrong
deterministically across **6/6 seed·size cells** through the SHIPPED public API.

## Score

| # | Target | Capability | Attack | Verdict |
|---|--------|-----------|--------|---------|
| 1 | sloppiness matrix-free **P6** | `inference/sloppiness` (`NUDGE-LIM-023`) | isolated EXACT structural null (2 params via their sum) in an otherwise well-conditioned FIM → iterative/`auto` Krylov path labels it **`well-constrained`** ("every parameter individually identifiable") | **HOLE — verified (6/6 seeds; `method="auto"` default + `method="iterative"`)** |
| 2 | OED structural-null CRLB | `inference/oed` (`NUDGE-LIM-024`) | target parameter structurally unidentifiable at every design → does `optimize_design` fake a resolution? | **HELD** — `min_eig` channel honestly reports `0.0→0.0`; abs. CRLB stays large (~154, std≈12); the residual `1.84×` ridge inflation is contradicted by `min_eig` (not a clean confident-wrong) |
| 3 | OED guarded ridge | `inference/oed` (`crlb`) | relative-ridge understates a flat direction's CRLB when trace is inflated | **HELD** (bounded) — ridge floor is over-cautious in absolute terms; no "falsely small CRLB → resolved" output |
| 4 | OED demo regression | `service.oed_demo` / `nudge oed` | headline gain drifted / regressed under the merge | **HELD** — logistic `crlb 31.5× / min_eig 17.8×` (matches documented headline) |
| 5 | differential P1/P4/P5 + multi_reporter P2 + design P3 | four merged fixes | merge-induced regression of the shipped guards | **HELD** — 54 passed / 1 skipped / 1 xfailed |
| 6 | `ode_identifiability` end-to-end isolated-null ODE | `inference/adjoint` | reach P6 through an ODE (not a linear map) | **NOT REACHED** (analyzed; an isolated exact null is hard to engineer inside a gLV/pathway ODE — the hole is demonstrated on the primary `analyze_model_matrixfree` public API instead) |

**Confident-wrong holes found: 1 verified (P6).** It extends the round-5 systemic insight in a
new direction: not a confound on the *perturbed* side, but a **guard that verifies the wrong
property** — the smallest-eigenvalue "fail-safe" checks *eigenpair-ness*, not *smallest-ness*,
so it certifies identifiability it never actually verified. The confound here is *numerical*
(a Krylov solver silently missing an isolated null), landing on the exact channel the module's
own docstring claims to protect.

---

## HOLE P6 — the matrix-free identifiability MOAT confidently mislabels a structurally-UNIDENTIFIABLE model `well-constrained`

**Capability:** `nudge.inference.sloppiness.sloppiness_diagnostic_matrixfree` /
`analyze_model_matrixfree` (`NUDGE-METHOD` scaling layer, `NUDGE-LIM-023`); the same path is
re-exported end-to-end by `nudge.inference.adjoint.ode_identifiability`.

**Repro:** `scripts/redteam/sloppiness_matrixfree_iterative_mislabel.py`
(`uv run python scripts/redteam/sloppiness_matrixfree_iterative_mislabel.py`, ~1 min, exits 0 =
hole reproduced). Discovery scratch: the exact-vs-near-dup robustness sweep + the root-cause
instrumentation are folded into the repro's docstring.

**The claim under attack.** `NUDGE-LIM-023` (STATE.md §Efficiency; the `sloppiness.py`
docstring) is the moat's fail-safe promise: the matrix-free Krylov path "is reliable for the
LARGEST FIM eigenvalues but NOT the smallest … so the iterative path certifies `unidentifiable`
via shape rank deficiency (`n_params > n_obs`), Rayleigh-residual-verifies any smallest
eigenpair, and **abstains rather than assert identifiability it cannot verify**." The
`sloppiness_diagnostic_matrixfree` docstring is even more explicit — it *names this exact
failure* and claims two fail-safes handle it:

> "The **smallest** eigenvalues of an ill-conditioned FIM are NOT reliably reachable by a
> matrix-free Krylov method (measured: `eigsh`/LOBPCG can return a large eigenvalue as
> "smallest" and **mislabel a rank-deficient model well-constrained**), so the smallest end is
> handled fail-safe: [`n_params > n_obs` ⇒ shape null ⇒ unidentifiable] … otherwise … the
> smallest eigenpairs are … **Rayleigh-residual verified**; if none converge, the diagnostic
> **abstains**."

**The attack — an isolated EXACT null with `n_params ≤ n_obs`.** Take a well-conditioned linear
observation map `y = M·θ` (`M`'s columns unit-normalized → a tight non-null spectrum) and set
`M`'s LAST column equal to its FIRST. Now parameters `p0` and `p{last}` enter the observation
only through `p0 + p{last}` — the `(1, 0, …, 0, −1)` direction is a **genuine zero** of the
Fisher information; those two parameters are provably non-recoverable from any amount of data.
This is exactly the structural non-identifiability the module's *own* validation model
`redundant_exponential_predict` (`A·e^{−(k₁+k₂)t}`) represents — just embedded in an otherwise
well-conditioned spectrum, and scaled so `method="auto"` selects the iterative path
(`n_params > dense_below=256`). Both shipped fail-safes miss it:

- **The shape-null certificate does not apply** (`n_params=300 ≤ n_obs=400`, or `40 ≤ 100`).
- **The Rayleigh verification is the wrong check.** `eigsh(which="SA")` converges to the
  well-conditioned CLUSTER and **misses the isolated ~0 eigenvalue**. The pairs it returns are
  *genuine* eigenpairs, so they PASS the Rayleigh residual (which verifies *eigenpair-ness*, not
  *smallest-ness*). Hence `smallest_certified=True`, `lam_min` is set to a large wrong value,
  `cond`/`span` are understated, `computed_null=0`, and the verdict tree lands on
  `well-constrained`.

**Confident-wrong output (verified, 6/6 across seeds; truth = `unidentifiable`):**

```
CASE 1 — method='auto' (DEFAULT), n_params=300 > dense_below=256  → auto picks iterative
 seed=0: matrix-free(auto)='well-constrained' n_null=0 cond=1.94e+02 lam_min=1.84e+02  |  dense(exact)='unidentifiable' n_null=1
 seed=1: matrix-free(auto)='well-constrained' n_null=0 cond=1.80e+02 lam_min=1.94e+02  |  dense(exact)='unidentifiable' n_null=1
 seed=2: matrix-free(auto)='well-constrained' n_null=0 cond=1.80e+02 lam_min=1.92e+02  |  dense(exact)='unidentifiable' n_null=1
    reason: "WELL-CONSTRAINED: the Fisher spectrum spans only 2.3 decades (cond 1.94e+02);
             every parameter is individually identifiable …"
CASE 2 — method='iterative' (explicit), n_params=40, vs analyze_model (jacfwd-SVD oracle)
 seed=0: matrix-free(iter)='well-constrained' n_null=0 lam_min=1.66e+03  |  dense-oracle='unidentifiable' n_null=1
 seed=1: matrix-free(iter)='well-constrained' n_null=0 lam_min=1.79e+03  |  dense-oracle='unidentifiable' n_null=1
 seed=2: matrix-free(iter)='well-constrained' n_null=0 lam_min=1.30e+03  |  dense-oracle='unidentifiable' n_null=1
```

The emitted call is the **strongest possible identifiability assertion** — *"every parameter is
individually identifiable"* — on a model with an **exact** structural null. The exact
dense-via-matvec path (`method="dense"`) and the dense `jacfwd`-SVD oracle (`analyze_model`)
both correctly return `unidentifiable` (`n_null=1`) on the identical model — so this is
specifically an **`auto`/`iterative`-path** confident-wrong, not a model artifact.

**Root cause (instrumented, measured).** For the `n_params=40, seed=0` model the true FIM
spectrum is `[−3.2e-12, 1661, 1932, …, 29458]` (one exact zero + a tight well-conditioned
cluster). `eigsh(which="SA")` returns as its six "smallest" `[1661, 1932, 2292, 2540, 3118,
3347]` — it **skips the ~0 entirely** — and every one passes the Rayleigh residual, so
`_verified_smallest_eigsh` reports `smallest_certified=True`. The gate then trusts
`lam_min=1661`, computes `span < 3 decades`, finds `computed_null=0`, and classifies
`well-constrained`.

**Which gate failed, and why this is NOT a documented bound.**
- **The `_verified_smallest_eigsh` fail-safe verifies the wrong property.** It checks each
  returned pair `(λ, v)` for `‖FIM·v − λv‖/λ_max < res_tol` — i.e. "is this an eigenpair" — and
  treats `converged=True` as license to classify. It never checks that the returned set actually
  *contains* the smallest eigenvalue. A Krylov solver that misses an isolated null therefore
  produces a *certified* but *understated* `λ_min`.
- **This contradicts the module's own reliability claim.** The docstring names "mislabel a
  rank-deficient model well-constrained" as the failure it protects against, and asserts it
  "abstains rather than assert identifiability it cannot verify." Here it asserts
  `well-constrained` on a provably rank-deficient model. So this is a **hole in the stated
  fail-safe**, not an accepted/registered bound. `NUDGE-LIM-023` documents that the smallest end
  is *unreliable* — but claims the response is *abstention*; the shipped code instead emits a
  confident positive.
- **The existing slow test does not exercise this geometry.** `test_ode_matrixfree_dense_
  matches_and_iterative_is_fail_safe` uses a gLV community whose rank-deficiency is *data-driven*
  (many near-null directions), where `eigsh` either finds a genuine small eigenvalue or fails to
  converge (→ abstain). An **isolated exact null inside an otherwise well-conditioned spectrum**
  is the untested case: `eigsh` converges happily past it and certifies. (Measured: the true
  `λ_min ≈ 0` is recovered exactly by the `method="dense"` reconstruct-and-`eigh` path on every
  one of these models.)

**Candidate decoy (described, not added).** `NUDGE-DECOY-0xx — matrix-free identifiability on an
isolated structural null`: a well-conditioned linear `predict_fn` with one exact duplicated
parameter column, at `n_params ∈ {40, 300}`, analyzed via `analyze_model_matrixfree` (auto) and
`sloppiness_diagnostic_matrixfree(method="iterative")`. Expected verdict: `unidentifiable`
(matching the dense oracle / `method="dense"`). NUDGE must NOT return `well-constrained` /
`sloppy-but-predictive`. Positive controls paired: `method="dense"` resolves it correctly; a
full-rank well-conditioned model → `well-constrained` on both paths (no over-abstention).
Generator: `scripts/redteam/sloppiness_matrixfree_iterative_mislabel.py`.

**Candidate limitation (described, not registered — SHARPENS `NUDGE-LIM-023`; do NOT register).**
*The matrix-free iterative path's smallest-eigenvalue "fail-safe" (`_verified_smallest_eigsh` +
Rayleigh residual) verifies that a returned pair IS an eigenpair, but NOT that it is the
SMALLEST eigenpair. When `n_params ≤ n_obs` (shape-null certificate inapplicable) and the FIM
has an ISOLATED (near-)zero eigenvalue in an otherwise well-conditioned spectrum,
`eigsh(which="SA")` converges to the well-conditioned cluster and misses the null; the returned
pairs pass Rayleigh, so `smallest_certified=True` and NUDGE emits a confident `well-constrained`
/ `sloppy-but-predictive` ("the model is usable; do not abstain") verdict on a structurally
non-identifiable model — contradicting the module's own reliability claim. Verified 6/6 across
seeds via `method="auto"` (default, `n_params=300`) and `method="iterative"` (`n_params=40`),
`scripts/redteam/sloppiness_matrixfree_iterative_mislabel.py`.* Severity: **major**,
safety-relevant (a false "identifiable" green-lights a downstream point-estimate / attribution
the data cannot support). Note: this is at **full float64 precision** (x64 enabled) — it is
NOT the separately-documented float32-downcast resolution caveat.

**Measured mitigation direction (NOT applied — main agent decides).** The `method="dense"`
reconstruct-and-`eigh` path already returns the correct `unidentifiable` on every one of these
models (measured, 6/6), so the smallest end IS recoverable matrix-free at moderate `n`. Concrete
directions: (a) when the iterative smallest end is "certified", cross-check its *completeness*
before trusting it — e.g. compare a matrix-free trace estimate (`Σ_i eᵢᵀ·FIM·eᵢ` or a Hutchinson
stochastic trace) against `Σ` of the reconstructed spectrum; a large deficit ⇒ missed small
eigenvalues ⇒ abstain; (b) certify `λ_min` *independently* of `which="SA"` via a few steps of
matrix-free **inverse iteration** (CG-solve `FIM·x = b` through the matvec — the dominant
eigenvector of `FIM⁻¹` is FIM's smallest, memory-flat), and abstain if it lands near zero;
(c) at minimum, make `method="auto"` DEFER to `method="dense"` (exact, O(n²) memory, still avoids
the `jacfwd` OOM) unless the smallest end is independently certified — the docstring already
recommends dense "for a definitive verdict on a moderate `n_params`", so auto silently trusting
the uncertified iterative smallest solve is the specific gap. Re-measure on the isolated-null
family (`n_params ∈ [16, 2000]`, 1–3 exact nulls) that the gLV test does not cover.

---

## HELD — OED structural-unidentifiability (target 1: false-precise CRLB / ridge masking)

**Repro logic** (scratch): a `DesignProblem` whose target parameter is structurally
unidentifiable at *every* design (`observe` depends on `θ0, θ1` only via `θ0 + θ1`), run through
`optimize_design` for `objective ∈ {crlb, e_opt, d_opt}`, seeds {0,1}. **HELD — not a clean
confident-wrong:**
- `min_eigenvalue` uses a raw `eigvalsh` (no ridge) and honestly reports `0.0 → 0.0` (the exact
  null is surfaced loudly); `min_eig_improvement` is `inf` (0/0), not a resolution claim.
- The absolute `target_crlb` stays large (`283 → 154`, i.e. a parameter std ≈ 12 in log-space —
  visibly unresolved). The `crlb_improvement = 1.84×` is a genuine relative-ridge artifact (the
  optimizer inflates the FIM trace, shrinking the *relative* ridge floor), but it is *bounded*
  and *contradicted* by the honest `min_eig=0` channel and the large absolute CRLB. A caller
  reading the full `OEDResult` sees the degeneracy. So the OED module does NOT fake a resolution
  on a structural null — the guarded ridge is over-cautious in absolute terms, exactly as
  documented. Recorded as a fail-safe win (the `min_eig` raw-`eigvalsh` channel is the load-
  bearing honesty here). *Minor observation (not a hole): the raw `OEDResult` does not carry the
  `NUDGE-LIM-024` local-OED caveat string — only the `oed_demo` wrapper does; a library caller
  reading `crlb_improvement` gets no caveat. This is a UX/doc gap, not a confident-wrong.*

## HELD — OED demo + the four merged differential/multi-reporter/design fixes (target 4 regression)

- **`service.oed_demo` (`nudge oed`)**: logistic → `crlb 31.5× / min_eig 17.8×`, `φ*` puts
  samples in the transient (`3.62 … 12.0`) — matches the documented headline; no merge
  regression.
- **Regression sweep** `tests/inference/test_differential.py` + `test_multi_reporter.py` +
  `tests/design/test_invert.py`: **54 passed, 1 skipped, 1 xfailed** (the strict-xfail decoys
  hold). The P1/P4/P5-gate differential guards, the P2 multi-reporter floor-consistency gate, and
  the P3 design absolute-near-fold gate all survive the moat merges intact.
- **Moat own tests** `test_sloppiness_matrixfree.py` + `test_oed.py`: **16 passed** (fast lane) —
  the build is healthy, so P6 is a genuine *coverage gap*, not a broken merge.

---

## Honest caveats & coverage (budget-aware)

- **Covered (heavy):** the P6 isolated-structural-null sweep (6/6 seed·size cells through
  `method="auto"` default + explicit `method="iterative"`, root-cause instrumented); the OED
  structural-null / ridge-masking probes (HELD, `min_eig` honest); the OED demo regression
  (HELD); the four-merged-fix regression sweep (HELD, 54 passed).
- **NOT reached (budget / construction difficulty):** (1) reaching P6 *through an ODE* via
  `ode_identifiability` — an isolated exact null is hard to engineer inside a gLV/pathway RHS
  (the gLV rank-deficiency is diffuse), so P6 is demonstrated on the primary
  `analyze_model_matrixfree` / `sloppiness_diagnostic_matrixfree` public API, which *is* the
  shipped surface for this capability (it is not wired to the CLI/MCP). The mechanism (eigsh
  misses an isolated null; Rayleigh certifies non-smallest pairs) is ODE-agnostic, so the same
  hole is expected for any ODE fit with an isolated near-null, but that specific end-to-end path
  is **analyzed, not run → not a claimed result.** (2) The `sloppy-but-predictive` sibling
  mislabel (measured 3/3 on a wide-spectrum + isolated-null model — same root cause as P6, folded
  into the finding, not separately scripted). (3) A partial-panel P5 re-hunt (owned by the
  concurrent fix loop; the differential regression sweep confirms the *shipped* P1/P4 gates did
  not regress).
- P6 is on a **synthetic** analytic `predict_fn` (a linear map with an exact duplicated column) —
  the canonical, un-gamed structural-null construction (the module's own `redundant_exponential`
  scaled up), at full float64 precision; not yet demonstrated on a real large-network ODE fit.
- The HELD results are genuine outcomes under the attacks tried; they do not prove no other
  attack exists.
