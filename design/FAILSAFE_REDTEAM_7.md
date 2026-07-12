# NUDGE fail-safe red-team — ROUND 7 (post-moat re-scan / STOP gate; run `000000025`)

**Mandate.** Same as rounds 1–6: adversarially force a NUDGE capability to emit a *confident,
specific, WRONG* call (a positive mechanism / interaction / verdict / quantitative claim where
the honest answer is abstention), past the shipped gates. A verified hole is a WIN; an abstention
or a documented bounded residual is a PASS. This document **reports and reproduces; it does NOT
fix.** No `src/` capability code, `fit.py`, `core/`, the decoy battery, or any fail-safe margin was
touched — only this report + two new deterministic `scripts/redteam/*.py` were added.

This is the **re-scan / STOP gate**: a genuine full sweep on the integration branch with **P5 and
P6 both fixed + merged** (`grep -c _NUISANCE_EARN_MARGIN differential.py` = 3;
`grep -c _smallest_eig_null_probe sloppiness.py` = 4; HEAD `c17bdcf`). Budget was spent on the two
surfaces the moat sweep (`runs/000000018`) left explicitly **UNREACHED** — `ode_identifiability`
through a real ODE and OED multi-knob / false-precision — then a P5+P6 fix-induced-regression pass
and a general sweep.

## VERDICT: `HOLES_FOUND: 0` — a genuine full sweep found **no new confident-wrong**. **STOP.**

Every priority-target attack HELD (abstention or a documented, self-flagged bound); the four prior
fixes (P1/P2/P3/P4) and the two just-merged fixes (P5/P6) show no merge-induced regression; the full
fast lane is green (310 passed, 2 xfailed = the locked decoys, 0 failed). This zero is itself the
load-bearing result: it hardens the fail-safe claim and STOPs the loop.

## Score

| # | Target | Capability | Attack | Verdict (gate that held) |
|---|--------|-----------|--------|--------------------------|
| 1 | **ode_identifiability** isolated exact null, DENSE (`auto`) | `inference/adjoint`→`sloppiness` (`LIM-023`) | ODE analogue of the P6 `redundant_exponential` (species-0 decay = `θ0+θ1`), n_theta=11, `float32`(default)+`float64` | **HELD** — `unidentifiable` (n_null=1); the sum-redundant sensitivity columns cancel to ~1e-16 relative even at float32 → the exact dense reconstruct+`eigh` catches the null |
| 2 | **ode_identifiability** isolated null, GENUINE ITERATIVE | `inference/adjoint`→`sloppiness` (P6 fix) | same null at n_theta=17, `method="iterative"` — the inverse-iteration probe must catch what `eigsh('SA')` misses | **HELD** — `unidentifiable` (n_null=1) both dtypes/seeds; the P6 `_smallest_eig_null_probe` fires end-to-end through a real ODE. **Structural:** the iterative path can ONLY ever return `unidentifiable` (null-found→unidentifiable; else→abstain-unidentifiable) |
| 3 | **ode_identifiability** diffuse, data-driven rank-deficient gLV | `inference/adjoint` | `make_glv_problem(12,120,24)`, `float32`+`float64` | **HELD** — `unidentifiable` (17–29 nulls); the realistic large-network case |
| 4 | **ode_identifiability** float32 silent trap | `inference/adjoint` | non-proportional 3-column null; and x64-OFF default | **HELD** — both still `unidentifiable`; XLA's jvp reduction cancels the null to ~3e-16, and x64-off additionally emits a visible float32 truncation warning |
| 5 | **OED shipped demo** silent regression / masking | `service.oed_demo` = `nudge oed` CLI | 144 configs (model×objective×n_obs×steps×seed): any `crlb_improvement < 1` or masked CRLB | **HELD** — 0/144; the logistic/glv showcase genuinely improves at every config |
| 6 | **OED** last-iterate honesty (`optimize_design` returns the LAST Adam iterate) | `inference/oed` | aggressive LR (0.5–20) overshoot; is a false improvement ever claimed? | **HELD** — reported `crlb_improvement` == the honest measured ratio at the returned design, always (never a false >1) |
| 7 | **OED** guarded-ridge masking a singularity | `inference/oed` | structurally-degenerate target (`k1,k2` via their sum), all 4 objectives | **HELD** — the relative ridge inflates `crlb_improvement` (11–243×), but the raw `min_eig_opt` ≈ 0 AND the ABSOLUTE `target_crlb_opt` ≈ 6.6e3–1.1e4 both expose the degeneracy (re-confirms round-6) |
| 8 | **differential gate 4d** a NEW non-affine confound | `inference/differential` (`LIM-016` P5) | perturbed-only OVERDISPERSION (isotropic measurement noise, OUTSIDE the uniform-affine null's span), sd 0.5–2.0 | **HELD** — `no-difference` at every level (the shared model absorbs the extra spread; the positive-call path / gate 4d is never even reached) |
| 9 | **P5/P6 + prior-fix regression** | differential / sloppiness / oed / design / multi_reporter | merge-induced regression of every shipped guard | **HELD** — full fast lane 310 passed / 2 xfailed / 0 failed; slow: differential 29p/1xf, P6 3p/1xf; round-6 P6 repro now 0/6 |

**Confident-wrong holes found: 0.** Both moat surfaces the prior sweep left UNREACHED are now reached
end-to-end and HELD; the two just-merged fixes create no new confident-wrong; no other capability
regressed.

---

## Priority target 1a — `adjoint.ode_identifiability` reached through a genuine ODE (the P6 root cause)

**The claim under test.** The P6 hole (an iterative Krylov solver missing an *isolated* Fisher null
and certifying `well-constrained`) was closed for the abstract `predict_fn`. The moat sweep left the
**end-to-end ODE path** as `NOT REACHED` (`runs/000000018` target 6). This round reaches it, incl. the
**float32-default `ODEProblem.dtype`** interaction the mandate flags: `ode_identifiability` passes a
float64 `theta` to the diagnostic, but the `ode_trajectory_predict_fn` silently downcasts the forward
model to `problem.dtype` (**float32 by default**), so the FIM carries float32 roundoff regardless.

**Result — HELD on every construction (repro `scripts/redteam/ode_identifiability_float32_null_rescan.py`,
exit 0, `0/12` confident-wrong across 2 seeds × {float32-default, float64} × 3 constructions):**

- **A. Isolated EXACT structural null through the DENSE (`auto`) route** (n_theta=11 ≤ `dense_below`):
  a linear decay cascade whose species-0 decay rate is `θ0+θ1` (two free params enter only via their
  sum) — the ODE analogue of the P6 `redundant_exponential`. Returns `unidentifiable` (n_null=1) at
  **both** precisions. Root cause of the HELD: a sum/product redundancy in an ODE produces sensitivity
  columns that are (near-)**exactly proportional**, so the null direction's FIM entries cancel to
  ~1e-16 **relative** *even in float32* (measured `rel λ_min` ≈ 3e-16 at float32) — well below the
  `rank_rtol² = 1e-14` floor → the exact dense reconstruct+`eigh` catches it. Float32 roundoff does not
  lift it, because the cancellation is arithmetic, not perturbative.
- **B. The same isolated null through the GENUINE ITERATIVE route** (n_theta=17, `method="iterative"`):
  the P6 inverse-iteration null probe (`_smallest_eig_null_probe`) catches the isolated null that
  `eigsh(which='SA')` misses, → `unidentifiable` (n_null=1), both dtypes/seeds. **Structural backstop:**
  reading the shipped code, in the non-shape-null iterative branch the ONLY outcomes are
  `null_found → unidentifiable` or `smallest_certified=False → unidentifiable` (abstention) — the
  iterative path is *structurally incapable* of emitting `well-constrained` / `sloppy-but-predictive`,
  so it cannot produce the mandate's confident-wrong at all.
- **C. Diffuse, data-driven rank-deficient gLV** (`make_glv_problem(12,120,24)`): `unidentifiable`
  (17–29 nulls), both precisions — the realistic large-network case (float32 finds *more* nulls, never
  fewer).

**Additional float32 probes (scratch, folded in): (i)** a NON-proportional 3-column null (a readout
`a0·x0 + a1·x1 + a2·(x0+x1)` over two independent decay states — the only geometry that could
perturbatively lift a null) still cancels to ~3e-16 relative (XLA's jvp reduction of the tangent
cancels it) → `unidentifiable`, 3 seeds. **(ii)** With **x64 OFF** (the true naive-user default — the
whole diagnostic runs float32, incl. numpy `eigh` on a float32-reconstructed FIM), the isolated null is
STILL caught (`unidentifiable`, n_null=1) *and* JAX emits a visible float32-truncation `UserWarning`, so
the trap is not even silent.

**Why this is not a hole.** On every genuinely-unidentifiable ODE fit, `ode_identifiability` abstains
(`unidentifiable`). No construction produced a confident `well-constrained` / `sloppy-but-predictive`.

## Priority target 1b — OED (`inference/oed`): multi-knob / silent-regression / false-precision

**Repro `scripts/redteam/oed_shipped_demo_and_masking_rescan.py` (exit 0, `0` confident-wrong).**

- **(a) Shipped `oed_demo` / `nudge oed` regression.** 144 configs (model × 4 objectives × n_obs∈{4,8,16}
  × steps∈{50,400} × 3 seeds): **0** with `crlb_improvement < 1` and **0** with a masked target CRLB
  (`target_crlb_opt` small while `min_eig_opt` ≈ 0). The showcase logistic/glv designs genuinely improve;
  the multi-knob (`n_obs` measurement-time vector) case is covered here and never regresses.
- **(b) "Returns the LAST Adam iterate, not best-of-history."** True as a code fact, but not a
  confident-wrong: `crlb_improvement = crlb(φ_init)/crlb(φ_opt)` is *measured at the returned design*, so
  even a divergent run reports the **honest** ratio. Driven at LR 0.5–20 (overshoot), the reported ratio
  equalled the independently-recomputed true ratio in every case (never a false >1). It is a latent
  *robustness* deficiency (a best-of-history return would be strictly better), **not** a false claim.
- **(c) Guarded-ridge masking a singularity.** On a target that is structurally unidentifiable at *every*
  design (`k1,k2` enter only via their sum), the relative ridge in `crlb()` inflates the *relative*
  `crlb_improvement` to 11–243× — but the two honest channels the `OEDResult` also carries expose the
  degeneracy unambiguously: the raw `min_eig_opt` (no ridge) is ≈ 0 (measured −4.5e-13 … 9e-13), and the
  ABSOLUTE `target_crlb_opt` ≈ 6.6e3–1.1e4 (a log-space std ≈ 80 — visibly unresolved). This re-confirms
  and extends the round-6 HELD (there via an abstract map; here end-to-end through `optimize_design` with
  a genuine structural sum-degeneracy). The relative-ridge number is misleading *in isolation* — a
  documented `NUDGE-LIM-024` local-OED / relative-metric caveat — but is not a confident-wrong because the
  report self-flags on two independent honest channels.

## Priority target 2 — P5 + P6 fix-induced regression

- **P6 (`sloppiness` matrix-free, `LIM-023`).** The round-6 hole repro
  (`scripts/redteam/sloppiness_matrixfree_iterative_mislabel.py`) now reports **`0/6`** confident-wrong
  (was the hole; auto + iterative both `unidentifiable`). Slow P6 tests: 3 passed / 1 xfailed (the
  huge-regime over-abstention bound). The inverse-iteration probe **never** falsely certifies a positive
  (it can only yield `unidentifiable`, an abstention — safe by construction, verified by code + probe B).
- **P5 (`differential` gate 4d, `LIM-016`).** Slow differential regression: **29 passed / 1 xfailed**
  (the deflation strict-xfail). A fresh NON-affine confound — a perturbed-only **overdispersion**
  (isotropic measurement noise, sd 0.5–2.0, OUTSIDE the uniform-affine null's span) — did NOT create a new
  confident-wrong: every case returned `no-difference` (the shared LNA model absorbs the extra spread; the
  candidate-positive path that invokes gate 4d is never reached). The uniform affine class (P1/P4/P5) stays
  closed; the documented residual (a non-uniform above-median scale, degenerate-with-a-genuine-ceiling per
  `runs/000000013`) is unchanged.
- **P1/P2/P3/P4 + no merge regression.** Full fast lane: **310 passed, 5 skipped, 2 xfailed
  (locked decoys), 0 failed**. The capability fast lane (design/multi_reporter/epistasis/lotka/
  fibrillization): 63 passed / 2 xfailed. The P5 and P6 fixes do not interact to produce a new
  confident-wrong (they live in disjoint modules — `differential` vs `sloppiness` — and both regression
  suites pass).

## Priority target 3 — general sweep

The full fast lane (310 passed) exercises every capability's fail-safe tests + the decoy battery
(2 strict-xfails hold: the epistasis additive-ambient synergy `LIM-009` and one differential deflation
bound). No capability emitted a confident-wrong. **Cross-capability composition into `design()`** was
analyzed: `design()`'s integrity gate refuses an unreliable fit and its reachability gate abstains on an
unreachable target, so a propagated confident-wrong would require an upstream `is_reliable=True` on a
wrong attribution — and the upstream emitters (dose_response, epistasis) are HELD/locked from prior
rounds. No reproducible composition hole was found (this is an **analysis, not a run → not a claimed
result**; see coverage below).

---

## Coverage & honesty (budget-aware — what was and was NOT swept)

**Covered (measured, this round):**
- `ode_identifiability` end-to-end: 3 ODE constructions × 2 dtypes × 2 seeds (dense `auto` + genuine
  `iterative`) + 2 extra float32 geometries (non-proportional null, x64-off) — the surface `runs/000000018`
  left UNREACHED; HELD, with a structural proof the iterative path cannot emit a positive verdict.
- OED: the full shipped-demo grid (144 configs), the last-iterate honesty probe (8 runs), and the
  structural-ridge-masking probe (12 runs); HELD on all.
- P6 regression: the round-6 repro (0/6) + slow P6 suite (3p/1xf).
- P5 regression: slow differential suite (29p/1xf) + a fresh non-affine (overdispersion) attack.
- Global regression: the full fast lane (310 passed, 0 failed) + the capability fast lane (63 passed).

**NOT fully swept / analyzed-only (explicitly not "fully swept"):**
- The **full slow lane for every capability** was not re-run end-to-end — only the P5/P6-relevant slow
  suites (differential, sloppiness) + the differential/design/multi_reporter slow decoys + the moat.
  Slow-marked lotka/fibrillization/epistasis validation beyond the fast lane were deselected (budget).
- **Cross-capability composition** (an upstream confident-wrong → `design()`) was reasoned about, not
  exhaustively driven — bounded by the upstream fail-safes, but **not a claimed no-hole result** for
  arbitrary upstream inputs.
- A **maximally-adversarial float32-lifted null** (a hand-built "different-computation-path" redundancy
  engineered so the tangent does NOT cancel arithmetically) was analyzed but not constructed: the three
  natural geometries all cancelled to ~1e-16 (XLA fuses the reduction), so no float32 lift was observed —
  this is a HELD observation, **not** a proof that no pathological ODE could ever lift a null.
- The OED **relative** `crlb_improvement` remains misleading *read in isolation* on a degenerate target
  (a `NUDGE-LIM-024` UX/relative-metric caveat, self-flagged by `min_eig` + absolute CRLB) — re-confirmed
  HELD, not re-litigated as a hole.
- The HELD results are genuine outcomes under the attacks tried; they do not prove no other attack exists.

**No decoy / limitation is registered here** (red-team does not fix). The two new scripts are additive,
deterministic (≥2 seeds), lint-clean (`uv run ruff check scripts` passes, line-length 100), and drive the
SHIPPED API; each exits 0 = HELD, 2 = a hole, so they double as regression guards for the STOP verdict.
