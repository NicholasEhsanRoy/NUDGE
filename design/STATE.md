# NUDGE — project state, roadmap & next-step plan (context-handoff)

**This is the single source of truth for "where are we and what's next."** Read it
first each session. Last updated 2026-07-08, after Phase 2 (proof of concept) +
overnight V&V calibration.

---

## 0. How to work here (operational essentials)

- **Toolchain: `uv`** (there is NO system `pip`). Run everything via `uv run`:
  - Tests: `uv run pytest -q` (fast lane). Slow lane: `uv run pytest -m slow`.
  - Lint/type: `uv run ruff check src tests scripts` · `uv run pyright src`.
  - Env is a local `.venv` (git-ignored); `uv pip install -e ".[dev]"` to set up.
- **Checks that must pass before every commit:** ruff, pyright, `pytest` (fast lane),
  and the three `scripts/check_*.py` validators.
- **Commits: always credit Claude.** Append `Co-Authored-By: Claude Opus 4.8
  <noreply@anthropic.com>`, write a real body saying what Claude did. See `CLAUDE.md`
  and the `/commit` skill. Commit + push on `main` (the user is fine with this).
- **Ruff/pyright are pinned** (`ruff==0.15.20`, `pyright==1.1.411`) so local == CI.
  `scripts/vv/*` has relaxed lint (throwaway analysis code).
- **Test lanes** (pyproject markers): default CI runs everything except
  `slow`/`validation`/`needs_llm`; `verification` + `decoy` run in default;
  heavy things are `slow` (scheduled lane). Real-data checks are `validation`+`needs_data`.
- **The approved structural plan** lives at
  `/home/nick/.claude/plans/great-let-s-now-work-ticklish-hamming.md` (outside the
  repo). The design docs (`design/WORKING_BACKWARDS.md`, `PITCH.md`,
  `GENERATOR_DESIGN.md`) hold the reasoning; `GENERATOR_DESIGN.md` especially — it
  has the two literature-review syntheses (count model + the bistability crux).

## 1. What NUDGE is (one paragraph)

Fits a compositional, differentiable gene-regulatory **circuit** ODE to single-cell
Perturb-seq counts and attributes each perturbation to a **mechanism** — does it
move a switch's **threshold** (K), **gain** (Hill n), or **ceiling** (v_max) — and
**abstains loudly** when it can't tell. Built on **MADDENING** (a differentiable
JAX graph-physics engine) — reuses its `ift_linear_solve` primitive and
`maddening.compliance` traceability, but NOT its `GraphManager` (see §4).

## 2. Current state — Phases 0–2 DONE (proof of concept closed + calibrated)

| Phase | Status | Key files |
|---|---|---|
| 0 Bootstrap | ✅ | `pyproject.toml`, CI `.github/workflows/`, `docs/known_limitations.yaml`, `scripts/check_*.py` |
| 1 Generative backbone | ✅ | `core/circuit.py`, `mechanisms/` (species, integrators, regulatory, readout), `data/synthetic.py`, `data/noise.py`, `data/ingest.py` |
| 2 Fit | ✅ (PoC) | `inference/losses.py`, `inference/fit.py`, `inference/classify.py` |
| V&V calibration | ✅ | `scripts/vv/` (harness + results + `FINDINGS.md`) |
| 3 Fail-loud | ◑ ~35% | gate logic + linear decoy + Tier-0.5 stochastic simulator (`data/stochastic.py`) + its inverse-crime guard test; battery/suite/Laplace NOT built |
| 4 Validation + provenance | ⬜ | T-cell SOS/RasGRP1; `provenance.py` is a stub |
| Stretch | ⬜ (homes reserved) | `design/invert.py`, `mcp/server.py`, `mechanisms/integrators/zero_order.py`, `data/loaders/tier{1,2}.py`, docs site, `scripts/ai/` (5 creative-AI hooks) |

**The PoC works end to end** (`tests/inference/test_fit_end_to_end.py`, slow lane):
generate ground-truth data → `fit()` → recover kinetics → attribute
threshold/gain/ceiling → and `off-model` when a linear model suffices.

## 3. Key empirical findings (overnight V&V — `scripts/vv/FINDINGS.md`)

- **0% misclassification at every `margin_k`** across 300 linear + 120 switch
  datasets. NUDGE never calls the wrong mechanism; it abstains. Fail-safe, measured.
- **Calibrated default `margin_k = 1.7`** → <2% false-positive rate on linear data.
  It's a specificity/sensitivity dial (1.0 → 88% correct/7.7% FP; 1.7 → 65%/1.7%).
- **Identifiability:** needs **≥~1000 cells/condition**; **gain > ceiling ≈
  threshold**; **ceiling is the most noise-fragile**; threshold hardest (K/v_max
  partial degeneracy — both shrink the ON signal).
- **Tier-0.5 (independent stochastic data) — fail-safe survives, with a boundary.**
  On data from the new tau-leaping SSA (`data/stochastic.py`, emergent bimodality),
  a matched-topology fit emits **0 wrong positives across seeds 0–3** (abstains, or
  recovers only gain) — the fail-safe property holds off the inverse crime. BUT
  fitting a *wrong* (feedforward) topology to the feedback data produced a confident
  wrong call (`gain→threshold`, every `margin_k`): **the guarantee is conditional on
  approximately-correct topology.** Full write-up in `scripts/vv/FINDINGS.md` §Tier-0.5.

## 4. Architecture decisions & gotchas (a fresh context MUST know these)

- **Circuit = self-contained differentiable JAX vector field, NOT `GraphManager`.**
  `GraphManager` bakes params as compile-time constants; we need params as a traced
  pytree (`{"species": {...}, "edges": {...}}`) to `vmap` over per-cell draws and take
  gradients. `Circuit.solve_population` = `jax.vmap` of a semi-implicit steady-state
  solve. (Documented as a MADDENING case study in
  `../plans/NUDGE_deterministic_solve_vs_graphmanager.md`.)
- **The population model = `vmap` over per-cell parameter draws** (extrinsic noise on
  basal/decay), NOT stochastic dynamics. This is the validated deterministic
  transfer-function route (Ochab-Marcinek & Tabaka 2010) — bimodality is *designed-in*
  (a Tier-0 feature). **This is exactly what Tier-0.5 must break** (see §6).
- **Count model = negative binomial, NO zero-inflation** (`data/noise.py`), via
  Poisson-Gamma. UMI droplet data is not zero-inflated (Svensson 2020 etc.).
- **Fit forward model** (`inference/fit.py`): the NB count *sample* is discrete/
  non-differentiable, so the loss uses a **reparameterized moment-matched Gaussian
  observation** `μ + √(μ+φμ²)·ζ`, **clamped ≥0** (counts can't be negative; also
  avoids negative-tail NaNs). A **`log1p` transform** makes the energy distance
  shape-sensitive (bimodality) — this is what separates the mechanisms.
- **classify.py has two levels:** (1) circuit-level **`switch_detected`** — the
  linear-baseline parsimony gate (mechanistic must beat linear on WT beyond the loss
  noise floor, else no switch → all `off-model`); (2) per-perturbation **`decide`** —
  no-effect / off-model(absolute) / unresolved / threshold-gain-ceiling. The gate is
  at the CIRCUIT level deliberately: per-perturbation it misfires because a strong
  gain/ceiling reduction genuinely linearizes a perturbed condition.
- **Attribution = three restricted fits** (free K / n / vmax of the target edge from
  the WT baseline); the winner is the mechanism. Noise floor = WT self-distance
  bootstrap (`_self_distance`).
- **Compile-cache gotcha:** the persistent on-disk JAX compile cache (MIME's conftest
  pattern) served corrupted `random.poisson` executables (silent -1). It is
  **disabled** in `tests/conftest.py` — do NOT re-enable it.
- **maddening pin:** `maddening[ift]>=0.3.1` (the `[ift]` extra pulls `lineax` for
  `ift_linear_solve`). 0.3.0 lacked it.

## 5. Public API surface (import-light where it matters)

`nudge.fit(adata, circuit) -> MechanismMap` · `nudge.Circuit` / `CircuitBuilder` ·
`nudge.generate_synthetic_perturbseq` · `nudge.PerturbationSpec` ·
`nudge.MechanismClass` (threshold/gain/ceiling + no-effect/unresolved/
technical-artifact/off-model) · `nudge.MechanismMap`/`MechanismCall`.
Lower-level: `inference.fit.fit_parameters` (the optax recovery engine),
`inference.classify.{decide, switch_detected}`, `inference.losses.{energy_distance,
rbf_mmd}`, `data.ingest.check_counts` (the raw-counts bouncer).

---

## 6. Tier-0.5 independent stochastic simulator — ✅ LANDED (simulator + guard)

**Status (updated after build).** Built: `data/stochastic.py`
(`generate_stochastic_perturbseq`, a tau-leaping SSA of a self-activating gene,
emergent bimodality, reusing the `Readout`+NB observation layer verbatim; re-exported
via `data/loaders/tier05.py` and `nudge.__init__`) and its guard test
`tests/verification/test_stochastic_inverse_crime.py` (a fast bimodality check +
the slow never-wrong fit assertion). **Result:** matched-topology fit → **0 wrong
positives across seeds 0–3** (fail-safe holds off the inverse crime); wrong-topology
fit → can misclassify (the fail-safe boundary is topology). See `FINDINGS.md`
§Tier-0.5. **Deferred follow-ons:** the To & Maheshri bimodality-without-bistability
*decoy* (needs a telegraph/promoter mechanism + a short lit search — user-confirmed
fast-follow), and a **multi-basin IC seeding** extension to the fit so it can
*represent* emergent feedback bistability — a Fable-5 async spike found it feasible,
and it is now **built into NUDGE** (`inference/losses.py::energy_distance_weighted`,
`inference/fit.py::{fit_multibasin_parameters, fit_multibasin}`, alongside the unchanged
`fit`). **Result: representation works (≈10× lower loss, `p` recovers) but attribution
degenerates** — a two-fixed-mode mixture conflates *gain* (graded) with *ceiling*
(shifted high mode), a confident wrong call → so `fit_multibasin` is EXPERIMENTAL /
not-fail-safe and the Tier-0.5 guard stays on single-basin `fit`. Full arc in
`scripts/vv/FINDINGS.md` §T0.5-3/§T0.5-4. **Under investigation:** a 3rd "transition"
mode at the unstable saddle to break the gain/ceiling degeneracy (user's idea).

The original plan is retained below for reference.

### (original plan) the Tier-0.5 independent stochastic simulator

**Why (the one caveat the whole PoC can't escape):** everything so far is **Tier-0
inverse crime** — the generator and the fitter share the same deterministic model +
noise params. The literature (`GENERATOR_DESIGN.md` "THE CRUX, RESOLVED"; Kepler &
Elston 2001; To & Maheshri 2010) says the honest robustness test is data from an
**independent, genuinely stochastic** process where bimodality is **emergent** (noise-
induced switching, mode occupancy set by the landscape/basin depths), NOT designed-in
by a parameter distribution. If NUDGE's deterministic fit still attributes mechanism
on such data, the approach generalizes. If it breaks, we learn the failure mode cheaply.

**What to build:**

1. **A self-contained stochastic simulator** — `src/nudge/data/stochastic.py` (or fill
   `data/loaders/tier05.py`). NO new heavy deps; a **tau-leaping SSA** of a
   self-activating gene is ~100 lines of numpy:
   - State = molecule count `X` per cell. Reactions per step `dt`:
     `production ~ Poisson((basal + vmax·X^n/(K^n+X^n))·dt)`,
     `degradation ~ Poisson(decay·X·dt)`; `X += prod − deg`, clip ≥ 0.
   - Run N independent cells from random ICs to time T (steady state) → snapshot `X`.
     Bimodality is **emergent** (Poisson noise + positive feedback → noise-induced
     switching), occupancy set by the landscape — the key difference from Tier-0.
   - Add mild per-cell extrinsic variation (rate constants) on top of intrinsic noise.
   - Map `X` → counts via the SAME observation layer (`data/noise.py`) so `fit()`
     consumes the AnnData identically. Emit the same schema as
     `generate_synthetic_perturbseq` (`.X` counts, `.obs` condition/true_mechanism,
     `.uns['ground_truth']`).
   - Perturbations move `K` / `n` / `vmax` in the propensity (ground-truth mechanism).
   - Give it a `generate_stochastic_perturbseq(...)` mirroring the Tier-0 signature.
   - Sanity: JAX is awkward for variable-length Gillespie; **tau-leaping is
     vectorizable over cells in numpy/jnp and fine** (generation isn't differentiated).

2. **The inverse-crime-guard test** — `tests/verification/` (mark `slow`+`verification`):
   generate Tier-0.5 switch data with strong K/n/vmax movers → `fit()` → assert
   **correct-or-unresolved, NEVER the wrong mechanism** (the fail-safe property must
   survive the model mismatch), and ideally correct for the clear (gain) case. This is
   the headline Tier-0.5 result.

3. **A "bimodality-without-bistability" decoy** (To & Maheshri 2010 route) — generate
   noise-induced bimodal data from a NON-cooperative (n≈1) stochastic feedback loop
   (bimodal but no deterministic switch) → `fit()` must return `off-model`/not-a-switch
   (its deterministic Hill fit must not beat linear beyond the floor). This is a
   scientifically-validated decoy the literature specifically flags.

**Success criterion:** on independent stochastic data, misclassification stays 0% and
the clear mechanisms (esp. gain) are still recovered. Expect threshold/ceiling to
abstain more (harder under mismatch) — that's acceptable and honest.

**After Tier-0.5, the rest of Phase 3:** the full decoy battery (`data/decoys.py` +
`tests/decoys/` — several buildable now: two-cell mixture, dropout zero-peak, dead
guide), the verification suite (confusion matrix as a test, calibration coverage),
Laplace uncertainty (`uncertainty/laplace.py` → intervals on `MechanismCall`). Then
Phase 4 (T-cell SOS/RasGRP1 + wire `provenance.py`) and stretch (`design()`, MCP).

## 7. Other de-risking V&V still open (lower priority than Tier-0.5)

- Misspecification robustness (fit with wrong dispersion/library σ than the data).
- Bistable-FEEDBACK circuits (self-activation loop, basin-spanning ICs) vs the current
  feedforward switch.
- A richer multi-reporter readout to break the K/v_max degeneracy.
- Replicate the identifiability sweep at higher fit budget.
