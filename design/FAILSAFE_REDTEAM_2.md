# NUDGE fail-safe red-team — ROUND 2 (the CORE ENGINE)

**Mandate.** Same as round 1 (`design/FAILSAFE_REDTEAM.md`): adversarially force a NUDGE
capability to emit a *confident, specific, WRONG* call where the honest answer is abstention,
past the gates. A verified hole is a WIN (a documented limitation/decoy). **This document
reports and reproduces; it does NOT fix.** No `src/` capability code, `fit.py`, the decoy
battery, or any fail-safe margin was touched — only this report + `scripts/redteam/*.py` were
added.

Round 1 hit the *peripheral* attribution capabilities (lyapunov-multi, epistasis, cross-
modality, dose-response, bifurcation). **Round 2 targets the CORE ENGINE** — the parsimony
gate `fit()→classify`, the multi-basin / saddle transition-mode path, topology
misspecification, the newly-shipped `constitutive` capability, and a regression on the two
round-1 fixes.

Repro scripts: `scripts/redteam/*.py` (lint-clean, ruff line-length 100; `uv run`). Every
verified hole below is reproduced with ≥2 seeds through the SHIPPED code path.

## Score

| # | Target | Capability | Attack | Verdict |
|---|--------|-----------|--------|---------|
| 1 | constitutive (#4) | `inference/constitutive` | control/population **capture-efficiency** mismatch → `biological-switch` on a LINEAR circuit | **HOLE — verified** |
| 2 | parsimony gate (#1) | `inference/fit.fit` + `classify` | static cell-type **mixture** (wide/balanced ratios) faking a switch at default budget | HELD |
| 3 | parsimony gate (#1) | `inference/fit.fit` + `classify` | **structured dropout / mixture** past the shipped decoys, default budget | HELD (see note) |
| 4 | transition mode (#2) | `inference/fit.fit_multibasin(transition_mode=True)` | non-gain (ceiling/threshold, fold-crossing) perturbation faking `gain` via `w_trans` | HELD |
| 5 | topology (#3) | `inference/fit.fit` | wrong-topology fit → wrong mechanism | KNOWN boundary (T0.5-2), not re-claimed |
| 6 | LIM-017 fix (#5) | `inference/lyapunov.attribute_lyapunov_multi` | near-fold 3rd point **just below** the `well_buffered_margin=0.15` gate (knife-edge) | **HOLE — verified (regression)** |
| 7 | LIM-009 lock (#5) | `inference/epistasis` | additive / multiplicative+additive variant faking synergy | LOCKED (re-confirmed by construction) |

**Confident-wrong holes found: 2 verified.** (1) The **constitutive** capture-efficiency
confound — a hole in the capability *built to remove* a confident-wrong, defeating that
capability's own `is_confident_wrong` guard structurally. (2) A **regression** on the round-1
LIM-017 fix: the `well_buffered_margin=0.15` fix is a **knife-edge** — the identical
ceiling→threshold corruption reappears at a 3rd point with proximity 0.146, immediately below
the gate. The core `fit()→classify` parsimony gate and the saddle transition-mode gain gate
**HELD** under every attack tried — genuine wins for the fail-safe claim on the load-bearing
engine.

---

## HOLE 1 — a control/population capture-efficiency mismatch makes `constitutive` assert `biological-switch` on a LINEAR circuit

**Capability:** `nudge.inference.constitutive` (`NUDGE-METHOD-011`, `NUDGE-LIM-018`) — the
shipped `NUDGE-LIM-006` mitigation, `constitutive_control_analysis` / `profile_circuit_n` /
`classify_constitutive`.
**Repro:** `scripts/redteam/constitutive_control_batch_confound.py` (`uv run python …`,
shipped default budget restarts=3/steps=600/n_model_cells=400; ~a few min/case).

**The claim under attack.** The module is advertised as **structurally fail-safe**:
`ConstitutiveResult.is_confident_wrong` returns `True` *only* for a bare
`threshold`/`gain`/`ceiling`, "so it can only move a confident false positive *toward* a
correct call or an abstention" (module docstring). Its strongest positive verdict,
`biological-switch` ("the ultrasensitivity is a real circuit switch, reject the readout-only
explanation"), is treated as never-confident-wrong. The shipped validation shows the LIM-006
hazard — a LINEAR (circuit `n=1`) population read through a nonlinear reporter — correctly
ABSTAINS (`unresolved`) when the control is clean (`test_constitutive.py::
test_linear_circuit_lim006_hazard_abstains_not_confident_wrong`).

**The blind spot.** `biological-switch` **is** a positive, falsifiable claim, and
`is_confident_wrong` does **not** count it. So if any realistic confound can push the profile
to reject `n=1` on a genuinely linear circuit, NUDGE emits a confident-wrong that its **own
fail-safe property cannot see** — a hole in the very capability built to *remove* a
confident-wrong (LIM-006).

**The attack (realistic, un-gated).** The constitutive control is a **separate population**
(constitutively-driven reporter cells). Single-cell samples routinely differ in **capture /
sequencing efficiency**, so the control's raw reporter counts sit on a different
multiplicative scale than the circuit population's counts. The module compares them
**directly** — `calibrate_readout` fits the reporter transfer function from the control's raw
responses assuming they are on the population's count scale, and there is **no** control→
population depth normalization anywhere in the path. Reading the control at **0.5×** the
population's efficiency mis-anchors the reporter `Vmax` low; the profile-over-`n` then develops
a spurious well **off** `n=1` and REJECTS "no switch".

**Confident-wrong output (verified — 2/2 seeds at full budget, 2/2 at reduced budget):**

```
seed=0  [clean control scale=1.0]  call='unresolved'       n1_rej=0.0000  5xspan=0.0047  argmin_n=1.0   # correct abstention
seed=0  [confounded scale=0.5]     call='biological-switch' n1_rej=0.0145  5xspan=0.0038  argmin_n=7.0   <== HOLE
seed=1  [clean control scale=1.0]  call='unresolved'       n1_rej=0.0000  5xspan=0.0040  argmin_n=1.0   # correct abstention
seed=1  [confounded scale=0.5]     call='biological-switch' n1_rej=0.0142  5xspan=0.0018  argmin_n=4.0   <== HOLE
```

Truth is a **linear circuit (`n=1`)**: the observed ultrasensitivity is entirely in the
reporter (`h=6`) — exactly the LIM-006 artifact the module exists to *reject*. With a clean
control NUDGE correctly abstains; with the capture-efficiency-mismatched control it asserts a
confident `biological-switch`. All three gate conditions pass by a margin: `n1_rejection`
≈0.014 clears the absolute margin `reject_abs=0.01`, exceeds `5×` the without-control span
(≈0.002–0.004), and the profile argmin moved off `n=1` (to 4–7). Directional: reading the
control *brighter* (scale=2.0) HELD (`unresolved`); reading it *dimmer* (scale=0.5)
manufactures the switch.

**Which gate failed.** None can see it. The module has **no** gate comparing the control's
depth/scale to the population's — `calibrate_readout` trusts the control's raw responses as
already on the population's count scale, and `profile_circuit_n` anchors the population fit to
that mis-scaled reporter. And the module's self-check, `is_confident_wrong`, **structurally
excludes** `biological-switch`, so the guarantee "0 confident-wrong" is defined too narrowly to
catch this class of error.

**Candidate decoy (described, not added).**
`NUDGE-DECOY-0xx — constitutive control read at a different capture efficiency`: a LINEAR
(`n=1`) circuit population + a constitutive control whose raw responses are scaled by 0.5
(a separate-sample capture-efficiency mismatch). Expected verdict through
`constitutive_control_analysis`: **`unresolved`** (NUDGE must NOT return `biological-switch`).
Positive control paired in the same case: the identical panel with scale=1.0 → `unresolved`
(so the decoy isolates the scale mismatch as the culprit). Generator: exactly
`scripts/redteam/constitutive_control_batch_confound.py::_analyze`.

**Candidate limitation (described, not added — next free id is `NUDGE-LIM-019`, do NOT
register).** *The constitutive control assumes the calibration population and the circuit
population share not only a reporter transfer function but a **count scale**.* A capture-
efficiency / library-depth mismatch between the two separate populations mis-anchors the
reporter `Vmax` and can manufacture a **confident `biological-switch` on a linear circuit** —
resurrecting the exact LIM-006 confident-wrong the control was built to remove. Compounding
it, `ConstitutiveResult.is_confident_wrong` structurally excludes `biological-switch`, so the
module's own fail-safe self-check is blind to this error class. Severity: **major**, safety-
relevant (a confident-wrong in a mitigation capability is worse than an honest gap — it is sold
as the fix). Verified: `scripts/redteam/constitutive_control_batch_confound.py`, 2/2 seeds at
full budget, gate margins ≈1.4× `reject_abs`. Mitigation direction (NOT applied — main agent
decides): (a) **normalize the control to the population's sequencing depth** (size factors /
shared spike-ins) before `calibrate_readout`, or add a **scale-consistency gate** that abstains
when the control-implied reporter `Vmax` is inconsistent with the population's observed dynamic
range; and (b) **broaden the honesty contract** so `biological-switch` is not treated as
unconditionally safe — e.g. `is_confident_wrong` (or a companion flag) should acknowledge that
`biological-switch` is a falsifiable positive that a control/population scale mismatch can make
wrong. Related: `NUDGE-LIM-006` (the confound this exists to break), `NUDGE-LIM-018` (does not
point-identify `n`), and the differential per-context depth-ratio abstention (`NUDGE-LIM-016`)
— which the constitutive path lacks the analogue of.

---

## HELD — the core parsimony gate `fit()→classify` (targets #1, #3)

**Repro:** `scripts/redteam/core_parsimony_mixture.py`.

The load-bearing "is there a switch at all" gate (`switch_detected`) and the per-perturbation
`decide` were attacked with **non-switch** data designed to look bimodal/switch-like, run
through the SHIPPED `fit()` at its **default** budget (`n_cells=256`, `steps=300`,
`margin_k=1.7`) — the budget a bare `nudge.fit(adata, circuit)` call uses (the decoy battery
tests at a *higher* 384/400 budget, so the default is the weaker, more attackable surface).

- **Static cell-type mixtures at ratios the shipped mixture decoy (`NUDGE-DECOY-002`,
  0.5→0.3, low 0.05 / high 2.0) does not cover** — wide/balanced (low 0.02 / high 4.0,
  0.5→0.2), very-wide (low 0.01 / high 8.0), high-fraction (0.7→0.3), crisp (high 3.0,
  0.6→0.25), each a two-level static mixture with a fraction-shifting "perturbation".
  **HELD 0/8** (all seeds, all configs): `beats_linear_baseline=False`, every call
  `off-model`. **Why it holds (a structural strength worth recording):** the default single-
  basin `fit()` solves the mechanistic hypothesis from `x0=0`, so the self-activation switch
  converges to its LOW fixed point for every cell — the mechanistic model is itself
  effectively **unimodal** and therefore cannot out-fit the linear baseline on a bimodal
  *static mixture*. The parsimony gate is structurally robust to static bimodality at the
  default budget, not merely calibrated to it.

**Note (target #3, structured dropout).** The shipped dropout witness (`NUDGE-DECOY-003`) is
already covered; the mixture sweep above is the adjacent, not-covered ratio family and it held
by the same structural mechanism. A *depth-structured* dropout aligned with the perturbation
(the additive-confound analogue) was not separately driven end-to-end here — it is the same
class as round-1 HOLE 2 (an additive/technical axis aligned with the condition), and the
single-basin gate's unimodality makes a spurious *positive mechanism* call unlikely; flagged as
a coverage gap, not a claimed hole.

## HELD — the saddle transition-mode GAIN gate (target #2)

**Repro:** `scripts/redteam/transition_mode_false_gain.py`.

The saddle transition-mode gain gate (`decide_with_transition`, fires `gain` when the free-`n`
restricted fit spends `w_trans > gain_wtrans_tau=0.5` on a 1-species saddle) was attacked with
**non-gain** perturbations on a 1-species self-activation switch, on **independent tau-leaping
SSA** data (`generate_stochastic_perturbseq` — off the inverse crime, the emergent-bistable
regime the path targets), through the SHIPPED `fit_multibasin(transition_mode=True)`:

- Ceiling knockdowns (`vmax×{0.6,0.5,0.45,0.35}`, incl. fold-crossing collapse) and threshold
  moves (`K×{0.5,2.0,3.0}`, incl. past the fold). **HELD 0 false-`gain`** across all cases /
  seeds. Outcomes were `off-model` / `unresolved` (the WT parsimony gate on this SSA data
  frequently declines to even detect the switch — a fail-safe abstention), and where the
  switch WAS detected and the attribution path genuinely ran (`K×0.5`, seed 1) it correctly
  resolved **`threshold`** (conf 1.0) — the gate was exercised and did **not** mis-fire to
  gain. The `w_trans`≈0.9-gain / ≈0.01-else separation (§T0.5-5) held under the attack.
- **Honest coverage caveat:** because the WT parsimony gate abstained on much of the SSA data,
  the GAIN gate itself was lightly exercised; a deep, reliably-detected SSA switch plus a
  strong ceiling collapse was also tried (`transition_mode_false_gain.py`, deep variant) and
  likewise produced no false gain, but a WT tuned to sit *at* the fold (large intrinsic
  intermediate mass) was not exhaustively swept — a residual gap.

## KNOWN boundary (not re-claimed) — topology misspecification (target #3)

Fitting a **wrong topology** to feedback data producing a confident wrong *mechanism* is
already documented as a NEGATIVE in `scripts/vv/FINDINGS.md` §T0.5-2 (a feedforward hypothesis
on feedback data called the gain mover `threshold` at conf 1.00, every `margin_k`, seed 0). It
is a known, stated boundary ("attribution presumes approximately-correct topology"), not a new
hole — so it is recorded here as **confirmed-known**, not re-reproduced. (Whether the real-data
`model_select` BIC gate reliably prevents *selecting* the wrong topology end-to-end is a larger
pipeline question left for a future round.)

## REGRESSION on the round-1 fixes (target #5)

### HOLE 2 (regression) — the NUDGE-LIM-017 `well_buffered_margin=0.15` fix is a KNIFE-EDGE

**Capability:** `nudge.inference.lyapunov.attribute_lyapunov_multi` (the patched round-1
HOLE 1).
**Repro:** `scripts/redteam/nearfold_knifeedge_regression.py` (`uv run python …`).

**The claim under attack.** Round 1's HOLE 1 (a near-fold 3rd operating point flips a true
`ceiling` to a confident `threshold`) was FIXED by gating the joint fit on the deterministic
bifurcation-proximity dial: abstain unless every operating point has `proximity ≤
well_buffered_margin` (0.15). The round-1 hole used toggle `basal=0.60` (proximity 0.231), now
correctly gated out.

**The attack.** Proximity is **continuous and monotone** in the toggle's `basal` (measured:
0.05→0.039 … 0.40→0.146, 0.42→0.153, crossing the 0.15 gate near `basal≈0.41`). So a 3rd
operating point at `basal=0.40` has proximity **0.146 — it PASSES the 0.15 gate** — yet its
Lyapunov covariance is *already* biased. Clean points `{0.05, 0.30}` + this 3rd point.

**Confident-wrong output (verified, 2/2 seeds; positive-controls in the same run):**

```
seed=0  clean 2-pt subset -> 'ceiling'  (correct; positive control)
  3rd basal=0.40  prox3=0.146  gate_pass=True   label='threshold'  gap=0.5328  <== HOLE (gate passed, WRONG)
      NLLs: gain=7.035  threshold=6.502  ceiling=7.134
  3rd basal=0.42  prox3=0.153  gate_pass=False  label='unresolved'   # fix WORKS just above the margin
  3rd basal=0.44  prox3=0.160  gate_pass=False  label='unresolved'   # fix WORKS just above the margin
seed=1  3rd basal=0.40  prox3=0.146  gate_pass=True  label='threshold'  gap=0.5371  <== HOLE
        (0.42/0.44 -> unresolved, same as seed 0)
```

Truth = **ceiling**; the clean 2-point subset resolves it. Adding a 3rd point at proximity
**0.146** (gate passes) flips it to a confident **`threshold`** (gap 0.53 ≫ `resolve_margin`
0.03) — the *identical* corruption round-1 HOLE 1 documented, reappearing **immediately below**
the 0.15 margin. The gate catches proximity 0.153 / 0.160 (correctly abstains) but not 0.146:
the fix is a knife-edge, not a robust buffer. `lna_reliable` is `True` at all three points
throughout (as in round 1). The 0.42/0.44 positive-controls prove the gate itself works — the
finding is that its threshold is set right at the edge of the corruption regime.

**Which gate failed.** The `well_buffered_margin` **value** (0.15). The corruption regime and
the gate boundary overlap: covariance bias sufficient to flip the call already exists at
proximity ≈0.146, but the gate only abstains above 0.15. The margin was calibrated to the
round-1 witness (`basal=0.60`, proximity 0.231) and does not leave headroom below it.

**Candidate decoy (described, not added).**
`NUDGE-DECOY-0xx — near-fold 3rd point just below the well-buffered margin`: the same true
`ceiling` toggle at 3 operating points `{0.05, 0.30, 0.40}`, the 3rd at proximity ≈0.146 (gate
passes). Expected verdict: **`unresolved`**. NUDGE must NOT return `threshold`. Positive
control (same run): a 3rd point at `basal≥0.42` (proximity >0.15) already abstains, and the
clean 2-point subset resolves `ceiling` — so the decoy isolates the sub-margin near-fold point.
Generator: `scripts/redteam/nearfold_knifeedge_regression.py`.

**Candidate limitation (described, not added — SHARPENS `NUDGE-LIM-017`).** The LIM-017 fix
gates on `proximity ≤ 0.15`, but the covariance bias that flips a true `ceiling` to a confident
`threshold` is already present at `proximity ≈ 0.146` — **below** the margin (verified,
`scripts/redteam/nearfold_knifeedge_regression.py`). The margin was set to exclude the round-1
witness (`proximity 0.231`) and does not leave a safety buffer below the corruption onset, so
the fix is a knife-edge: the same confident-wrong reappears for a 3rd point tuned just under the
gate. Severity: **major**, safety-relevant (a regression on a shipped fail-safe patch).
Mitigation direction (NOT applied — main agent decides): **lower** `well_buffered_margin` (a
proximity sweep locating the corruption onset would set it empirically — the onset here is
below 0.146, so a value nearer ~0.10 with margin), and/or replace the single hard threshold
with a **graded down-weighting** of near-fold points in the joint fit (so a point's
contribution decays smoothly with proximity rather than switching on/off at a knife-edge), and
add this sub-margin decoy as the regression lock.

- **NUDGE-LIM-009 (additive-ambient synergy — LOCKED, no runtime gate):** round 1 established
  there is **no safe runtime gate** (any additive, gene-level offset aligned with A+B defeats
  size-factor normalization, and a real synergy would false-abstain under any blunt gate), so
  it was LOCKED as a strict-xfail decoy rather than fixed. It follows **by construction** that
  an additive-offset *variant* (or a multiplicative+additive mix) still fakes `synergistic` —
  this re-confirms the LOCKED status, it is not a new hole. Not re-run at length here (round-1
  `epistasis_pipeline_confound.py` already demonstrates the class); flagged as re-confirmed.

---

## Honest caveats on these findings

- HOLE 1 is on **synthetic** data with an engineered (but realistic and un-gated) capture-
  efficiency mismatch between the control and the circuit population. It is not yet demonstrated
  on a real paired constitutive-control screen (none is in the repo). Its value: it is a
  **confident-wrong in the capability built to remove a confident-wrong**, and it defeats that
  capability's own `is_confident_wrong` guard *structurally* (the guard never considers
  `biological-switch`), so the "0 confident-wrong / structurally fail-safe" claim for
  `constitutive` is too strong as stated.
- The HELD results (core parsimony gate, transition-mode gain gate) are genuine wins for the
  fail-safe claim on the **load-bearing engine** under the attacks tried; they do not prove no
  attack exists (see the coverage caveats above).
- Coverage gaps not probed this round: the real-data `model_select`/BIC topology gate end-to-
  end; a WT tuned exactly at the fold for the transition gain gate; multi-reporter's shared-
  additive-offset analogue (flagged already in round 1).
