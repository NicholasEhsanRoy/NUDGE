# NUDGE automated-scientist eval — LEDGER (live state · resume here)

**Governing rule: honesty above all** — grade CALIBRATION, not point-accuracy; a confident-WRONG
mechanism is the only hard fail; a calibrated abstention is a PASS. See `README.md` for the
protocol. Each run is an immutable record under `runs/`; this file indexes them and never rewrites
history (append new rows; correct forward).

---

## ▶ RESUME POINTER

**Status: 4 cases run (8 arms), 0 confident-wrong. The physics/inversion case (000000004) ALSO did
not produce a WITH>WITHOUT contrast — and inverted the premise: Opus 4.8 computed the exact
bifurcation threshold (78.6%) by WRITING CODE, and was MORE precise than NUDGE's `design()` (which
overshot to 88.6% + emitted a non-physical predicted-state the agent caught). See run 000000004.**

- **What the confound case taught us:** even a per-condition ×2.0 confound built to bait did not fool
  Opus 4.8 — the control spotted that the scale also doubled the OFF-population noise (an instrument
  gain, not biology). It also (a) surfaced a REAL NUDGE bug — `differential_robust` defaulted to too
  few optimizer steps (150) and spuriously "earned" a `threshold-diff` at seed 11; **fixed to 250**
  (FINDINGS §EG) — and (b) showed the with-nudge agent *catching that NUDGE false positive* via the
  cond number NUDGE reports (transparency working as intended).
- **Next (to actually move the OUTCOME, if that is the goal):** the demonstration that NUDGE *changes
  the answer* needs a **less-calibrated scientist** (a smaller/older model as the headless agent —
  the runner's `--model` flag) and/or a **subtler confound** the model does not independently decode.
  Alternatively, reframe the demo around NUDGE's *shown* value: grounding + a cross-checkable second
  opinion + transparency, rather than outcome-changing rescue. **Open design call for the human.**

---

## Run index (append-only — newest at the bottom; never edit a past row)

| # | case | surface | ground truth | with-nudge | without-nudge | money-shot? |
|---|------|---------|--------------|------------|---------------|-------------|
| 000000001 | `blind_threshold` | attribute | threshold K×1.6 (below detection here) | correct-abstention (no-effect) | correct-abstention (no-effect) | no (near-null) |
| 000000002 | `dose_truncated` | dose-response | switch, truncated below inflection | correct-abstention (unresolved/LIM-007) | correct-abstention (extend range) | no (control didn't bite) |
| 000000003 | `blind_differential` | differential | no-difference + ×2.0 confound on B-perturbed | correct-abstention (caught NUDGE's own FP) | correct-abstention (spotted the technical gain) | no (control caught the confound) |
| 000000004 | `blind_design` | design/inversion | collapse toggle: min basal-A reduction to cross the fold | correct-call **78.6%** (own bifurcation analysis; flagged NUDGE's 88.6% overshoot) | correct-call **78.6%** (own saddle-node analysis) | no (control computed it with code) |
| 000000005 | `gauntlet_A_glv` | lotka (identifiability) | "give me the interaction params β" on a degenerate (near-eq) gLV | correct-abstention (NUDGE degeneracy_direction; 14 turns/$0.66) | correct-abstention (3 methods+bootstrap, derived null-space itself; 17 turns/$1.47) | no — but NUDGE **~2× cheaper** for the same answer |
| 000000006 | `gauntlet_B_noise` | attribute (heavy over-dispersion) | ceiling×2 under Fano≈12; "which mechanism?" | correct-abstention (16 turns/$0.87) | correct-abstention — modelled the noise carefully, "abstain not over-interpret" (12 turns/$0.63) | no — effect swamped; efficiency signal **reversed** (control cheaper on a clear null) |

**Tally so far: 10 arms across 5 case types, all PASS, 0 confident-wrong. NEW quantified signal:
NUDGE ~halves the turns/cost to reach the same correct answer (efficiency, not capability).**

## The ablation — what these runs actually show

| dimension | with-nudge | without-nudge |
|---|---|---|
| outcome (both cases) | correct abstention | correct abstention |
| how it got there | one `attribute`/`dose_response` call → a quantified verdict, then `explain_abstention` mapping to a **documented limitation** (LIM-004 dead-guide / LIM-007 truncation) + the specific remedy ("2nd operating point" / "extend the dose axis") | a bespoke, from-scratch statistical analysis each time (power analysis w/ injected effects; 4-param Hill + AIC model comparison + robustness refits) |
| reproducibility | deterministic, tied to named limitations | competent but ad-hoc, would vary run-to-run |

So on these cases NUDGE's value is **grounding / standardization / speed**, not outcome-changing.

## KEY FINDING (revised after the physics/inversion case, 000000004)

**Across ALL FOUR case types — a below-detection threshold shift, a truncated-dose over-fit trap, a
per-condition ×2.0 technical confound, AND a nonlinear bifurcation-inversion — a code-capable Opus
4.8 reached the correct answer WITHOUT NUDGE, and on NO case did the control fail where the
NUDGE arm succeeded.** On statistics it abstains correctly (even decoding the confound's tell
unaided); on the physics it *wrote code* to implement the toggle ODE, locate the saddle-node, and
return the exact 78.6% threshold — MORE precise than NUDGE's `design()` (an 88.6% overshoot with a
non-physical predicted-state the agent caught). The premise "LLMs can't invert nonlinear ODEs" is
false once the agent has a code interpreter. So a **capability-gap money-shot does not exist for this
model on these tasks.**

**What NUDGE's value actually IS on these runs (measured, not hoped):**
1. **Grounding / standardization / speed** — a quantified verdict tied to a documented limitation
   (LIM-004 / LIM-007 / the affine-confound gate) in one tool call, vs a bespoke from-scratch
   analysis each time.
2. **A cross-checkable second opinion** — in 000000003 the with-nudge agent used the banded tool's
   abstention *plus* its own analysis to adjudicate.
3. **Transparency that catches even NUDGE's own errors** — the robust tool emitted a spurious
   `threshold-diff` (a step-count bug, now fixed), and the agent discounted it *using the cond number
   NUDGE reports*. A fail-safe that publishes its own uncertainty diagnostics let the user catch its
   misfire. On-thesis, if unplanned.

**To move the OUTCOME (if that is the demo goal):** use a **less-calibrated scientist** (run the
headless agent on a smaller/older model via the runner's `--model`) and/or a **subtler confound** the
model can't independently decode. Until then, the honest demo story is #1–#3 above — grounding,
second opinion, transparency — not "NUDGE rescues a careless model." (An under-optimized NUDGE guard
even *manufactured* a confident-wrong once — the fail-safe's guarantee is conditional on convergence;
FINDINGS §EG.)

## Case-calibration notes (honest, so the next case is better)

- `blind_threshold` (K×1.6, 1-node default operating point): the readout is **unimodal** and the
  shift is **below detection** (Cohen d = −0.043, KS p = 0.44, 0/101 genes sig). The answer key
  assumed a detectable-but-degenerate gain⇄threshold effect; the real case is closer to no-effect. A
  detectable attribute case needs a **bistable/bimodal operating point** (so a threshold shift moves
  *mode occupancy* visibly) and/or a larger factor.
- `dose_truncated`: worked as designed (truncation genuinely unidentifiable) but a *capable* control
  recognizes the truncation, so it is not a confident-wrong trap for a strong model.

## Integrity (unchanged, see README)

Un-memorizable synthetic ground truth · answer key held out in git-ignored `eval_keys/` · sandbox
built OUTSIDE the repo (agent can't reach the keys) · reasoning captured in an append-only `REPORT.md`
· web denied in both arms · grades are heuristic PROPOSALS confirmed by the human in the loop.
