# NUDGE automated-scientist eval — LEDGER (live state · resume here)

**Governing rule: honesty above all** — grade CALIBRATION, not point-accuracy; a confident-WRONG
mechanism is the only hard fail; a calibrated abstention is a PASS. See `README.md` for the
protocol. Each run is an immutable record under `runs/`; this file indexes them and never rewrites
history (append new rows; correct forward).

---

## ▶ RESUME POINTER

**Status: 2 cases run (4 arms), 0 confident-wrong. KEY FINDING below changes the next case design.**

- **Next:** build a **confounded** blind case — the money-shot the clean cases can't produce (see
  "Key finding"). Concretely: a `differential`-surface case with a per-condition **affine technical
  nuisance** on the perturbed cells (the P1/P4/P5 family we just hardened), where a careful generic
  analysis is *genuinely baited* into calling a `ceiling`/`gain` change but NUDGE's guard abstains.
  Requires a new `@register("differential")` builder in `blind_harness.py` that injects the confound
  + holds out "truth = no-difference".
- Then re-run the ablation on that case and record it as `runs/000000003-*`.

---

## Run index (append-only — newest at the bottom; never edit a past row)

| # | case | surface | ground truth | with-nudge | without-nudge | money-shot? |
|---|------|---------|--------------|------------|---------------|-------------|
| 000000001 | `blind_threshold` | attribute | threshold K×1.6 (below detection here) | correct-abstention (no-effect) | correct-abstention (no-effect) | no (near-null) |
| 000000002 | `dose_truncated` | dose-response | switch, truncated below inflection | correct-abstention (unresolved/LIM-007) | correct-abstention (extend range) | no (control didn't bite) |

**Tally so far: 4 arms, 4 PASS (all correct-abstention), 0 confident-wrong.**

## The ablation — what these runs actually show

| dimension | with-nudge | without-nudge |
|---|---|---|
| outcome (both cases) | correct abstention | correct abstention |
| how it got there | one `attribute`/`dose_response` call → a quantified verdict, then `explain_abstention` mapping to a **documented limitation** (LIM-004 dead-guide / LIM-007 truncation) + the specific remedy ("2nd operating point" / "extend the dose axis") | a bespoke, from-scratch statistical analysis each time (power analysis w/ injected effects; 4-param Hill + AIC model comparison + robustness refits) |
| reproducibility | deterministic, tied to named limitations | competent but ad-hoc, would vary run-to-run |

So on these cases NUDGE's value is **grounding / standardization / speed**, not outcome-changing.

## KEY FINDING (drives the next case)

**Opus 4.8 is already well-calibrated on clean synthetic abstention cases — it abstains correctly
*without* NUDGE.** Across a below-detection threshold shift and a truncated dose curve (a *designed*
switch-over-fit trap), the control arm never committed a confident-wrong; it built its own rigorous
identifiability analysis and abstained. Therefore the "NUDGE prevents a confident-wrong the control
commits" demonstration **cannot come from a clean case** — the control is too careful.

**Where NUDGE's UNIQUE value lives (the next case):** a **confound** a careful generic analysis does
not know to check for — a per-condition **affine technical nuisance** (scale/offset on the perturbed
cells only) that *aliases onto a real mechanism* (a scale looks like a ceiling `v_max` change; an
offset shifts modes → threshold/gain). A generic analyst sees a cleanly shifted/scaled distribution
and calls the apparent mechanism (a confident-wrong); NUDGE's engineered guards (differential gates
4b/4c, the opt-in Earn-Guard — the exact P1/P4/P5 hardening just merged to `main`) recognize the
affine confound and **abstain**. That is the outcome-changing money-shot, and it ties the demo
directly to the depth-of-execution hardening work.

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
