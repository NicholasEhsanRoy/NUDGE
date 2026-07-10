# NUDGE fail-safe red-team — ROUND 3 (the not-yet-exercised capabilities)

**Mandate.** Same as rounds 1–2 (`design/FAILSAFE_REDTEAM.md`, `..._2.md`): adversarially
force a NUDGE capability to emit a *confident, specific, WRONG* call where the honest answer
is abstention, past the gates. A verified hole is a WIN (a documented limitation/decoy).
**This document reports and reproduces; it does NOT fix.** No `src/` capability code,
`fit.py`, `core/`, the decoy battery, or any fail-safe margin was touched — only this report
+ new `scripts/redteam/*.py` were added.

Rounds 1–2 hit the peripheral attribution capabilities, the core engine, and the round-1
fixes. **Round 3 targets the four capabilities NOT yet adversarially exercised** —
`differential`, `multi_reporter`, `design/invert`, `hidden_node` — plus a round-2-fix
stress-test on `constitutive`.

Repro scripts: `scripts/redteam/*.py` (lint-clean, ruff line-length 100; `uv run`). Every
verified hole is reproduced with ≥ 2 seeds (or deterministically) through the SHIPPED path.

## Score

| # | Target | Capability | Attack | Verdict |
|---|--------|-----------|--------|---------|
| 1 | differential (#1) | `inference/differential` | ADDITIVE offset on ONE context's **perturbed** cells (control clean) → confident `gain-diff` | **HOLE — verified** |
| 2 | multi_reporter (#2) | `inference/multi_reporter` | per-CONDITION batch/depth scale on the **perturbed** panel → confident `ceiling` | **HOLE — verified** |
| 3 | design/invert (#3) | `design/invert.design` | intervention pushed INTO the near-fold regime cleared as "OK, stays away from the fold" | **HOLE — verified** |
| 4 | hidden_node (#4) | `inference/hidden_node` | drive a huge off-axis residual → bare positive / lone-leading hidden-node | HELD |
| 5 | constitutive (#5) | `inference/constitutive` | SECOND route to `biological-switch` on a linear circuit — misspecified **dispersion** | HELD |

**Confident-wrong holes found: 3 verified** (differential additive-perturbed-offset;
multi_reporter per-condition batch scale; design safety-gate absolute-proximity blind spot).
`hidden_node` **HELD** — its rank cap + always-present competing causes make a lone-leading
or bare-positive hidden-node claim structurally unreachable. The constitutive alt-route
(misspecified dispersion) also **HELD** — the `structure_ratio` (≥ 5×-span) gate scales with
the noise and catches it.

**A unifying theme across holes 1 & 2 (and the round-1/2 confounds).** NUDGE's confound
guards are keyed on the **control / WT** condition (per-context depth from each control;
the multi-reporter consistency check on control curves). A technical confound applied to the
**perturbed** condition *only* — the exact place a real batch/ambient/depth artifact aligned
with the perturbation lands — is **structurally invisible** to a control-keyed guard, and
maps straight onto a mechanism call. This is the same class as round-1 HOLE 2 (additive
ambient on A+B) and round-2 HOLE 1 (control-vs-population capture scale), now shown to hit
`differential` and `multi_reporter` too.

---

## HOLE 1 — an additive offset on ONE context's PERTURBED cells fakes a confident `gain-diff`

**Capability:** `nudge.inference.differential.attribute_differential` (`NUDGE-METHOD-010`,
`NUDGE-LIM-016`). This is the differential hole round 1 flagged as "the most likely
differential hole" and never tested.
**Repro:** `scripts/redteam/differential_additive_confound.py` (`uv run python …`, ~10 min;
2 seeds × {0,1,2,3,5} additive offset).

**The claim under attack.** `NUDGE-LIM-016`: a depth/batch difference aligned with the
context axis is defended by pinning depth **per context from each context's own control**
(`calibrate_from_wt`) and abstaining (gate 2 of `classify_differential`) when the two
controls' pinned depths differ beyond `depth_ratio_max=1.5`. The stated exception is that a
cleanly-resolved **threshold/gain** difference "reshapes the distribution, orthogonal to a
global scale, so it survives a depth difference and is still callable."

**The attack.** Add a constant additive offset to context B's **perturbed** cells only —
its control left **clean** (an ambient/batch shift concentrated on the perturbed cells of
one context, aligned with the context axis). Because the depth guard's `depth_ratio` is
computed from the two **controls** (both clean), it stays ≈ 1.009 and **gate 2 never
engages**. The offset shifts B's perturbed modes and compresses their relative separation,
which the joint LNA-BIC fit reads as **reduced cooperativity** — a per-context `n`
difference.

**Confident-wrong output (verified, 3 confident-wrong calls across 2 seeds):**

```
seed=0 [offset=0.0 control] call='no-difference'  depth_ratio=1.009 selected='shared'  # positive control
seed=0 [offset=3.0]         call='gain-diff'  <== HOLE  depth_ratio=1.009 best_diff='n'
    dBIC vs shared=21.2   dBIC vs runner (vmax)=29.6   (n_A=5.09 vs n_B=2.83, log2 -0.85)
seed=0 [offset=5.0]         call='gain-diff'  <== HOLE  dBIC vs shared=201.4  dBIC vs runner (K)=9.4
seed=1 [offset=5.0]         call='gain-diff'  <== HOLE  dBIC vs shared=231.0  dBIC vs runner (K)=16.9
```

Truth = **no-difference** (`mechanism="none"`). The clean (offset 0) run correctly returns
`no-difference` at both seeds; adding the perturbed-only offset manufactures a confident
`gain-diff` (dBIC vs shared 21–231 ≫ margin 6.0; beats the runner-up by 9–30 ≫ 6.0).

**Which gate failed, and why it is worse than a bypass.** Gate 2 (the LIM-016 depth guard)
never engages because it keys on the controls, not the perturbed cells. But even if it *did*
see a depth difference, the confound fakes **`gain-diff`** — and gate 2 **explicitly exempts
gain/threshold diffs** as "depth-robust, orthogonal to a global scale." So the offset lands
on the *one channel the guard trusts most*: the guard's own exception would wave it through.
The premise "only a global multiplicative scale confounds a mechanism difference, and it can
only mimic *ceiling*" is false — an **additive** perturbed-condition offset mimics **gain**.

**Candidate decoy (described, not added).** `NUDGE-DECOY-0xx — additive offset on one
context's perturbed cells`: a `mechanism="none"` context pair with a fixed additive offset
on context B's perturbed activity (control clean). Expected verdict: `no-difference` /
`unresolved`. NUDGE must NOT return any `*-diff`. Positive control paired (offset 0 →
`no-difference`), isolating the offset as the culprit. Generator:
`scripts/redteam/differential_additive_confound.py`.

**Candidate limitation (described, not added — SHARPENS `NUDGE-LIM-016`; next free id
`NUDGE-LIM-021`, do NOT register).** *The differential depth/batch guard is keyed on the
per-context **controls** and models the confound as a global multiplicative scale (mimicking
ceiling). An **additive** technical offset on a single context's **perturbed** cells is
(a) invisible to the control-derived `depth_ratio`, so gate 2 never engages, and (b) mimics
a **gain** difference, which gate 2's exception explicitly treats as depth-robust and
callable — so it is confidently mis-called `gain-diff` where the truth is `no-difference`
(verified, `scripts/redteam/differential_additive_confound.py`, 3 confident-wrong across
2 seeds).* Severity: **major**, safety-relevant. Mitigation direction (NOT applied — main
agent decides): pin depth/baseline per context from **both** the control AND the perturbed
cells' OFF-mode (a perturbed-cell library/baseline diagnostic), or require the per-context
control to come from the same library as its perturbed cells AND add an OFF-baseline-shift
abstention (the module already computes `off_shift_ratio` but explicitly declares it
non-load-bearing — this hole is the argument for making a *relative* version of it a gate,
or at least abstaining when the OFF baseline moves between contexts inconsistently with the
called knob).

---

## HOLE 2 — a per-condition batch scale on the perturbed panel fakes a confident `ceiling`

**Capability:** `nudge.inference.multi_reporter.attribute_multi_reporter`
(`NUDGE-METHOD-008`, `NUDGE-LIM-014`). The round-1/2 "shared-additive analogue" flagged and
never tested.
**Repro:** `scripts/redteam/multi_reporter_batch_confound.py` (`uv run python …`, ~1 min;
2 seeds × {0.5, 0.6, 0.75} batch factor).

**The claim under attack.** The multi-reporter consistency guard (gate 1 of
`classify_multi_reporter`, `NUDGE-LIM-014`) "strengthens the guarantee": a spurious mechanism
must be consistent across ALL reporters, and if the reporters do not share one latent it
abstains `off-model`. A ceiling call means "every reporter's ON amplitude drops by the same
fraction."

**The attack.** The consistency guard computes `panel_r2` / `consistency_ratio` / the
per-reporter R² from the **CONTROL** curves only. A confound on the **perturbed** condition
is invisible to it. And multi_reporter pins each reporter's affine from the control and fits
the perturbed panel with **no per-condition depth/batch normalization** (unlike
`differential`). Apply a single multiplicative factor `c` to **every** perturbed reporter —
a batch / sequencing-depth / instrument-gain difference between the control-condition and
perturbed-condition measurement, consistent across the panel. `c·(floor + gain·f)` is
aliased to a shared latent-ceiling change `A = c`: every reporter's ON amplitude scales by
the same fraction — the *exact* ceiling signature.

**Confident-wrong output (verified, 6/6 across 2 seeds × 3 factors):**

```
seed=0 [factor=1.00 control] call='no-effect'  panel_r2=0.999 consistency=1.05   # positive control
seed=0 [factor=0.50] call='ceiling' <== HOLE  A=0.496 knob_m=756.9 eff_m=1315.5 ci_log2=(-1.03,-1.00)
seed=0 [factor=0.60] call='ceiling' <== HOLE  A=0.597 knob_m=414.6 eff_m=640.1  ci_log2=(-0.76,-0.73)
seed=0 [factor=0.75] call='ceiling' <== HOLE  A=0.747 knob_m=130.1 eff_m=173.7  ci_log2=(-0.44,-0.41)
seed=1 [factor=0.50] call='ceiling' <== HOLE  A=0.496 knob_m=1002  eff_m=1712    ci_log2=(-1.03,-0.99)
```

Truth = **no-effect** (`mechanism="none"` — the perturbation did not move the latent). The
recovered `A_perturbed/A_wt` equals the batch factor `c` to 3 digits; the bootstrap CI
excludes 0; the knob margin (>100) and effect margin (>170) crush the gates. Positive
controls (factor 1.0) correctly return `no-effect`. **Not a tiny-floor artifact** — it
survives realistic floors: at `floor_range=(0.2, 0.6)`, factor 0.5 → still `ceiling`,
knob_margin 6.6, effect_margin 20.9 (both ≫ their thresholds).

**Which gate failed.** None can see it. The consistency guard is control-only; there is no
per-condition depth normalization; the knob/effect/CI gates all see a clean, consistent
`ceiling` because a per-condition scale genuinely *is* consistent across the panel. The
guard's "strengthened guarantee" holds against a confound that breaks the shared latent —
but a confound *consistent with* the shared latent (a uniform per-condition scale) sails
through.

**Candidate decoy (described, not added).** `NUDGE-DECOY-0xx — per-condition batch scale on
the perturbed reporter panel`: a `mechanism="none"` panel whose perturbed curves are all
scaled by one factor `c` (control clean). Expected verdict: `no-effect` / `unresolved`.
NUDGE must NOT return `ceiling`. Positive control paired (factor 1.0 → `no-effect`).
Generator: `scripts/redteam/multi_reporter_batch_confound.py`.

**Candidate limitation (described, not added — SHARPENS `NUDGE-LIM-014`).** *The
multi-reporter consistency guard validates that the reporters share ONE latent using the
**control** curves only, and the module applies **no per-condition depth/batch
normalization** before fitting the perturbed panel. A uniform multiplicative batch/depth
difference between the control-condition and perturbed-condition measurement (very common:
different plate/day/instrument-gain/sequencing depth), consistent across the panel, is
aliased 1:1 to a shared latent-ceiling change `A = batch_factor` and produces a confident
`ceiling` (CI excludes 0, margins ≫ thresholds) where the truth is `no-effect` (verified,
`scripts/redteam/multi_reporter_batch_confound.py`, 6/6 across 2 seeds × 3 factors, robust
to realistic floors).* Severity: **major**, safety-relevant. Mitigation direction (NOT
applied): normalize the perturbed condition to the control's depth from a shared reference
(spike-in / housekeeping reporter / a designated no-response reporter), or add a
scale-consistency gate that abstains when a *ceiling* win is indistinguishable from a global
perturbed-condition rescale (e.g. require an independent depth anchor before a `ceiling`
call, mirroring the differential per-context depth pin the multi-reporter path lacks).

---

## HOLE 3 — design()'s safety gate clears an intervention pushed INTO the near-fold regime

**Capability:** `nudge.design.invert.design` (`NUDGE-METHOD-007`, `NUDGE-LIM-013`) — the
flagship inverse-design verb. **The highest-harm output class: a confident-wrong SAFETY flag
on a proposed intervention.**
**Repro:** `scripts/redteam/design_safety_gate_absolute_proximity.py` (`uv run python …`,
deterministic — an optimization to a fixed target, no seed dependence).

**The claim under attack.** `design()`'s bifurcation **safety gate** "flags an intervention
that pushes a bistable switch toward a tipping point, reusing the shipped Cap-5
`bifurcation_proximity` dial." The stated protection is `SafetyReport.high_risk_of_instability`
/ `crosses_fold`.

**The blind spot.** `_safety_report` sets `high_risk_of_instability` **only** when the
*increase* in proximity `delta = proximity_after − proximity_before` exceeds `margin`
(default 0.15). It **never** compares the **absolute** `proximity_after` against the shipped
near-fold threshold `bifurcation.NEAR_FOLD = 0.55`. So an intervention that moves a robust
switch **across** 0.55 into the near-fold regime — but by a sub-margin increment — is cleared
as safe, and the reason string literally reads "OK, stays away from the fold" while the
intervened circuit is in NUDGE's *own* near-fold zone.

**Construction.** base = `ras_switch_1node(n=2.0, vmax=3.0, K=1.5)` → proximity **0.500**
(`robust`). The reachable inversion scales `K` by ×0.667 (→ K=1.0) to reach the target ON
level; the intervened circuit has proximity **0.589** (`near-fold`), a rise of only **0.089
< margin 0.15**.

**Confident-wrong output (verified, deterministic):**

```
base proximity        = 0.500  (robust)
proposed delta        = edge[0].K  ×0.667   (closes 100% of the target gap)
safety.proximity_after = 0.589
safety.delta           = 0.089
safety.high_risk_of_instability = False
safety.crosses_fold             = False
design REASON: "... — safety: OK, stays away from the fold (proximity 0.50->0.59)."  <== HOLE

# independent classify_robustness on the SAME intervened circuit:
proximity_after = 0.589 -> 'near-fold'  ("proximity ≥ 0.55 — close to a saddle-node fold ...")
```

`design()` asserts "stays away from the fold" for a circuit its *own* `classify_robustness`
calls `near-fold`. A confident-wrong safety flag on a proposal — the highest-harm output.

**Aggravating factor (the caveat is dropped on the safe path).** In `_circuit_reason`, the
one-sided-lower-bound caveat ("a one-sided LOWER bound") is emitted only on the
`high_risk_of_instability` branch. The **safe** branch ("OK, stays away from the fold")
carries **no** one-sided caveat — yet the Cap-5 dial is a one-sided lower bound near the
fold (`NUDGE-LIM-012`), so `proximity_after` can *under*-report the true proximity. The one
path that most needs the "this is a lower bound, the truth may be worse" hedge is the one
that omits it.

**Which gate failed.** The safety gate's decision rule is `delta > margin`, a *relative*
test with no *absolute* near-fold check. An intervention can therefore land the switch in the
near-fold regime (or leave an already-near-fold base there) with a sub-margin delta and be
reported safe. The threshold `NEAR_FOLD` that `classify_robustness` uses to call the *same*
circuit `near-fold` is never consulted.

**Candidate decoy (described, not added).** `NUDGE-DECOY-0xx — design pushes a robust switch
across NEAR_FOLD with a sub-margin delta`: the deterministic construction above. Expected:
the plan's safety must NOT say "stays away from the fold"; it must flag near-fold (or abstain)
because `proximity_after ≥ NEAR_FOLD`. Positive control: a target reachable without raising
proximity above 0.55 → correctly "OK". Generator:
`scripts/redteam/design_safety_gate_absolute_proximity.py`.

**Candidate limitation (described, not added — SHARPENS `NUDGE-LIM-013`).** *design()'s
bifurcation safety gate flags only a proximity **increase** beyond `margin`; it never checks
the **absolute** `proximity_after` against `NEAR_FOLD`. An intervention that pushes a robust
switch across the near-fold threshold (or holds an already-near-fold base there) with a
sub-margin delta is reported "OK, stays away from the fold" — contradicting
`classify_robustness` on the identical circuit (verified, deterministic,
`scripts/redteam/design_safety_gate_absolute_proximity.py`). Compounding it, the "safe"
reason branch omits the one-sided-lower-bound caveat that near-fold proximities carry
(`NUDGE-LIM-012`), so the falsely-safe number is also presented without its "may be worse"
hedge.* Severity: **major**, safety-relevant (a confident-wrong on a *proposal*'s safety
label). Mitigation direction (NOT applied): make the safety verdict fire on `high_risk` OR
`proximity_after ≥ NEAR_FOLD` (an absolute near-fold check, not just a delta), route the
absolute-near-fold case through `classify_robustness`'s wording, and carry the `one_sided`
caveat on the SAFE branch too whenever `proximity_after` is a lower bound.

---

## HELD — hidden_node cannot be pushed to a bare positive or a lone-leading claim

**Capability:** `nudge.inference.hidden_node.diagnose_inadequacy` (`NUDGE-METHOD-009`,
`NUDGE-LIM-015`).
**Repro:** `scripts/redteam/hidden_node_lone_leading_probe.py`.

The attack drove a huge off-axis / neomorphic residual (up to 99) as the ONLY firing
diagnostic (`off_model=False`), trying to make hidden-node the single top-ranked cause or an
outright positive. **HELD 0/4.** Two structural guards hold jointly:

1. **Rank cap.** `_cause_hidden_node` returns at most `plausible` (never `leading`) — no
   input can raise it.
2. **Always-present competitors.** `_cause_not_a_switch`, `_cause_nonlinear_readout`, and
   `_cause_off_target` are ALWAYS added and are `plausible` whenever the trigger is a bare
   residual (`off_model=False`, `readout_flag` unset), so hidden-node is always in a ≥ 4-way
   tie at the top rank — never the lone leading answer. When `off_model=True`, "not-a-switch"
   becomes the lone `leading` cause and hidden-node stays `plausible`.

No `CandidateCause` text ever asserts a hidden node exists (checked against a
bare-positive phrase list). The abstention-only contract is structurally sound under the
attacks tried.

---

## Round-2-fix stress-test — constitutive alternate route (target #5)

**Attack:** a SECOND route to `biological-switch` on a LINEAR (n=1) circuit besides the
round-2 capture-efficiency mismatch — a **misspecified count dispersion** (the module trusts
`profile_circuit_n(dispersion=…)` as known; overdispersion is rarely known exactly).
**Repro:** `scripts/redteam/constitutive_dispersion_route.py` (full shipped budget; 2 seeds ×
true dispersion ∈ {0.1 matched, 0.4, 0.8}, assumed 0.1).

**HELD (0/6).** Every case (both seeds, all three true dispersions) returned `unresolved`;
0 confident-wrong `biological-switch`. Why it holds (a structural strength worth recording):
as the true dispersion rises above the assumed 0.1, the extra count spread inflates the
`n1_rejection` (0.0000 → ~0.0105–0.0118 at true_disp 0.8, marginally clearing
`reject_abs=0.01`), **but it inflates the WITHOUT-control profile span by the same token**
(5×span 0.0047 → ~0.021–0.025), so the second, *relative* gate — `n1_rejection ≥
structure_ratio(=5) × span_no_control` — is NEVER cleared, and the fail-safe conjunction
abstains. The dispersion confound moves both the signal and the noise floor together, so the
ratio gate is robust to it. (Contrast the round-2 capture-efficiency route, which mis-anchors
the reporter `Vmax` *without* inflating the without-control span, and so DID clear the ratio
gate.) This is a genuine win for the constitutive fail-safe under a second, realistic,
un-gated misspecification.

```
seed=0 true_disp=0.1 [matched] unresolved  n1_rej=0.0000 5xspan=0.0047 argmin_n=1.0  # positive control
seed=0 true_disp=0.4           unresolved  n1_rej=0.0007 5xspan=0.0202 argmin_n=1.5
seed=0 true_disp=0.8           unresolved  n1_rej=0.0105 5xspan=0.0210 argmin_n=2.0  # n1_rej>reject_abs but < 5xspan
seed=1 true_disp=0.8           unresolved  n1_rej=0.0118 5xspan=0.0251 argmin_n=2.0
```

---

## Honest caveats & coverage gaps

- All three holes are on **synthetic** data with engineered (but realistic and un-gated)
  confounds: an additive perturbed-condition offset (differential), a per-condition batch
  scale (multi_reporter), and a bistable switch tuned near the robust/near-fold boundary
  (design). None is yet demonstrated on a real screen.
- Holes 1–2 lie in the *class* the prior rounds already conceded ("a confound perfectly
  aligned with the condition is not separable"), but they **verify** that the concession is a
  **confident, specific mechanism call** (a `gain-diff` / a `ceiling`), not a graceful
  abstention, and — the sharp new point — that the confound lands on the channel each guard
  **trusts most** (gain, which the differential guard exempts as depth-robust; a *consistent*
  per-condition scale, which the multi-reporter consistency guard cannot see). That is a real
  sharpening of `NUDGE-LIM-016` / `NUDGE-LIM-014`, not just a re-statement.
- Hole 3 is a **logic** blind spot (relative-delta vs absolute-near-fold), independent of
  noise — it reproduces deterministically and contradicts NUDGE's own `classify_robustness`.
- **Not reached this round:** the LIM-017 best-buffered-pair CORROBORATION collusion attack
  (engineer the two *most-buffered* points to both be corrupted) — flagged as the highest-
  value remaining round-2-fix probe; the `design` curve-mode inversion (no safety gate by
  design); and an end-to-end `design`-off-a-confident-wrong-attribution chain (Hole 1/2 → a
  confident-wrong proposal), which follows by composition but was not driven end-to-end.
- The three HELD/HOLE results are genuine outcomes under the attacks tried; they do not prove
  no other attack exists.
