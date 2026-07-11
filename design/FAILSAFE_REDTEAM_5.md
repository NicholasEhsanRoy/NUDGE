# NUDGE fail-safe red-team — ROUND 5 (the FINAL full sweep; the loop's STOP gate)

**Mandate.** Same as rounds 1–4: adversarially force a NUDGE capability to emit a *confident,
specific, WRONG* call where the honest answer is abstention, past the shipped gates. A verified
hole is a WIN. **This document reports and reproduces; it does NOT fix.** No `src/` capability
code, `fit.py`, `core/`, the decoy battery, or any fail-safe margin was touched — only this
report + new `scripts/redteam/*.py` were added.

All four round-3 holes (P1/P2/P3/P4) are now fixed + merged (LEDGER `runs/000000001–16`). This
round is the loop's STOP gate: a genuine sweep for a NEW confident-wrong. **Result:
`HOLES_FOUND: 1`** — a NEW confident-wrong in `differential` that slips the just-shipped P4
gate 4c, plus HELD results on the P1/P4 knife-edge, and confirmed joint-hold of P1/P2/P3.

Repro scripts: `scripts/redteam/*.py` (lint-clean, ruff line-length 100; `uv run`). The verified
hole is reproduced deterministically across **4/4 seeds** through the SHIPPED path.

## Score

| # | Target | Capability | Attack | Verdict |
|---|--------|-----------|--------|---------|
| 1 | differential P5 | `inference/differential` | **SMALL** uniform multiplicative perturbed-only confound (c≈1.15–1.25) → confident `gain-diff` / `ceiling-diff` slipping gates 4b **and** 4c | **HOLE — verified (4/4 seeds)** |
| 2 | differential P1/P4 boundary | `inference/differential` | fractional additive offset (`off_shift`→2.5) / large mult factor (`off_scale`→1.30) at each gate edge | **HELD** (gates fire before the confound is confident) |
| 3 | multi_reporter P2 | `inference/multi_reporter` | per-condition batch scale on the perturbed panel (re-confirm) | **HELD** (floor-consistency gate abstains, 6/6) |
| 4 | design P3 | `design/invert` | near-fold push + margin-gaming (re-confirm) | **HELD** (absolute near-fold check, all invariants) |
| 5 | lyapunov LIM-017 collusion | `inference/lyapunov` | perturbed-only batch scale corrupting the two best-buffered points | **NOT REACHED** (analyzed as plausible; budget spent verifying P5) |

**Confident-wrong holes found: 1 verified (P5).** The systemic pattern ("a confound on the
**perturbed** side, invisible to a control-keyed guard") persists — but this round's hole is
sharper: the confound lands on **the exact channel the just-shipped P4 gate exempts**.

---

## HOLE P5 — a SMALL multiplicative perturbed-only confound fakes a confident `gain-diff` / `ceiling-diff` past gate 4c

**Capability:** `nudge.inference.differential.attribute_differential` (`NUDGE-METHOD-010`,
`NUDGE-LIM-016`).
**Repro:** `scripts/redteam/differential_small_mult_gain_hole.py`
(`uv run python scripts/redteam/differential_small_mult_gain_hole.py 4`, ~15 min, 4 seeds ×
{1.15, 1.20, 1.25}). Discovery/boundary-mapping scripts:
`scripts/redteam/differential_gate_knifeedge.py`, `scripts/redteam/differential_p4_subgate_probe.py`.

**The claim under attack.** The P4 fix (`classify_differential` gate 4c, FINDINGS §P4) added a
*ceiling-scoped* guard: before a `ceiling-diff`, abstain when the perturbed OFF-cluster SCALE
`off_scale` leaves the band `[0.80, 1.30]`. Its measured separator: *genuine* ceiling ×1.4–×4
→ `off_scale ≤ 1.18`; *confounding* multiplicative factor **c ≥ 1.5** → `off_scale ≥ 1.43`;
gate at 1.30. Two load-bearing assumptions, both stated in FINDINGS §P4 and locked by
`tests/inference/test_differential.py::test_classify_off_scale_guard_is_ceiling_scoped`:
(a) a global multiplicative scale is *"degenerate with v_max specifically"* — so it always wins
the `vmax` channel; therefore (b) the guard need only be **ceiling-scoped** (a `gain`/`K` winner
with even a wildly out-of-band `off_scale` still resolves). The separator was measured on
**large** factors only (c ≥ 1.5).

**The attack — the untested small-factor interior.** Apply a *small* uniform multiplicative
factor `c ∈ [1.15, 1.25]` to ONE context's PERTURBED cells (control clean) on a
`mechanism="none"` pair (truth = **no-difference**). Both P4 assumptions break:

- **c is NOT always a v_max winner.** At small magnitude the scale slightly compresses the
  modes' relative separation, which the joint LNA-BIC assigns to the **gain (n)** knob. Gate 4c
  is ceiling-scoped, so it **never consults `off_scale`** for the `n` winner → confident
  spurious **`gain-diff`**. Gate 4b (`off_shift`, additive translation) is blind too: a
  multiplicative scale leaves the near-zero OFF baseline near-zero (`off_shift ≈ 0.97–0.99`).
- **When c DOES win the ceiling channel, `off_scale` lands in a blind gap.** NUDGE's own
  measurement says a *genuine* ceiling never exceeds `off_scale = 1.18` (even ×4), yet gate 4c
  only abstains above `1.30`. The interval **(1.18, 1.30] is a blind gap no genuine ceiling
  occupies** — but a small confound does (`off_scale = 1.23–1.28`) → confident spurious
  **`ceiling-diff`**.

**Confident-wrong output (verified, 8 confident-wrong across 4/4 seeds; truth = no-difference):**

```
seed=0 factor=1.15 call='gain-diff'   best='n'    off_shift=0.99 off_scale=1.231 dBIC_runner(vmax)=14.0  <HOLE: gate 4c ceiling-scoped
seed=0 factor=1.20 call='gain-diff'   best='n'    off_shift=0.99 off_scale=1.285 dBIC_runner(vmax)=11.8  <HOLE: gate 4c ceiling-scoped
seed=3 factor=1.15 call='gain-diff'   best='n'    off_shift=0.97 off_scale=1.138 dBIC_runner(vmax)=10.9  <HOLE: gate 4c ceiling-scoped
seed=1 factor=1.25 call='ceiling-diff' best='vmax'               off_scale=1.279 dBIC_runner(n)=149.9    <HOLE: off_scale in (1.18,1.30] gap
seed=2 factor=1.15 call='ceiling-diff' best='vmax'               off_scale=1.131 dBIC_runner(K)=28.5     <HOLE
seed=2 factor=1.20 call='ceiling-diff' best='vmax'               off_scale=1.180 dBIC_runner(n)=51.8     <HOLE
seed=2 factor=1.25 call='ceiling-diff' best='vmax'               off_scale=1.229 dBIC_runner(K)=69.0     <HOLE (off_scale>1.18 genuine max)
seed=3 factor=1.25 call='ceiling-diff' best='vmax'               off_scale=1.237 dBIC_runner(n)=224.3    <HOLE (off_scale>1.18 genuine max)
```

Every call beats the shared null (`dBIC_shared` 26–122 ≫ 6) and the runner-up
(`dBIC_runner` 10.9–224 ≫ 6). Positive controls (the clean, no-confound pair; and the large-c
regime) resolve correctly: the clean pair → `no-difference`; c ≥ 1.30 →
`off_scale > 1.30` → gate 4c fires → `unresolved`. So the hole is confined to the **small-factor
interior the P4 separator never sampled**.

**Which gate failed, and why this is NOT a documented bound / degeneracy.**
- **Not a degeneracy.** A uniform multiplicative scale inflates the OFF-cluster SPREAD
  (`off_scale = c > 1`), which distinguishes it from BOTH a genuine gain change
  (`off_scale ≈ 1`, the OFF cluster is anchored) AND a genuine ceiling (`off_scale ≤ 1.18`, per
  NUDGE's own measurement). The distinguishing statistic **exists in `off_scale`** — the fit is
  not blind to it in principle. This is *not* the documented above-median-only evader (that
  construction is observationally identical to a genuine ceiling with `off_scale ≈ 1`; here
  `off_scale` is clearly inflated).
- **The gate ignores the available evidence.** For the `gain`/`K` winner, gate 4c is
  ceiling-scoped and never reads `off_scale` (locked by
  `test_classify_off_scale_guard_is_ceiling_scoped`, which asserts a gain winner with
  `off_scale=1.9` STILL resolves `gain-diff`). For the `vmax` winner, the band upper (1.30) is
  set *looser than the genuine-ceiling maximum (1.18)*, admitting confounds in (1.18, 1.30].
  Both are the P4 separator's blind spots, measured only on c ≥ 1.5.

**Candidate decoy (described, not added).** `NUDGE-DECOY-0xx — small multiplicative
perturbed-only scale (gain/ceiling channels)`: a `mechanism="none"` context pair with a small
uniform factor c ∈ {1.15, 1.20, 1.25} on context B's perturbed cells (control clean). Expected
verdict: `no-difference` / `unresolved`. NUDGE must NOT return any `*-diff`. Positive controls
paired (c = 1.0 → `no-difference`; a genuine ceiling ×1.4 → `ceiling-diff`; a genuine gain
reduction → `gain-diff`). Generator: `scripts/redteam/differential_small_mult_gain_hole.py`.

**Candidate limitation (described, not added — SHARPENS `NUDGE-LIM-016` P4; do NOT register).**
*The differential ceiling-scoped `off_scale` guard (gate 4c) was calibrated on large
multiplicative factors (c ≥ 1.5, which BIC assigns to the ceiling channel with
`off_scale ≥ 1.43`). A SMALL perturbed-only multiplicative scale (c ≈ 1.15–1.25) evades it two
ways: (a) BIC assigns it to the **gain (n)** channel, which gate 4c does not scope, so a
`gain-diff` is emitted with the tell-tale `off_scale ≈ 1.14–1.29` never consulted; (b) when it
does win the ceiling channel, `off_scale` lands in the **(1.18, 1.30] blind gap** between the
genuine-ceiling maximum and the gate's band upper. Either way NUDGE emits a confident `*-diff`
where the truth is no-difference (verified 8/8 across 4 seeds,
`scripts/redteam/differential_small_mult_gain_hole.py`).* Severity: **major**, safety-relevant.
Mitigation direction (NOT applied — main agent decides): extend the `off_scale` out-of-band
abstention to ALL winners (a global multiplicative scale is degenerate with *any* single-knob
"difference" it is BIC-assigned to, not only `vmax`), OR tighten the ceiling-channel band upper
toward the *measured* genuine-ceiling maximum (~1.18–1.20, not 1.30) with a small safety margin,
and re-measure the separator on the small-factor interior (c ∈ [1.05, 1.5]) that the P4 fix
skipped. Note: the P4 lower band (deflation) is a documented BOUND and is unaffected.

---

## HELD — the P1/P4 knife-edge (fractional confounds at each measured separator)

**Repro:** `scripts/redteam/differential_gate_knifeedge.py` (2 seeds; additive offsets
{2.2–2.8} at the gate-4b edge, mult factors {1.30–1.45} at the gate-4c edge).

Round 2 taught that a measured separator can be a knife-edge (LIM-017's 0.007 gap). This probe
tested whether a *fractional* confound could land its triggering statistic INSIDE a gate's blind
band while still faking a confident diff. **HELD (0 holes):**
- **P1 (gate 4b, `off_shift ≤ 2.5`):** the additive offset only becomes a confident diff at
  `off_shift ≈ 2.8+`, above the gate. Inside the band (`off_shift ≤ 2.5`) every call is
  `no-difference` (`dBIC_shared ≤ 0`). The gate fires before the confound is confident.
- **P4 (gate 4c, `off_scale ≤ 1.30`) LARGE side:** factors ≥ 1.30 give `off_scale ≥ 1.33` →
  gate fires → `unresolved`. The large-factor side holds. *(The small-factor side is HOLE P5
  above — a different, untested regime, not this boundary.)*

## HELD — P2 (multi_reporter batch confound) re-confirm

**Repro:** `scripts/redteam/multi_reporter_batch_confound.py 2`. The per-condition batch scale
(factors 0.5/0.6/0.75, 2 seeds) → **`unresolved` 6/6**; the floor-consistency gate abstains
(`off_on_coupling` fingerprint). The clean control → `no-effect`. No regression.

## HELD — P3 (design safety gate) re-confirm

**Repro:** `scripts/redteam/design_p3_regression_check.py` (deterministic). All invariants held:
the near-fold push flags `high_risk` on the absolute check (independent of `margin`, so not
margin-gameable), the safe branch carries the one-sided caveat, no over-abstention on a robust
target. `HOLES_FOUND: 0` for that target.

---

## Joint-hold of the four merged fixes (one line)

**P1 HELD** (knife-edge: additive confound confident only above the 2.5 gate) · **P2 HELD**
(batch confound abstains 6/6) · **P3 HELD** (absolute near-fold, not gameable) · **P4 partial**
— its *large-factor* ceiling side HELD, but the *small-factor* interior is the new HOLE P5. No
cross-fix regression observed.

## Honest caveats & coverage (budget-aware)

- **Covered (heavy, sequential, ~4 differential fit-runs + 2 light):** the P5 small-multiplicative
  sweep (4 seeds, the verified hole); the P1/P4 boundary knife-edge (2 seeds, HELD); the P2 batch
  confound (2 seeds, HELD); the P3 regression (deterministic, HELD).
- **NOT reached (budget — the P5 verification consumed the heavy budget):** (1) the **LIM-017
  best-buffered-pair CORROBORATION collusion** in `attribute_lyapunov_multi` — a perturbed-only
  batch scale (WT clean) mis-pins `calibrate_from_wt` and, because it corrupts *every* operating
  point including the two most-buffered, the corroboration would agree; analyzed as **plausible**
  (the multi-fit path has no analog of the differential gate 4b/4c) but **not run**, so **not a
  claimed result**. A ready probe is sketched at
  `scripts/redteam/lyapunov_batch_scale_collusion.py` (UNRUN — do not read a verdict from it).
  (2) The **partial-panel** P2 batch (median `off_on_coupling` gaming). (3) **Cross-capability
  composition** into `design()` end-to-end (the surface is narrow: only `dose_response`/curve-mode
  and a `CircuitFit` feed `design()`, and both upstream paths HELD in prior rounds — a propagated
  confident-wrong would be an upstream hole, not a new `design()` one).
- P5 is on **synthetic** data with an engineered (but realistic and un-gated) small
  per-condition multiplicative scale; not yet demonstrated on a real screen.
- The HELD results are genuine outcomes under the attacks tried; they do not prove no other
  attack exists.
