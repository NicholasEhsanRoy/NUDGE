# Fail-safe red-team — round 6 (FINAL FULL sweep, hardening loop)

**Role:** `nudge-red-team` (role 1), worktree-isolated. The FINAL full sweep — the loop's
STOP gate, run with all five round-3/4/5 holes (P3/P1/P4/P2/P5) fixed + merged (shared HEAD
`8e5f10f`).

**Result: `HOLES_FOUND: 1` — NOT a STOP.** The five merged fixes HELD jointly (no cross-fix
regression), but the highest-value UNREACHED probe from the prior final sweep
(`runs/000000017`, priority 1) — the LIM-017 best-buffered-pair **corroboration collusion on
WELL-BUFFERED points** in `attribute_lyapunov_multi` — was RUN this round and is a genuine,
reproduced confident-wrong. The loop resumes on the new hole (**P6**).

---

## The verified hole — P6 (lyapunov multi-point breaker, perturbed-only batch/depth scale)

| field | value |
|---|---|
| **id** | P6 |
| **capability** | `nudge.inference.lyapunov.attribute_lyapunov_multi` (the M3 multi-operating-point covariance BREAKER — the machinery sold as *more* trustworthy than one snapshot) |
| **truth** | a genuine **threshold** (`K` ×1.6) difference on a mutual-repression toggle; **NO** ceiling (`v_max`) change |
| **confound** | a per-condition **multiplicative batch/depth scale ×2.0 on the PERTURBED cells only** (WT/control clean) — a routine Perturb-seq lane/capture/guide-dependent depth difference |
| **confident-wrong output** | a bare `ceiling` (asserts a nonexistent `v_max` change AND misses the true threshold), gap ≫ `resolve_margin` 0.03, gate passing (`gate_all_ok=True`) |
| **reproduced** | full sweep `lyapunov_batch_scale_collusion.py 4`: **3/4 seeds** (1,2,3) `ceiling` at batch 2.0; focused `lyapunov_perturbed_batch_ceiling_hole.py 1 2`: **2/2 seeds**, byte-identical NLLs (deterministic) |
| **repro** | `scripts/redteam/lyapunov_perturbed_batch_ceiling_hole.py` (focused, authored + run this round) and the pre-existing `scripts/redteam/lyapunov_batch_scale_collusion.py` (RUN this round) |

### The exact failing output (focused repro, seeds 1 2 — my authored artifact)

```
seed=1 batch=1.0 [control]  gate_all_ok=True  label='unresolved'  gap=0.2203
    NLLs: gain=7.064  threshold=6.841  ceiling=7.061
seed=1 batch=2.0          gate_all_ok=True  label='ceiling'  gap=0.3204  <== CONFIDENT-WRONG HOLE
    NLLs: gain=8.840  threshold=8.607  ceiling=8.287
seed=2 batch=1.0 [control]  gate_all_ok=True  label='threshold'  gap=0.2244
    NLLs: gain=7.067  threshold=6.840  ceiling=7.064
seed=2 batch=2.0          gate_all_ok=True  label='ceiling'  gap=0.3061  <== CONFIDENT-WRONG HOLE
    NLLs: gain=8.898  threshold=8.602  ceiling=8.296
>>> confident-wrong holes: 2 / 2 seeds
```

Full sweep (`lyapunov_batch_scale_collusion.py 4`) at batch 2.0:
`seed=1 ceiling (gap 0.320)`, `seed=2 ceiling (gap 0.306)`, `seed=3 ceiling (gap 0.241)`,
`seed=0 unresolved` (corroboration disagreed → HELD). **The positive controls (batch 1.0)
return `threshold` or `unresolved` — NEVER `ceiling`** across all seeds, proving the ceiling
call is BATCH-INDUCED, not a pre-existing bug. The effect is monotone in the batch factor:
batch 1.0 → threshold/unresolved, batch 1.5 → threshold/unresolved, batch 2.0 → ceiling — a
clean dose-dependent confound exactly as the scale⇄v_max mechanism predicts.

### Which gate fails and why

`fit_lyapunov_multi` PINS each operating point's depth `scale` from that point's **clean WT**
(`calibrate_from_wt`) and holds it fixed for the perturbed condition — it **ASSUMES WT and
perturbed share one capture/depth scale** (the `calibrate_from_wt` docstring even notes it
pins from WT *precisely so a ceiling change is not absorbed into depth*). A perturbed-only
batch factor `c` violates that assumption: the pinned scale is now wrong by `c`, and the
free-`v_max` restricted fit absorbs `c` into the ceiling (both multiply the ON mode), so
`ceiling` wins the joint NLL.

**Why the two LIM-017 mechanisms do NOT catch it (the collusion):**

1. **Graded near-fold down-weighting** keys on each point's bifurcation *proximity*. The
   batch is applied to the DATA, not the circuit, so every point stays well-buffered
   (`gate_all_ok=True`, all proximities low). Nothing is down-weighted.
2. **Best-buffered-pair corroboration** accepts a bare mechanism only if the two
   MOST-BUFFERED points agree. But the batch corrupts **every** operating point *identically*
   (a uniform scale), so the most-buffered pair also reads `ceiling` → it **agrees** → the
   confident-wrong call is *certified*. The corroboration is designed to catch a *marginal*
   (near-fold) point that changed the answer; here no point is marginal — they are all
   corrupted equally, which is exactly the case the corroboration cannot see.

**There is NO OFF-cluster / depth-ratio gate in the lyapunov multi path** — no analog of the
differential module's gate 2 (per-context depth ratio), gate 4b (`off_shift`) or gate 4c
(`off_scale`, the P4/P5 fix). The differential module guards this *identical* confound (a
perturbed-only multiplicative scale, LIM-016 P4/P5); the lyapunov breaker does not, and it has
no documented bound for it.

### Not a degeneracy, not a documented bound (honesty)

- **Not observationally identical to a genuine effect.** A genuine `v_max` (ceiling) change
  scales ONLY the ON mode and leaves the OFF mode at basal; the batch scales the ENTIRE
  distribution including the OFF mode. So the OFF-cluster level/spread fingerprint DISTINGUISHES
  batch from ceiling (this is the very `off_scale` statistic the differential module keys on) —
  the distinguishing information EXISTS in the data; the lyapunov path simply never inspects it.
  Additionally the truth is a *threshold* shift (mode positions reshaped across basals), so the
  data is not a pure scale either. NUDGE asserting `ceiling` is a genuine mis-attribution, not a
  degenerate tie.
- **Not a documented bound.** LIM-017 documents ONLY the near-fold-point corruption of the
  joint fit; its best-buffered-pair corroboration is presented as the fail-safe that closes it.
  This hole is on WELL-BUFFERED points and DEFEATS that corroboration by a uniform perturbed-only
  batch — a surface the prior final sweep explicitly recorded as plausible-but-UNRUN
  (`runs/000000017`, LIM-017 corroboration collusion = "NOT REACHED"). LIM-016 documents the
  perturbed-only-scale confound only for the **differential** module (with gates 2/4b/4c); the
  lyapunov module carries no such gate and no such limitation. This is NEW.

### Candidate decoy + limitation (DESCRIBED only — NOT registered, NOT fixed)

- **Candidate limitation (a new `NUDGE-LIM` or a sharpening of LIM-017):** the multi-point
  covariance breaker assumes WT and the perturbed condition share one capture/depth scale
  (`calibrate_from_wt` pins from WT). A perturbed-only batch/depth scale violates this and the
  free-`v_max` fit absorbs it into a confident `ceiling`; the corroboration is blind because the
  corruption is uniform across operating points. Fail-safe status: currently a **confident-wrong
  hole**, safety-relevant (the breaker looks *more* authoritative than a snapshot).
- **Candidate fix direction (for the fixer to MEASURE + decide):** port the differential
  module's OFF-cluster check into `attribute_lyapunov_multi`. Since `calibrate_from_wt` already
  pins the WT depth, compare each perturbed point's OFF-cluster scale (MAD of below-median
  activity) against its WT control's; a uniform batch inflates it by `c` (≈2.0 here) while a
  genuine ceiling leaves it ≈1 — the exact P4 `off_scale` separator. Abstain when it departs a
  measured band. Expect this to CLOSE the inflating batch and possibly leave a deflating-batch
  BOUND (degenerate with a genuine ceiling reduction), mirroring the honest P4/P5 outcome.
- **Candidate decoy:** lock `lyapunov_perturbed_batch_ceiling_hole.py` (batch 2.0, seeds 1/2/3)
  as a `@pytest.mark.decoy` fail-safe assertion — *never a confident `ceiling` under a
  perturbed-only batch on a genuine threshold* — alongside a positive control (batch 1.0 still
  resolves/abstains, no over-abstention).

---

## Score table (final sweep, round 6)

| Probe | Verdict |
|-------|---------|
| **P6: LIM-017 corroboration collusion — perturbed-only batch ×2.0 on well-buffered points → confident `ceiling`** | **HOLE — verified 3/4 (sweep) + 2/2 (focused), deterministic** |
| P2 multi_reporter batch confound re-confirm | HELD (0 confident-wrong; abstains `unresolved`) |
| P3 design safety gate (absolute near-fold + margin-gaming) re-confirm | HELD (fires `high_risk`; agrees with `classify_robustness` 0.589 ≥ 0.55) |
| P5 differential small-mult confound re-confirm | HELD (0 confident-wrong across 2 seeds; all `unresolved`) |
| P4 differential large-mult gate-4c machinery (exercised by the P5 repro's `off_scale`/resolvability path) | HELD (same gate code; `off_scale` computed + acted on) |
| P1 differential additive gate-4b (`_OFF_SHIFT_INFLATION_MAX=2.5` present; `off_shift` computed in the P5 run) | HELD-by-code + regression tests (not heavy re-run — budget) |
| Cross-capability composition into `design()` (a confident-wrong upstream label) | HELD-by-construction (safety gate recomputes proximity from the intervened Circuit, does not trust the upstream label; P3 absolute check present) — analyzed, not heavy-run |

## Coverage vs skipped (budget-honest — ~16 GB / 4 cores, heavy jobs run STRICTLY sequentially)

- **RAN (heavy, sequential):** the LIM-017 lyapunov corroboration-collusion probe at 2 seeds
  (found 1/2), then 4 seeds (confirmed 3/4), then the focused authored repro at 2 seeds (2/2,
  deterministic) — priority 1, the whole point of this sweep; the P5 differential small-mult
  re-confirm at 2 seeds.
- **RAN (light):** P2 multi_reporter batch confound; P3 design safety gate.
- **Analyzed, not heavy-run (budget + already-strong prior coverage):** composition into
  `design()` (HELD-by-construction — the safety gate is independent of upstream labels and P3 is
  re-confirmed); P1/P4 differential gates (the gate CODE is present and unchanged post-merge, the
  P5 repro exercises the shared gate-4c/`off_scale` path, and both are regression-locked). These
  are stated as HELD-by-code, NOT as fresh heavy reproductions.
- **Not attempted this round:** partial-panel P2 gaming; a fresh dose-response / cross-modality /
  epistasis sweep (HELD in prior rounds, no new attack vector identified within budget).

## Joint-hold confirmation of the 5 shipped fixes

**All five HOLD jointly, no cross-fix regression.** P2 (0 confident-wrong) and P3 (high_risk
fires, agrees with `classify_robustness`) directly re-run HELD; P5 (0 confident-wrong, all
`unresolved`) directly re-run HELD; P4's gate-4c machinery is exercised by the P5 repro (HELD);
P1's gate-4b constant is present and regression-locked (HELD-by-code). The NEW hole P6 is in a
DIFFERENT capability (`lyapunov`), not a regression of any of the five.

## LIM-017 lyapunov collusion probe result (the mandated statement)

**RAN → HOLE.** The well-buffered-points corroboration collusion (perturbed-only batch ×2.0)
returns a confident-wrong `ceiling` at 3/4 seeds (sweep) and 2/2 seeds (focused, deterministic).
This is the priority-1 unreached probe, now reached and reproduced.
