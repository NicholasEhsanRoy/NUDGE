# 000000022 · red-team · FINAL FULL sweep (round 2) → HOLES_FOUND: 1 (NEW hole P6) — NOT a STOP

*Immutable run record (write-once). This is currently the highest-numbered record, so its
`NEXT →` block below is the AUTHORITATIVE resume pointer (`../LEDGER.md` mirrors it).*

## ▶ NEXT →

```
NEXT  →  nudge-uq-fixer  on  P6  (lyapunov corroboration-collusion → confident ceiling)
then  →  nudge-audit → orchestrator merge → nudge-red-team (sweep)
STOP when nudge-red-team reports HOLES_FOUND: 0 after a genuine FULL sweep.
```

## Role / target

- **Role:** `nudge-red-team` (role 1), worktree-isolated. The 2nd final full sweep, run with
  all FIVE prior holes (P3/P1/P4/P2/P5) fixed + merged, directed at the previously-UNREACHED
  probes (the LIM-017 lyapunov collusion, cross-capability composition into `design()`).
- **Result:** **HOLES_FOUND: 1** → NOT a STOP. The five merged fixes HELD jointly (no cross-fix
  regression); the sweep RAN the mandated unreached probe and it produced a NEW hole (**P6**).

## The NEW hole — P6 (verified 3/4 sweep + 2/2 focused-deterministic; orchestrator re-confirming)

**Capability:** `nudge.inference.lyapunov.attribute_lyapunov_multi` (the multi-operating-point
covariance breaker; `NUDGE-LIM-017` best-buffered-pair corroboration). **The LIM-017
corroboration-collusion the prior sweep (`runs/000000017`) flagged plausible-but-UNRUN — now
RAN → HOLE.**

A **perturbed-only batch/depth scale ×2.0** (WT clean) applied on top of a genuine **threshold**
difference (K×1.6, no v_max change) drives `attribute_lyapunov_multi` to a confident bare
**`ceiling`**. Truth = threshold. Repro: `scripts/redteam/lyapunov_perturbed_batch_ceiling_hole.py`
(focused, deterministic) + the sweep `scripts/redteam/lyapunov_batch_scale_collusion.py`.

```
batch=2.0  label='ceiling'  gate_all_ok=True   gap 0.24–0.32 ≫ resolve_margin 0.03
  focused NLLs: seed1 gain=8.840 threshold=8.607 ceiling=8.287 ; seed2 gain=8.898 thr=8.602 ceil=8.296
batch=1.0 (control) → 'threshold'/'unresolved' (never 'ceiling') — proving the call is batch-induced
```

**Why it evades:** `calibrate_from_wt` pins depth from the CLEAN WT and assumes WT/perturbed
share it; the free-`v_max` fit absorbs the perturbed-only batch into the ceiling. The LIM-017
best-buffered-pair CORROBORATION is DEFEATED because a **uniform** batch corrupts EVERY operating
point identically — so the two most-buffered points agree on the (wrong) ceiling. The lyapunov
path has **no OFF-cluster/`off_scale` analog** of the differential gate 4c and no documented bound
for this vector. **Not a degeneracy:** the batch scales the OFF mode too, distinguishable in
principle from a genuine ceiling (the info exists; the gate does not) — the same systemic class as
P1/P2/P4/P5 (a perturbed-condition confound invisible to a control-keyed guard, aliasing to
ceiling), now shown to hit the core multi-point breaker.

## Score table (this sweep)

| Probe | Verdict |
|-------|---------|
| **P6: LIM-017 corroboration collusion (perturbed-only batch ×2.0 → confident `ceiling`)** | **HOLE — 3/4 sweep + 2/2 focused, deterministic** |
| P2 multi_reporter batch confound | HELD (0 confident-wrong) |
| P3 design safety gate (absolute near-fold + margin-gaming) | HELD |
| P5 differential small-mult confound | HELD (0 confident-wrong) |
| P4 differential large-mult gate-4c | HELD-by-code (exercised via P5's off_scale path) |
| P1 differential additive gate-4b | HELD-by-code + regression tests |
| Composition into `design()` (confident-wrong upstream label) | HELD-by-construction (safety gate recomputes proximity from the intervened Circuit; P3 absolute check present) |

## Coverage (budget-honest; heavy jobs strictly sequential)

- RAN (heavy): the LIM-017 lyapunov collusion probe 2→4 seeds + a focused authored repro (2
  seeds); the P5 differential re-confirm.
- RAN (light): P2 multi_reporter, P3 design safety gate.
- Analyzed, not heavy-run: composition into `design()` (HELD-by-construction); P1/P4 gates
  (code present + regression-locked; the P5 repro exercises the shared gate-4c path).
- **Skipped (explicitly, budget):** partial-panel P2 gaming; fresh
  dose-response/cross-modality/epistasis sweeps (HELD prior rounds, no new vector within budget).
  **These remain open probe surface for a future sweep.**

## Fix direction (red-team's pointer — the P6 fixer MEASURES + decides)

Port the differential module's OFF-cluster / perturbed-vs-control scale check into
`attribute_lyapunov_multi`: compare each perturbed operating point's OFF-cluster scale against its
pinned WT control; abstain the ceiling call when the OFF mode is rescaled (the batch fingerprint).
Expect **CLOSED-inflating / BOUNDED-deflating**, mirroring P4/P5. `lyapunov.py` is NOT frozen core
(only `fit.py`/`core/` are) and `attribute_lyapunov_multi` was already hardened once (round-2
LIM-017) — but it is a SHARED foundational module (differential/multi_reporter import its helpers),
so the fix must be additive + localized and must not regress the P1–P5 paths.

## Independent verification (orchestrator)

Merged the additive artifacts (`design/FAILSAFE_REDTEAM_6.md` + the repro) — no `src/` touched.
Re-running the focused repro to confirm before queueing.

## Branch + commit (merged additively by the orchestrator)

- Branch: `worktree-agent-a5a84dba8a46feaae` · Commit: `fe0cb9244e6803938794d70dee19a484f53579e2`
