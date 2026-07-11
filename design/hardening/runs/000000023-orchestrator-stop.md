# 000000023 · orchestrator · LOOP PAUSED by user decision (P6 left OPEN, documented)

*Immutable run record (write-once). This is currently the highest-numbered record, so its
`NEXT →` block below is the AUTHORITATIVE resume pointer (`../LEDGER.md` mirrors it).*

## ▶ NEXT →

```
NEXT  →  nudge-uq-fixer  on  P6  (lyapunov corroboration-collusion — WHEN THE LOOP RESUMES)
STATUS:  PAUSED BY USER — do NOT dispatch any agent until the user says go.
```

## Why the loop paused HERE (not at HOLES_FOUND: 0) — honest record

The 2nd final full sweep (`runs/000000022`) found a genuine NEW confident-wrong, **P6**
(`attribute_lyapunov_multi`: a perturbed-only batch ×2.0 defeats the LIM-017 best-buffered-pair
corroboration → confident bare `ceiling` where truth is threshold). Independently re-confirmed by
the orchestrator: `lyapunov_perturbed_batch_ceiling_hole.py` → **3/3 seeds** deterministic
(batch=2.0 → `ceiling`, gap 0.24–0.32 ≫ resolve_margin; controls batch=1.0 → `unresolved`/
`threshold`, never `ceiling`).

Because P6 is a real hole, the loop's own STOP condition (`HOLES_FOUND: 0`) was **NOT** met. The
orchestrator surfaced a scope decision to the user (6 holes found, all the same systemic class — a
perturbed-condition batch/scale confound aliasing to `ceiling`, invisible to a control-keyed guard;
un-probed capabilities likely carry the same vector; each round ~2.5 h compute). **The user chose
to STOP here**, judging the underlying *class* problem sufficiently demonstrated/solved by the
merged P1–P5 fixes, and accepting P6 as a documented open hole for now ("it's okay").

## HONEST STATE at pause (no overclaim)

- **Closed/bounded + merged (5):** P3 (design safety gate), P1 (differential additive), P4
  (differential large-mult), P5 (differential small-mult), P2 (multi_reporter batch) — each
  independently audited (PASS) + orchestrator-re-verified through the full gate; frozen core never
  touched. See the "Closed problems" table.
- **OPEN (1) — P6:** a LIVE, UNGUARDED confident-wrong in the shipped `attribute_lyapunov_multi`
  (not a documented bound). It remains in the problem queue with its repro
  (`scripts/redteam/lyapunov_perturbed_batch_ceiling_hole.py`), the report
  (`design/FAILSAFE_REDTEAM_6.md`), and the fix direction (port the differential OFF-cluster /
  perturbed-vs-control scale check into `attribute_lyapunov_multi`; expect CLOSED-inflating /
  BOUNDED-deflating, mirroring P4/P5). `lyapunov.py` is NOT frozen core.
- **Un-probed for this vector (recorded so nothing reads as "fully swept"):** fresh
  dose-response / cross-modality / epistasis / constitutive sweeps for the same
  perturbed-condition confound; partial-panel P2 gaming. These are open probe surface, NOT
  cleared.

## To RESUME (a future session)

Read `../LEDGER.md`; the queue holds **P6** (and the un-probed surface above). Resume the fix
loop `[ nudge-uq-fixer → nudge-audit → nudge-red-team ]` on P6, then continue sweeping until a
genuine full sweep returns `HOLES_FOUND: 0`.

## This session's loop runs (immutable): `runs/000000001`–`000000023`.
