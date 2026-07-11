# 000000024 · orchestrator · P6 merged (audit PASSED) + independent re-measurement/correction

*Immutable run record (write-once). This is currently the highest-numbered record, so its
`NEXT →` block below is the AUTHORITATIVE resume pointer (`../LEDGER.md` mirrors it).*

## ▶ NEXT →

```
NEXT  →  nudge-red-team  (re-scan — the loop's STOP gate)
         MUST cover: (a) P5+P6 fix-induced-regression check;
                     (b) the two surfaces the moat sweep left UNREACHED (runs/000000018):
                         adjoint.ode_identifiability reached through a real large ODE (same P6 root
                         cause), and OED multi-knob design-space / end-to-end composition.
STOP when nudge-red-team reports HOLES_FOUND: 0 after a genuine FULL sweep.
```

## What the orchestrator did

`nudge-audit` returned **PASS** on P6 (`runs/000000023`). Before merging, the orchestrator
**independently re-verified** and acted on the audit's one honesty flag:

- **Independent re-measurement (not trusting the fixer OR the audit):** ran the fixed diagnostic
  (`method="iterative"`, seeds 0–2 × n∈{40,300}) — every case `unidentifiable (n_null=1)`, certified
  smallest eigenvalue **~1.1e-12 (n=40) … ~2.3e-10 (n=300)**, `λ_max ~2.7e4–3.6e4`. This confirms the
  hole is closed AND that the recorded "~1e-19" probe RQ was a non-reproducing (sub-cancellation-
  floor) figure.
- **Merge:** `git merge --no-edit claude/p6-uq-fix` (`2d54b57`) → merge commit **`3cd1f41`** (clean,
  `ort`; the 4 shared docs — CHANGELOG/STATE/known_limitations/FINDINGS — auto-merged additively with
  the P5 content, no conflicts).
- **Honesty correction (additive, commit `ed3c381`):** corrected "~1e-19" → the measured
  ~1e-12…1e-10 (≪ the rank floor `rank_rtol²·λ_max ≈ 3e-10`) in `FINDINGS §P6` (table + a
  re-measurement note), `NUDGE-LIM-023`, and `CHANGELOG` — the load-bearing separation is the
  ratio-to-floor, which reproduces. `check_anomalies` still OK (schema-valid).
- **Frozen core:** `git diff --name-only origin/main..HEAD -- src/nudge/inference/fit.py
  src/nudge/core/` = empty. ✔

## Outcome

**P6 CLOSED (isolated structural-null mislabel) / BOUNDED (huge-regime over-abstention, fail-safe,
locked by strict-xfail decoy).** The matrix-free identifiability path can no longer certify a
structurally-unidentifiable model `well-constrained`: `auto` defers to the exact reconstruction to a
measured `dense_below=2048`, and the huge regime abstains (or the inverse-iteration probe catches the
null) — never a confident-wrong. Positive controls resolve (no over-abstention); honesty record
re-measured + corrected. Moved to LEDGER "Closed problems".

- Merge commit: `3cd1f41` · Correction: `ed3c381` · Fix commit: `2d54b57` · Audit: `runs/000000023`
  (PASS). Frozen core untouched; `main` untouched.
