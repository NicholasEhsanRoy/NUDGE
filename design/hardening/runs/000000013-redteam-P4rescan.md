# 000000013 · red-team · P4-fix re-scan → HOLES_FOUND: 0 (+ orchestrator precision tightening)

*Immutable run record (write-once). This is currently the highest-numbered record, so its
`NEXT →` block below is the AUTHORITATIVE resume pointer (`../LEDGER.md` mirrors it).*

## ▶ NEXT →

```
NEXT  →  nudge-uq-fixer  on  P2  (multi_reporter per-condition batch-scale confound, LIM-014)
then  →  nudge-audit → orchestrator merge → nudge-red-team (re-scan)
STOP when nudge-red-team reports HOLES_FOUND: 0 after a genuine FULL sweep (P2 fixed).
```

## Role / target

- **Role:** `nudge-red-team` (role 1), worktree-isolated. A **regression re-scan** after the
  P4 fix merged (`ebda9c6`), scoped to gate 4c regressions + a remaining `differential`
  confident-wrong (dual-gate evasion, band-edge gaming, over-abstention).
- **Target:** the merged P4 gate 4c (+ P1 gate 4b) in `src/nudge/inference/differential.py`.

## Verdict: **HOLES_FOUND: 0** (both gates hold; no new confident-wrong in `differential`)

| Attack (truth = no biological difference) | off_shift | off_scale | Result | caught by |
|---|---|---|---|---|
| uniform multiplicative c∈{1.5,2,2.4,0.7,0.5} (P4 regression) | 0.96–2.58 | out-of-band | **HELD** 0/12 | gate 4c (+4b at 2.4) |
| doublet-rate difference (realistic) | 0.99–1.48 | 1.11–1.23 (in band) | **HELD** 0/4 | gate 4 (Δ-model tie) |
| smooth content-dependent capture bias (realistic) | 0.99–1.88 | 1.45–1.81 | **HELD** 0/4 | gate 4c |
| knife-edge scale on strictly-above-median cells | 0.99–1.07 | 0.95–1.07 (in band) | in-band positive → **NOT a valid confident-wrong** (see below) | evades 4c |

- **Probe 1 (dual-gate evasion):** a multiplicative scale confined **strictly** to the
  above-median (ON-mode) cells evades gate 4c (off_scale measures the untouched OFF cluster
  → ≈1.0) and 4b (off_shift ≈1.0), yielding a confident `ceiling-diff`. **Correctly judged
  NOT a hole (honesty):** the evading data is observationally *identical* to a genuine ceiling
  change (raises the ON mode, anchored OFF spread = the exact real-ceiling fingerprint) —
  indistinguishable without an external depth anchor, the same degeneracy P4 abstains on for
  a deflating scale — and the median-step construction has no plausible physical generator.
  Every **realistic** sibling is caught (smooth content bias → gate 4c; doublet rate →
  gate-4 tie). A multiplicative scale is intrinsically a ceiling operation, so it does not
  produce a gain/threshold-channel evader either.
- **Probe 2 (band-edge robustness):** [0.80, 1.30] robust across seeds/N/floors; no confound
  straddles just inside 1.30 while faking a confident ceiling-diff.
- **Probe 3 (over-abstention):** none new — the gate-4c abstentions triggered were on
  confounds (off_scale 1.45–1.81), not on genuine effects.

## Orchestrator honesty follow-through — precision tightening APPLIED

The red-team flagged (documentation precision, NOT a hole) that P4's "INFLATION is CLOSED" is,
exactly, "closed against a **uniform or smoothly content-dependent** inflating scale"; the
above-median-only construction evades the OFF-cluster fingerprint but is observationally
degenerate with a genuine ceiling change. To keep the claim exactly matched to the measurement
(the #1 honesty rule), the orchestrator tightened the wording (a doc-only change, no logic /
no gate touched):
- `src/nudge/inference/differential.py` — the module docstring + the `_OFF_SCALE_INFLATION_MAX`
  rationale comment now qualify "INFLATION CLOSED" and note the degenerate above-median evader.
- `scripts/vv/FINDINGS.md` §P4 — a "Precision" paragraph with the three re-scan repros.
Verified after the edit: ruff clean, pyright 0, import OK (gate-4c band unchanged 0.80/1.30),
all 5 doc checkers OK. No behavior change ⇒ the audited gate results (repro HOLES:0, slow
suite 22+1) still hold; the fast lane was re-confirmed green earlier at `ebda9c6` (261).

## Independent verification (orchestrator)

Merged the 3 additive repro scripts (no `src/` touched). The red-team's non-hole verdict on
the above-median evader is sound (indistinguishable-from-genuine, no physical generator).

## P2 status

Untouched (as instructed) — remains **OPEN**. Next fix target.

## Branch + commit (merged additively by the orchestrator)

- Branch: `worktree-agent-ace2d6a628fe93c4e` · Commit: `47a840047dde4acbebec38c5f9fc8c4008da310c`
- Repros: `differential_subset_scale_confound.py` (degenerate evader, HELD),
  `differential_content_capture_confound.py`, `differential_doublet_rate_confound.py`.
