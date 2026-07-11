# 000000009 · red-team · P1-fix re-scan → HOLES_FOUND: 1 (NEW hole P4)

*Immutable run record (write-once). This is currently the highest-numbered record, so its
`NEXT →` block below is the AUTHORITATIVE resume pointer (`../LEDGER.md` mirrors it).*

## ▶ NEXT →

```
NEXT  →  nudge-uq-fixer  on  P4  (differential MULTIPLICATIVE perturbed-only scale → ceiling-diff)
then  →  nudge-audit → orchestrator merge → nudge-red-team (re-scan) ; then P2
STOP when nudge-red-team reports HOLES_FOUND: 0 after a genuine FULL sweep (P4 + P2 fixed).
```

## Role / target

- **Role:** `nudge-red-team` (role 1), worktree-isolated. A **regression re-scan** after the
  P1 fix merged (`99d73b8`), scoped to hunt a P1-fix-induced / adjacent confident-wrong.
- **Target:** the merged P1 gate 4b in `src/nudge/inference/differential.py`.

## Verdict: **HOLES_FOUND: 1** (NEW hole P4; the P1 additive fix itself HELD)

| Probe | verdict |
|-------|---------|
| 1a — ADDITIVE perturbed-only offset, sub-2.5 band (N=6000, obs_sd=0.3) | **HELD** (tighter modes raise off_shift; no additive sub-2.5 gap) |
| 1b — **MULTIPLICATIVE** perturbed-only scale c∈{1.5,2.0} at both seeds | **HOLE — verified (NEW = P4)** |
| 2 — deflating MULTIPLICATIVE c∈{0.5,0.7} | confident ceiling-diff — but on the deflating side already declared unguarded → **sharpens the documented P1 bound, NOT new** |
| 2 — deflating ADDITIVE (subtract) | HELD (`no-difference`, matches FINDINGS §P1) |
| 4 — seed robustness of the 2.5 cut (c=2.4) | guard **FLAKY at the boundary** (off_shift straddles 2.5 across seeds; robustly-under at c=1.5/2.0) |

## The NEW hole — P4 (verified, independently re-confirmed by the orchestrator)

A constant MULTIPLICATIVE factor `c` on ONE context's PERTURBED cells only (control clean),
`mechanism="none"` (truth = no-difference) → a confident **`ceiling-diff`**. It evades BOTH
guards: gate 2 keys `depth_ratio` on the clean controls (≈1.009), and gate 4b's `off_shift`
measures the near-zero OFF baseline — a multiplicative factor scales near-zero to near-zero,
so `off_shift ≈ 1` (< 2.5), gate 4b silent. Meanwhile `vmax` aliases a global scale on the
ON mode 1:1 (depth pinned from the clean control) → a massive-margin ceiling-diff.

**Orchestrator independent spot-check** (`differential_multiplicative_confound.py 1`, run
alone, rc=2): seed 0, factor 1.5 → `ceiling-diff` vmax 2.01→3.24, dBIC-vs-shared **319.0**,
beats runner-up threshold-diff by 132.1, `off_infl=0.994 < 2.5`, `depth_ratio=1.009`. Factors
2.0 (dBIC 1019) and 2.4 (dBIC 5893) likewise `ceiling-diff`. The red-team verified 5
confident-wrong across 2 seeds.

**Why it is NEW, not a P1 regression/overclaim:** the P1 fix scoped itself honestly to the
ADDITIVE channel (CLOSED-inflating / BOUNDED-deflating); it never claimed to close a
multiplicative confound. P4 is a pre-existing latent hole in `differential` that this re-scan
surfaced. It is the differential analogue of the still-OPEN **P2** (multi_reporter per-
condition batch scale → ceiling) — the SAME class: a per-condition multiplicative scale that
aliases the ceiling channel, invisible to a control-keyed guard.

**Honesty follow-through required (recorded here, actioned in the P4 fix):** P1's living-doc
bound was characterized only against the additive channel, so it is honest-but-incomplete —
the P4 fix must sharpen `NUDGE-LIM-016` to name the multiplicative channel (inflating AND
deflating) explicitly, so nothing implies `differential`'s perturbed-side confounds are all
handled.

## Mitigation direction (red-team's pointer — the P4 fixer MEASURES + decides)

The `off_shift` diagnostic must probe the **ON mode / mode-amplitude ratio**, not only the OFF
baseline (a multiplicative scale is visible in ON-mode inflation; an additive offset in the
OFF-mode translation) — OR require an independent per-context depth anchor before any
`ceiling-diff`. The ceiling channel is fundamentally degenerate with a perturbed-condition
global scale, so an honest outcome may be a documented BOUND (abstain-on-ceiling-diff-without-
an-anchor), like the P1 deflating bound / constitutive LIM-019.

## Independent verification (orchestrator)

Merged the additive artifacts (3 `scripts/redteam/*.py`) — no `src/` touched. Re-ran the
multiplicative repro (rc=2, hole confirmed). P1's additive fix HELD (probes 1a/2-additive).

## P2 status

Untouched this round (as instructed) — remains **OPEN**.

## Branch + commit (merged additively by the orchestrator)

- Branch: `worktree-agent-a16b809b9417c133b` · Commit: `63516d4`
- Repros: `differential_multiplicative_confound.py` (P4), `differential_multiplicative_deflating.py`
  (sharpens the P1 bound), `differential_sub25_band_probe.py` (additive sub-2.5 HELD).
