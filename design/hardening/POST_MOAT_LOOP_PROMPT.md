# Cloud hardening-loop brief — bulletproof the differentiability moat (post-merge)

**Paste this whole file as the task for a cloud web agent working in the NUDGE repo.**

---

## Who you are and the one rule

You are orchestrating NUDGE's **4-way fail-safe hardening loop** on freshly-merged code.
NUDGE is a mechanism-attribution tool for perturbation screens whose entire thesis is:
**never claim more than you have measured; abstain over being confidently wrong; a
confident-wrong output is the ONLY hard failure.** A calibrated abstention or a
documented, bounded residual is a *pass*, not a gap. A polished-but-false claim scores
worse than an honest "cannot resolve this." Read `CLAUDE.md` and `design/STATE.md` first.

## Hard constraints (non-negotiable)

- **Never edit `src/nudge/inference/fit.py` or anything under `src/nudge/core/`.** They are
  the frozen stability contract. Every fix is **additive** (a new guard, a new module, a
  wrapper) — same discipline that shipped every capability to date.
- **Fixes must be MEASURED, not arbitrary.** A guard threshold has to be grounded in a
  measured degeneracy/curvature/separator (e.g. "confident-wrong cases show `X ≥ 2.99` vs
  genuine `≤ 1.96`, so gate at 2.5"), never a hand-picked constant. If you can't measure a
  clean separator, the honest fix is to **abstain and document the bound** — that is a valid
  outcome.
- **Do all of this on a SEPARATE BRANCH — do NOT touch `main`.** See "Branching" below.

## What to harden (priority order)

The recently-merged **differentiability moat** is the least-hardened surface — prioritize it:

1. **`src/nudge/inference/oed.py`** (`NUDGE-METHOD-014`, `NUDGE-LIM-024`) — gradient-based
   optimal experimental design. Red-team questions: can it ever report a *false-precise* CRLB
   / identifiability gain (understate the bound) on a design that is actually still
   degenerate? Does the guarded ridge ever mask a genuine singularity and let it claim a
   parameter is resolved when it is not? Does `optimize_design` ever return a "converged"
   optimum that is worse than the naive design (a silent regression)? Is the local-OED caveat
   (`NUDGE-LIM-024`) actually enforced, or can a caller read the gain as unconditional?
2. **The matrix-free additions in `src/nudge/inference/sloppiness.py`**
   (`sloppiness_diagnostic_matrixfree`, `analyze_model_matrixfree`, `fim_matvec`) and
   **`ode_identifiability` in `src/nudge/inference/adjoint.py`** (`NUDGE-LIM-023`). Red-team
   questions: can the iterative Krylov path ever label a rank-deficient / unidentifiable model
   `well-constrained` or `sloppy-but-predictive` (the one dangerous mislabel)? Does the
   shape-rank-deficiency certificate hold on every path, or is there an `n_params ≤ n_obs`
   case where an unconverged smallest-eigenvalue estimate leaks a confident identifiable
   verdict? Does `method="auto"` ever pick the iterative path where it should abstain?
3. **The wiring** that exposes the above — `service.oed_demo` / `service.*`, the `nudge oed`
   CLI, and any MCP tool — for a confident-wrong that only appears end-to-end.
4. **Then a general full sweep** of the existing capabilities to catch any merge-induced
   regression (the two merges touched shared docs + `sloppiness.py`/`adjoint.py`).

Also **fold in the pre-existing OPEN hole already in the queue**: **P5** in
`design/hardening/LEDGER.md` (differential small-multiplicative confound, `c≈1.15–1.25`,
`LIM-016`) — repro at `scripts/redteam/differential_small_mult_gain_hole.py`. It must be
fixed or honestly bounded before release, and its now-falsified P4 "INFLATION CLOSED" wording
corrected.

## The loop (roles + sequence)

Follow the protocol in **`design/hardening/README.md`** exactly. Four roles — if your
environment has the dedicated subagent types use them, otherwise perform each role yourself,
keeping them **independent** (the auditor must not trust the fixer; re-run everything):

| # | role | does |
|---|------|------|
| 1 | **red-team** | Hunts confident-wrong holes; **reports + reproduces**, never fixes. A deterministic repro script under `scripts/redteam/` is required per hole. Its verdict `HOLES_FOUND: n` drives the loop. |
| 3 | **uq-fixer** | Takes ONE hole → independently re-reproduces it → **measured, targeted, additive** fix → a regression test + a decoy + the honesty record (a new/*sharpened* `NUDGE-LIM-*` in `docs/known_limitations.yaml` + a `scripts/vv/FINDINGS.md` entry). |
| 4 | **audit** | **Independently re-verifies**: hole now abstains? positive control still resolves (no over-abstention)? honesty record accurate? frozen core untouched? full gate green? Returns `AUDIT: PASS/FAIL`. |

**Sequence (start with red-team, as requested):**
`1 (full sweep, moat-first)` → for each hole `[ 3 → 4 ]` → after each fix **return to `1`** to
re-scan for fix-induced regressions → **STOP when `1` reports `HOLES_FOUND: 0` after a genuine
full sweep.** A `4 = FAIL` sends the hole back to `3` (do not advance).

## The append-only audit trail (do not break it)

Everything the loop does is recorded immutably under `design/hardening/`:
- Append one immutable record per agent invocation to
  `design/hardening/runs/NNNNNNNNN-<role>-<target>.md` (9-digit zero-padded, monotonic; the
  current max is `000000017`, so continue at `000000018`). **Never edit or delete a past
  record.** Each record MUST begin with a `NEXT →` block naming the next agent + target.
- Mirror the resume pointer + problem-queue transitions in `design/hardening/LEDGER.md`
  (statuses move in place `OPEN → IN-PROGRESS → FIXED → CLOSED`; rows are never deleted).
- `scripts/check_hardening_append_only.py` **enforces** this on every change — run it as part
  of the gate; a deletion/mutation of a `runs/` record fails CI.

## Branching (keep `main` clean for the release)

- Branch off the current `main`: `git checkout -b hardening/post-moat-release`.
- Each role may work in its own worktree/branch, but **integrate everything into
  `hardening/post-moat-release`** — the orchestrator merges role branches into *that* branch,
  **never into `main`.** `main` must be untouched when you finish.
- When red-team reaches `HOLES_FOUND: 0`, push `hardening/post-moat-release` and open a PR
  to `main` titled `hardening: post-moat red-team loop → release candidate`. A human merges
  it; that merge is what unblocks the `nudge-bio 0.1.0` PyPI release
  (`design/RELEASE_CHECKLIST.md`).

## The full gate (run before every merge, and at the end)

```bash
uv sync --extra dev
uv run ruff check src tests scripts
uv run pyright src
uv run python scripts/check_mechanism_cards.py
uv run python scripts/check_impl_mapping.py
uv run python scripts/check_citations.py
uv run python scripts/check_anomalies.py
uv run python scripts/check_hardening_append_only.py
uv run pytest -q            # fast lane (should be ~296+ passed, 0 failed)
uv run pytest -q -m slow    # the slow identifiability/scaling suite
```

Additive/opt-in only; never touch the energy-distance `fit()` default or the decoy battery.
Credit Claude in commits with a `Co-Authored-By: Claude <model> <noreply@anthropic.com>`
trailer, naming only a model you can verify actually ran.

## Deliverable

A pushed `hardening/post-moat-release` branch + PR where: red-team's final full sweep reports
`HOLES_FOUND: 0`; every hole found is either fixed (measured guard + test + decoy) or honestly
bounded (documented `NUDGE-LIM` + strict-xfail decoy); P5 is closed/bounded; the full gate is
green; `main` is untouched; and the `design/hardening/` trail tells the whole story. End with
a short summary: holes found, how each was resolved, and the residual bounds a release note
should mention.
