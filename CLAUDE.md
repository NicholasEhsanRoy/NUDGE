# NUDGE — project guide for Claude

**NUDGE** (Node/edge Ultrasensitivity Diagnostic for Gene-regulatory Effects) is
a mechanism-attribution tool for perturbation (Perturb-seq) screens. It fits a
compositional, differentiable circuit model to single-cell data and distinguishes
perturbations that move a switch's **threshold** from ones that change its
**gain** — a distinction linear screen analysis can't make — then inverts the fit
to propose untested interventions. Built on **MADDENING** (a differentiable
graph-physics engine) for the [Built with Claude: Life Sciences] hackathon.

## ⭐ Start here each session: `design/STATE.md`

**`design/STATE.md` is the single source of truth for where the project is and what's
next** — operational essentials (uv toolchain, checks), current state (Phases 0–2 done,
proof of concept closed + calibrated), the full roadmap, architecture decisions &
gotchas, empirical findings, and the ready-to-execute **Tier-0.5 stochastic simulator**
plan. Read it before substantive work.

Design docs (deeper reasoning):
- `design/GENERATOR_DESIGN.md` — literature-grounded generator design (count model +
  the resolved bistability crux; the two `/deep-research` syntheses)
- `design/WORKING_BACKWARDS.md` — full PR/FAQ + engineering reasoning
- `design/PITCH.md` — plain-language version
- `scripts/vv/FINDINGS.md` — overnight V&V calibration results
- `brief.md` — the original concept brief

## Commits — always credit Claude explicitly

This project is deliberately an experiment in Claude-assisted development, so the
git history must make Claude's involvement **obvious and auditable**. On every
commit that involved Claude:

- **Append a `Co-Authored-By` trailer** as the final line of the message:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  ```
  Name the exact model that did the work.
- **Write a real body** for non-trivial changes, briefly saying *what Claude did*
  (analysis, drafting, research, code) so the credit is specific, not boilerplate.
- **Never fabricate co-authorship** — credit Claude only for work Claude actually
  did. Honesty over ritual.
- Only commit or push when the user asks.

The `/commit` skill (`.claude/skills/commit/SKILL.md`) documents the same policy
with examples; this file is the always-loaded backstop so the rule holds even
when the skill isn't explicitly invoked.
