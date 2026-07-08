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
next** — operational essentials (uv toolchain, checks), current state, the full roadmap,
architecture decisions & gotchas, and empirical findings. Read it before substantive work.

Design docs (deeper reasoning):
- `design/GENERATOR_DESIGN.md` — literature-grounded generator design (count model +
  the resolved bistability crux; the two `/deep-research` syntheses)
- `design/WORKING_BACKWARDS.md` — full PR/FAQ + engineering reasoning
- `design/PITCH.md` — plain-language version
- `scripts/vv/FINDINGS.md` — V&V calibration + Tier-0.5 findings (the measured results)
- `JUDGES_GUIDE.md` — guided tour of the project against the hackathon judging criteria
- `brief.md` — the original concept brief (a historical record; **do not update it**)

## Keep the living docs fresh (don't let them go stale)

Several docs describe *current* state and will rot silently if not maintained. When you
land a substantive change (a new capability, a resolved/failed experiment, a calibration
result), update the affected ones **in the same change**, and prefer editing the existing
section over appending a new one:

- **`design/STATE.md`** — current state, roadmap %, architecture gotchas. The first thing
  a fresh session reads; keep its "what's done / what's next" honest.
- **`scripts/vv/FINDINGS.md`** — the measured results. Add a numbered finding; don't let a
  claim outlive the evidence.
- **`JUDGES_GUIDE.md`** — the judges-facing tour. Keep its criterion pointers and status
  claims matched to reality (it must never overclaim — see the honesty rule below).
- **`README.md` "Status"** and **`CHANGELOG.md` `[Unreleased]`** — the public front door.
- Historical records (`brief.md`, past design PR/FAQs) are **not** living docs — leave them
  as written; correct forward in the living docs instead.

## Optimize for the judging criteria (and proactively suggest how)

This is a hackathon entry. Keep the four judging criteria in mind during **all** work,
and — this is a standing instruction — **whenever you can see a concrete way to raise the
project on any of them, say so proactively**, don't wait to be asked. Prefer the highest-
leverage, currently-weakest area. When you propose an improvement, tie it to a specific
criterion and be concrete (a file to add, a demo to build, a result to measure).

**Hard constraint: improving a criterion must never mean overclaiming.** The project's
whole thesis is *never claim more than you've measured* (see the honesty rule below).
Strengthen Impact/Demo by making things *actually* better or *actually* demoable — never
by inflating language. A polished-but-false claim scores worse than an honest gap.

1. **Impact (25%)** — real-world potential; who benefits and how much. Could it become
   something people use (builder) / a finding others can build on (researcher)? Fit the
   track's problem statement.
2. **Claude Use (25%)** — creative use of Claude Code beyond a basic application;
   surfacing capabilities that surprise. *(Current strength.)*
3. **Depth & Execution (20%)** — pushed past the first idea; sound, thoughtfully-refined
   engineering; evidence it was genuinely wrestled with. *(Current strength.)*
4. **Demo (30%)** — a working, compelling demo; software you could use / findings you
   trust; genuinely cool to watch. *(Highest weight and currently the weakest — a guided
   notebook or visual walkthrough is the biggest single score lever; flag chances to build
   toward it.)*

`JUDGES_GUIDE.md` is our answer to these criteria; keep it honest and current.

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
- **Only credit a model you can actually verify.** A subagent's `model:` override can
  silently no-op (it once did — subagents launched with `model:"fable"` did *not* run on
  Fable; docs and two commits mis-credited "Claude Fable 5"). Do **not** assert a model
  ran unless you have evidence. When unsure, credit the model you can confirm (the main
  loop's) or describe the work without naming a specific sub-model — never guess. This
  applies to any tool/model override: don't claim it took effect without proof.
- Only commit or push when the user asks.

The `/commit` skill (`.claude/skills/commit/SKILL.md`) documents the same policy
with examples; this file is the always-loaded backstop so the rule holds even
when the skill isn't explicitly invoked.
