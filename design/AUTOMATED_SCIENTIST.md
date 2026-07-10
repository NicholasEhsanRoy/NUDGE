# Automated-scientist evaluation + demo harness — design & feasibility

**Goal.** A self-improving flywheel that (1) rigorously tests whether NUDGE helps an AI reach a
*correct* mechanism (or an honest abstention), (2) produces watchable demo material, and (3)
generates concrete tickets to improve NUDGE's MCP / UX — optimizing NUDGE both for *automated*
scientific research and for *human* users of Claude Science (the Demo criterion, 30%).

**Honesty is the governing rule** (as everywhere in this project). The eval does NOT grade
point-accuracy alone — it grades **calibration**: did the scientist reach the correct mechanism
*or correctly abstain when the data can't decide*, and did it avoid a confident-wrong? That is
the exact property NUDGE exists to have, now measured on an AI (and human) user.

---

## Three corrections the feasibility research forced (read first)

1. **Claude Science is a code-execution notebook, not a GUI-automation target.** It runs
   Python/R/shell in a sandbox, connects **MCP connectors** (so `nudge-mcp` plugs straight in),
   versions artifacts, and has a built-in **Reviewer** that checks claims against the execution
   log. So it is the *surface the scientist works in*, not something we drive by clicking. Its
   preserved code + artifacts are themselves the "figure code is preserved" grain.
2. **Native computer-use is macOS-only in the CLI, Pro/Max-only, research-preview, and NOT
   available headless.** On Linux (our box) the enabler is the community MCP
   **`agent-sh/computer-use-linux`** (`screenshot`/`click`/`type_text`/`press_key`/AT-SPI window
   control; Wayland-first; `claude mcp add --scope user computer-use-linux -- computer-use-linux
   mcp`; needs AT-SPI + `ydotoold` + `/dev/uinput` in the `input` group). **It does NOT record
   video** — only static screenshots — so screen recording stays a DIY ffmpeg + Wayland-portal
   (or X11) layer.
3. **Headless transcripts capture tool calls/results but NOT the model's reasoning.** So both
   modes must instruct the agent to **narrate its reasoning into a report file as it works**
   (`REPORT.md`), which is the auditable "think-out-loud" artifact — do not rely on the session
   JSONL for reasoning.

---

## Mode 1 — Headless automated scientist (the rigorous eval; lowest risk, build first)

A fully non-interactive run graded for calibration. All tooling exists today.

- **Run:** `claude --bare -p "<task>" --output-format json > result.json` in a scratch repo/dir.
- **No web (enforced):** `settings.json` → `"permissions": {"deny": ["WebSearch", "WebFetch"]}`
  (or `--allowedTools "Read,Bash,mcp__nudge__*"`). The point is a clean test of *the data +
  NUDGE*, not the model's memory of the literature.
- **NUDGE available:** project `.mcp.json` with `nudge` (stdio `nudge-mcp`), auto-loaded in `-p`
  mode. Wrap it in a health check — a crashed stdio server is silent otherwise.
- **Think-out-loud report:** the task prompt requires the agent to keep an append-only
  `REPORT.md` — hypothesis, each NUDGE call + how it read the output, the abstentions it hit and
  why, and its final mechanistic conclusion **with a stated confidence / explicit abstention.**
- **Grading (calibration rubric):** a second (grader) run — or a deterministic script — compares
  `REPORT.md`'s conclusion to the known answer and scores: `correct-call` / `correct-abstention`
  / `wrong-confident` (a fail) / `over-abstained`. **A confident-wrong is the only hard failure**;
  a calibrated abstention where the data can't decide is a *pass*. Also flag *contamination*: did
  the transcript show it reasoning from NUDGE's output, or did it guess from priors?
- **Deliverable:** `REPORT.md` + `result.json` + the transcript JSONL + a grade — a reproducible,
  auditable eval record. Strong Impact/Claude-Use artifact on its own.

## Mode 2 — Human-imitating scientist in Claude Science (the watchable demo)

The demo-facing run; surfaces MCP/UX friction a headless run can't. Two variants, in risk order:

- **2a — Claude Science does the science (recommended first).** Run the same problem *inside
  Claude Science* with `nudge-mcp` as a connector: Claude writes analysis code, calls NUDGE, and
  the Reviewer verifies. Screen-record the session at the OS level (`ffmpeg` via the Wayland
  screencast portal, or an X11 grab) for the video; the notebook's preserved code+artifacts are
  the reproducible record; the agent narrates into a report. This *is* "Claude doing science with
  NUDGE," it's naturally watchable, and it directly exercises the connector/UX we want to improve.
- **2b — Drive the GUI like a human (heaviest; add only if we want literal click-through UX).**
  A Claude Code session with the `computer-use-linux` MCP operates the Claude Science (or browser)
  GUI by screenshot→click→type, imitating a human. This is the truest "human imitation" and
  surfaces GUI friction ("the graph looks weird", a confusing button), but computer-use on a beta
  app is the least reliable piece — gate it behind 2a working.
- Either way: **no web**, narrate-to-report, and OS-level screen recording (`ffmpeg`) is a
  separate layer (`computer-use-linux` gives screenshots, not video).

## Mode 3 — The feedback flywheel (report → human → ticket → Claude Code)

- A **summary agent** reads Mode-1 + Mode-2 reports (+ the video/screenshots) and drafts, for the
  **human in the loop (you)**, a concise findings note + a *proposed* improvement ticket — both
  the machine-visible gaps (a NUDGE-MCP tool that was awkward, an abstention that read as an
  error) and the human-only ones ("the dose-response panel's axis is confusing").
- You approve/edit → it becomes a **GitHub issue** (`gh issue create`) → a **Claude Code** run
  (locally, or the `anthropics/claude-code` GitHub Action on the issue) implements the MCP/UX
  improvement → the loop re-runs to confirm it helped. This closes "tool for AI science" **and**
  "tool for human scientists in Claude Science." (No bidirectional loop exists out of the box —
  it's ~a few scripts of glue: run → parse → issue → review → act.)

---

## The linchpin — problem selection (needs your domain input)

Make-or-break for validity. The target must satisfy **all** of:
- **Post-cutoff** (published after the model's ~Jan-2026 training) so the answer isn't memorized.
- A **public dataset NUDGE can actually run** (Perturb-seq counts / a dose-response series / a
  cross-modality readout / a two-context comparison — something in NUDGE's vocabulary).
- A **known-correct mechanistic answer** to grade against — ideally a **threshold-vs-gain-vs-
  ceiling** (or synergy / differential) call that is **non-obvious from priors**, so the eval
  genuinely tests "did it use the data," and where a **calibrated abstention** may be the correct
  outcome.
- **Contamination guard:** prefer a problem whose answer a strong model cannot guess without the
  data; enforce no-web; and check the transcript reasons from NUDGE's output.

*(If you don't have a candidate in hand, a second research pass can shortlist 2026 screens/dose-
response results with public data + a clean mechanistic answer.)*

---

## Phased MVP (scoped to the freeze) + the biggest risks

| Phase | What | Risk | Blocker |
|---|---|---|---|
| **0** | Pick the problem+dataset; stand up the scratch harness (no-web `.mcp.json` + `nudge-mcp` health check) | low | problem selection (yours) |
| **1** | **Mode 1** headless graded run on 1 problem → `REPORT.md` + grade | **low** — all tooling exists | none |
| **2** | **Mode 2a** recorded Claude Science session (nudge-mcp connector + ffmpeg) | medium | Claude Science beta; screen-record plumbing |
| **3** | **Mode 3** summary → issue → Claude Code UX/MCP fix | medium | DIY glue; gh/action auth |
| **2b** | GUI-driven human-imitation via `computer-use-linux` | **high** | computer-use reliability on a beta GUI |

**Biggest single blocker:** the computer-use GUI-driving (2b) — macOS-only natively, and on Linux
it depends on the community `computer-use-linux` MCP + AT-SPI/ydotoold plumbing + a beta target
app. **Mitigation:** the flywheel's scientific value lives in Mode 1 + Mode 2a; treat 2b as an
optional demo flourish, not a dependency.

**Do NOT let the demo harness overclaim.** If the automated scientist got it wrong, that is a
finding to *show*, not hide — an honest "the AI over-read the data here, and NUDGE's abstention
would have saved it" is a *better* demo than a staged success, and it's on-thesis.

## Feasibility summary (sources)

Possible today: headless `-p` + `--output-format json`, `permissions.deny` web tools, `.mcp.json`
auto-load of `nudge-mcp`, the Agent SDK tool-runner (beta), `gh` + the Claude Code GitHub Action,
Claude Science with MCP connectors + Reviewer, `computer-use-linux` MCP for Linux GUI control.
Beta/gated: computer use (macOS-only CLI, Pro/Max, research preview; **no headless**), Claude
Science (beta), routines (web-only beta), `computer-use-linux` (v0.3.x, Ubuntu-tested). Not a
thing: a "locally-hosted Claude Science" separate from the desktop app; a built-in screen
recorder (DIY ffmpeg); a bidirectional report↔ticket loop (DIY glue). Sources: code.claude.com
(headless, permission-modes, mcp, computer-use), platform.claude.com (tool-runner, managed-agents,
computer-use tool), claude.com/docs/claude-science, github.com/agent-sh/computer-use-linux.
