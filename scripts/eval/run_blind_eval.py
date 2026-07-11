#!/usr/bin/env python3
"""Headless runner for the automated-scientist blind eval — SET UP, dry-run by default.

Given a blind case (from ``blind_harness.py``), this builds an isolated sandbox holding ONLY the
scrubbed data + a task prompt, wires up (or withholds) NUDGE via MCP, denies web search, and
constructs the headless ``claude`` command. **It does NOT run anything unless you pass ``--run``**
— so you can set it all up now and execute it later on a stable connection.

The ABLATION is a first-class flag:
  --mode with-nudge     the agent gets nudge-mcp (attribute/dose_response/... MCP tools)
  --mode without-nudge  the agent gets only generic tools + the raw data (the control arm)

The held-out answer key is NEVER copied into the sandbox — grading (``grade_blind_eval.py``)
reads it separately.

Usage (dry-run — prints the command + builds the sandbox, runs nothing):
  uv run python scripts/eval/run_blind_eval.py --agent-dir eval_cases/dose_switch \
      --surface dose-response --mode with-nudge
Add --run to actually invoke claude (do this when on a stable connection).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# A neutral, honest cover: the agent is TOLD it's a synthetic/fictional test (not a lie) — which
# is exactly the signal to use the tools, not memory. It must NOT hint at threshold/gain/ceiling.
_TASK = """\
You are an automated scientist working a FICTIONAL, synthetic test case (it is not real data and
the answer is not in the literature — so you cannot look it up; you must reason from the data and
the tools, exactly as on a genuinely novel problem). A lab reports that a newly-characterized
perturbation was applied and profiled; the data is in this directory ({data_files}).

Your task: determine the MECHANISM of the perturbation as precisely as the data supports — and,
crucially, ABSTAIN honestly if the data cannot decide (a calibrated "I can't tell / need X" is a
correct scientific answer; a confident wrong answer is the worst outcome).

Rules:
- Do NOT search the web (it is disabled). Reason only from the data{tools_line}.
- Keep an append-only REPORT.md in this directory: your hypothesis, every tool call and how you
  read its output, the abstentions you hit and why, and a final conclusion WITH an explicit
  confidence level or an explicit abstention.
- Be honest and calibrated. State what the data supports and what it does not.
"""

_WITH_NUDGE_TOOLS = (
    ", using the NUDGE tools available to you via MCP (attribute / dose_response / cross_modality "
    "/ etc.) — NUDGE is a mechanism-attribution tool that fits circuit models and abstains when it "
    "cannot tell"
)
_WITHOUT_NUDGE_TOOLS = " using your own analysis (you have generic Python/file tools, but no NUDGE)"

# The differential surface asks a COMPARATIVE question (two contexts), so it gets its own framing.
_TASK_DIFFERENTIAL = """\
You are an automated scientist working a FICTIONAL, synthetic test case (it is not real data and
the answer is not in the literature — so you cannot look it up; you must reason from the data and
the tools, exactly as on a genuinely novel problem). The SAME perturbation was applied in TWO
experimental contexts, A and B (think two cell lines / donors), each profiled together with its
OWN untreated control. The data is in this directory ({data_files}) — a `.npz` with activity-space
arrays `data_a` / `control_a` (context A's perturbed cells + its control) and `data_b` /
`control_b` (context B).

Your task: determine whether the perturbation acts via a DIFFERENT MECHANISM in context B than in
context A — a difference in the switch's threshold, gain, or ceiling — OR whether any apparent
difference is NOT a real mechanistic difference (e.g. a per-condition technical/batch effect).
Crucially, ABSTAIN honestly if the data cannot decide: a confident WRONG mechanism-difference call
is the worst outcome.

Rules:
- Do NOT search the web (it is disabled). Reason only from the data{tools_line}.
- Keep an append-only REPORT.md in this directory: your hypothesis, every tool call and how you
  read its output, the abstentions you hit and why, and a final conclusion WITH an explicit
  confidence level or an explicit abstention.
- Be honest and calibrated. State what the data supports and what it does not.
"""

_WITH_NUDGE_TOOLS_DIFF = (
    ", using the NUDGE tools via MCP. For a two-context comparison NUDGE offers `differential` "
    "(BIC-selects which switch knob — threshold / gain / ceiling — differs between the contexts, "
    "or abstains) and `differential_robust` (the same, but hardened against a per-condition "
    "technical nuisance on one context's perturbed cells — it returns a mechanism difference only "
    "if the knob EARNS its parameter over a free affine confound, else it abstains). Both read the "
    "`.npz` with arrays `data_a` / `control_a` / `data_b` / `control_b`"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent-dir", required=True, help="the blind case dir (scrubbed data)")
    ap.add_argument("--surface", required=True)
    ap.add_argument("--mode", choices=["with-nudge", "without-nudge"], default="with-nudge")
    ap.add_argument("--sandbox", default=None, help="where to build the run sandbox (default: "
                    "<agent-dir>/_run_<mode>)")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--run", action="store_true", help="ACTUALLY invoke claude (default: dry-run)")
    args = ap.parse_args()

    agent_dir = Path(args.agent_dir).resolve()
    data_files = sorted(p.name for p in agent_dir.iterdir() if p.is_file())
    if not data_files:
        raise SystemExit(f"no data files in {agent_dir} — generate a case with blind_harness.py")

    sandbox = Path(args.sandbox).resolve() if args.sandbox else agent_dir / f"_run_{args.mode}"
    sandbox.mkdir(parents=True, exist_ok=True)
    # Copy ONLY the scrubbed data into the sandbox (never the answer key).
    for f in data_files:
        shutil.copy2(agent_dir / f, sandbox / f)

    with_nudge = args.mode == "with-nudge"
    # Deny web tools (both arms). Enforced via project settings.
    settings = {"permissions": {"deny": ["WebSearch", "WebFetch"]}}
    (sandbox / ".claude").mkdir(exist_ok=True)
    (sandbox / ".claude" / "settings.json").write_text(json.dumps(settings, indent=2))
    # Wire nudge-mcp only in the with-nudge arm (run it in the repo env regardless of cwd).
    if with_nudge:
        mcp = {"mcpServers": {"nudge": {
            "command": "uv", "args": ["run", "--project", str(REPO), "nudge-mcp"]}}}
        (sandbox / ".mcp.json").write_text(json.dumps(mcp, indent=2))
    elif (sandbox / ".mcp.json").exists():
        (sandbox / ".mcp.json").unlink()

    is_diff = args.surface == "differential"
    template = _TASK_DIFFERENTIAL if is_diff else _TASK
    if with_nudge:
        tools_line = _WITH_NUDGE_TOOLS_DIFF if is_diff else _WITH_NUDGE_TOOLS
    else:
        tools_line = _WITHOUT_NUDGE_TOOLS
    task = template.format(data_files=", ".join(data_files), tools_line=tools_line)
    (sandbox / "TASK.md").write_text(task)

    allowed = "Read,Write,Edit,Bash" + (",mcp__nudge__*" if with_nudge else "")
    # NOTE: the prompt MUST be the value of ``-p`` (or piped via stdin); a trailing positional
    # prompt is rejected by the CLI ("Input must be provided ... when using --print").
    cmd = ["claude", "-p", task, "--output-format", "json", "--model", args.model,
           "--allowedTools", allowed]

    print(f"=== blind eval · surface={args.surface} · mode={args.mode} ===")
    print(f"sandbox: {sandbox}")
    print(f"  data (scrubbed): {data_files}   nudge-mcp: {'YES' if with_nudge else 'no (control)'}"
          "   web: DENIED")
    print("  answer key: HELD OUT (not in sandbox) — grade with grade_blind_eval.py")
    print("command (run from the sandbox dir):")
    print("  cd " + str(sandbox) + " && \\\n    " + " ".join(
        (repr(c) if " " in c or "\n" in c else c) for c in cmd) + " > run_result.json")

    if not args.run:
        print("\n[DRY RUN] nothing executed. Re-run with --run on a stable connection.")
        return 0

    print("\n[RUN] invoking claude headless …")
    proc = subprocess.run(cmd, cwd=sandbox, capture_output=True, text=True)
    (sandbox / "run_result.json").write_text(proc.stdout)
    if proc.stderr:  # keep stderr on disk so a failed run is diagnosable (never silently lost)
        (sandbox / "run_stderr.txt").write_text(proc.stderr)
    print(f"exit={proc.returncode}; wrote {sandbox / 'run_result.json'} and "
          f"{sandbox / 'REPORT.md'} (if the agent produced it)")
    if proc.returncode != 0:
        print(f"  [nonzero exit] see {sandbox / 'run_stderr.txt'} — first line: "
              f"{(proc.stderr or '').splitlines()[0] if proc.stderr else '(no stderr)'}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
