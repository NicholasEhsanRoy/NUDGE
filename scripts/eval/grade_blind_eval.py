#!/usr/bin/env python3
"""Grade a blind-eval run for CALIBRATION — proposes a grade for human confirmation.

Reads the agent's ``REPORT.md`` (its think-out-loud conclusion) + the HELD-OUT answer key, and
proposes one of:
  correct-call        — the agent reached the ground-truth mechanism (a confident, correct answer)
  correct-abstention  — the data can't decide and the agent honestly abstained (a PASS)
  over-abstained      — the agent abstained where the data supported a call (weak, not a fail)
  CONFIDENT-WRONG     — the agent confidently asserted the WRONG mechanism (the ONLY hard fail)

This is a heuristic PROPOSAL (free-text conclusions are judgment calls), surfaced for the human in
the loop to confirm — deliberately, per the honesty rule: the grader does not overclaim certainty
about a calibration judgment. For an at-scale battery, swap the heuristic for an LLM judge given
the same rubric.

Usage:
  uv run python scripts/eval/grade_blind_eval.py \
      --report eval_cases/dose_switch/_run_with-nudge/REPORT.md \
      --key eval_keys/dose_switch_answer_key.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_ABSTAIN = ("unresolved", "abstain", "cannot tell", "can't tell", "cannot determine",
            "insufficient", "need a", "need more", "not identifiable", "underdetermined",
            "inconclusive", "gain_or_threshold", "cannot separate")
_MECHS = ("threshold", "gain", "ceiling", "switch", "graded", "synergistic", "buffering",
          "growth", "interaction", "susceptibility")


def _truth_terms(key: dict) -> list[str]:
    """The ground-truth mechanism term(s) to look for in the report."""
    for field in ("ground_truth_mechanism", "case"):
        if field in key:
            return [str(key[field])]
    gt = key.get("ground_truth", {})
    if isinstance(gt, dict) and "mechanism" in gt:
        return [str(gt["mechanism"])]
    if "expected_nudge_verdict" in key:
        return [str(key["expected_nudge_verdict"])]
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True, help="the agent's REPORT.md")
    ap.add_argument("--key", required=True, help="the HELD-OUT answer_key.json")
    args = ap.parse_args()

    key = json.loads(Path(args.key).read_text())
    report = Path(args.report).read_text().lower() if Path(args.report).exists() else ""
    if not report:
        print("PROPOSED GRADE: NO-REPORT (the agent produced no REPORT.md) — needs human review")
        return 0

    # Look only at the report's conclusion region (last ~1500 chars) for the final call.
    tail = report[-1500:]
    truth = [t.lower() for t in _truth_terms(key)]
    abstained = any(a in tail for a in _ABSTAIN)
    asserted = [m for m in _MECHS if re.search(rf"\b{re.escape(m)}\b", tail)]
    wrong_assert = [m for m in asserted if m not in truth and not abstained]

    # Conservative + honest: only auto-propose a PASS when it is unambiguous; otherwise DEFER to
    # the human (keyword matching cannot parse negation, e.g. "no switch"). The grader must not
    # overclaim certainty about a calibration judgment.
    truth_present = any(t in tail for t in truth)
    if abstained:
        grade = "correct-abstention (honest abstention — a PASS if the data can't decide)"
    elif truth_present and not wrong_assert:
        grade = f"correct-call (reached {truth})"
    elif wrong_assert:
        grade = (f"NEEDS-HUMAN-REVIEW: asserted {wrong_assert} (truth {truth}) with no abstention "
                 "— LIKELY CONFIDENT-WRONG (the hard fail), but confirm; the heuristic can't parse "
                 "negation")
    else:
        grade = "UNCLEAR — no clear call or abstention; needs human review"

    expected = key.get("expected", key.get("expected_nudge_verdict"))
    rubric = key.get("grading_rubric", "calibration; confident-wrong is the only hard fail")
    print("=== blind-eval grade (PROPOSED — confirm with the human in the loop) ===")
    print(f"  case: {key.get('case_id')}  surface: {key.get('surface')}")
    print(f"  ground truth: {truth}   |   expected: {expected}")
    print(f"  report says — abstained: {abstained}   asserted mechanisms: {asserted or 'none'}")
    print(f"  PROPOSED GRADE: {grade}")
    print(f"  rubric: {rubric}")
    print("  -> human confirms/overrides this grade (feeds the summary -> ticket loop).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
