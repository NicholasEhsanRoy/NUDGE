#!/usr/bin/env python3
"""Creative-AI idea 1 — AI as the biological red team.

Prompts an LLM (the ``ai`` extra) to author a generator for a synthetic dataset
that contains no real bistability but a technical artifact mimicking a switch,
then registers it as a ``DecoyCase(authored_by="ai", prompt_ref=...)`` in
``src/nudge/data/decoys.py``. Dev-time only; never runs in CI. Phase-3 stub.
"""

from __future__ import annotations

import sys

_MESSAGE = "needs the ai extra; emits an AI-authored DecoyCase"


def main() -> int:
    print(f"generate_decoy: stub — {_MESSAGE}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
