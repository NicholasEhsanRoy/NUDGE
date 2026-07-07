#!/usr/bin/env python3
"""Creative-AI idea 2 — PDF/paper to JAX Mechanism Card.

Extracts the ODEs from a paper, translates them into a registered JAX mechanism
class, and drafts a Mechanism Card from ``docs/mechanism_cards/_template.md``.
The ``new-mechanism`` skill is the human review checklist for its output.
Stretch stub.
"""

from __future__ import annotations

import sys

_MESSAGE = "needs the ai extra; drafts a mechanism class + Mechanism Card"


def main() -> int:
    print(f"paper_to_mechanism: stub — {_MESSAGE}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
