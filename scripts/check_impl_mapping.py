#!/usr/bin/env python3
"""Verify Mechanism-Card Implementation-Mapping references resolve to real code.

Each ``docs/mechanism_cards/*.md`` may reference ``nudge.module.Class.method``
qualnames (in backticks); this checks each imports and resolves. Template files
(``_*.md``) are skipped. Exit 0 if all resolve, 1 otherwise.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

_QUAL = re.compile(r"`(nudge\.[A-Za-z0-9_.]+)`")


def _resolves(qual: str) -> bool:
    parts = qual.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        try:
            obj: object = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        for attr in parts[split:]:
            obj = getattr(obj, attr, None)
            if obj is None:
                return False
        return True
    return False


def main() -> int:
    cards = Path("docs/mechanism_cards")
    bad: list[str] = []
    for md in sorted(cards.rglob("*.md")):
        if md.name.startswith("_"):
            continue
        for match in _QUAL.finditer(md.read_text()):
            if not _resolves(match.group(1)):
                bad.append(f"{md}: unresolved reference `{match.group(1)}`")
    for msg in bad:
        print(f"check_impl_mapping: {msg}", file=sys.stderr)
    if bad:
        return 1
    print("check_impl_mapping: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
