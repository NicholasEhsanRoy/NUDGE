#!/usr/bin/env python3
"""Verify every ``[@Key]`` citation in ``docs/**/*.md`` resolves in the bibliography.

CI gate: a dangling citation fails the build. Exit 0 if all resolve, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_CITE = re.compile(r"\[@([A-Za-z0-9_:-]+)\]")
_BIBKEY = re.compile(r"@\w+\{\s*([^,\s]+)\s*,")


def main() -> int:
    docs = Path("docs")
    bib = docs / "bibliography.bib"
    keys = set(_BIBKEY.findall(bib.read_text())) if bib.exists() else set()
    missing: list[str] = []
    for md in sorted(docs.rglob("*.md")):
        for match in _CITE.finditer(md.read_text()):
            if match.group(1) not in keys:
                missing.append(f"{md}: undefined citation [@{match.group(1)}]")
    for msg in missing:
        print(f"check_citations: {msg}", file=sys.stderr)
    if missing:
        return 1
    print("check_citations: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
