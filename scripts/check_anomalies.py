#!/usr/bin/env python3
"""Validate ``docs/known_limitations.yaml`` against MADDENING's anomaly schema.

Reuses ``maddening.compliance.validate_anomaly_registry`` with the ``NUDGE-LIM-``
ID prefix — the Appendix F downstream-bootstrap path. Exit 0 if valid, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

from maddening.compliance import validate_anomaly_registry

DEFAULT_PATH = "docs/known_limitations.yaml"
PREFIX = "NUDGE-LIM-"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = args[0] if args else DEFAULT_PATH
    if not Path(path).exists():
        print(f"check_anomalies: {path} not found", file=sys.stderr)
        return 1
    errors = validate_anomaly_registry(path, prefix=PREFIX)
    for err in errors:
        print(f"check_anomalies: {err}", file=sys.stderr)
    if errors:
        return 1
    print(f"check_anomalies: {path} OK (prefix {PREFIX})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
