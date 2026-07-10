#!/usr/bin/env python3
"""Enforce the hardening audit trail is APPEND-ONLY — no deletions, records immutable.

`design/hardening/` is the never-deleted, fully-traceable record of everything the
hardening-loop agents do (see design/hardening/README.md). This check fails CI if a change:

  * DELETES any file under ``design/hardening/`` (nothing in the audit trail may vanish), or
  * MODIFIES or RENAMES an existing immutable run record under ``design/hardening/runs/``
    (a run record is written once and never edited; corrections are NEW records).

Additions anywhere, and modifications to the live index files (``LEDGER.md`` / ``README.md``),
are allowed. Compares the working ``HEAD`` against a base ref (a PR base or the pre-push SHA).

Usage: ``python scripts/check_hardening_append_only.py [BASE_REF]``  (default: origin/main)
Exit 0 if the invariant holds (or the base ref can't be resolved — nothing to compare),
1 on a violation.
"""

from __future__ import annotations

import subprocess
import sys

HARDENING = "design/hardening/"
RUNS = "design/hardening/runs/"


def _run(*args: str) -> tuple[int, str]:
    p = subprocess.run(["git", *args], capture_output=True, text=True)
    return p.returncode, p.stdout


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    base = args[0] if args else "origin/main"

    # If the base ref can't be resolved (first push, all-zeros SHA, shallow clone), there is
    # nothing to diff against — pass rather than block.
    rc, _ = _run("rev-parse", "--verify", "--quiet", f"{base}^{{commit}}")
    if rc != 0 or base.strip("0") == "":
        print(f"check_hardening_append_only: base ref {base!r} unresolved — skipping (OK)")
        return 0

    rc, out = _run("diff", "--name-status", "--find-renames", base, "HEAD")
    if rc != 0:
        print(f"check_hardening_append_only: git diff against {base!r} failed — skipping (OK)")
        return 0

    violations: list[str] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status, paths = parts[0], parts[1:]
        code = status[0]
        # A rename row is "Rxxx\told\tnew"; a delete/modify/add is "X\tpath".
        if code == "D" and paths[0].startswith(HARDENING):
            violations.append(f"DELETED (not allowed anywhere in the audit trail): {paths[0]}")
        elif code == "M" and paths[0].startswith(RUNS):
            violations.append(f"MODIFIED immutable run record (records are write-once): {paths[0]}")
        elif code == "R":
            old, new = paths[0], paths[-1]
            if old.startswith(RUNS) or new.startswith(RUNS):
                violations.append(f"RENAMED/moved immutable run record: {old} -> {new}")
            if old.startswith(HARDENING) and not new.startswith(HARDENING):
                violations.append(f"MOVED a record OUT of the audit trail: {old} -> {new}")

    if violations:
        print("check_hardening_append_only: APPEND-ONLY INVARIANT VIOLATED", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        print(
            "\nThe hardening audit trail (design/hardening/) is append-only and never deleted. "
            "Add a new record instead of editing or removing one.",
            file=sys.stderr,
        )
        return 1

    print(f"check_hardening_append_only: OK (no deletions/record-mutations vs {base})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
