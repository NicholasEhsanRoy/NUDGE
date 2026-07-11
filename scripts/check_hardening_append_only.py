#!/usr/bin/env python3
"""Enforce the project's audit trails are APPEND-ONLY — no deletions, records immutable.

Two never-deleted, fully-traceable audit trails share this guard:
  * ``design/hardening/`` — the hardening-loop agent record (see design/hardening/README.md).
  * ``design/automated_scientist/`` — the automated-scientist blind-eval record
    (see design/automated_scientist/README.md).

This check fails CI if a change, in EITHER trail:

  * DELETES any file under the trail root (nothing in an audit trail may vanish), or
  * MODIFIES or RENAMES an existing immutable run record under the trail's ``runs/`` dir
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

# (trail root, immutable-records dir) — every trail is guarded identically.
TRAILS: tuple[tuple[str, str], ...] = (
    ("design/hardening/", "design/hardening/runs/"),
    ("design/automated_scientist/", "design/automated_scientist/runs/"),
)


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
        for root, runs in TRAILS:
            if code == "D" and paths[0].startswith(root):
                violations.append(
                    f"DELETED (not allowed anywhere in the audit trail): {paths[0]}")
            elif code == "M" and paths[0].startswith(runs):
                violations.append(
                    f"MODIFIED immutable run record (records are write-once): {paths[0]}")
            elif code == "R":
                old, new = paths[0], paths[-1]
                if old.startswith(runs) or new.startswith(runs):
                    violations.append(f"RENAMED/moved immutable run record: {old} -> {new}")
                if old.startswith(root) and not new.startswith(root):
                    violations.append(f"MOVED a record OUT of the audit trail: {old} -> {new}")

    if violations:
        print("check_hardening_append_only: APPEND-ONLY INVARIANT VIOLATED", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        print(
            "\nThe audit trails (design/hardening/, design/automated_scientist/) are "
            "append-only and never deleted. Add a new record instead of editing or removing one.",
            file=sys.stderr,
        )
        return 1

    print(f"check_hardening_append_only: OK (no deletions/record-mutations vs {base})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
