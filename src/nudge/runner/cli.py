"""The ``nudge`` command-line interface.

Core surface: ``nudge fit`` (fit a circuit to a dataset), ``nudge design``
(invert a fit), ``nudge check-data`` (run the raw-count ingestion guardrail).
Phase-0: the parser is wired; subcommands raise until their engines land.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``nudge`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="nudge",
        description="Mechanism attribution for Perturb-seq screens.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fit", help="fit a circuit to a Perturb-seq dataset")
    sub.add_parser("design", help="invert a fitted circuit to propose interventions")
    sub.add_parser("check-data", help="run the raw-count ingestion guardrail")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``[project.scripts]`` → ``nudge``)."""
    args = build_parser().parse_args(argv)
    raise NotImplementedError(f"nudge {args.command} — not yet implemented")
