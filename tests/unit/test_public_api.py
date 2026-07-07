"""The public API surface is importable and stable."""

from __future__ import annotations

import nudge


def test_public_api_exports() -> None:
    for name in (
        "fit",
        "design",
        "Circuit",
        "CircuitBuilder",
        "MechanismClass",
        "MechanismMap",
        "generate_synthetic_perturbseq",
    ):
        assert hasattr(nudge, name), f"missing public export: {name}"


def test_version() -> None:
    assert nudge.__version__ == "0.1.0"
