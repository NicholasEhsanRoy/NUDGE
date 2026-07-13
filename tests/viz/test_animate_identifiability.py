"""Identifiability animator smoke + honesty tests (viz/identifiability.py build_animation).

Drives ``build_animation`` DIRECTLY (not via ``viz.render(kind=...)``) — the animator is
additive and its registry entry lands separately, so these tests must not depend on it.
The load-bearing assertion is the honesty one: ``sloppy-but-predictive`` is USABLE (NO
overlay, ``abstained is False``) while ``unidentifiable`` IS an abstention (``abstained is
True``) — exactly the "sloppy ≠ unidentifiable" distinction the figure exists to make.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402

from nudge.viz.identifiability import build_animation  # noqa: E402


def _sloppy_dict() -> dict[str, Any]:
    """A sloppy-but-predictive spectrum: many decades, but a real (non-abstaining) call."""
    return {
        "kind": "identifiability",
        "label": "sum-of-exponentials",
        "call": "sloppy-but-predictive",
        "verdict": "sloppy-but-predictive",
        "reason": "SLOPPY but PREDICTIVE: loose parameters, tight predictions, no null.",
        "fim_eigenvalues": [0.0044, 0.079, 7.16, 133.6, 1443.0, 8342.0, 42407.0, 196142.0],
        "cond_number": 4.42e7,
        "span_decades": 7.6,
        "smallest_eigenvalue": 0.0044,
        "largest_eigenvalue": 196142.0,
        "sloppy_decade_threshold": 3.0,
    }


def _unidentifiable_dict() -> dict[str, Any]:
    """A structural-null spectrum → UNIDENTIFIABLE (an abstention: the overlay must fire)."""
    return {
        "kind": "identifiability",
        "label": "degenerate-circuit",
        "call": "unidentifiable",
        "verdict": "unidentifiable",
        "reason": "structural null direction — a parameter combination is unrecoverable.",
        "fim_eigenvalues": [1e-13, 0.5, 3.0, 21.0, 150.0, 900.0],
        "cond_number": 9e15,
        "span_decades": 15.9,
        "smallest_eigenvalue": 1e-13,
        "largest_eigenvalue": 900.0,
        "sloppy_decade_threshold": 3.0,
    }


def test_identifiability_animation_sloppy_renders_no_abstain(tmp_path: Path) -> None:
    """sloppy-but-predictive → renders, is NOT an abstention, and has ≥2 frames."""
    spec = build_animation(_sloppy_dict(), frames=5)
    assert spec.abstained is False
    out = tmp_path / "sloppy.gif"
    spec.anim.save(str(out), writer=PillowWriter(fps=6))
    plt.close(spec.fig)
    assert out.stat().st_size > 0
    from PIL import Image

    with Image.open(out) as im:
        assert getattr(im, "n_frames", 1) >= 2


def test_identifiability_animation_unidentifiable_abstains(tmp_path: Path) -> None:
    """LOAD-BEARING: unidentifiable IS an abstention → overlay fires, abstained is True."""
    spec = build_animation(_unidentifiable_dict(), frames=5)
    assert spec.abstained is True
    out = tmp_path / "unident.gif"
    spec.anim.save(str(out), writer=PillowWriter(fps=6))
    plt.close(spec.fig)
    assert out.stat().st_size > 0
