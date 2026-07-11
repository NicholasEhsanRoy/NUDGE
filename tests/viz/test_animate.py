"""Smoke tests for the animation engine (viz/animate.py; design §5.2).

A few-frame GIF must render to a non-empty file, and — the honesty lock — an abstaining
fit must report ``abstained is True`` (the overlay is stamped per-frame off the verdict).
Kept to a handful of frames so it is fast under CI (design §6: keep animation smoke tests
≤ a few frames).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

import nudge.viz as viz  # noqa: E402


def _constitutive(call: str) -> dict[str, Any]:
    return {
        "kind": "constitutive", "call": call, "reason": "test", "label": "circuit",
        "asserts_biological_switch": call == "biological-switch",
        "n_grid": [1.0, 2.0, 3.0, 4.0, 5.0],
        "loss_no_control": [1.0, 1.0, 1.0, 1.0, 1.0],
        "loss_with_control": [5.0, 2.0, 1.0, 1.5, 2.0],
        "n1_rejection": 4.0, "argmin_n_with_control": 3.0,
        "calibration": {"h": 6.0, "km": 0.5, "vmax": 20.0, "base": 0.1, "r2": 0.99,
                        "is_nonlinear": True},
        "ground_truth": None,
    }


def test_constitutive_flip_gif_renders(tmp_path: Path) -> None:
    out = tmp_path / "flip.gif"
    fr = viz.render(_constitutive("biological-switch"), str(out), kind="constitutive",
                    animate=True, anim_frames=4, anim_fps=6)
    assert fr.path == str(out)
    assert out.exists() and out.stat().st_size > 0
    assert fr.kind == "constitutive"
    assert fr.abstained is False
    # Provenance code + data sidecar are emitted for the animation too.
    assert fr.code_path is not None and Path(fr.code_path).exists()
    assert fr.data_path is not None and Path(fr.data_path).exists()

    from PIL import Image

    with Image.open(out) as im:
        assert getattr(im, "n_frames", 1) >= 2


def test_abstaining_flip_reports_abstained(tmp_path: Path) -> None:
    """LOAD-BEARING: an unresolved fit's animation still carries the abstention."""
    out = tmp_path / "flip_ab.gif"
    fr = viz.render(_constitutive("unresolved"), str(out), kind="constitutive",
                    animate=True, anim_frames=4, emit_code=False)
    assert out.exists() and out.stat().st_size > 0
    assert fr.abstained is True


def test_animate_requires_out() -> None:
    with pytest.raises(ValueError, match="requires an out"):
        viz.render(_constitutive("biological-switch"), None, kind="constitutive",
                   animate=True)


def test_unanimatable_kind_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no animation for kind"):
        viz.render({"kind": "epistasis", "call": "additive"},
                   str(tmp_path / "x.gif"), kind="epistasis", animate=True)
