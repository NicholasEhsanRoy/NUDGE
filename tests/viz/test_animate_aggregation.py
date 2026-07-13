"""Aggregation gauge-orbit animator smoke + honesty tests (viz/aggregation.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

import nudge.viz as viz  # noqa: E402


def _agg_anim(identifiable: bool = False) -> dict[str, Any]:
    orbit = [
        {"alpha": 1.0, "k_n": 0.45, "k_plus": 0.10, "k_2": 5.0},
        {"alpha": 2.0, "k_n": 0.225, "k_plus": 0.20, "k_2": 2.5},
        {"alpha": 0.5, "k_n": 0.90, "k_plus": 0.05, "k_2": 10.0},
    ]
    return {
        "kind": "aggregation", "call": "composites-identified", "reason": "gauge",
        "label": "amyloid aggregation", "kappa": 1.0, "lambda": 0.3,
        "individual_k_identifiable": identifiable,
        "null_direction": [0.577, -0.577, 0.577],
        "animation": {
            "t": [0.0, 1.0, 2.0, 3.0, 4.0], "m": [0.0, 0.1, 0.5, 0.9, 1.0],
            "kappa": 1.0, "lambda": 0.3, "gauge_check": 1e-7,
            "k_labels": ["kₙ", "k₊", "k₂"], "orbit": orbit,
        },
    }


def test_aggregation_gauge_orbit_renders(tmp_path: Path) -> None:
    out = tmp_path / "agg.gif"
    fr = viz.render(_agg_anim(), str(out), kind="aggregation", animate=True,
                    anim_frames=6, anim_fps=8)
    assert fr.path == str(out) and out.exists() and out.stat().st_size > 0
    assert fr.kind == "aggregation"
    from PIL import Image

    with Image.open(out) as im:
        assert getattr(im, "n_frames", 1) >= 2


def test_aggregation_abstains_on_unidentifiable_constants(tmp_path: Path) -> None:
    """LOAD-BEARING: the individual-constant panel abstains (gauge-degenerate) by default."""
    fr = viz.render(_agg_anim(identifiable=False), str(tmp_path / "agg_ab.gif"),
                    kind="aggregation", animate=True, anim_frames=6, emit_code=False)
    assert fr.abstained is True


def test_aggregation_resolved_when_constants_identifiable(tmp_path: Path) -> None:
    """A series+anchor case (individual constants identifiable) does NOT abstain."""
    fr = viz.render(_agg_anim(identifiable=True), str(tmp_path / "agg_ok.gif"),
                    kind="aggregation", animate=True, anim_frames=6, emit_code=False)
    assert fr.abstained is False


def test_aggregation_animation_needs_frames(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="enriched 'animation' block"):
        viz.render({"kind": "aggregation", "call": "composites-identified"},
                   str(tmp_path / "x.gif"), kind="aggregation", animate=True)
