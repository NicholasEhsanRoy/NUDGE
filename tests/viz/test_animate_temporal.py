"""Temporal / gLV animator smoke + honesty tests (viz/temporal.py build_animation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

import nudge.viz as viz  # noqa: E402


def _temporal_anim(call: str = "susceptibility") -> dict[str, Any]:
    t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ref = [[1.0, 0.8, 0.6]] * 7
    pert = [[1.0, 0.8, 0.6], [0.9, 0.8, 0.6], [0.7, 0.8, 0.65], [0.4, 0.8, 0.7],
            [0.3, 0.75, 0.75], [0.5, 0.7, 0.7], [0.7, 0.72, 0.66]]
    return {
        "kind": "temporal", "label": "gLV community", "call": call,
        "reason": "test", "selected_knob": call,
        "identifiability": {"cond_number": 50.0, "degenerate": call == "unresolved"},
        "animation": {
            "t": t, "reference": ref, "perturbed": pert, "pulse_window": [2.0, 4.0],
            "target": 0, "species_labels": ["taxon 0 (target)", "taxon 1", "taxon 2"],
        },
    }


def test_temporal_animation_renders(tmp_path: Path) -> None:
    out = tmp_path / "temporal.gif"
    fr = viz.render(_temporal_anim(), str(out), kind="temporal", animate=True,
                    anim_frames=5, anim_fps=6)
    assert fr.path == str(out) and out.exists() and out.stat().st_size > 0
    assert fr.kind == "temporal" and fr.abstained is False
    from PIL import Image

    with Image.open(out) as im:
        assert getattr(im, "n_frames", 1) >= 2


def test_temporal_abstention_overlay_fires(tmp_path: Path) -> None:
    """LOAD-BEARING: the degenerate α⇄β case (unresolved) stamps the overlay."""
    fr = viz.render(_temporal_anim("unresolved"), str(tmp_path / "temporal_ab.gif"),
                    kind="temporal", animate=True, anim_frames=5, emit_code=False)
    assert fr.abstained is True


def test_temporal_animation_needs_frames(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="enriched 'animation' block"):
        viz.render({"kind": "temporal", "call": "susceptibility"},
                   str(tmp_path / "x.gif"), kind="temporal", animate=True)
