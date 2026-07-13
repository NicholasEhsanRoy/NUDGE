"""OED animator smoke + honesty tests (viz/oed.py build_animation).

A few-frame GIF must render, and — the honesty lock — an abstaining verdict must stamp the
overlay (``abstained is True``). Kept to a handful of frames + a synthetic animation dict
(no slow re-fit) so it is fast under CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

import nudge.viz as viz  # noqa: E402


def _oed_anim(call: str = "") -> dict[str, Any]:
    return {
        "kind": "oed", "call": call, "reason": "test", "label": "OED",
        "model": "logistic", "objective": "crlb", "target_parameter": "log_alpha",
        "crlb_improvement": 31.0, "min_eig_improvement": 18.0,
        "animation": {
            "param_labels": ["log α", "log |β|"], "theta0": [0.0, -1.0],
            "t_bounds": [0.0, 12.0], "traj_t": [0.0, 3.0, 6.0, 12.0],
            "traj_x": [0.05, 1.0, 1.8, 2.0],
            "frames": [
                {"step": 0, "phi": [7.0, 9.0, 11.0],
                 "ellipse": {"width": 0.6, "height": 0.06, "angle": 50.0},
                 "target_crlb": 0.5},
                {"step": 299, "phi": [0.5, 1.5, 3.0],
                 "ellipse": {"width": 0.16, "height": 0.05, "angle": 68.0},
                 "target_crlb": 0.02},
            ],
        },
    }


def test_oed_animation_renders(tmp_path: Path) -> None:
    out = tmp_path / "oed.gif"
    fr = viz.render(_oed_anim(), str(out), kind="oed", animate=True, anim_frames=4, anim_fps=6)
    assert fr.path == str(out) and out.exists() and out.stat().st_size > 0
    assert fr.kind == "oed" and fr.abstained is False
    assert fr.code_path and Path(fr.code_path).exists()
    from PIL import Image

    with Image.open(out) as im:
        assert getattr(im, "n_frames", 1) >= 2


def test_oed_abstention_overlay_fires(tmp_path: Path) -> None:
    """LOAD-BEARING: an abstaining verdict still stamps the overlay through the animation."""
    fr = viz.render(_oed_anim("unresolved"), str(tmp_path / "oed_ab.gif"), kind="oed",
                    animate=True, anim_frames=4, emit_code=False)
    assert fr.abstained is True


def test_oed_animation_needs_frames(tmp_path: Path) -> None:
    bare = {"kind": "oed", "call": "", "label": "OED"}
    with pytest.raises(ValueError, match="enriched 'animation' block"):
        viz.render(bare, str(tmp_path / "x.gif"), kind="oed", animate=True)
