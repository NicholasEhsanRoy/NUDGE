"""Robustness animator smoke + honesty tests (viz/robustness.py build_animation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

import nudge.viz as viz  # noqa: E402


def _robust_anim(final_call: str = "not-bistable") -> dict[str, Any]:
    x = [0.0, 0.5, 1.0, 1.5, 2.0]
    return {
        "kind": "robustness", "label": "1-node switch", "call": final_call,
        "reason": "swept to the fold", "proximity": None, "one_sided": False,
        "animation": {
            "x": x, "u_max": 1.0,
            "frames": [
                {"n": 6.0, "U": [0.0, 0.4, 0.2, 0.5, 0.1], "proximity": 0.08,
                 "one_sided": False, "channel_proximities":
                 {"critical_slowing": 0.1, "basin_collapse": 0.05, "lobe_overlap": 0.0},
                 "call": "robust", "reason": "deep basin",
                 "fixed_points": [[0.2, "stable"], [1.0, "saddle-index1"], [1.8, "stable"]]},
                {"n": 2.0, "U": [0.0, 0.15, 0.1, 0.12, 0.0], "proximity": 0.6,
                 "one_sided": True, "channel_proximities":
                 {"critical_slowing": 0.6, "basin_collapse": 0.5, "lobe_overlap": 0.4},
                 "call": "near-fold", "reason": "close to fold",
                 "fixed_points": [[0.4, "stable"], [0.9, "saddle-index1"], [1.4, "stable"]]},
                {"n": 1.4, "U": [0.0, 0.02, 0.0, 0.03, 0.05], "proximity": None,
                 "one_sided": False, "channel_proximities":
                 {"critical_slowing": 1.0, "basin_collapse": 1.0, "lobe_overlap": 1.0},
                 "call": final_call, "reason": "monostable",
                 "fixed_points": [[0.8, "stable"]]},
            ],
        },
    }


def test_robustness_animation_renders(tmp_path: Path) -> None:
    out = tmp_path / "rob.gif"
    fr = viz.render(_robust_anim(), str(out), kind="robustness", animate=True,
                    anim_frames=5, anim_fps=6)
    assert fr.path == str(out) and out.exists() and out.stat().st_size > 0
    assert fr.kind == "robustness"
    from PIL import Image

    with Image.open(out) as im:
        assert getattr(im, "n_frames", 1) >= 2


def test_robustness_abstention_overlay_fires(tmp_path: Path) -> None:
    """LOAD-BEARING: a sweep that ends past the fold (not-bistable) reads as an abstention."""
    fr = viz.render(_robust_anim("not-bistable"), str(tmp_path / "rob_ab.gif"),
                    kind="robustness", animate=True, anim_frames=5, emit_code=False)
    assert fr.abstained is True


def test_robustness_animation_needs_frames(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="enriched 'animation' block"):
        viz.render({"kind": "robustness", "call": "robust"},
                   str(tmp_path / "x.gif"), kind="robustness", animate=True)
