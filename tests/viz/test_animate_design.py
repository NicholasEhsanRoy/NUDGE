"""Design (inverse-design) animator smoke + honesty tests (viz/design.py build_animation).

Drives ``build_animation`` DIRECTLY (the animator registry entry lands later), on synthetic
dicts, kept to a handful of frames so it is fast under CI. The honesty lock: a *reachable*
plan is a confident WARNING (``abstained is False``, red HIGH-RISK banner ≠ the grey abstain
hatch); an *unreachable* plan is an ABSTENTION (``abstained is True``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402

from nudge.viz.design import build_animation  # noqa: E402

_NAN = float("nan")


def _intervention() -> dict[str, Any]:
    """A reachable, HIGH-RISK circuit-mode plan (crosses the fold, one-sided bound)."""
    return {
        "kind": "design", "design_kind": "intervention", "call": "",
        "label": "flip ON", "verdict": "", "reason": "reachable", "mode": "circuit",
        "deltas": [{"name": "edge0.K", "factor": 0.6}, {"name": "edge1.n", "factor": 1.8}],
        "dose": _NAN, "predicted_response": _NAN,
        "safety": {"proximity_before": 0.3, "proximity_after": _NAN,
                   "crosses_fold": True, "high_risk": True, "one_sided": True},
    }


def _abstention() -> dict[str, Any]:
    return {
        "kind": "design", "design_kind": "abstention", "call": "abstain",
        "label": "flip ON", "verdict": "unreachable",
        "reason": "requested state out of the reachable range", "mode": "",
        "deltas": [], "dose": _NAN, "predicted_response": _NAN, "safety": None,
    }


def test_design_intervention_renders(tmp_path: Path) -> None:
    """A reachable plan (with a high_risk safety block) renders and is NOT an abstention."""
    spec = build_animation(_intervention(), frames=5)
    assert spec.abstained is False
    out = tmp_path / "design.gif"
    spec.anim.save(str(out), writer=PillowWriter(fps=6))
    plt.close(spec.fig)
    assert out.stat().st_size > 0
    from PIL import Image

    with Image.open(out) as im:
        assert getattr(im, "n_frames", 1) >= 2


def test_design_abstention_overlay_fires(tmp_path: Path) -> None:
    """LOAD-BEARING: an unreachable design reads as an abstention (grey hatch, no trajectory)."""
    spec = build_animation(_abstention(), frames=5)
    assert spec.abstained is True
    out = tmp_path / "design_ab.gif"
    spec.anim.save(str(out), writer=PillowWriter(fps=6))
    plt.close(spec.fig)
    assert out.stat().st_size > 0
    # The overlay stamped its machine-detectable marker on the trajectory axis.
    assert any(getattr(a, "_nudge_abstained", False) for a in spec.fig.axes)
