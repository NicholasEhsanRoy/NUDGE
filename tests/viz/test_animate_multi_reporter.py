"""Multi-reporter animator smoke + honesty tests (viz/multi_reporter.py build_animation).

Drives ``build_animation`` DIRECTLY (not ``viz.render(..., animate=True)`` — the ``_ANIMATORS``
registry entry is landed later by the integrator, so these tests must not depend on it). A
few-frame GIF must render, and — the honesty lock — an abstaining verdict (``off-model``: no
single shared latent) must stamp ``abstained is True``, while a resolved ``ceiling`` (the JOINT
genuinely resolved) must not. Synthetic dicts only (no re-fit) so it is fast under CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402

from nudge.viz.multi_reporter import build_animation  # noqa: E402


def _mr(call: str = "ceiling", winner: str = "ceiling") -> dict[str, Any]:
    return {
        "kind": "multi_reporter", "call": call, "reason": "test", "label": "3-reporter panel",
        "winner": winner, "knob_margin": 3.0, "effect_margin": 6.0, "n_reporters": 3,
        "losses": {"no_effect": 60.0, "threshold": 11.0, "gain": 35.0, "ceiling": 1.0,
                   "full": 0.9},
        "reporters": [
            {"name": "R0", "r2_shared": 0.98, "r2_independent": 0.99},
            {"name": "R1", "r2_shared": 0.97, "r2_independent": 0.95},
            {"name": "R2", "r2_shared": 0.96, "r2_independent": 0.88},
        ],
    }


def test_multi_reporter_animation_renders(tmp_path: Path) -> None:
    """A resolved 'ceiling' result renders a multi-frame GIF and does NOT abstain."""
    spec = build_animation(_mr("ceiling"), frames=5)
    out = tmp_path / "mr.gif"
    spec.anim.save(str(out), writer=PillowWriter(fps=6))
    plt.close(spec.fig)
    assert out.stat().st_size > 0
    assert spec.abstained is False
    from PIL import Image

    with Image.open(out) as im:
        assert getattr(im, "n_frames", 1) >= 2


def test_multi_reporter_abstention_overlay_fires(tmp_path: Path) -> None:
    """LOAD-BEARING: an 'off-model' panel (no single shared latent) reads as an abstention."""
    spec = build_animation(_mr("off-model", winner=""), frames=5)
    out = tmp_path / "mr_ab.gif"
    spec.anim.save(str(out), writer=PillowWriter(fps=6))
    plt.close(spec.fig)
    assert out.stat().st_size > 0
    assert spec.abstained is True
