"""Dose-response animator smoke + honesty tests (viz/dose_response.py build_animation).

Drives ``build_animation`` DIRECTLY (not via ``viz.render`` — the animator registry lands
later) and saves a few-frame GIF with matplotlib's bundled ``PillowWriter``. The load-bearing
checks are the honesty locks: a resolved switch strip must NOT abstain, a strip carrying a
flat/one-sided panel MUST (``spec.abstained``), and the SAME sweep must render a
cross-modality variants dict. Kept tiny + synthetic (no re-fit) so it is fast under CI.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
pytest.importorskip("PIL")
mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402

from nudge.viz.dose_response import build_animation  # noqa: E402


def _switch(dose_max: float = 1.0, k: float = 0.5) -> SimpleNamespace:
    """A lightweight resolved-switch result (the full ``_coerce_panel`` path — a namespace,
    not a dict, so dose/response ARE attached and it is not read as a canonical panel)."""
    return SimpleNamespace(
        call="switch", reason="", direction="repress", n=4.0, k_threshold=k, amp=1.0,
        floor=0.05, r2=0.98, ci_n=(3.0, 5.0), ci_k=(0.4, 0.6), spans_inflection=True,
        dose_min=0.05, dose_max=dose_max,
    )


def _abstain_onesided(dose_max: float = 0.6) -> SimpleNamespace:
    """An honest abstention whose K sits PAST the sampled range (one-sided bound)."""
    return SimpleNamespace(
        call="unresolved", reason="K past max dose → gain unidentifiable",
        direction="repress", n=2.0, k_threshold=5.0, amp=1.0, floor=0.05, r2=0.3,
        ci_n=(float("nan"), float("nan")), ci_k=(float("nan"), float("nan")),
        spans_inflection=False, dose_min=0.05, dose_max=dose_max,
    )


def _doses(dmax: float) -> tuple[list[float], list[float]]:
    d = np.linspace(0.05, dmax, 8)
    r = 0.05 + 1.0 * (0.5**4) / (0.5**4 + d**4)  # a plausible repressive response
    return [float(x) for x in d], [float(x) for x in r]


def _save(spec: Any, path: Path) -> None:
    spec.anim.save(str(path), writer=PillowWriter(fps=6))
    plt.close(spec.fig)


def test_resolved_switch_strip_renders_and_does_not_abstain(tmp_path: Path) -> None:
    d, r = _doses(1.0)
    spec = build_animation([("switch-like", _switch(), d, r)], frames=5)
    out = tmp_path / "switch.gif"
    _save(spec, out)
    assert out.stat().st_size > 0
    assert spec.abstained is False
    from PIL import Image

    with Image.open(out) as im:
        assert getattr(im, "n_frames", 1) >= 2  # a real multi-frame sweep


def test_strip_with_flat_abstain_panel_abstains(tmp_path: Path) -> None:
    """LOAD-BEARING: any abstaining panel flips ``spec.abstained`` (its overlay fires)."""
    d0, r0 = _doses(1.0)
    d1, r1 = _doses(0.6)
    entries = [("switch-like", _switch(), d0, r0),
               ("flat / abstain", _abstain_onesided(), d1, r1)]
    spec = build_animation(entries, frames=5)
    out = tmp_path / "mixed.gif"
    _save(spec, out)
    assert out.stat().st_size > 0
    assert spec.abstained is True


def test_cross_modality_variants_dict_renders(tmp_path: Path) -> None:
    """The continuous-readout path: a ``{"variants": [...]}`` dict reuses the same sweep."""
    xmod = {
        "variants": [
            {"variant": "WT", "call": "reference", "K_threshold": 1.0,
             "n_apparent_gain": 2.0, "amp": 1.0, "floor": 0.05, "direction": "activate"},
            {"variant": "mutant", "call": "threshold", "K_threshold": 3.0,
             "n_apparent_gain": 2.0, "amp": 1.0, "floor": 0.05, "direction": "activate"},
        ]
    }
    spec = build_animation(xmod, frames=5)
    out = tmp_path / "xmod.gif"
    _save(spec, out)
    assert out.stat().st_size > 0
    assert spec.abstained is False  # both variants resolved
    assert spec.data.get("kind") == "cross_modality"  # provenance routes back here


def test_cross_modality_abstaining_variant_abstains(tmp_path: Path) -> None:
    """Honesty holds in the REUSED path: a non-responsive variant abstains (one-sided)."""
    xmod = {
        "variants": [
            {"variant": "WT", "call": "reference", "K_threshold": 1.0,
             "n_apparent_gain": 2.0, "amp": 1.0, "floor": 0.05, "direction": "activate"},
            {"variant": "dead", "call": "non-responsive", "K_threshold": 1.0,
             "n_apparent_gain": 1.0, "amp": 0.02, "floor": 0.05, "direction": "activate"},
        ]
    }
    spec = build_animation(xmod, frames=5)
    _save(spec, tmp_path / "xmod_ab.gif")
    assert spec.abstained is True
