"""Cross-modality readout renderer (``NUDGE-METHOD-002``; the Chure LacI benchmark).

The SAME threshold/gain/ceiling vocabulary, read from a CONTINUOUS single channel
(fluorescence / activity / fold-change) instead of counts. Because a variant's fit has the
identical Hill geometry as a dose-response fit — a K/n/v_max localisation of one curve —
this renderer **reuses the dose-response panel** rather than duplicating it: each variant
becomes one Hill panel in a strip, with the fold-change axis labels and the same honest
one-sided / abstention grammar. The only new code is the variant→panel adapter.
"""

from __future__ import annotations

from typing import Any

from nudge.viz._util import get
from nudge.viz.base import RenderedFigure, is_abstention

_XLABEL = "dose (continuous readout)"
_YLABEL = "readout (rel. control)"


def _f(x: Any, d: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def _pair(x: Any) -> list[float]:
    if x is None:
        return [float("nan"), float("nan")]
    seq = list(x)
    return [_f(seq[0]), _f(seq[1])]


def _variant_to_panel(v: Any) -> dict[str, Any]:
    """Adapt one variant attribution to a canonical dose-response panel dict."""
    call = str(get(v, "call", default="unresolved"))
    return {
        "label": str(get(v, "variant", "label", default="variant")),
        "call": call,
        "reason": str(get(v, "reason", default="")),
        "direction": str(get(v, "direction", default="activate")),
        "n": _f(get(v, "n_apparent_gain", "n", default=1.0), 1.0),
        "k_threshold": _f(get(v, "K_threshold", "k_threshold", default=1.0), 1.0),
        "amp": _f(get(v, "amp", default=1.0), 1.0),
        "floor": _f(get(v, "floor", default=0.0), 0.0),
        "r2": _f(get(v, "r2")),
        "ci_n": _pair(get(v, "ci_n")),
        "ci_k": _pair(get(v, "ci_K", "ci_k")),
        # A resolved knob localisation spans the inflection; an abstaining variant does not
        # get a confident K (it still gets the abstain banner via the pipeline).
        "spans_inflection": not is_abstention(call),
        "dose_min": None,
        "dose_max": None,
        "dose": None,
        "response": None,
    }


def cross_modality_data(obj: Any) -> dict[str, Any]:
    """Normalise a variant panel (``cross_modality_panel_file`` / list / replay) to a dict."""
    if isinstance(obj, dict) and obj.get("kind") == "cross_modality":
        return obj
    if isinstance(obj, dict) and "variants" in obj:
        variants = obj["variants"]
    elif isinstance(obj, (list, tuple)):
        variants = list(obj)
    else:
        variants = [obj]
    return {
        "kind": "cross_modality",
        "panels": [_variant_to_panel(v) for v in variants],
        "xlabel": _XLABEL,
        "ylabel": _YLABEL,
    }


def build(obj: Any, *, theme: str = "auto") -> RenderedFigure:
    """Build the cross-modality strip by REUSING the dose-response renderer."""
    from nudge.viz import dose_response as dr

    data = cross_modality_data(obj)
    # Hand the canonical panels to the dose-response engine (identical Hill geometry).
    dr_data = {
        "kind": "dose_response",
        "panels": data["panels"],
        "xlabel": data["xlabel"],
        "ylabel": data["ylabel"],
    }
    rf = dr.build(dr_data, theme=theme)
    # Re-tag as cross-modality so the FigureResult / provenance replay route back here.
    return RenderedFigure(
        fig=rf.fig, panels=rf.panels, kind="cross_modality", caption=rf.caption,
        data=data,
    )
