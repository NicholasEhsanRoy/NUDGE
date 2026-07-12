"""Differential attribution renderer (``NUDGE-METHOD-010``).

The SAME perturbation in two contexts (resistant vs sensitive; disease vs healthy): which
single knob — **threshold (K)** / **gain (n)** / **ceiling (v_max)** — differs, a call
linear differential expression structurally cannot make, or an honest abstention.

Two panels: (left) the **model-selection BIC** of each single-knob-difference model
relative to the shared (no-difference) null — bars below zero beat the null, and the
selected knob is highlighted; and (right) the **winning knob's Δ** (log2 ratio B/A) with
its bootstrap CI against the zero (no-difference) line — the verdict panel, hatched when
the result abstains (``no-difference`` / ``unresolved``). Reads a ``DifferentialResult``,
the ``service.differential_to_dict`` flat dict, or the canonical replay dict.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nudge.viz._util import get, verdict_color
from nudge.viz.base import Panel, RenderedFigure, is_abstention
from nudge.viz.theme import apply_theme

# BIC model key → the human label drawn on the bar.
_MODEL_LABEL = {
    "shared": "shared\n(null)",
    "K": "threshold\n(K)",
    "n": "gain\n(n)",
    "vmax": "ceiling\n(v_max)",
}
_KNOB_OF_CALL = {"threshold-diff": "K", "gain-diff": "n", "ceiling-diff": "vmax"}


def _f(x: Any, d: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def differential_data(obj: Any) -> dict[str, Any]:
    """Normalise a differential result (dataclass / dict / replay) to a figure dict."""
    if isinstance(obj, dict) and obj.get("kind") == "differential":
        return obj
    fit = get(obj, "fit", default=obj)
    bic = get(fit, "bic", default={}) or {}
    ci = get(fit, "ci_log2", default=(float("nan"), float("nan")))
    return {
        "kind": "differential",
        "label": str(get(obj, "label", default="A vs B")),
        "call": str(get(obj, "call", default="unresolved")),
        "reason": str(get(obj, "reason", default="")),
        "selected": str(get(fit, "selected", "selected_model", default="shared")),
        "best_diff": str(get(fit, "best_diff", "best_diff_model", default="K")),
        "bic": {k: _f(v) for k, v in dict(bic).items()},
        "log2_ratio": _f(get(fit, "log2_ratio")),
        "ci_log2": [_f(ci[0]), _f(ci[1])],
    }


def _caption(d: dict[str, Any]) -> str:
    call = d["call"].upper()
    if is_abstention(d["call"]):
        return f"{d['label']} → {call} ({d['reason'][:80] or 'no single knob resolved'})"
    r = d["log2_ratio"]
    lo, hi = d["ci_log2"]
    return f"{d['label']} → {call} (log2 B/A={r:+.2f}, CI [{lo:+.2f}, {hi:+.2f}])"


def _draw_bic(ax: Any, d: dict[str, Any], pal: dict[str, str], color: str) -> None:
    bic = d["bic"]
    shared = bic.get("shared", 0.0)
    models = [m for m in ("shared", "K", "n", "vmax") if m in bic]
    xs = np.arange(len(models))
    deltas = [bic[m] - shared for m in models]  # <0 ⇒ beats the no-difference null
    selected = d["selected"]
    colors = [color if m == selected and not is_abstention(d["call"]) else pal["muted"]
              for m in models]
    ax.bar(xs, deltas, color=colors, alpha=0.9, zorder=3, width=0.6)
    ax.axhline(0.0, color=pal["text"], lw=1.2, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([_MODEL_LABEL.get(m, m) for m in models], fontsize=8)
    ax.set_ylabel("ΔBIC vs shared null  (lower = favoured)")
    ax.set_title("which knob differs?", fontweight="bold", color=pal["text"])
    ax.annotate("no-difference null", xy=(0.0, 0.0), xycoords=("axes fraction", "data"),
                xytext=(3, 3), textcoords="offset points", fontsize=7.5,
                color=pal["muted"], ha="left", va="bottom")


def _draw_delta(ax: Any, d: dict[str, Any], pal: dict[str, str], color: str) -> None:
    r = d["log2_ratio"]
    lo, hi = d["ci_log2"]
    knob = _KNOB_OF_CALL.get(d["call"], d["best_diff"])
    ax.axhline(0.0, color=pal["text"], lw=1.2, zorder=2)
    if np.isfinite(r):
        yerr = np.array([[max(r - lo, 0.0)], [max(hi - r, 0.0)]]) if np.isfinite(lo) \
            else None
        ax.errorbar([0.0], [r], yerr=yerr, fmt="o", color=color, ecolor=color,
                    elinewidth=2.0, capsize=6, markersize=11, zorder=4,
                    label=f"{knob} log2(B/A)")
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_ylabel("log2 ratio  B / A")
    ax.set_title("winning-knob difference", fontweight="bold", color=pal["text"])
    ax.annotate("equal (no difference)", xy=(0.02, 0.0), xycoords=("axes fraction", "data"),
                xytext=(0, 3), textcoords="offset points", fontsize=7.5,
                color=pal["muted"], ha="left", va="bottom")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


def build(obj: Any, *, theme: str = "auto") -> RenderedFigure:
    """Build the differential figure (no overlay — the pipeline stamps it off the call)."""
    import matplotlib.pyplot as plt

    pal = apply_theme(theme)
    d = differential_data(obj)
    color = verdict_color(d["call"], pal)
    fig, (ax_bic, ax_delta) = plt.subplots(1, 2, figsize=(10.2, 4.4),
                                           gridspec_kw={"width_ratios": [1.5, 1.0]})
    _draw_bic(ax_bic, d, pal, color)
    _draw_delta(ax_delta, d, pal, color)
    fig.suptitle(f"{d['label']}  →  {d['call'].upper()}", fontweight="bold",
                 color=pal["text"], fontsize=13)
    fig.tight_layout()
    panels = [
        Panel(ax=ax_bic, call="", reason="", label=d["label"]),
        Panel(ax=ax_delta, call=d["call"], reason=d["reason"], label=d["label"]),
    ]
    return RenderedFigure(
        fig=fig, panels=panels, kind="differential", caption=_caption(d), data=d
    )
