"""Epistasis / synergy renderer (``NUDGE-METHOD-003``; Norman combos).

Two panels that make the combination geometry legible: (left) the **additive (Bliss)
null** — A, B, their additive prediction, and the OBSERVED A+B side by side with bootstrap
CIs, so the gap between "predicted if independent" and "observed" is the interaction; and
(right) the **interaction estimate with its CI against the zero (additive) line** — the
verdict panel. A CI that brackets zero is an honest abstention (the overlay fires here).
The off-axis residual is annotated with its standing caveat: it is *consistent with, does
not prove* a hidden regulator (never a hidden-node claim; NUDGE-LIM-015).

Reads an ``EpistasisResult`` (``.fit`` fields), the ``service.synergy_to_dict`` flat dict,
or the canonical replay dict — never re-fits.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nudge.viz._util import get, verdict_color
from nudge.viz.base import Panel, RenderedFigure, is_abstention
from nudge.viz.layout import freest_corner
from nudge.viz.theme import apply_theme


def _f(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _pair(x: Any) -> list[float]:
    if x is None:
        return [float("nan"), float("nan")]
    seq = list(x)
    return [_f(seq[0]), _f(seq[1])]


def epistasis_data(obj: Any) -> dict[str, Any]:
    """Normalise an epistasis result (dataclass / ``synergy_to_dict`` / replay) to a dict."""
    if isinstance(obj, dict) and obj.get("kind") == "epistasis":
        return obj
    fit = get(obj, "fit", default=obj)
    return {
        "kind": "epistasis",
        "label": str(get(obj, "label", default="A × B")),
        "call": str(get(obj, "call", default="unresolved")),
        "reason": str(get(obj, "reason", default="")),
        "effect_a": _f(get(fit, "effect_a")),
        "effect_b": _f(get(fit, "effect_b")),
        "effect_ab": _f(get(fit, "effect_ab")),
        "additive_pred": _f(get(fit, "additive_pred")),
        "interaction": _f(get(fit, "interaction")),
        "ci_a": _pair(get(fit, "ci_a")),
        "ci_b": _pair(get(fit, "ci_b")),
        "ci_ab": _pair(get(fit, "ci_ab")),
        "ci_interaction": _pair(get(fit, "ci_interaction")),
        "off_axis_residual": _f(get(fit, "off_axis_residual")),
        "neomorphic_ratio": _f(get(fit, "neomorphic_ratio")),
        "effect_space": str(get(fit, "effect_space", default="log-fold-change")),
    }


def _caption(d: dict[str, Any]) -> str:
    call = d["call"].upper()
    if is_abstention(d["call"]):
        return f"{d['label']} → {call} ({d['reason'] or 'interaction not resolved'})"
    inter = d["interaction"]
    lo, hi = d["ci_interaction"]
    return f"{d['label']} → {call} (interaction={inter:+.2f}, CI [{lo:+.2f}, {hi:+.2f}])"


def _err(val: float, ci: list[float]) -> np.ndarray:
    lo, hi = ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.array([[0.0], [0.0]])
    return np.array([[max(val - lo, 0.0)], [max(hi - val, 0.0)]])


def _draw_bars(ax: Any, d: dict[str, Any], pal: dict[str, str], color: str) -> None:
    labels = ["A", "B", "A+B\n(additive)", "A+B\n(observed)"]
    vals = [d["effect_a"], d["effect_b"], d["additive_pred"], d["effect_ab"]]
    colors = [pal["muted"], pal["muted"], pal["muted"], color]
    xs = np.arange(len(labels))
    ax.bar(xs, vals, color=colors, alpha=0.9, zorder=3, width=0.62)
    err_bars: list[tuple[int, float, list[float]]] = [
        (0, d["effect_a"], d["ci_a"]),
        (1, d["effect_b"], d["ci_b"]),
        (3, d["effect_ab"], d["ci_ab"]),
    ]
    for i, v, ci in err_bars:
        ax.errorbar(i, v, yerr=_err(v, ci), fmt="none", ecolor=pal["text"],
                    elinewidth=1.1, capsize=3, zorder=4)
    # The additive-null level as a reference line across the observed bar.
    ax.axhline(0.0, color=pal["grid"], lw=1.0, zorder=1)
    if np.isfinite(d["additive_pred"]):
        ax.plot([2.6, 3.4], [d["additive_pred"]] * 2, ls="--", color=pal["muted"],
                lw=1.4, zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(f"effect ({d['effect_space']})")
    ax.set_title("additive null vs observed", fontweight="bold", color=pal["text"])


def _draw_interaction(ax: Any, d: dict[str, Any], pal: dict[str, str], color: str) -> None:
    inter = d["interaction"]
    lo, hi = d["ci_interaction"]
    ax.axhline(0.0, color=pal["text"], lw=1.2, ls="-", zorder=2)
    ax.annotate("additive null", xy=(0.0, 0.0), xytext=(0.0, 0.0),
                xycoords=("axes fraction", "data"), textcoords="offset points",
                fontsize=8, color=pal["muted"], ha="left", va="bottom")
    if np.isfinite(inter):
        ax.errorbar([0.0], [inter], yerr=_err(inter, [lo, hi]), fmt="o", color=color,
                    ecolor=color, elinewidth=2.0, capsize=6, markersize=10, zorder=4,
                    label="interaction (CI)")
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_ylabel("A+B − additive prediction")
    ax.set_title("interaction estimate", fontweight="bold", color=pal["text"])
    # Off-axis residual — the standing honesty caveat, never a hidden-node claim.
    resid = d["off_axis_residual"]
    if np.isfinite(resid):
        ax.text(0.5, 0.02, f"off-axis residual = {resid:.2f}\n(consistent with — not proof of"
                " — a hidden regulator)", transform=ax.transAxes, fontsize=7.5,
                color=pal["muted"], ha="center", va="bottom", zorder=6)
    ax.legend(loc=freest_corner(ax, avoid_top=True), fontsize=8, framealpha=0.9)


def build(obj: Any, *, theme: str = "auto") -> RenderedFigure:
    """Build the epistasis figure (no overlay — the pipeline stamps it off the call)."""
    import matplotlib.pyplot as plt

    pal = apply_theme(theme)
    d = epistasis_data(obj)
    color = verdict_color(d["call"], pal)
    fig, (ax_bar, ax_int) = plt.subplots(1, 2, figsize=(10.4, 4.4))
    _draw_bars(ax_bar, d, pal, color)
    _draw_interaction(ax_int, d, pal, color)
    fig.suptitle(f"{d['label']}  →  {d['call'].upper()}", fontweight="bold",
                 color=pal["text"], fontsize=13)
    fig.tight_layout()
    # The interaction panel carries the verdict → the overlay fires there on abstention.
    panels = [
        Panel(ax=ax_bar, call="", reason="", label=d["label"]),
        Panel(ax=ax_int, call=d["call"], reason=d["reason"], label=d["label"]),
    ]
    return RenderedFigure(
        fig=fig, panels=panels, kind="epistasis", caption=_caption(d), data=d
    )
