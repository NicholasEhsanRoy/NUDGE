"""Bifurcation-proximity "robustness dial" renderer (``NUDGE-METHOD-006``, ``NUDGE-LIM-012``).

How close is a bistable switch to its fold (the tipping point past which it collapses to the
other basin)? The fused **0–1 proximity dial** plus the three raw channels that feed it —
critical slowing (the slowest eigenvalue → 0), basin collapse (the node–saddle distance →
0), and LNA lobe swell (the noise lobes overlapping). Near the fold the number is a
**one-sided lower bound** — the LNA Gaussian breaks down there, so NUDGE can only say
"at LEAST this close", drawn as an open-ended arrow, never a point estimate (``NUDGE-LIM-012``).

Two panels: (left) the proximity dial (0 = deep, safe basin → 1 = at the fold) with the
one-sided arrow when applicable; and (right) the three channels. Verdict panel = the dial;
hatched on ``not-bistable`` / ``unresolved``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nudge.viz._util import get, verdict_color
from nudge.viz.base import Panel, RenderedFigure, is_abstention
from nudge.viz.theme import apply_theme

_CHANNEL_LABELS = {
    "critical_slowing": "critical\nslowing",
    "basin_collapse": "basin\ncollapse",
    "lobe_overlap": "LNA lobe\nswell",
}


def _f(x: Any, d: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def robustness_data(obj: Any) -> dict[str, Any]:
    """Normalise a bifurcation result (``robustness_circuit`` / ``bifurcation_to_dict``)."""
    if isinstance(obj, dict) and obj.get("kind") == "robustness":
        return obj
    score = get(obj, "score", default=None)
    src = score if score is not None else obj
    channels = get(obj, "channels", default=None)
    if channels is None:
        channels = get(src, "channels", default={})
    # The three fold channels live under ``channel_proximities`` in the raw score dict.
    if isinstance(channels, dict) and "channel_proximities" in channels:
        channels = channels["channel_proximities"]
    return {
        "kind": "robustness",
        "label": str(get(obj, "label", "topology", default="switch")),
        "call": str(get(obj, "call", default="unresolved")),
        "reason": str(get(obj, "reason", default="")),
        "proximity": _f(get(obj, "proximity", default=get(src, "proximity"))),
        "one_sided": bool(get(obj, "one_sided", default=get(src, "one_sided",
                                                            default=False))),
        "channels": {k: _f(v) for k, v in dict(channels).items()},
    }


def _caption(d: dict[str, Any]) -> str:
    call = d["call"].upper()
    prox = d["proximity"]
    if is_abstention(d["call"]):
        return f"{d['label']} → {call} ({d['reason'][:80] or 'proximity not resolved'})"
    bound = "≥ " if d["one_sided"] else ""
    tag = " (one-sided lower bound, NUDGE-LIM-012)" if d["one_sided"] else ""
    return f"{d['label']} → {call} (fold proximity {bound}{prox:.2f}{tag})"


def _draw_dial(ax: Any, d: dict[str, Any], pal: dict[str, str], color: str) -> None:
    prox = d["proximity"]
    ax.barh([0], [1.0], height=0.5, color=pal["grid"], zorder=1)  # the full 0..1 track
    if np.isfinite(prox):
        ax.barh([0], [prox], height=0.5, color=color, alpha=0.9, zorder=3)
        if d["one_sided"]:
            # LNA breaks down near the fold → an OPEN-ENDED arrow toward the fold, not a
            # point estimate. "At least this close."
            ax.annotate("", xy=(min(prox + 0.28, 1.0), 0), xytext=(prox, 0),
                        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.4})
            ax.text(prox, 0.42, f"≥ {prox:.2f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=color)
        else:
            ax.text(prox, 0.42, f"{prox:.2f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=color)
    ax.axvline(1.0, ls="--", color=pal["gain"], lw=1.8, zorder=4)
    ax.text(1.0, -0.52, "fold\n(tipping point)", ha="center", va="top", fontsize=8,
            color=pal["gain"])
    ax.text(0.0, -0.52, "deep basin\n(safe)", ha="center", va="top", fontsize=8,
            color=pal["muted"])
    ax.set_xlim(-0.02, 1.14)
    ax.set_ylim(-0.9, 0.9)
    ax.set_yticks([])
    ax.set_xlabel("fold proximity  (0 = deep · 1 = at the fold)")
    ax.set_title("robustness dial", fontweight="bold", color=pal["text"])
    if d["one_sided"]:
        ax.text(0.5, 0.02, "one-sided lower bound — LNA breaks down near the fold "
                "(NUDGE-LIM-012)", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.5, color=pal["muted"], zorder=6)


def _draw_channels(ax: Any, d: dict[str, Any], pal: dict[str, str], color: str) -> None:
    order = [k for k in ("critical_slowing", "basin_collapse", "lobe_overlap")
             if k in d["channels"]]
    if not order:
        ax.text(0.5, 0.5, "channels unavailable", transform=ax.transAxes, ha="center",
                va="center", color=pal["muted"])
        ax.set_title("channels", fontweight="bold", color=pal["text"])
        return
    xs = np.arange(len(order))
    vals = [d["channels"][k] for k in order]
    bar_color = pal["abstain"] if is_abstention(d["call"]) else color
    ax.bar(xs, vals, color=bar_color, alpha=0.9, zorder=3, width=0.6)
    ax.axhline(1.0, ls="--", color=pal["gain"], lw=1.4, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([_CHANNEL_LABELS.get(k, k) for k in order], fontsize=8)
    ax.set_ylabel("proximity channel  (→1 = at the fold)")
    ax.set_ylim(0.0, 1.12)
    ax.set_title("three channels approaching the fold", fontweight="bold",
                 color=pal["text"])


def build(obj: Any, *, theme: str = "auto") -> RenderedFigure:
    """Build the robustness-dial figure (no overlay — the pipeline stamps it off call)."""
    import matplotlib.pyplot as plt

    pal = apply_theme(theme)
    d = robustness_data(obj)
    color = verdict_color(d["call"], pal)
    fig, (ax_dial, ax_ch) = plt.subplots(1, 2, figsize=(10.6, 4.2),
                                         gridspec_kw={"width_ratios": [1.5, 1.0]})
    _draw_dial(ax_dial, d, pal, color)
    _draw_channels(ax_ch, d, pal, color)
    fig.suptitle(f"{d['label']}  →  {d['call'].upper()}", fontweight="bold",
                 color=pal["text"], fontsize=13)
    fig.tight_layout()
    panels = [
        Panel(ax=ax_dial, call=d["call"], reason=d["reason"], label=d["label"],
              one_sided=d["one_sided"]),
        Panel(ax=ax_ch, call="", reason="", label=d["label"]),
    ]
    return RenderedFigure(
        fig=fig, panels=panels, kind="robustness", caption=_caption(d), data=d
    )
