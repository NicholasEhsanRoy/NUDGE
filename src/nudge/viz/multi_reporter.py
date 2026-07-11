"""Multi-reporter joint attribution renderer (``NUDGE-METHOD-008``).

Several downstream reporters of ONE latent switch, fit jointly — the panel over-determines
the latent, so the joint fit RESOLVES threshold/gain/ceiling where a single reporter is
degenerate (the identifiability force-multiplier). A panel that no single shared latent can
explain abstains ``off-model``.

Two panels: (left) the **restricted-loss** of each single-knob hypothesis (free K / free n
/ free A) against the no-effect and full-freedom brackets — the winner (the resolved knob)
is highlighted, and the gap to ``no-effect`` (there IS an effect) and to the runner-up (the
knob is identifiable) is what a single reporter cannot see; and (right) each reporter's fit
quality under the **shared latent vs its own independent fit** — a shared latent that
explains every reporter is why the joint call resolves. The loss panel is the verdict panel
(hatched on ``no-effect`` / ``unresolved`` / ``off-model``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nudge.viz._util import get, verdict_color
from nudge.viz.base import Panel, RenderedFigure, is_abstention
from nudge.viz.theme import apply_theme


def _f(x: Any, d: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def multi_reporter_data(obj: Any) -> dict[str, Any]:
    """Normalise a multi-reporter result (dataclass / dict / replay) to a figure dict."""
    if isinstance(obj, dict) and obj.get("kind") == "multi_reporter":
        return obj
    fit = get(obj, "fit", default=obj)
    losses = get(fit, "losses", default=None)
    if losses is None:  # dataclass form: individual loss_* fields
        losses = {
            "no_effect": get(fit, "loss_no_effect"),
            "threshold": get(fit, "loss_threshold"),
            "gain": get(fit, "loss_gain"),
            "ceiling": get(fit, "loss_ceiling"),
            "full": get(fit, "loss_full"),
        }
    reporters = get(fit, "reporters", default=[]) or []
    rep_out = [
        {
            "name": str(get(r, "name", default=f"R{i}")),
            "r2_shared": _f(get(r, "r2_shared")),
            "r2_independent": _f(get(r, "r2_independent")),
        }
        for i, r in enumerate(reporters)
    ]
    return {
        "kind": "multi_reporter",
        "label": str(get(obj, "label", default="joint panel")),
        "call": str(get(obj, "call", default="unresolved")),
        "reason": str(get(obj, "reason", default="")),
        "winner": str(get(fit, "winner", default="")),
        "knob_margin": _f(get(fit, "knob_margin")),
        "effect_margin": _f(get(fit, "effect_margin")),
        "n_reporters": int(_f(get(fit, "n_reporters"), len(rep_out)) or len(rep_out)),
        "losses": {k: _f(v) for k, v in dict(losses).items()},
        "reporters": rep_out,
    }


def _caption(d: dict[str, Any]) -> str:
    call = d["call"].upper()
    if is_abstention(d["call"]):
        return f"{d['label']} → {call} ({d['reason'][:80] or 'not jointly resolvable'})"
    return (f"{d['label']} → {call} (winner={d['winner']}, "
            f"knob margin={d['knob_margin']:.2f}, {d['n_reporters']} reporters)")


def _draw_losses(ax: Any, d: dict[str, Any], pal: dict[str, str], color: str) -> None:
    order = ["no_effect", "threshold", "gain", "ceiling", "full"]
    labels = ["no-effect", "threshold\n(K)", "gain\n(n)", "ceiling\n(A)", "full\n(all free)"]
    losses = d["losses"]
    vals = [losses.get(k, float("nan")) for k in order]
    winner = d["winner"].lower()
    colors = []
    for k in order:
        if k == winner and not is_abstention(d["call"]):
            colors.append(color)
        elif k in ("no_effect", "full"):
            colors.append(pal["grid"])
        else:
            colors.append(pal["muted"])
    xs = np.arange(len(order))
    ax.bar(xs, vals, color=colors, alpha=0.95, zorder=3, width=0.62)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("restricted loss  (lower = better fit)")
    ax.set_title("which single knob explains the panel?", fontweight="bold",
                 color=pal["text"])


def _draw_reporters(ax: Any, d: dict[str, Any], pal: dict[str, str], color: str) -> None:
    reps = d["reporters"]
    if not reps:
        ax.text(0.5, 0.5, "per-reporter R²\nunavailable", transform=ax.transAxes,
                ha="center", va="center", color=pal["muted"], fontsize=10)
        ax.set_xticks([])
        ax.set_title("shared vs independent fit", fontweight="bold", color=pal["text"])
        return
    names = [r["name"] for r in reps]
    xs = np.arange(len(reps))
    w = 0.38
    ax.bar(xs - w / 2, [r["r2_shared"] for r in reps], width=w, color=color, alpha=0.9,
           zorder=3, label="shared latent")
    ax.bar(xs + w / 2, [r["r2_independent"] for r in reps], width=w, color=pal["muted"],
           alpha=0.8, zorder=3, label="independent")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("R²")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("shared latent explains each reporter", fontweight="bold",
                 color=pal["text"])
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


def build(obj: Any, *, theme: str = "auto") -> RenderedFigure:
    """Build the multi-reporter figure (no overlay — the pipeline stamps it off the call)."""
    import matplotlib.pyplot as plt

    pal = apply_theme(theme)
    d = multi_reporter_data(obj)
    color = verdict_color(d["call"], pal)
    fig, (ax_loss, ax_rep) = plt.subplots(1, 2, figsize=(10.6, 4.4))
    _draw_losses(ax_loss, d, pal, color)
    _draw_reporters(ax_rep, d, pal, color)
    fig.suptitle(f"{d['label']}  →  {d['call'].upper()}", fontweight="bold",
                 color=pal["text"], fontsize=13)
    fig.tight_layout()
    panels = [
        Panel(ax=ax_loss, call=d["call"], reason=d["reason"], label=d["label"]),
        Panel(ax=ax_rep, call="", reason="", label=d["label"]),
    ]
    return RenderedFigure(
        fig=fig, panels=panels, kind="multi_reporter", caption=_caption(d), data=d
    )
