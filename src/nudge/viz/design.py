"""Inverse-design renderer (``NUDGE-METHOD-007``; the flagship prescription).

Turn a reliable diagnosis into a proposed, untested intervention — behind an integrity
gate and a bifurcation **safety** gate. This renders the proposal + the safety dial.

Two panels: (left) the **proposed knob changes** (the ranked Δ factors, log axis, or a
single dose for curve-mode); and (right) the **safety dial** — the switch's fold proximity
BEFORE vs AFTER the intervention, with a loud HIGH-RISK marker when the plan crosses the
fold (destabilises the current basin). A design that is *unreachable* / gated returns an
abstention, and the whole figure carries the abstention overlay. A risky-but-decided plan
is NOT an abstention — it is a confident "this works but is dangerous", drawn with the risk
banner rather than the can't-tell hatch.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nudge.viz._util import get
from nudge.viz.base import Panel, RenderedFigure
from nudge.viz.theme import apply_theme


def _f(x: Any, d: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def design_data(obj: Any) -> dict[str, Any]:
    """Normalise a design result (``design_to_dict`` / ``design_circuit`` / replay)."""
    if isinstance(obj, dict) and obj.get("kind") == "design":
        return obj
    kind = str(get(obj, "kind", default="intervention"))
    is_abst = kind == "abstention"
    deltas_raw = get(obj, "deltas", default=[]) or []
    deltas = [
        {
            "name": str(get(get(dd, "param", default={}), "name", default="knob")
                        if isinstance(dd, dict) else get(dd, "name", default="knob")),
            "factor": _f(dd.get("factor") if isinstance(dd, dict) else get(dd, "factor")),
        }
        for dd in deltas_raw
    ]
    safety = get(obj, "safety", default=None)
    safety_out = None
    if isinstance(safety, dict):
        safety_out = {
            "proximity_before": _f(safety.get("proximity_before")),
            "proximity_after": _f(safety.get("proximity_after")),
            "crosses_fold": bool(safety.get("crosses_fold", False)),
            "high_risk": bool(safety.get("high_risk_of_instability", False)),
            "one_sided": bool(safety.get("one_sided", False)),
        }
    return {
        "kind": "design",
        "label": str(get(obj, "label", default="intervention")),
        "design_kind": kind,
        "mode": str(get(obj, "mode", default="")),
        "reason": str(get(obj, "reason", default="")),
        "verdict": str(get(obj, "verdict", default="")),
        "call": "abstain" if is_abst else "",
        "deltas": deltas,
        "dose": _f(get(obj, "dose")),
        "predicted_response": _f(get(obj, "predicted_response")),
        "safety": safety_out,
    }


def _caption(d: dict[str, Any]) -> str:
    if d["design_kind"] == "abstention":
        return f"{d['label']} → ABSTAIN ({d['verdict'] or d['reason'][:80]})"
    s = d["safety"]
    risk = " — HIGH RISK (crosses fold)" if s and s["high_risk"] else ""
    if d["mode"] == "dose":
        return f"{d['label']} → dose={d['dose']:.3g} (predicted={d['predicted_response']:.3g})"
    knobs = ", ".join(f"{x['name']}×{x['factor']:.2f}" for x in d["deltas"][:3])
    return f"{d['label']} → intervention [{knobs}]{risk}"


def _draw_plan(ax: Any, d: dict[str, Any], pal: dict[str, str]) -> None:
    if d["design_kind"] == "abstention":
        ax.set_axis_off()
        ax.text(0.5, 0.55, "no reachable intervention", transform=ax.transAxes,
                ha="center", fontsize=13, fontweight="bold", color=pal["abstain"])
        ax.text(0.5, 0.4, d["reason"][:110], transform=ax.transAxes, ha="center",
                fontsize=8.5, color=pal["muted"])
        return
    if d["mode"] == "dose":
        ax.bar([0], [d["dose"]], width=0.5, color=pal["ceiling"], alpha=0.9, zorder=3)
        ax.set_xticks([0])
        ax.set_xticklabels(["proposed dose"], fontsize=9)
        ax.set_ylabel("dose")
        ax.set_title("proposed intervention", fontweight="bold", color=pal["text"])
        return
    deltas = d["deltas"]
    names = [x["name"] for x in deltas]
    factors = [x["factor"] for x in deltas]
    xs = np.arange(len(deltas))
    log2f = [np.log2(f) if f and f > 0 else 0.0 for f in factors]
    colors = [pal["threshold"] if lf < 0 else pal["gain"] for lf in log2f]
    ax.bar(xs, log2f, color=colors, alpha=0.9, zorder=3, width=0.55)
    ax.axhline(0.0, color=pal["text"], lw=1.1, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("log2 fold change in knob")
    ax.set_title("proposed knob changes", fontweight="bold", color=pal["text"])


def _draw_safety(ax: Any, d: dict[str, Any], pal: dict[str, str]) -> None:
    s = d["safety"]
    if s is None:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no bifurcation safety gate\n(curve mode: no circuit/fold)",
                transform=ax.transAxes, ha="center", va="center", fontsize=10,
                color=pal["muted"])
        return
    before, after = s["proximity_before"], s["proximity_after"]
    risky = s["high_risk"] or s["crosses_fold"]
    after_color = pal["no_effect"] if risky else pal["ceiling"]
    ax.bar([0], [before], color=pal["muted"], alpha=0.9, width=0.5, zorder=3)
    if np.isfinite(after):
        ax.bar([1], [after], color=after_color, alpha=0.9, width=0.5, zorder=3)
    elif s["crosses_fold"]:
        # Crossing the fold destabilises the current basin → proximity past it is
        # undefined. Draw an OPEN-ENDED arrow past the fold, never a fake bar height.
        ax.annotate("", xy=(1.0, 1.12), xytext=(1.0, 0.35),
                    xycoords=("data", "axes fraction"),
                    arrowprops={"arrowstyle": "-|>", "color": after_color, "lw": 2.2})
        ax.text(1.0, 0.30, "past fold\n(basin destabilised)", ha="center", va="top",
                fontsize=7.5, color=after_color, transform=ax.get_xaxis_transform())
    ax.axhline(1.0, ls="--", color=pal["gain"], lw=1.6, zorder=2)
    ax.text(1.02, 1.0, "fold (bifurcation)", transform=ax.get_yaxis_transform(),
            fontsize=7.5, color=pal["gain"], va="center", ha="left")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["before", "after"], fontsize=9)
    ax.set_ylabel("fold proximity  (→1 = at the fold)")
    tops = [1.2] + [v * 1.1 for v in (before, after) if np.isfinite(v)]
    ax.set_ylim(0.0, max(tops))
    ax.set_title("safety dial", fontweight="bold", color=pal["text"])
    if risky:
        ax.text(0.5, 0.92, "HIGH RISK — crosses the fold", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=pal["no_effect"], zorder=7,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": pal["surface"],
                      "alpha": 0.9, "edgecolor": pal["no_effect"]})
    if s["one_sided"]:
        ax.text(0.5, 0.02, "proximity is a one-sided lower bound near the fold "
                "(NUDGE-LIM-012)", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.5, color=pal["muted"])


def build(obj: Any, *, theme: str = "auto") -> RenderedFigure:
    """Build the inverse-design figure (no overlay — the pipeline stamps it off the call)."""
    import matplotlib.pyplot as plt

    pal = apply_theme(theme)
    d = design_data(obj)
    fig, (ax_plan, ax_safe) = plt.subplots(1, 2, figsize=(10.2, 4.4))
    _draw_plan(ax_plan, d, pal)
    _draw_safety(ax_safe, d, pal)
    # A meaningful suptitle: say what the figure IS (a proposed intervention, or a target
    # that could not be reached → ABSTAIN) rather than echoing the default label into the
    # verdict (the old "intervention → INTERVENTION" / "intervention → ABSTAIN" smell).
    generic = d["label"].strip().lower() in ("", "intervention", "design", "flip")
    head = (
        "target unreachable  →  ABSTAIN"
        if d["design_kind"] == "abstention"
        else "proposed intervention"
    )
    title = head if generic else f"{d['label']}  —  {head}"
    fig.suptitle(title, fontweight="bold", color=pal["text"], fontsize=13)
    fig.tight_layout()
    # The plan panel carries the verdict → the overlay fires there on an abstaining design.
    panels = [
        Panel(ax=ax_plan, call=d["call"], reason=d["reason"], label=d["label"]),
        Panel(ax=ax_safe, call="", reason="", label=d["label"]),
    ]
    return RenderedFigure(
        fig=fig, panels=panels, kind="design", caption=_caption(d), data=d
    )
