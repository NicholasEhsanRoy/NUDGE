"""Hidden-node diagnosis renderer (``NUDGE-METHOD-009``; abstention packaging).

When the parsimony gate returns ``off-model``, NUDGE never asserts a hidden node — it
returns a *differential* of candidate causes (unmeasured regulator, off-target, readout
nonlinearity, wrong topology, depth confound, …), each with its documented limitation and
distinguishing experiment (``NUDGE-LIM-015``). This renders that differential as a ranked
card, and — because an off-model result IS an abstention — the whole figure carries the
abstention overlay, so it can never read as a positive hidden-node claim.

One panel: a horizontal ranked list of candidate causes (leading → less-likely), each
tagged with its limitation reference. Verdict panel = the card; hatched whenever the model
is inadequate.
"""

from __future__ import annotations

from typing import Any

from nudge.viz._util import get
from nudge.viz.base import ABSTAIN_CALLS, Panel, RenderedFigure
from nudge.viz.theme import apply_theme

_RANK_ORDER = {"leading": 0, "plausible": 1, "less-likely": 2}
_RANK_LEN = {"leading": 1.0, "plausible": 0.66, "less-likely": 0.4}


def diagnose_data(obj: Any) -> dict[str, Any]:
    """Normalise an ``InadequacyReport`` / ``inadequacy_to_dict`` / replay to a dict."""
    if isinstance(obj, dict) and obj.get("kind") == "diagnose":
        return obj
    raw_causes: list[Any] = get(obj, "causes", default=None)
    if raw_causes is None:
        ranked = get(obj, "ranked_causes", default=None)  # dataclass method
        raw_causes = list(ranked()) if callable(ranked) else []
    causes = [
        {
            "name": str(get(c, "name", default="")),
            "qualitative_rank": str(get(c, "qualitative_rank", default="plausible")),
            "limitation_ref": str(get(c, "limitation_ref", default="")),
            "limitation_title": str(get(c, "limitation_title", default="")),
        }
        for c in raw_causes
    ]
    causes.sort(key=lambda c: _RANK_ORDER.get(c["qualitative_rank"], 99))
    is_adequate = bool(get(obj, "is_adequate", default=False))
    verdict = str(get(obj, "verdict", default="off-model"))
    # An inadequate model is an abstention (never a positive hidden-node claim).
    call = "" if is_adequate else (verdict if verdict in ABSTAIN_CALLS else "off-model")
    return {
        "kind": "diagnose",
        "label": str(get(obj, "label", default="attribution")),
        "is_adequate": is_adequate,
        "verdict": verdict,
        "call": call,
        "reason": str(get(obj, "reason", default="")),
        "causes": causes,
    }


def _caption(d: dict[str, Any]) -> str:
    if d["is_adequate"]:
        return f"{d['label']} → model ADEQUATE (no hidden-node differential needed)"
    names = ", ".join(c["name"] for c in d["causes"][:3])
    return (f"{d['label']} → {d['verdict'].upper()} — differential (NOT a verdict): "
            f"{names or 'candidate causes'}")


def build(obj: Any, *, theme: str = "auto") -> RenderedFigure:
    """Build the diagnosis card (no overlay — the pipeline stamps it off the call)."""
    import matplotlib.pyplot as plt

    pal = apply_theme(theme)
    d = diagnose_data(obj)
    causes = d["causes"]
    height = 5.6 if len(causes) > 4 else 4.6
    fig, ax = plt.subplots(figsize=(8.8, height))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    if d["is_adequate"] or not causes:
        ax.text(0.5, 0.6, "model ADEQUATE", transform=ax.transAxes, ha="center",
                fontsize=15, fontweight="bold", color=pal["ceiling"])
        ax.text(0.5, 0.45, d["reason"][:120], transform=ax.transAxes, ha="center",
                fontsize=9, color=pal["muted"])
        ax.set_ylim(0, 1)
    else:
        n = len(causes)
        # Extra top headroom so the abstain banner sits in a clear band above every bar.
        ax.set_ylim(-1.0, n + 0.9)
        for i, c in enumerate(causes):
            y = n - 1 - i
            length = _RANK_LEN.get(c["qualitative_rank"], 0.5)
            ax.barh(y, length, height=0.5, left=0.04, color=pal["abstain"], alpha=0.85,
                    zorder=3)
            ax.text(0.06, y, f"{i + 1}. {c['name']}  ({c['qualitative_rank']})",
                    va="center", ha="left", fontsize=9.5, fontweight="bold",
                    color=pal["text"], zorder=6)
            ref = c["limitation_ref"] or c["limitation_title"]
            if ref:
                ax.text(0.06, y - 0.30, ref, va="center", ha="left", fontsize=7.5,
                        color=pal["muted"], zorder=6)
        # Honesty note at the BOTTOM (the top band is reserved for the abstain banner).
        ax.text(0.5, 0.01, "differential of candidate causes — NOT a positive "
                "hidden-node claim", transform=ax.transAxes, ha="center", fontsize=8.5,
                color=pal["muted"], zorder=6)
    ax.set_title(f"{d['label']}  →  "
                 f"{'ADEQUATE' if d['is_adequate'] else d['verdict'].upper()}",
                 fontweight="bold", color=pal["text"], fontsize=13)
    fig.tight_layout()
    panels = [Panel(ax=ax, call=d["call"], reason=d["reason"], label=d["label"])]
    return RenderedFigure(
        fig=fig, panels=panels, kind="diagnose", caption=_caption(d), data=d
    )
