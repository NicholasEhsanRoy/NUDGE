"""Core attribution renderer — the ``AttributionReport`` across operating points.

This is the capstone pipeline's picture (``nudge.inference.pipeline.AttributionReport``):
a target perturbation attributed across one or more operating points. A **single**
operating point can only ever return an abstention-class call (``gain_or_threshold`` — the
measured gain/threshold confound — or ``ceiling`` / ``unresolved``, never a bare
gain/threshold); the **joint** multi-operating-point fit is the *breaker* that can resolve
a bare ``gain`` / ``threshold`` / ``ceiling`` when the operating points are well-buffered.
Operating points that were unusable (too few cells / an untrustworthy LNA) are drawn as
first-class **skipped** cells, never hidden.

Two panels: (left) the **per-operating-point verdict chips** — one row per operating
point, coloured by its call, width = target-condition cell count, with skipped points as
hatched grey cells carrying their skip reason; and (right) the **restricted-NLL profile**
of the headline fit (the joint breaker if it ran, else the single point) — the gain /
threshold / ceiling hypotheses as ΔNLL bars, the winner highlighted, hatched by the
honesty overlay when the headline call abstains. Reads an ``AttributionReport`` dataclass,
the ``service.report_to_dict`` dict, or the canonical replay dict.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nudge.viz._util import get, verdict_color
from nudge.viz.base import Panel, RenderedFigure, is_abstention
from nudge.viz.theme import apply_theme

# nlls hypothesis keys are ``n``/``K``/``vmax`` from a single-condition fit and
# ``gain``/``threshold``/``ceiling`` from the joint fit — canonicalise to the mechanism.
_MECH_OF_KEY = {
    "n": "gain", "gain": "gain",
    "K": "threshold", "threshold": "threshold",
    "vmax": "ceiling", "ceiling": "ceiling",
}
_MECH_LABEL = {"gain": "gain\n(n)", "threshold": "threshold\n(K)", "ceiling": "ceiling\n(v_max)"}
_MECH_ORDER = ("gain", "threshold", "ceiling")


def _f(x: Any, d: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def _canon_nlls(nlls: Any) -> dict[str, float]:
    """Canonicalise a restricted-NLL dict to mechanism keys (gain/threshold/ceiling)."""
    out: dict[str, float] = {}
    for k, v in dict(nlls or {}).items():
        mech = _MECH_OF_KEY.get(str(k))
        if mech is not None:
            out[mech] = _f(v)
    return out


def _single_items(single: Any) -> list[tuple[str, str, dict[str, float]]]:
    """Read ``single`` in either shape → ``[(op_label, call, nlls)]``.

    The dataclass carries ``{label: (call, nlls)}``; ``report_to_dict`` carries
    ``{label: {"call": ..., "nlls": ...}}``.
    """
    items: list[tuple[str, str, dict[str, float]]] = []
    for label, val in dict(single or {}).items():
        if isinstance(val, dict):
            call, nlls = str(val.get("call", "")), val.get("nlls", {})
        else:  # (call, nlls) tuple
            call, nlls = str(val[0]), val[1]
        items.append((str(label), call, _canon_nlls(nlls)))
    return items


def attribution_data(obj: Any) -> dict[str, Any]:
    """Normalise an ``AttributionReport`` (dataclass / ``report_to_dict`` / replay) to a dict."""
    if isinstance(obj, dict) and obj.get("kind") == "attribution":
        return obj
    target = str(get(obj, "target", "label", default="target"))
    n_cells = dict(get(obj, "n_cells", default={}) or {})
    singles = _single_items(get(obj, "single", default={}))
    skipped_raw = dict(get(obj, "skipped", default={}) or {})

    multi = get(obj, "multi", default=None)
    joint: dict[str, Any] | None = None
    if isinstance(multi, dict):
        joint = {"call": str(multi.get("call", "")), "nlls": _canon_nlls(multi.get("nlls"))}
    elif isinstance(multi, (list, tuple)) and len(multi) == 2:
        joint = {"call": str(multi[0]), "nlls": _canon_nlls(multi[1])}

    ops = [
        {"label": lbl, "call": call, "n_cells": int(n_cells.get(lbl, 0)), "nlls": nlls}
        for lbl, call, nlls in singles
    ]
    skipped = [
        {"label": lbl, "reason": str(reason), "n_cells": int(n_cells.get(lbl, 0))}
        for lbl, reason in skipped_raw.items()
    ]

    # The headline verdict that governs the honesty overlay: the joint breaker if it ran,
    # else the lone operating point's call, else unresolved (no usable point).
    if joint is not None:
        headline = joint["call"]
    elif len(ops) == 1:
        headline = ops[0]["call"]
    else:
        headline = "unresolved"

    return {
        "kind": "attribution",
        "label": target,
        "call": headline,
        "reason": _headline_reason(joint, ops, skipped),
        "ops": ops,
        "skipped": skipped,
        "joint": joint,
    }


def _headline_reason(joint: Any, ops: list[Any], skipped: list[Any]) -> str:
    if joint is not None:
        if is_abstention(joint["call"]):
            return "the joint breaker could not resolve a single knob across the operating points"
        return "the joint multi-operating-point fit resolved a single knob"
    if len(ops) == 1:
        return "a single operating point cannot separate gain from threshold (measured confound)"
    if not ops:
        return f"no usable operating point ({len(skipped)} skipped)"
    return "fewer than two usable operating points — the joint breaker did not run"


def _headline_nlls(d: dict[str, Any]) -> tuple[dict[str, float], str]:
    """The restricted NLLs to draw (joint breaker if present, else the lone op) + a tag."""
    if d["joint"] is not None and d["joint"]["nlls"]:
        return d["joint"]["nlls"], "joint (breaker)"
    if len(d["ops"]) == 1 and d["ops"][0]["nlls"]:
        return d["ops"][0]["nlls"], f"single op: {d['ops'][0]['label']}"
    return {}, ""


def _caption(d: dict[str, Any]) -> str:
    call = d["call"].upper()
    n_ok, n_skip = len(d["ops"]), len(d["skipped"])
    scope = f"{n_ok} op(s), {n_skip} skipped"
    if is_abstention(d["call"]):
        return f"{d['label']} → {call} ({scope}; {d['reason'][:70]})"
    return f"{d['label']} → {call} (resolved across {scope})"


def _draw_chips(ax: Any, d: dict[str, Any], pal: dict[str, str]) -> None:
    """One row per operating point: bar width = target-cell count, colour = its call.

    Skipped operating points are first-class hatched grey cells carrying the skip reason.
    """
    rows: list[tuple[str, float, str, str, bool]] = []  # (label, n_cells, call, note, skip)
    for op in d["ops"]:
        rows.append((op["label"], float(op["n_cells"]), op["call"],
                     op["call"].replace("_", " "), False))
    for sk in d["skipped"]:
        rows.append((sk["label"], float(sk["n_cells"]), "abstain",
                     f"skipped — {sk['reason']}", True))

    if not rows:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no operating points", transform=ax.transAxes, ha="center",
                va="center", fontsize=11, color=pal["muted"])
        return

    ys = np.arange(len(rows))[::-1]  # first op at the top
    widths = [max(w, 1.0) for _, w, _, _, _ in rows]
    for y, (_, w, call, note, skip) in zip(ys, rows, strict=True):
        color = pal["abstain"] if skip else verdict_color(call, pal)
        ax.barh(y, max(w, 1.0), color=color, alpha=0.85, zorder=3, height=0.62,
                hatch="////" if skip else None, edgecolor=color)
        ax.text(max(w, 1.0), y, f"  {note}  (n={int(w)})", va="center", ha="left",
                fontsize=8, color=pal["muted"] if skip else pal["text"], zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(0.0, max(widths) * 1.7)
    ax.set_xlabel("target-condition cells")
    ax.set_title("per-operating-point verdicts", fontweight="bold", color=pal["text"])


def _draw_nll(ax: Any, d: dict[str, Any], pal: dict[str, str], color: str) -> None:
    """ΔNLL of the gain / threshold / ceiling hypotheses for the headline (breaker) fit."""
    nlls, tag = _headline_nlls(d)
    mechs = [m for m in _MECH_ORDER if m in nlls]
    if not mechs:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no restricted-NLL profile\n(no usable operating point)",
                transform=ax.transAxes, ha="center", va="center", fontsize=10,
                color=pal["muted"])
        ax.set_title("restricted-NLL profile", fontweight="bold", color=pal["text"])
        return
    best = min(nlls[m] for m in mechs)
    xs = np.arange(len(mechs))
    deltas = [nlls[m] - best for m in mechs]  # 0 = the winning hypothesis
    resolved = not is_abstention(d["call"])
    winner = _MECH_OF_KEY.get(d["call"], d["call"]) if resolved else None
    colors = [color if (resolved and m == winner) else pal["muted"] for m in mechs]
    ax.bar(xs, deltas, color=colors, alpha=0.9, zorder=3, width=0.6)
    ax.axhline(0.0, color=pal["text"], lw=1.2, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([_MECH_LABEL[m] for m in mechs], fontsize=8)
    ax.set_ylabel("ΔNLL vs best  (lower = favoured)")
    ax.set_title(f"restricted-NLL profile — {tag}", fontweight="bold", color=pal["text"])


def build(obj: Any, *, theme: str = "auto") -> RenderedFigure:
    """Build the attribution figure (no overlay — the pipeline stamps it off the call)."""
    import matplotlib.pyplot as plt

    pal = apply_theme(theme)
    d = attribution_data(obj)
    color = verdict_color(d["call"], pal)
    fig, (ax_chips, ax_nll) = plt.subplots(1, 2, figsize=(10.6, 4.5),
                                           gridspec_kw={"width_ratios": [1.35, 1.0]})
    _draw_chips(ax_chips, d, pal)
    _draw_nll(ax_nll, d, pal, color)
    fig.suptitle(f"attribute {d['label']}  →  {d['call'].upper()}", fontweight="bold",
                 color=pal["text"], fontsize=13)
    fig.tight_layout()
    # The NLL panel carries the headline verdict → the overlay fires there on an abstention.
    panels = [
        Panel(ax=ax_chips, call="", reason="", label=d["label"]),
        Panel(ax=ax_nll, call=d["call"], reason=d["reason"], label=d["label"]),
    ]
    return RenderedFigure(
        fig=fig, panels=panels, kind="attribution", caption=_caption(d), data=d
    )
