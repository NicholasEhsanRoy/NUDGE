"""Amyloid fibrillization / aggregation renderer (``NUDGE-METHOD-013``).

One aggregation curve resolves the identifiable composites (κ, λ) but the three microscopic
rate constants (primary nucleation, elongation, secondary nucleation) are non-identifiable
up to an exact **gauge null** (``NUDGE-LIM-021``). The honest output is: the composites,
WITH their CIs — and the null-space direction along which the three constants trade off,
NOT three point estimates.

Panels: an optional (left) **aggregation curve** (mass fraction vs time, when supplied);
(centre) the **identified composites** κ and λ with bootstrap CIs; and (right) the **gauge
non-identifiability** — the null-space direction over (k_n, k_+, k_2) with the condition
number, i.e. the exact combination the single curve cannot separate. Verdict panel = the
gauge panel; hatched only when the fit is ``unresolved``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nudge.viz._util import get, verdict_color
from nudge.viz.base import Panel, RenderedFigure, is_abstention
from nudge.viz.theme import apply_theme

_CONST_LABELS = ["primary\n(k_n)", "elongation\n(k_+)", "secondary\n(k_2)"]


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


def aggregation_data(obj: Any) -> dict[str, Any]:
    """Normalise a fibrillization result (``fibrillization_demo`` / dict / replay)."""
    if isinstance(obj, dict) and obj.get("kind") == "aggregation":
        return obj
    nd = get(obj, "null_direction", default=None)
    ident = get(obj, "identifiability", default=None)
    if nd is None and ident is not None:
        nd = get(ident, "null_direction", default=None)
    cond = get(obj, "cond_number", default=None)
    if cond is None and ident is not None:
        cond = get(ident, "cond_number", default=None)
    return {
        "kind": "aggregation",
        "label": str(get(obj, "label", default="aggregation")),
        "call": str(get(obj, "call", default="unresolved")),
        "reason": str(get(obj, "reason", default="")),
        "kappa": _f(get(obj, "kappa")),
        "lam": _f(get(obj, "lambda", "lam")),
        "kappa_ci": _pair(get(obj, "kappa_ci")),
        "lambda_ci": _pair(get(obj, "lambda_ci")),
        "individual_k_identifiable": bool(get(obj, "individual_k_identifiable",
                                             default=False)),
        "cond_number": _f(cond),
        "null_direction": None if nd is None else [float(x) for x in nd],
        "curve": get(obj, "curve", default=None),
    }


def _caption(d: dict[str, Any]) -> str:
    call = d["call"].upper()
    if is_abstention(d["call"]):
        return f"{d['label']} → {call} ({d['reason'][:90] or 'kinetics unresolved'})"
    return (f"{d['label']} → {call} (κ={d['kappa']:.2g}, λ={d['lam']:.2g} identified; "
            "3 microscopic constants non-identifiable — gauge null)")


def _draw_curve(ax: Any, curve: dict[str, Any], pal: dict[str, str], color: str) -> None:
    t = np.asarray(curve["t"], dtype=float)
    m = np.asarray(curve["m"], dtype=float)
    ax.plot(t, m, color=color, lw=2.2, zorder=3)
    ax.set_xlabel("time")
    ax.set_ylabel("mass fraction aggregated")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("aggregation curve", fontweight="bold", color=pal["text"])


def _draw_composites(ax: Any, d: dict[str, Any], pal: dict[str, str], color: str) -> None:
    vals = [d["kappa"], d["lam"]]
    cis = [d["kappa_ci"], d["lambda_ci"]]
    xs = np.arange(2)
    for i, (v, ci) in enumerate(zip(vals, cis, strict=False)):
        if np.isfinite(v):
            lo, hi = ci
            yerr = np.array([[max(v - lo, 0.0)], [max(hi - v, 0.0)]]) \
                if np.isfinite(lo) else None
            ax.errorbar(i, v, yerr=yerr, fmt="o", color=color, ecolor=color,
                        elinewidth=2.0, capsize=6, markersize=11, zorder=4)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(["κ (rate)", "λ (lag)"], fontsize=9)
    ax.set_ylabel("value")
    ax.set_title("identified composites (κ, λ)", fontweight="bold", color=pal["text"])


def _draw_gauge(ax: Any, d: dict[str, Any], pal: dict[str, str], color: str) -> None:
    nd = d["null_direction"]
    xs = np.arange(3)
    if nd is not None and len(nd) >= 3:
        ax.bar(xs, nd[:3], color=pal["abstain"], alpha=0.9, zorder=3, width=0.6)
        ax.axhline(0.0, color=pal["text"], lw=1.0, zorder=2)
    else:
        ax.text(0.5, 0.5, "null direction\nunavailable", transform=ax.transAxes,
                ha="center", va="center", color=pal["muted"])
    ax.set_xticks(xs)
    ax.set_xticklabels(_CONST_LABELS, fontsize=8)
    ax.set_ylabel("gauge null-space direction")
    ax.set_title("microscopic constants: non-identifiable", fontweight="bold",
                 color=pal["text"])
    if np.isfinite(d["cond_number"]):
        ax.text(0.5, 0.02, f"cond # = {d['cond_number']:.1f}  (individual constants trade "
                "off along this direction)", transform=ax.transAxes, fontsize=7.5,
                color=pal["muted"], ha="center", va="bottom", zorder=6)


def build(obj: Any, *, theme: str = "auto") -> RenderedFigure:
    """Build the aggregation figure (no overlay — the pipeline stamps it off the call)."""
    import matplotlib.pyplot as plt

    pal = apply_theme(theme)
    d = aggregation_data(obj)
    color = verdict_color(d["call"], pal)
    curve = d["curve"]
    if curve is not None:
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
        _draw_curve(axes[0], curve, pal, color)
        ax_comp, ax_gauge = axes[1], axes[2]
        context = [axes[0], ax_comp]
    else:
        fig, (ax_comp, ax_gauge) = plt.subplots(1, 2, figsize=(9.8, 4.4))
        context = [ax_comp]
    _draw_composites(ax_comp, d, pal, color)
    _draw_gauge(ax_gauge, d, pal, color)
    fig.suptitle(f"{d['label']}  →  {d['call'].upper()}", fontweight="bold",
                 color=pal["text"], fontsize=13)
    fig.tight_layout()
    panels = [Panel(ax=a, call="", reason="", label=d["label"]) for a in context]
    panels.append(Panel(ax=ax_gauge, call=d["call"], reason=d["reason"], label=d["label"]))
    return RenderedFigure(
        fig=fig, panels=panels, kind="aggregation", caption=_caption(d), data=d
    )
