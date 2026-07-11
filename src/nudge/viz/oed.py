"""Optimal-experimental-design renderer (``NUDGE-METHOD-014``; the differentiability moat).

Gradient-optimise WHEN to measure so an otherwise-degenerate parameter becomes identifiable,
and report the MEASURED identifiability gain — never asserted (local OED at θ₀,
``NUDGE-LIM-024``).

Two panels: (left) the **measurement schedule** — the naive (near-equilibrium) design vs the
gradient-optimised design on the time axis, so you can see the optimiser pull samples into
the informative transient; and (right) the **FIM lift** — the target parameter's Cramér–Rao
lower bound and the FIM's smallest eigenvalue, before vs after, with the improvement factor.
OED always returns a measured result (no abstention in the default path), but the renderer
still routes any verdict through the honesty overlay for uniformity.
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


def _list(x: Any) -> list[float]:
    return [] if x is None else [_f(v) for v in x]


def oed_data(obj: Any) -> dict[str, Any]:
    """Normalise an ``oed_demo`` dict / replay to a figure dict."""
    if isinstance(obj, dict) and obj.get("kind") == "oed":
        return obj
    return {
        "kind": "oed",
        "label": str(get(obj, "label", default="experimental design")),
        "call": str(get(obj, "call", default="")),
        "reason": str(get(obj, "reason", default="")),
        "model": str(get(obj, "model", default="")),
        "objective": str(get(obj, "objective", default="")),
        "target_parameter": str(get(obj, "target_parameter", default="θ")),
        "phi_init": _list(get(obj, "phi_init")),
        "phi_opt": _list(get(obj, "phi_opt")),
        "target_crlb_init": _f(get(obj, "target_crlb_init")),
        "target_crlb_opt": _f(get(obj, "target_crlb_opt")),
        "crlb_improvement": _f(get(obj, "crlb_improvement")),
        "min_eig_init": _f(get(obj, "min_eig_init")),
        "min_eig_opt": _f(get(obj, "min_eig_opt")),
        "min_eig_improvement": _f(get(obj, "min_eig_improvement")),
    }


def _caption(d: dict[str, Any]) -> str:
    return (f"{d['label']} ({d['model']}/{d['objective']}) → CRLB of "
            f"{d['target_parameter']} improves ×{d['crlb_improvement']:.2g}, "
            f"FIM min-eig ×{d['min_eig_improvement']:.2g} (MEASURED at θ₀; NUDGE-LIM-024)")


def _draw_schedule(ax: Any, d: dict[str, Any], pal: dict[str, str]) -> None:
    init, opt = d["phi_init"], d["phi_opt"]
    if init:
        ax.plot(init, [1.0] * len(init), "o", color=pal["muted"], ms=9, zorder=3,
                label="naive (near-equilibrium)")
    if opt:
        ax.plot(opt, [0.0] * len(opt), "o", color=pal["threshold"], ms=9, zorder=3,
                label="optimised")
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["optimised", "naive"], fontsize=9)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("measurement time")
    ax.set_title("measurement schedule", fontweight="bold", color=pal["text"])
    ax.legend(loc="best", fontsize=8, framealpha=0.9)


def _draw_lift(ax: Any, d: dict[str, Any], pal: dict[str, str]) -> None:
    eig_i, eig_o = d["min_eig_init"], d["min_eig_opt"]
    xs = np.arange(2)
    vals = [eig_i, eig_o]
    ax.bar(xs, vals, color=[pal["muted"], pal["ceiling"]], alpha=0.9, width=0.5, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(["naive", "optimised"], fontsize=9)
    ax.set_ylabel("FIM smallest eigenvalue  (↑ = more identifiable)")
    ax.set_title("identifiability lift", fontweight="bold", color=pal["text"])
    fac = d["min_eig_improvement"]
    if np.isfinite(fac):
        ax.text(0.5, 0.92, f"×{fac:.2g} lift", transform=ax.transAxes, ha="center",
                va="center", fontsize=12, fontweight="bold", color=pal["ceiling"],
                zorder=6)
    if np.isfinite(d["crlb_improvement"]):
        # Left-upper region is empty (the optimised bar is on the right) — keep the caveat
        # off the bars.
        ax.text(0.03, 0.62, f"CRLB of {d['target_parameter']}\n×{d['crlb_improvement']:.2g}"
                "  ·  MEASURED at θ₀\n(local OED, NUDGE-LIM-024)", transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5, color=pal["muted"])


def build(obj: Any, *, theme: str = "auto") -> RenderedFigure:
    """Build the OED figure (no overlay — the pipeline stamps it off the call)."""
    import matplotlib.pyplot as plt

    pal = apply_theme(theme)
    d = oed_data(obj)
    fig, (ax_sched, ax_lift) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    _draw_schedule(ax_sched, d, pal)
    _draw_lift(ax_lift, d, pal)
    fig.suptitle(f"{d['label']}  —  optimal experimental design", fontweight="bold",
                 color=pal["text"], fontsize=13)
    fig.tight_layout()
    panels = [
        Panel(ax=ax_lift, call=d["call"], reason=d["reason"], label=d["label"]),
        Panel(ax=ax_sched, call="", reason="", label=d["label"]),
    ]
    return RenderedFigure(
        fig=fig, panels=panels, kind="oed", caption=_caption(d), data=d
    )
