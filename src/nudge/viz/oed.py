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
from nudge.viz.base import AnimationSpec, Panel, RenderedFigure, abstain_overlay, is_abstention
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


def build_animation(obj: Any, *, theme: str = "auto", frames: int = 28) -> AnimationSpec:
    """Animate the OED gradient: the measurement times slide from the naive near-equilibrium
    design into the informative transient while the (α,β) 95% confidence ellipse COLLAPSES
    (``NUDGE-METHOD-014``; the differentiability moat, in motion).

    Reads the enriched ``animation`` block (``service.oed_animation_demo``: a checkpointed
    design-φ trajectory + the per-step covariance ellipse) and draws it — it never re-runs
    the optimiser. Left panel: the transient backdrop with the current measurement times.
    Right panel: the parameter-uncertainty ellipse shrinking. Everything is MEASURED at θ₀
    (local OED, ``NUDGE-LIM-024``).
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Ellipse

    pal = apply_theme(theme)
    d = oed_data(obj)
    call = d.get("call", "")
    abst = is_abstention(call)
    anim_data = obj.get("animation", {}) if isinstance(obj, dict) else {}
    cps = anim_data.get("frames", [])
    if not cps:
        raise ValueError("oed animation needs the enriched 'animation' block "
                         "(use service.oed_animation_demo / demo_result('oed', animate=True))")

    labels = anim_data.get("param_labels", ["θ₁", "θ₂"])
    theta0 = anim_data.get("theta0", [0.0, 0.0])
    tb = anim_data.get("t_bounds", [0.0, 1.0])
    traj_t = np.asarray(anim_data.get("traj_t", []), dtype=float)
    traj_x = np.asarray(anim_data.get("traj_x", []), dtype=float)
    n_cp = len(cps)
    w0 = float(cps[0]["ellipse"]["width"])
    h0 = float(cps[0]["ellipse"]["height"])
    span = 0.62 * max(w0, h0)

    fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(10.6, 4.4))
    hold = max(frames // 5, 2)

    def draw(i: int) -> None:
        ci = min(int(round(min(i, frames - hold) / max(frames - hold, 1) * (n_cp - 1))),
                 n_cp - 1)
        fr = cps[ci]
        # left — the transient the samples slide into
        ax_t.clear()
        if len(traj_t):
            ax_t.plot(traj_t, traj_x, color=pal["muted"], lw=2.0, zorder=2,
                      label="growth transient x(t)")
        for x in fr["phi"]:
            ax_t.axvline(x, color=pal["threshold"], lw=1.6, alpha=0.85, zorder=3)
        ax_t.plot(fr["phi"], [traj_x.min() if len(traj_x) else 0.0] * len(fr["phi"]),
                  "o", color=pal["threshold"], ms=7, zorder=4, label="measurement times")
        ax_t.set_xlim(tb[0], tb[1])
        ax_t.set_xlabel("time")
        ax_t.set_ylabel("abundance x(t)")
        ax_t.set_title("measurement schedule → the transient", fontweight="bold",
                       color=pal["text"])
        ax_t.legend(loc="lower right", fontsize=8, framealpha=0.9)
        # right — the confidence ellipse collapsing
        ax_e.clear()
        ell = fr["ellipse"]
        ax_e.add_patch(Ellipse((theta0[0], theta0[1]), float(ell["width"]),
                               float(ell["height"]), angle=float(ell["angle"]),
                               facecolor=pal["threshold"], alpha=0.28,
                               edgecolor=pal["threshold"], lw=2.0, zorder=3))
        ax_e.plot([theta0[0]], [theta0[1]], "+", color=pal["text"], ms=11, mew=2, zorder=4)
        ax_e.set_xlim(theta0[0] - span, theta0[0] + span)
        ax_e.set_ylim(theta0[1] - span, theta0[1] + span)
        ax_e.set_xlabel(labels[0])
        ax_e.set_ylabel(labels[1])
        crlb = fr.get("target_crlb", float("nan"))
        ax_e.set_title(f"95% confidence ellipse  (CRLB→{crlb:.2g})", fontweight="bold",
                       color=pal["text"])
        if abst:
            abstain_overlay(ax_e, call, d.get("reason", ""), palette=pal)

    imprv = d.get("crlb_improvement", float("nan"))
    fig.suptitle(f"gradient OED — CRLB of {d['target_parameter']} ×{imprv:.2g} "
                 "(MEASURED at θ₀)", fontweight="bold", color=pal["text"], fontsize=12)
    fig.tight_layout()
    anim = FuncAnimation(
        fig, draw, frames=frames, interval=1000 // 8, blit=False,  # type: ignore[arg-type]
    )
    caption = (f"{d['label']} → measurement times slide into the transient; the (α,β) 95% "
               f"ellipse collapses (CRLB ×{imprv:.2g}, MEASURED at θ₀; NUDGE-LIM-024)")
    # the animation block rides in the sidecar so fig.py replays it with no re-fit
    out = dict(d)
    out["animation"] = anim_data
    return AnimationSpec(fig=fig, anim=anim, caption=caption, abstained=abst, data=out)
