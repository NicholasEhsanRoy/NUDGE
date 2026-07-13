"""Temporal / generalized-Lotka–Volterra renderer (``NUDGE-METHOD-012``).

A perturbed microbial community's trajectories, and which knob the perturbation moved:
**growth (α)** / **interaction (β)** / **susceptibility (ε)** — or an honest abstention
when the growth ⇄ self-limitation pair (α⇄βᵢᵢ) is degenerate near equilibrium
(``NUDGE-LIM-020``). That degeneracy is a *direction in parameter space*, so the honest
output is the null-space direction, not a point estimate.

Panels: an optional (left) **trajectory** panel (reference vs perturbed species means, when
the caller supplies them); (centre) the **per-knob BIC** relative to the no-change null,
winner highlighted; and (right) the **identifiability** panel — |corr(α, βᵢᵢ)| against the
degeneracy threshold plus the condition number and the null-space direction — which is the
verdict panel, hatched when the fit is ``unresolved`` (degenerate).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nudge.viz._util import get, verdict_color
from nudge.viz.base import AnimationSpec, Panel, RenderedFigure, abstain_overlay, is_abstention
from nudge.viz.theme import apply_theme

_KNOBS = ("growth", "interaction", "susceptibility")
_KNOB_SHORT = {"growth": "growth\n(α)", "interaction": "interaction\n(β)",
               "susceptibility": "suscept.\n(ε)"}


def _f(x: Any, d: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def temporal_data(obj: Any) -> dict[str, Any]:
    """Normalise a gLV result (``lotka_demo`` / ``lotka_file`` dict / replay) to a dict."""
    if isinstance(obj, dict) and obj.get("kind") == "temporal":
        return obj
    ident = get(obj, "identifiability", default={}) or {}
    fit = get(obj, "fit", default=obj)
    bic = get(obj, "bic", default=None)
    if bic is None:
        bic = get(fit, "bic", default={})
    delta = get(obj, "delta", "fitted_delta", default=None)
    if delta is None:
        delta = get(fit, "delta", default={})
    dd = get(obj, "degeneracy_direction", default=None)
    return {
        "kind": "temporal",
        "label": str(get(obj, "label", default="community")),
        "call": str(get(obj, "call", default="unresolved")),
        "reason": str(get(obj, "reason", default="")),
        "selected_knob": str(get(obj, "selected_knob", default=get(fit, "selected",
                                                                   default=""))),
        "bic": {k: _f(v) for k, v in dict(bic).items()},
        "delta": {k: _f(v) for k, v in dict(delta).items()},
        "cond_number": _f(get(ident, "cond_number", default=get(fit, "cond_number"))),
        "abs_corr": _f(get(ident, "abs_corr_alpha_beta",
                           default=get(fit, "corr_alpha_beta"))),
        "degenerate": bool(get(ident, "degenerate", default=get(fit, "degenerate",
                                                                default=False))),
        "degeneracy_direction": None if dd is None else [float(x) for x in dd],
        "trajectories": get(obj, "trajectories", default=None),
    }


def _caption(d: dict[str, Any]) -> str:
    call = d["call"].upper()
    if is_abstention(d["call"]):
        return (f"{d['label']} → {call} (α⇄βᵢᵢ degenerate, |corr|={d['abs_corr']:.2f}; "
                "null-space direction reported, not a point estimate)")
    return f"{d['label']} → {call} (knob moved = {d['selected_knob']})"


def _draw_traj(ax: Any, traj: dict[str, Any], pal: dict[str, str]) -> None:
    t = np.asarray(traj["t"], dtype=float)
    ref = np.asarray(traj["reference"], dtype=float)
    pert = np.asarray(traj["perturbed"], dtype=float)
    for s in range(ref.shape[0]):
        ax.plot(t, ref[s], color=pal["muted"], lw=1.6, alpha=0.8,
                label="reference" if s == 0 else None)
    for s in range(pert.shape[0]):
        ax.plot(t, pert[s], color=pal["gain"], lw=1.8, ls="--",
                label="perturbed" if s == 0 else None)
    ax.set_xlabel("time")
    ax.set_ylabel("abundance")
    ax.set_title("community trajectories", fontweight="bold", color=pal["text"])
    ax.legend(loc="best", fontsize=8, framealpha=0.9)


def _draw_bic(ax: Any, d: dict[str, Any], pal: dict[str, str], color: str) -> None:
    bic = d["bic"]
    null = bic.get("null", 0.0)
    knobs = [k for k in _KNOBS if k in bic]
    xs = np.arange(len(knobs))
    deltas = [bic[k] - null for k in knobs]  # <0 ⇒ beats the no-change null
    sel = d["selected_knob"]
    colors = [color if k == sel and not is_abstention(d["call"]) else pal["muted"]
              for k in knobs]
    ax.bar(xs, deltas, color=colors, alpha=0.95, zorder=3, width=0.6)
    ax.axhline(0.0, color=pal["text"], lw=1.2, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([_KNOB_SHORT.get(k, k) for k in knobs], fontsize=8)
    ax.set_ylabel("ΔBIC vs no-change  (lower = favoured)")
    ax.set_title("which knob moved?", fontweight="bold", color=pal["text"])


def _draw_ident(ax: Any, d: dict[str, Any], pal: dict[str, str], color: str) -> None:
    corr = d["abs_corr"]
    thresh = 0.9  # the degeneracy cut used in the classifier's spirit
    bar_color = pal["abstain"] if d["degenerate"] else color
    ax.bar([0], [corr if np.isfinite(corr) else 0.0], width=0.5, color=bar_color,
           alpha=0.9, zorder=3)
    ax.axhline(thresh, ls="--", color=pal["muted"], lw=1.4, zorder=2)
    ax.text(0.0, thresh + 0.01, "degeneracy threshold", fontsize=7.5, color=pal["muted"],
            ha="center", va="bottom")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([0])
    ax.set_xticklabels(["|corr(α, βᵢᵢ)|"], fontsize=8)
    ax.set_ylabel("α ⇄ self-limitation correlation")
    ax.set_title("identifiability", fontweight="bold", color=pal["text"])
    lines = []
    if np.isfinite(d["cond_number"]):
        lines.append(f"cond # = {d['cond_number']:.1f}")
    dd = d["degeneracy_direction"]
    if dd is not None and len(dd) >= 2:
        lines.append("null dir ≈ [" + ", ".join(f"{x:+.2f}" for x in dd[:3]) + "]")
    if lines:
        ax.text(0.5, 0.02, "\n".join(lines), transform=ax.transAxes, fontsize=7.5,
                color=pal["muted"], ha="center", va="bottom", zorder=6)


def build(obj: Any, *, theme: str = "auto") -> RenderedFigure:
    """Build the temporal/gLV figure (no overlay — the pipeline stamps it off the call)."""
    import matplotlib.pyplot as plt

    pal = apply_theme(theme)
    d = temporal_data(obj)
    color = verdict_color(d["call"], pal)
    traj = d["trajectories"]
    if traj is not None:
        fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4),
                                 gridspec_kw={"width_ratios": [1.4, 1.2, 1.0]})
        _draw_traj(axes[0], traj, pal)
        ax_bic, ax_ident = axes[1], axes[2]
        context_axes = [axes[0], ax_bic]
    else:
        fig, (ax_bic, ax_ident) = plt.subplots(1, 2, figsize=(10.2, 4.4),
                                               gridspec_kw={"width_ratios": [1.4, 1.0]})
        context_axes = [ax_bic]
    _draw_bic(ax_bic, d, pal, color)
    _draw_ident(ax_ident, d, pal, color)
    fig.suptitle(f"{d['label']}  →  {d['call'].upper()}", fontweight="bold",
                 color=pal["text"], fontsize=13)
    fig.tight_layout()
    panels = [Panel(ax=a, call="", reason="", label=d["label"]) for a in context_axes]
    panels.append(Panel(ax=ax_ident, call=d["call"], reason=d["reason"], label=d["label"]))
    return RenderedFigure(
        fig=fig, panels=panels, kind="temporal", caption=_caption(d), data=d
    )


def build_animation(obj: Any, *, theme: str = "auto", frames: int = 28) -> AnimationSpec:
    """Animate the gLV community integrating under the antibiotic pulse — perturbed vs
    reference DIVERGING (``NUDGE-METHOD-012``; the temporal capability, in motion).

    Reads the enriched ``animation`` block (``service.temporal_animation_demo``: the per-
    timepoint reference vs perturbed mean trajectories + the pulse window) and sweeps a time
    cursor so the two communities visibly separate as the drug pulse hits — susceptibility is
    the identifiable positive. It only READS the simulated trajectories + the fit's verdict
    (no re-fit); a near-equilibrium growth change is the degenerate α⇄βᵢᵢ case that abstains
    (``NUDGE-LIM-020``), stamped by the overlay.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    pal = apply_theme(theme)
    d = temporal_data(obj)
    call = d["call"]
    abst = is_abstention(call)
    color = verdict_color(call, pal)
    anim = obj.get("animation", {}) if isinstance(obj, dict) else {}
    t = np.asarray(anim.get("t", []), dtype=float)
    ref = np.asarray(anim.get("reference", []), dtype=float)   # (T, S)
    pert = np.asarray(anim.get("perturbed", []), dtype=float)  # (T, S)
    if not len(t) or ref.ndim != 2:
        raise ValueError("temporal animation needs the enriched 'animation' block "
                         "(use service.temporal_animation_demo / "
                         "demo_result('temporal', animate=True))")
    pulse = anim.get("pulse_window", [float("nan"), float("nan")])
    target = int(anim.get("target", 0))
    labels = anim.get("species_labels", [f"taxon {i}" for i in range(ref.shape[1])])
    n_sp = ref.shape[1]
    # a distinct hue per taxon (target gets the verdict colour, others muted variants)
    cmap = [color if i == target else pal["muted"] for i in range(n_sp)]
    ymax = float(np.nanmax([ref.max(), pert.max()])) * 1.1 or 1.0
    hold = max(frames // 6, 2)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    def draw(i: int) -> None:
        ax.clear()
        cut = min(int(round(min(i, frames - hold) / max(frames - hold, 1) * (len(t) - 1))) + 1,
                  len(t))
        if np.isfinite(pulse[0]):
            ax.axvspan(pulse[0], pulse[1], color=pal["gain"], alpha=0.14, zorder=1)
            ax.text(0.5 * (pulse[0] + pulse[1]), ymax * 0.98, "antibiotic pulse",
                    ha="center", va="top", fontsize=8, color=pal["gain"])
        for s in range(n_sp):
            ax.plot(t[:cut], ref[:cut, s], ls="--", color=cmap[s], lw=1.6, alpha=0.7,
                    zorder=2)
            ax.plot(t[:cut], pert[:cut, s], ls="-", color=cmap[s], lw=2.4, zorder=3,
                    label=labels[s] if s == 0 or s == target else None)
        ax.plot([], [], ls="--", color=pal["muted"], label="reference")
        ax.plot([], [], ls="-", color=pal["muted"], label="perturbed")
        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.set_ylim(0.0, ymax)
        ax.set_xlabel("time")
        ax.set_ylabel("abundance")
        ax.set_title("gLV community: perturbed diverges from reference under the pulse",
                     fontweight="bold", color=pal["text"], fontsize=10.5)
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9, ncol=2)
        if abst:
            abstain_overlay(ax, call, d.get("reason", ""), palette=pal)

    anim_obj = FuncAnimation(
        fig, draw, frames=frames, interval=1000 // 8, blit=False,  # type: ignore[arg-type]
    )
    knob = d.get("selected_knob", "") or call
    caption = (f"{d['label']} → {call.upper()} "
               f"(perturbed vs reference diverge under the pulse; knob: {knob})")
    return AnimationSpec(fig=fig, anim=anim_obj, caption=caption, abstained=abst,
                         data=dict(obj) if isinstance(obj, dict) else {})
