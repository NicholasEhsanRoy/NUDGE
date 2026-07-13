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

from nudge.viz._util import ease, get
from nudge.viz.base import AnimationSpec, Panel, RenderedFigure, abstain_overlay, is_abstention
from nudge.viz.theme import apply_theme

#: Warning-red for the safety WARNING (a decided-but-dangerous plan) — deliberately NOT the
#: grey ``abstain`` slot, so the HIGH-RISK banner reads as a red WARNING and can never be
#: mistaken for the grey "I can't tell" abstention hatch (they must stay visually distinct).
_RISK = "#d62728"


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


def _draw_trajectory(ax: Any, d: dict[str, Any], p: float, pal: dict[str, str]) -> None:
    """Left panel at progress ``p`` — the knobs ramping 1.0 → their target ``factor``.

    Each knob is a bar of ``log2`` fold change that grows from 0 (×1.0, the untouched
    circuit) toward its target ``log2(factor)`` as ``p`` advances (a ghost dashed outline
    marks the target). ``factor < 1`` = a reduction (blue, threshold hue); ``> 1`` = an
    increase (orange, gain hue). Dose-mode plans ramp a single dose bar instead.
    """
    lbl = d["label"] or "the target"
    verb = "" if lbl.strip().lower().startswith("flip") else "flip "
    title = f"intervention to {verb}{lbl}"

    # dose-mode plan: a single dose bar ramping 0 → the proposed dose.
    if d["mode"] == "dose" or (not d["deltas"] and np.isfinite(d["dose"])):
        dose = d["dose"]
        cur = p * dose if np.isfinite(dose) else 0.0
        if np.isfinite(dose):
            ax.bar([0], [dose], width=0.5, facecolor="none", edgecolor=pal["muted"],
                   linestyle="--", linewidth=1.3, zorder=2)  # target ghost
        ax.bar([0], [cur], width=0.5, color=pal["ceiling"], alpha=0.9, zorder=3)
        if np.isfinite(cur):
            ax.text(0, cur, f"{cur:.3g}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=pal["text"], zorder=5)
        ax.set_xticks([0])
        ax.set_xticklabels(["proposed dose"], fontsize=9)
        ax.set_ylabel("dose")
        ax.set_title(title, fontweight="bold", color=pal["text"])
        return

    deltas = d["deltas"]
    if not deltas:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no knob changes proposed", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color=pal["muted"])
        ax.set_title(title, fontweight="bold", color=pal["text"])
        return

    names = [x["name"] for x in deltas]
    factors = [x["factor"] for x in deltas]
    xs = np.arange(len(deltas))
    tgt = [np.log2(f) if f and f > 0 else 0.0 for f in factors]
    cur = [p * t for t in tgt]
    colors = [pal["threshold"] if t < 0 else pal["gain"] for t in tgt]
    ax.bar(xs, tgt, facecolor="none", edgecolor=colors, linestyle="--", linewidth=1.3,
           width=0.55, zorder=2)  # ghost target the bars grow toward
    ax.bar(xs, cur, color=colors, alpha=0.9, width=0.55, zorder=3)
    ax.axhline(0.0, color=pal["text"], lw=1.1, zorder=2)
    for xi, ci in zip(xs, cur, strict=True):
        ax.text(xi, ci, f"×{2.0 ** ci:.2f}", ha="center",
                va="bottom" if ci >= 0 else "top", fontsize=8, fontweight="bold",
                color=pal["text"], zorder=5)
    lim = max((abs(t) for t in tgt), default=0.5) * 1.35 or 0.5
    ax.set_ylim(-max(lim, 0.5), max(lim, 0.5))
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("log2 fold change in knob  (→ target)")
    ax.set_title(title, fontweight="bold", color=pal["text"])
    ax.text(0.98, 0.02, f"progress to target: {p:.0%}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=pal["muted"], zorder=6)


def _draw_safety_gauge(ax: Any, d: dict[str, Any], p: float, at_target: bool,
                       pal: dict[str, str], near: float) -> None:
    """Right panel at progress ``p`` — the fold-proximity dial climbing before → after.

    A 0→1 proximity gauge (0 = deep, safe basin · 1 = at the fold) whose fill climbs
    ``proximity_before`` → ``proximity_after`` as ``p`` advances, with the ``NEAR_FOLD``
    danger threshold as a red dashed line. A red HIGH-RISK banner fires when the current
    proximity reaches ``NEAR_FOLD`` OR the fit flagged ``high_risk`` / ``crosses_fold`` at
    full progress — a decided-but-dangerous WARNING (red), NOT the grey abstain hatch.
    """
    s = d["safety"]
    if s is None:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no bifurcation safety gate\n(curve mode: no circuit/fold)",
                transform=ax.transAxes, ha="center", va="center", fontsize=10,
                color=pal["muted"])
        ax.set_title("safety dial", fontweight="bold", color=pal["text"])
        return

    before, after = s["proximity_before"], s["proximity_after"]
    crosses, high_risk, one_sided = s["crosses_fold"], s["high_risk"], s["one_sided"]
    b0 = before if np.isfinite(before) else 0.0
    if np.isfinite(after):
        target = after
    elif crosses:
        target = 1.0  # crossing the fold → proximity past 1 is undefined; ramp to the fold
    else:
        target = b0
    cur = b0 + p * (target - b0)
    risky_final = high_risk or crosses
    show_risk = cur >= near or (at_target and risky_final)
    bar_color = _RISK if show_risk else pal["ceiling"]

    ax.barh([0], [1.0], height=0.5, color=pal["grid"], zorder=1)  # the full 0..1 track
    ax.barh([0], [min(cur, 1.0)], height=0.5, color=bar_color, alpha=0.9, zorder=3)
    ax.text(min(cur, 1.0), 0.42, f"{cur:.2f}", ha="center", va="bottom", fontsize=9,
            fontweight="bold", color=bar_color, zorder=5)
    if crosses and not np.isfinite(after) and cur >= 1.0 - 1e-9:
        # basin destabilised past the fold → an OPEN-ENDED arrow, never a fake height.
        ax.annotate("", xy=(1.13, 0), xytext=(1.0, 0),
                    arrowprops={"arrowstyle": "-|>", "color": _RISK, "lw": 2.4}, zorder=4)

    ax.axvline(near, ls="--", color=_RISK, lw=1.8, zorder=4)
    ax.text(near, -0.55, f"near-fold\n({near:g})", ha="center", va="top", fontsize=7.5,
            color=_RISK)
    ax.axvline(1.0, ls="--", color=pal["gain"], lw=1.6, zorder=4)
    ax.text(1.0, -0.55, "fold\n(tipping point)", ha="center", va="top", fontsize=7.5,
            color=pal["gain"])
    ax.text(0.0, -0.55, "deep basin\n(safe)", ha="center", va="top", fontsize=7.5,
            color=pal["muted"])
    ax.set_xlim(-0.02, 1.16)
    ax.set_ylim(-0.98, 1.05)
    ax.set_yticks([])
    ax.set_xlabel("fold proximity  (0 = deep · 1 = at the fold)")
    ax.set_title("safety dial", fontweight="bold", color=pal["text"])

    if show_risk:
        msg = ("CROSSES THE FOLD — destroys bistability" if crosses
               else "HIGH RISK — approaches the fold")
        ax.text(0.5, 0.9, msg, transform=ax.transAxes, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=_RISK, zorder=7,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": pal["surface"],
                      "alpha": 0.92, "edgecolor": _RISK})
    if one_sided:
        ax.text(0.5, 0.02, "proximity is a one-sided lower bound near the fold "
                "(NUDGE-LIM-012)", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.5, color=pal["muted"], zorder=6)


def _draw_unreachable(ax_traj: Any, ax_safe: Any, d: dict[str, Any],
                      pal: dict[str, str]) -> None:
    """The static UNREACHABLE state — an abstention has no trajectory to animate."""
    ax_traj.set_axis_off()
    ax_traj.text(0.5, 0.58, "no reachable intervention", transform=ax_traj.transAxes,
                 ha="center", fontsize=13, fontweight="bold", color=pal["abstain"])
    ax_traj.text(0.5, 0.42, d["reason"][:110], transform=ax_traj.transAxes,
                 ha="center", fontsize=8.5, color=pal["muted"])
    ax_safe.set_axis_off()
    ax_safe.text(0.5, 0.5, "no safety dial\n(target unreachable — nothing to gate)",
                 transform=ax_safe.transAxes, ha="center", va="center", fontsize=10,
                 color=pal["muted"])


def build_animation(obj: Any, *, theme: str = "auto", frames: int = 28) -> AnimationSpec:
    """Animate the intervention TRAJECTORY to target + the safety dial climbing to the fold.

    The frame variable is intervention progress ``p`` (0 = the untouched circuit, every knob
    ×1.0 → 1 = the target, each knob at its ``factor``), with a short hold at the end. Left
    panel: the knobs ramp toward their target factors ("intervention to flip <label>"). Right
    panel: the fold-proximity **safety dial** climbs ``proximity_before`` → ``proximity_after``
    with the ``NEAR_FOLD`` threshold as a red dashed line, raising a red **HIGH RISK** WARNING
    when the plan reaches the near-fold band (``NUDGE-METHOD-007``; the safety gate, in motion).

    Everything is computed from the frozen result dict (:func:`design_data`) — it NEVER re-fits
    (only the :data:`~nudge.inference.bifurcation.NEAR_FOLD` float is read). An *unreachable*
    design is an ABSTENTION: the abstention overlay fires and no trajectory is drawn. A
    reachable-but-risky plan is NOT an abstention — the red HIGH-RISK banner is a confident
    WARNING, kept visually distinct from the grey "I can't tell" hatch.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    from nudge.inference.bifurcation import NEAR_FOLD  # a plain float constant, not the fit

    pal = apply_theme(theme)
    d = design_data(obj)
    abst = d["design_kind"] == "abstention" or is_abstention(d["call"])
    overlay_verdict = d["verdict"] or d["call"] or "unreachable"

    fig, (ax_traj, ax_safe) = plt.subplots(
        1, 2, figsize=(11.0, 4.7), gridspec_kw={"width_ratios": [1.3, 1.0]}
    )
    hold = max(frames // 6, 2)
    span = max(frames - 1 - hold, 1)

    def draw(i: int) -> None:
        raw = min(i, span) / span
        p = ease(raw)
        at_target = raw >= 1.0 - 1e-9
        for a in (ax_traj, ax_safe):
            a.clear()
        if abst:
            # An unreachable design has no trajectory — draw the static UNREACHABLE state and
            # stamp the abstention overlay (the honesty lock; distinct from a risk WARNING).
            _draw_unreachable(ax_traj, ax_safe, d, pal)
            abstain_overlay(ax_traj, overlay_verdict, d["reason"], palette=pal)
            return
        _draw_trajectory(ax_traj, d, p, pal)
        _draw_safety_gauge(ax_safe, d, p, at_target, pal, NEAR_FOLD)

    generic = d["label"].strip().lower() in ("", "intervention", "design", "flip")
    head = ("target unreachable  →  ABSTAIN" if abst
            else "intervention trajectory to target")
    title = head if generic else f"{d['label']}  —  {head}"
    fig.suptitle(title, fontweight="bold", color=pal["text"], fontsize=13)
    fig.tight_layout()
    anim = FuncAnimation(
        fig, draw, frames=frames, interval=1000 // 8, blit=False,  # type: ignore[arg-type]
    )

    if abst:
        caption = (f"{d['label']} → UNREACHABLE "
                   f"({d['verdict'] or d['reason'][:70] or 'no reachable intervention'}) "
                   "— no trajectory drawn")
    else:
        knobs = ", ".join(f"{x['name']}×{x['factor']:.2f}" for x in d["deltas"][:3]) or "—"
        s = d["safety"]
        risk = ""
        if s is not None:
            after = s["proximity_after"]
            if s["crosses_fold"]:
                risk = " — CROSSES THE FOLD (destroys bistability)"
            elif s["high_risk"] or (np.isfinite(after) and after >= NEAR_FOLD):
                risk = " — HIGH RISK (approaches the fold)"
            if s["one_sided"]:
                risk += " [one-sided lower bound, NUDGE-LIM-012]"
        caption = f"{d['label']} → intervention to target [{knobs}]{risk}"

    return AnimationSpec(fig=fig, anim=anim, caption=caption, abstained=abst, data=dict(d))
