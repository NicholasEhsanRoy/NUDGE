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

from nudge.viz._util import ease, get, verdict_color
from nudge.viz.base import AnimationSpec, Panel, RenderedFigure, abstain_overlay, is_abstention
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


def build_animation(obj: Any, *, theme: str = "auto", frames: int = 28) -> AnimationSpec:
    """Animate the joint panel RESOLVING as reporters are added one at a time
    (``NUDGE-METHOD-008``; the identifiability force-multiplier, in motion).

    Frame variable ``k`` = the number of reporters included, sweeping 1 → ``n_reporters``
    with a short hold at the end (GIF frame → ``k`` exactly as the reference animators map
    frame → checkpoint). Left panel: the reporters popping in one at a time with their
    independent-fit R². Right panel: a **mechanism-resolution landscape** over the four
    hypotheses {no-effect, threshold, gain, ceiling}, whose bars start FLAT / near-equal at
    ``k=1`` (the single-reporter K⇄v_max / gain⇄threshold degeneracy — ambiguous) and SHARPEN
    toward the real, peaked-at-``winner`` shape as ``k`` → ``n``, so the winner visibly lights
    up only once the joint panel over-determines the shared latent.

    Reads ONLY the frozen result dict (``multi_reporter_data``) — it never re-fits. Honesty is
    the same as the static path: the abstention overlay fires off the result's OWN ``call`` (an
    ``off-model`` panel — no single shared latent — hatches every frame; a resolved ``ceiling``
    does not, because the JOINT genuinely resolved), and ``abstained`` is stamped off that same
    verdict. The single→joint story is carried by the flat→peaked bars + labels, never a faked
    per-frame verdict.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    pal = apply_theme(theme)
    d = multi_reporter_data(obj)
    call = d["call"]
    abst = is_abstention(call)
    color = verdict_color(call, pal)
    reps = d["reporters"]
    n = max(int(d["n_reporters"]), len(reps), 1)

    # Mechanism-resolution landscape: lower restricted-loss ⇒ taller "resolution" bar. The
    # k=1 shape is FLAT (every bar at the mean — the single-reporter degeneracy); the k=n
    # shape is the real, peaked-at-winner shape (normalised so the winner reaches 1.0). Frames
    # interpolate flat → peaked, so the landscape visibly sharpens as reporters accumulate.
    hyps = ["no_effect", "threshold", "gain", "ceiling"]
    hyp_labels = ["no-effect", "threshold\n(K)", "gain\n(n)", "ceiling\n(A)"]
    raw = [_f(d["losses"].get(h)) for h in hyps]
    finite = [v for v in raw if np.isfinite(v)]
    max_loss = max(finite) if finite else 1.0
    target = [(max_loss - v) if np.isfinite(v) else 0.0 for v in raw]
    peak = max(target)
    target = [t / peak for t in target] if peak > 0 else target  # winner → 1.0
    flat = sum(target) / len(target)  # the degenerate, near-equal single-reporter shape
    winner = d["winner"].lower()

    fig, (ax_rep, ax_land) = plt.subplots(1, 2, figsize=(10.6, 4.4))
    hold = max(frames // 5, 2)
    span = max(frames - hold, 1)

    def draw(i: int) -> None:
        frac = min(i, frames - hold) / span
        k = min(int(round(frac * (n - 1))) + 1, n)
        w = ease(frac)
        resolved = (k >= n) and not abst
        bars = [(1.0 - w) * flat + w * t for t in target]

        # left — reporters accumulate one at a time (ghosted slots wait to be filled)
        ax_rep.clear()
        for j in range(n):
            r2 = _f(reps[j].get("r2_independent")) if j < len(reps) else float("nan")
            if j < k and np.isfinite(r2):
                ax_rep.barh(j, r2, height=0.55, color=pal["muted"], alpha=0.9, zorder=3)
                ax_rep.text(min(r2 + 0.02, 1.12), j, f"{r2:.2f}", va="center", fontsize=8,
                            color=pal["text"], zorder=4)
            elif j < k:  # included but no per-reporter R² available
                ax_rep.barh(j, 1.0, height=0.55, color=pal["muted"], alpha=0.25, zorder=2)
            else:  # a not-yet-added slot
                ax_rep.barh(j, 1.0, height=0.55, color=pal["grid"], alpha=0.4, zorder=1)
        ax_rep.set_yticks(range(n))
        ax_rep.set_yticklabels(
            [str(reps[j].get("name", f"R{j}")) if j < len(reps) else f"R{j}" for j in range(n)],
            fontsize=8)
        ax_rep.set_xlim(0.0, 1.15)
        ax_rep.set_ylim(-0.6, n - 0.4)
        ax_rep.invert_yaxis()
        ax_rep.set_xlabel("R²  (independent fit)")
        ax_rep.set_title(f"{k} of {n} reporters", fontweight="bold", color=pal["text"])

        # right — the mechanism-resolution landscape sharpening flat → peaked-at-winner
        ax_land.clear()
        xs = np.arange(len(hyps))
        colors = [
            color if (h == winner and resolved)
            else pal["grid"] if h == "no_effect"
            else pal["muted"]
            for h in hyps
        ]
        ax_land.bar(xs, bars, color=colors, alpha=0.95, zorder=3, width=0.62)
        ax_land.set_xticks(xs)
        ax_land.set_xticklabels(hyp_labels, fontsize=8)
        ax_land.set_ylim(0.0, 1.2)
        ax_land.set_ylabel("resolution score  (taller = better fit ⇒ more resolved)")
        ax_land.set_title("mechanism-resolution landscape", fontweight="bold",
                          color=pal["text"])
        if resolved and winner in hyps:
            wi = hyps.index(winner)
            ax_land.text(wi, bars[wi] + 0.04, "winner", ha="center", va="bottom",
                         fontsize=9, fontweight="bold", color=color, zorder=6)
        if abst:
            abstain_overlay(ax_land, call, d["reason"], palette=pal)
        else:
            msg = "single reporter — degenerate" if k < n else "joint panel — resolved"
            ax_land.text(0.5, 0.95, msg, transform=ax_land.transAxes, ha="center", va="top",
                         fontsize=8.5, color=pal["muted"], zorder=6)

    fig.suptitle(f"{d['label']}  —  joint attribution as reporters accumulate",
                 fontweight="bold", color=pal["text"], fontsize=13)
    fig.tight_layout()
    anim = FuncAnimation(
        fig, draw, frames=frames, interval=1000 // 8, blit=False,  # type: ignore[arg-type]
    )
    if abst:
        caption = (f"{d['label']} → {call.upper()} — reporters add but the joint panel can't "
                   f"resolve ({d['reason'][:60] or 'not jointly resolvable'})")
    else:
        caption = (f"{d['label']} → {call.upper()}: {n} reporters added one at a time; the "
                   f"resolution landscape sharpens to the winner ({d['winner']}, knob margin "
                   f"{d['knob_margin']:.2f})")
    return AnimationSpec(
        fig=fig, anim=anim, caption=caption, abstained=abst,
        data=dict(obj) if isinstance(obj, dict) else d,
    )
