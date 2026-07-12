"""Sloppiness / identifiability renderer — the Fisher (FIM) eigenvalue spectrum.

This is the picture for :class:`nudge.inference.sloppiness.SloppinessReport`: it separates
the three verdicts a naive condition-number test conflates — **well-constrained** (narrow
spectrum), **sloppy-but-predictive** (a Fisher spectrum spanning many decades, so the
individual parameters are loose, yet the *predictions* are tight — NUDGE must NOT abstain
on this), and **unidentifiable** (a structural null, or loose directions that also make
predictions loose — NUDGE abstains). The figure exists to make "sloppy ≠ unidentifiable"
legible, so a prettier picture never tempts an over-abstention (or a false confident fit).

Two panels: (left) the **FIM eigenvalue spectrum** on a log scale — each eigenvalue as a
stem, the sloppy band (small eigenvalues) shaded, any structural-null eigenvalues pinned at
the rank floor in the abstain grey, annotated with the spectral span (decades) and the
condition number; and (right) the **naive-vs-measured verdict** — what a
condition-number-only test would say against NUDGE's measured verdict, plus the relative
prediction uncertainty against its tolerance, so a *sloppy-but-predictive* model reads as
usable (naive "unidentifiable" but tight predictions) rather than abstained. Reads a
``SloppinessReport`` dataclass or the canonical replay dict. The renderer routes the
verdict through the honesty overlay (``unidentifiable`` is an abstention).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nudge.viz._util import get
from nudge.viz.base import Panel, RenderedFigure
from nudge.viz.theme import apply_theme

_VERDICT_COLOR = {
    "well-constrained": "ceiling",
    "sloppy-but-predictive": "threshold",
    "unidentifiable": "abstain",
}


def _f(x: Any, d: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def _floats(x: Any) -> list[float]:
    if x is None:
        return []
    return [_f(v) for v in np.asarray(x).ravel().tolist()]


def identifiability_data(obj: Any) -> dict[str, Any]:
    """Normalise a ``SloppinessReport`` (dataclass / replay dict) → a figure dict.

    ``SloppinessReport.label`` IS the verdict (well-constrained / sloppy-but-predictive /
    unidentifiable), so it maps to ``call`` (not a display name); the display name comes
    from an optional ``model_label`` / ``name`` the caller passes (else "model").
    """
    if isinstance(obj, dict) and obj.get("kind") == "identifiability":
        return obj
    verdict = str(get(obj, "verdict", default=None) or get(obj, "label", default="unidentifiable"))
    nulls = get(obj, "null_directions", default=()) or ()
    null_hint = ""
    if nulls:
        first = nulls[0]
        null_hint = str(get(first, "hint", default=""))
    return {
        "kind": "identifiability",
        "label": str(get(obj, "model_label", "model_name", "name", default="model")),
        "call": verdict,
        "verdict": verdict,
        "reason": str(get(obj, "reason", default="")),
        "param_names": [str(n) for n in (get(obj, "param_names", default=()) or ())],
        "fim_eigenvalues": _floats(get(obj, "fim_eigenvalues")),
        "cond_number": _f(get(obj, "cond_number")),
        "span_decades": _f(get(obj, "spectral_span_decades", "span_decades")),
        "smallest_eigenvalue": _f(get(obj, "smallest_eigenvalue")),
        "largest_eigenvalue": _f(get(obj, "largest_eigenvalue")),
        "n_sloppy_dims": int(_f(get(obj, "n_sloppy_dims"), 0)),
        "n_null_dims": int(_f(get(obj, "n_null_dims"), 0)),
        "is_sloppy": bool(get(obj, "is_sloppy", default=False)),
        "predictive": bool(get(obj, "predictive", default=False)),
        "relative_prediction_std": _f(get(obj, "relative_prediction_std")),
        "pred_rel_tol": _f(get(obj, "pred_rel_tol"), 0.05),
        "naive_verdict": str(get(obj, "naive_verdict", default="")),
        "naive_is_wrong": bool(get(obj, "naive_is_wrong", default=False)),
        "sloppy_decade_threshold": _f(get(obj, "sloppy_decade_threshold"), 3.0),
        "null_hint": null_hint,
    }


def _caption(d: dict[str, Any]) -> str:
    v = d["verdict"].upper()
    span = d["span_decades"]
    span_txt = "∞" if not np.isfinite(span) else f"{span:.1f}"
    tail = ""
    if d["naive_is_wrong"]:
        tail = " — a condition-number-only test would WRONGLY call this unidentifiable"
    return (f"{d['label']} → {v} (FIM spectrum spans {span_txt} decades, "
            f"rel. prediction std {d['relative_prediction_std']:.1%}){tail}")


def _color(d: dict[str, Any], pal: dict[str, str]) -> str:
    return pal[_VERDICT_COLOR.get(d["verdict"], "threshold")]


def _draw_spectrum(ax: Any, d: dict[str, Any], pal: dict[str, str], color: str) -> None:
    evals = np.asarray(d["fim_eigenvalues"], dtype=float)
    if evals.size == 0:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no FIM spectrum", transform=ax.transAxes, ha="center",
                va="center", fontsize=11, color=pal["muted"])
        return
    evals = np.sort(evals)
    lam_max = float(evals[-1]) if evals[-1] > 0 else 1.0
    floor = lam_max * 1e-16  # plot-floor so a structural-null (~0) eigenvalue stays on log-y
    sloppy_cut = lam_max * 10.0 ** (-d["sloppy_decade_threshold"])
    null_cut = lam_max * (1e-7**2)  # rank_rtol² (FIM eigenvalue = sv²/σ²); see sloppiness.py
    xs = np.arange(1, evals.size + 1)

    # shade the sloppy band (small eigenvalues that make individual parameters loose).
    ax.axhspan(floor, sloppy_cut, color=pal["muted"], alpha=0.10, zorder=0)
    ax.axhline(sloppy_cut, ls=":", lw=1.1, color=pal["muted"], zorder=1)
    ax.annotate("sloppy band", xy=(0.02, sloppy_cut), xycoords=("axes fraction", "data"),
                xytext=(0, -2), textcoords="offset points", fontsize=7.5,
                color=pal["muted"], ha="left", va="top")

    for x, lam in zip(xs, evals, strict=True):
        lp = max(float(lam), floor)
        is_null = float(lam) < null_cut
        c = pal["abstain"] if is_null else (color if lp >= sloppy_cut else pal["gain"])
        ax.plot([x, x], [floor, lp], lw=1.2, color=c, zorder=2)
        ax.plot([x], [lp], "o", ms=8, color=c, zorder=3,
                markerfacecolor="none" if is_null else c)
    ax.set_yscale("log")
    ax.set_xlim(0.4, evals.size + 0.6)
    ax.set_xticks(xs)
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("FIM eigenvalue (log-parameter)")
    ax.set_title("Fisher-information spectrum", fontweight="bold", color=pal["text"])
    span = d["span_decades"]
    span_txt = "∞ (structural null)" if not np.isfinite(span) else f"{span:.1f} decades"
    cond = d["cond_number"]
    cond_txt = "∞" if not np.isfinite(cond) else f"{cond:.1e}"
    ax.annotate(f"span {span_txt}\ncond {cond_txt}",
                xy=(0.98, 0.02), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=8, color=pal["text"],
                bbox={"boxstyle": "round,pad=0.3", "facecolor": pal["surface"],
                      "alpha": 0.9, "edgecolor": pal["muted"]})


def _draw_verdict(ax: Any, d: dict[str, Any], pal: dict[str, str], color: str) -> None:
    """Naive (condition-number-only) verdict vs the measured one, + prediction tightness."""
    rel = d["relative_prediction_std"]
    tol = d["pred_rel_tol"]
    # prediction-uncertainty bar against the tolerance line.
    rel_plot = rel if np.isfinite(rel) else tol * 3
    bar_c = color if (np.isfinite(rel) and rel < tol) else pal["abstain"]
    ax.bar([0], [rel_plot], width=0.5, color=bar_c, alpha=0.9, zorder=3)
    ax.axhline(tol, ls="--", lw=1.5, color=pal["gain"], zorder=2)
    ax.annotate(f"predictions 'tight' below {tol:.0%}", xy=(0.5, tol),
                xycoords=("axes fraction", "data"), xytext=(0, 3),
                textcoords="offset points", fontsize=7.5, color=pal["gain"],
                ha="center", va="bottom")
    ax.set_xticks([0])
    ax.set_xticklabels(["prediction"], fontsize=9)
    ax.set_ylabel("max relative prediction std")
    ax.set_ylim(0.0, max(rel_plot, tol) * 1.6)
    ax.set_title("naive vs measured verdict", fontweight="bold", color=pal["text"])

    naive = d["naive_verdict"].upper() or "?"
    measured = d["verdict"].upper()
    lines = [f"cond-only test:  {naive}", f"NUDGE (measured):  {measured}"]
    if d["naive_is_wrong"]:
        lines.append("→ the naive test is WRONG here:")
        lines.append("  loose parameters, but tight predictions")
    elif d["verdict"] == "unidentifiable" and d["n_null_dims"] > 0:
        lines.append(f"→ {d['n_null_dims']} structural null direction(s)")
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, color=pal["text"],
            bbox={"boxstyle": "round,pad=0.4", "facecolor": pal["surface"],
                  "alpha": 0.92, "edgecolor": color})


def build(obj: Any, *, theme: str = "auto") -> RenderedFigure:
    """Build the identifiability figure (no overlay — the pipeline stamps it off the verdict)."""
    import matplotlib.pyplot as plt

    pal = apply_theme(theme)
    d = identifiability_data(obj)
    color = _color(d, pal)
    fig, (ax_spec, ax_verd) = plt.subplots(1, 2, figsize=(10.6, 4.5),
                                           gridspec_kw={"width_ratios": [1.4, 1.0]})
    _draw_spectrum(ax_spec, d, pal, color)
    _draw_verdict(ax_verd, d, pal, color)
    fig.suptitle(f"{d['label']}  —  identifiability  →  {d['verdict'].upper()}",
                 fontweight="bold", color=pal["text"], fontsize=13)
    fig.tight_layout()
    # The verdict panel carries the call → the overlay fires there on 'unidentifiable'.
    panels = [
        Panel(ax=ax_spec, call="", reason="", label=d["label"]),
        Panel(ax=ax_verd, call=d["call"], reason=d["reason"], label=d["label"]),
    ]
    return RenderedFigure(
        fig=fig, panels=panels, kind="identifiability", caption=_caption(d), data=d
    )
