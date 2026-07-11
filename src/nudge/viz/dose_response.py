"""The flagship dose-response renderer (``design`` §5.1).

Draws one panel per dose-response result — a Hill fit + guide-dose scatter + the
threshold marker — and lays several side by side for the flagship OCT4 (resolved
``switch``) vs NANOG (honest ``unresolved`` abstention) dual panel. The Hill curve is
drawn from the fit's OUTPUT with a pure-numpy primitive (no JAX at plot time); the
abstention overlay is applied by the render pipeline, not here.

Honesty grammar: a resolved threshold (``spans_inflection``) is a dashed vertical line
at ``K``; an *unresolved* one (``K`` past the dose range — one arm of a sigmoid, gain
unidentifiable) is drawn as an **open-ended arrow**, never a point estimate or a closed
error bar.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nudge.viz.base import Panel, RenderedFigure, is_abstention
from nudge.viz.layout import freest_corner, place_label, reserve_top_band
from nudge.viz.theme import apply_theme, call_color

_DEFAULT_XLABEL = "perturbation dose"
_DEFAULT_YLABEL = "response (rel. control)"


def _hill(direction: str, d: Any, floor: float, amp: float, k: float, n: float) -> Any:
    d = np.maximum(np.asarray(d, dtype=float), 1e-9)
    frac = d**n / (k**n + d**n)
    if direction == "repress":
        frac = k**n / (k**n + d**n)
    return floor + amp * frac


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Read an attribute or dict key by any of ``names`` (dataclass ⇄ dict duality)."""
    for name in names:
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
        elif hasattr(obj, name):
            return getattr(obj, name)
    return default


def _coerce_panel(
    obj: Any,
    *,
    dose: Any = None,
    response: Any = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Normalise a dose-response result (dataclass / ``*_to_dict`` / panel dict) to a panel.

    Accepts:
    - a :class:`~nudge.inference.dose_response.DoseResponseResult` (or ``DoseResponseFit``)
      + ``dose`` / ``response`` arrays;
    - the ``service.dose_response_to_dict`` form (draws the curve; scatter only if points
      are carried);
    - an already-canonical panel dict (the ``fig.data.json`` replay path).
    """
    if isinstance(obj, dict) and "n" in obj and "k_threshold" in obj and "call" in obj:
        # Already a canonical viz panel dict (the replay path).
        panel = dict(obj)
        panel.setdefault("label", label or panel.get("label", ""))
        return panel

    fit = _get(obj, "fit", default=obj)  # DoseResponseResult.fit, else obj itself
    call = str(_get(obj, "call", default="unresolved"))
    reason = str(_get(obj, "reason", default=""))
    direction = str(_get(fit, "direction", default="repress"))
    n = float(_get(fit, "n", "n_apparent_gain", default=1.0))
    k = float(_get(fit, "k_threshold", "K_threshold", default=1.0))
    amp = float(_get(fit, "amp", default=1.0))
    floor = float(_get(fit, "floor", default=0.0))
    r2 = float(_get(fit, "r2", default=float("nan")))
    ci_n = _get(fit, "ci_n", default=(float("nan"), float("nan")))
    ci_k = _get(fit, "ci_k", "ci_K", default=(float("nan"), float("nan")))
    spans = bool(_get(fit, "spans_inflection", default=True))

    dr = _get(fit, "dose_range", default=None)
    dmin = _get(fit, "dose_min", default=None)
    dmax = _get(fit, "dose_max", default=None)
    if dr is not None:
        dmin, dmax = float(dr[0]), float(dr[1])

    d_list = None if dose is None else [float(x) for x in np.asarray(dose).ravel()]
    r_list = (
        None if response is None else [float(x) for x in np.asarray(response).ravel()]
    )
    if d_list is not None:
        dmin = float(min(d_list)) if dmin is None else dmin
        dmax = float(max(d_list)) if dmax is None else dmax

    return {
        "label": label if label is not None else str(_get(obj, "label", default="")),
        "call": call,
        "reason": reason,
        "direction": direction,
        "n": n,
        "k_threshold": k,
        "amp": amp,
        "floor": floor,
        "r2": r2,
        "ci_n": [float(ci_n[0]), float(ci_n[1])],
        "ci_k": [float(ci_k[0]), float(ci_k[1])],
        "spans_inflection": spans,
        "dose_min": None if dmin is None else float(dmin),
        "dose_max": None if dmax is None else float(dmax),
        "dose": d_list,
        "response": r_list,
    }


def dose_response_data(entries: Any, *, xlabel: str | None = None,
                       ylabel: str | None = None) -> dict[str, Any]:
    """Build the canonical figure-data dict from one or more dose-response entries.

    ``entries`` is a single result, a ``(label, result, dose, response)`` tuple, a list of
    such tuples / results / panel dicts, or an already-built figure-data dict. The output
    (``{"kind": "dose_response", "panels": [...]}``) is what gets written to
    ``fig.data.json`` and replayed by :func:`nudge.viz.render`.
    """
    if isinstance(entries, dict) and entries.get("kind") == "dose_response":
        return entries

    items = entries if isinstance(entries, (list, tuple)) else [entries]
    # A bare (label, result, dose, response) tuple must not be split into panels.
    if (
        isinstance(entries, tuple)
        and len(entries) in (2, 3, 4)
        and not isinstance(entries[0], (list, tuple, dict))
    ):
        items = [entries]

    panels: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, tuple):
            label = item[0]
            result = item[1]
            dose = item[2] if len(item) > 2 else None
            response = item[3] if len(item) > 3 else None
            panels.append(
                _coerce_panel(result, dose=dose, response=response, label=str(label))
            )
        else:
            panels.append(_coerce_panel(item))
    return {
        "kind": "dose_response",
        "panels": panels,
        "xlabel": xlabel or _DEFAULT_XLABEL,
        "ylabel": ylabel or _DEFAULT_YLABEL,
    }


def _short_reason(panel: dict[str, Any]) -> str:
    call = panel["call"].lower()
    if not panel["spans_inflection"]:
        return "K past max dose → gain unidentifiable"
    if call == "no-effect":
        return "flat within noise"
    return "unresolved"


def _caption(data: dict[str, Any]) -> str:
    parts = []
    for p in data["panels"]:
        lbl = p["label"] or "curve"
        call = p["call"].upper()
        if is_abstention(p["call"]):
            parts.append(f"{lbl} → {call} ({_short_reason(p)})")
        else:
            r2 = p["r2"]
            r2s = "" if not np.isfinite(r2) else f", R²={r2:.2f}"
            parts.append(f"{lbl} → {call} (n≈{p['n']:.1f}{r2s})")
    return " · ".join(parts)


def _draw_panel(ax: Any, p: dict[str, Any], pal: dict[str, str]) -> None:
    color = call_color(p["call"], pal)
    direction = p["direction"]
    dmax = p["dose_max"] if p["dose_max"] is not None else max(p["k_threshold"], 1.0)
    xmax = max(dmax, p["k_threshold"]) * 1.05
    xs = np.linspace(0.0, xmax, 200)
    ys = _hill(direction, xs, p["floor"], p["amp"], p["k_threshold"], p["n"])
    ax.plot(xs, ys, color=color, lw=2.0, zorder=3, label=f"Hill fit (n={p['n']:.1f})")

    if p["dose"] is not None and p["response"] is not None:
        ax.scatter(
            p["dose"], p["response"], s=42, color=color, zorder=4,
            edgecolors=pal["surface"], linewidths=0.8, label="guide-dose points",
        )
    if p["dose_max"] is not None:
        ax.axvline(p["dose_max"], ls=":", color=pal["muted"], lw=1.0, zorder=2)

    abst = is_abstention(p["call"])
    if p["spans_inflection"] and not abst:
        ax.axvline(
            p["k_threshold"], ls="--", color=color, lw=1.2, alpha=0.8, zorder=2,
        )
        # Reserve a clear top band and drop the K-label there, anchored to the K line but
        # clamped inside the frame — so it never lands on the curve or under the legend.
        # (Only for a RESOLVED call: an abstained panel gets the banner in this band and
        # must not assert a confident K.)
        band_y = reserve_top_band(ax, band=0.18)
        xmin_ax, xmax_ax = ax.get_xlim()
        k_frac = (p["k_threshold"] - xmin_ax) / max(xmax_ax - xmin_ax, 1e-9)
        ha = "left" if k_frac < 0.5 else "right"
        ax.annotate(
            f"K={p['k_threshold']:.2f} (inside range)",
            xy=(p["k_threshold"], band_y),
            xytext=(5 if ha == "left" else -5, 0),
            textcoords="offset points",
            xycoords=("data", "axes fraction"),
            fontsize=8, color=color, ha=ha, va="center", zorder=7,
        )
    elif not p["spans_inflection"]:
        # One-sided lower bound: an OPEN-ENDED arrow, never a point estimate. K sits
        # past the observed dose range, so gain is unidentifiable. The arrow sits mid-axes
        # (the abstain banner will occupy the reserved top band); the label is placed in
        # the freest region so it clears both the arrow and the curve.
        x0 = dmax * 0.55
        ax.annotate(
            "", xy=(xmax, 0.5), xytext=(x0, 0.5),
            xycoords=("data", "axes fraction"),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.6,
                        "linestyle": "--"},
        )
        place_label(
            ax, "K past max dose →\ngain unidentifiable",
            [(0.06, 0.30), (0.06, 0.62), (0.40, 0.30)],
            color=color, ha="left", va="center",
        )

    lbl = p["label"] or "curve"
    ax.set_title(f"{lbl}  →  {p['call'].upper()}", fontweight="bold", color=pal["text"])
    ax.legend(loc=freest_corner(ax, avoid_top=True), fontsize=8, framealpha=0.9)


def build(data_or_entries: Any, *, theme: str = "auto") -> RenderedFigure:
    """Build the dose-response figure (NO overlay — the render pipeline applies that).

    Returns a :class:`~nudge.viz.base.RenderedFigure` whose panels carry each verdict, so
    the pipeline can stamp the abstention overlay off the same ``call`` the text prints.
    """
    import matplotlib.pyplot as plt

    pal = apply_theme(theme)
    data = dose_response_data(data_or_entries)
    panels_data = data["panels"]
    ncols = max(len(panels_data), 1)
    fig, axes = plt.subplots(1, ncols, figsize=(5.4 * ncols, 4.2), squeeze=False)
    row = axes[0]

    panels: list[Panel] = []
    for ax, p in zip(row, panels_data, strict=False):
        _draw_panel(ax, p, pal)
        ax.set_xlabel(data.get("xlabel", _DEFAULT_XLABEL))
        ax.set_ylabel(data.get("ylabel", _DEFAULT_YLABEL))
        panels.append(
            Panel(
                ax=ax,
                call=p["call"],
                reason=p["reason"],
                label=p["label"],
                one_sided=not p["spans_inflection"],
            )
        )
    fig.tight_layout()
    return RenderedFigure(
        fig=fig,
        panels=panels,
        kind="dose_response",
        caption=_caption(data),
        data=data,
    )
