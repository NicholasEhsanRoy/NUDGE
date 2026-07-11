"""Constitutive-control renderer (``NUDGE-METHOD-011``; the NUDGE-LIM-006 flip).

The mitigation for the readout-nonlinearity false positive: a constitutive control drives
the reporter at KNOWN activity doses (bypassing the circuit), anchoring the readout. A
profile over the circuit's Hill ``n`` then asks whether the observed ultrasensitivity is
BIOLOGICAL or lives in the measurement.

The flagship panel is the **n-profile flip**: WITHOUT the control the loss-vs-``n`` profile
is FLAT (you cannot even tell a switch exists — any ``n`` fits), but WITH the control the
``n=1`` (no-switch) point is REJECTED and the profile dips at the true ``n``. The second
panel shows the calibrated reporter Hill (its nonlinearity is what made the readout
deceptive). Verdict panel = the profile; hatched on ``unresolved``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nudge.viz._util import get, verdict_color
from nudge.viz.base import Panel, RenderedFigure, is_abstention
from nudge.viz.layout import freest_corner
from nudge.viz.theme import apply_theme


def _f(x: Any, d: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def _list(x: Any) -> list[float]:
    if x is None:
        return []
    return [_f(v) for v in x]


def constitutive_data(obj: Any) -> dict[str, Any]:
    """Normalise a constitutive result (``constitutive_demo`` / dict / replay) to a dict."""
    if isinstance(obj, dict) and obj.get("kind") == "constitutive":
        return obj
    cal = get(obj, "calibration", default={})
    if not isinstance(cal, dict):  # dataclass form
        cal = {
            "reporter_hill_h": get(cal, "h"), "km": get(cal, "km"),
            "vmax": get(cal, "vmax"), "base": get(cal, "base"),
            "r2": get(cal, "r2"), "is_nonlinear": get(cal, "is_nonlinear"),
        }
    gt = get(obj, "ground_truth", default=None)
    return {
        "kind": "constitutive",
        "label": str(get(obj, "label", default="circuit")),
        "call": str(get(obj, "call", default="unresolved")),
        "reason": str(get(obj, "reason", default="")),
        "asserts_biological_switch": bool(get(obj, "asserts_biological_switch",
                                              default=False)),
        "n_grid": _list(get(obj, "n_grid")),
        "loss_no_control": _list(get(obj, "loss_no_control")),
        "loss_with_control": _list(get(obj, "loss_with_control")),
        "n1_rejection": _f(get(obj, "n1_rejection")),
        "argmin_n_with_control": _f(get(obj, "argmin_n_with_control")),
        "calibration": {
            "h": _f(cal.get("reporter_hill_h", cal.get("h"))),
            "km": _f(cal.get("km")), "vmax": _f(cal.get("vmax")),
            "base": _f(cal.get("base")), "r2": _f(cal.get("r2")),
            "is_nonlinear": bool(cal.get("is_nonlinear", False)),
        },
        "ground_truth": dict(gt) if isinstance(gt, dict) else None,
    }


def _caption(d: dict[str, Any]) -> str:
    call = d["call"].upper()
    if is_abstention(d["call"]):
        return f"{d['label']} → {call} ({d['reason'][:90] or 'switch existence unresolved'})"
    if d["call"] == "biological-switch":
        return (f"{d['label']} → BIOLOGICAL-SWITCH (n=1 rejected by "
                f"{d['n1_rejection']:.1f}; argmin n≈{d['argmin_n_with_control']:.1f})")
    return f"{d['label']} → {call}"


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if not len(v) or not np.isfinite(v).any():
        return v
    lo = np.nanmin(v)
    return v - lo


def draw_profile(ax: Any, d: dict[str, Any], pal: dict[str, str], color: str, *,
                 reveal_with: float = 1.0) -> None:
    """Draw the n-profile flip. ``reveal_with`` in [0,1] partially draws the WITH-control
    curve (for the animation); 1.0 draws it fully."""
    ng = np.asarray(d["n_grid"], dtype=float)
    if not len(ng):
        ax.text(0.5, 0.5, "no n-profile", transform=ax.transAxes, ha="center",
                va="center", color=pal["muted"])
        return
    lno = _norm(d["loss_no_control"])
    lwith = _norm(d["loss_with_control"])
    ax.plot(ng, lno, color=pal["muted"], lw=2.0, marker="o", ms=4, zorder=3,
            label="WITHOUT control (flat → can't tell)")
    k = max(1, int(round(reveal_with * len(ng))))
    ax.plot(ng[:k], lwith[:k], color=color, lw=2.4, marker="o", ms=4, zorder=4,
            label="WITH control (n=1 rejected)")
    # Mark the n=1 (no-switch) hypothesis.
    ax.axvline(1.0, ls=":", color=pal["text"], lw=1.2, zorder=2)
    ax.annotate("n=1\n(no switch)", xy=(1.0, 0.0), xycoords=("data", "axes fraction"),
                xytext=(4, 6), textcoords="offset points", fontsize=7.5,
                color=pal["text"], ha="left", va="bottom")
    am = d["argmin_n_with_control"]
    if np.isfinite(am) and reveal_with >= 0.999:
        ax.axvline(am, ls="--", color=color, lw=1.4, alpha=0.8, zorder=2)
    ax.set_xlabel("circuit Hill n (hypothesis)")
    ax.set_ylabel("profile loss  (Δ from min)")
    ax.set_title("n-profile: does a switch exist?", fontweight="bold", color=pal["text"])
    # Keep the legend clear of the reserved top band (the abstain banner lives there).
    ax.legend(loc=freest_corner(ax, avoid_top=True), fontsize=8, framealpha=0.9)


def _draw_calibration(ax: Any, d: dict[str, Any], pal: dict[str, str]) -> None:
    c = d["calibration"]
    h, km, vmax, base = c["h"], c["km"], c["vmax"], c["base"]
    if not all(np.isfinite([h, km, vmax, base])):
        ax.text(0.5, 0.5, "calibration unavailable", transform=ax.transAxes,
                ha="center", va="center", color=pal["muted"])
        ax.set_title("reporter calibration", fontweight="bold", color=pal["text"])
        return
    act = np.linspace(0.0, max(2.0 * km, 1.0), 200)
    resp = base + vmax * act**h / (km**h + act**h)
    ax.plot(act, resp, color=pal["ceiling"], lw=2.2, zorder=3)
    lin = base + (vmax) * (act / max(2.0 * km, 1e-9))
    ax.plot(act, np.clip(lin, None, base + vmax), color=pal["muted"], ls="--", lw=1.4,
            zorder=2, label="linear ref")
    ax.set_xlabel("driven activity (known dose)")
    ax.set_ylabel("reporter readout")
    tag = "NONLINEAR" if c["is_nonlinear"] else "≈linear"
    ax.set_title(f"reporter calibration  (Hill h={h:.1f}, {tag})", fontweight="bold",
                 color=pal["text"])
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


def build(obj: Any, *, theme: str = "auto") -> RenderedFigure:
    """Build the constitutive-flip figure (no overlay — the pipeline stamps it off call)."""
    import matplotlib.pyplot as plt

    pal = apply_theme(theme)
    d = constitutive_data(obj)
    color = verdict_color(d["call"], pal)
    fig, (ax_prof, ax_cal) = plt.subplots(1, 2, figsize=(11.0, 4.5))
    draw_profile(ax_prof, d, pal, color)
    _draw_calibration(ax_cal, d, pal)
    fig.suptitle(f"{d['label']}  →  {d['call'].upper()}", fontweight="bold",
                 color=pal["text"], fontsize=13)
    fig.tight_layout()
    panels = [
        Panel(ax=ax_prof, call=d["call"], reason=d["reason"], label=d["label"]),
        Panel(ax=ax_cal, call="", reason="", label=d["label"]),
    ]
    return RenderedFigure(
        fig=fig, panels=panels, kind="constitutive", caption=_caption(d), data=d
    )
