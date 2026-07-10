"""NUDGE viz house style — a SEMANTIC mechanism palette, light/dark, headless Agg.

The colours are the ``dataviz`` skill's validated default palette (colorblind-safe;
see ``design/VISUALIZATION_DESIGN.md`` §2.4). The mechanism palette is *semantic*, not
decorative: the same hue always means the same mechanism (threshold ``K`` / gain ``n`` /
ceiling ``v_max``) across every renderer, and the ``abstain`` slot (muted grey + hatch)
is visually distinct from any positive call — the honesty guarantee is load-bearing, so
it must be legible.

matplotlib is imported **lazily** here (inside :func:`apply_theme`), matching how the
rest of the codebase defers ``anndata`` / ``scanpy`` — importing ``nudge.viz`` never
imports matplotlib until a figure is actually drawn.
"""

from __future__ import annotations

from typing import Any

# The dataviz-validated categorical hues, keyed by the mechanism they mean. Two
# selected columns (light / dark) — the SAME hues stepped for each surface.
_PALETTE = {
    "light": {
        "threshold": "#2a78d6",  # blue   — K
        "gain": "#eb6834",  # orange — n
        "ceiling": "#008300",  # green  — v_max
        "graded": "#2a78d6",  # blue   — a resolved, graded response
        "switch": "#eb6834",  # orange — a resolved, ultrasensitive gain
        "no_effect": "#898781",  # muted grey
        "abstain": "#898781",  # muted grey (+ hatch) — visually distinct from a call
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#898781",
        "grid": "#e1e0d9",
    },
    "dark": {
        "threshold": "#3987e5",
        "gain": "#d95926",
        "ceiling": "#008300",
        "graded": "#3987e5",
        "switch": "#d95926",
        "no_effect": "#898781",
        "abstain": "#898781",
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#898781",
        "grid": "#2c2c2a",
    },
}


def resolve_theme(theme: str) -> str:
    """Resolve ``"auto"`` to a concrete ``"light"`` / ``"dark"`` (auto → light).

    Saved PNGs and notebook-embedded figures default to the light surface (the safe,
    print-friendly default); ``theme="dark"`` opts in to the dark surface.
    """
    t = (theme or "auto").strip().lower()
    if t in ("light", "dark"):
        return t
    return "light"


def palette(theme: str = "auto") -> dict[str, str]:
    """The resolved semantic mechanism palette for ``theme``."""
    return dict(_PALETTE[resolve_theme(theme)])


def _select_headless_backend() -> None:
    """Force the headless-safe ``Agg`` backend UNLESS an inline/interactive one is live.

    In a notebook ``%matplotlib inline`` has already selected the inline backend; we must
    not clobber it (that would break embedded output). Everywhere else — a plain script, a
    CLI run, CI — we force ``Agg`` so a figure renders with no ``$DISPLAY``.
    """
    import contextlib

    import matplotlib

    backend = matplotlib.get_backend().lower()
    if any(tag in backend for tag in ("inline", "nbagg", "ipympl", "widget")):
        return
    # pragma: no cover - backend may already be fixed; savefig still works headless
    with contextlib.suppress(Exception):
        matplotlib.use("Agg", force=True)


def apply_theme(theme: str = "auto") -> dict[str, str]:
    """Select the headless backend, set NUDGE rcParams, and return the palette.

    Called at the top of every renderer (never at import time). Returns the resolved
    semantic palette so a renderer draws each mechanism in its canonical colour.
    """
    _select_headless_backend()
    import matplotlib.pyplot as plt

    pal = palette(theme)
    plt.rcParams.update(
        {
            "figure.facecolor": pal["surface"],
            "axes.facecolor": pal["surface"],
            "savefig.facecolor": pal["surface"],
            "axes.edgecolor": pal["muted"],
            "axes.labelcolor": pal["text"],
            "text.color": pal["text"],
            "xtick.color": pal["muted"],
            "ytick.color": pal["muted"],
            "axes.grid": True,
            "grid.color": pal["grid"],
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "figure.dpi": 120,
        }
    )
    return pal


def call_color(call: str, pal: dict[str, str]) -> Any:
    """Map a verdict to its semantic colour (abstentions → the grey abstain slot)."""
    key = (call or "").strip().lower()
    if key == "switch":
        return pal["switch"]
    if key == "graded":
        return pal["graded"]
    return pal["abstain"]
