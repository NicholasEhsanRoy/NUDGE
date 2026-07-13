"""Shared viz primitives — ``FigureResult``, the panel model, and the ABSTENTION overlay.

The overlay is the load-bearing honesty mechanism (``design/VISUALIZATION_DESIGN.md``
§2.5): it is applied by the render pipeline itself, off each panel's OWN verdict, so a
per-mechanism renderer *cannot forget it*. A positive call is never drawn where the
result abstained, because the overlay is keyed off the same ``call`` field the CLI / MCP
print — the picture and the text can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Verdict classes that are abstentions — the overlay fires on any of these. Keyed off
#: the SAME ``call`` field the CLI/MCP report, across every result type.
ABSTAIN_CALLS = frozenset(
    {
        "unresolved",
        "no-effect",
        "off-model",
        "technical-artifact",
        "no-difference",
        "not-bistable",
        "abstention",
        "abstain",
        # cross-modality variant abstentions (NUDGE-METHOD-002): the panel cannot localise
        # a knob for this variant, so it must read as an abstention, not a positive call.
        "non-responsive",
        "inconclusive",
        # single-condition covariance attribution: gain and threshold are the measured
        # confound NUDGE deliberately abstains *between* (never a bare gain/threshold from
        # one snapshot — nudge.inference.lyapunov._decide_lyapunov), so the chip and the
        # picture must read as an abstention.
        "gain_or_threshold",
        # sloppiness / identifiability (nudge.inference.sloppiness): an unrecoverable
        # parameter direction is an abstention, not a positive call.
        "unidentifiable",
    }
)


def is_abstention(call: str | None) -> bool:
    """True when ``call`` is an abstention verdict (the overlay-firing set)."""
    return (call or "").strip().lower() in ABSTAIN_CALLS


@dataclass(frozen=True)
class FigureResult:
    """The record :func:`nudge.viz.render` returns (see design §2.2).

    ``abstained`` is True when the overlay fired on any panel — the figure carries an
    abstention and a caller (CLI / MCP / notebook) can surface it without re-reading the
    picture. ``caption`` always carries the honest verdict + reason, so a figure lifted
    out of context still states its caveat.
    """

    path: str | None
    code_path: str | None
    data_path: str | None
    png_base64: str | None
    caption: str
    abstained: bool
    kind: str


@dataclass
class Panel:
    """One axis + the verdict that governs whether it gets the abstention overlay."""

    ax: Any
    call: str
    reason: str
    label: str = ""
    one_sided: bool = False

    @property
    def abstained(self) -> bool:
        return is_abstention(self.call)


@dataclass
class RenderedFigure:
    """A built figure + its panels + the provenance data — pre-save.

    Returned by :func:`nudge.viz.figure` (for notebooks / composition / tests); consumed
    by :func:`nudge.viz.render` (which saves + emits provenance). ``fig`` is a live
    matplotlib ``Figure``; ``data`` is the figure-data dict written to ``fig.data.json``.
    """

    fig: Any
    panels: list[Panel]
    kind: str
    caption: str
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        return any(p.abstained for p in self.panels)


@dataclass
class AnimationSpec:
    """The record an animator's ``build_animation`` returns — the animation battery contract.

    Each animated ``kind`` exposes ``build_animation(result_or_data, *, theme, frames) ->
    AnimationSpec`` in its own renderer module (mirroring the static ``build``), so
    :func:`nudge.viz.animate.render_animation` is generic. ``anim`` is a live
    ``FuncAnimation``; ``data`` is the FULL, serialisable frame-sequence spec written to the
    sidecar so the standalone ``fig.py`` replays the animation exactly (no re-fit). The
    load-bearing honesty is the same as the static path: ``abstained`` is stamped off the
    result's OWN verdict, and the per-frame abstention overlay fires when it is True.
    """

    fig: Any
    anim: Any
    caption: str
    abstained: bool
    data: dict[str, Any] = field(default_factory=dict)


def abstain_overlay(
    ax: Any,
    verdict: str,
    reason: str = "",
    *,
    one_sided: bool = False,
    palette: dict[str, str] | None = None,
) -> Any:
    """Stamp the abstention marker on ``ax`` — a grey hatch + an ``I CAN'T TELL`` banner.

    This is applied by the render pipeline (never by a per-mechanism renderer). It greys
    and hatches the plot region and stamps the verdict so the panel cannot be mistaken
    for a confident call. ``one_sided`` softens the banner to the "at least this far /
    unidentifiable past here" grammar (the open-ended-bound case). Sets a
    ``_nudge_abstained`` marker on the axis so the honesty test can detect the overlay
    without OCR.
    """
    import matplotlib.patches as mpatches

    from nudge.viz.layout import reserve_top_band

    pal = palette or {}
    grey = pal.get("abstain", "#898781")
    surface = pal.get("surface", "#fcfcfb")

    patch = mpatches.Rectangle(
        (0.0, 0.0),
        1.0,
        1.0,
        transform=ax.transAxes,
        zorder=5,
        facecolor=grey,
        alpha=0.16,
        hatch="////",
        edgecolor=grey,
        linewidth=0.0,
    )
    ax.add_patch(patch)

    # Clear a top band of data so the banner never sits ON the data points, then anchor
    # the banner in the centre of that reserved band (collision-aware; see viz.layout).
    band_y = reserve_top_band(ax, band=0.24)
    head = "CAN'T TELL — ONE-SIDED BOUND" if one_sided else "I CAN'T TELL"
    banner = f"{head}\n{verdict.upper()}"
    ax.text(
        0.5,
        band_y,
        banner,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=grey,
        zorder=6,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": surface,
            "alpha": 0.9,
            "edgecolor": grey,
        },
    )
    # Machine-detectable marker (the honesty test asserts this fired).
    ax._nudge_abstained = True  # type: ignore[attr-defined]
    ax._nudge_one_sided = one_sided  # type: ignore[attr-defined]
    return patch
