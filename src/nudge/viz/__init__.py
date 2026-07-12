"""``nudge.viz`` — an opt-in, provenance-carrying figure layer for NUDGE results.

One :func:`render` surface keyed off the existing frozen result dataclasses (and their
``*_to_dict()`` dicts) turns an honest *result* into an honest *picture*. It is purely
additive: it only READS result objects, never re-attributes, and never imports the fit
engine. matplotlib is an optional ``[viz]`` extra, imported lazily — importing this
package without it is fine until you actually draw.

The load-bearing guarantee: the **abstention overlay is applied by the render pipeline
itself**, off each result's OWN verdict (:func:`figure` → :func:`_apply_honesty`), so a
per-mechanism renderer cannot draw a positive call where the result abstained. Every
figure also ships a standalone ``fig.py`` + a data sidecar that replays the exact plot
from the fit's output (no re-fit) — the Claude Science provenance grain.

This is the first slice: the flagship dose-response dual panel. Other result types are
designed (``design/VISUALIZATION_DESIGN.md`` §2.2) and land in later slices.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from nudge.viz.base import (
    ABSTAIN_CALLS,
    FigureResult,
    Panel,
    RenderedFigure,
    abstain_overlay,
    is_abstention,
)

__all__ = [
    "ABSTAIN_CALLS",
    "FigureResult",
    "Panel",
    "RenderedFigure",
    "abstain_overlay",
    "figure",
    "is_abstention",
    "plot_dose_response",
    "render",
]

_INSTALL_HINT = (
    "nudge.viz needs matplotlib. Install the optional extra:\n"
    "    uv pip install -e '.[viz]'   (or '.[dev]', which includes viz)"
)

# Inline PNG size cap for the MCP path (base64 chars ≈ 1.5 MB).
_PNG_B64_CAP = 1_500_000


def _require_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(_INSTALL_HINT) from exc


# The renderer registry — ``kind`` → the ``module`` exposing ``build(result, *, theme)``.
# Each renderer builds its overlay-FREE figure with per-panel verdicts; the pipeline
# (:func:`_apply_honesty`) stamps the abstention overlay off each panel's own ``call``, so
# a new renderer inherits the honesty guarantee for free. Imports are lazy (per-kind) so
# ``import nudge.viz`` stays cheap and matplotlib-optional.
_RENDERERS: dict[str, str] = {
    "dose_response": "nudge.viz.dose_response",
    "attribution": "nudge.viz.attribution",  # the core AttributionReport across ops
    "cross_modality": "nudge.viz.cross_modality",  # reuses the Hill panel (fold-change axis)
    "epistasis": "nudge.viz.epistasis",
    "differential": "nudge.viz.differential",
    "multi_reporter": "nudge.viz.multi_reporter",
    "temporal": "nudge.viz.temporal",
    "aggregation": "nudge.viz.aggregation",
    "constitutive": "nudge.viz.constitutive",
    "diagnose": "nudge.viz.diagnose",
    "design": "nudge.viz.design",
    "oed": "nudge.viz.oed",
    "identifiability": "nudge.viz.identifiability",  # FIM sloppiness spectrum
    "robustness": "nudge.viz.robustness",
}


def _dispatch_build(result: Any, ctx: dict[str, Any], theme: str) -> RenderedFigure:
    """Build the (overlay-free) figure for ``result``'s kind via the renderer registry."""
    kind = ctx.get("kind")
    if kind is None and isinstance(result, dict):
        kind = result.get("kind")
    if kind in (None, "dose_response"):
        entries = _dose_entries(result, ctx)
        return _build_from_kind("dose_response", entries, theme)
    if kind in _RENDERERS:
        return _build_from_kind(kind, result, theme)
    raise ValueError(
        f"nudge.viz has no renderer for kind={kind!r} "
        f"(known: {sorted(_RENDERERS)})"
    )


def _build_from_kind(kind: str, entries: Any, theme: str) -> RenderedFigure:
    import importlib

    module_name = _RENDERERS.get(kind)
    if module_name is None:
        raise ValueError(f"unknown figure kind {kind!r}")
    build = importlib.import_module(module_name).build
    return build(entries, theme=theme)


def _dose_entries(result: Any, ctx: dict[str, Any]) -> Any:
    """Fold single-result + ``dose``/``response`` context into the dose-response entries."""
    if isinstance(result, dict) and result.get("kind") == "dose_response":
        return result
    if isinstance(result, (list, tuple)):
        return result
    dose = ctx.get("dose")
    response = ctx.get("response")
    label = ctx.get("label")
    if dose is not None or response is not None or label is not None:
        return (label if label is not None else "", result, dose, response)
    return result


def _apply_honesty(rf: RenderedFigure, theme: str) -> RenderedFigure:
    """The load-bearing step: stamp the abstention overlay on every abstaining panel.

    Runs in the render pipeline (not the per-mechanism renderer), off each panel's own
    ``call`` — so a renderer cannot forget it and the picture can never claim more than
    the verdict.
    """
    from nudge.viz.theme import palette

    pal = palette(theme)
    for p in rf.panels:
        if p.abstained:
            abstain_overlay(
                p.ax, p.call, p.reason, one_sided=p.one_sided, palette=pal
            )
    return rf


def figure(result: Any, *, theme: str = "auto", **ctx: Any) -> RenderedFigure:
    """Build a figure + apply the honesty overlay, WITHOUT saving (notebooks / tests).

    Accepts a frozen result dataclass (with ``dose=`` / ``response=`` for dose-response),
    its ``*_to_dict()`` dict, a list of ``(label, result, dose, response)`` entries for a
    multi-panel figure, or a replay figure-data dict. Returns the live
    :class:`~nudge.viz.base.RenderedFigure` (its ``.fig`` displays inline; ``.panels``
    carry the verdicts; ``.abstained`` reports whether the overlay fired).
    """
    _require_matplotlib()
    rf = _dispatch_build(result, ctx, theme)
    return _apply_honesty(rf, theme)


def render(
    result: Any,
    out: str | None = None,
    *,
    emit_code: bool = True,
    theme: str = "auto",
    self_contained: bool = False,
    animate: bool = False,
    inline_png: bool = False,
    cli_call: str | None = None,
    **ctx: Any,
) -> FigureResult:
    """Render a NUDGE result to a figure — the one-call dispatcher (design §2.2).

    Dispatches on ``result``'s type (or a figure-data dict's ``kind``), applies the
    abstention overlay off the result's OWN verdict, writes ``out`` (PNG) if given, and —
    unless ``emit_code=False`` — the regenerating ``fig.py`` + data sidecar beside it.
    Returns a :class:`~nudge.viz.base.FigureResult`; ``abstained`` is True when any panel
    abstained. ``animate`` is accepted for API stability but not implemented in this slice.
    """
    if animate:
        _require_matplotlib()
        if out is None:
            raise ValueError("animate=True requires an out= path (a .gif)")
        from nudge.viz.animate import render_animation

        kind = ctx.get("kind")
        if kind is None and isinstance(result, dict):
            kind = result.get("kind")
        return render_animation(
            result, out, kind=kind, theme=theme,
            frames=int(ctx.get("anim_frames", 28)),
            fps=int(ctx.get("anim_fps", 8)),
            emit_code=emit_code, self_contained=self_contained, cli_call=cli_call,
        )
    rf = figure(result, theme=theme, **ctx)

    path: str | None = None
    code_path: str | None = None
    data_path: str | None = None
    png_b64: str | None = None

    if out is not None:
        import matplotlib.pyplot as plt

        outp = Path(out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        rf.fig.savefig(
            outp, dpi=120, bbox_inches="tight", metadata={"Software": "NUDGE viz"}
        )
        path = str(outp)

        if emit_code:
            from nudge.viz.provenance import emit_code as _emit

            code_path, data_path = _emit(
                rf.data, path, self_contained=self_contained, cli_call=cli_call
            )
        if inline_png:
            raw = base64.b64encode(outp.read_bytes()).decode("ascii")
            png_b64 = raw if len(raw) <= _PNG_B64_CAP else None
        plt.close(rf.fig)

    return FigureResult(
        path=path,
        code_path=code_path,
        data_path=data_path,
        png_base64=png_b64,
        caption=rf.caption,
        abstained=rf.abstained,
        kind=rf.kind,
    )


def plot_dose_response(
    result: Any,
    dose: Any = None,
    response: Any = None,
    *,
    out: str | None = None,
    label: str | None = None,
    emit_code: bool = True,
    theme: str = "auto",
    self_contained: bool = False,
    **ctx: Any,
) -> FigureResult:
    """Convenience entry for the dose-response figure (single or ``(label, res, ...)`` list).

    Thin wrapper over :func:`render` with the kind pinned. Pass a single result +
    ``dose``/``response`` for one panel, or a list of ``(label, result, dose, response)``
    tuples for the flagship dual panel.
    """
    if dose is not None or response is not None or label is not None:
        ctx = {**ctx, "dose": dose, "response": response, "label": label}
    return render(
        result,
        out,
        emit_code=emit_code,
        theme=theme,
        self_contained=self_contained,
        kind="dose_response",
        **ctx,
    )
