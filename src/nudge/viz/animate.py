"""Animation engine (design §5.2) — ``FuncAnimation`` → Pillow GIF, no external deps.

The **animation battery**: each animated result ``kind`` exposes a
``build_animation(result_or_data, *, theme, frames) -> AnimationSpec`` in its own renderer
module (mirroring the static ``build``), and this module is the generic dispatcher +
GIF/provenance writer over them. So a new animator is purely additive — an
``build_animation`` in its module + a one-line ``_ANIMATORS`` entry — and inherits the
honesty guarantee for free.

Honesty is preserved exactly as in the static path: the animation is built from the SAME
frozen result the static figure uses (its ``*_animation_data`` normaliser reads the result,
never re-fits), and if the fit ABSTAINS the abstention overlay is stamped on every frame off
the result's OWN verdict — the picture can never claim more than the text. Every animation
also ships the standalone ``fig.py`` + a data sidecar carrying the FULL frame-sequence spec,
so it replays the exact GIF with no re-fit (the Claude Science provenance grain). GIFs are
written with matplotlib's bundled ``PillowWriter`` (Pillow is already a matplotlib
dependency); no ffmpeg / mp4.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from nudge.viz.base import AnimationSpec, FigureResult

# kind → the module exposing ``build_animation`` (mirrors the static ``_RENDERERS``).
# The animation battery (design §5.2): each entry is an additive animator, registered as it
# lands. Kinds with no natural frame variable (differential / diagnose / attribution /
# epistasis) are deliberately NOT animated; cross_modality reuses the dose-response animation.
_ANIMATORS: dict[str, str] = {
    "constitutive": "nudge.viz.constitutive",
}


def _resolve_kind(result_or_data: Any, kind: str | None) -> str:
    if kind is None and isinstance(result_or_data, dict):
        kind = result_or_data.get("kind")
    if kind not in _ANIMATORS:
        raise ValueError(
            f"nudge.viz has no animation for kind={kind!r} "
            f"(animated kinds: {sorted(_ANIMATORS)})"
        )
    return kind


def build_animation(
    result_or_data: Any, *, kind: str | None = None, theme: str = "auto", frames: int = 28
) -> AnimationSpec:
    """Build the live :class:`AnimationSpec` for ``result_or_data`` (no save).

    Dispatches on ``kind`` (or the data dict's ``kind``) through the animator registry.
    Used by :func:`render_animation` and by notebooks/tests that want the live ``anim``.
    """
    kind = _resolve_kind(result_or_data, kind)
    mod = importlib.import_module(_ANIMATORS[kind])
    spec = mod.build_animation(result_or_data, theme=theme, frames=frames)
    if not isinstance(spec, AnimationSpec):  # pragma: no cover - contract guard
        raise TypeError(
            f"{_ANIMATORS[kind]}.build_animation must return an AnimationSpec, "
            f"got {type(spec).__name__}"
        )
    return spec


def render_animation(
    result_or_data: Any,
    out: str,
    *,
    kind: str | None = None,
    theme: str = "auto",
    frames: int = 28,
    fps: int = 8,
    emit_code: bool = True,
    self_contained: bool = False,
    cli_call: str | None = None,
) -> FigureResult:
    """Render a NUDGE result to an animated GIF (the ``render(..., animate=True)`` path).

    Dispatches on ``kind`` through the animator battery (:data:`_ANIMATORS`). Writes ``out``
    (a ``.gif``) via ``PillowWriter`` and, unless ``emit_code=False``, the regenerating
    ``fig.py`` + data sidecar (the provenance grain). The abstention overlay is applied
    per-frame off the fit's own verdict inside each animator.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    spec = build_animation(result_or_data, kind=kind, theme=theme, frames=frames)
    resolved_kind = _resolve_kind(result_or_data, kind)

    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    spec.anim.save(str(outp), writer=PillowWriter(fps=fps))
    plt.close(spec.fig)

    code_path: str | None = None
    data_path: str | None = None
    if emit_code:
        code_path, data_path = _emit_animation_code(
            spec.data, str(outp), fps=fps, frames=frames,
            self_contained=self_contained, cli_call=cli_call,
        )

    return FigureResult(
        path=str(outp), code_path=code_path, data_path=data_path, png_base64=None,
        caption=spec.caption, abstained=spec.abstained, kind=resolved_kind,
    )


def _emit_animation_code(
    data: dict[str, Any], gif_path: str, *, fps: int, frames: int,
    self_contained: bool, cli_call: str | None,
) -> tuple[str, str | None]:
    """Emit a standalone ``fig.py`` (+ data sidecar) that regenerates the GIF (no re-fit).

    The sidecar carries the FULL frame-sequence spec (``data``), so replay recomputes
    nothing — it re-runs the SAME animator over the fit's OUTPUT, and the abstention overlay
    still fires where the fit abstained.
    """
    import base64
    import json
    from datetime import datetime, timezone

    gif = Path(gif_path)
    stem = gif.stem
    code_path = gif.with_name(f"{stem}.py")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = (
        f"# Generated by NUDGE viz — regenerates {gif.name} exactly (no re-fit).\n"
        f"# {stamp}" + (f"  ·  call: {cli_call}" if cli_call else "") + "\n"
        "# Replays the SAME animation code over the fit's OUTPUT (the abstention overlay\n"
        "# still fires where the fit abstains).\n"
    )
    call = (f'viz.render(data, out="{gif.name}", animate=True, emit_code=False, '
            f'anim_fps={fps}, anim_frames={frames})')
    if self_contained:
        blob = base64.b64encode(json.dumps(data, sort_keys=True).encode()).decode("ascii")
        script = (f"{header}import base64, json\n\nimport nudge.viz as viz\n\n"
                  f'_DATA = "{blob}"\ndata = json.loads(base64.b64decode(_DATA))\n{call}\n')
        code_path.write_text(script, encoding="utf-8")
        return str(code_path), None
    data_path = gif.with_name(f"{stem}.data.json")
    data_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    script = (f"{header}import json\n\nimport nudge.viz as viz\n\n"
              f'with open("{data_path.name}") as _fh:\n    data = json.load(_fh)\n{call}\n')
    code_path.write_text(script, encoding="utf-8")
    return str(code_path), str(data_path)
