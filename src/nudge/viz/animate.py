"""Animation engine (design §5.2) — ``FuncAnimation`` → Pillow GIF, no external deps.

The flagship is the **constitutive-flip** (NUDGE-LIM-006): the circuit-``n`` profile going
from FLAT — WITHOUT the constitutive control you cannot even tell a switch exists, any
``n`` fits — to the ``n=1`` (no-switch) point being REJECTED once the control is switched
ON. It is NUDGE's sharpest fail-safe story (a documented confident-false-positive turned
into a correct call) and it is *dynamic*, so it earns the "cool to watch" half of Demo.

Honesty is preserved: the animation is built from the SAME frozen result the static figure
uses, and if the fit ABSTAINS (``unresolved``) the abstention overlay is stamped on every
frame — the control can turn on and the profile still stays flat, and the picture says so.
GIFs are written with matplotlib's bundled ``PillowWriter`` (Pillow is already a matplotlib
dependency); no ffmpeg / mp4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nudge.viz.base import FigureResult, abstain_overlay, is_abstention
from nudge.viz.layout import freest_corner
from nudge.viz.theme import apply_theme

# kind → the module + its data-normaliser (mirrors the static registry).
_ANIMATORS = {"constitutive": "nudge.viz.constitutive"}


def _norm(v: Any) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if not len(v) or not np.isfinite(v).any():
        return v
    return v - np.nanmin(v)


def _ease(frac: float) -> float:
    """Smooth 0→1 ease-in-out so the control 'switches on' fluidly, then holds."""
    return float(0.5 - 0.5 * np.cos(np.pi * np.clip(frac, 0.0, 1.0)))


def _constitutive_anim(data: dict[str, Any], theme: str, frames: int) -> Any:
    """Build the constitutive-flip FuncAnimation. Returns ``(fig, anim, caption, abst)``."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    from nudge.viz._util import verdict_color

    pal = apply_theme(theme)
    color = verdict_color(data["call"], pal)
    ng = np.asarray(data["n_grid"], dtype=float)
    lno = _norm(data["loss_no_control"])
    lwith = _norm(data["loss_with_control"])
    abst = is_abstention(data["call"])
    ymax = float(np.nanmax([lno.max() if len(lno) else 0.0,
                            lwith.max() if len(lwith) else 1.0])) or 1.0

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    hold = max(frames // 4, 2)  # hold the final ("control ON") state at the end

    def draw(i: int) -> None:
        ax.clear()
        t = _ease(min(i, frames - hold) / max(frames - hold, 1))
        cur = (1.0 - t) * lno + t * lwith  # morph the WITH-control curve as control ramps
        ax.plot(ng, lno, color=pal["muted"], lw=2.0, marker="o", ms=4, zorder=3,
                label="WITHOUT control (flat → can't tell)")
        ax.plot(ng, cur, color=color, lw=2.6, marker="o", ms=4, zorder=4,
                label="WITH control")
        ax.axvline(1.0, ls=":", color=pal["text"], lw=1.2, zorder=2)
        ax.annotate("n=1\n(no switch)", xy=(1.0, 0.0), xycoords=("data", "axes fraction"),
                    xytext=(4, 6), textcoords="offset points", fontsize=7.5,
                    color=pal["text"], ha="left", va="bottom")
        ax.set_xlabel("circuit Hill n (hypothesis)")
        ax.set_ylabel("profile loss  (Δ from min)")
        ax.set_ylim(-0.03 * ymax, 1.15 * ymax)
        state = "ON" if t > 0.5 else "OFF"
        ax.set_title(f"constitutive control: {state}   —   does a switch exist?",
                     fontweight="bold", color=pal["text"])
        # Keep the legend out of the reserved top band (the abstain banner may live there).
        ax.legend(loc=freest_corner(ax, avoid_top=True), fontsize=8, framealpha=0.9)
        if abst:
            # The fit abstains → the profile stays flat even with the control on; say so.
            abstain_overlay(ax, data["call"], data.get("reason", ""), palette=pal)

    anim = FuncAnimation(
        fig, draw, frames=frames, interval=1000 // 8, blit=False,  # type: ignore[arg-type]
    )
    call = data["call"].upper()
    if abst:
        caption = f"{data.get('label', 'circuit')} → {call} (control ON, still flat → can't tell)"
    else:
        caption = (f"{data.get('label', 'circuit')} → {call} — WITH the constitutive "
                   f"control, n=1 is rejected (by {data.get('n1_rejection', float('nan')):.1f})")
    return fig, anim, caption, abst


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

    Dispatches on ``kind`` (only ``constitutive`` is animated so far — design §5.2's
    flagship). Writes ``out`` (a ``.gif``) via ``PillowWriter`` and, unless
    ``emit_code=False``, the regenerating ``fig.py`` + data sidecar (the provenance grain).
    The abstention overlay is applied per-frame off the fit's own verdict.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    if kind is None and isinstance(result_or_data, dict):
        kind = result_or_data.get("kind")
    if kind not in _ANIMATORS:
        raise ValueError(
            f"nudge.viz has no animation for kind={kind!r} "
            f"(animated kinds: {sorted(_ANIMATORS)})"
        )

    import importlib

    mod = importlib.import_module(_ANIMATORS[kind])
    data = mod.constitutive_data(result_or_data)
    fig, anim, caption, abst = _constitutive_anim(data, theme, frames)

    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(outp), writer=PillowWriter(fps=fps))
    plt.close(fig)

    code_path: str | None = None
    data_path: str | None = None
    if emit_code:
        code_path, data_path = _emit_animation_code(
            data, str(outp), fps=fps, frames=frames,
            self_contained=self_contained, cli_call=cli_call,
        )

    return FigureResult(
        path=str(outp), code_path=code_path, data_path=data_path, png_base64=None,
        caption=caption, abstained=abst, kind=kind,
    )


def _emit_animation_code(
    data: dict[str, Any], gif_path: str, *, fps: int, frames: int,
    self_contained: bool, cli_call: str | None,
) -> tuple[str, str | None]:
    """Emit a standalone ``fig.py`` (+ data sidecar) that regenerates the GIF (no re-fit)."""
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
