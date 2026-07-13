"""Inline-transport helpers — base64 + GIF size discipline for the MCP inline path.

Claude Science's connector can deliver a figure only as **inline base64** (its shared dir
is mounted read-only → the connector can't write a path the client can read; its own temp
is invisible to the client). So the ``render_figure`` inline branch base64-encodes the
image, and — because a GIF is far heavier than a static PNG — applies a strict size
discipline before it does:

1. **downscale** every frame to a max dimension + **frame-limit** (sample down to a frame
   cap) + a **tight palette** re-encode;
2. a **never-inflate guard** — keep the re-encoded bytes only if they are *smaller* than the
   original, else keep the original;
3. a **hard cap** (~1.5 MB of base64): if even the downscaled GIF is over the cap, fall back
   to a **reduced static preview** (the final frame as a small PNG); if THAT is still over,
   omit the image with a clear ``too large — <reason>`` note. It never silently truncates.

Static PNGs are already small, so they take the plain capped-base64 path. Pure byte helpers
— they read image bytes, never a result — so they live in ``nudge.viz`` beside the drawing
layer without touching the fit engine.
"""

from __future__ import annotations

import base64
from typing import Any

#: Inline base64 cap (characters). ~1.5 MB — the Claude Science inline limit; base64 counts
#: as characters over the wire, so the guard is on the encoded length, not the raw bytes.
INLINE_B64_CAP = 1_500_000

#: GIF size-discipline defaults (authors should also keep GIFs small at build time).
_GIF_MAX_PX = 480
_GIF_MAX_FRAMES = 30
_PREVIEW_MAX_PX = 640


def encode_b64(data: bytes) -> str:
    """Base64-encode ``data`` to an ASCII string (the inline wire form)."""
    return base64.b64encode(data).decode("ascii")


def compress_gif(
    gif_bytes: bytes, *, max_px: int = _GIF_MAX_PX, max_frames: int = _GIF_MAX_FRAMES
) -> tuple[bytes, str | None]:
    """Downscale + frame-limit + tight-palette re-encode a GIF; **never inflate**.

    Returns ``(bytes, note)``. On any failure — or if the re-encode did not shrink the file
    — returns the ORIGINAL bytes (the never-inflate guard), so this can only help, never
    hurt. ``note`` is a short human-readable record of what happened (for provenance).
    """
    try:
        import io

        from PIL import Image, ImageSequence
    except Exception as exc:  # pragma: no cover - Pillow ships with matplotlib
        return gif_bytes, f"gif compression skipped ({type(exc).__name__})"

    try:
        im = Image.open(io.BytesIO(gif_bytes))
        duration = im.info.get("duration", 80)
        frames = [f.copy() for f in ImageSequence.Iterator(im)]
        n = len(frames)
        if n == 0:
            return gif_bytes, None
        if n > max_frames:  # frame-limit: keep an even sample (always incl. the last frame)
            idx = sorted({round(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)})
            frames = [frames[i] for i in idx]
        out_frames = []
        for f in frames:
            rgb = f.convert("RGB")
            w, h = rgb.size
            scale = min(1.0, max_px / max(w, h))
            if scale < 1.0:
                rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))))
            out_frames.append(rgb.quantize(colors=64, method=Image.Quantize.MEDIANCUT))
        buf = io.BytesIO()
        out_frames[0].save(
            buf, format="GIF", save_all=True, append_images=out_frames[1:], loop=0,
            duration=duration, optimize=True, disposal=2,
        )
        out = buf.getvalue()
    except Exception as exc:  # pragma: no cover - defensive
        return gif_bytes, f"gif compression skipped ({type(exc).__name__})"

    if len(out) < len(gif_bytes):  # never-inflate guard
        return out, (f"gif compressed {len(gif_bytes)}→{len(out)} bytes "
                     f"({len(out_frames)} frames, ≤{max_px}px)")
    return gif_bytes, "gif kept original (re-encode did not shrink it)"


def gif_final_frame_png(gif_bytes: bytes, *, max_px: int = _PREVIEW_MAX_PX) -> bytes | None:
    """Extract the GIF's FINAL frame as a small PNG — the reduced-preview fallback.

    Our animations resolve to their end-state on the last frame (the switch resolved, the
    ellipse collapsed, the dial at the fold), so the final frame is the honest still to show
    when the animation itself is too large to inline. Returns ``None`` on failure.
    """
    try:
        import io

        from PIL import Image, ImageSequence

        im = Image.open(io.BytesIO(gif_bytes))
        frames = [f.copy() for f in ImageSequence.Iterator(im)]
        if not frames:
            return None
        last = frames[-1].convert("RGB")
        w, h = last.size
        scale = min(1.0, max_px / max(w, h))
        if scale < 1.0:
            last = last.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        last.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception:  # pragma: no cover - defensive
        return None


def prepare_inline_image(
    image_bytes: bytes, mime_type: str, *, cap: int = INLINE_B64_CAP
) -> dict[str, Any]:
    """Apply the size discipline and return the inline image fields.

    Returns ``{image_base64, mime_type, image_base64_omitted_reason, compression_note}``.
    GIFs are compressed first (never-inflate); if the encoded image is still over ``cap`` a
    reduced static PNG preview of the final frame is substituted; if even that is over, the
    image is omitted with a ``too large — <reason>`` note (never a silent truncation).
    """
    note: str | None = None
    if mime_type == "image/gif":
        image_bytes, note = compress_gif(image_bytes)

    b64 = encode_b64(image_bytes)
    if len(b64) <= cap:
        return {
            "image_base64": b64, "mime_type": mime_type,
            "image_base64_omitted_reason": None, "compression_note": note,
        }

    # Over the cap. For a GIF, fall back to a reduced static preview of the final frame.
    if mime_type == "image/gif":
        png = gif_final_frame_png(image_bytes)
        if png is not None:
            pb64 = encode_b64(png)
            if len(pb64) <= cap:
                return {
                    "image_base64": pb64, "mime_type": "image/png",
                    "image_base64_omitted_reason": (
                        f"animation exceeded the {cap:,}-char inline cap; showing the final "
                        "frame as a static PNG preview (read the full GIF from its path/code)"
                    ),
                    "compression_note": note,
                }

    return {
        "image_base64": None, "mime_type": mime_type,
        "image_base64_omitted_reason": (
            f"too large — the base64 image ({len(b64):,} chars) exceeds the "
            f"{cap:,}-char inline cap even after size reduction"
        ),
        "compression_note": note,
    }
