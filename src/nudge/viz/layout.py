"""Collision-aware placement — keep annotations, legends, and banners OFF the data.

A shared layout layer every renderer uses so text never lands on top of data points or
other text (the user's #1 complaint). It is deliberately dependency-light: it reads the
already-drawn artists (``Line2D`` / scatter ``PathCollection`` / bar ``Rectangle``s),
builds a coarse occupancy grid in **axes-fraction** coordinates, and answers three
questions renderers keep asking:

* :func:`reserve_top_band` — bump the y-limit so the top strip of the axes is EMPTY of
  data, giving the abstention banner (and the ``K``-line label) a clear band to sit in.
* :func:`freest_corner` — pick the legend corner (``loc=``) whose region is least
  occupied by data, so ``legend(loc="best")`` never lands on the ``K``-label.
* :func:`place_label` — drop a text label at the freest of several candidate anchors and
  return where it went, so a second label can avoid it.

Everything is computed from the *data's* occupancy, not hard-coded positions, so the same
helper works for a full sigmoid, a flat no-effect line, a bar chart, or a trajectory.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _artist_points_axes(ax: Any) -> np.ndarray:
    """All drawn data points (lines + scatter + bar tops) in axes-fraction coords.

    Returns an ``(N, 2)`` array in ``[0, 1]²`` (points outside the view are kept but
    clipped by callers via the grid). Uses ``ax.transLimits`` (data → axes [0,1]) so it
    reflects the CURRENT view limits — call it AFTER the limits are final.
    """
    pts: list[np.ndarray] = []
    for line in getattr(ax, "lines", []):
        xy = line.get_xydata()
        if xy is not None and len(xy):
            pts.append(np.asarray(xy, dtype=float))
    for coll in getattr(ax, "collections", []):
        try:
            off = np.asarray(coll.get_offsets(), dtype=float)
        except Exception:  # pragma: no cover - exotic collection
            off = None
        if off is not None and off.ndim == 2 and off.shape[0]:
            pts.append(off)
    for patch in getattr(ax, "patches", []):
        # Bars: sample the top edge so tall bars read as occupied.
        get_xy = getattr(patch, "get_xy", None)
        get_w = getattr(patch, "get_width", None)
        get_h = getattr(patch, "get_height", None)
        if callable(get_xy) and callable(get_w) and callable(get_h):
            try:
                xy0 = np.asarray(get_xy(), dtype=float).ravel()
                w = float(np.asarray(get_w(), dtype=float))
                h = float(np.asarray(get_h(), dtype=float))
            except Exception:  # pragma: no cover
                continue
            if xy0.size >= 2 and np.isfinite([xy0[0], xy0[1], w, h]).all():
                x0, y0 = float(xy0[0]), float(xy0[1])
                xs = np.linspace(x0, x0 + w, 4)
                pts.append(np.column_stack([xs, np.full_like(xs, y0 + h)]))
    if not pts:
        return np.empty((0, 2), dtype=float)
    data = np.vstack(pts)
    finite = np.isfinite(data).all(axis=1)
    data = data[finite]
    if not len(data):
        return np.empty((0, 2), dtype=float)
    return np.asarray(ax.transLimits.transform(data), dtype=float)


def _occupancy(ax: Any, nx: int = 10, ny: int = 10) -> np.ndarray:
    """A coarse ``(ny, nx)`` boolean grid: True where data occupies that axes cell."""
    grid = np.zeros((ny, nx), dtype=bool)
    ap = _artist_points_axes(ax)
    if not len(ap):
        return grid
    inside = (ap[:, 0] >= 0) & (ap[:, 0] <= 1) & (ap[:, 1] >= 0) & (ap[:, 1] <= 1)
    ap = ap[inside]
    if not len(ap):
        return grid
    ix = np.clip((ap[:, 0] * nx).astype(int), 0, nx - 1)
    iy = np.clip((ap[:, 1] * ny).astype(int), 0, ny - 1)
    grid[iy, ix] = True
    return grid


def reserve_top_band(ax: Any, band: float = 0.22, pad: float = 0.04) -> float:
    """Bump the upper y-limit so the top ``band`` fraction of the axes has NO data.

    Returns the axes-fraction y at the centre of the cleared band (where a banner / label
    should be anchored). Idempotent-ish: if the data already clears the band, only a small
    ``pad`` of headroom is added. Works for both directions of curve.
    """
    ap = _artist_points_axes(ax)
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    if span <= 0:
        return 1.0 - band / 2.0
    if len(ap):
        inside = (ap[:, 0] >= -0.02) & (ap[:, 0] <= 1.02)
        yfrac = ap[inside, 1] if inside.any() else ap[:, 1]
        data_top = float(np.nanmax(yfrac)) if len(yfrac) else 0.0
    else:
        data_top = 0.0
    target_top = 1.0 - band
    if data_top > target_top:
        # Rescale so the current data-top maps to (1 - band): new_ymax s.t.
        # (data_ymax - ymin)/(new_ymax - ymin) = target_top.
        data_ymax = ymin + data_top * span
        new_span = (data_ymax - ymin) / max(target_top, 1e-6)
        ax.set_ylim(ymin, ymin + new_span * (1.0 + pad))
    else:
        ax.set_ylim(ymin, ymax + span * pad)
    return 1.0 - band / 2.0


def freest_corner(ax: Any, *, avoid_top: bool = True) -> str:
    """Pick the legend ``loc`` whose region is least covered by data (and not the banner).

    Scores the four/six standard corners against the occupancy grid; when ``avoid_top`` is
    set (a banner or K-label lives up there) the upper corners are penalised so the legend
    drops to a clear bottom corner.
    """
    grid = _occupancy(ax, nx=8, ny=8)
    ny, nx = grid.shape
    # Regions as (row_slice, col_slice) in grid space; row 0 = bottom.
    regions = {
        "lower left": (slice(0, ny // 2), slice(0, nx // 2)),
        "lower right": (slice(0, ny // 2), slice(nx // 2, nx)),
        "upper left": (slice(ny // 2, ny), slice(0, nx // 2)),
        "upper right": (slice(ny // 2, ny), slice(nx // 2, nx)),
        "center left": (slice(ny // 4, 3 * ny // 4), slice(0, nx // 3)),
        "center right": (slice(ny // 4, 3 * ny // 4), slice(2 * nx // 3, nx)),
    }
    best_loc, best_score = "lower left", np.inf
    for loc, (rs, cs) in regions.items():
        score = float(grid[rs, cs].sum())
        if avoid_top and loc.startswith("upper"):
            score += 100.0  # keep the legend clear of the reserved top band
        if score < best_score:
            best_loc, best_score = loc, score
    return best_loc


def place_label(
    ax: Any,
    text: str,
    candidates: list[tuple[float, float]],
    *,
    color: str,
    fontsize: float = 8.0,
    avoid: list[tuple[float, float]] | None = None,
    **kw: Any,
) -> tuple[float, float]:
    """Place ``text`` at the freest candidate anchor (axes-fraction) and return where.

    Scores each ``(x, y)`` candidate by nearby data occupancy plus distance to any
    ``avoid`` anchors already placed, and draws at the best one. ``kw`` passes through to
    ``ax.text`` (e.g. ``ha`` / ``va``).
    """
    grid = _occupancy(ax, nx=10, ny=10)
    ny, nx = grid.shape
    avoid = avoid or []
    best_xy, best_score = candidates[0], np.inf
    for (x, y) in candidates:
        cx, cy = int(np.clip(x * nx, 0, nx - 1)), int(np.clip(y * ny, 0, ny - 1))
        r0, r1 = max(cy - 1, 0), min(cy + 2, ny)
        c0, c1 = max(cx - 1, 0), min(cx + 2, nx)
        score = float(grid[r0:r1, c0:c1].sum())
        for (ax_, ay_) in avoid:
            d = np.hypot(x - ax_, y - ay_)
            score += max(0.0, 1.0 - d) * 3.0
        if score < best_score:
            best_xy, best_score = (x, y), score
    ax.text(
        best_xy[0], best_xy[1], text, transform=ax.transAxes,
        fontsize=fontsize, color=color, zorder=7, **kw,
    )
    return best_xy
