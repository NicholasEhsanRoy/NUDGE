"""Distributional losses for fitting cell *populations*, not means.

Threshold vs gain lives in the *shape* of the single-cell distribution (the
off/on split and its sharpness), so the fit compares the simulated population to
the observed one with a **sample-based distance** — the parameter-free **energy
distance** (default) or an **RBF-MMD**. Both are differentiable w.r.t. the
simulated samples, so gradients flow back to the circuit parameters.

Energy distance is exactly MMD with the negative-Euclidean kernel; both are zero
iff the two empirical distributions match. Cost is O(n·m·d) — use minibatches of
cells for large populations. See ``design/GENERATOR_DESIGN.md`` §1–2.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

__all__ = ["energy_distance", "rbf_mmd"]

_EPS = 1e-12


def _as_2d(x: Array) -> Array:
    return x[:, None] if x.ndim == 1 else x


def _pairwise_sq_dists(a: Array, b: Array) -> Array:
    """Squared Euclidean distances ``||a_i − b_j||²`` via the inner-product identity."""
    aa = jnp.sum(a * a, axis=-1)[:, None]
    bb = jnp.sum(b * b, axis=-1)[None, :]
    return jnp.maximum(aa + bb - 2.0 * (a @ b.T), 0.0)


def _mean_distance(a: Array, b: Array) -> Array:
    # +EPS under the sqrt keeps the gradient finite where the distance is zero.
    return jnp.sqrt(_pairwise_sq_dists(a, b) + _EPS).mean()


def energy_distance(x: Array, y: Array) -> Array:
    """Energy distance between two samples ``x`` ``(n, d)`` and ``y`` ``(m, d)``.

    ``ED = 2·E‖x−y‖ − E‖x−x'‖ − E‖y−y'‖`` — ``≥ 0``, and ``0`` iff the empirical
    distributions match. 1-D inputs are treated as ``(n, 1)``.
    """
    x, y = _as_2d(x), _as_2d(y)
    return 2.0 * _mean_distance(x, y) - _mean_distance(x, x) - _mean_distance(y, y)


def rbf_mmd(x: Array, y: Array, *, sigma: float = 1.0) -> Array:
    """Squared MMD between ``x`` and ``y`` under a Gaussian (RBF) kernel of width sigma.

    ``≥ 0``, and ``0`` iff the distributions match. Pick ``sigma`` near the median
    pairwise distance of the pooled data (the median heuristic).
    """
    x, y = _as_2d(x), _as_2d(y)
    scale = 2.0 * sigma * sigma

    def kernel(a: Array, b: Array) -> Array:
        return jnp.exp(-_pairwise_sq_dists(a, b) / scale)

    return kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean()
