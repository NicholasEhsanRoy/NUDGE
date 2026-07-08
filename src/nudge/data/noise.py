"""Technical observation model — latent expression Λ → raw UMI counts.

The literature-grounded second layer (see ``design/GENERATOR_DESIGN.md`` §1–2):
UMI droplet counts are **negative binomial, not zero-inflated** (Svensson 2020;
Townes 2019; Sarkar & Stephens 2021; Jiang 2022). We sample NB as **Poisson-Gamma**
— the Gamma carries intrinsic/technical overdispersion, the Poisson the capture —
with dispersion a function of the mean and per-cell library-size scaling.

There is **no Bernoulli dropout mask**: an explicit zero-inflation term
double-counts the zeros already produced by low-depth sampling, biases the
mean-variance trend, and injects spurious bimodality that a switch-detector would
misread as ultrasensitivity. High zero fractions here are *emergent* from low
depth / low expression, and genuine biological OFF-state zeros (``Λ = 0``) pass
through as true zeros — exactly the signal NUDGE must detect.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

#: Dispersion φ: either a constant or a function of the mean (mean-dependent
#: dispersion enforces the empirical mean-variance trend — Splatter's mechanism).
Dispersion = float | Callable[[Array], Array]


def _phi(dispersion: Dispersion, mean: Array) -> Array:
    """Resolve φ to an array broadcastable to ``mean``."""
    value = dispersion(mean) if callable(dispersion) else dispersion
    return jnp.broadcast_to(jnp.asarray(value, dtype=mean.dtype), mean.shape)


def nb_variance(mean: Array, dispersion: Dispersion) -> Array:
    """Negative-binomial variance ``μ + φ·μ²`` (overdispersed vs Poisson)."""
    phi = _phi(dispersion, mean)
    return mean + phi * mean**2


def nb_zero_fraction(mean: Array, dispersion: Dispersion) -> Array:
    """NB probability of a zero count, ``(r / (r + μ))^r`` with ``r = 1/φ``.

    This is the *emergent* dropout-vs-mean curve — decreasing in the mean — that a
    correct model reproduces without any explicit dropout term.
    """
    phi = _phi(dispersion, mean)
    r = 1.0 / phi
    return jnp.where(mean > 0, (r / (r + mean)) ** r, 1.0)


def sample_library_sizes(
    key: Array, n_cells: int, *, log_mean: float = 0.0, log_sd: float = 0.35
) -> Array:
    """Draw per-cell library-size *factors* (lognormal — real total-count spread).

    Default is a mean-≈1 multiplicative factor; ``Λ`` carries the count scale.
    """
    return jnp.exp(log_mean + log_sd * jax.random.normal(key, (n_cells,)))


def sample_counts(
    key: Array,
    expression: Array,
    library_size: Array,
    *,
    dispersion: Dispersion = 0.1,
) -> Array:
    """Sample raw UMI counts ``~ NB(mean = library_size · Λ, φ)`` via Poisson-Gamma.

    Parameters
    ----------
    key
        PRNG key.
    expression
        Biological expression rate ``Λ``, shape ``(n_cells, n_genes)``, ``≥ 0``.
    library_size
        Per-cell size factor, shape ``(n_cells,)``.
    dispersion
        φ, a constant or a callable of the mean; ``var = μ + φ·μ²``.

    Returns
    -------
    Integer counts, shape ``(n_cells, n_genes)``. No dropout mask is applied.
    """
    mean = library_size[:, None] * expression  # μ
    if not callable(dispersion) and float(dispersion) == 0.0:
        # φ = 0 → no overdispersion → pure Poisson capture (realism level 0).
        return jax.random.poisson(key, mean).astype(jnp.int32)
    r = 1.0 / _phi(dispersion, mean)  # NB "size" (number of failures)
    key_gamma, key_pois = jax.random.split(key)
    # Gamma(shape=r, scale=μ/r) has mean μ and variance μ²/r; Poisson-sampling it
    # yields the NB marginal. mean == 0 (biological OFF) stays an exact zero.
    rate = jax.random.gamma(key_gamma, r) * jnp.where(mean > 0, mean / r, 0.0)
    counts = jax.random.poisson(key_pois, rate)
    return counts.astype(jnp.int32)
