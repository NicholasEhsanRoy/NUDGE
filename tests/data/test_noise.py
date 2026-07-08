"""The observation model reproduces NB (Poisson-Gamma) statistics, not zero-inflation.

These are the countsimQC-style realism self-tests on raw counts: the mean-variance
relationship, the *emergent* zero-fraction-vs-mean curve (no explicit dropout),
and library-size scaling. See design/GENERATOR_DESIGN.md §1, §6.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from nudge.data.noise import (
    nb_variance,
    nb_zero_fraction,
    sample_counts,
    sample_library_sizes,
)

_N = 40_000
_MEANS = jnp.array([0.5, 2.0, 10.0])
_PHI = 0.2


def _counts(key: Array, means: Array = _MEANS, library: Array | None = None) -> Array:
    expr = jnp.broadcast_to(means, (_N, means.shape[0]))
    lib = jnp.ones(_N) if library is None else library
    return sample_counts(key, expr, lib, dispersion=_PHI)


def test_empirical_mean_matches_mu() -> None:
    counts = _counts(jax.random.key(0))
    assert jnp.allclose(counts.mean(0), _MEANS, rtol=0.05)


def test_overdispersion_matches_nb_variance() -> None:
    counts = _counts(jax.random.key(1))
    emp_var = counts.astype(jnp.float32).var(0)
    assert jnp.allclose(emp_var, nb_variance(_MEANS, _PHI), rtol=0.12)
    assert bool(jnp.all(emp_var > _MEANS))  # overdispersed vs Poisson


def test_zero_fraction_matches_nb_not_inflated() -> None:
    counts = _counts(jax.random.key(2))
    emp_zero = (counts == 0).mean(0)
    # Empirical zeros match the NB curve with NO excess → not zero-inflated.
    assert jnp.allclose(emp_zero, nb_zero_fraction(_MEANS, _PHI), atol=0.02)


def test_zero_fraction_decreases_with_mean() -> None:
    emp_zero = (_counts(jax.random.key(3)) == 0).mean(0)
    assert bool(jnp.all(jnp.diff(emp_zero) < 0))  # emergent dropout curve


def test_biological_zeros_pass_through() -> None:
    counts = _counts(jax.random.key(4), means=jnp.array([0.0, 5.0]))
    assert int(counts[:, 0].sum()) == 0  # Λ = 0 → exact zeros (a genuine OFF state)
    assert int(counts[:, 1].sum()) > 0


def test_library_size_scales_mean() -> None:
    key = jax.random.key(5)
    base = _counts(key, library=jnp.ones(_N))
    doubled = _counts(key, library=jnp.full((_N,), 2.0))
    assert jnp.allclose(doubled.mean(0), 2.0 * base.mean(0), rtol=0.06)


def test_sample_library_sizes_positive_lognormal() -> None:
    sizes = sample_library_sizes(jax.random.key(6), 1000)
    assert bool(jnp.all(sizes > 0))
    assert 0.7 < float(sizes.mean()) < 1.5  # mean-≈1 multiplicative factor
