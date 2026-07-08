"""Integrator kernels: the steady state is exactly the root of the rate."""

from __future__ import annotations

import jax.numpy as jnp

from nudge.mechanisms.integrators.linear import LinearIntegrator
from nudge.mechanisms.integrators.saturating import SaturatingIntegrator


def test_linear_steady_state_is_rate_root() -> None:
    ig = LinearIntegrator(decay=2.0)
    prod = jnp.array(6.0)
    ss = ig.steady_state(prod)
    assert jnp.allclose(ss, 3.0)
    assert jnp.allclose(ig.rate(ss, prod), 0.0, atol=1e-6)


def test_saturating_production_half_max_at_km() -> None:
    ig = SaturatingIntegrator(vmax=4.0, km=1.0, decay=1.0)
    # drive == km → half of vmax
    assert jnp.allclose(ig.production(jnp.array(1.0)), 2.0, atol=1e-6)
    assert jnp.allclose(ig.production(jnp.array(1e4)), 4.0, atol=1e-2)  # → vmax


def test_saturating_steady_state_is_rate_root() -> None:
    ig = SaturatingIntegrator(vmax=4.0, km=2.0, decay=0.5)
    drive = jnp.array(2.0)
    ss = ig.steady_state(drive)
    assert jnp.allclose(ig.rate(ss, drive), 0.0, atol=1e-6)
