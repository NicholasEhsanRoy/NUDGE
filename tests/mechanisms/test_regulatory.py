"""Regulatory effect kernels encode NUDGE's threshold / gain / ceiling vocabulary.

These tests are executable documentation of what each attribution term *means*:
K is the threshold (half-max), n is the gain (Hill steepness), vmax is the ceiling.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nudge.mechanisms.regulatory import (
    HillActivationEffect,
    HillRepressionEffect,
    LinearEffect,
)


def test_linear_effect_is_proportional() -> None:
    e = LinearEffect(weight=2.0)
    assert float(e.response(jnp.array(3.0))) == 6.0


def test_threshold_is_half_max_input() -> None:
    # K is the threshold: response(K) == vmax / 2, whatever the gain.
    e = HillActivationEffect(K=1.5, n=3.0, vmax=4.0)
    assert jnp.allclose(e.response(jnp.array(1.5)), 2.0, atol=1e-5)


def test_ceiling_and_floor() -> None:
    e = HillActivationEffect(K=1.0, n=2.0, vmax=5.0)
    assert jnp.allclose(e.response(jnp.array(0.0)), 0.0, atol=1e-6)
    assert jnp.allclose(e.response(jnp.array(1e3)), 5.0, atol=1e-2)  # → ceiling vmax


def test_activation_is_monotonic_increasing() -> None:
    e = HillActivationEffect(K=1.0, n=2.0)
    ys = e.response(jnp.linspace(0.0, 5.0, 50))
    assert bool(jnp.all(jnp.diff(ys) >= -1e-6))


def test_gain_is_steepness_at_threshold() -> None:
    # Higher n (gain) ⇒ steeper response at the half-max point (ultrasensitivity).
    def slope(n: float) -> float:
        e = HillActivationEffect(K=1.0, n=n, vmax=1.0)
        return float(jax.grad(e.response)(jnp.array(1.0)))

    assert slope(4.0) > slope(2.0) > slope(1.0)


def test_repression_is_activation_complement() -> None:
    k, n, vmax = 1.2, 2.5, 3.0
    rep = HillRepressionEffect(K=k, n=n, vmax=vmax)
    act = HillActivationEffect(K=k, n=n, vmax=vmax)
    x = jnp.array(0.7)
    assert jnp.allclose(rep.response(x) + act.response(x), vmax, atol=1e-5)
    # full effect at zero input
    assert jnp.allclose(rep.response(jnp.array(0.0)), vmax, atol=1e-6)
