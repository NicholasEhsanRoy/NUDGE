"""Readout is a non-negative affine map from species activity to expression Λ."""

from __future__ import annotations

import jax.numpy as jnp

from nudge.mechanisms.readout import Readout


def test_affine_map() -> None:
    weight = jnp.array([[1.0, 0.0], [0.0, 2.0]])  # (n_genes=2, n_species=2)
    base = jnp.array([0.5, 1.0])
    readout = Readout(weight=weight, base=base)
    activity = jnp.array([[3.0, 4.0]])  # one cell, two species
    # gene0 = 0.5 + 1*3 + 0*4 = 3.5 ; gene1 = 1.0 + 0*3 + 2*4 = 9.0
    assert jnp.allclose(readout.expression(activity), jnp.array([[3.5, 9.0]]))


def test_non_negative_clamp() -> None:
    readout = Readout(weight=jnp.array([[1.0]]), base=jnp.array([0.0]))
    activity = jnp.array([[-5.0]])  # -5 → clamped to 0 (expression rates are ≥ 0)
    assert float(readout.expression(activity)[0, 0]) == 0.0
