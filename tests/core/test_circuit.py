"""Circuit solve: relaxation, bistability, routing, baseline, population vmap."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from nudge.core.circuit import Circuit, EdgeDef, Params, SpeciesDef


def test_relaxation_steady_state() -> None:
    # Single species, no edges: steady state = basal / decay.
    c = Circuit([SpeciesDef("A", basal=6.0, decay=2.0)], [])
    ss = c.steady_state(c.base_params(), jnp.zeros(1))
    assert jnp.allclose(ss, 3.0, atol=1e-3)


def test_vector_field_zero_at_steady_state() -> None:
    c = Circuit(
        [SpeciesDef("A", basal=1.0, decay=1.0), SpeciesDef("B", basal=0.2, decay=1.0)],
        [EdgeDef(0, 1, "hill_activation", K=1.0, n=2.0, vmax=3.0)],
    )
    p = c.base_params()
    ss = c.steady_state(p, jnp.zeros(2))
    assert jnp.allclose(c.vector_field(ss, p), 0.0, atol=1e-3)


def test_feedforward_routing_matches_hill() -> None:
    # A relaxes to basal_A/decay_A = 1; B = basal_B + hill(A) (decay_B = 1).
    c = Circuit(
        [SpeciesDef("A", basal=1.0, decay=1.0), SpeciesDef("B", basal=0.2, decay=1.0)],
        [EdgeDef(0, 1, "hill_activation", K=1.0, n=2.0, vmax=3.0)],
    )
    ss = c.steady_state(c.base_params(), jnp.zeros(2))
    expected_b = 0.2 + 3.0 * 1.0**2 / (1.0**2 + 1.0**2)  # 0.2 + 1.5 = 1.7
    assert jnp.allclose(ss[0], 1.0, atol=1e-3)
    assert jnp.allclose(ss[1], expected_b, atol=3e-3)


def test_bistability_from_cooperative_self_activation() -> None:
    # dA/dt = 0.2 + 2·A^4/(1+A^4) − A has two stable fixed points (~0.2 and ~2.1).
    c = Circuit(
        [SpeciesDef("A", basal=0.2, decay=1.0)],
        [EdgeDef(0, 0, "hill_activation", K=1.0, n=4.0, vmax=2.0)],
    )
    p = c.base_params()
    low = c.steady_state(p, jnp.array([0.0]))
    high = c.steady_state(p, jnp.array([3.0]))
    assert float(low[0]) < 0.5  # OFF basin
    assert float(high[0]) > 1.5  # ON basin
    assert jnp.allclose(c.vector_field(low, p), 0.0, atol=1e-3)
    assert jnp.allclose(c.vector_field(high, p), 0.0, atol=1e-3)


def test_linear_baseline_swaps_effects() -> None:
    c = Circuit(
        [SpeciesDef("A"), SpeciesDef("B")],
        [EdgeDef(0, 1, "hill_activation", K=1.0, n=3.0)],
    )
    base = c.linear_baseline()
    assert all(e.effect == "linear" for e in base.edges)
    assert base.species == c.species  # same topology


def test_solve_population_matches_per_cell() -> None:
    c = Circuit([SpeciesDef("A", decay=1.0)], [])
    n_cells = 5
    basal = jnp.linspace(1.0, 5.0, n_cells)

    def tile(a: Array) -> Array:
        return jnp.broadcast_to(a, (n_cells, *a.shape))

    base = c.base_params()
    params: Params = {
        "species": {k: tile(v) for k, v in base["species"].items()},
        "edges": {k: tile(v) for k, v in base["edges"].items()},
    }
    params["species"]["basal"] = basal[:, None]  # per-cell basal
    ss = c.solve_population(params, jnp.zeros((n_cells, 1)))
    # single species, no edges → steady state = basal / decay = basal
    assert jnp.allclose(ss[:, 0], basal, atol=1e-3)
