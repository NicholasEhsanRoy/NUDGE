"""CircuitBuilder assembles a Circuit (feedback = an edge closing a cycle)."""

from __future__ import annotations

import pytest

from nudge.core.builder import CircuitBuilder


def test_builder_assembles_topology() -> None:
    c = (
        CircuitBuilder()
        .add_species("A", basal=1.0, decay=1.0)
        .add_species("B", basal=0.2, decay=1.0)
        .regulate("A", "B", effect="hill_activation", K=1.0, n=2.0, vmax=3.0)
        .feedback("B", "A", effect="hill_activation", K=1.0, n=2.0)
        .build()
    )
    assert c.n_species == 2
    assert c.n_edges == 2
    assert c.index("B") == 1
    assert (c.edges[0].source, c.edges[0].target) == (0, 1)
    assert (c.edges[1].source, c.edges[1].target) == (1, 0)  # feedback closes the cycle


def test_builder_duplicate_species_raises() -> None:
    builder = CircuitBuilder().add_species("A")
    with pytest.raises(ValueError, match="already added"):
        builder.add_species("A")
