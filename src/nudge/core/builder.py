"""``CircuitBuilder`` — the fluent, typed path to a ``Circuit``.

Power users get IDE autocompletion and static type checking; the config/YAML path
(``CircuitSpec``) produces the same ``Circuit`` under the hood. Species must be
added before edges that reference them (edges resolve names to indices eagerly).
"""

from __future__ import annotations

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef

__all__ = ["CircuitBuilder"]


class CircuitBuilder:
    """Fluent builder: ``CircuitBuilder().add_species("SOS").regulate(...).build()``."""

    def __init__(self) -> None:
        self._species: list[SpeciesDef] = []
        self._edges: list[EdgeDef] = []
        self._index: dict[str, int] = {}

    def add_species(
        self, name: str, *, integrator: str = "linear", **params: float
    ) -> CircuitBuilder:
        """Add a species node with an integrator (``linear`` | ``saturating``)."""
        if name in self._index:
            raise ValueError(f"species {name!r} already added")
        self._index[name] = len(self._species)
        self._species.append(SpeciesDef(name=name, integrator=integrator, **params))
        return self

    def regulate(
        self,
        source: str,
        target: str,
        *,
        effect: str = "hill_activation",
        **params: float,
    ) -> CircuitBuilder:
        """Add a regulatory edge ``source → target`` (both must already be added)."""
        self._edges.append(
            EdgeDef(
                source=self._index[source],
                target=self._index[target],
                effect=effect,
                **params,
            )
        )
        return self

    def feedback(
        self,
        source: str,
        target: str,
        *,
        effect: str = "hill_activation",
        **params: float,
    ) -> CircuitBuilder:
        """Add a feedback edge — just an edge that closes a cycle (no special type)."""
        return self.regulate(source, target, effect=effect, **params)

    def build(self) -> Circuit:
        """Return the assembled ``Circuit``."""
        return Circuit(self._species, self._edges)
