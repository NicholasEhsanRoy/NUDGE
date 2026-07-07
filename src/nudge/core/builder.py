"""``CircuitBuilder`` — the fluent, typed façade over the mechanism registry.

Power users get IDE autocompletion and static type checking; the config/YAML
path (``CircuitSpec``) produces the same ``Circuit`` under the hood. Neither uses
a bare global dict — both resolve mechanisms through a ``MechanismRegistry``.

Phase-0 stub: the fluent surface is fixed; assembly lands in Phase 1.
"""

from __future__ import annotations

from typing import Any

from nudge.core.circuit import Circuit
from nudge.mechanisms.registry import MechanismRegistry, default_registry

__all__ = ["CircuitBuilder"]


class CircuitBuilder:
    """Fluent builder: ``CircuitBuilder().add_species("SOS").regulate(...).build()``."""

    def __init__(self, registry: MechanismRegistry | None = None) -> None:
        self._registry = registry if registry is not None else default_registry
        self._circuit = Circuit()

    def add_species(self, name: str, **params: Any) -> CircuitBuilder:
        """Add a species node governed by an integrator."""
        raise NotImplementedError("CircuitBuilder.add_species — Phase 1")

    def regulate(self, source: str, target: str, effect: Any) -> CircuitBuilder:
        """Add a regulatory edge from ``source`` to ``target``."""
        raise NotImplementedError("CircuitBuilder.regulate — Phase 1")

    def feedback(self, source: str, target: str, effect: Any) -> CircuitBuilder:
        """Add a feedback edge (an edge that closes a cycle — no special type)."""
        raise NotImplementedError("CircuitBuilder.feedback — Phase 1")

    def build(self) -> Circuit:
        """Return the assembled ``Circuit``."""
        return self._circuit
