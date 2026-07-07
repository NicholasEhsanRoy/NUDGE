"""The ``Circuit`` object — NUDGE's compositional gene-regulatory model.

A circuit is assembled from mechanisms (``Species`` nodes + ``RegulatoryEffect``
edges) and compiles to a MADDENING ``GraphManager`` — one JIT-compiled,
differentiable ``state -> state`` function. ``fit`` and ``design`` operate on the
same ``Circuit``; design is simply the fit run backwards. The linear baseline is
the same topology with every edge swapped to ``LinearEffect`` — apples-to-apples
model comparison through one codebase.

Phase-0 stub: the public surface is fixed; the compile path lands in Phase 1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nudge.core.spec import CircuitSpec

__all__ = ["Circuit"]


class Circuit:
    """A differentiable gene-regulatory circuit that compiles to a GraphManager."""

    def __init__(self) -> None:
        self._species: list[str] = []
        self._edges: list[tuple[str, str, Any]] = []

    @classmethod
    def from_spec(cls, spec: CircuitSpec) -> Circuit:
        """Build a ``Circuit`` from a ``CircuitSpec`` via the registry."""
        raise NotImplementedError("Circuit.from_spec — Phase 1")

    def compile(self) -> Any:
        """Compile to a ``maddening.GraphManager`` (JIT, differentiable)."""
        raise NotImplementedError("Circuit.compile — Phase 1")

    def steady_state(self, theta: Any) -> Any:
        """Solve the circuit to steady state for parameter vector ``theta``."""
        raise NotImplementedError("Circuit.steady_state — Phase 1")

    def solve_population(self, theta_dist: Any) -> Any:
        """vmap the steady-state solve over a per-cell parameter distribution."""
        raise NotImplementedError("Circuit.solve_population — Phase 2")

    def linear_baseline(self) -> Circuit:
        """Return the same topology with every edge swapped to ``LinearEffect``."""
        raise NotImplementedError("Circuit.linear_baseline — Phase 1")
