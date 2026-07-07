"""Base contracts for mechanisms.

A *mechanism* is a node (``Species``, integrators, ``Readout``) or an edge
(regulatory effect) that carries a ``MechanismMeta`` and knows how to emit its
MADDENING representation. Phase-0 defines the Protocol; concrete mechanisms land
in Phase 1 as ``SimulationNode`` subclasses / edges registered on the registry.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nudge.core.metadata import MechanismMeta


@runtime_checkable
class Mechanism(Protocol):
    """A registrable circuit mechanism carrying traceability metadata."""

    meta: MechanismMeta
