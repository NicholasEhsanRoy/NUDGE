"""Pure-Python mechanism metadata (no JAX import).

The in-code half of a Mechanism Card. Reuses MADDENING's compliance metadata
vocabulary where it fits and adds the NUDGE-specific fields a card needs: the
identifiability regime (from the synthetic power sweep) and the decoy(s) that
exercise each known failure mode. See ``design/WORKING_BACKWARDS.md`` Part 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MechanismRole(str, Enum):
    """Where a mechanism sits in the circuit graph."""

    SPECIES = "species"
    INTEGRATOR = "integrator"
    REGULATORY_EDGE = "regulatory-edge"
    READOUT = "readout"
    PERTURBATION = "perturbation"


@dataclass(frozen=True)
class IdentifiabilityRegime:
    """The data regime under which a mechanism's parameters are recoverable.

    Populated from the synthetic identifiability power sweep (Part 5). Empty
    fields mean "not yet characterised".
    """

    min_cells_per_condition: int | None = None
    min_perturbations: int | None = None
    max_dropout_rate: float | None = None
    notes: str = ""


@dataclass(frozen=True)
class MechanismMeta:
    """Traceability metadata for a mechanism (the Mechanism Card, in code)."""

    algorithm_id: str  # e.g. "NUDGE-MECH-001"
    role: MechanismRole
    summary: str
    assumptions: tuple[str, ...] = ()
    known_failure_modes: tuple[str, ...] = ()
    decoy_refs: tuple[str, ...] = ()  # decoy IDs that exercise the failure modes
    limitation_refs: tuple[str, ...] = ()  # NUDGE-LIM-* ids in known_limitations.yaml
    verification_refs: tuple[str, ...] = ()  # NUDGE-VER-* test ids
    identifiability: IdentifiabilityRegime = field(
        default_factory=IdentifiabilityRegime
    )
    references: tuple[str, ...] = ()  # bibliography [@Key] citations
