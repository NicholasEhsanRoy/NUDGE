"""Pydantic circuit specification — the config/YAML-facing surface.

A ``CircuitSpec`` is a declarative, serializable description of a circuit whose
mechanism ``type`` fields resolve through a ``MechanismRegistry``. It is the
biologist- and (future) visual-builder-friendly path; ``CircuitBuilder`` is the
fluent-code path; both produce a ``Circuit``. Phase-0 stub schema.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MechanismSpec(BaseModel):
    """A single mechanism reference, resolved via the registry ``type`` name."""

    type: str
    params: dict[str, float] = Field(default_factory=dict)


class EdgeSpec(BaseModel):
    """A regulatory edge: ``source`` acts on ``target`` via ``effect``."""

    source: str
    target: str
    effect: MechanismSpec


class SpeciesSpec(BaseModel):
    """A species node and the integrator governing its production/decay."""

    name: str
    integrator: MechanismSpec


class CircuitSpec(BaseModel):
    """Declarative circuit description (serializable to/from YAML)."""

    species: list[SpeciesSpec] = Field(default_factory=list)
    edges: list[EdgeSpec] = Field(default_factory=list)
    readout: MechanismSpec | None = None
