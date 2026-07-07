"""The serializable output of ``fit`` — a ``MechanismMap``.

This is the artifact the CLI prints, the MCP server returns, ``provenance``
stamps, and the AI reviewer hook (creative-AI idea 3) inspects. Keeping it a
typed, serializable pydantic model is what lets an antagonistic reviewer
mechanically check that no confident claim has overlapping uncertainty bounds.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from nudge.core.vocabulary import MechanismClass


class MechanismCall(BaseModel):
    """NUDGE's verdict for a single perturbation."""

    perturbation: str
    mechanism: MechanismClass
    confidence: float = Field(ge=0.0, le=1.0)
    #: Per-parameter Laplace intervals, keyed by parameter name.
    intervals: dict[str, tuple[float, float]] = Field(default_factory=dict)
    #: Present for abstention classes; explains why NUDGE declined to attribute.
    rationale: str = ""


class MechanismMap(BaseModel):
    """The full result of a fit: one call per perturbation + fit-level diagnostics."""

    calls: list[MechanismCall] = Field(default_factory=list)
    #: Whether the mechanistic fit beat the linear baseline by a margin that
    #: survives Laplace uncertainty (the false-positive guard).
    beats_linear_baseline: bool = False
    fit_quality: float | None = None
    #: Provenance stamp (data hash, circuit, versions, seed, metrics).
    provenance: dict[str, str] = Field(default_factory=dict)
