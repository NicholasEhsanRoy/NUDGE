"""Per-result provenance stamping.

Every fit/design result carries a ``ResultProvenance`` — the data hash, the
circuit hypothesis, the NUDGE/MADDENING/JAX versions, the random seed, and the
loss/metric values. This composes with Claude Science's per-figure provenance: a
NUDGE claim generated inside the workbench is then reproducible at two layers.
Phase-0 stub schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResultProvenance:
    """Reproducibility stamp attached to a ``MechanismMap``."""

    data_hash: str
    circuit_id: str
    seed: int
    versions: dict[str, str] = field(default_factory=dict)  # nudge / maddening / jax
    metrics: dict[str, float] = field(default_factory=dict)
    authored_by: str = "nudge"  # "nudge" | "ai" for AI-in-the-loop artifacts

    def as_dict(self) -> dict[str, str]:
        """Flatten to the string map stored in ``MechanismMap.provenance``."""
        out: dict[str, str] = {
            "data_hash": self.data_hash,
            "circuit_id": self.circuit_id,
            "seed": str(self.seed),
            "authored_by": self.authored_by,
        }
        out.update({f"version.{k}": v for k, v in self.versions.items()})
        out.update({f"metric.{k}": str(v) for k, v in self.metrics.items()})
        return out
