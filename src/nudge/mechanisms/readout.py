"""Readout — maps latent species activity to per-gene expression rate Λ.

The biology→measurement boundary (``design/GENERATOR_DESIGN.md`` §2): the circuit
produces per-cell species activities; the ``Readout`` links them to per-gene
expression rates ``Λ ≥ 0``, which the technical layer (``nudge.data.noise``) then
turns into raw counts. Keeping this an explicit, separate layer is what keeps the
mechanism parameters identifiable.

Phase-1 minimal: a non-negative affine reporter map ``Λ = max(base + A·Wᵀ, 0)``.
The count model lives in ``nudge.data.noise`` — the Readout emits rates, not counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp
from jax import Array

from nudge.core.metadata import MechanismMeta, MechanismRole
from nudge.mechanisms.registry import default_registry


@default_registry.register("Readout")
@dataclass(frozen=True)
class Readout:
    """Affine, non-negative reporter map from species activity to expression Λ.

    ``weight`` has shape ``(n_genes, n_species)`` and ``base`` shape ``(n_genes,)``.
    """

    weight: Array
    base: Array

    meta: ClassVar[MechanismMeta] = MechanismMeta(
        algorithm_id="NUDGE-MECH-020",
        role=MechanismRole.READOUT,
        summary="Affine latent→expression reporter; emits rates Λ, not counts.",
        assumptions=("expression rate is affine in activity, clamped non-negative",),
    )

    def expression(self, activity: Array) -> Array:
        """Map activity ``(n_cells, n_species)`` → expression ``Λ``."""
        return jnp.maximum(self.base + activity @ self.weight.T, 0.0)
