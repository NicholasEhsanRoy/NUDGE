"""LinearIntegrator — simple relaxation dynamics.

``dx/dt = production − decay · x``, so the steady state is ``production / decay``.
The natural integrator for a species whose own production is not saturating
(e.g. SOS, RasGRP1 activity). The *right-hand side* here is deterministic and
review-invariant; how a circuit is *solved* (single-node algebra vs a coupled
population solve) is decided separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from jax import Array

from nudge.core.metadata import MechanismMeta, MechanismRole
from nudge.mechanisms.registry import default_registry


@default_registry.register("LinearIntegrator")
@dataclass(frozen=True)
class LinearIntegrator:
    """Relaxation integrator ``dx/dt = production − decay · x``."""

    decay: float = 1.0

    meta: ClassVar[MechanismMeta] = MechanismMeta(
        algorithm_id="NUDGE-MECH-010",
        role=MechanismRole.INTEGRATOR,
        summary="Relaxation dx/dt = production - decay*x; steady state prod/decay.",
    )

    def rate(self, x: Array, production: Array) -> Array:
        """Time derivative ``dx/dt`` given current ``x`` and drive ``production``."""
        return production - self.decay * x

    def steady_state(self, production: Array) -> Array:
        """Algebraic steady state ``production / decay`` (the root of ``rate``)."""
        return production / self.decay
