"""LinearIntegrator — simple relaxation dynamics.

``dx/dt = production − decay · x``, so the steady state is ``production / decay``.
The natural integrator for a species whose own production is not saturating
(e.g. SOS, RasGRP1 activity). ``linear_rate`` is the single source of truth for
the math (used by the kernel and the circuit vector field). How a circuit is
*solved* is decided separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from nudge.core.metadata import MechanismMeta, MechanismRole
from nudge.mechanisms.registry import default_registry


def linear_rate(x: ArrayLike, production: ArrayLike, decay: ArrayLike) -> Array:
    """Relaxation time-derivative ``production − decay · x``."""
    return jnp.asarray(production) - jnp.asarray(decay) * jnp.asarray(x)


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
        return linear_rate(x, production, self.decay)

    def steady_state(self, production: Array) -> Array:
        """Algebraic steady state ``production / decay`` (the root of ``rate``)."""
        return production / self.decay
