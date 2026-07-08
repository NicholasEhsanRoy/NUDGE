"""SaturatingIntegrator — Michaelis-Menten production with linear decay.

``dx/dt = vmax · drive / (km + drive) − decay · x``. Production saturates as the
drive grows, so the species has its own ceiling ``vmax / decay`` independent of
any regulatory edge. The pure functions ``saturating_production`` / ``saturating_rate``
are the single source of truth for the math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from nudge.core.metadata import MechanismMeta, MechanismRole
from nudge.mechanisms.registry import default_registry


def saturating_production(drive: ArrayLike, vmax: ArrayLike, km: ArrayLike) -> Array:
    """Michaelis-Menten production ``vmax · drive / (km + drive)``."""
    drive = jnp.asarray(drive)
    return jnp.asarray(vmax) * drive / (jnp.asarray(km) + drive)


def saturating_rate(
    x: ArrayLike, drive: ArrayLike, vmax: ArrayLike, km: ArrayLike, decay: ArrayLike
) -> Array:
    """Time-derivative ``vmax · drive/(km+drive) − decay · x``."""
    return saturating_production(drive, vmax, km) - jnp.asarray(decay) * jnp.asarray(x)


@default_registry.register("SaturatingIntegrator")
@dataclass(frozen=True)
class SaturatingIntegrator:
    """Saturating-production integrator ``dx/dt = vmax·drive/(km+drive) − decay·x``."""

    vmax: float = 1.0
    km: float = 1.0
    decay: float = 1.0

    meta: ClassVar[MechanismMeta] = MechanismMeta(
        algorithm_id="NUDGE-MECH-011",
        role=MechanismRole.INTEGRATOR,
        summary="MM production dx/dt = vmax*drive/(km+drive) - decay*x.",
        assumptions=("Michaelis-Menten quasi-steady-state on production",),
    )

    def production(self, drive: Array) -> Array:
        """Saturating production term ``vmax · drive / (km + drive)``."""
        return saturating_production(drive, self.vmax, self.km)

    def rate(self, x: Array, drive: Array) -> Array:
        """Time derivative ``dx/dt`` given current ``x`` and ``drive``."""
        return saturating_rate(x, drive, self.vmax, self.km, self.decay)

    def steady_state(self, drive: Array) -> Array:
        """Algebraic steady state ``production(drive) / decay``."""
        return self.production(drive) / self.decay
