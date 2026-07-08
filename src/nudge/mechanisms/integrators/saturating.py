"""SaturatingIntegrator — Michaelis-Menten production with linear decay.

``dx/dt = vmax · drive / (km + drive) − decay · x``. Production saturates as the
drive grows, so the species has its own ceiling ``vmax / decay`` independent of
any regulatory edge. Deterministic, review-invariant right-hand side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from jax import Array

from nudge.core.metadata import MechanismMeta, MechanismRole
from nudge.mechanisms.registry import default_registry


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
        return self.vmax * drive / (self.km + drive)

    def rate(self, x: Array, drive: Array) -> Array:
        """Time derivative ``dx/dt`` given current ``x`` and ``drive``."""
        return self.production(drive) - self.decay * x

    def steady_state(self, drive: Array) -> Array:
        """Algebraic steady state ``production(drive) / decay``."""
        return self.production(drive) / self.decay
