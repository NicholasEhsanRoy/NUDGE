"""Regulatory edge effects — how a regulator's activity modulates its target.

Each effect is a differentiable JAX response function ``f(x)`` of the regulator
activity ``x``. The parameters carry NUDGE's entire attribution vocabulary:

- ``K``  (half-max)       → the switch **threshold**
- ``n``  (Hill exponent)  → the **gain** / ultrasensitivity (``n > 1`` = switch-like)
- ``vmax``                → the **ceiling** (maximal effect)

Feedback is *not* a separate type — it is any of these on an edge that closes a
cycle (per the brief). All three register on the default ``MechanismRegistry``;
importing this module is what populates them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp
from jax import Array

from nudge.core.metadata import MechanismMeta, MechanismRole
from nudge.mechanisms.registry import default_registry


@default_registry.register("LinearEffect")
@dataclass(frozen=True)
class LinearEffect:
    """Baseline linear edge ``f(x) = weight * x`` (the linear model's edge)."""

    weight: float = 1.0

    meta: ClassVar[MechanismMeta] = MechanismMeta(
        algorithm_id="NUDGE-MECH-001",
        role=MechanismRole.REGULATORY_EDGE,
        summary="Linear regulatory effect f(x)=w*x (the linear-baseline edge).",
        references=("Yuan2021CellBox",),
    )

    def response(self, x: Array) -> Array:
        """Regulatory contribution of activity ``x``."""
        return self.weight * x


@default_registry.register("HillActivation")
@dataclass(frozen=True)
class HillActivationEffect:
    """Activating Hill response ``f(x) = vmax * x^n / (K^n + x^n)``.

    ``K`` is the threshold (half-max input), ``n`` the gain (Hill coefficient;
    ``n > 1`` is ultrasensitive/switch-like), ``vmax`` the ceiling.
    """

    K: float
    n: float = 1.0
    vmax: float = 1.0

    meta: ClassVar[MechanismMeta] = MechanismMeta(
        algorithm_id="NUDGE-MECH-002",
        role=MechanismRole.REGULATORY_EDGE,
        summary="Hill activation; K=threshold, n=gain, vmax=ceiling.",
        assumptions=("quasi-steady-state binding", "a single cooperative site set"),
        references=("HuangFerrell1996", "Das2009"),
    )

    def response(self, x: Array) -> Array:
        """Activating contribution of activity ``x``."""
        xn = jnp.power(x, self.n)
        return self.vmax * xn / (jnp.power(self.K, self.n) + xn)


@default_registry.register("HillRepression")
@dataclass(frozen=True)
class HillRepressionEffect:
    """Repressing Hill response ``f(x) = vmax * K^n / (K^n + x^n)``."""

    K: float
    n: float = 1.0
    vmax: float = 1.0

    meta: ClassVar[MechanismMeta] = MechanismMeta(
        algorithm_id="NUDGE-MECH-003",
        role=MechanismRole.REGULATORY_EDGE,
        summary="Hill repression; K=threshold, n=gain, vmax=ceiling.",
        assumptions=("quasi-steady-state binding",),
        references=("HuangFerrell1996",),
    )

    def response(self, x: Array) -> Array:
        """Repressing contribution of activity ``x``."""
        kn = jnp.power(self.K, self.n)
        return self.vmax * kn / (kn + jnp.power(x, self.n))
