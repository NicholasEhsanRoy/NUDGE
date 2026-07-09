"""The NUDGE mechanism library.

Importing this package **deterministically populates** the default mechanism
registry: each submodule below registers its mechanism(s) as an import side
effect, so ``default_registry.list()`` is complete regardless of which entry
point pulled the package in. (Previously registration was incidental — whichever
modules the circuit happened to import — which silently dropped
``LinearIntegrator`` from the registry.)
"""

from __future__ import annotations

from nudge.mechanisms import readout as _readout  # noqa: F401
from nudge.mechanisms import regulatory as _regulatory  # noqa: F401
from nudge.mechanisms.integrators import linear as _linear  # noqa: F401
from nudge.mechanisms.integrators import saturating as _saturating  # noqa: F401
from nudge.mechanisms.registry import MechanismRegistry, default_registry

__all__ = ["MechanismRegistry", "default_registry"]
