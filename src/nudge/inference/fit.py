"""The fit engine entry point.

``fit(adata, circuit)`` runs a minibatch optax loop over cells, fitting the
circuit's kinetic parameters and per-cell parameter distribution to the observed
single-cell count distribution (a distributional loss — energy distance / MMD —
not a mean fit), then classifies each perturbation into a ``MechanismClass`` with
Laplace uncertainty. Phase-0 stub.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nudge.core.results import MechanismMap

if TYPE_CHECKING:
    from nudge.core.circuit import Circuit

__all__ = ["fit"]


def fit(adata: Any, circuit: Circuit) -> MechanismMap:
    """Fit ``circuit`` to ``adata`` (raw-count Perturb-seq) → a ``MechanismMap``.

    ``adata`` must carry raw integer counts; ``nudge.data.ingest`` enforces this
    at the boundary. Phase-0 stub.
    """
    raise NotImplementedError("fit — Phase 2")
