"""Synthetic Perturb-seq generator — the CI backbone.

``generate_synthetic_perturbseq(...)`` runs a NUDGE circuit as a *generative*
model: per-cell parameters (extrinsic noise) → vmapped steady-state solve →
``Readout`` link → negative-binomial counts, returning an ``AnnData`` with the
ground-truth parameters and mechanism labels stashed in ``.uns['ground_truth']``.
Because the circuit is itself the generator, we get a ground-truth simulator for
free. Phase-0 stub.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nudge.core.circuit import Circuit

__all__ = ["generate_synthetic_perturbseq"]


def generate_synthetic_perturbseq(
    circuit: Circuit,
    conditions: Any,
    *,
    n_cells_per_condition: int = 1000,
    noise_model: Any = None,
    realism_level: int = 1,
    seed: int = 0,
) -> Any:
    """Generate a synthetic Perturb-seq ``AnnData`` with ground truth in ``.uns``.

    ``realism_level`` is the 0–3 difficulty dial: 0 exact model, 1 +count noise,
    2 +extrinsic heterogeneity, 3 +misspecification (the honest, inverse-crime-free
    test). Phase-0 stub.
    """
    raise NotImplementedError("generate_synthetic_perturbseq — Phase 1")
