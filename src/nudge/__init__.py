"""NUDGE — mechanism attribution for Perturb-seq screens.

NUDGE fits a compositional, differentiable gene-regulatory circuit model to
single-cell perturbation data and classifies each perturbation by *mechanism*
— does it move a switch's threshold, its gain, or its ceiling — with explicit
abstention when the data cannot say. Built on MADDENING.

See ``design/WORKING_BACKWARDS.md`` for the full rationale.
"""

from __future__ import annotations

from nudge.core.builder import CircuitBuilder
from nudge.core.circuit import Circuit
from nudge.core.results import MechanismCall, MechanismMap
from nudge.core.vocabulary import MechanismClass
from nudge.data.synthetic import PerturbationSpec, generate_synthetic_perturbseq
from nudge.design.invert import design
from nudge.inference.fit import fit

__version__ = "0.1.0"

__all__ = [
    "Circuit",
    "CircuitBuilder",
    "MechanismCall",
    "MechanismClass",
    "MechanismMap",
    "PerturbationSpec",
    "__version__",
    "design",
    "fit",
    "generate_synthetic_perturbseq",
]
