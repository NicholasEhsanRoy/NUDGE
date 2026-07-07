"""The attribution vocabulary NUDGE can return for a perturbation.

Deliberately richer than ``{threshold, gain, ceiling}``: the abstention
classes are first-class, because the product's core promise is to *fail
safely and loudly* rather than emit a confident false positive. The decoy
battery (``nudge.data.decoys``) exists to force these classes to be used.
"""

from __future__ import annotations

from enum import Enum


class MechanismClass(str, Enum):
    """How a perturbation acts on a switch — or why NUDGE declined to say."""

    # --- Positive attributions ---
    THRESHOLD = "threshold"  # moves where the switch trips
    GAIN = "gain"  # changes how sharply it commits
    CEILING = "ceiling"  # changes the maximal output
    COMBO = "combo"  # a combination of the above

    # --- Abstention / negative attributions (the fail-loud vocabulary) ---
    NO_EFFECT = "no-effect"  # perturbation-strength latent ~ 0
    UNRESOLVED = "unresolved"  # overlapping posteriors; cannot decide
    TECHNICAL_ARTIFACT = "technical-artifact"  # population/technical, not a switch
    OFF_MODEL = "off-model"  # poor global fit / high residual → distrust


POSITIVE_CLASSES = frozenset(
    {
        MechanismClass.THRESHOLD,
        MechanismClass.GAIN,
        MechanismClass.CEILING,
        MechanismClass.COMBO,
    }
)
ABSTENTION_CLASSES = frozenset(MechanismClass) - POSITIVE_CLASSES
