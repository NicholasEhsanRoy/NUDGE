"""Named circuit motifs — reusable bistable switches for tests + real-data attribution.

Small, hand-built :class:`~nudge.core.circuit.Circuit` instances in NUDGE's
``SpeciesDef``/``EdgeDef`` vocabulary. The two Ras motifs are the **candidate
topologies**
for the Gladstone T-cell validation: NUDGE does not presume which one the biology uses —
model selection (``inference.model_select``) fits both and lets the data + a parsimony
penalty choose (a 2-node model must *earn* its extra parameter over the 1-node one).

The Ras activation switch (Das 2009): RASGRP1 (graded input GEF) primes Ras-GTP; SOS1
(digital GEF) is allosterically activated *by* Ras-GTP → positive feedback →
bistability.
As a small circuit this is either a **1-node self-activation** switch (the activation
program self-amplifies) or a **2-node mutual-activation** switch (SOS ⇄ Ras-GTP).
"""

from __future__ import annotations

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef

__all__ = ["ras_switch_1node", "ras_switch_2node", "toggle"]


def toggle(
    *, n: float = 4.0, vmax: float = 2.0, K: float = 1.0, basal: float = 0.05
) -> Circuit:
    """2-node mutual-**repression** toggle (Gardner–Collins): anti-correlated modes."""
    return Circuit(
        [
            SpeciesDef("A", basal=basal, decay=1.0),
            SpeciesDef("B", basal=basal, decay=1.0),
        ],
        [
            EdgeDef(1, 0, "hill_repression", K=K, n=n, vmax=vmax),  # B ⊣ A
            EdgeDef(0, 1, "hill_repression", K=K, n=n, vmax=vmax),  # A ⊣ B
        ],
    )


def ras_switch_1node(
    *, n: float = 6.0, vmax: float = 2.0, K: float = 1.0, basal: float = 0.05
) -> Circuit:
    """1-node self-activation switch — the activation program amplifying itself.

    One "Activation" species with a self ``hill_activation`` edge (the SOS positive
    feedback). Bistable (low/high) for cooperative ``n``; the low/high lobes are the
    resting/activated T-cell states read out by the IEG panel. Perturbations move the
    self-edge: gain = ``n`` (cooperativity), threshold = ``K``, ceiling = ``vmax``.
    """
    return Circuit(
        [SpeciesDef("Activation", basal=basal, decay=1.0)],
        [EdgeDef(0, 0, "hill_activation", K=K, n=n, vmax=vmax)],
    )


def ras_switch_2node(
    *, n: float = 4.0, vmax: float = 2.0, K: float = 1.0, basal: float = 0.05
) -> Circuit:
    """2-node mutual-**activation** switch — SOS ⇄ Ras-GTP positive feedback.

    Two species (``RasGTP``, ``SOS``) each ``hill_activation``-driving the other, so the
    lobes are *correlated* (both-low / both-high) — the co-activation form.
    ``target_edge=0`` (SOS → RasGTP) carries the attributable gain/threshold/ceiling.
    """
    return Circuit(
        [
            SpeciesDef("RasGTP", basal=basal, decay=1.0),
            SpeciesDef("SOS", basal=basal, decay=1.0),
        ],
        [
            EdgeDef(1, 0, "hill_activation", K=K, n=n, vmax=vmax),  # SOS → RasGTP
            EdgeDef(0, 1, "hill_activation", K=K, n=n, vmax=vmax),  # RasGTP → SOS
        ],
    )
