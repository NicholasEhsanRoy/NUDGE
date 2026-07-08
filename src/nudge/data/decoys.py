"""The decoy battery — synthetic negatives engineered to look positive.

Each ``DecoyCase`` is a dataset a naive method would call a confident hit; the
pass condition is that NUDGE returns the correct *negative* / abstention verdict.
The registry lives here (in ``src``, not ``tests``) so Mechanism Cards and
``known_limitations.yaml`` can cross-reference decoy IDs. It accepts human- and
AI-authored generators (creative-AI idea 1) via ``authored_by`` / ``prompt_ref``.

Phase-0: the schema + an empty battery; the nine cases land in Phase 3.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from nudge.core.vocabulary import MechanismClass
from nudge.data.decoy_generators import (
    generate_dropout_decoy,
    generate_mixture_decoy,
)
from nudge.data.stochastic import generate_telegraph_perturbseq


@dataclass(frozen=True)
class DecoyCase:
    """A single adversarial dataset and the verdict NUDGE must return."""

    decoy_id: str  # e.g. "NUDGE-DECOY-001"
    summary: str
    generate: Callable[[], Any]  # () -> AnnData
    expected_verdict: MechanismClass
    limitation_ref: str = ""  # the NUDGE-LIM-* it maps to
    authored_by: Literal["human", "ai"] = "human"
    prompt_ref: str = ""  # for AI-authored decoys: prompt/model hash (idea 1)


def _telegraph_decoy() -> Any:
    """Bimodal-but-monostable telegraph data (To & Maheshri 2010) — not a switch."""
    return generate_telegraph_perturbseq(n_cells_per_condition=2000, seed=0)


def _mixture_decoy() -> Any:
    """Two-population (cell-type / doublet) mixture faking bimodality — not a switch."""
    return generate_mixture_decoy(n_cells_per_condition=2000, seed=0)


def _dropout_decoy() -> Any:
    """Technical dropout zero-peak on a monostable population — not a switch."""
    return generate_dropout_decoy(n_cells_per_condition=2000, seed=0)


#: The decoy battery. Grows with the mechanism library and the AI decoy generator.
DECOY_BATTERY: list[DecoyCase] = [
    DecoyCase(
        decoy_id="NUDGE-DECOY-001",
        summary=(
            "Noise-induced bimodality without bistability: a non-cooperative "
            "positive-feedback loop with slow promoter switching is deterministically "
            "monostable yet bimodal. NUDGE must return off-model (not-a-switch)."
        ),
        generate=_telegraph_decoy,
        expected_verdict=MechanismClass.OFF_MODEL,
        limitation_ref="NUDGE-LIM-001",
    ),
    DecoyCase(
        decoy_id="NUDGE-DECOY-002",
        summary=(
            "Two-population mixture (cell types / doublets): a static mix of a low- "
            "and a high-expressing subpopulation is bimodal but has no switch. "
            "NUDGE must return off-model, not attribute ultrasensitivity."
        ),
        generate=_mixture_decoy,
        expected_verdict=MechanismClass.OFF_MODEL,
        limitation_ref="NUDGE-LIM-002",
    ),
    DecoyCase(
        decoy_id="NUDGE-DECOY-003",
        summary=(
            "Dropout zero-peak: a monostable population read out at bimodal library "
            "depth (a fraction captured at very low depth → near-all-zeros) mimics an "
            "OFF/ON switch. The bimodality is a measurement artifact — off-model."
        ),
        generate=_dropout_decoy,
        expected_verdict=MechanismClass.OFF_MODEL,
        limitation_ref="NUDGE-LIM-003",
    ),
]
