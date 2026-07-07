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


#: The decoy battery. Populated in Phase 3 and by ``scripts/ai/generate_decoy.py``.
DECOY_BATTERY: list[DecoyCase] = []
