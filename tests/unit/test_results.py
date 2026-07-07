"""The MechanismMap output schema round-trips through serialization."""

from __future__ import annotations

from nudge.core.results import MechanismCall, MechanismMap
from nudge.core.vocabulary import MechanismClass


def test_mechanism_map_roundtrip() -> None:
    m = MechanismMap(
        calls=[
            MechanismCall(
                perturbation="SOS",
                mechanism=MechanismClass.THRESHOLD,
                confidence=0.9,
                intervals={"K": (0.4, 0.6)},
            ),
            MechanismCall(
                perturbation="RASGRP1",
                mechanism=MechanismClass.UNRESOLVED,
                confidence=0.2,
                rationale="posteriors overlap",
            ),
        ],
        beats_linear_baseline=True,
    )
    restored = MechanismMap.model_validate(m.model_dump())
    assert restored.calls[0].mechanism is MechanismClass.THRESHOLD
    assert restored.calls[1].mechanism is MechanismClass.UNRESOLVED
    assert restored.beats_linear_baseline is True
