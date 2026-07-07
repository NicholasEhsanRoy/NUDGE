"""The decoy battery — each case is a green/red CI test where NUDGE must return
the correct *negative* / abstention verdict.

Phase-0: the battery is empty, so the parametrized test collects zero cases; the
schema check proves the harness is wired. The nine adversarial cases (and any
AI-generated ones, creative-AI idea 1) land in Phase 3.
"""

from __future__ import annotations

import pytest

from nudge.data.decoys import DECOY_BATTERY, DecoyCase

pytestmark = pytest.mark.decoy


def test_battery_is_a_registry_of_decoycases() -> None:
    assert isinstance(DECOY_BATTERY, list)
    assert all(isinstance(c, DecoyCase) for c in DECOY_BATTERY)


@pytest.mark.parametrize("case", DECOY_BATTERY, ids=lambda c: c.decoy_id)
def test_decoy_returns_expected_verdict(case: DecoyCase) -> None:
    # Phase 3: assert classify(fit(case.generate())) == case.expected_verdict
    pytest.skip("decoy battery is populated in Phase 3")
