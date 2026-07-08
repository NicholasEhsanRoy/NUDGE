"""The decoy battery — each case is a green/red CI test where NUDGE must return
the correct *negative* / abstention verdict.

The schema check is fast; the per-case fit assertions are ``slow`` (a full fit +
the transition-mode path). A decoy passes only if NUDGE abstains on BOTH the
single-basin and the powerful transition-mode path — a decoy that fools the
stronger engine is not caught by the weaker one.
"""

from __future__ import annotations

import pytest

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef
from nudge.core.vocabulary import POSITIVE_CLASSES, MechanismClass
from nudge.data.decoys import DECOY_BATTERY, DecoyCase
from nudge.inference.fit import fit, fit_multibasin

pytestmark = pytest.mark.decoy


def _switch_hypothesis() -> Circuit:
    # The bistable-switch hypothesis a naive bimodality detector would try to fit.
    return Circuit(
        [SpeciesDef("SW", basal=0.05, decay=1.0)],
        [EdgeDef(0, 0, "hill_activation", K=1.0, n=6.0, vmax=2.0)],
    )


def test_battery_is_a_registry_of_decoycases() -> None:
    assert isinstance(DECOY_BATTERY, list)
    assert all(isinstance(c, DecoyCase) for c in DECOY_BATTERY)


@pytest.mark.slow
@pytest.mark.parametrize("case", DECOY_BATTERY, ids=lambda c: c.decoy_id)
def test_decoy_returns_expected_verdict(case: DecoyCase) -> None:
    adata = case.generate()
    circuit = _switch_hypothesis()
    # Single-basin path: every call must be the expected abstention verdict.
    mm = fit(adata, circuit, n_cells=384, steps=400, margin_k=1.7, seed=0)
    for call in mm.calls:
        assert call.mechanism is case.expected_verdict, (
            f"{case.decoy_id}: single-basin returned {call.mechanism.value}, "
            f"expected {case.expected_verdict.value}"
        )
    # Powerful transition-mode path must not be fooled into any positive attribution.
    mm_t = fit_multibasin(
        adata, circuit, n_cells=384, steps=400, margin_k=1.7,
        transition_mode=True, seed=0,
    )
    assert not any(c.mechanism in POSITIVE_CLASSES for c in mm_t.calls), (
        f"{case.decoy_id}: transition-mode path emitted a spurious positive"
    )
    # Sanity: OFF_MODEL decoys must genuinely fail the parsimony gate.
    if case.expected_verdict is MechanismClass.OFF_MODEL:
        assert not mm.beats_linear_baseline
