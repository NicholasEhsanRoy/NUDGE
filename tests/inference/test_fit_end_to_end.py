"""End-to-end fit() → MechanismMap: attribution + the linear-baseline parsimony gate.

Marked slow (it runs the full fit orchestration — a WT fit plus restricted fits per
condition). Runs in the scheduled lane, not on every PR.
"""

from __future__ import annotations

import pytest

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef
from nudge.core.vocabulary import MechanismClass
from nudge.data.synthetic import PerturbationSpec, generate_synthetic_perturbseq
from nudge.inference.fit import fit

pytestmark = [pytest.mark.verification, pytest.mark.slow]


def _switch() -> Circuit:
    return Circuit(
        [
            SpeciesDef("IN", basal=1.0, decay=1.0),
            SpeciesDef("SW", basal=0.05, decay=1.0),
        ],
        [EdgeDef(0, 1, "hill_activation", K=1.0, n=6.0, vmax=2.0)],
    )


def test_fit_attributes_threshold_gain_ceiling() -> None:
    perts = [
        PerturbationSpec("thr", "edge", 0, "K", 3.0),  # threshold
        PerturbationSpec("gai", "edge", 0, "n", 0.2),  # gain
        PerturbationSpec("cei", "edge", 0, "vmax", 0.3),  # ceiling
    ]
    adata = generate_synthetic_perturbseq(
        _switch(), perts, n_cells_per_condition=3000, realism_level=1, seed=0
    )
    mm = fit(adata, _switch(), n_cells=384, steps=400, seed=0)
    calls = {c.perturbation: c.mechanism for c in mm.calls}
    assert calls["thr"] is MechanismClass.THRESHOLD
    assert calls["gai"] is MechanismClass.GAIN
    assert calls["cei"] is MechanismClass.CEILING
    assert mm.beats_linear_baseline


def test_fit_linear_data_returns_off_model() -> None:
    # The linear-baseline gate: data from a LINEAR circuit has no switch, so the
    # mechanistic hypothesis must NOT invent one — every call is off-model.
    linear_circuit = Circuit(
        [
            SpeciesDef("IN", basal=1.0, decay=1.0),
            SpeciesDef("SW", basal=0.5, decay=1.0),
        ],
        [EdgeDef(0, 1, "linear", weight=1.0)],
    )
    adata = generate_synthetic_perturbseq(
        linear_circuit,
        [PerturbationSpec("lin", "edge", 0, "weight", 0.4)],
        n_cells_per_condition=2000, realism_level=1, seed=0,
    )
    mm = fit(adata, _switch(), seed=0)  # fit with the mechanistic (switch) hypothesis
    assert all(c.mechanism is MechanismClass.OFF_MODEL for c in mm.calls)
    assert not mm.beats_linear_baseline
