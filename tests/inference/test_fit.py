"""The fit engine recovers ground-truth K (threshold) and n (gain) from scramble."""

from __future__ import annotations

import numpy as np
import pytest

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef
from nudge.data.ingest import IngestError
from nudge.data.synthetic import generate_synthetic_perturbseq
from nudge.inference.fit import fit_parameters

pytestmark = pytest.mark.verification


def _true_circuit() -> Circuit:
    return Circuit(
        [
            SpeciesDef("IN", basal=1.0, decay=1.0),
            SpeciesDef("SW", basal=0.05, decay=1.0),
        ],
        [EdgeDef(0, 1, "hill_activation", K=1.0, n=6.0, vmax=2.0)],
    )


def test_recovers_threshold_and_gain_from_scrambled_init() -> None:
    true = _true_circuit()
    adata = generate_synthetic_perturbseq(
        true, [], n_cells_per_condition=2000, realism_level=1, seed=0
    )
    # Start far from the truth: K 1.0 → 2.5, n 6.0 → 3.0.
    scrambled = Circuit(
        true.species, [EdgeDef(0, 1, "hill_activation", K=2.5, n=3.0, vmax=2.0)]
    )
    free = [("edge", 0, "K"), ("edge", 0, "n")]
    recovered, history = fit_parameters(
        adata, scrambled, free, condition="WT", steps=300, seed=1
    )
    assert recovered[("edge", 0, "K")] == pytest.approx(1.0, abs=0.35)  # true 1.0
    assert 4.5 < recovered[("edge", 0, "n")] < 7.5  # true 6.0
    assert history[-1] < 0.2 * history[0]  # the fit converged


def test_fit_rejects_non_raw_input() -> None:
    # The bouncer (check_counts) is wired into the fit boundary.
    adata = generate_synthetic_perturbseq(
        _true_circuit(), [], n_cells_per_condition=200, realism_level=1, seed=0
    )
    adata.X = np.log1p(adata.X).astype(np.float32)
    with pytest.raises(IngestError):
        fit_parameters(adata, _true_circuit(), [("edge", 0, "K")], steps=1)
