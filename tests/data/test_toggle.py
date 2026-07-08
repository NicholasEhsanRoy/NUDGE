"""The 2-node toggle-switch generator: emergent bimodality across both attractors.

Fast guard (no fit). The full N-D attribution test (does the saddle gain gate recover
gain on a toggle) is the slow toggle guard test in tests/verification/.
"""

from __future__ import annotations

import numpy as np
import pytest

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef
from nudge.data.ingest import check_counts
from nudge.data.stochastic import generate_toggle_perturbseq

pytestmark = pytest.mark.verification


def _toggle() -> Circuit:
    return Circuit(
        [
            SpeciesDef("A", basal=0.05, decay=1.0),
            SpeciesDef("B", basal=0.05, decay=1.0),
        ],
        [
            EdgeDef(1, 0, "hill_repression", K=1.0, n=4.0, vmax=2.0),
            EdgeDef(0, 1, "hill_repression", K=1.0, n=4.0, vmax=2.0),
        ],
    )


def test_toggle_is_emergently_bimodal_two_genes() -> None:
    adata = generate_toggle_perturbseq(_toggle(), n_cells_per_condition=1500, seed=0)
    check_counts(adata)  # raw non-negative integer counts
    assert adata.shape[1] == 2 and list(adata.var_names) == ["A", "B"]
    assert adata.uns["ground_truth"]["tier"] == "0.5-toggle"
    wt = np.asarray(adata[adata.obs["condition"] == "WT"].X)
    # both mutually-exclusive attractors populated: (A-hi, B-lo) and (B-hi, A-lo)
    a_hi = float(((wt[:, 0] > wt[:, 1]) & (wt[:, 0] >= 5)).mean())
    b_hi = float(((wt[:, 1] > wt[:, 0]) & (wt[:, 1] >= 5)).mean())
    assert a_hi > 0.2, f"A-high attractor underpopulated ({a_hi:.2f})"
    assert b_hi > 0.2, f"B-high attractor underpopulated ({b_hi:.2f})"
