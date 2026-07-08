"""The telegraph decoy generator: bimodal snapshot, monostable deterministic system.

Fast guard (no fit) — the full decoy assertion (NUDGE abstains) is the slow
battery test in ``tests/decoys/test_battery.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from nudge.data.decoys import DECOY_BATTERY
from nudge.data.ingest import check_counts
from nudge.data.stochastic import generate_telegraph_perturbseq

pytestmark = pytest.mark.verification


def test_telegraph_is_bimodal_but_monostable() -> None:
    adata = generate_telegraph_perturbseq(n_cells_per_condition=1500, seed=0)
    check_counts(adata)  # raw non-negative integer counts
    # Deterministically monostable — no real switch to attribute (the decoy premise).
    assert adata.uns["ground_truth"]["deterministically_monostable"] is True
    assert len(adata.uns["ground_truth"]["mean_field_fixed_points"]) == 1
    # ...yet the WT snapshot is clearly bimodal (a low/OFF spike + a populated ON mode).
    wt = np.asarray(adata[adata.obs["condition"] == "WT"].X).ravel()
    frac_low = float((wt <= 1).mean())
    frac_high = float((wt >= 8).mean())
    assert frac_low > 0.2, f"no OFF spike (frac_low={frac_low:.2f})"
    assert frac_high > 0.1, f"no ON mode (frac_high={frac_high:.2f})"


def test_telegraph_decoy_registered() -> None:
    ids = {c.decoy_id for c in DECOY_BATTERY}
    assert "NUDGE-DECOY-001" in ids
