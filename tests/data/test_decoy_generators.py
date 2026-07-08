"""Decoy generators must produce genuinely bimodal, raw-count data — otherwise the
battery's "NUDGE declines" assertion is vacuous. The slow "NUDGE abstains" checks
live in tests/decoys/test_battery.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from nudge.data.decoy_generators import generate_dropout_decoy, generate_mixture_decoy
from nudge.data.decoys import DECOY_BATTERY
from nudge.data.ingest import check_counts

pytestmark = pytest.mark.verification


def _wt_counts(adata) -> np.ndarray:
    return np.asarray(adata[adata.obs["condition"] == "WT"].X).ravel()


def test_mixture_decoy_is_bimodal_raw_counts() -> None:
    adata = generate_mixture_decoy(n_cells_per_condition=1500, seed=0)
    check_counts(adata)
    wt = _wt_counts(adata)
    assert float((wt <= 1).mean()) > 0.2  # OFF-like spike
    assert float((wt >= 8).mean()) > 0.2  # ON-like mode


def test_dropout_decoy_is_bimodal_raw_counts() -> None:
    adata = generate_dropout_decoy(n_cells_per_condition=1500, seed=0)
    check_counts(adata)
    wt = _wt_counts(adata)
    assert float((wt <= 1).mean()) > 0.2  # depth-driven zero peak
    assert float((wt >= 5).mean()) > 0.15  # the expression mode


def test_battery_covers_diverse_decoys() -> None:
    ids = {c.decoy_id for c in DECOY_BATTERY}
    assert {f"NUDGE-DECOY-00{i}" for i in range(1, 6)} <= ids
    # Every case cross-references a known limitation.
    assert all(c.limitation_ref for c in DECOY_BATTERY)
