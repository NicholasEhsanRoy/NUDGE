"""Counts → activity bridge (Phase 4 M2): real counts → the Lyapunov path's input.

Checks the two jobs: (1) ``counts_to_activity`` reduces a panel to per-species activity
and **divides out per-cell sequencing depth** (a 2× deeper cell reads the same); (2)
``adata_to_operating_point`` turns one condition of an AnnData into a valid
``OperatingPoint`` (activity + WT-calibrated scale/obs) — the AnnData → attribution.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from nudge.circuits import ras_switch_1node
from nudge.inference.bridge import adata_to_operating_point, counts_to_activity

_MARKERS = {"Activation": ("IL2", "CD69", "EGR1")}
_GENES = ["IL2", "CD69", "EGR1", "ACTB"]


def _bimodal_adata(n: int = 200, seed: int = 0) -> ad.AnnData:
    """A 1-D activation switch read out by 3 IEG genes: half resting, half activated."""
    rng = np.random.default_rng(seed)
    levels = np.where(np.arange(n) < n // 2, 1.0, 15.0)  # low / high activation
    lib = rng.integers(800, 1600, size=n).astype(float)  # per-cell depth varies
    x = np.zeros((n, len(_GENES)), dtype=np.int32)
    for g in range(3):  # the 3 IEG markers track activation × depth
        x[:, g] = rng.poisson(levels * lib / 1000.0)
    x[:, 3] = rng.poisson(lib / 100.0)  # ACTB: a depth-only gene (not a marker)
    cond = np.where(np.arange(n) < n // 4, "WT", "SOS1")  # mix WT + one condition
    obs = pd.DataFrame(
        {"condition": pd.Categorical(cond), "total_counts": lib},
        index=pd.Index([f"c{i}" for i in range(n)]),
    )
    return ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=pd.Index(_GENES)))


def test_activity_shape_and_bimodal() -> None:
    adata = _bimodal_adata()
    act = counts_to_activity(adata, ras_switch_1node(), _MARKERS)
    assert act.shape == (adata.n_obs, 1)  # 1 species column
    # the two activation levels are clearly separated
    lo = act[np.asarray(adata.obs.index.str[1:].astype(int)) < adata.n_obs // 2]
    hi = act[np.asarray(adata.obs.index.str[1:].astype(int)) >= adata.n_obs // 2]
    assert float(hi.mean()) > 3 * float(lo.mean())


def test_depth_is_divided_out() -> None:
    adata = _bimodal_adata()
    act = counts_to_activity(adata, ras_switch_1node(), _MARKERS)
    deep = adata.copy()  # make cell 0 twice as deep (counts + library both ×2)
    deep.X = np.asarray(deep.X).copy()
    deep.X[0] *= 2
    deep.obs["total_counts"] = deep.obs["total_counts"].to_numpy().copy()
    deep.obs.loc[deep.obs.index[0], "total_counts"] *= 2
    act2 = counts_to_activity(deep, ras_switch_1node(), _MARKERS)
    assert np.allclose(act2[0], act[0], rtol=1e-6)  # depth divided out


def test_missing_marker_raises() -> None:
    adata = _bimodal_adata()
    with pytest.raises(KeyError, match="NOPE"):
        counts_to_activity(adata, ras_switch_1node(), {"Activation": ("IL2", "NOPE")})


@pytest.mark.slow
def test_adata_to_operating_point_calibrates() -> None:
    adata = _bimodal_adata()
    op = adata_to_operating_point(adata, ras_switch_1node(), _MARKERS, "SOS1")
    assert op.data.shape[1] == 1 and op.data.shape[0] > 0
    assert op.scale > 0 and op.obs_sd > 0  # WT-calibrated depth + noise
    assert op.circuit.n_species == 1
