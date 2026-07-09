"""End-to-end attribution pipeline (Phase 4 M4 scaffold) on a synthetic fixture.

Validates the real-data plumbing before the real data lands: raw counts (Gladstone-ish
obs) → activity bridge → per-operating-point single call + the multi-op joint call, with
honest **skip** accounting for operating points that are unusable (too few cells / an
unreliable LNA). The exact mechanism call is noisy on synthetic counts and covered by
Lyapunov tests; here we check the orchestration, the guard, and the report shape.
"""

from __future__ import annotations

import anndata as ad
import jax
import numpy as np
import pandas as pd
import pytest

from nudge.circuits import ras_switch_1node
from nudge.inference.lyapunov import sample_lna_mixture
from nudge.inference.pipeline import attribute_across_operating_points

_MARKERS = {"Activation": ("IL2", "CD69", "EGR1")}
_GENES = ["IL2", "CD69", "EGR1", "ACTB"]


def _op_adata(
    n_wt: int, n_target: int, *, basal: float = 0.05, seed: int = 0
) -> ad.AnnData:
    """One operating point: WT + target cells; IEG counts track a bimodal activation."""
    key = jax.random.PRNGKey(seed)
    kw, kt = jax.random.split(key)
    wt = sample_lna_mixture(ras_switch_1node(basal=basal), n_wt, kw, scale=1.0)
    tg = sample_lna_mixture(
        ras_switch_1node(basal=basal, n=3.0), n_target, kt, scale=1.0
    )
    act = np.clip(np.concatenate([wt, tg])[:, 0], 0, None)  # (n_wt+n_target,)
    rng = np.random.default_rng(seed)
    lib = rng.integers(1500, 3000, size=act.size).astype(float)
    x = np.zeros((act.size, len(_GENES)), dtype=np.int32)
    for g in range(3):  # 3 IEG markers ~ Poisson(activity × depth)
        x[:, g] = rng.poisson(act * lib / 80.0)
    x[:, 3] = rng.poisson(lib / 100.0)  # a depth-only non-marker gene
    cond = ["WT"] * n_wt + ["SOS1"] * n_target
    obs = pd.DataFrame(
        {"condition": pd.Categorical(cond), "total_counts": lib},
        index=pd.Index([f"c{i}" for i in range(act.size)]),
    )
    return ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=pd.Index(_GENES)))


@pytest.mark.slow
def test_pipeline_reports_across_operating_points() -> None:
    ops = {
        "Stim8hr": _op_adata(1500, 1500, basal=0.05, seed=0),
        "Stim48hr": _op_adata(1500, 1500, basal=0.15, seed=1),  # a 2nd operating point
        "Rest": _op_adata(1500, 40, basal=0.05, seed=2),  # too few target cells
    }
    rep = attribute_across_operating_points(
        ops, ras_switch_1node(), _MARKERS, "SOS1", steps=150, min_cells=200
    )
    assert rep.target == "SOS1"
    assert rep.n_cells == {"Stim8hr": 1500, "Stim48hr": 1500, "Rest": 40}
    assert "Rest" in rep.skipped  # too few cells, recorded not dropped
    assert set(rep.single) == {"Stim8hr", "Stim48hr"}  # the two usable ops got a call
    for _label, (call, _nlls) in rep.single.items():
        assert call in {"gain_or_threshold", "ceiling", "unresolved"}  # never bare
    assert rep.multi is not None  # ≥2 usable ops → the joint breaker ran


@pytest.mark.slow
def test_pipeline_abstains_on_low_depth() -> None:
    # Very shallow counts → lna_reliable trips → the op is skipped, not force-called.
    op = _op_adata(1500, 1500, seed=0)
    op.X = (np.asarray(op.X) // 50).astype(np.int32)  # crush the depth
    op.obs["total_counts"] = np.asarray(op.obs["total_counts"]) / 50.0
    rep = attribute_across_operating_points(
        {"shallow": op}, ras_switch_1node(), _MARKERS, "SOS1", steps=120
    )
    assert "shallow" in rep.skipped and rep.multi is None
