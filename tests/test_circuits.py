"""Named circuit motifs — both Ras candidate topologies must be bistable (Phase 4 M2).

The Lyapunov attribution needs ≥2 stable modes; a Ras candidate that isn't bistable
can't carry the switch. 1-node is a 1-D low/high switch; 2-node is *correlated*
(co-activation); the toggle is *anti-correlated* — sanity that each one builds.
"""

from __future__ import annotations

import numpy as np

from nudge.circuits import ras_switch_1node, ras_switch_2node, toggle


def test_all_motifs_bistable_and_psd() -> None:
    for build in (ras_switch_1node, ras_switch_2node, toggle):
        modes = build().mode_covariances()
        assert modes is not None and len(modes) == 2, build.__name__
        for _mean, cov in modes:
            assert (np.linalg.eigvalsh(cov) > 0).all()  # positive-definite lobe


def test_topology_shapes() -> None:
    m1 = ras_switch_1node().mode_covariances()
    assert m1 is not None and all(mean.shape == (1,) for mean, _ in m1)  # 1-D switch

    m2 = ras_switch_2node().mode_covariances()
    assert m2 is not None
    low = min(m2, key=lambda mc: float(mc[0].sum()))[0]
    assert abs(float(low[0]) - float(low[1])) < 0.1  # co-activation: both species low

    tog = toggle().mode_covariances()
    assert tog is not None
    lobe = min(tog, key=lambda mc: float(mc[0][0]))[0]
    assert abs(float(lobe[0]) - float(lobe[1])) > 0.5  # anti-correlated: hi/lo
