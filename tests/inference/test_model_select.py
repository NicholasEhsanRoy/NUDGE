"""Topology model-selection (Phase 4 M3): let the data + BIC choose the circuit.

The gate must (a) abstain — pick the no-switch null — on unimodal data (no bistability);
(b) pick the mechanistic switch when the data is genuinely bimodal from it; and
(c) not over-select complexity — a switch wins only when its likelihood gain beats the
BIC penalty for its extra parameters.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from nudge.circuits import ras_switch_1node, ras_switch_2node
from nudge.inference.lyapunov import sample_lna_mixture
from nudge.inference.model_select import Candidate, select_topology

SCALE, OBS_SD = 20.0, 0.5
_KIN1 = [("edge", 0, "n"), ("edge", 0, "K"), ("edge", 0, "vmax")]
_KIN2 = _KIN1 + [("edge", 1, "n"), ("edge", 1, "K"), ("edge", 1, "vmax")]


def _cand1() -> Candidate:
    return Candidate("1-node", ras_switch_1node(), _KIN1)


def _cand2() -> Candidate:
    return Candidate("2-node", ras_switch_2node(), _KIN2)


@pytest.mark.slow
def test_unimodal_data_selects_no_switch() -> None:
    # One Gaussian blob, no bistability → the null must win (fail-safe "no switch").
    rng = np.random.default_rng(0)
    data = rng.normal(10.0, 1.5, size=(2000, 1))
    res = select_topology(data, [_cand1()], steps=150)
    assert res.selected == "no-switch" and res.is_switch is False
    # the switch either loses on BIC or is unstable to fit on a blob — never selected.
    assert "1-node" not in res.bic or res.bic["1-node"] > res.bic["no-switch"]


@pytest.mark.slow
def test_1node_bimodal_data_selects_1node() -> None:
    # Data genuinely from the 1-node switch → it must beat the null by BIC.
    data = sample_lna_mixture(
        ras_switch_1node(), 3000, jax.random.PRNGKey(0), scale=SCALE, obs_sd=OBS_SD
    )
    res = select_topology(data, [_cand1()], steps=150)
    assert res.selected == "1-node" and res.is_switch is True
    assert res.bic["1-node"] < res.bic["no-switch"]


@pytest.mark.slow
def test_2node_2d_data_selects_2node_over_null() -> None:
    # Genuinely 2-D bimodal data → the 2-node switch beats the 2-D no-switch null.
    data = sample_lna_mixture(
        ras_switch_2node(), 3000, jax.random.PRNGKey(0), scale=SCALE, obs_sd=OBS_SD
    )
    res = select_topology(data, [_cand2()], steps=150)
    assert res.selected == "2-node" and res.is_switch is True


def test_dimension_mismatch_raises() -> None:
    # A 2-node (2-species) candidate cannot score 1-D data — the honest guard.
    data = np.zeros((10, 1))
    with pytest.raises(ValueError, match="species"):
        select_topology(data, [_cand2()], steps=1, include_no_switch=False)
