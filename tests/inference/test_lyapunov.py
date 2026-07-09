"""The Lyapunov (linear-noise Gaussian-mixture) attribution path — M1.

Validates the covariance-structured fit end to end on **inverse-crime** data (cells
sampled from the same LNA mixture): the differentiable forward model — IFT-diff mode
means + the Lyapunov covariance — plus the optax NLL fit recover the perturbed
kinetic parameter. ``scale`` / ``obs_sd`` are fixed at their known values here (a free
global scale is degenerate with vmax — for real data the scale must be calibrated).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef
from nudge.inference.lyapunov import (
    OperatingPoint,
    attribute_lyapunov_multi,
    attribute_lyapunov_single,
    calibrate_from_wt,
    fit_lyapunov_parameters,
    sample_lna_mixture,
)

SCALE, OBS_SD = 20.0, 0.5  # Gaussian-friendly (higher-count) regime


def _toggle() -> Circuit:
    return Circuit(
        [SpeciesDef("A", basal=0.05, decay=1.0), SpeciesDef("B", basal=0.05, decay=1)],
        [
            EdgeDef(1, 0, "hill_repression", K=1.0, n=4.0, vmax=2.0),
            EdgeDef(0, 1, "hill_repression", K=1.0, n=4.0, vmax=2.0),
        ],
    )


def test_sample_lna_mixture_structure() -> None:
    # Two well-separated lobes near the toggle's stable fixed points.
    data = sample_lna_mixture(_toggle(), 800, jax.random.PRNGKey(0), scale=SCALE)
    assert data.shape == (800, 2)
    # each cell is closer to one lobe than the other → clear bimodality in A - B
    diff = data[:, 0] - data[:, 1]
    assert (diff > 5).any() and (diff < -5).any()


def test_fit_requires_bistable() -> None:
    # A monostable switch has one stable mode → the k_modes=2 fit must refuse cleanly.
    mono = Circuit(
        [SpeciesDef("SW", basal=0.05, decay=1.0)],
        [EdgeDef(0, 0, "hill_activation", K=1.0, n=1.2, vmax=2.0)],
    )
    data = np.zeros((10, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="stable modes"):
        fit_lyapunov_parameters(data, mono, [("edge", 0, "n")], k_modes=2, steps=1)


@pytest.mark.parametrize(
    ("free", "true_val", "tol"),
    [
        (("edge", 0, "n"), 2.8, 0.15),  # gain (Hill n × 0.7)
        (("edge", 0, "vmax"), 1.2, 0.15),  # ceiling (vmax × 0.6)
    ],
)
@pytest.mark.slow
def test_inverse_crime_recovers_kinetic(free, true_val, tol) -> None:
    wt = _toggle()
    data = sample_lna_mixture(
        wt, 3000, jax.random.PRNGKey(0),
        free=[free], vals=np.array([true_val]), scale=SCALE, obs_sd=OBS_SD,
    )
    rec, _aux, _hist = fit_lyapunov_parameters(
        data, wt, [free], k_modes=2, steps=200, seed=0,
        scale_init=SCALE, obs_sd_init=OBS_SD, fit_scale=False, fit_obs=False,
    )
    got = list(rec.values())[0]
    assert abs(got - true_val) / true_val < tol


@pytest.mark.parametrize(
    ("mech", "param", "val", "expected"),
    [
        ("ceiling", "vmax", 1.5, "ceiling"),          # identifiable
        ("gain", "n", 3.0, "gain_or_threshold"),      # confounded → abstain
        ("threshold", "K", 1.3, "gain_or_threshold"), # confounded → abstain
    ],
)
@pytest.mark.slow
def test_single_condition_correct_or_abstains(mech, param, val, expected) -> None:
    # The honest single-snapshot call: identify ceiling; abstain between gain and
    # threshold (the measured confound). Crucially it is NEVER confidently wrong — it
    # never returns the *other* specific mechanism.
    wt = _toggle()
    wt_data = sample_lna_mixture(
        wt, 3000, jax.random.PRNGKey(100), scale=SCALE, obs_sd=OBS_SD
    )
    cond = sample_lna_mixture(
        wt, 3000, jax.random.PRNGKey(0),
        free=[("edge", 0, param)], vals=np.array([val]), scale=SCALE, obs_sd=OBS_SD,
    )
    label, _nlls = attribute_lyapunov_single(
        cond, wt, wt_data=wt_data, target_edge=0, steps=200, seed=0
    )
    assert label == expected
    assert label != "gain"  # a single snapshot must never claim a bare gain/threshold
    assert label != "threshold"


def _toggle_bB(bB: float) -> Circuit:
    return Circuit(
        [SpeciesDef("A", basal=0.05, decay=1.0), SpeciesDef("B", basal=bB, decay=1)],
        [
            EdgeDef(1, 0, "hill_repression", K=1.0, n=4.0, vmax=2.0),
            EdgeDef(0, 1, "hill_repression", K=1.0, n=4.0, vmax=2.0),
        ],
    )


def _operating_point(bB: float, param: str, val: float) -> OperatingPoint:
    wt = _toggle_bB(bB)
    wt_data = sample_lna_mixture(
        wt, 3000, jax.random.PRNGKey(500), scale=SCALE, obs_sd=OBS_SD
    )
    scale, obs = calibrate_from_wt(wt_data, wt)
    cond = sample_lna_mixture(
        wt, 3000, jax.random.PRNGKey(0),
        free=[("edge", 0, param)], vals=np.array([val]), scale=SCALE, obs_sd=OBS_SD,
    )
    return OperatingPoint(data=cond, circuit=wt, scale=scale, obs_sd=obs)


@pytest.mark.parametrize(
    ("mech", "param", "val"),
    [("gain", "n", 2.4), ("threshold", "K", 2.0)],
)
@pytest.mark.slow
def test_second_operating_point_breaks_confound(mech, param, val) -> None:
    # A single operating point abstains between gain and threshold (test above); a
    # SECOND operating point (a basal-B shift) lets the joint fit RESOLVE the true one.
    op1 = _operating_point(0.05, param, val)
    op2 = _operating_point(0.30, param, val)
    label, nlls = attribute_lyapunov_multi([op1, op2], target_edge=0, steps=200, seed=0)
    assert label == mech  # the confound is broken → the true mechanism is named
    # the true mechanism's joint NLL is the clear minimum
    assert nlls[mech] == min(nlls.values())
