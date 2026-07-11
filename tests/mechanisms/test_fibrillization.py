"""Tests for the protein aggregation / fibrillization mechanism (``NUDGE-METHOD-013``).

Fast lane: the differentiable moment field + RK4 integrator + composites helper + the
**exact gauge symmetry** (a numerical check that a 100× ``k_+`` rescale, compensated on
``k_n`` / ``k_2``, leaves the mass-fraction curve identical) + the inhibitor classifier's
result-level plumbing.

Slow lane (``verification`` / ``decoy``): the synthetic round-trips — a single curve
**recovers the composites κ, λ and ABSTAINS on the three individual constants** (the
measured gauge null); a concentration series **with** a seeded anchor **resolves** all
three; a concentration series **without** the anchor **stays degenerate**; and the
inhibitor battery is **0 confident-wrong** (recover the microscopic target or abstain,
never a wrong step). ``NUDGE-LIM-021``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nudge.mechanisms.fibrillization import (
    _BALANCED_TRUTH,
    _K_NAMES,
    AggregationParams,
    attribute_aggregation,
    attribute_inhibitor,
    composite_lambda_kappa,
    fit_composites,
    individual_k_identifiability,
    moment_vector_field,
    resolve_series,
    simulate_aggregation,
    simulate_aggregation_curve,
    simulate_concentration_series,
    simulate_inhibitor_pair,
)

_TARGETS = {"primary_nucleation", "elongation", "secondary_nucleation"}


# --------------------------------------------------------------------------- #
# fast lane — field, integrator, composites, gauge symmetry
# --------------------------------------------------------------------------- #
def test_moment_field_matches_the_master_equation_moments() -> None:
    # P=1, M=0.4, m = m_tot - M = 0.6; k_n=2, k_+=3, k_2=5, n_c=2, n_2=2, m_tot=1.
    state = jnp.array([1.0, 0.4])
    d = moment_vector_field(
        state, jnp.asarray(2.0), jnp.asarray(3.0), jnp.asarray(5.0),
        n_c=2.0, n_2=2.0, m_tot=1.0,
    )
    m = 0.6
    expected_dp = 2.0 * m**2 + 5.0 * m**2 * 0.4  # k_n m^nc + k_2 m^n2 M
    expected_dm = 2.0 * 3.0 * m * 1.0            # 2 k_+ m P
    assert float(d[0]) == pytest.approx(expected_dp, rel=1e-6)
    assert float(d[1]) == pytest.approx(expected_dm, rel=1e-6)


def test_curve_is_sigmoidal_and_saturates() -> None:
    c = simulate_aggregation_curve(t_max=40.0, n_obs=40, dt=0.02, n_replicates=4, seed=0)
    y = c.mean
    assert y[0] < 0.1                      # starts near zero (lag)
    assert y[-1] > 0.9                     # saturates near full mass
    assert np.all(np.diff(y) > -0.05)      # monotone up to noise


def test_composites_helper_arithmetic() -> None:
    p = AggregationParams(k_n=5e-4, k_plus=0.1, k_2=5.0, n_c=2.0, n_2=2.0)
    lam, kappa = composite_lambda_kappa(p, 1.0)
    assert lam == pytest.approx(0.01, rel=1e-9)   # √(2·0.1·5e-4)
    assert kappa == pytest.approx(1.0, rel=1e-9)  # √(2·0.1·5)


def test_gauge_symmetry_is_exact() -> None:
    """(k_n, k_+, k_2) → (k_n/α, α k_+, k_2/α) must leave the mass-fraction curve
    identical — the exact continuous degeneracy that makes the three individual constants
    non-identifiable from one curve."""
    t_obs = np.linspace(0.0, 40.0, 40)
    n_steps = 2000
    obs_idx = jnp.asarray(np.clip(np.round(t_obs / 0.02).astype(int), 0, n_steps))
    kw = dict(m_tot=1.0, dt=0.02, n_steps=n_steps, obs_idx=obs_idx, n_c=2.0, n_2=2.0)
    base = np.asarray(simulate_aggregation(
        (jnp.asarray(5e-4), jnp.asarray(0.1), jnp.asarray(5.0)), **kw))
    alpha = 100.0
    scaled = np.asarray(simulate_aggregation(
        (jnp.asarray(5e-4 / alpha), jnp.asarray(0.1 * alpha), jnp.asarray(5.0 / alpha)),
        **kw))
    assert np.max(np.abs(base - scaled)) < 1e-4  # float32 grid; still machine-tiny


# --------------------------------------------------------------------------- #
# slow lane — the single-curve headline: composites identified, individuals abstained
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.verification
def test_single_curve_identifies_composites_and_abstains_on_individuals() -> None:
    curve = simulate_aggregation_curve(seed=0)  # κ=1, λ=0.01
    res = attribute_aggregation(curve, steps=600)

    # the two composites ARE recovered (the identifiable dof).
    assert res.kappa == pytest.approx(1.0, rel=0.1)
    assert res.lam == pytest.approx(0.01, rel=0.5)  # λ weakly constrained but recovered

    # the three individual constants are NOT — a MEASURED degeneracy, not asserted.
    ident = res.identifiability
    assert res.call == "composites-identified"
    assert res.individual_k_identifiable is False
    assert ident.degenerate is True
    assert ident.cond_number > 1e3                  # near-singular (→ ∞ at the exact gauge)
    assert set(ident.unidentifiable) == set(_K_NAMES)
    assert ident.gauge_check < 1e-6                 # the exact symmetry, confirmed

    # the null direction is the k_+ ⇄ (k_n, k_2) gauge: (+k_n, −k_+, +k_2) up to sign.
    v = ident.null_direction
    assert np.sign(v[0]) == np.sign(v[2])           # k_n and k_2 move together
    assert np.sign(v[0]) != np.sign(v[1])           # against k_+
    assert np.allclose(np.abs(v), 1 / np.sqrt(3), atol=0.08)


# --------------------------------------------------------------------------- #
# slow lane — the concentration series resolves the three (with a seeded anchor)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.verification
def test_concentration_series_with_anchor_resolves_three_constants() -> None:
    series = simulate_concentration_series(with_anchor=True, seed=0)  # balanced regime
    sr = resolve_series(series, use_anchor=True, steps=1500)
    assert sr.identifiable is True
    # 0 confident-wrong: all three recovered close to truth.
    assert sr.k_n == pytest.approx(_BALANCED_TRUTH.k_n, rel=0.25)
    assert sr.k_plus == pytest.approx(_BALANCED_TRUTH.k_plus, rel=0.25)
    assert sr.k_2 == pytest.approx(_BALANCED_TRUTH.k_2, rel=0.25)


@pytest.mark.slow
@pytest.mark.decoy
def test_concentration_series_without_anchor_stays_degenerate() -> None:
    """The honesty half: a mass-fraction concentration series ALONE cannot separate the
    individuals — the gauge is concentration-independent, so NUDGE must NOT claim a
    resolution (``NUDGE-LIM-021``)."""
    series = simulate_concentration_series(with_anchor=False, seed=0)
    sr = resolve_series(series, use_anchor=False, steps=1200)
    assert sr.identifiable is False
    assert not np.isfinite(sr.cond_number) or sr.cond_number > 1e6


# --------------------------------------------------------------------------- #
# slow lane — inhibitor attribution: 0 confident-wrong across the battery
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.verification
@pytest.mark.parametrize(
    "target", ["secondary_nucleation", "elongation", "primary_nucleation", "none"]
)
def test_inhibitor_battery_recover_or_abstain_never_wrong(target: str) -> None:
    ctrl, inhibited, gt = simulate_inhibitor_pair(target=target, factor=0.25, seed=3)
    res = attribute_inhibitor(ctrl, inhibited, steps=500)
    if target == "none":
        assert res.call in ("no-effect", "unresolved")   # never a positive on a null
        assert res.is_reliable is False
    else:
        # recover the true microscopic target, or abstain — NEVER a different target.
        assert res.call in (target, "unresolved")
        wrong = _TARGETS - {target}
        assert res.call not in wrong


@pytest.mark.slow
@pytest.mark.verification
def test_inhibitor_battery_zero_confident_wrong() -> None:
    """The headline fail-safe: across every microscopic target, NUDGE never names the
    WRONG step (it recovers the true one or abstains)."""
    confident_wrong = 0
    for target in ("secondary_nucleation", "elongation", "primary_nucleation"):
        ctrl, inhibited, _ = simulate_inhibitor_pair(target=target, factor=0.25, seed=7)
        res = attribute_inhibitor(ctrl, inhibited, steps=500)
        if res.call in _TARGETS and res.call != target:
            confident_wrong += 1
    assert confident_wrong == 0


# --------------------------------------------------------------------------- #
# fast lane — composite fit is well-posed even when individuals are not
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_composite_fit_recovers_kappa_lambda_point_estimate() -> None:
    curve = simulate_aggregation_curve(seed=1)
    fit = fit_composites(curve, steps=600)
    assert fit.kappa == pytest.approx(1.0, rel=0.1)
    assert fit.lam == pytest.approx(0.01, rel=0.6)
    # and the measured individual-k degeneracy is reproducible from the fit.
    ident = individual_k_identifiability(curve, fit)
    assert ident.degenerate is True
