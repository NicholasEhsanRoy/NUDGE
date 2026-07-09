"""Laplace posterior uncertainty layer — curvature CIs + the degeneracy guard.

Two tiers. **Fast (CI lane):** pure-math checks of ``laplace_posterior`` on
hand-built losses — a well-conditioned quadratic (the CI matches the analytic
Hessian⁻¹), a rank-deficient loss (the guarded inverse stays finite and the flat
direction is flagged *unidentifiable / CI unbounded*, never false-precise), a
non-positive-definite loss (abstain), and ``mechanism_confidence``'s abstention.
**Slow (full lane):** the load-bearing validation on the Lyapunov Gaussian-mixture
NLL — the K⇄v_max / gain⇄threshold degeneracy reproduces as a high condition number
and strong negative correlation, drops with a second operating point (the measured
×16 Fisher result), and the marginal CIs cover the truth at ~nominal rate.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef
from nudge.inference.lyapunov import (
    _stable_roots,
    fit_lyapunov_parameters,
    sample_lna_mixture,
)
from nudge.inference.uncertainty import (
    laplace_posterior,
    lyapunov_nll_loss,
    mechanism_confidence,
)

SCALE, OBS_SD = 20.0, 0.5  # Gaussian-friendly (higher-count) regime


# ---------------------------------------------------------------------------
# Fast, pure-math checks of the curvature → posterior machinery (no circuit).
# ---------------------------------------------------------------------------


def test_wellconditioned_quadratic_matches_analytic_hessian() -> None:
    # loss = ½ (θ−θ*)ᵀ A (θ−θ*): Hessian = A, cov = A⁻¹, so the log-space marginal sd
    # is √diag(A⁻¹) and the natural-unit CI is exp(θ* ± z·sd). A is SPD + well-mixed.
    a_mat = jnp.array([[4.0, 1.0], [1.0, 9.0]])
    theta_star = jnp.array([np.log(2.0), np.log(5.0)])

    def loss(theta: Array) -> Array:
        d = theta - theta_star
        return 0.5 * d @ a_mat @ d

    post = laplace_posterior(loss, theta_star, names=["K", "vmax"], n_data=1)
    assert not post.degenerate
    cov_true = np.linalg.inv(np.asarray(a_mat))
    np.testing.assert_allclose(post.cov, cov_true, atol=1e-6)
    sd = np.sqrt(np.diag(cov_true))
    for i, ci in enumerate(post.marginal_ci):
        assert ci.identifiable
        assert ci.lo < ci.point < ci.hi
        np.testing.assert_allclose(ci.lo, float(np.exp(theta_star[i] - 1.96 * sd[i])),
                                   rtol=1e-4)
        np.testing.assert_allclose(ci.hi, float(np.exp(theta_star[i] + 1.96 * sd[i])),
                                   rtol=1e-4)


def test_ndata_scales_covariance() -> None:
    # A mean NLL over N cells has MLE covariance H⁻¹/N: the CI must shrink like 1/√N.
    def loss(theta: Array) -> Array:
        return 0.5 * jnp.sum(theta**2)

    p1 = laplace_posterior(loss, jnp.array([0.0]), n_data=1)
    p100 = laplace_posterior(loss, jnp.array([0.0]), n_data=100)
    np.testing.assert_allclose(p100.cov, p1.cov / 100.0, atol=1e-9)
    np.testing.assert_allclose(p100.marginal_ci[0].log_sd,
                               p1.marginal_ci[0].log_sd / 10.0, rtol=1e-5)


def test_guarded_inverse_on_singular_hessian_widens_not_nans() -> None:
    # loss depends only on (a − b): the (1,1) direction is perfectly flat (a rank-1
    # Hessian). A plain pseudo-inverse would ZERO that direction's variance (false
    # precision); the guarded ridge inverse must instead keep it finite + PSD, flag the
    # posterior degenerate, and report both knobs unidentifiable with an unbounded CI.
    def loss(theta: Array) -> Array:
        return (theta[0] - theta[1]) ** 2

    post = laplace_posterior(loss, jnp.array([0.0, 0.0]), names=["n", "K"])
    assert post.degenerate
    assert np.isfinite(post.cov).all()  # no NaN / inf: the guard held
    assert post.cond_number > 1e3
    for ci in post.marginal_ci:
        assert not ci.identifiable
        assert ci.hi == float("inf") and ci.log_sd == float("inf")
    conf = mechanism_confidence(post, [("edge", 0, "n"), ("edge", 0, "K")])
    assert conf["confidence"] == 0.0 and conf["unidentifiable"]
    assert set(conf["unidentifiable_knobs"]) == {"n", "K"}


def test_partial_degeneracy_flags_only_the_flat_knob() -> None:
    # One stiff direction (θ0) + one flat (θ1): θ0 stays identifiable, θ1 abstains.
    def loss(theta: Array) -> Array:
        return 5.0 * theta[0] ** 2 + 1e-9 * theta[1] ** 2

    post = laplace_posterior(loss, jnp.array([0.0, 0.0]), names=["vmax", "K"])
    assert post.degenerate
    assert post.marginal_ci[0].identifiable
    assert not post.marginal_ci[1].identifiable


def test_non_positive_definite_hessian_abstains() -> None:
    # A saddle (a² − b²) is not a minimum → curvature can't be a posterior precision.
    def loss(theta: Array) -> Array:
        return theta[0] ** 2 - theta[1] ** 2

    post = laplace_posterior(loss, jnp.array([0.0, 0.0]))
    assert post.degenerate and not np.isfinite(post.cond_number)
    assert "positive definite" in post.reason


def test_mechanism_confidence_clean_case_is_bounded() -> None:
    def loss(theta: Array) -> Array:
        return 0.5 * jnp.sum(theta**2)  # identity Hessian → sd = 1 per knob

    post = laplace_posterior(loss, jnp.zeros(3), names=["K", "n", "vmax"])
    conf = mechanism_confidence(
        post, [("edge", 0, "K"), ("edge", 0, "n"), ("edge", 0, "vmax")]
    )
    assert not conf["unidentifiable"]
    assert 0.0 < conf["confidence"] <= 1.0
    np.testing.assert_allclose(conf["confidence"], np.exp(-1.0), rtol=1e-4)


# ---------------------------------------------------------------------------
# Slow: the load-bearing validation on the real Lyapunov NLL (inverse-crime data).
# ---------------------------------------------------------------------------


def _toggle(bB: float = 0.05) -> Circuit:
    return Circuit(
        [SpeciesDef("A", basal=0.05, decay=1.0), SpeciesDef("B", basal=bB, decay=1.0)],
        [
            EdgeDef(1, 0, "hill_repression", K=1.0, n=4.0, vmax=2.0),
            EdgeDef(0, 1, "hill_repression", K=1.0, n=4.0, vmax=2.0),
        ],
    )


def _sorted_roots(circuit: Circuit) -> np.ndarray:
    roots = sorted(
        _stable_roots(circuit, [], np.asarray([], dtype=float)),
        key=lambda r: tuple(float(x) for x in r),
    )
    return np.stack(roots)


_FREE3 = [("edge", 0, "n"), ("edge", 0, "vmax"), ("edge", 0, "K")]
_NAMES3 = ["n", "vmax", "K"]


@pytest.mark.slow
def test_gain_threshold_degeneracy_reproduced_and_broken() -> None:
    # The load-bearing reproduction of FINDINGS §2 through the Laplace covariance.
    # SINGLE operating point: the toggle's gain(n)⇄threshold(K) confound must show up as
    # a near-singular Hessian — a high condition number (measured ≈ 211, matching the
    # FIM's ≈ 210) and a strong correlation (≈ 0.99). NOTE the SIGN: the covariance
    # correlation is +0.99 while the *Fisher* correlation is -0.99 (inverting a 2x2
    # with a negative off-diagonal flips the sign) -- the same degeneracy, seen through
    # H^-1 rather than H. So gain + threshold are flagged unidentifiable, ceiling not.
    c1 = _toggle(0.05)
    data1 = sample_lna_mixture(
        c1, 2000, jax.random.PRNGKey(0), scale=SCALE, obs_sd=OBS_SD
    )
    theta_true = np.log(np.array([4.0, 2.0, 1.0]))  # (n, vmax, K)
    loss1 = lyapunov_nll_loss(
        data1, c1, _FREE3, roots=_sorted_roots(c1), scale=SCALE, obs_sd=OBS_SD
    )
    post1 = laplace_posterior(loss1, theta_true, names=_NAMES3, n_data=data1.shape[0])
    assert post1.degenerate
    # Order-10² near-singularity, reproducing the FIM's ≈ 210. The exact value is a
    # finite-sample quantity (the flat direction is *barely* curved, so its empirical
    # curvature is noisy: ~150-250+ at N~2000, ~211 at N~4000) -- the load-bearing
    # claim is that it far exceeds the guard, which it robustly does.
    assert post1.cond_number > 120.0
    assert abs(post1.correlation[0, 2]) > 0.95  # |corr(n, K)| — the sloppy pair
    conf1 = mechanism_confidence(post1, _FREE3)
    assert conf1["confidence"] == 0.0
    # gain (n) + threshold (K) abstain; ceiling (vmax) stays identifiable
    assert set(conf1["unidentifiable_knobs"]) == {"n", "K"}

    # SECOND operating point (a basal-B shift) breaks it — mirroring the measured ×16
    # Fisher improvement: the condition number collapses (≈ 211 → ≈ 27) below the guard,
    # the posterior resolves, and every knob becomes identifiable.
    c2 = _toggle(0.30)
    data2 = sample_lna_mixture(
        c2, 2000, jax.random.PRNGKey(1), scale=SCALE, obs_sd=OBS_SD
    )
    loss2 = lyapunov_nll_loss(
        data2, c2, _FREE3, roots=_sorted_roots(c2), scale=SCALE, obs_sd=OBS_SD
    )

    def combined(log_theta: Array) -> Array:
        return loss1(log_theta) + loss2(log_theta)

    n1 = data1.shape[0]
    post2 = laplace_posterior(combined, theta_true, names=_NAMES3, n_data=n1)
    assert not post2.degenerate
    assert post2.cond_number < 50.0
    assert post2.cond_number < post1.cond_number / 4.0  # the degeneracy-break
    conf2 = mechanism_confidence(post2, _FREE3)
    assert conf2["unidentifiable_knobs"] == []
    assert conf2["confidence"] > 0.0


@pytest.mark.slow
def test_marginal_ci_covers_truth() -> None:
    # Calibration: fit the identifiable knob (ceiling / vmax) on fresh inverse-crime
    # samples, build the Laplace CI at each θ*, and check it covers the true value at
    # ~nominal rate across seeds. Coverage ≥ nominal is the fail-safe direction (a wider
    # honest interval is fine; a too-narrow one is not).
    c = _toggle(0.05)
    v_true = 2.0
    free = [("edge", 0, "vmax")]
    n_seeds, n_cells = 16, 1200
    hits = 0
    for s in range(n_seeds):
        data = sample_lna_mixture(
            c, n_cells, jax.random.PRNGKey(2000 + s), scale=SCALE, obs_sd=OBS_SD
        )
        rec, aux, _ = fit_lyapunov_parameters(
            data, c, free, k_modes=2, steps=70, seed=s,
            scale_init=SCALE, obs_sd_init=OBS_SD, fit_scale=False, fit_obs=False,
        )
        v_hat = rec[free[0]]
        roots = np.stack(
            sorted(_stable_roots(c, free, np.array([v_hat])),
                   key=lambda r: tuple(float(x) for x in r))
        )
        loss = lyapunov_nll_loss(
            data, c, free, roots=roots, scale=SCALE, obs_sd=OBS_SD,
            log_weights=aux["weights"],
        )
        post = laplace_posterior(
            loss, np.log([v_hat]), names=["vmax"], n_data=n_cells
        )
        ci = post.marginal_ci[0]
        assert ci.identifiable and np.isfinite(ci.hi)
        assert ci.hi / ci.lo < 1.5  # an informative (not vacuous) interval
        hits += int(ci.lo <= v_true <= ci.hi)
    assert hits / n_seeds >= 0.85  # covers the truth at ~nominal (95%) rate or better
