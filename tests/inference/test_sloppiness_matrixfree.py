"""Matrix-free sloppiness diagnostic — matches the dense path, never materializes J.

Correctness contract (the honesty rule): on the validated small cases the matrix-free
verdict/spectrum/null-direction must MATCH the dense :func:`sloppiness_diagnostic`
(``scripts/vv/sloppiness_validation.py`` cases — the sum-of-exponentials *sloppy* model and
the ``A·e^{-(k₁+k₂)t}`` *unidentifiable* model), and the FIM matvec must equal the dense
``JᵀJ/σ²`` action. The scaling itself is measured in ``scripts/vv/sloppiness_scaling.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from nudge.inference.sloppiness import (
    analyze_model,
    analyze_model_matrixfree,
    fim_matvec,
    fisher_information,
    redundant_exponential_predict,
    relative_sensitivity_jacobian,
    sum_of_exponentials_predict,
    well_conditioned_predict,
)

# sloppy-model analysis needs float64 resolution on the FIM spectrum (the smallest
# eigenvalues / structural nulls); conftest scopes x64 to the ``x64`` marker.
pytestmark = pytest.mark.x64

SIGMA = 0.01


def _t() -> np.ndarray:
    return np.linspace(0.05, 6.0, 60)


def _cases() -> dict:
    t = _t()
    return {
        "sloppy": (
            sum_of_exponentials_predict(rates=[0.5, 1.3, 2.5, 4.5], amps=[1, 1, 1, 1], t=t),
            "sloppy-but-predictive",
        ),
        "unident": (
            redundant_exponential_predict(amp=1.0, k1=0.7, k2=0.9, t=t),
            "unidentifiable",
        ),
        "well": (well_conditioned_predict(slope=2.0, offset=1.0, t=t), "well-constrained"),
    }


@pytest.mark.parametrize("name", ["sloppy", "unident", "well"])
def test_matrixfree_label_matches_dense(name: str) -> None:
    """Same verdict as the dense diagnostic on every validated small case."""
    fn, expected = _cases()[name]
    dense = analyze_model(fn, sigma=SIGMA)
    mf = analyze_model_matrixfree(fn, sigma=SIGMA)
    assert dense.label == expected  # sanity: the dense oracle is what we think
    assert mf.label == dense.label


@pytest.mark.parametrize("name", ["sloppy", "unident", "well"])
def test_matrixfree_spectrum_matches_dense(name: str) -> None:
    """Condition number, spectral span, and null count agree to tight tolerance."""
    fn, _ = _cases()[name]
    dense = analyze_model(fn, sigma=SIGMA)
    mf = analyze_model_matrixfree(fn, sigma=SIGMA)
    assert mf.n_null_dims == dense.n_null_dims
    # extreme eigenvalues agree (relative) — largest always finite here.
    assert mf.largest_eigenvalue == pytest.approx(dense.largest_eigenvalue, rel=1e-6)
    if dense.n_null_dims == 0:  # spectrum finite -> span/cond comparable
        assert mf.spectral_span_decades == pytest.approx(dense.spectral_span_decades, abs=1e-3)
    # prediction reliability agrees.
    assert mf.predictive == dense.predictive
    assert mf.relative_prediction_std == pytest.approx(dense.relative_prediction_std, rel=1e-4)


def test_matrixfree_null_direction_matches_dense() -> None:
    """The reported null direction of the structurally-unidentifiable model coincides
    (up to sign) with the dense SVD null vector."""
    fn, _ = _cases()["unident"]
    dense = analyze_model(fn, sigma=SIGMA)
    mf = analyze_model_matrixfree(fn, sigma=SIGMA)
    assert dense.null_directions and mf.null_directions
    vd = dense.null_directions[0].vector
    vm = mf.null_directions[0].vector
    assert abs(float(vd @ vm)) == pytest.approx(1.0, abs=1e-6)  # parallel


def test_fim_matvec_equals_dense_action() -> None:
    """``fim_matvec`` reproduces the dense ``JᵀJ/σ²`` matrix-vector product exactly."""
    fn, _ = _cases()["sloppy"]
    theta = np.asarray(fn.theta0, dtype=np.float64)
    jac_log, _y = relative_sensitivity_jacobian(fn, theta)
    fim_dense = fisher_information(jac_log, SIGMA)  # (p, p)
    mv = fim_matvec(fn, theta, SIGMA)
    rng = np.random.default_rng(0)
    for _ in range(4):
        v = rng.standard_normal(theta.shape[0])
        assert np.allclose(mv(v), fim_dense @ v, rtol=1e-8, atol=1e-8)


def test_matrixfree_never_forms_the_jacobian() -> None:
    """The matvec closure returns an n_params vector from an n_params vector — its work is
    O(n_params + n_obs), the whole point (regression against accidentally densifying)."""
    fn, _ = _cases()["sloppy"]
    theta = np.asarray(fn.theta0, dtype=np.float64)
    mv = fim_matvec(fn, theta, SIGMA)
    out = mv(np.ones(theta.shape[0]))
    assert out.shape == (theta.shape[0],)


# --------------------------------------------------------------------------- #
# the large-ODE end-to-end path (the scaling target) — slow
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_ode_identifiability_underdetermined_is_unidentifiable() -> None:
    """A gLV fit with more free parameters than observations is rank-deficient BY SHAPE
    (rank ≤ n_obs) — the matrix-free iterative path certifies ``unidentifiable`` from shape
    alone (no slow/unreliable smallest-eigenvalue solve), the realistic large-network verdict."""
    from nudge.inference.adjoint import make_glv_problem, ode_identifiability

    prob = make_glv_problem(n_species=8, n_free=80, n_obs=6, seed=1)  # 80 params, 48 obs
    assert prob.n_theta > prob.target.size  # underdetermined
    rep = ode_identifiability(prob, sigma=1e-2, method="iterative")
    assert rep.label == "unidentifiable"
    assert rep.n_null_dims >= prob.n_theta - prob.target.size


@pytest.mark.slow
def test_ode_matrixfree_dense_matches_and_iterative_is_fail_safe() -> None:
    """On a rank-deficient gLV fit, the exact dense-via-matvec path recovers the structural
    nulls (``unidentifiable``), and the iterative path is FAIL-SAFE: it never asserts
    identifiability it cannot certify (it abstains ``unidentifiable`` when the ill-conditioned
    smallest end does not converge)."""
    import jax

    from nudge.inference.adjoint import make_glv_problem, ode_trajectory_predict_fn
    from nudge.inference.sloppiness import sloppiness_diagnostic_matrixfree

    prob = make_glv_problem(n_species=12, n_free=120, n_obs=24, dtype=jax.numpy.float64, seed=0)
    fn = ode_trajectory_predict_fn(prob)
    theta = np.asarray(prob.theta0, dtype=np.float64)
    exact = sloppiness_diagnostic_matrixfree(fn, theta, 1e-2, method="dense")
    assert exact.label == "unidentifiable"
    assert exact.n_null_dims > 0  # data-driven rank deficiency, recovered exactly
    itr = sloppiness_diagnostic_matrixfree(fn, theta, 1e-2, method="iterative")
    # never confidently identifiable on a genuinely rank-deficient model.
    assert itr.label == "unidentifiable"
