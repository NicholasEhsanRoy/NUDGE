"""P6 regression (``NUDGE-LIM-023``): the matrix-free iterative path must NEVER label a
structurally-UNIDENTIFIABLE model ``well-constrained``.

The hole (red-team round 6, ``scripts/redteam/sloppiness_matrixfree_iterative_mislabel.py``):
on a well-conditioned linear map with ONE exact duplicated parameter column — parameters
``p0`` and ``p{last}`` enter only through their sum, a provable Fisher null — the iterative /
``method="auto"`` (``n_params > dense_below``) path returned ``well-constrained``
(``n_null=0``, "every parameter individually identifiable"), because ``eigsh(which='SA')``
misses the isolated zero and the Rayleigh check only verifies eigenpair-ness, not
smallest-ness.

The fix (measured, FINDINGS §P6): ``auto`` defers to the EXACT dense-via-matvec reconstruction
up to ``dense_below=2048`` (recovers the null exactly), and where that is not affordable the
smallest end is probed by INVERSE ITERATION (shift-invert via CG) which reliably CATCHES the
isolated null ``eigsh('SA')`` misses; if no null is found the path ABSTAINS
(``unidentifiable``) — it never asserts ``well-constrained`` / ``sloppy-but-predictive`` it
cannot certify. This test locks: (a) the null is caught across the reported seeds/sizes via
the DEFAULT ``method="auto"``; (b) genuine positive controls still resolve (no
over-abstention); (c) the residual huge-regime over-abstention is strict-xfail decoyed.
"""

from __future__ import annotations

import numpy as np
import pytest

from nudge.inference.sloppiness import (
    analyze_model,
    analyze_model_matrixfree,
    sloppiness_diagnostic_matrixfree,
    sum_of_exponentials_predict,
    well_conditioned_predict,
)

pytestmark = pytest.mark.x64  # smallest-eig / null resolution needs float64

SIGMA = 0.01


def _linear_model(mat: np.ndarray):
    """Wrap a numpy design matrix ``M`` as a differentiable ``predict(theta) = M·theta`` with
    the ``.theta0`` / ``.names`` attributes the analyzers read."""
    import jax.numpy as jnp

    mat_j = jnp.asarray(mat)
    n_params = mat.shape[1]

    def predict(theta):
        return mat_j @ theta

    predict.theta0 = np.ones(n_params, dtype=np.float64)  # type: ignore[attr-defined]
    predict.names = tuple(f"p{i}" for i in range(n_params))  # type: ignore[attr-defined]
    return predict, predict.theta0  # type: ignore[attr-defined]


def _structural_null_model(n_params: int, n_obs: int, seed: int):
    """A well-conditioned linear map with the LAST column set equal to the FIRST — an EXACT
    structural null (the ``(1,0,…,0,-1)`` Fisher-zero direction). Truth: ``unidentifiable``."""
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((n_obs, n_params))
    mat = mat / np.linalg.norm(mat, axis=0, keepdims=True)  # tight non-null spectrum
    mat[:, n_params - 1] = mat[:, 0]  # exact duplicate column
    return _linear_model(mat)


def _well_conditioned_fullrank(n_params: int, n_obs: int, seed: int):
    """Same construction WITHOUT the duplicate column — genuinely well-constrained, no null."""
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((n_obs, n_params))
    mat = mat / np.linalg.norm(mat, axis=0, keepdims=True)
    return _linear_model(mat)


# --------------------------------------------------------------------------- #
# (a) THE HOLE IS CLOSED — structural null never labelled well-constrained
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_p6_auto_default_structural_null_is_unidentifiable(seed: int) -> None:
    """CASE 1: n_params=300 > dense_below ⇒ the DEFAULT ``method="auto"`` path. Must return
    ``unidentifiable`` (matching the exact dense oracle), NEVER ``well-constrained`` /
    ``sloppy-but-predictive``."""
    fn, _theta = _structural_null_model(300, 400, seed)
    rep = analyze_model_matrixfree(fn, sigma=SIGMA)  # default method="auto"
    assert rep.label == "unidentifiable"
    assert rep.label != "well-constrained"
    assert rep.n_null_dims >= 1  # the null is caught (probe) — not n_null=0
    # the exact dense-via-matvec oracle agrees on the identical model
    dense = sloppiness_diagnostic_matrixfree(fn, fn.theta0, SIGMA, method="dense")
    assert dense.label == "unidentifiable" and dense.n_null_dims >= 1


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_p6_explicit_iterative_structural_null_is_unidentifiable(seed: int) -> None:
    """CASE 2: explicit ``method="iterative"`` at n_params=40 (forces the iterative path). The
    inverse-iteration null probe must catch the isolated null the shipped ``eigsh('SA')``
    missed — ``unidentifiable``, matching the ``analyze_model`` jacfwd-SVD oracle."""
    fn, theta = _structural_null_model(40, 100, seed)
    itr = sloppiness_diagnostic_matrixfree(fn, theta, SIGMA, method="iterative")
    oracle = analyze_model(fn, sigma=SIGMA)
    assert oracle.label == "unidentifiable"  # sanity: the reference truth
    assert itr.label == "unidentifiable"
    assert itr.label != "well-constrained"


@pytest.mark.slow
def test_p6_iterative_regime_probe_catches_null_above_dense_below() -> None:
    """n_params=3000 > dense_below=2048 ⇒ the genuine iterative regime (not the dense fallback).
    The inverse-iteration probe must still CATCH the isolated null → ``unidentifiable`` with a
    reported near-zero smallest eigenvalue."""
    fn, _theta = _structural_null_model(3000, 4000, 0)
    rep = analyze_model_matrixfree(fn, sigma=SIGMA)
    assert rep.label == "unidentifiable"
    assert rep.n_null_dims >= 1
    assert rep.smallest_eigenvalue < 1e-6 * rep.largest_eigenvalue  # a genuine near-null


# --------------------------------------------------------------------------- #
# (b) POSITIVE CONTROLS — the fix does NOT over-abstain on the affordable path
# --------------------------------------------------------------------------- #
def test_p6_positive_control_well_constrained_still_resolves() -> None:
    """A genuinely well-conditioned full-rank model at n_params=300 (< dense_below) must still
    resolve ``well-constrained`` via ``auto`` (routed to the exact dense path)."""
    fn, _theta = _well_conditioned_fullrank(300, 400, 0)
    rep = analyze_model_matrixfree(fn, sigma=SIGMA)
    assert rep.label == "well-constrained"
    assert rep.n_null_dims == 0


def test_p6_positive_control_sloppy_still_resolves() -> None:
    """The canonical sum-of-exponentials (sloppy but predictive, no structural null) must still
    resolve ``sloppy-but-predictive`` via ``auto`` — and match the dense oracle bit-for-bit."""
    t = np.linspace(0.05, 6.0, 60)
    fn = sum_of_exponentials_predict(rates=[0.5, 1.3, 2.5, 4.5], amps=[1, 1, 1, 1], t=t)
    dense = analyze_model(fn, sigma=SIGMA)
    auto = analyze_model_matrixfree(fn, sigma=SIGMA)
    assert dense.label == "sloppy-but-predictive"
    assert auto.label == "sloppy-but-predictive"


def test_p6_dense_matrixfree_equivalence_on_well_conditioned() -> None:
    """Preserve the advertised dense/matrix-free agreement: on the well-conditioned control the
    ``method="dense"`` and ``method="auto"`` labels + null counts coincide."""
    fn = well_conditioned_predict(slope=2.0, offset=1.0, t=np.linspace(0.05, 6.0, 60))
    dense = sloppiness_diagnostic_matrixfree(fn, fn.theta0, SIGMA, method="dense")
    auto = analyze_model_matrixfree(fn, sigma=SIGMA)
    assert dense.label == auto.label == "well-constrained"
    assert dense.n_null_dims == auto.n_null_dims == 0


# --------------------------------------------------------------------------- #
# (c) DECOY (strict-xfail) — the residual bound: the iterative path CANNOT certify
#     a genuine well-constrained model matrix-free (it abstains). Locking this as a
#     strict xfail means: if a future change makes the iterative path return a positive
#     verdict here, the suite XPASSES and forces a re-audit (is the new certificate sound,
#     or did the fail-safe regress into a possible confident-wrong?). See NUDGE-LIM-023.
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="NUDGE-LIM-023 residual bound: a matrix-free Krylov path cannot certify the SMALLEST "
    "eigenvalue, so a genuine well-constrained model above dense_below ABSTAINS "
    "(unidentifiable) rather than assert well-constrained. Over-abstains, never confident-wrong.",
)
def test_p6_decoy_huge_wellconditioned_cannot_be_certified_iterative() -> None:
    """DECOY: we WISH a genuine well-constrained model at n_params > dense_below resolved
    ``well-constrained`` via ``auto`` (iterative), but the smallest end is not certifiable
    matrix-free — so it honestly abstains. This assert therefore FAILS by design (strict xfail).
    (The exact verdict remains available on demand via ``method="dense"``.)"""
    fn, _theta = _well_conditioned_fullrank(3000, 4000, 0)
    rep = analyze_model_matrixfree(fn, sigma=SIGMA)  # auto -> iterative (3000 > 2048)
    assert rep.label == "well-constrained"  # EXPECTED TO FAIL — it abstains (unidentifiable)
