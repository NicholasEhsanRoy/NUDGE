"""Tests for gradient-based Optimal Experimental Design (``nudge.inference.oed``).

The load-bearing claims, each MEASURED (never asserted):

- the design gradient ``∂criterion/∂φ`` exists and is finite (the white-box moat);
- a naive near-equilibrium design is *measurably* degenerate on the α⇄β pair (tiny FIM
  smallest eigenvalue / near-singular curvature) — the sloppy starting point;
- gradient-ascending the design **improves** the target parameter's Cramér–Rao bound and
  the FIM's smallest eigenvalue by a measured factor (identifiability recovered);
- the improvement holds across objectives (D-/E-optimality, targeted CRLB) and generalizes
  to the multi-species gLV builder.

Fast lane keeps ``t_max`` short, ``dt`` coarse, and the optimizer budget small; the deeper
default-scale recovery is in the slow lane.
"""

from __future__ import annotations

import numpy as np
import pytest

from nudge.inference.oed import (
    criterion_value,
    crlb,
    design_gradient,
    fisher_information,
    grid_search_design,
    make_glv_design_problem,
    make_logistic_design_problem,
    min_eigenvalue,
    optimize_design,
)


def _fast_logistic():
    """A small, fast logistic problem for the CI lane."""
    return make_logistic_design_problem(t_max=6.0, dt=0.05)


def _naive(problem, m=6):
    """A naive near-equilibrium design (the realistic 'measure at steady state' default)."""
    lo, hi = problem.phi_bounds
    return np.linspace(0.6 * hi, hi, m)


# --------------------------------------------------------------------------- #
# FIM + criterion primitives
# --------------------------------------------------------------------------- #
def test_fim_symmetric_and_psd() -> None:
    prob = _fast_logistic()
    fim = fisher_information(prob, _naive(prob))
    assert fim.shape == (2, 2)
    np.testing.assert_allclose(fim, fim.T, rtol=1e-9, atol=1e-9)
    assert min_eigenvalue(fim) >= -1e-8  # PSD (up to numerical floor)


def test_naive_near_equilibrium_design_is_degenerate() -> None:
    """The α⇄β confound is MEASURED: near-equilibrium sampling → near-singular FIM."""
    prob = _fast_logistic()
    fim_naive = fisher_information(prob, _naive(prob))
    # a design that also samples the transient is far better conditioned.
    lo, hi = prob.phi_bounds
    transient = np.linspace(lo, hi, 6)
    fim_trans = fisher_information(prob, transient)
    cond_naive = np.linalg.cond(fim_naive)
    cond_trans = np.linalg.cond(fim_trans)
    assert cond_naive > cond_trans  # transient sampling breaks the confound
    # the growth CRLB is strictly worse under the naive design.
    assert crlb(fim_naive)[0] > crlb(fim_trans)[0]


def test_design_gradient_is_finite_and_nonzero() -> None:
    """The white-box moat: ∂criterion/∂φ exists (a black-box solver cannot form it)."""
    prob = _fast_logistic()
    g = design_gradient(prob, _naive(prob), objective="crlb", target="log_alpha")
    assert g.shape == (6,)
    assert np.all(np.isfinite(g))
    assert np.linalg.norm(g) > 1e-6


def test_criterion_transient_beats_equilibrium() -> None:
    prob = _fast_logistic()
    lo, hi = prob.phi_bounds
    c_naive = criterion_value(prob, _naive(prob), objective="crlb", target="log_alpha")
    c_trans = criterion_value(prob, np.linspace(lo, hi, 6), objective="crlb",
                              target="log_alpha")
    assert c_trans > c_naive  # higher reciprocal-CRLB = better identified


def test_unknown_objective_raises() -> None:
    prob = _fast_logistic()
    with pytest.raises(ValueError, match="unknown objective"):
        criterion_value(prob, _naive(prob), objective="not-a-thing")
    with pytest.raises(ValueError, match="unknown objective"):
        optimize_design(prob, _naive(prob), objective="nope")


# --------------------------------------------------------------------------- #
# the optimizer — measured identifiability gain
# --------------------------------------------------------------------------- #
def test_optimize_improves_target_crlb() -> None:
    """Gradient-ascent measurably resolves the previously-sloppy growth parameter."""
    prob = _fast_logistic()
    res = optimize_design(
        prob, _naive(prob), objective="crlb", target="log_alpha",
        steps=200, learning_rate=0.2,
    )
    # the optimum is at least as good as the start on the objective (ascent),
    assert res.criterion_opt >= res.criterion_init
    # and the target CRLB improves (this coarse/fast config measures ~2.5x; the large
    # ~31x headline gain is validated in the slow lane at default resolution).
    assert res.crlb_improvement > 1.5
    assert res.target_crlb_opt < res.target_crlb_init
    assert res.min_eig_improvement > 1.0
    # the optimal design stays inside the physical window.
    lo, hi = prob.phi_bounds
    assert np.all(res.phi_opt >= lo - 1e-9) and np.all(res.phi_opt <= hi + 1e-9)


def test_grid_search_returns_best_candidate() -> None:
    prob = _fast_logistic()
    lo, hi = prob.phi_bounds
    rng = np.random.default_rng(0)
    cands = [np.sort(rng.uniform(lo, hi, 6)) for _ in range(40)]
    gs = grid_search_design(prob, cands, objective="crlb", target="log_alpha")
    assert gs.n_evaluations == 40
    assert gs.best_criterion == pytest.approx(float(np.max(gs.all_criteria)))


# --------------------------------------------------------------------------- #
# slow lane — default-scale recovery, all objectives, gLV generalization
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.x64
def test_optimize_default_scale_large_gain() -> None:
    prob = make_logistic_design_problem()  # default t_max=12
    res = optimize_design(
        prob, _naive(prob, m=8), objective="crlb", target="log_alpha",
        steps=400, learning_rate=0.2,
    )
    assert res.crlb_improvement > 10.0  # the headline: a large, measured resolution gain
    assert res.min_eig_improvement > 5.0


@pytest.mark.slow
@pytest.mark.x64
@pytest.mark.parametrize("objective", ["d_opt", "e_opt", "crlb"])
def test_objectives_all_improve_identifiability(objective: str) -> None:
    prob = make_logistic_design_problem()
    res = optimize_design(
        prob, _naive(prob, m=8), objective=objective, target="log_alpha",
        steps=400, learning_rate=0.2,
    )
    # every objective raises its own criterion AND resolves the sloppy pair.
    assert res.criterion_opt >= res.criterion_init
    assert res.min_eig_improvement > 3.0


@pytest.mark.slow
@pytest.mark.x64
def test_glv_multispecies_generalizes() -> None:
    prob = make_glv_design_problem(n_species=3, target=0)
    res = optimize_design(
        prob, _naive(prob, m=8), objective="crlb", target="log_alpha_t",
        steps=400, learning_rate=0.2,
    )
    assert res.crlb_improvement > 10.0
    assert res.target_crlb_opt < res.target_crlb_init


# --------------------------------------------------------------------------- #
# rank-deficient-naive honesty (NUDGE-LIM-029): the guard is MEASURED (min-eig vs the
# ridge floor at the FIM's own scale) and must not fire on a well-conditioned design.
# --------------------------------------------------------------------------- #
from nudge.inference.oed import (  # noqa: E402
    is_rank_deficient,
    ridge_floor,
    target_ridge_dominated,
)


def test_rank_deficiency_helpers_on_constructed_fims() -> None:
    """UNIT: the curvature-grounded rank test agrees with construction — a well-conditioned
    FIM is not rank-deficient; an exactly-singular FIM (a zero flat direction) is, and its
    target CRLB is a ridge artifact (var ~halves when the ridge doubles)."""
    well = np.diag([10.0, 4.0])
    assert not is_rank_deficient(well)
    assert min_eigenvalue(well) > ridge_floor(well)
    dominated, ratio = target_ridge_dominated(well, 1)
    assert not dominated and ratio < 1.5  # ridge-insensitive → genuine information

    singular = np.array([[10.0, 0.0], [0.0, 0.0]])  # parameter 1 is a flat direction
    assert is_rank_deficient(singular)
    assert min_eigenvalue(singular) <= ridge_floor(singular)
    dominated, ratio = target_ridge_dominated(singular, 1)
    assert dominated and ratio > 1.5  # var(r)/var(2r) → 2 → pure ridge artifact


@pytest.mark.x64
def test_logistic_default_naive_is_not_flagged_rank_deficient() -> None:
    """A SECOND non-singular model: the logistic OED default naive schedule is informative
    (min_eig ≫ ridge floor), so the guard must NOT fire and the finite gain stands."""
    prob = make_logistic_design_problem()
    res = optimize_design(
        prob, _naive(prob, m=8), objective="crlb", target="log_alpha",
        steps=120, learning_rate=0.2,
    )
    assert res.naive_rank_deficient is False
    assert res.naive_target_identifiable is True
    assert res.crlb_improvement_is_lower_bound is False
    assert res.note == ""


def test_decoy_well_conditioned_design_is_never_flagged() -> None:
    """DECOY (NUDGE-LIM-029): a well-conditioned, genuinely-informative design must NEVER be
    flagged rank-deficient — the guard fires ONLY on true ridge-floor degeneracy, so it can
    never silently over-abstain on an informative design. The hole (a false-precise finite
    factor on a rank-deficient baseline) is CLOSED, so this decoy PASSES (not strict-xfail)."""
    prob = _fast_logistic()
    fim = fisher_information(prob, _naive(prob, m=6))
    assert not is_rank_deficient(fim)
    res = optimize_design(
        prob, _naive(prob, m=6), objective="crlb", target="log_alpha", steps=40,
    )
    assert res.naive_rank_deficient is False
    assert res.crlb_improvement_is_lower_bound is False
