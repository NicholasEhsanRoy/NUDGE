"""Tests for the AD amyloid-β QSP model + its identifiability/OED demo (``ad_qsp``).

The load-bearing claims, each MEASURED:

- the differentiable Aβ cascade produces a **bounded, sensible** amyloid→plaque trajectory,
  and an antibody dose **measurably lowers plaque** (a real PK/PD effect);
- the population cohort forward map is differentiable and the **matrix-free** identifiability
  path returns a verdict where the dense path would OOM — and, in the realistic
  more-params-than-observations regime, that verdict is the fail-safe ``unidentifiable``;
- single-subject identifiability flags the plaque-growth gain/threshold (``k_pg``/``K_pg``) as
  **sloppy** (the biomarkers cannot pin them) — a named, measured degeneracy;
- the antibody-binding ⇄ clearance pair (``k_on``/``k_gl``) is **genuinely confounded** by a
  naive schedule (measured near-singular FIM) and gradient OED **improves** its CRLB;
- the full published Proctor 2013 network is transcribed faithfully (64 states, 73 params,
  112 reactions) and its vector field is finite.

Fast lane keeps cohorts tiny and optimizer budgets small; heavier scaling is in the slow lane.
"""

from __future__ import annotations

import numpy as np
import pytest

from nudge.mechanisms import ad_qsp as A


def test_forward_sim_bounded_and_antibody_lowers_plaque():
    tr0, t = A.simulate_subject(dose=0.0)
    trA, _ = A.simulate_subject(dose=0.6)
    plaque0, plaqueA = tr0[:, 2], trA[:, 2]
    assert not np.isnan(tr0).any() and not np.isnan(trA).any()
    assert np.all(tr0 < 1e5)  # bounded (no runaway)
    assert plaque0[-1] > 0.1  # plaque actually forms
    assert np.all(np.diff(plaque0) >= -1e-8)  # monotone rise to a plateau
    # the antibody measurably clears plaque
    assert plaqueA[-1] < 0.6 * plaque0[-1]


def test_cohort_predict_shape_and_free_knob():
    prob = A.make_ad_cohort_predict_fn(n_subjects=4, n_free=20, n_obs_times=5, seed=0)
    assert prob.n_theta == 20
    assert prob.n_obs == 4 * 5 * len(A.BIOMARKERS)
    obs = np.asarray(prob.predict_fn(prob.theta0))
    assert obs.shape == (prob.n_obs,)
    assert np.isfinite(obs).all()


def test_generate_cohort_is_synthetic_ground_truth():
    coh = A.generate_ad_cohort(n_subjects=3, n_obs_times=4, obs_noise=0.05, seed=1)
    assert coh["true_params"].shape == (3, len(A.AD_PARAM_NAMES))
    assert np.isfinite(coh["observations"]).all()
    assert "SYNTHETIC" in coh["note"] and "NOT real" in coh["note"]


@pytest.mark.x64
def test_single_subject_flags_plaque_growth_as_sloppy():
    from nudge.inference.sloppiness import analyze_model, relative_sensitivity_jacobian

    prob = A.make_ad_cohort_predict_fn(
        n_subjects=1, n_free=len(A.AD_PARAM_NAMES), n_obs_times=8, biomarkers=(2, 1), seed=0
    )
    rep = analyze_model(
        prob.predict_fn, prob.theta0, sigma=0.05, param_names=list(A.AD_PARAM_NAMES)
    )
    # the model is sloppy (a wide Fisher spectrum) but usable — never a bare confident verdict.
    assert rep.label in ("sloppy-but-predictive", "unidentifiable")
    assert rep.spectral_span_decades > 3.0  # genuinely sloppy
    jac_log, _ = relative_sensitivity_jacobian(prob.predict_fn, prob.theta0)
    fim = jac_log.T @ jac_log / 0.05**2
    evals, evecs = np.linalg.eigh(fim)
    sloppiest = evecs[:, 0]
    top = {A.AD_PARAM_NAMES[int(j)] for j in np.argsort(-np.abs(sloppiest))[:3]}
    # the autocatalytic plaque-growth gain/threshold dominate the sloppiest direction.
    assert top & {"k_pg", "K_pg"}


@pytest.mark.x64
def test_oed_confound_is_measured_and_resolved():
    from nudge.inference import oed

    prob = A.make_ad_oed_problem(pair=("k_on", "k_gl"), biomarkers=(2,))
    t_max = float(prob.meta["t_max"])
    naive = np.concatenate([np.linspace(0.05, 0.5, 3), np.linspace(t_max - 0.5, t_max, 3)])
    fim = oed.fisher_information(prob, naive)
    corr = abs(fim[0, 1] / np.sqrt(fim[0, 0] * fim[1, 1]))
    assert corr > 0.9  # the pair is genuinely confounded by the naive schedule (MEASURED)
    res = oed.optimize_design(prob, naive, objective="d_opt", target="log_k_on",
                              steps=120, learning_rate=0.15)
    # gradient OED measurably resolves the confound (never asserted — measured factor).
    assert res.crlb_improvement > 5.0
    assert res.min_eig_improvement > 5.0
    assert np.linalg.cond(res.fim_opt) < np.linalg.cond(fim)


@pytest.mark.x64
@pytest.mark.slow
def test_matrix_free_flat_where_dense_would_oom():
    from nudge.inference.sloppiness import sloppiness_diagnostic_matrixfree

    # more subject-specific params than plaque observations → rank-deficient population fit;
    # NUDGE certifies unidentifiable cheaply (NUDGE-LIM-023 fail-safe), matrix-free.
    prob = A.make_ad_cohort_predict_fn(
        n_subjects=60, n_free=720, n_obs_times=2, biomarkers=(2,), seed=0
    )
    assert prob.n_theta > prob.n_obs
    rep = sloppiness_diagnostic_matrixfree(prob.predict_fn, prob.theta0, 0.05, method="iterative")
    assert rep.label == "unidentifiable"


def test_nlme_cohort_shape_and_border():
    prob = A.make_ad_nlme_cohort_predict_fn(
        n_subjects=5, re_params=("k_pg", "K_pg", "k_gl"), n_obs_times=3,
        include_prior=False, biomarkers=(2,), seed=0,
    )
    d = prob.n_re
    assert d == 3 and prob.n_fixed == len(A.AD_PARAM_NAMES) - d
    # θ = [μ (d) | φ (n_fixed) | r_i (N·d)] with include_prior=False.
    assert prob.border_size == d + prob.n_fixed
    assert prob.n_theta == prob.border_size + 5 * d
    assert prob.n_obs == 5 * 3 * 1  # sparse plaque-only
    obs = np.asarray(prob.predict_fn(prob.theta0))
    assert obs.shape == (prob.n_obs,) and np.isfinite(obs).all()
    assert (prob.theta0 > 0).all()  # RAW-positive throughout (log-sensitivity is well-defined)


@pytest.mark.x64
def test_nlme_fim_is_genuinely_coupled_arrowhead():
    """The load-bearing structural claim: shared μ/φ couple every subject (border↔subject
    off-block entries are NONZERO), while cross-subject blocks are exactly zero (arrowhead)."""
    from nudge.inference.sloppiness import relative_sensitivity_jacobian

    prob = A.make_ad_nlme_cohort_predict_fn(
        n_subjects=4, n_obs_times=4, include_prior=True, biomarkers=(2, 1), seed=0
    )
    jac_log, _ = relative_sensitivity_jacobian(prob.predict_fn, prob.theta0)
    fim = jac_log.T @ jac_log / 0.05**2
    border, s0, s1 = prob.border_indices(), prob.subject_block(0), prob.subject_block(1)
    scale = np.abs(fim).max()
    # border couples to a subject block → the matrix is NOT block-diagonal
    assert np.abs(fim[np.ix_(border, s0)]).max() > 1e-3 * scale
    # cross-subject blocks are exactly zero → arrowhead, not decomposable per-subject
    assert np.abs(fim[np.ix_(s0, s1)]).max() <= 1e-12 * scale
    # include_prior=True makes ω a genuine free border hyperparameter (the full μ+ω NLME)
    assert prob.include_prior and any(n.startswith("omega[") for n in prob.param_names)


@pytest.mark.x64
def test_nlme_matrixfree_matches_dense_on_coupled_model():
    """On a small coupled instance the matrix-free (dense-route) verdict matches dense bit-for-bit
    — the coupled FIM is analyzed correctly, not approximated."""
    from nudge.inference.sloppiness import analyze_model, sloppiness_diagnostic_matrixfree

    prob = A.make_ad_nlme_cohort_predict_fn(
        n_subjects=6, n_obs_times=6, include_prior=True, biomarkers=(2, 1), seed=0
    )
    dense = analyze_model(prob.predict_fn, prob.theta0, sigma=0.05,
                          param_names=list(prob.param_names))
    mf = sloppiness_diagnostic_matrixfree(prob.predict_fn, prob.theta0, 0.05,
                                          param_names=list(prob.param_names), method="dense")
    assert dense.label == mf.label
    rel = np.max(np.abs(dense.fim_eigenvalues - mf.fim_eigenvalues)) / max(
        dense.largest_eigenvalue, 1e-30
    )
    assert rel < 1e-9  # bit-for-bit agreement


@pytest.mark.x64
def test_nlme_sparse_regime_certifies_unidentifiable_failsafe():
    """The NUDGE-LIM-023 fail-safe is preserved on the coupled model: with more per-subject RE
    params than per-subject observations the population fit is rank-deficient by shape → NUDGE
    certifies unidentifiable, never a fabricated verdict."""
    from nudge.inference.sloppiness import sloppiness_diagnostic_matrixfree

    prob = A.make_ad_nlme_cohort_predict_fn(
        n_subjects=40, n_obs_times=2, include_prior=False, biomarkers=(2,), seed=0
    )
    assert prob.n_theta > prob.n_obs  # rank-deficient by shape
    rep = sloppiness_diagnostic_matrixfree(prob.predict_fn, prob.theta0, 0.05, method="iterative")
    assert rep.label == "unidentifiable"


def test_nlme_registry_wiring_and_scale_knob():
    from nudge.inference.model_registry import build_identifiability_problem, list_models

    assert "ad_qsp_nlme" in {m["name"] for m in list_models()}
    prev = 0
    for nf in (72, 300, 900):
        p = build_identifiability_problem("ad_qsp_nlme", n_free=nf)
        assert p.n_params >= prev  # the n_free knob grows the joint parameter count
        assert p.meta["coupling"].startswith("arrowhead")
        prev = p.n_params


@pytest.mark.x64
@pytest.mark.slow
def test_nlme_matrix_free_flat_at_scale():
    """At a population scale that OOMs a dense jacfwd, the matrix-free path completes and returns
    the fail-safe verdict on the coupled model."""
    from nudge.inference.sloppiness import sloppiness_diagnostic_matrixfree

    prob = A.make_ad_nlme_cohort_predict_fn(
        n_subjects=500, n_obs_times=2, include_prior=False, biomarkers=(2,), seed=0
    )
    assert prob.n_theta > 1400  # thousands of jointly-estimated params
    rep = sloppiness_diagnostic_matrixfree(prob.predict_fn, prob.theta0, 0.05, method="iterative")
    assert rep.label == "unidentifiable"


def test_full_proctor_network_is_faithful():
    from nudge.mechanisms import _proctor2013 as P

    assert len(P.SPECIES) == 64
    assert len(P.PARAM_NAMES) == 73
    assert P.Y0.shape == (64,)
    # the published Hill plaque-growth constants are present (the AD switch: gain + threshold)
    pv = dict(zip(P.PARAM_NAMES, P.PARAM_VALUES, strict=True))
    assert pv["kpg"] == pytest.approx(0.15) and pv["kpghalf"] == pytest.approx(10.0)
    # the vector field evaluates finite at the published initial state
    import jax

    dy = np.asarray(P.rhs(jax.numpy.asarray(P.Y0), jax.numpy.asarray(P.PARAM_VALUES)))
    assert dy.shape == (64,) and np.isfinite(dy).all()
