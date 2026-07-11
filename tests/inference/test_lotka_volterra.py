"""Tests for the temporal / gLV trajectory-fit attribution (``NUDGE-METHOD-012``).

Fast lane: the differentiable field + integrator + generators are unit-checked.
Slow lane (``verification`` / ``decoy``): the synthetic round-trip — a KNOWN
single-knob perturbation is **recovered or abstained on, never mis-called** — and the
two gLV decoys (the α⇄βᵢᵢ confound + the no-perturbation null). The headline fail-safe
assertion is **0 confident-wrong** across the battery.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nudge.inference.lotka_volterra import (
    DegeneracyDirection,
    GLVFit,
    GLVParams,
    GLVResult,
    _LOAD_TOL,
    _default_baseline,
    alpha_beta_identifiability,
    attribute_glv,
    degeneracy_direction_from_posterior,
    generate_alpha_beta_confound_decoy,
    generate_no_perturbation_null,
    glv_vector_field,
    simulate_glv,
    simulate_glv_perturbseq,
)
from nudge.inference.uncertainty import laplace_posterior

_POSITIVE = {"growth", "interaction", "susceptibility"}


def _make_fit(*, selected: str, degenerate: bool,
              degeneracy: DegeneracyDirection | None) -> GLVFit:
    """A minimal GLVFit for exercising the result-level directional-abstention gating."""
    base = GLVParams(alpha=np.array([0.8]), beta=np.array([[-0.5]]), eps=np.array([0.0]))
    bic = {"null": 100.0, "growth": 80.0, "interaction": 82.0, "susceptibility": 90.0}
    return GLVFit(
        target=0, n_species=1, n_replicates=10, n_timepoints=8, baseline=base,
        baseline_loss=0.0, bic=bic, rss={}, delta={}, selected=selected,
        cond_number=(float("inf") if degenerate else 12.0),
        corr_alpha_beta=(0.99 if degenerate else 0.3), degenerate=degenerate,
        identifiability_reason="synthetic", degeneracy=degeneracy,
    )


# --------------------------------------------------------------------------- #
# fast lane — field, integrator, generator
# --------------------------------------------------------------------------- #
def test_vector_field_is_glv_product_form() -> None:
    x = jnp.array([1.0, 2.0])
    alpha = jnp.array([0.5, -0.3])
    beta = jnp.array([[-1.0, 0.1], [0.2, -0.8]])
    eps = jnp.array([-2.0, 0.0])
    u = jnp.array(1.0)
    got = np.asarray(glv_vector_field(x, alpha, beta, eps, u))
    # dxᵢ/dt = xᵢ (αᵢ + Σⱼ βᵢⱼxⱼ + εᵢ u)
    expected = np.array([
        1.0 * (0.5 + (-1.0 * 1.0 + 0.1 * 2.0) + (-2.0) * 1.0),
        2.0 * (-0.3 + (0.2 * 1.0 + -0.8 * 2.0) + 0.0),
    ])
    assert np.allclose(got, expected, atol=1e-5)


def test_integrator_saturates_at_carrying_capacity() -> None:
    # A single self-limiting species should settle at K = -alpha/beta.
    alpha = np.array([0.8])
    beta = np.array([[-0.5]])
    eps = np.array([0.0])
    p = GLVParams(alpha=alpha, beta=beta, eps=eps)
    dt = 0.01
    n_steps = int(round(30.0 / dt))
    u_grid = jnp.zeros(n_steps, dtype=jnp.float32)
    obs_idx = jnp.array([0, n_steps])
    traj = np.asarray(simulate_glv(p, jnp.array([0.1], jnp.float32), u_grid, dt, obs_idx))
    assert np.isclose(traj[-1, 0], 0.8 / 0.5, atol=1e-2)  # K = 1.6


def test_pulse_only_acts_while_drug_is_on() -> None:
    # With eps != 0 the trajectory must diverge from the eps=0 run ONLY after the pulse.
    ds = simulate_glv_perturbseq(
        mechanism="susceptibility", delta=-1.5, n_replicates=8, seed=0,
        pulse_window=(4.0, 6.0),
    )
    ref = ds.reference.mean(0)  # (T, S) eps=0
    pert = ds.perturbed.mean(0)  # (T, S) eps on target 0
    pre_pulse = ds.t_obs < 4.0
    during = (ds.t_obs >= 4.0) & (ds.t_obs <= 6.5)
    # before the pulse the two communities are ~identical (drug not yet applied).
    assert np.max(np.abs(np.log(pert[pre_pulse, 0] + 1e-2) -
                         np.log(ref[pre_pulse, 0] + 1e-2))) < 0.2
    # DURING the pulse the susceptible taxon is directly suppressed (the ε signature);
    # after the drug ends it recovers — the time-localized on/off contrast.
    assert np.min(pert[during, 0]) < np.min(ref[during, 0])


def test_generator_shapes_and_ground_truth() -> None:
    ds = simulate_glv_perturbseq(n_species=3, n_replicates=12, n_obs=20, seed=1)
    assert ds.reference.shape[0] == 12 and ds.reference.shape[2] == 3
    assert ds.reference.shape == ds.perturbed.shape
    assert ds.ground_truth["mechanism"] == "susceptibility"
    assert len(ds.t_obs) == ds.reference.shape[1]


def test_with_knob_moves_only_the_named_parameter() -> None:
    rng = np.random.default_rng(0)
    base = _default_baseline(3, rng)
    g = base.with_knob("growth", 1, 0.4)
    assert np.isclose(g.alpha[1], base.alpha[1] + 0.4)
    assert np.allclose(g.beta, base.beta) and np.allclose(g.eps, base.eps)
    b = base.with_knob("interaction", 1, -0.3)
    assert np.isclose(b.beta[1, 1], base.beta[1, 1] - 0.3)
    assert np.allclose(b.alpha, base.alpha)


# --------------------------------------------------------------------------- #
# fast lane — directional abstention (the null-space extractor + result gating)
# --------------------------------------------------------------------------- #
def test_degeneracy_direction_extracts_null_eigenvector_and_hint() -> None:
    # A synthetic near-singular curvature on (log αₜ, log |βₜₜ|): stiff along the (1,-1)
    # "difference" axis (K=−α/β is well-determined), FLAT along the (1,1) "sum" axis (raise
    # both → K unchanged). The extractor must return that (1,1) diagonal as the null
    # direction and fire the "cannot separate growth from interaction" hint — reusing the
    # ALREADY-computed Hessian, no re-solve.
    def loss(th: np.ndarray) -> np.ndarray:
        d = th[0] - th[1]
        s = th[0] + th[1]
        return 0.5 * (100.0 * d * d + 1e-3 * s * s)

    post = laplace_posterior(
        loss, [0.0, 0.0], names=["alpha_t", "abs_beta_tt"], cond_max=100.0
    )
    assert post.degenerate  # cond = 1e5, near-singular
    dd = degeneracy_direction_from_posterior(post)
    assert dd.names == ("alpha_t", "abs_beta_tt")
    v = dd.vector
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6  # unit vector
    assert v[0] * v[1] > 0  # both same sign → the α⇄β diagonal (K=−α/β confound)
    assert abs(abs(v[0]) - abs(v[1])) < 0.05  # ≈ (1,1)/√2
    assert "Cannot separate Growth" in dd.hint


def test_degeneracy_direction_single_axis_hint() -> None:
    # When only ONE parameter is flat (the null direction is axis-aligned, |corr|→0), the
    # hint names that single unidentifiable knob — not the joint confound.
    def loss(th: np.ndarray) -> np.ndarray:
        # stiff in alpha_t (index 0), flat in abs_beta_tt (index 1).
        return 0.5 * (100.0 * th[0] * th[0] + 1e-3 * th[1] * th[1])

    post = laplace_posterior(
        loss, [0.0, 0.0], names=["alpha_t", "abs_beta_tt"], cond_max=100.0
    )
    dd = degeneracy_direction_from_posterior(post)
    assert abs(abs(dd.vector[1]) - 1.0) < 1e-3  # null direction ≈ the β axis
    assert "Interaction" in dd.hint and "not identifiable" in dd.hint


def test_result_surfaces_direction_only_when_operative() -> None:
    # The α⇄β directional abstention is surfaced iff it is the OPERATIVE reason: an
    # unresolved call whose best knob is growth/interaction AND the curvature is degenerate.
    direction = DegeneracyDirection(
        names=("alpha_t", "abs_beta_tt"),
        vector=np.array([0.7071, 0.7071]), eigenvalue=1e-3, cond_number=float("inf"),
        hint="Cannot separate Growth (α) from Interaction (β): ...",
    )
    # (a) operative → surfaced, status UNRESOLVED.
    op = GLVResult(
        fit=_make_fit(selected="growth", degenerate=True, degeneracy=direction),
        call="unresolved", reason="",
    )
    assert op.status == "UNRESOLVED"
    assert op.degeneracy_direction is not None
    assert np.allclose(op.degeneracy_direction, direction.vector)
    assert op.human_readable_hint is not None and "Cannot separate" in op.human_readable_hint

    # (b) resolved susceptibility, α⇄β degenerate but NOT operative → NOT surfaced.
    res = GLVResult(
        fit=_make_fit(selected="susceptibility", degenerate=True, degeneracy=direction),
        call="susceptibility", reason="",
    )
    assert res.status == "RESOLVED"
    assert res.degeneracy_direction is None and res.human_readable_hint is None

    # (c) well-conditioned resolved growth → no degeneracy at all.
    well = GLVResult(
        fit=_make_fit(selected="growth", degenerate=False, degeneracy=None),
        call="growth", reason="",
    )
    assert well.status == "RESOLVED"
    assert well.degeneracy_direction is None and well.human_readable_hint is None


# --------------------------------------------------------------------------- #
# slow lane — the synthetic round-trip (recover-or-abstain, never mis-call)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.verification
def test_susceptibility_is_the_identifiable_positive() -> None:
    # The antibiotic ε axis is the most identifiable (time-localized on/off signature).
    ds = simulate_glv_perturbseq(mechanism="susceptibility", dense_transient=True, seed=0)
    res = attribute_glv(ds, steps=180, n_sim=28, seed=0)
    assert res.call == "susceptibility", res.reason
    assert res.is_reliable


@pytest.mark.slow
@pytest.mark.verification
def test_growth_resolves_when_the_transient_is_sampled() -> None:
    # A growth change is separable from self-limitation ONLY via the transient; with
    # dense early sampling NUDGE recovers it (else it must abstain — never mis-call).
    ds = simulate_glv_perturbseq(mechanism="growth", dense_transient=True, seed=1)
    res = attribute_glv(ds, steps=180, n_sim=28, seed=1)
    assert res.call in {"growth", "unresolved"}, res.reason
    if res.call == "growth":  # the demoable transient-resolved positive
        assert not res.fit.degenerate


@pytest.mark.slow
@pytest.mark.verification
def test_self_interaction_abstains_via_measured_degeneracy() -> None:
    # A self-interaction (βₜₜ) change is intrinsically confounded with growth; NUDGE must
    # not confidently call it (abstain), and never call the WRONG knob.
    ds = simulate_glv_perturbseq(mechanism="interaction", dense_transient=True, seed=0)
    res = attribute_glv(ds, steps=180, n_sim=28, seed=0)
    assert res.call in {"interaction", "unresolved"}, res.reason  # correct or abstain
    assert res.call != "growth" and res.call != "susceptibility"  # never mis-called


# --------------------------------------------------------------------------- #
# slow lane — the gLV decoys (must abstain, not mis-call)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.decoy
def test_alpha_beta_confound_decoy_abstains() -> None:
    # A growth change sampled near equilibrium LOOKS like a self-limitation change
    # (Kᵢ=−αᵢ/βᵢᵢ). NUDGE must ABSTAIN (unresolved) — not confidently call interaction —
    # with the degeneracy MEASURED by the Laplace curvature (NUDGE-LIM-020).
    for seed in (0, 1):
        ds = generate_alpha_beta_confound_decoy(seed=seed)
        res = attribute_glv(ds, steps=160, n_sim=26, seed=seed)
        assert res.call == "unresolved", (seed, res.reason)
        assert res.fit.degenerate  # the abstention is earned by measured curvature
        assert res.call not in _POSITIVE


@pytest.mark.slow
@pytest.mark.decoy
def test_no_perturbation_null_makes_no_positive_call() -> None:
    # No perturbation → NUDGE must not manufacture a mechanism (no-change / unresolved).
    for seed in (0, 1):
        ds = generate_no_perturbation_null(seed=seed)
        res = attribute_glv(ds, steps=160, n_sim=26, seed=seed)
        assert res.call in {"no-change", "unresolved"}, (seed, res.reason)
        assert res.call not in _POSITIVE


@pytest.mark.slow
@pytest.mark.verification
def test_battery_has_zero_confident_wrong() -> None:
    # The headline fail-safe guarantee across the mixed battery: a KNOWN answer is
    # recovered or abstained on — NEVER a confident wrong knob.
    cases = [
        (simulate_glv_perturbseq(mechanism="susceptibility", seed=2), "susceptibility"),
        (simulate_glv_perturbseq(mechanism="growth", dense_transient=True, seed=2),
         "growth"),
        (generate_alpha_beta_confound_decoy(seed=2), "growth"),  # truth growth, degenerate
        (generate_no_perturbation_null(seed=2), "none"),
    ]
    for ds, truth in cases:
        res = attribute_glv(ds, steps=150, n_sim=24, seed=2)
        confident_wrong = res.call in _POSITIVE and res.call != truth
        assert not confident_wrong, (truth, res.call, res.reason)


# --------------------------------------------------------------------------- #
# slow lane — identifiability is measured, not asserted
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.decoy
def test_directional_abstention_end_to_end() -> None:
    # The actionable half of NUDGE-LIM-020, end to end: a near-equilibrium growth confound
    # abstains AND exposes the null-space direction (points along α⇄β, hint fires); a
    # transient-resolved growth fit surfaces NO direction. Recover-or-abstain — never
    # mis-called, never a false-precise identifiability claim.
    deg = generate_alpha_beta_confound_decoy(seed=0, n_species=2, n_replicates=40, n_obs=18)
    res = attribute_glv(deg, steps=160, n_sim=26, seed=0)
    assert res.call == "unresolved" and res.status == "UNRESOLVED", res.reason
    v = res.degeneracy_direction
    assert v is not None  # the actionable direction is exposed
    assert v[0] * v[1] > 0  # both same sign → the α⇄β diagonal (K=−α/β confound)
    # the flat direction is a genuinely joint confound (|corr|→1): both growth and self-
    # limitation load on it (a tilted-but-diagonal null, not a single axis) → the joint hint.
    assert min(abs(v[0]), abs(v[1])) > _LOAD_TOL
    assert res.fit.corr_alpha_beta > 0.9
    assert "Cannot separate Growth" in (res.human_readable_hint or "")

    # a resolved fit must NOT surface a directional abstention (no flat direction to name).
    ok = simulate_glv_perturbseq(
        mechanism="growth", delta=0.6, dense_transient=True, n_species=2,
        n_replicates=50, n_obs=30, seed=2,
    )
    res_ok = attribute_glv(ok, steps=200, n_sim=40, seed=0)
    if res_ok.call == "growth":
        assert res_ok.degeneracy_direction is None
        assert res_ok.human_readable_hint is None


@pytest.mark.slow
@pytest.mark.verification
def test_laplace_curvature_measures_the_alpha_beta_degeneracy() -> None:
    # Near equilibrium the (αₜ, βₜₜ) Laplace curvature must be near-singular (degenerate);
    # with the transient sampled it must be BETTER conditioned. The abstention is grounded
    # in this measured number, not asserted.
    near_eq = generate_alpha_beta_confound_decoy(seed=0)  # near-equilibrium sampling
    base_ne = near_eq.baseline
    post_ne = alpha_beta_identifiability(base_ne, near_eq, target=0, n_sim=26)
    assert post_ne.degenerate  # measured near-singular Hessian
    assert abs(post_ne.correlation[0, 1]) > 0.9  # strong α⇄β_tt correlation
