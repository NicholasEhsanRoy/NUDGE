"""Self-contained HIERARCHICAL / NLME Alzheimer's Aβ QSP forward model — NO ``nudge`` dependency.

This is the *coupled-population* companion to ``ad_qsp_forward.py`` (which is the same model
with **independent** subjects). It re-expresses NUDGE's ``make_ad_nlme_cohort_predict_fn`` as
plain NumPy so a raw agent WITHOUT NUDGE can load the identical coupled model + cohort and
attempt the full joint identifiability analysis.

**Why the coupling matters (the A/B point).** In ``ad_qsp_forward.py`` every subject has its own
private 12 kinetic constants, so subject *i*'s observations depend only on subject *i*'s
parameters — the population Fisher-information matrix is **block-diagonal** and a competent
analyst decomposes it into per-subject blocks (never a big matrix). Here the subjects share
population **hyperparameters**: each subject's random-effect kinetic value is drawn around a
SHARED geometric-mean ``μ`` and the non-random kinetics are a SHARED fixed-effect vector ``φ``.
Because ``μ`` and ``φ`` enter EVERY subject's predicted observations, the joint FIM over

    θ = [ μ (d) | φ (n_fixed) | r₀ (d) | r₁ (d) | … | r_{N-1} (d) ]

(``r_i`` = subject *i*'s multiplicative random effect; its RE kinetic value is ``μ ⊙ r_i``) is
**NOT block-diagonal** — it is a bordered / **arrowhead** matrix (a dense border of shared-
hyperparameter rows/cols that couples every subject, plus per-subject blocks; cross-subject
blocks are exactly zero). **Block-summing no longer applies:** you cannot analyze subject *i*
in isolation because ``μ``/``φ`` tie all subjects together.

**Why this is the WITHOUT-arm wall.** A raw agent asked for the FULL joint identifiability
spectrum of this coupled model most naturally forms the dense Jacobian ``J = ∂(obs)/∂θ`` (here
by finite differences, since there is no autodiff) and the dense FIM ``JᵀJ`` — an
``(n_params × n_params)`` array that reaches ~O(10 GB) and OOMs at population scale. Its only
matrix-free alternative, finite-difference matvecs, costs O(n_params) forward solves PER matvec
(one perturbed re-simulation of the whole cohort per parameter) — intractable at N≈2000+. NUDGE
avoids both because the model is differentiable: it forms the FIM only through ``jvp∘vjp``
matvecs (one forward + one reverse sweep, no J), staying flat in memory.

**Honesty (do not drop):** synthetic cohort, never real patients; demo-scaled dimensionless
constants (``NUDGE-LIM-026``); the arrowhead is in-principle Schur-decomposable so the claim is
the MEASURED dense-OOM-vs-matrix-free-flat contrast, not impossibility (``NUDGE-LIM-028``).

Run ``python make_nlme_dataset.py`` to (re)generate ``cohort_nlme.npz`` from this model.
"""

from __future__ import annotations

import numpy as np
from ad_qsp_forward import (
    PARAM_NAMES,
    PARAM_VALUES,
    observe_biomarkers,
    simulate_subject,
)

#: the kinetic parameters given a population random effect (plaque-switch gain/threshold +
#: microglial clearance). The rest are shared fixed effects.
RE_PARAMS: tuple[str, ...] = ("k_pg", "K_pg", "k_gl")


def _split_indices(re_params: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    name_to_idx = {n: i for i, n in enumerate(PARAM_NAMES)}
    re_idx = np.array([name_to_idx[n] for n in re_params], dtype=int)
    fixed_idx = np.array([i for i in range(len(PARAM_NAMES)) if i not in set(re_idx.tolist())],
                         dtype=int)
    return re_idx, fixed_idx


def unpack_theta(
    theta: np.ndarray, n_subjects: int, re_params: tuple[str, ...] = RE_PARAMS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the joint vector ``θ = [μ | φ | r₀ … r_{N-1}]`` → ``(mu (d,), phi (n_fixed,),
    r (N, d))``. The first ``d + n_fixed`` entries are the SHARED border; the rest are the
    per-subject random-effect blocks."""
    re_idx, fixed_idx = _split_indices(re_params)
    d, n_fixed = len(re_idx), len(fixed_idx)
    mu = theta[:d]
    phi = theta[d:d + n_fixed]
    r = theta[d + n_fixed:].reshape(n_subjects, d)
    return mu, phi, r


def nlme_predict(
    theta: np.ndarray,
    n_subjects: int,
    obs_times: np.ndarray,
    *,
    re_params: tuple[str, ...] = RE_PARAMS,
    dose: float = 0.6,
    t_max: float = 12.0,
    dt: float = 0.08,
    biomarkers: tuple[int, ...] = (2,),
) -> np.ndarray:
    """The coupled forward map ``θ -> stacked log-biomarker observations`` (flat vector).

    ``μ`` and ``φ`` (the shared border) enter every subject, so the joint Jacobian/FIM of this
    map is a coupled arrowhead — the whole point of the WITHOUT-arm."""
    re_idx, fixed_idx = _split_indices(re_params)
    mu, phi, r = unpack_theta(theta, n_subjects, re_params)
    out = []
    for i in range(n_subjects):
        p = np.empty(len(PARAM_NAMES), dtype=np.float64)
        p[re_idx] = mu * r[i]      # μ ⊙ r_i : subject-specific random-effect kinetics
        p[fixed_idx] = phi         # shared fixed effects
        traj, times = simulate_subject(p, dose=dose, t_max=t_max, dt=dt)
        obs = observe_biomarkers(traj, times, obs_times, biomarkers)
        out.append(np.log(np.clip(obs, 0.0, 1e6) + 1e-3).reshape(-1))
    return np.concatenate(out)


def joint_theta0(
    n_subjects: int, *, re_params: tuple[str, ...] = RE_PARAMS, omega: float = 0.2, seed: int = 0
) -> np.ndarray:
    """The nominal joint vector ``[μ | φ | r₀ … r_{N-1}]`` at population truth (μ,φ = base
    kinetics) with per-subject random effects ``r_i = exp(ω·η_i)``, ``η_i ~ N(0, I)``."""
    re_idx, fixed_idx = _split_indices(re_params)
    rng = np.random.default_rng(seed)
    mu0 = PARAM_VALUES[re_idx]
    phi0 = PARAM_VALUES[fixed_idx]
    eta = rng.standard_normal((n_subjects, len(re_idx)))
    r0 = np.exp(omega * eta)
    return np.concatenate([mu0, phi0, r0.reshape(-1)])


def finite_difference_jacobian(
    theta: np.ndarray, n_subjects: int, obs_times: np.ndarray, *, eps: float = 1e-6, **kw
) -> np.ndarray:
    """The dense log-parameter Jacobian ``J = ∂(obs)/∂ log θ`` by CENTRAL FINITE DIFFERENCES.

    This is the *only* route a from-scratch NumPy model has (no autodiff): it costs ``2·n_params``
    full cohort re-simulations, and materializes an ``(n_obs × n_params)`` array. Forming ``JᵀJ``
    then gives the ``(n_params × n_params)`` dense FIM that OOMs at population scale — the wall the
    A/B comparison rests on. Provided so the WITHOUT-arm is runnable at SMALL N (do not call it at
    N≈2000+: 2·n_params cohort solves is intractable, which is exactly the point)."""
    y0 = nlme_predict(theta, n_subjects, obs_times, **kw)
    n_obs, n_params = y0.shape[0], theta.shape[0]
    jac = np.empty((n_obs, n_params), dtype=np.float64)
    for j in range(n_params):
        h = eps * max(abs(theta[j]), 1e-8)
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        yp = nlme_predict(tp, n_subjects, obs_times, **kw)
        ym = nlme_predict(tm, n_subjects, obs_times, **kw)
        jac[:, j] = (yp - ym) / (2.0 * h) * theta[j]  # ∂/∂log θ = θ·∂/∂θ
    return jac


if __name__ == "__main__":  # a tiny coupling demonstration at small N
    N = 5
    obs_times = np.linspace(0.0, 12.0, 2)
    theta = joint_theta0(N)
    re_idx, fixed_idx = _split_indices(RE_PARAMS)
    border = len(re_idx) + len(fixed_idx)
    d = len(re_idx)
    J = finite_difference_jacobian(theta, N, obs_times)
    fim = J.T @ J / 0.05**2
    b = np.arange(border)
    s0 = np.arange(border, border + d)
    s1 = np.arange(border + d, border + 2 * d)
    print("n_params:", theta.shape[0], " n_obs:", J.shape[0], " border:", border)
    print("max |FIM[border, subj0]| =", round(float(np.abs(fim[np.ix_(b, s0)]).max()), 3),
          "(>0: shared border couples every subject)")
    print("max |FIM[subj0, subj1]|  =", round(float(np.abs(fim[np.ix_(s0, s1)]).max()), 6),
          "(~0: block-diagonal in the random effects → an arrowhead, not decomposable)")
