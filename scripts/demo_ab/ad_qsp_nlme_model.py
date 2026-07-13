"""Self-contained *differentiable* AD amyloid-β **NLME / hierarchical** model — NO ``nudge`` import.

The coupled (arrowhead) population sibling of ``ad_qsp_model.py``: subjects share population
hyperparameters (geometric-mean ``μ`` for the random-effect kinetics + a fixed-effect vector
``φ``), so the joint Fisher-information matrix over ``θ = [ μ | φ | r₀ … r_{N-1} ]`` is a genuine
**arrowhead** — a dense border (``μ``/``φ`` couple every subject) plus per-subject blocks;
cross-subject blocks are exactly zero. Block-summing no longer applies.

This is the SAME coupled model NUDGE's registered ``ad_qsp_nlme`` analyses, re-expressed as a
standalone **JAX float64** builder (no ``nudge`` import) so it can be ingested via
``identifiability(model_path=".../ad_qsp_nlme_model.py")`` and reproduce the registry result to
machine precision (verified in the dynamic-ingestion tests). It mirrors
``nudge.mechanisms.ad_qsp.make_ad_nlme_cohort_predict_fn`` verbatim.

**The builder interface** (a convention; the file needs no ``nudge`` import):
``nudge_identifiability(n_free=0, seed=0, sigma=None) -> {"predict_fn", "theta0", "param_names",
"sigma"}`` — the coupled NLME population, ``n_free`` growing the joint parameter count (mostly
per-subject random effects) at fixed integrated state.

**Honesty labels (do not drop).** Synthetic hierarchical cohort, never real patients. Demo-scaled
dimensionless constants (``NUDGE-LIM-026``); the arrowhead is in-principle Schur-decomposable so
the coupled-scale claim is the MEASURED dense-OOM-vs-matrix-free-flat contrast, not an
impossibility (``NUDGE-LIM-028``). With a sparse plaque-only budget the population is
rank-deficient by shape → NUDGE certifies ``unidentifiable`` (the ``NUDGE-LIM-023`` fail-safe).
Dynamic ingestion executes this file as Python in the caller's process (``NUDGE-LIM-030``):
local, trusted-input only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

# reuse the verbatim vector field / integrator / constants from the single-subject model file.
from ad_qsp_model import PARAM_NAMES, PARAM_VALUES, X0, _dose_grid, _rk4_full
from jax import Array

# NOTE on precision: like ``ad_qsp_model.py``, this file does NOT flip ``jax_enable_x64`` globally
# at import (that would perturb an unrelated caller's numerics). NUDGE's ``identifiability`` tool
# enables x64 around the diagnostic; a raw agent enables it first (see the ``__main__`` block).

_X_CAP = 1e6
_LOG_OFFSET = 1e-3
DOSE_WINDOW: tuple[float, float] = (2.0, 8.0)


def _omega_vec(omega: float | Sequence[float], d: int) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(omega, dtype=np.float64))
    if arr.shape[0] == 1:
        arr = np.repeat(arr, d)
    if arr.shape[0] != d:
        raise ValueError(f"omega must be a scalar or length-{d}; got shape {arr.shape}")
    return arr


def _nlme_predict_fn(
    *,
    n_subjects: int,
    re_params: Sequence[str] = ("k_pg", "K_pg", "k_gl"),
    n_obs_times: int = 2,
    t_max: float = 12.0,
    dt: float = 0.08,
    dose: float = 0.6,
    omega: float | Sequence[float] = 0.2,
    include_prior: bool = False,
    biomarkers: tuple[int, ...] = (2,),
    seed: int = 0,
) -> tuple[Any, np.ndarray, tuple[str, ...], int]:
    """Build ``(predict_fn, theta0, param_names, border_size)`` for the coupled NLME calibration.

    Verbatim numerical mirror of ``nudge.mechanisms.ad_qsp.make_ad_nlme_cohort_predict_fn``: each
    subject draws a multiplicative random effect ``r_i = exp(ω ⊙ η_i)`` around the shared
    geometric-mean ``μ`` (= population truth) for the ``re_params`` kinetics; the rest are shared
    fixed effects ``φ``. The joint vector is ``θ = [ μ | φ | r₀ … r_{N-1} ]`` (``include_prior``
    False → no ``ω`` border).
    """
    n_params = PARAM_VALUES.shape[0]
    name_to_idx = {n: i for i, n in enumerate(PARAM_NAMES)}
    re_params_t = tuple(re_params)
    missing = [n for n in re_params_t if n not in name_to_idx]
    if missing:
        raise ValueError(f"unknown re_params {missing}; available: {PARAM_NAMES}")
    re_idx = np.array([name_to_idx[n] for n in re_params_t], dtype=int)
    fixed_idx = np.array([i for i in range(n_params) if i not in set(re_idx.tolist())], dtype=int)
    d = int(re_idx.shape[0])
    n_fixed = int(fixed_idx.shape[0])
    n_subjects = int(max(1, n_subjects))
    omega_vec = _omega_vec(omega, d)

    mu0 = PARAM_VALUES[re_idx].astype(np.float64)       # (d,) population geometric means
    phi0 = PARAM_VALUES[fixed_idx].astype(np.float64)   # (n_fixed,) shared fixed effects
    rng = np.random.default_rng(seed)
    eta = rng.standard_normal((n_subjects, d))          # (N, d) subject random effects
    r0 = np.exp(omega_vec[None, :] * eta)               # (N, d) multiplicative REs (≈1)

    border_size = d + (d if include_prior else 0) + n_fixed
    theta0_parts = [mu0]
    if include_prior:
        theta0_parts.append(omega_vec.copy())
    theta0_parts.append(phi0)
    theta0_parts.append(r0.reshape(-1))
    theta0 = np.concatenate(theta0_parts).astype(np.float64)

    n = int(round(t_max / dt))
    grid_t = np.arange(n) * dt
    obs_t = np.linspace(0.0, t_max, n_obs_times)
    obs_idx = np.clip(np.round(obs_t / dt).astype(int), 0, n)
    u_grid = np.asarray(_dose_grid(jnp.asarray(grid_t), dose, DOSE_WINDOW))
    x0 = np.tile(X0, (n_subjects, 1))

    re_idx_j = jnp.asarray(re_idx)
    fixed_idx_j = jnp.asarray(fixed_idx)
    x0_j = jnp.asarray(x0)
    grid_j = jnp.asarray(grid_t)
    u_j = jnp.asarray(u_grid)
    obs_idx_j = jnp.asarray(obs_idx)
    biom_j = jnp.asarray(np.asarray(biomarkers))

    def one_subject(p_row: Array, x0_row: Array) -> Array:
        traj = _rk4_full(p_row, x0_row, grid_j, u_j, dt)  # (G+1, 6)
        obs = traj[obs_idx_j][:, biom_j]  # (n_obs_times, n_biom)
        return jnp.log(jnp.clip(obs, 0.0, _X_CAP) + _LOG_OFFSET).reshape(-1)

    om_end = d + (d if include_prior else 0)  # end of the [μ | ω?] prefix
    phi_end = om_end + n_fixed                 # end of the [μ | ω? | φ] border

    def predict(theta: Array) -> Array:
        theta = jnp.asarray(theta)
        mu = theta[:d]
        phi = theta[om_end:phi_end]
        r = theta[phi_end:].reshape(n_subjects, d)
        re_vals = mu[None, :] * r  # (N, d) subject-specific RE kinetics
        full = jnp.zeros((n_subjects, n_params), dtype=theta.dtype)
        full = full.at[:, re_idx_j].set(re_vals.astype(theta.dtype))
        full = full.at[:, fixed_idx_j].set(
            jnp.broadcast_to(phi.astype(theta.dtype), (n_subjects, n_fixed))
        )
        data = jax.vmap(one_subject)(full, x0_j).reshape(-1)
        if include_prior:
            om = theta[d:om_end]
            prior = (jnp.log(jnp.clip(r, 1e-12, None)) / om[None, :]).reshape(-1)
            return jnp.concatenate([data, prior])
        return data

    names: list[str] = [f"mu[{p}]" for p in re_params_t]
    if include_prior:
        names += [f"omega[{p}]" for p in re_params_t]
    names += [f"phi[{PARAM_NAMES[i]}]" for i in fixed_idx]
    names += [
        f"r[{re_params_t[j]}][subj{i}]" for i in range(n_subjects) for j in range(d)
    ]
    return predict, theta0, tuple(names), int(border_size)


def nudge_identifiability(
    n_free: int = 0, seed: int = 0, sigma: float | None = None, **_: Any
) -> dict[str, Any]:
    """The coupled NLME population identifiability problem (loader interface).

    Mirrors ``nudge.inference.model_registry._ad_qsp_nlme_ident``: ``re_params =
    (k_pg, K_pg, k_gl)`` (``d=3``), a ``border`` of ``μ (3) + φ (9) = 12`` shared entries, and
    ``n_free`` (default 72) growing the joint count; ``n_subjects = max(20, ceil((n_free-12)/3))``.
    Analysed by the SAME matrix-free Fisher diagnostic as ``model="ad_qsp_nlme"``, to machine
    precision (a sparse plaque-only budget → certified ``unidentifiable``).
    """
    re_params = ("k_pg", "K_pg", "k_gl")
    d = len(re_params)
    border = len(PARAM_NAMES)  # μ (d) + φ (n_fixed = 12 − d) — include_prior=False
    nf = int(n_free) if n_free and n_free > 0 else 72
    n_subjects = max(20, -(-(nf - border) // d))  # ceil((nf-border)/d), ≥20
    predict, theta0, names, border_size = _nlme_predict_fn(
        n_subjects=n_subjects, re_params=re_params, n_obs_times=2, include_prior=False,
        biomarkers=(2,), seed=seed,
    )
    return {
        "predict_fn": predict,
        "theta0": theta0,
        "param_names": names,
        "sigma": 0.05 if sigma is None else float(sigma),
        "domain": "clinical pharmacology (Alzheimer's Aβ, NLME)",
        "meta": {"model": "ad_qsp_nlme (standalone)", "n_subjects": n_subjects,
                 "border_size": border_size, "n_re": d,
                 "coupling": "arrowhead (shared μ/φ border + per-subject r_i blocks)",
                 "note": "synthetic hierarchical cohort, demo-scaled (NUDGE-LIM-026/028); "
                         "dynamic ingestion NUDGE-LIM-030"},
    }


if __name__ == "__main__":  # a tiny sanity check of the builder shape
    jax.config.update("jax_enable_x64", True)  # analysing this model by hand wants float64
    ident = nudge_identifiability()
    print("nlme identifiability: n_params =", ident["theta0"].shape[0],
          "| first names:", list(ident["param_names"][:4]))
