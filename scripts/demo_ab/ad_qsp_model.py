"""Self-contained *differentiable* AD amyloid-β QSP model — NO ``nudge`` import.

This is the shared model file that BOTH arms of the "with-vs-without NUDGE" demo use: a raw
agent reads the forward function below and autodiffs / finite-differences it directly, while
NUDGE ingests this same file via ``identifiability(model_path=".../ad_qsp_model.py")`` /
``oed(model_path=".../ad_qsp_model.py")``. It mirrors the math of
:mod:`nudge.mechanisms.ad_qsp` **verbatim** (same rate-law forms, same demo-scaled constants,
same RK4 ``lax.scan`` integrator, same cohort draws), so the tools' results are numerically
**identical** to the registered ``ad_qsp`` model — verified to machine precision in
``tests/inference/test_model_loader.py`` and ``tests/mcp/test_dynamic_model_ingestion.py``.

Unlike ``ad_qsp_forward.py`` (a plain-NumPy forward model for hand-fitting), this file is
**JAX float64** so its ``predict_fn`` / ``observe`` are autodiff-differentiable — the interface
NUDGE's white-box Fisher / OED machinery needs.

**The builder interface** (a convention; the file needs no ``nudge`` import):

- ``nudge_identifiability(n_free=0, seed=0, sigma=None) -> dict`` — the population-cohort
  identifiability problem: ``{"predict_fn", "theta0", "param_names", "sigma"}`` with the
  ``n_free`` scale knob (how many subject-specific parameters are jointly estimated).
- ``nudge_oed(target="k_on", sigma=None, seed=0) -> dict`` — the plaque-only ``k_on``⇄``k_gl``
  design problem: ``{"observe", "theta0", "param_names", "phi_bounds", "sigma"}``.

**Honesty labels (do not drop).** Synthetic cohort, never real patients. Demo-scaled
dimensionless constants — the published stiff seconds-to-years Proctor et al. 2013 model
(BioModels ``BIOMD0000000488``, CC0) cannot be integrated by an explicit RK4; the reaction
topology + rate-law forms are preserved, the constants non-dimensionalized (``NUDGE-LIM-026``).
The identifiability *structure* is a property of the preserved rate-law forms. Dynamic
ingestion executes this file as Python in the caller's process (``NUDGE-LIM-030``): local,
trusted-input only.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

# NOTE on precision: this file does NOT flip ``jax_enable_x64`` globally at import — doing so
# would be an intrusive side effect that could change an unrelated caller's numerics. The FIM's
# smallest eigenvalues DO need float64; NUDGE's ``identifiability`` tool enables x64 around the
# diagnostic itself, and a raw agent analysing this file by hand should ``jax.config.update(
# "jax_enable_x64", True)`` first (as the ``__main__`` block below does).

# --------------------------------------------------------------------------- #
# the model definition (mirrors nudge.mechanisms.ad_qsp, verbatim rate-law forms)
# --------------------------------------------------------------------------- #
SPECIES: tuple[str, ...] = (
    "Abeta_monomer",     # M — soluble Aβ monomer
    "Abeta_oligomer",    # O — soluble oligomer / dimer
    "Abeta_plaque",      # P — aggregated plaque (≈ amyloid-PET)
    "antibody",          # A — therapeutic anti-Aβ mAb
    "antibody_complex",  # C — antibody-bound Aβ, cleared
    "microglia",         # G — activated microglia clearance capacity
)

#: the 12 kinetic parameters (order = the free-parameter vector).
PARAM_NAMES: tuple[str, ...] = (
    "s_M", "d_M", "k_agg", "k_dis", "k_pf", "k_pg",
    "K_pg", "k_dp", "k_on", "d_A", "k_gl", "k_ga",
)

#: demo-scaled nominal (population-truth) values (dimensionless; NUDGE-LIM-026).
PARAM_VALUES: np.ndarray = np.array(
    [0.05, 0.05, 0.6, 0.12, 0.5, 1.3, 1.0, 0.06, 0.9, 0.5, 0.2, 0.35], dtype=np.float64
)

#: default seeded initial state (a physiological monomer pool; everything else 0).
X0: np.ndarray = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

#: antibody dosing window (infusion on ``[t_on, t_off)``) — the transient that separates the
#: antibody-binding ⇄ microglial-clearance confound.
DOSE_WINDOW: tuple[float, float] = (2.0, 8.0)

#: biomarker readouts (indices into SPECIES): plaque (≈ amyloid-PET), soluble oligomer (≈ CSF).
BIOMARKERS: tuple[int, ...] = (2, 1)

_X_CAP = 1e6
#: log-offset for the log-abundance observation transform (abundances ≥ 0, ≈ 0 in OFF state).
_LOG_OFFSET = 1e-3


def ad_field(x: Array, p: Array, u: Array) -> Array:
    """The differentiable Aβ-cascade vector field ``dx/dt = f(x, p, u)`` (RAW params ``p``).

    The plaque-growth term ``k_pg·O·P²/(K_pg²+P²)`` is the published Proctor ``AbetaPlaqueGrowth``
    autocatalytic Hill switch (gain ``k_pg`` / threshold ``K_pg``); the rest is the mass-action
    aggregation cascade + antibody binding/clearance + microglial clearance. Identical algebra to
    ``nudge.mechanisms.ad_qsp.ad_field``. ``u`` is the scalar antibody-infusion input.
    """
    M, Ol, Pl, A, C, G = x
    s_M, d_M, k_agg, k_dis, k_pf, k_pg, K_pg, k_dp, k_on, d_A, k_gl, k_ga = (
        p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11]
    )
    r_agg = 0.5 * k_agg * M * M
    r_dis = k_dis * Ol
    r_pf = 0.5 * k_pf * Ol * Ol
    r_pg = k_pg * Ol * Pl * Pl / (K_pg * K_pg + Pl * Pl)  # Proctor AbetaPlaqueGrowth (Hill)
    r_dp = k_dp * Pl
    r_clr = k_gl * G * Pl
    dM = s_M - d_M * M - 2.0 * r_agg + 2.0 * r_dis - k_on * A * M
    dO = r_agg - r_dis - 2.0 * r_pf - k_on * A * Ol + r_dp
    dPl = r_pf + r_pg - r_dp - r_clr
    dA = u - d_A * A
    dC = k_on * A * (M + Ol) - 0.5 * d_A * C
    dG = k_ga * Pl * (1.0 - G) - 0.2 * G
    return jnp.stack([dM, dO, dPl, dA, dC, dG])


def _dose_grid(grid_t: Array, dose: float, window: tuple[float, float]) -> Array:
    t_on, t_off = window
    return jnp.where((grid_t >= t_on) & (grid_t < t_off), dose, 0.0)


def _rk4_full(p: Array, x0: Array, grid_t: Array, u_grid: Array, dt: float) -> Array:
    """Integrate ``dx/dt = ad_field(x, p, u)`` on the fine grid → full trajectory ``(G+1, 6)``.

    A plain RK4 ``lax.scan`` (differentiable w.r.t. ``p``); index 0 = ``t0`` so the trajectory can
    be interpolated at continuous observation times (the OED design ``φ``). ``grid_t`` is carried
    for interface parity; the scan advances over ``u_grid``.
    """
    def step(x: Array, u: Array) -> tuple[Array, Array]:
        k1 = ad_field(x, p, u)
        k2 = ad_field(x + 0.5 * dt * k1, p, u)
        k3 = ad_field(x + 0.5 * dt * k2, p, u)
        k4 = ad_field(x + dt * k3, p, u)
        x_next = jnp.clip(x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0, _X_CAP)
        return x_next, x_next

    _final, traj = jax.lax.scan(step, x0, u_grid)
    return jnp.concatenate([x0[None, :], traj], axis=0)


def _observe_at(traj: Array, grid_t: Array, phi: Array, biom: Array | np.ndarray) -> Array:
    """Interpolate a fine trajectory at continuous times ``phi`` and log the biomarkers."""
    cols = [jnp.interp(phi, grid_t, traj[:, int(s)]) for s in np.asarray(biom)]
    obs = jnp.stack(cols, axis=1)  # (m, n_biom)
    return jnp.log(jnp.clip(obs, 0.0, _X_CAP) + _LOG_OFFSET).reshape(-1)


# --------------------------------------------------------------------------- #
# a plain forward function a raw agent can autodiff / finite-difference
# --------------------------------------------------------------------------- #
def forward(
    p: Array | np.ndarray,
    obs_times: np.ndarray,
    *,
    dose: float = 0.6,
    t_max: float = 12.0,
    dt: float = 0.06,
    biomarkers: tuple[int, ...] = BIOMARKERS,
) -> Array:
    """Forward-simulate one subject and return the log-biomarker observations at ``obs_times``.

    A clear, self-contained differentiable map ``p -> log-biomarkers`` (JAX float64). A raw agent
    can ``jax.jacobian(forward)`` it (or finite-difference it) to build the sensitivity matrix by
    hand; NUDGE builds the same Fisher information from the ``nudge_identifiability`` /
    ``nudge_oed`` builders below. ``p`` is the 12 RAW kinetic parameters (:data:`PARAM_NAMES`).
    """
    p_j = jnp.asarray(np.asarray(p, dtype=np.float64))
    n = int(round(t_max / dt))
    grid_t = jnp.asarray(np.arange(n) * dt)
    u_grid = _dose_grid(grid_t, dose, DOSE_WINDOW)
    traj = _rk4_full(p_j, jnp.asarray(X0), grid_t, u_grid, dt)
    grid_full = jnp.asarray(np.concatenate([[0.0], np.arange(n) * dt + dt]))
    return _observe_at(traj, grid_full, jnp.asarray(np.asarray(obs_times, dtype=np.float64)),
                       np.asarray(biomarkers))


# --------------------------------------------------------------------------- #
# builder 1 — population-cohort identifiability (mirrors make_ad_cohort_predict_fn)
# --------------------------------------------------------------------------- #
def _cohort_params(n_subjects: int, seed: int, isv: float) -> np.ndarray:
    """Subject-specific parameters: population truth × log-normal inter-subject variability."""
    rng = np.random.default_rng(seed)
    base = PARAM_VALUES[None, :]
    log_shift = isv * rng.standard_normal((n_subjects, PARAM_VALUES.shape[0]))
    return base * np.exp(log_shift)  # (n_subjects, n_params) raw, positive


def _cohort_predict_fn(
    *,
    n_subjects: int,
    n_free: int,
    n_obs_times: int = 8,
    t_max: float = 12.0,
    dt: float = 0.06,
    dose: float = 0.6,
    isv: float = 0.15,
    biomarkers: tuple[int, ...] = BIOMARKERS,
    seed: int = 0,
) -> tuple[Any, np.ndarray, tuple[str, ...]]:
    """Build ``(predict_fn, theta0, param_names)`` for the population calibration.

    Verbatim numerical mirror of ``nudge.mechanisms.ad_qsp.make_ad_cohort_predict_fn``: each
    subject carries its own copy of the 12 kinetics (population truth × ``isv`` log-normal
    variability); the first ``n_free`` of the stacked ``n_subjects × 12`` vector are free. The
    forward map ``vmap``\\s the single-subject RK4 solve and returns log-biomarker observations.
    """
    n_params = PARAM_VALUES.shape[0]
    p_full = n_subjects * n_params
    n_free = int(np.clip(n_free, 1, p_full))

    subj_p = _cohort_params(n_subjects, seed, isv)  # (n_subjects, n_params)
    full = subj_p.reshape(-1)  # (n_subjects*n_params,)
    free_idx = np.arange(n_free)

    n = int(round(t_max / dt))
    grid_t = np.arange(n) * dt
    obs_t = np.linspace(0.0, t_max, n_obs_times)
    obs_idx = np.clip(np.round(obs_t / dt).astype(int), 0, n)  # index into (G+1) trajectory
    u_grid = np.asarray(_dose_grid(jnp.asarray(grid_t), dose, DOSE_WINDOW))
    x0 = np.tile(X0, (n_subjects, 1))

    full_j = jnp.asarray(full)
    free_idx_j = jnp.asarray(free_idx)
    x0_j = jnp.asarray(x0)
    grid_j = jnp.asarray(grid_t)
    u_j = jnp.asarray(u_grid)
    obs_idx_j = jnp.asarray(obs_idx)
    biom_j = jnp.asarray(np.asarray(biomarkers))

    def one_subject(p_row: Array, x0_row: Array) -> Array:
        traj = _rk4_full(p_row, x0_row, grid_j, u_j, dt)  # (G+1, 6)
        obs = traj[obs_idx_j][:, biom_j]  # (n_obs_times, n_biom)
        return jnp.log(jnp.clip(obs, 0.0, _X_CAP) + _LOG_OFFSET).reshape(-1)

    def predict(theta: Array) -> Array:
        full_p = full_j.at[free_idx_j].set(theta.astype(full_j.dtype))
        P = full_p.reshape(n_subjects, n_params)
        obs = jax.vmap(one_subject)(P, x0_j)  # (n_subjects, n_obs_times*n_biom)
        return obs.reshape(-1)

    theta0 = full[free_idx].astype(np.float64)
    param_names = tuple(
        f"{PARAM_NAMES[i % n_params]}[subj{i // n_params}]" for i in free_idx
    )
    return predict, theta0, param_names


def nudge_identifiability(
    n_free: int = 0, seed: int = 0, sigma: float | None = None, **_: Any
) -> dict[str, Any]:
    """The population-cohort identifiability problem (loader interface).

    Mirrors ``nudge.inference.model_registry._ad_qsp_ident``: ``n_free`` (default 24) is the
    population scale knob; ``n_subjects = max(20, ceil(n_free/12))``. Returns the builder dict the
    dynamic loader wraps into an ``IdentifiabilityProblem`` — analysed by the SAME matrix-free
    Fisher diagnostic as ``model="ad_qsp"``, to machine precision.
    """
    per = PARAM_VALUES.shape[0]
    nf = int(n_free) if n_free and n_free > 0 else 24
    n_subjects = max(20, -(-nf // per))  # ceil(nf/per), ≥20
    predict, theta0, names = _cohort_predict_fn(n_subjects=n_subjects, n_free=nf, seed=seed)
    return {
        "predict_fn": predict,
        "theta0": theta0,
        "param_names": names,
        "sigma": 0.05 if sigma is None else float(sigma),
        "domain": "clinical pharmacology (Alzheimer's Aβ)",
        "meta": {"model": "ad_qsp (standalone)", "n_subjects": n_subjects, "n_free": nf,
                 "note": "synthetic cohort, demo-scaled (NUDGE-LIM-026); dynamic ingestion "
                         "NUDGE-LIM-030"},
    }


# --------------------------------------------------------------------------- #
# builder 2 — plaque-only k_on ⇄ k_gl design problem (mirrors make_ad_oed_problem)
# --------------------------------------------------------------------------- #
def nudge_oed(
    target: str | None = "k_on", sigma: float | None = None, seed: int = 0, **_: Any
) -> dict[str, Any]:
    """The plaque-only ``k_on``⇄``k_gl`` OED design problem (loader interface).

    Mirrors ``nudge.mechanisms.ad_qsp.make_ad_oed_problem`` (the confounded antibody-binding ⇄
    microglial-clearance pair, plaque-only ≈ amyloid-PET readout). ``target`` selects which of the
    pair to resolve downstream (default ``k_on``); the returned ``observe`` covers both. Returns
    the builder dict the dynamic loader wraps into a ``DesignProblem`` — designed by the SAME
    gradient OED as ``model="ad_qsp"``, reproducing the ×259 CRLB lift to machine precision.
    """
    pair = ("k_on", "k_gl")
    t_max, dt, dose = 12.0, 0.03, 0.6
    t_min = 0.05
    sig = 0.05 if sigma is None else float(sigma)
    biomarkers = (2,)  # plaque only (≈ amyloid-PET): the realistic single readout

    name_idx = {n: i for i, n in enumerate(PARAM_NAMES)}
    i0, i1 = name_idx[pair[0]], name_idx[pair[1]]
    base = PARAM_VALUES.copy()
    n = int(round(t_max / dt))
    grid_np = (np.arange(n + 1) * dt).astype(np.float64)
    u_np = np.asarray(_dose_grid(jnp.asarray(grid_np[:-1]), dose, DOSE_WINDOW))
    x0_np = X0.copy()
    biom_np = np.asarray(biomarkers)

    def observe(theta: Array, phi: Array) -> Array:
        p = jnp.asarray(base, theta.dtype)
        p = p.at[i0].set(jnp.exp(theta[0])).at[i1].set(jnp.exp(theta[1]))
        grid_t = jnp.asarray(grid_np, theta.dtype)
        u_grid = jnp.asarray(u_np, theta.dtype)
        x0 = jnp.asarray(x0_np, theta.dtype)
        traj = _rk4_full(p, x0, grid_t[:-1], u_grid, dt)
        return _observe_at(traj, grid_t, phi, biom_np)

    theta0 = np.array([np.log(base[i0]), np.log(base[i1])], dtype=np.float64)
    return {
        "observe": observe,
        "theta0": theta0,
        "param_names": (f"log_{pair[0]}", f"log_{pair[1]}"),
        "phi_bounds": (t_min, t_max),
        "sigma": sig,
        "domain": "clinical pharmacology (Alzheimer's Aβ)",
        "meta": {"model": "ad_qsp (standalone)", "pair": pair, "target": f"log_{target}",
                 "dose": dose, "dose_window": DOSE_WINDOW, "biomarkers": biomarkers,
                 "note": "synthetic, demo-scaled (NUDGE-LIM-026); dynamic ingestion NUDGE-LIM-030"},
    }


if __name__ == "__main__":  # a tiny sanity forward-sim + the two builder shapes
    jax.config.update("jax_enable_x64", True)  # analysing this model by hand wants float64
    obs_t = np.linspace(0.0, 12.0, 8)
    y0 = np.asarray(forward(PARAM_VALUES, obs_t, dose=0.0))
    y1 = np.asarray(forward(PARAM_VALUES, obs_t, dose=0.6))
    print("log-plaque(end), no antibody :", round(float(y0[-2]), 4))
    print("log-plaque(end), with antibody:", round(float(y1[-2]), 4), "(antibody lowers plaque)")
    ident = nudge_identifiability()
    print("identifiability: n_params =", ident["theta0"].shape[0])
    design = nudge_oed()
    print("oed: params =", design["param_names"], "phi_bounds =", design["phi_bounds"])
