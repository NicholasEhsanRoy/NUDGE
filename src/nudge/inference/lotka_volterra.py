"""Temporal / trajectory-fit attribution for generalized Lotka–Volterra (gLV) systems.

**The extensibility thesis, made concrete.** Everything else in NUDGE observes a
**steady-state snapshot** — a distribution over per-cell equilibria — and attributes a
switch's ``K`` / ``n`` / ``v_max``. A gLV microbial community is a different dynamical
system whose parameter information lives in **transients**, not an equilibrium, so this
module re-instantiates a *trajectory-matching* fit loop and a new attribution vocabulary
— **growth ``α`` / interaction ``β`` / antibiotic-susceptibility ``ε``** — while reusing
NUDGE's mechanism-agnostic scaffolding verbatim: the distributional
:func:`~nudge.inference.losses.energy_distance` (the fit driver), the BIC restricted-fit
parsimony pattern (:mod:`nudge.inference.model_select`,
:mod:`nudge.inference.dose_response`), and the Laplace/Fisher identifiability guard
(:func:`~nudge.inference.uncertainty.laplace_posterior`). It touches **neither
``fit.py`` nor ``core/circuit.py``** — the design constraint that keeps the frozen core
frozen (see ``design/EXTENSIBILITY_SPIKE.md`` / ``design/MICROBIOME_DATA_GATE.md``).

## The model

Each community of ``S`` taxa evolves under

```
dxᵢ/dt = xᵢ · (αᵢ + Σⱼ βᵢⱼ xⱼ + εᵢ · u(t))
```

- ``αᵢ`` — intrinsic **growth** rate of taxon ``i``;
- ``βᵢⱼ`` — **interaction** of taxon ``j`` on ``i`` (``βᵢᵢ < 0`` = self-limitation /
  carrying capacity ``Kᵢ = −αᵢ/βᵢᵢ``);
- ``εᵢ`` — **susceptibility** of taxon ``i`` to a known external input ``u(t)`` (an
  antibiotic pulse); ``εᵢ·u`` acts **only while the drug is on**.

## What it attributes (and where it abstains — the load-bearing honesty)

Given a *reference* community and a *perturbed* one (the SAME external ``u(t)``, one knob
of a target taxon moved), it BIC-selects **which single knob** moved — ``growth`` /
``interaction`` / ``susceptibility`` — or **abstains**. gLV inference is famously
ill-posed, so abstaining is on-thesis and expected:

- **``ε`` (susceptibility) is the most identifiable** axis — the drug window is an on/off
  contrast, so a direct-kill signature is time-localized and distinct from a constant
  growth/interaction shift. This is where a demoable *positive* appears.
- **``α`` ⇄ ``βᵢᵢ`` (growth vs self-limitation) is degenerate near equilibrium** —
  ``Kᵢ = −αᵢ/βᵢᵢ`` means a growth change and a carrying-capacity change produce the same
  steady state, separable only by the **transient**. When the sampling does not resolve
  the transient, the two restricted fits tie *and* the Laplace curvature on
  ``(αₜ, βₜₜ)`` is near-singular — so NUDGE returns ``unresolved`` with the degeneracy
  **measured, not asserted** (``NUDGE-LIM-020``).

Fail-safety is paramount: a synthetic case with a known answer must **recover it or
abstain — never a confident wrong knob**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from nudge.inference.losses import energy_distance
from nudge.inference.uncertainty import LaplacePosterior, laplace_posterior

__all__ = [
    "GLVParams",
    "GLVDataset",
    "GLVFit",
    "GLVResult",
    "DegeneracyDirection",
    "glv_vector_field",
    "simulate_glv",
    "simulate_glv_perturbseq",
    "generate_alpha_beta_confound_decoy",
    "generate_no_perturbation_null",
    "fit_baseline_glv",
    "fit_glv_attribution",
    "classify_glv",
    "attribute_glv",
    "alpha_beta_identifiability",
    "degeneracy_direction_from_posterior",
]

#: The three attributable knobs, in a fixed order, and the verdict each names.
_KNOBS: tuple[str, ...] = ("growth", "interaction", "susceptibility")
_CALL_OF: dict[str, str] = {
    "growth": "growth",
    "interaction": "interaction",
    "susceptibility": "susceptibility",
}
#: log-offset for the log-abundance transform (abundances are ≥ 0, often ≈ 0 in the OFF
#: state); ``log(x + _LOG_OFFSET)`` keeps the transform finite and shape-sensitive.
_LOG_OFFSET = 1e-2
#: Hard cap on abundance inside the integrator — a diverging gLV orbit must not NaN the
#: loss (it is simply a bad fit); clip ≥ 0 because abundances cannot be negative.
_X_CAP = 1e6


# --------------------------------------------------------------------------- #
# parameter container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GLVParams:
    """A gLV community's kinetics: growth ``alpha`` ``(S,)``, interaction ``beta``
    ``(S, S)``, susceptibility ``eps`` ``(S,)``. Stored as numpy; :meth:`jax` packs
    them into device arrays for the integrator."""

    alpha: np.ndarray
    beta: np.ndarray
    eps: np.ndarray

    @property
    def n_species(self) -> int:
        return int(self.alpha.shape[0])

    def jax(self) -> tuple[Array, Array, Array]:
        return (
            jnp.asarray(self.alpha, dtype=jnp.float32),
            jnp.asarray(self.beta, dtype=jnp.float32),
            jnp.asarray(self.eps, dtype=jnp.float32),
        )

    def with_knob(self, knob: str, target: int, delta: float) -> GLVParams:
        """Return a copy with one knob of ``target`` moved by an **additive** ``delta``.

        ``growth`` shifts ``alpha[target]``; ``susceptibility`` shifts ``eps[target]``;
        ``interaction`` shifts the **self-interaction** ``beta[target, target]`` — the
        carrying-capacity axis that is degenerate with growth near equilibrium
        (``Kᵢ = −αᵢ/βᵢᵢ``; ``NUDGE-LIM-020``).
        """
        alpha, beta, eps = self.alpha.copy(), self.beta.copy(), self.eps.copy()
        if knob == "growth":
            alpha[target] += delta
        elif knob == "susceptibility":
            eps[target] += delta
        elif knob == "interaction":
            beta[target, target] += delta
        else:
            raise ValueError(f"unknown knob {knob!r}")
        return GLVParams(alpha=alpha, beta=beta, eps=eps)


# --------------------------------------------------------------------------- #
# the differentiable gLV vector field + integrator (self-contained, no diffrax)
# --------------------------------------------------------------------------- #
def glv_vector_field(
    x: Array, alpha: Array, beta: Array, eps: Array, u: Array
) -> Array:
    """The gLV RHS ``dx/dt = x · (α + β·x + ε·u)`` (elementwise in ``x``)."""
    growth = alpha + beta @ x + eps * u
    return x * growth


def _rk4_step(
    x: Array, alpha: Array, beta: Array, eps: Array, u: Array, dt: float
) -> Array:
    k1 = glv_vector_field(x, alpha, beta, eps, u)
    k2 = glv_vector_field(x + 0.5 * dt * k1, alpha, beta, eps, u)
    k3 = glv_vector_field(x + 0.5 * dt * k2, alpha, beta, eps, u)
    k4 = glv_vector_field(x + dt * k3, alpha, beta, eps, u)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return jnp.clip(x_next, 0.0, _X_CAP)


def simulate_glv(
    params: GLVParams | tuple[Array, Array, Array],
    x0: Array,
    u_grid: Array,
    dt: float,
    obs_idx: Array,
) -> Array:
    """Integrate one community on a fine grid (RK4) and gather the observation times.

    ``u_grid`` is ``u(t)`` sampled on the fine grid (length ``G``); ``obs_idx`` indexes
    the fine grid at the observation times. Returns the trajectory ``(T, S)`` at those
    times. Differentiable w.r.t. the kinetics (a plain ``lax.scan`` of RK4 steps) — the
    gradient the trajectory fit needs, with **no ``diffrax`` dependency**.
    """
    if isinstance(params, GLVParams):
        alpha, beta, eps = params.jax()
    else:
        alpha, beta, eps = params

    def body(x: Array, u: Array) -> tuple[Array, Array]:
        x_next = _rk4_step(x, alpha, beta, eps, u, dt)
        return x_next, x_next

    _final, traj = jax.lax.scan(body, x0, u_grid)
    # traj[g] = state AFTER stepping with u_grid[g]; prepend x0 so index 0 = t0.
    traj = jnp.concatenate([x0[None, :], traj], axis=0)
    return traj[obs_idx]


def _fine_grid(
    t_max: float, dt: float, t_obs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build the fine integration grid and the nearest-grid indices for ``t_obs``."""
    n_steps = int(round(t_max / dt))
    grid_t = np.arange(n_steps + 1) * dt
    obs_idx = np.clip(np.round(t_obs / dt).astype(int), 0, n_steps)
    return grid_t, obs_idx


def _pulse(grid_t: np.ndarray, t_on: float, t_off: float) -> np.ndarray:
    """A unit external input ``u(t) = 1`` on ``[t_on, t_off)`` (the antibiotic window)."""
    return ((grid_t >= t_on) & (grid_t < t_off)).astype(np.float32)


# --------------------------------------------------------------------------- #
# dataset container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GLVDataset:
    """A reference vs perturbed pair of replicate trajectory ensembles + the truth.

    ``reference`` / ``perturbed`` are ``(R, T, S)`` replicate × time × species
    abundance arrays under the SAME external input ``u_grid`` (on the fine grid).
    ``t_obs`` are the observation times; ``obs_idx`` indexes ``u_grid``'s fine grid.
    ``baseline`` is the reference community's true kinetics; ``ground_truth`` records the
    perturbation (``mechanism`` ∈ knobs ∪ ``{"none"}``, ``target``, ``delta``) and the
    sampling regime, so a test can assert recover-or-abstain.
    """

    reference: np.ndarray
    perturbed: np.ndarray
    t_obs: np.ndarray
    u_grid: np.ndarray
    obs_idx: np.ndarray
    dt: float
    baseline: GLVParams
    ground_truth: dict[str, Any]

    @property
    def n_species(self) -> int:
        return int(self.reference.shape[2])


# --------------------------------------------------------------------------- #
# synthetic generators (synthetic-first — nothing real until the round-trip passes)
# --------------------------------------------------------------------------- #
def _default_baseline(n_species: int, rng: np.random.Generator) -> GLVParams:
    """A stable, diagonally-dominant gLV community (strong self-limitation, weak cross).

    Diagonal dominance keeps orbits bounded (they saturate at a carrying capacity), so
    the round-trip is not dominated by integrator blow-up — the honest, controllable
    regime the synthetic gate needs.
    """
    alpha = rng.uniform(0.6, 1.0, size=n_species)
    beta = np.zeros((n_species, n_species))
    for i in range(n_species):
        beta[i, i] = -rng.uniform(0.8, 1.2)  # strong self-limitation
        for j in range(n_species):
            if i != j:
                beta[i, j] = rng.uniform(-0.15, 0.15)  # weak coupling
    eps = np.zeros(n_species)  # baseline: no drug susceptibility unless a knob sets it
    return GLVParams(alpha=alpha, beta=beta, eps=eps)


def simulate_glv_perturbseq(
    *,
    n_species: int = 3,
    n_replicates: int = 60,
    mechanism: str = "susceptibility",
    target: int = 0,
    delta: float | None = None,
    t_max: float = 12.0,
    n_obs: int = 25,
    dt: float = 0.02,
    pulse_window: tuple[float, float] = (4.0, 6.0),
    param_noise: float = 0.04,
    obs_noise: float = 0.05,
    baseline: GLVParams | None = None,
    dense_transient: bool = True,
    seed: int = 0,
) -> GLVDataset:
    """Simulate a reference vs perturbed pair with a KNOWN single-knob perturbation.

    ``mechanism`` ∈ ``{"growth", "interaction", "susceptibility", "none"}`` names which
    knob of taxon ``target`` is moved by ``delta`` (a sensible default per knob if
    ``None``). Both groups feel the SAME antibiotic pulse ``u(t) = 1`` on
    ``pulse_window``; for a ``susceptibility`` perturbation the reference is *insusceptible*
    (``ε = 0``) and only the perturbed community is hit — the on/off contrast that makes
    ``ε`` the identifiable axis. ``dense_transient`` samples densely early (capturing the
    growth transient that separates ``α`` from ``βᵢᵢ``); set it ``False`` for the
    near-equilibrium regime where that pair is degenerate. Per-replicate multiplicative
    ``param_noise`` (extrinsic community variation) + log-space ``obs_noise`` mimic real
    ensembles. Returns a :class:`GLVDataset`.
    """
    if mechanism not in (*_KNOBS, "none"):
        raise ValueError(f"unknown mechanism {mechanism!r}")
    rng = np.random.default_rng(seed)
    base = baseline if baseline is not None else _default_baseline(n_species, rng)

    # observation times: dense early (transient) then sparse, or near-equilibrium tail.
    if dense_transient:
        early = np.linspace(0.0, pulse_window[1] + 1.0, n_obs - n_obs // 3)
        late = np.linspace(pulse_window[1] + 1.5, t_max, n_obs // 3)
        t_obs = np.unique(np.concatenate([early, late]))
    else:
        t_obs = np.linspace(0.6 * t_max, t_max, n_obs)
    grid_t, obs_idx = _fine_grid(t_max, dt, t_obs)
    u_grid = _pulse(grid_t[:-1], *pulse_window)

    if delta is None:
        delta = {"growth": 0.5, "interaction": -0.5, "susceptibility": -1.2,
                 "none": 0.0}[mechanism]
    pert = base if mechanism == "none" else base.with_knob(mechanism, target, delta)

    x0 = rng.uniform(0.3, 0.7, size=n_species).astype(np.float32)

    def ensemble(p: GLVParams) -> np.ndarray:
        trajs = []
        for _ in range(n_replicates):
            # extrinsic per-replicate multiplicative variation on the kinetics.
            a = p.alpha * (1.0 + param_noise * rng.standard_normal(n_species))
            b = p.beta * (1.0 + param_noise * rng.standard_normal((n_species, n_species)))
            e = p.eps + param_noise * rng.standard_normal(n_species)
            x0r = x0 * (1.0 + param_noise * rng.standard_normal(n_species))
            traj = np.asarray(
                simulate_glv(
                    (jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32),
                     jnp.asarray(e, jnp.float32)),
                    jnp.asarray(np.clip(x0r, 1e-3, None), jnp.float32),
                    jnp.asarray(u_grid), dt, jnp.asarray(obs_idx),
                )
            )
            # log-space observation noise (measurement error on the density).
            traj = traj * np.exp(obs_noise * rng.standard_normal(traj.shape))
            trajs.append(traj)
        return np.stack(trajs)

    reference = ensemble(base)
    perturbed = ensemble(pert)
    ground_truth = {
        "mechanism": mechanism,
        "target": target,
        "delta": float(delta),
        "dense_transient": dense_transient,
        "pulse_window": pulse_window,
    }
    return GLVDataset(
        reference=reference,
        perturbed=perturbed,
        t_obs=t_obs,
        u_grid=u_grid,
        obs_idx=obs_idx,
        dt=dt,
        baseline=base,
        ground_truth=ground_truth,
    )


def generate_alpha_beta_confound_decoy(*, seed: int = 0, **kwargs: Any) -> GLVDataset:
    """The α⇄βᵢᵢ confound decoy — a **growth** change sampled near equilibrium.

    The truth is a growth perturbation, but with only settled (near-equilibrium) samples
    a growth change and a self-limitation (``βᵢᵢ``) change are degenerate
    (``Kᵢ = −αᵢ/βᵢᵢ`` → same steady state). A naive attributor could confidently call
    ``interaction``; NUDGE must **abstain** (``unresolved``), with the degeneracy measured
    by the Laplace curvature (``NUDGE-LIM-020``). This is the "apparent β-change is really
    an α-change routed through the coupling → must abstain, not mis-call" case.
    """
    kwargs.setdefault("mechanism", "growth")
    kwargs.setdefault("dense_transient", False)  # near-equilibrium → degenerate
    return simulate_glv_perturbseq(seed=seed, **kwargs)


def generate_no_perturbation_null(*, seed: int = 0, **kwargs: Any) -> GLVDataset:
    """The no-perturbation null — reference and "perturbed" are the same community.

    NUDGE must return ``no-change`` (no restricted knob earns its parameter over the
    null): the fail-safe guard against manufacturing a mechanism from noise.
    """
    kwargs["mechanism"] = "none"
    return simulate_glv_perturbseq(seed=seed, **kwargs)


# --------------------------------------------------------------------------- #
# the trajectory fit loop (re-instantiated here; reuses losses.energy_distance)
# --------------------------------------------------------------------------- #
def _log_abund(x: Array) -> Array:
    return jnp.log(jnp.clip(x, 0.0, _X_CAP) + _LOG_OFFSET)


def _traj_energy_loss(sim_ensemble: Array, obs_ensemble: Array) -> Array:
    """Sum over timepoints of :func:`~nudge.inference.losses.energy_distance`.

    Both ensembles are ``(R, T, S)`` in **log-abundance** space; at each timepoint the
    replicate cloud ``(R, S)`` of the simulation is matched to the observation's — a
    genuinely *distributional* trajectory fit (matches the per-time spread across
    replicate communities, not just the mean), reusing the shipped energy distance
    verbatim.
    """
    n_t = sim_ensemble.shape[1]
    total = jnp.asarray(0.0)
    for t in range(n_t):
        total = total + energy_distance(sim_ensemble[:, t, :], obs_ensemble[:, t, :])
    return total


def _sim_ensemble(
    params_jax: tuple[Array, Array, Array],
    x0_ens: Array,
    u_grid: Array,
    dt: float,
    obs_idx: Array,
    noise_z: Array,
    obs_noise: float,
) -> Array:
    """A differentiable simulated ensemble: vmap the integrator over per-replicate ``x0``,
    then add **fixed** reparameterized log-noise ``noise_z`` (common random numbers) so
    the energy-distance loss is deterministic and its gradient is clean."""
    def one(x0: Array) -> Array:
        return simulate_glv(params_jax, x0, u_grid, dt, obs_idx)

    det = jax.vmap(one)(x0_ens)  # (R, T, S)
    return det * jnp.exp(obs_noise * noise_z)


def fit_baseline_glv(
    dataset: GLVDataset,
    *,
    steps: int = 400,
    learning_rate: float = 0.05,
    obs_noise: float = 0.05,
    n_sim: int = 40,
    seed: int = 0,
) -> tuple[GLVParams, float]:
    """Recover the reference community's kinetics from its control replicates.

    A full gLV fit (α, β, ε) by minimizing the per-timepoint energy distance between a
    simulated ensemble and the reference ensemble, initialized from the data (early
    log-growth → ``α``; steady-state means → ``βᵢᵢ = −αᵢ/x*``). Returns ``(params,
    final_loss)``. This is the honest baseline — the attribution then asks which single
    knob the perturbation moved *relative to this fitted reference* (not the truth).
    """
    ref = dataset.reference
    n_species = dataset.n_species
    rng = np.random.default_rng(seed)
    n_t = len(dataset.t_obs)

    # data-driven init.
    mean_traj = ref.mean(axis=0)  # (T, S)
    tail = max(n_t // 4, 1)
    x_star = np.clip(mean_traj[-tail:].mean(axis=0), 1e-2, None)
    head = max(n_t // 4, 2)
    early = np.clip(mean_traj[:head], 1e-3, None)
    dt_head = max(float(dataset.t_obs[head - 1] - dataset.t_obs[0]), 1e-3)
    slope = (np.log(early[-1]) - np.log(early[0])) / dt_head
    alpha0 = np.clip(slope, 0.1, 2.0)
    beta0 = np.zeros((n_species, n_species))
    for i in range(n_species):
        beta0[i, i] = -alpha0[i] / x_star[i]
    eps0 = np.zeros(n_species)

    theta = {
        "alpha": jnp.asarray(alpha0, jnp.float32),
        "beta": jnp.asarray(beta0, jnp.float32),
        "eps": jnp.asarray(eps0, jnp.float32),
    }
    x0_ens = jnp.asarray(ref[:n_sim, 0, :], jnp.float32)
    obs_log = jnp.asarray(_log_abund(jnp.asarray(ref[:n_sim], jnp.float32)))
    noise_z = jnp.asarray(rng.standard_normal((n_sim, n_t, n_species)), jnp.float32)
    u_grid = jnp.asarray(dataset.u_grid)
    obs_idx = jnp.asarray(dataset.obs_idx)

    def loss(th: dict[str, Array]) -> Array:
        sim = _sim_ensemble(
            (th["alpha"], th["beta"], th["eps"]), x0_ens, u_grid, dataset.dt,
            obs_idx, noise_z, obs_noise,
        )
        return _traj_energy_loss(_log_abund(sim), obs_log)

    opt = optax.adam(learning_rate)
    state = opt.init(theta)

    @jax.jit
    def step(th: dict[str, Array], st: optax.OptState) -> tuple[Any, Any, Array]:
        val, grad = jax.value_and_grad(loss)(th)
        updates, st = opt.update(grad, st)
        return optax.apply_updates(th, updates), st, val

    val = jnp.asarray(0.0)
    for _ in range(steps):
        theta, state, val = step(theta, state)
    params = GLVParams(
        alpha=np.asarray(theta["alpha"]),
        beta=np.asarray(theta["beta"]),
        eps=np.asarray(theta["eps"]),
    )
    return params, float(val)


def _bic(rss: float, n_obs: int, k: int) -> float:
    """Gaussian-residual BIC (unknown variance), matching :mod:`dose_response`."""
    rss = max(rss, 1e-12)
    return k * np.log(n_obs) + n_obs * (np.log(2 * np.pi) + np.log(rss / n_obs) + 1.0)


def _knob_apply(
    base_jax: tuple[Array, Array, Array], knob: str | None, target: int, d: Array
) -> tuple[Array, Array, Array]:
    """Apply an additive ``delta`` (``d[0]``) to one knob of ``target`` (``None`` = null)."""
    a0, b0, e0 = base_jax
    if knob == "growth":
        return a0.at[target].add(d[0]), b0, e0
    if knob == "susceptibility":
        return a0, b0, e0.at[target].add(d[0])
    if knob == "interaction":
        return a0, b0.at[target, target].add(d[0]), e0
    return a0, b0, e0


def _obs_contrast(dataset: GLVDataset) -> np.ndarray:
    """The measured reference→perturbed effect: ``log(pert_mean) − log(ref_mean)`` ``(T, S)``.

    Attribution is inherently about the **contrast**, and scoring the contrast (not the
    absolute level) **cancels the baseline fit's mean-bias** — the leak that would
    otherwise let any free knob shrink a null's residual and fake a mechanism. Under a
    true null this contrast is ~measurement noise, so no knob earns its parameter.
    """
    ref_mean = np.asarray(_log_abund(jnp.asarray(dataset.reference, jnp.float32))
                          ).mean(axis=0)
    pert_mean = np.asarray(_log_abund(jnp.asarray(dataset.perturbed, jnp.float32))
                           ).mean(axis=0)
    return pert_mean - ref_mean


def _model_contrast(
    baseline: GLVParams, knob: str, target: int, delta: float, dataset: GLVDataset,
    x0_rep: Array,
) -> np.ndarray:
    """The knob's PREDICTED effect: ``log(sim_{base+δ}) − log(sim_{base})`` from a common
    ``x0`` (so the baseline dynamics cancel, leaving only the delta's signature)."""
    base_jax = baseline.jax()
    u_grid = jnp.asarray(dataset.u_grid)
    obs_idx = jnp.asarray(dataset.obs_idx)
    sim_base = _log_abund(simulate_glv(base_jax, x0_rep, u_grid, dataset.dt, obs_idx))
    pert_jax = _knob_apply(base_jax, knob, target, jnp.asarray([delta], jnp.float32))
    sim_pert = _log_abund(simulate_glv(pert_jax, x0_rep, u_grid, dataset.dt, obs_idx))
    return np.asarray(sim_pert - sim_base)


def _fit_delta(
    baseline: GLVParams,
    dataset: GLVDataset,
    knob: str | None,
    target: int,
    *,
    steps: int,
    learning_rate: float,
    obs_noise: float,
    n_sim: int,
    seed: int,
) -> tuple[float, float]:
    """Fit a single additive ``delta`` on ``knob`` and score its **contrast** RSS.

    The delta is fit by the distributional energy distance on the perturbed replicate
    ensemble (the reused trajectory-matching loop). The BIC likelihood target, though, is
    the **contrast** RSS — how well the delta's predicted effect
    (:func:`_model_contrast`) matches the measured reference→perturbed contrast
    (:func:`_obs_contrast`) — which cancels the baseline mean-bias so a null cannot be
    beaten by a spurious knob. ``knob is None`` is the null (``delta = 0``, contrast = 0):
    ``rss = Σ obs_contrast²``. Returns ``(delta, rss)``.
    """
    pert = dataset.perturbed
    n_species = dataset.n_species
    rng = np.random.default_rng(seed)
    base_jax = baseline.jax()
    x0_ens = jnp.asarray(pert[:n_sim, 0, :], jnp.float32)
    obs_log = jnp.asarray(_log_abund(jnp.asarray(pert[:n_sim], jnp.float32)))
    noise_z = jnp.asarray(
        rng.standard_normal((n_sim, len(dataset.obs_idx), n_species)), jnp.float32
    )
    u_grid = jnp.asarray(dataset.u_grid)
    obs_idx = jnp.asarray(dataset.obs_idx)
    obs_contrast = _obs_contrast(dataset)

    theta = jnp.zeros(1, dtype=jnp.float32)
    if knob is not None:
        def loss(d: Array) -> Array:
            sim = _sim_ensemble(
                _knob_apply(base_jax, knob, target, d), x0_ens, u_grid, dataset.dt,
                obs_idx, noise_z, obs_noise,
            )
            return _traj_energy_loss(_log_abund(sim), obs_log)

        opt = optax.adam(learning_rate)
        state = opt.init(theta)

        @jax.jit
        def step(d: Array, st: optax.OptState) -> tuple[Any, Any, Array]:
            val, grad = jax.value_and_grad(loss)(d)
            updates, st = opt.update(grad, st)
            return optax.apply_updates(d, updates), st, val

        for _ in range(steps):
            theta, state, _val = step(theta, state)

    delta = float(theta[0])
    if knob is None:
        rss = float(np.sum(obs_contrast ** 2))
    else:
        x0_rep = jnp.asarray(pert[:, 0, :].mean(axis=0), jnp.float32)
        model_contrast = _model_contrast(baseline, knob, target, delta, dataset, x0_rep)
        rss = float(np.sum((model_contrast - obs_contrast) ** 2))
    return delta, rss


# --------------------------------------------------------------------------- #
# identifiability — the α⇄βᵢᵢ degeneracy, MEASURED via the Laplace curvature
# --------------------------------------------------------------------------- #
def alpha_beta_identifiability(
    baseline: GLVParams,
    dataset: GLVDataset,
    target: int,
    *,
    obs_noise: float = 0.05,
    n_sim: int = 40,
    cond_max: float = 100.0,
    seed: int = 0,
) -> LaplacePosterior:
    """Measure the growth ⇄ self-limitation (``αₜ`` ⇄ ``βₜₜ``) degeneracy.

    Builds a smooth Gaussian trajectory NLL of the perturbed group over the **log
    magnitudes** of the confounded pair ``(αₜ, |βₜₜ|)`` and hands its Hessian to
    :func:`~nudge.inference.uncertainty.laplace_posterior` — exactly how NUDGE measures
    the gain⇄threshold degeneracy elsewhere. A near-singular Hessian (high condition
    number, ``|corr| → 1``) *earns* the abstention (``NUDGE-LIM-020``): it is the
    ``Kᵢ = −αᵢ/βᵢᵢ`` trade-off made quantitative, not asserted. Reuses ``uncertainty.py``
    verbatim (both knobs are positive magnitudes, so its log-space CIs apply cleanly).
    """
    pert = dataset.perturbed
    n_species = dataset.n_species
    a0, b0, e0 = baseline.jax()
    alpha_t = float(baseline.alpha[target])
    beta_tt = float(baseline.beta[target, target])
    theta_opt = np.array([np.log(max(alpha_t, 1e-3)), np.log(max(-beta_tt, 1e-3))])

    obs_mean = jnp.asarray(_log_abund(jnp.asarray(pert, jnp.float32)).mean(axis=0))
    obs_sd = float(np.std(np.asarray(_log_abund(jnp.asarray(pert, jnp.float32)))))
    obs_var = max(obs_sd, 1e-2) ** 2
    x0_ens = jnp.asarray(pert[:n_sim, 0, :], jnp.float32)
    u_grid = jnp.asarray(dataset.u_grid)
    obs_idx = jnp.asarray(dataset.obs_idx)

    def loss_fn(log_theta: Array) -> Array:
        alpha_i = jnp.exp(log_theta[0])
        beta_ii = -jnp.exp(log_theta[1])
        a = a0.at[target].set(alpha_i)
        b = b0.at[target, target].set(beta_ii)

        def one(x0: Array) -> Array:
            return simulate_glv((a, b, e0), x0, u_grid, dataset.dt, obs_idx)

        sim_mean = _log_abund(jax.vmap(one)(x0_ens)).mean(axis=0)
        # mean Gaussian NLL over (T, S) — a proper likelihood curvature.
        return jnp.mean(0.5 * (sim_mean - obs_mean) ** 2 / obs_var)

    n_data = int(pert.shape[0] * len(dataset.t_obs) * n_species)
    return laplace_posterior(
        loss_fn, theta_opt, names=["alpha_t", "abs_beta_tt"],
        n_data=n_data, cond_max=cond_max,
    )


# --------------------------------------------------------------------------- #
# directional abstention — turn UNRESOLVED into an ACTIONABLE null-space direction
# --------------------------------------------------------------------------- #
#: A parameter loads "substantially" on the flat direction above this magnitude (the
#: eigenvector is unit-norm). A genuinely axis-aligned flat direction has ~0 load on the
#: OTHER axis; a confounded pair (|corr|→1) has the flat direction on the diagonal, where
#: even a tilted direction keeps both loads non-trivial. ~0.2 (≈4% of the direction's power
#: on that axis, ≳11° off-axis) cleanly separates "both confounded" from "one flat".
_LOAD_TOL = 0.2


@dataclass(frozen=True)
class DegeneracyDirection:
    """The **null-space direction** the α⇄βᵢᵢ degeneracy lives along — the actionable half
    of an ``unresolved`` abstention (``NUDGE-LIM-020``).

    When the Laplace/Fisher curvature on the confounded pair ``(αₜ, |βₜₜ|)`` is
    near-singular, the fit cannot move *along its flat eigenvector* without changing the
    likelihood — that eigenvector is exactly the combination of parameters the data does
    not constrain. Exposing it converts a bare "cannot tell" into "cannot tell **these two
    things apart, in this direction** — here is the experiment that would."

    - ``names`` — the confounded parameters, in vector order (``("alpha_t", "abs_beta_tt")``).
    - ``vector`` — the **unit null eigenvector** (smallest-eigenvalue direction) of the
      already-computed Hessian, in the **log**-parameter space of ``names`` (sign-canonical:
      the dominant component is positive).
    - ``eigenvalue`` — its (near-zero) curvature; ``cond_number`` — the pair's condition
      number (∞ when a direction is perfectly flat).
    - ``hint`` — the null direction mapped to a plain-language phrase.
    """

    names: tuple[str, ...]
    vector: np.ndarray
    eigenvalue: float
    cond_number: float
    hint: str


def _degeneracy_hint(alpha_load: float, beta_load: float) -> str:
    """Map the null eigenvector's (α, β) loadings to a human-readable phrase.

    ``alpha_load`` / ``beta_load`` are the null direction's components on ``log αₜ`` and
    ``log |βₜₜ|`` (sign already canonicalized). Both substantial → the classic confound
    ("cannot separate growth from interaction"); one axis dominant → that single knob is
    the unidentifiable one.
    """
    a, b = abs(float(alpha_load)), abs(float(beta_load))
    both = a >= _LOAD_TOL and b >= _LOAD_TOL
    if both:
        together = float(alpha_load) * float(beta_load) > 0.0
        how = (
            "raises growth αₜ and self-limitation |βₜₜ| together, leaving the carrying "
            "capacity Kₜ=−αₜ/βₜₜ (hence the steady state) unchanged"
            if together
            else "trades growth αₜ against self-limitation |βₜₜ| at fixed steady state"
        )
        return (
            "Cannot separate Growth (α) from Interaction (β): the fit's flat direction "
            f"{how}, so this sampling cannot tell an intrinsic-growth change from a "
            "carrying-capacity / interaction change. Resolve it by sampling the growth "
            "TRANSIENT (denser observations while the community is still climbing to "
            "equilibrium), which breaks the Kₜ=−αₜ/βₜₜ tie (NUDGE-LIM-020)."
        )
    if a > b:
        return (
            "Growth (α) is not identifiable here: the flat direction lies almost entirely "
            "along αₜ while self-limitation |βₜₜ| is comparatively constrained — the growth "
            "rate is under-determined by this sampling (NUDGE-LIM-020)."
        )
    return (
        "Interaction / self-limitation (βₜₜ) is not identifiable here: the flat direction "
        "lies almost entirely along |βₜₜ| while growth αₜ is comparatively constrained — "
        "the carrying capacity is under-determined by this sampling (NUDGE-LIM-020)."
    )


def degeneracy_direction_from_posterior(
    post: LaplacePosterior,
) -> DegeneracyDirection:
    """Extract the α⇄βᵢᵢ **null-space direction** from an ALREADY-COMPUTED Laplace posterior.

    Reuses ``post.hessian`` (the Hessian :func:`alpha_beta_identifiability` already built —
    this does **not** recompute the autodiff curvature); a cheap 2×2 ``eigh`` gives its
    eigenvectors, and the **smallest-eigenvalue eigenvector** is the flat / unconstrained
    direction. Names are read back from ``post.marginal_ci`` so the α and β axes are mapped
    by name, not position. Returns a :class:`DegeneracyDirection` with the sign-canonical
    unit vector, its curvature, and the human-readable hint.
    """
    names = tuple(ci.name for ci in post.marginal_ci)
    evals, evecs = np.linalg.eigh(np.asarray(post.hessian, dtype=np.float64))
    null = np.asarray(evecs[:, 0], dtype=np.float64)  # smallest-eigenvalue eigenvector
    lam0 = float(evals[0])

    # locate the α and β axes by name (robust to ordering); fall back to positional.
    alpha_i = next((i for i, n in enumerate(names) if "alpha" in n.lower()), 0)
    beta_i = next((i for i, n in enumerate(names) if "beta" in n.lower()), 1)

    # sign-canonicalize: make the dominant component positive (a direction is ±-ambiguous).
    dom = int(np.argmax(np.abs(null)))
    if null[dom] < 0.0:
        null = -null

    hint = _degeneracy_hint(null[alpha_i], null[beta_i])
    return DegeneracyDirection(
        names=names,
        vector=null,
        eigenvalue=lam0,
        cond_number=float(post.cond_number),
        hint=hint,
    )


# --------------------------------------------------------------------------- #
# result containers + the fail-safe classifier
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GLVFit:
    """The restricted-fit outcome: per-model BIC / RSS / Δ, the winner, and the measured
    α⇄βᵢᵢ degeneracy from the Laplace curvature."""

    target: int
    n_species: int
    n_replicates: int
    n_timepoints: int
    baseline: GLVParams
    baseline_loss: float
    bic: dict[str, float]
    rss: dict[str, float]
    delta: dict[str, float]
    selected: str  # the min-BIC model among the three knobs
    cond_number: float  # α⇄βᵢᵢ Laplace condition number
    corr_alpha_beta: float  # |correlation| of the confounded pair
    degenerate: bool  # the Laplace degeneracy verdict
    identifiability_reason: str
    extras: dict[str, Any] = field(default_factory=dict)
    #: the α⇄βᵢᵢ null-space direction — populated iff the Laplace curvature is degenerate,
    #: else ``None`` (a well-conditioned fit has no flat direction to report).
    degeneracy: DegeneracyDirection | None = None


#: the ``call`` → coarse ``status`` map (the abstentions get their own explicit states).
_STATUS_OF: dict[str, str] = {
    "growth": "RESOLVED",
    "interaction": "RESOLVED",
    "susceptibility": "RESOLVED",
    "no-change": "NO_CHANGE",
    "unresolved": "UNRESOLVED",
}


@dataclass(frozen=True)
class GLVResult:
    """A gLV attribution + its conservative verdict and human-readable reason."""

    fit: GLVFit
    call: str  # growth | interaction | susceptibility | no-change | unresolved
    reason: str

    @property
    def is_reliable(self) -> bool:
        """A resolved single-knob attribution is trustworthy; the abstentions are not."""
        return self.call in _KNOBS

    @property
    def status(self) -> str:
        """A coarse status enum — ``RESOLVED`` / ``NO_CHANGE`` / ``UNRESOLVED``."""
        return _STATUS_OF.get(self.call, "UNRESOLVED")

    @property
    def _alpha_beta_abstention(self) -> bool:
        """True iff the verdict is the **α⇄βᵢᵢ directional abstention** (classify gate 2):
        an ``unresolved`` call whose best knob is growth/interaction AND the curvature is
        degenerate. Only then is the null direction the *operative* reason to surface — a
        cleanly-resolved ``susceptibility`` call can co-exist with a degenerate α⇄β pair
        (ε is orthogonal to it), and there we must NOT cry "cannot separate growth from
        interaction"."""
        return (
            self.call == "unresolved"
            and self.fit.selected in ("growth", "interaction")
            and self.fit.degeneracy is not None
        )

    @property
    def degeneracy(self) -> DegeneracyDirection | None:
        """The α⇄βᵢᵢ null-space direction **when it is the operative reason for the
        abstention**, else ``None`` — the actionable half of ``NUDGE-LIM-020``.

        On an ``UNRESOLVED`` α⇄βᵢᵢ verdict this exposes *which* combination of (growth,
        interaction) the data cannot separate, not just that it cannot. A resolved fit, a
        ``no-change`` fit, or a tie-driven ``unresolved`` reports ``None``. (The raw
        curvature measurement, present whenever the pair is degenerate, lives on
        :attr:`GLVFit.degeneracy`.)
        """
        return self.fit.degeneracy if self._alpha_beta_abstention else None

    @property
    def degeneracy_direction(self) -> np.ndarray | None:
        """The unit null eigenvector in ``(alpha_t, abs_beta_tt)`` log-parameter space when
        the α⇄βᵢᵢ abstention is operative, else ``None``. See :attr:`degeneracy`."""
        d = self.degeneracy
        return None if d is None else d.vector

    @property
    def human_readable_hint(self) -> str | None:
        """The plain-language mapping of :attr:`degeneracy_direction` (e.g. *"Cannot
        separate Growth (α) from Interaction (β)"*), or ``None`` when not operative."""
        d = self.degeneracy
        return None if d is None else d.hint


def fit_glv_attribution(
    dataset: GLVDataset,
    *,
    baseline: GLVParams | None = None,
    target: int | None = None,
    steps: int = 300,
    learning_rate: float = 0.05,
    obs_noise: float = 0.05,
    n_sim: int = 40,
    cond_max: float = 100.0,
    seed: int = 0,
) -> GLVFit:
    """Fit the null + three restricted single-knob models and measure identifiability.

    If ``baseline`` is ``None`` it is recovered from the reference replicates
    (:func:`fit_baseline_glv`) — the honest round-trip. ``target`` defaults to the
    ground-truth target taxon (else 0). Returns a :class:`GLVFit`; the verdict is
    :func:`classify_glv`'s job.
    """
    if target is None:
        target = int(dataset.ground_truth.get("target", 0))
    if baseline is None:
        baseline, base_loss = fit_baseline_glv(
            dataset, steps=max(steps, 300), learning_rate=learning_rate,
            obs_noise=obs_noise, n_sim=n_sim, seed=seed,
        )
    else:
        base_loss = float("nan")

    bic: dict[str, float] = {}
    rss: dict[str, float] = {}
    delta: dict[str, float] = {}
    n_species = dataset.n_species
    n_obs_pts = len(dataset.t_obs) * n_species

    d0, rss0 = _fit_delta(
        baseline, dataset, None, target, steps=steps, learning_rate=learning_rate,
        obs_noise=obs_noise, n_sim=n_sim, seed=seed,
    )
    bic["null"] = _bic(rss0, n_obs_pts, 1)  # variance only
    rss["null"] = rss0
    delta["null"] = d0

    for knob in _KNOBS:
        dk, rssk = _fit_delta(
            baseline, dataset, knob, target, steps=steps, learning_rate=learning_rate,
            obs_noise=obs_noise, n_sim=n_sim, seed=seed,
        )
        bic[knob] = _bic(rssk, n_obs_pts, 2)  # delta + variance
        rss[knob] = rssk
        delta[knob] = dk

    selected = min(_KNOBS, key=lambda k: bic[k])

    post = alpha_beta_identifiability(
        baseline, dataset, target, obs_noise=obs_noise, n_sim=n_sim,
        cond_max=cond_max, seed=seed,
    )
    corr = float(abs(post.correlation[0, 1])) if post.correlation.shape == (2, 2) else 0.0

    # Directional abstention: when the α⇄βᵢᵢ curvature is degenerate, extract the flat
    # null-space direction from the SAME Hessian (no recompute) so an UNRESOLVED verdict
    # can say WHICH combination the data cannot separate (NUDGE-LIM-020).
    degeneracy = (
        degeneracy_direction_from_posterior(post) if bool(post.degenerate) else None
    )

    return GLVFit(
        target=target,
        n_species=n_species,
        n_replicates=int(dataset.perturbed.shape[0]),
        n_timepoints=len(dataset.t_obs),
        baseline=baseline,
        baseline_loss=base_loss,
        bic=bic,
        rss=rss,
        delta=delta,
        selected=selected,
        cond_number=float(post.cond_number),
        corr_alpha_beta=corr,
        degenerate=bool(post.degenerate),
        identifiability_reason=post.reason,
        degeneracy=degeneracy,
    )


def classify_glv(
    fit: GLVFit,
    *,
    bic_margin: float = 10.0,
    resolve_margin: float = 6.0,
) -> tuple[str, str]:
    """Turn a gLV fit into a conservative verdict — fail-safe, 0 confident-wrong.

    Gates, most-conservative first:

    1. **no-change** — no restricted knob beats the null by ``bic_margin`` (the
       perturbation is inert or uncaptured; do not manufacture a mechanism).
    2. **unresolved — the α⇄βᵢᵢ degeneracy (``NUDGE-LIM-020``).** The winner is
       ``growth`` or ``interaction`` AND the Laplace curvature on ``(αₜ, βₜₜ)`` is
       degenerate (``cond_number > cond_max`` / a flat direction). Growth and
       self-limitation are confounded (``Kᵢ = −αᵢ/βᵢᵢ``) and the data does not resolve
       the transient that separates them — abstain, with the degeneracy **measured**.
    3. **unresolved — the two best knobs tie.** The winner does not beat the runner-up by
       ``resolve_margin`` — which knob moved is unidentifiable; abstain, don't guess.
    4. **growth / interaction / susceptibility** — the winner earns its parameter over the
       null AND beats the runner-up AND (if in the confounded pair) is identifiable.
    """
    d_null = fit.bic["null"] - fit.bic[fit.selected]
    others = [k for k in _KNOBS if k != fit.selected]
    runner = min(others, key=lambda k: fit.bic[k])
    d_runner = fit.bic[runner] - fit.bic[fit.selected]

    # 1. nothing earns its parameter over the null.
    if d_null < bic_margin:
        return "no-change", (
            f"no single-knob model beats the no-change null by ΔBIC≥{bic_margin:g} "
            f"(best ΔBIC={d_null:.1f}, knob={fit.selected}) — the perturbation is inert "
            "or not captured by an α/β/ε change; NUDGE declines to manufacture a "
            "mechanism"
        )

    # 2. the α⇄βᵢᵢ degeneracy — measured, not asserted (NUDGE-LIM-020).
    if fit.selected in ("growth", "interaction") and fit.degenerate:
        return "unresolved", (
            f"the winning knob is '{fit.selected}', but growth (αₜ) and self-limitation "
            f"(βₜₜ) are DEGENERATE here: the Laplace curvature on the pair is near-"
            f"singular (condition number {fit.cond_number:.0f}, |corr|="
            f"{fit.corr_alpha_beta:.3f}). Kᵢ=−αᵢ/βᵢᵢ means a growth change and a "
            "carrying-capacity change give the same steady state — separable only by the "
            "transient, which this sampling does not resolve. NUDGE abstains "
            f"(NUDGE-LIM-020). [{fit.identifiability_reason}]"
        )

    # 3. which knob moved must be identifiable.
    if d_runner < resolve_margin:
        return "unresolved", (
            f"a perturbation is real (ΔBIC vs null={d_null:.1f}) but WHICH knob moved is "
            f"unidentifiable: '{fit.selected}' beats runner-up '{runner}' by only "
            f"ΔBIC={d_runner:.1f} < {resolve_margin:g} — NUDGE abstains rather than guess"
        )

    # 4. a resolved single-knob attribution.
    call = _CALL_OF[fit.selected]
    guidance = {
        "growth": "the perturbation retuned a taxon's intrinsic growth rate α",
        "interaction": "the perturbation retuned a taxon's self-limitation / carrying "
                       "capacity βᵢᵢ (identifiable here — the transient separated it "
                       "from α)",
        "susceptibility": "the perturbation changed a taxon's DIRECT susceptibility ε to "
                          "the external drug — a time-localized on/off signature the "
                          "pulse window makes identifiable",
    }[fit.selected]
    return call, (
        f"{call}: the '{fit.selected}' model earns its parameter over the null "
        f"(ΔBIC={d_null:.1f}) and beats the runner-up '{runner}' (ΔBIC={d_runner:.1f}); "
        f"fitted Δ={fit.delta[fit.selected]:+.3f}. Read: {guidance}"
    )


def attribute_glv(
    dataset: GLVDataset,
    *,
    baseline: GLVParams | None = None,
    target: int | None = None,
    steps: int = 300,
    learning_rate: float = 0.05,
    obs_noise: float = 0.05,
    n_sim: int = 40,
    cond_max: float = 100.0,
    bic_margin: float = 10.0,
    resolve_margin: float = 6.0,
    seed: int = 0,
) -> GLVResult:
    """Fit + classify a gLV perturbation in one call — the CLI / notebook entry point.

    Attributes which single knob — **growth ``α`` / interaction ``β`` / susceptibility
    ``ε``** — the perturbation moved, or abstains (``no-change`` / ``unresolved``) with the
    α⇄βᵢᵢ degeneracy measured by the Laplace curvature. Fail-safe: recover-or-abstain,
    never a confident wrong knob.
    """
    fit = fit_glv_attribution(
        dataset, baseline=baseline, target=target, steps=steps,
        learning_rate=learning_rate, obs_noise=obs_noise, n_sim=n_sim,
        cond_max=cond_max, seed=seed,
    )
    call, reason = classify_glv(fit, bic_margin=bic_margin, resolve_margin=resolve_margin)
    return GLVResult(fit=fit, call=call, reason=reason)
