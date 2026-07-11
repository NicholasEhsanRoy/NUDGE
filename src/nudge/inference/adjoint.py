"""Adjoint-method gradients of an ODE-fitting loss — O(1) in the parameter count.

**The computational capability gap.** Fitting a mechanistic ODE network to data needs
the gradient of a trajectory loss ``L(θ)`` w.r.t. **every** kinetic parameter ``θ``. The
obvious 0-shot way an un-aided coding agent writes this is **forward sensitivity
analysis**: propagate the sensitivity matrix ``S = ∂x/∂θ`` alongside the state, i.e.
integrate an **augmented system of size ``n_x·(1 + n_θ)``** whose cost (time *and*
memory) grows **linearly with the number of parameters** (Sengupta 2014;
``design/DEEP_RESEARCH_drug_discovery_directions.md`` finding 5). On a 15-state network
with a few hundred parameters that augmented system is large enough to time-out / OOM an
ad-hoc script.

The **adjoint method** computes the *same* gradient at a cost effectively **independent
of ``n_θ``** (``O(1)`` in the parameter count): one forward solve + one reverse solve,
regardless of how many parameters you differentiate w.r.t. Here the adjoint is the
**discrete adjoint** — reverse-mode automatic differentiation *through* the ``lax.scan``
RK4 integrator. Reverse-mode AD of a scan **is** the discrete adjoint recursion (the
VJP walks the same recurrence backwards), so :func:`adjoint_gradient` is a genuine
adjoint, not a black box we relabelled — and it is bit-for-bit equal to the forward-
sensitivity gradient (both are the *exact* gradient of the same discretised loss; see
:func:`forward_sensitivity_gradient` and ``tests/inference/test_adjoint.py``).

This module is **additive / opt-in** and self-contained: it never touches the frozen
``fit.py`` or ``core/``. It provides

- :func:`rk4_integrate` — a differentiable ``lax.scan`` RK4 integrator (mirrors the gLV
  integrator in :mod:`nudge.inference.lotka_volterra`);
- :class:`ODEProblem` — a parameterised ODE + observed trajectory + a flat **free
  parameter** vector, so the parameter count is a knob (used by the scaling benchmark);
- :func:`make_glv_problem` / :func:`make_linear_pathway_problem` — large-network
  generators (scalable to 15+ states and hundreds of parameters);
- :func:`adjoint_gradient` (the discrete adjoint) vs
  :func:`forward_sensitivity_gradient` (the explicit augmented-system baseline that
  blows up), which **agree to tight tolerance**;
- :func:`fit_ode_adjoint` — an opt-in optax fit path driven by the adjoint gradient.

**Honesty.** The claim "``O(1)`` in ``n_θ``" is *measured*, not asserted:
``scripts/vv/adjoint_scaling.py`` records wall-time and peak RSS of both methods as
``n_θ`` grows (5 → 200+), and ``scripts/vv/FINDINGS.md`` carries the curve. The adjoint's
cost is independent of ``n_θ`` but **not** free — reverse-mode stores the forward tape,
so its memory grows with the **trajectory length** (``NUDGE-LIM-022``); this module does
not implement gradient checkpointing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

__all__ = [
    "VectorField",
    "ODEProblem",
    "ODEFitResult",
    "rk4_integrate",
    "simulate",
    "residual_loss",
    "adjoint_gradient",
    "forward_sensitivity_gradient",
    "fit_ode_adjoint",
    "make_glv_problem",
    "make_linear_pathway_problem",
]

#: A parameterised vector field ``f(x, θ, u) -> dx/dt`` — ``x`` the state ``(n_x,)``,
#: ``θ`` the flat **free** parameter vector ``(n_θ,)``, ``u`` the scalar external input.
VectorField = Callable[[Array, Array, Array], Array]

#: Abundances / concentrations are clipped into ``[0, _X_CAP]`` inside the integrator so a
#: diverging orbit is merely a bad fit (a large finite loss), never a NaN gradient.
_X_CAP = 1e6


# --------------------------------------------------------------------------- #
# the differentiable RK4 integrator (a lax.scan; no diffrax)
# --------------------------------------------------------------------------- #
def rk4_integrate(
    field: VectorField,
    x0: Array,
    theta: Array,
    u_grid: Array,
    dt: float,
    obs_idx: Array,
) -> Array:
    """Integrate ``dx/dt = field(x, θ, u)`` with RK4 on a fine grid; gather obs times.

    ``u_grid`` is ``u(t)`` on the fine grid (length ``G``); ``obs_idx`` indexes the fine
    grid (0 = ``t0``) at the observation times. Returns the trajectory ``(T, n_x)`` at
    those times. A plain ``lax.scan`` of RK4 steps — differentiable w.r.t. ``theta`` by
    both forward-mode (forward sensitivity) and reverse-mode (the discrete adjoint).
    """

    def rk4_step(x: Array, u: Array) -> Array:
        k1 = field(x, theta, u)
        k2 = field(x + 0.5 * dt * k1, theta, u)
        k3 = field(x + 0.5 * dt * k2, theta, u)
        k4 = field(x + dt * k3, theta, u)
        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return jnp.clip(x_next, 0.0, _X_CAP)

    def body(x: Array, u: Array) -> tuple[Array, Array]:
        x_next = rk4_step(x, u)
        return x_next, x_next

    _final, traj = jax.lax.scan(body, x0, u_grid)
    traj = jnp.concatenate([x0[None, :], traj], axis=0)  # index 0 = t0
    return traj[obs_idx]


# --------------------------------------------------------------------------- #
# problem container + generators
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ODEProblem:
    """A parameterised ODE fit problem: a field, an initial state, an input, and data.

    ``field`` closes over any *fixed* parameters and reads the **free** parameter vector
    ``theta`` as its second argument — so ``n_theta`` (the length of :attr:`theta0`) is a
    free knob independent of the state dimension ``n_states``. That decoupling is what
    lets the scaling benchmark grow the parameter count at fixed state count and isolate
    the ``O(n_θ)`` (forward-sensitivity) vs ``O(1)`` (adjoint) contrast.

    ``target`` is the observed trajectory ``(T, n_states)`` the loss fits; ``theta0`` is
    the true / initial free parameters; ``obs_idx`` indexes ``u_grid``'s fine grid.
    """

    field: VectorField
    x0: np.ndarray
    u_grid: np.ndarray
    dt: float
    obs_idx: np.ndarray
    target: np.ndarray
    theta0: np.ndarray
    n_states: int
    #: the working precision; float64 (opt-in via ``jax_enable_x64``) is what makes the
    #: adjoint-vs-forward-sensitivity gradients agree to ~1e-11 in the correctness test.
    dtype: Any = jnp.float32

    @property
    def n_theta(self) -> int:
        return int(self.theta0.shape[0])

    def jax_args(self) -> tuple[Array, Array, Array, Array]:
        """Device copies of ``(x0, u_grid, obs_idx, target)`` for the fit/gradient paths."""
        return (
            jnp.asarray(self.x0, self.dtype),
            jnp.asarray(self.u_grid, self.dtype),
            jnp.asarray(self.obs_idx),
            jnp.asarray(self.target, self.dtype),
        )


def _pulse(grid_t: np.ndarray, t_on: float, t_off: float) -> np.ndarray:
    return ((grid_t >= t_on) & (grid_t < t_off)).astype(np.float32)


def _make_field_from_full(
    full: np.ndarray,
    free_idx: np.ndarray,
    unpack: Callable[[Array], Any],
    rhs: Callable[[Array, Any, Array], Array],
    dtype: Any = jnp.float32,
) -> VectorField:
    """Build a :data:`VectorField` that scatters the free ``theta`` back into the full
    parameter set (the rest held fixed), unpacks it, and evaluates ``rhs``.

    Keeping the fixed parameters as a constant and scattering only ``free_idx`` is what
    makes ``n_theta`` a clean knob: the same dynamics, differentiated w.r.t. any chosen
    subset of parameters.
    """
    full_j = jnp.asarray(full, dtype)
    idx_j = jnp.asarray(free_idx)

    def field(x: Array, theta: Array, u: Array) -> Array:
        full_p = full_j.at[idx_j].set(theta.astype(full_j.dtype))
        params = unpack(full_p)
        return rhs(x, params, u)

    return field


def make_glv_problem(
    *,
    n_species: int = 15,
    n_free: int = 50,
    t_max: float = 12.0,
    n_obs: int = 24,
    dt: float = 0.02,
    pulse_window: tuple[float, float] = (4.0, 6.0),
    obs_noise: float = 0.0,
    dtype: Any = jnp.float32,
    seed: int = 0,
) -> ODEProblem:
    """A generalized Lotka–Volterra community — a compositional ODE scalable to 15+ states.

    ``S = n_species`` taxa evolve under ``dxᵢ/dt = xᵢ (αᵢ + Σⱼ βᵢⱼ xⱼ + εᵢ u(t))`` with a
    drug pulse ``u`` on ``pulse_window``. The full parameter vector is
    ``[α (S) ‖ vec(β) (S²) ‖ ε (S)]`` (length ``S² + 2S``); the first ``n_free`` entries
    are the free parameters the gradient is taken w.r.t. (so ``n_free`` sweeps 5 → S²+2S
    at fixed ``S``). The community is diagonally dominant (bounded orbits), and ``target``
    is the trajectory at the true parameters (optionally with log-normal ``obs_noise``).
    """
    rng = np.random.default_rng(seed)
    s = n_species
    p_full = s * s + 2 * s
    n_free = int(np.clip(n_free, 1, p_full))

    alpha = rng.uniform(0.6, 1.0, size=s)
    beta = np.zeros((s, s))
    for i in range(s):
        beta[i, i] = -rng.uniform(0.8, 1.2)
        for j in range(s):
            if i != j:
                beta[i, j] = rng.uniform(-0.08, 0.08)
    eps = np.zeros(s)
    eps[: max(s // 3, 1)] = -rng.uniform(0.4, 0.8, size=max(s // 3, 1))
    full = np.concatenate([alpha, beta.reshape(-1), eps]).astype(np.float32)
    free_idx = np.arange(n_free)

    def unpack(full_p: Array) -> tuple[Array, Array, Array]:
        a = full_p[:s]
        b = full_p[s : s + s * s].reshape(s, s)
        e = full_p[s + s * s :]
        return a, b, e

    def rhs(x: Array, params: tuple[Array, Array, Array], u: Array) -> Array:
        a, b, e = params
        return x * (a + b @ x + e * u)

    field = _make_field_from_full(full, free_idx, unpack, rhs, dtype)

    n_steps = int(round(t_max / dt))
    grid_t = np.arange(n_steps + 1) * dt
    t_obs = np.linspace(0.0, t_max, n_obs)
    obs_idx = np.clip(np.round(t_obs / dt).astype(int), 0, n_steps)
    u_grid = _pulse(grid_t[:-1], *pulse_window)

    x0 = rng.uniform(0.3, 0.7, size=s).astype(np.float32)
    theta0 = full[free_idx]
    target = np.asarray(
        rk4_integrate(
            field, jnp.asarray(x0, dtype), jnp.asarray(theta0, dtype),
            jnp.asarray(u_grid, dtype), dt, jnp.asarray(obs_idx),
        )
    )
    if obs_noise > 0.0:
        target = target * np.exp(obs_noise * rng.standard_normal(target.shape))

    return ODEProblem(
        field=field, x0=x0, u_grid=u_grid, dt=dt, obs_idx=obs_idx,
        target=target.astype(np.float64), theta0=theta0.astype(np.float64),
        n_states=s, dtype=dtype,
    )


def make_linear_pathway_problem(
    *,
    n_states: int = 15,
    n_free: int | None = None,
    t_max: float = 8.0,
    n_obs: int = 24,
    dt: float = 0.01,
    dtype: Any = jnp.float32,
    seed: int = 0,
) -> ODEProblem:
    """A linear reaction cascade ``x₀ → x₁ → … → x_{n−1}`` (mass-action, no pulse).

    ``dx₀/dt = −k₀ x₀``; ``dxᵢ/dt = k_{i−1} xᵢ₋₁ − kᵢ xᵢ``. The free parameters are the
    ``n_states`` rate constants (a compositional network with a well-posed spectrum). A
    simpler, monotone alternative to the gLV community for the scaling benchmark;
    ``n_free`` defaults to all rates.
    """
    rng = np.random.default_rng(seed)
    n = n_states
    n_free = n if n_free is None else int(np.clip(n_free, 1, n))
    k = rng.uniform(0.4, 1.6, size=n).astype(np.float32)
    free_idx = np.arange(n_free)

    def unpack(full_p: Array) -> Array:
        return full_p

    def rhs(x: Array, params: Array, _u: Array) -> Array:
        kk = params
        inflow = jnp.concatenate([jnp.zeros(1), kk[:-1] * x[:-1]])
        return inflow - kk * x

    field = _make_field_from_full(k, free_idx, unpack, rhs, dtype)

    n_steps = int(round(t_max / dt))
    t_obs = np.linspace(0.0, t_max, n_obs)
    obs_idx = np.clip(np.round(t_obs / dt).astype(int), 0, n_steps)
    u_grid = np.zeros(n_steps, dtype=np.float32)

    x0 = np.zeros(n, dtype=np.float32)
    x0[0] = 1.0
    theta0 = k[free_idx]
    target = np.asarray(
        rk4_integrate(
            field, jnp.asarray(x0, dtype), jnp.asarray(theta0, dtype),
            jnp.asarray(u_grid, dtype), dt, jnp.asarray(obs_idx),
        )
    )
    return ODEProblem(
        field=field, x0=x0, u_grid=u_grid, dt=dt, obs_idx=obs_idx,
        target=target.astype(np.float64), theta0=theta0.astype(np.float64),
        n_states=n, dtype=dtype,
    )


# --------------------------------------------------------------------------- #
# the loss + the two gradient routes
# --------------------------------------------------------------------------- #
def simulate(problem: ODEProblem, theta: Array | np.ndarray) -> Array:
    """Integrate ``problem`` at free parameters ``theta`` → trajectory ``(T, n_states)``."""
    x0, u_grid, obs_idx, _target = problem.jax_args()
    return rk4_integrate(
        problem.field, x0, jnp.asarray(theta, problem.dtype), u_grid, problem.dt, obs_idx
    )


def _loss_fn(problem: ODEProblem) -> Callable[[Array], Array]:
    """Build the scalar mean-squared trajectory loss ``L(θ)`` as a closure over data."""
    x0, u_grid, obs_idx, target = problem.jax_args()

    def loss(theta: Array) -> Array:
        sim = rk4_integrate(problem.field, x0, theta, u_grid, problem.dt, obs_idx)
        return 0.5 * jnp.mean((sim - target) ** 2)

    return loss


def residual_loss(problem: ODEProblem, theta: Array | np.ndarray) -> float:
    """The scalar fit loss ``0.5·mean((x(θ) − y)²)`` at ``theta``."""
    return float(_loss_fn(problem)(jnp.asarray(theta, problem.dtype)))


def adjoint_gradient(problem: ODEProblem, theta: Array | np.ndarray) -> np.ndarray:
    """``dL/dθ`` by the **discrete adjoint** — reverse-mode AD through the scan integrator.

    Reverse-mode AD of a ``lax.scan`` *is* the discrete adjoint recursion: the VJP walks
    the RK4 recurrence backwards accumulating the adjoint state, so the whole gradient is
    obtained with **one** reverse sweep whose cost is **independent of ``n_θ``** (it does
    not integrate a per-parameter sensitivity). Returns the gradient as a numpy array.
    """
    grad = jax.grad(_loss_fn(problem))(jnp.asarray(theta, problem.dtype))
    return np.asarray(grad)


def _augmented_forward_sensitivity(
    problem: ODEProblem, theta: Array
) -> tuple[Array, Array]:
    """Integrate the augmented system ``(x, S)`` with ``S = ∂x/∂θ`` and gather obs times.

    The forward-sensitivity ODE is ``dx/dt = f`` and
    ``dS/dt = (∂f/∂x)·S + (∂f/∂θ)`` — an augmented state of size ``n_x·(1 + n_θ)`` whose
    per-step Jacobian ``∂f/∂θ`` (``n_x × n_θ``) and matmul both scale with ``n_θ``. This
    is the explicit baseline whose time *and* memory grow linearly in the parameter count.
    Returns ``(x_obs (T, n_x), S_obs (T, n_x, n_θ))``.
    """
    x0, u_grid, obs_idx, _target = problem.jax_args()
    field = problem.field
    dt = problem.dt
    n_x = problem.n_states
    n_theta = int(theta.shape[0])
    s0 = jnp.zeros((n_x, n_theta))

    def aug_rhs(x: Array, s: Array, u: Array) -> tuple[Array, Array]:
        fx = field(x, theta, u)
        jx = jax.jacobian(field, argnums=0)(x, theta, u)  # (n_x, n_x)
        jth = jax.jacobian(field, argnums=1)(x, theta, u)  # (n_x, n_θ)
        ds = jx @ s + jth
        return fx, ds

    def rk4_step(x: Array, s: Array, u: Array) -> tuple[Array, Array]:
        fx1, ds1 = aug_rhs(x, s, u)
        fx2, ds2 = aug_rhs(x + 0.5 * dt * fx1, s + 0.5 * dt * ds1, u)
        fx3, ds3 = aug_rhs(x + 0.5 * dt * fx2, s + 0.5 * dt * ds2, u)
        fx4, ds4 = aug_rhs(x + dt * fx3, s + dt * ds3, u)
        x_next = x + (dt / 6.0) * (fx1 + 2 * fx2 + 2 * fx3 + fx4)
        s_next = s + (dt / 6.0) * (ds1 + 2 * ds2 + 2 * ds3 + ds4)
        # match rk4_integrate's clip; freeze the sensitivity where the state saturates.
        clipped = (x_next <= 0.0) | (x_next >= _X_CAP)
        x_next = jnp.clip(x_next, 0.0, _X_CAP)
        s_next = jnp.where(clipped[:, None], 0.0, s_next)
        return x_next, s_next

    def body(carry: tuple[Array, Array], u: Array) -> tuple[tuple[Array, Array], Any]:
        x, s = carry
        x_next, s_next = rk4_step(x, s, u)
        return (x_next, s_next), (x_next, s_next)

    _final, (x_traj, s_traj) = jax.lax.scan(body, (x0, s0), u_grid)
    x_traj = jnp.concatenate([x0[None, :], x_traj], axis=0)
    s_traj = jnp.concatenate([s0[None, :, :], s_traj], axis=0)
    return x_traj[obs_idx], s_traj[obs_idx]


def forward_sensitivity_gradient(
    problem: ODEProblem, theta: Array | np.ndarray
) -> np.ndarray:
    """``dL/dθ`` by **explicit forward sensitivity** (the augmented ``n_x·(1+n_θ)`` system).

    Integrates ``S = ∂x/∂θ`` alongside the state (:func:`_augmented_forward_sensitivity`),
    then contracts the loss's state-gradient with the sensitivity:
    ``dL/dθ = mean_over(t,i) (xᵢ(t) − yᵢ(t)) · Sᵢₖ(t)``. Mathematically the *same* gradient
    as :func:`adjoint_gradient` (verified bit-close), but at ``O(n_θ)`` cost — the baseline
    the adjoint beats. Returns the gradient as a numpy array.
    """
    theta_j = jnp.asarray(theta, problem.dtype)
    _x0, _u, _idx, target = problem.jax_args()
    x_obs, s_obs = _augmented_forward_sensitivity(problem, theta_j)
    resid = x_obs - target  # (T, n_x)
    n_elem = resid.shape[0] * resid.shape[1]
    grad = jnp.einsum("ti,tik->k", resid, s_obs) / n_elem
    return np.asarray(grad)


# --------------------------------------------------------------------------- #
# the opt-in fit path (adjoint-driven; never touches fit.py)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ODEFitResult:
    """The recovered free parameters + loss history from :func:`fit_ode_adjoint`."""

    theta: np.ndarray
    loss: float
    loss_history: np.ndarray
    n_steps: int
    n_theta: int


def fit_ode_adjoint(
    problem: ODEProblem,
    theta_init: Array | np.ndarray | None = None,
    *,
    steps: int = 400,
    learning_rate: float = 0.05,
) -> ODEFitResult:
    """Fit ``problem``'s free parameters by Adam on the **adjoint** gradient (opt-in).

    A sibling fit loop that uses reverse-mode-through-the-scan (the discrete adjoint,
    :func:`adjoint_gradient`) so the per-step gradient cost is independent of the
    parameter count — the whole point of the adjoint for large networks. Never modifies
    the frozen ``fit.py``. ``theta_init`` defaults to a perturbed copy of the truth (a
    recovery test); pass your own for a real fit. Returns the recovered parameters + the
    loss trajectory.
    """
    loss_fn = _loss_fn(problem)
    if theta_init is None:
        rng = np.random.default_rng(0)
        theta_init = problem.theta0 * (1.0 + 0.1 * rng.standard_normal(problem.n_theta))
    theta = jnp.asarray(theta_init, problem.dtype)

    opt = optax.adam(learning_rate)
    state = opt.init(theta)

    @jax.jit
    def step(th: Array, st: optax.OptState) -> tuple[Array, Any, Array]:
        val, grad = jax.value_and_grad(loss_fn)(th)
        updates, st = opt.update(grad, st)
        return jnp.asarray(optax.apply_updates(th, updates)), st, jnp.asarray(val)

    history = np.empty(steps, dtype=np.float64)
    val = jnp.asarray(0.0)
    for i in range(steps):
        theta, state, val = step(theta, state)
        history[i] = float(val)

    return ODEFitResult(
        theta=np.asarray(theta),
        loss=float(val),
        loss_history=history,
        n_steps=steps,
        n_theta=problem.n_theta,
    )