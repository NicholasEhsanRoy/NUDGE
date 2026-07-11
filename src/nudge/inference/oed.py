"""Gradient-based Optimal Experimental Design (OED) — the differentiability moat.

**The white-box advantage a black-box ODE solver cannot offer.** Everywhere else in
NUDGE we take gradients of a *fit* loss w.r.t. the *parameters*. Here we take gradients of
an *identifiability criterion* w.r.t. the **experimental design** itself. Because NUDGE
fits a **differentiable** white-box mechanistic model, the Fisher Information Matrix
``FIM(φ)`` — and therefore any experimental-design criterion built from it (D-, A-,
E-optimality, or the reciprocal Cramér–Rao bound of one target parameter) — is itself a
**differentiable function of an experimental-design parameter ``φ``** (the measurement
times, a pulse window, a dose). So we can take ``∂criterion/∂φ`` by autodiff — straight
through the ODE solve *and* the FIM assembly — and **gradient-ascend ``φ`` to the exact
optimal experiment ``φ*``** that maximally resolves a sloppy / degenerate parameter.

A legacy black-box solver has no ``∂/∂φ``: it can only **grid-search** the design space,
whose cost is exponential in the number of design knobs. NUDGE gets the **exact continuous
optimum** by gradients. This module makes the Lotka–Volterra directional abstention
(:mod:`nudge.inference.lotka_volterra`: *"the growth α ⇄ self-interaction βᵢᵢ pair is
degenerate near equilibrium — sample the transient to break the tie"*) **exact**: the OED
gradient tells you *precisely which measurement times* break the tie, and by *what measured
factor* they resolve the previously-sloppy parameter.

## What it computes

Given a :class:`DesignProblem` — a differentiable forward model ``observe(θ, φ)`` at a
nominal parameter ``θ₀``, an observation noise ``σ``, and a valid range for ``φ`` — it:

1. assembles ``FIM(φ) = J(φ)ᵀ J(φ) / σ²`` where ``J = ∂observe/∂θ`` (autodiff through the
   ODE solve; :func:`fisher_information`);
2. scores a smooth, differentiable **design criterion** of the FIM
   (:func:`d_optimality` / :func:`a_optimality` / :func:`e_optimality` /
   :func:`neg_log_crlb` — the last targets *one* sloppy parameter's Cramér–Rao bound);
3. **gradient-ascends ``φ``** to the optimum ``φ*`` (:func:`optimize_design`), clipping to
   the physically valid window; and
4. contrasts ``φ*`` against a naive / uniform design and a **black-box grid search**
   (:func:`grid_search_design`) so the "white-box gradient beats grid search" claim is
   **measured, not asserted** — the identifiability gain (the factor by which the target
   parameter's CRLB / the FIM's smallest eigenvalue improves) is a computed number.

## Honesty (the load-bearing points)

- The FIM is the **local** curvature at ``θ₀``; an optimal design is only optimal for
  parameters near ``θ₀`` (the standard local-OED caveat, ``NUDGE-LIM-023``). We report the
  gain we *measure* at ``θ₀`` and do not extrapolate it.
- Near-singular FIMs are inverted with a **guarded ridge** (never a plain pseudo-inverse,
  which would *zero* a flat direction's variance and *understate* the CRLB — the opposite
  of safe), mirroring :func:`nudge.inference.uncertainty.laplace_posterior`.
- Small-eigenvalue and matrix-inverse criteria need **float64**; the reported FIM / CRLB /
  eigenvalues are assembled in numpy ``float64`` from the (autodiff) Jacobian so the
  measured numbers are precision-safe regardless of the global ``jax_enable_x64`` flag.

Additive / opt-in and self-contained: it re-instantiates its own differentiable RK4
integrator (a ``lax.scan``, no ``diffrax``) and touches **neither ``fit.py`` nor
``core/``** — the frozen-core constraint that keeps the guarantees from rotting. It reuses
the sloppiness / Fisher grammar of :mod:`nudge.inference.sloppiness` and the guarded-inverse
discipline of :mod:`nudge.inference.uncertainty`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

__all__ = [
    "DesignProblem",
    "OEDResult",
    "GridSearchResult",
    "fisher_information",
    "d_optimality",
    "a_optimality",
    "e_optimality",
    "neg_log_crlb",
    "crlb",
    "min_eigenvalue",
    "criterion_value",
    "design_gradient",
    "optimize_design",
    "grid_search_design",
    "make_logistic_design_problem",
    "make_glv_design_problem",
]

#: log-offset for the log-abundance observation transform (abundances are ≥ 0, ≈ 0 in the
#: OFF state); ``log(x + _LOG_OFFSET)`` keeps the transform finite and scale-free — the same
#: transform :mod:`nudge.inference.lotka_volterra` uses.
_LOG_OFFSET = 1e-2
#: hard abundance cap inside the integrator so a diverging orbit is a bad design, not a NaN.
_X_CAP = 1e6
#: the four supported design objectives (all phrased as "maximize me").
_OBJECTIVES: tuple[str, ...] = ("d_opt", "a_opt", "e_opt", "crlb")


# --------------------------------------------------------------------------- #
# problem container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DesignProblem:
    """A differentiable forward model + the design knob ``φ`` we optimize over.

    ``observe(theta, phi) -> Array`` is the **differentiable** observation map: given the
    (log-space) parameters ``theta`` and a design vector ``phi`` (here the ``m`` measurement
    *times*), it returns the stacked observation vector ``(m·n_out,)`` the FIM is built from
    — autodiff-differentiable in **both** arguments (through the ODE solve, via a smooth
    interpolation of the trajectory at the continuous times ``phi``). ``theta0`` is the
    nominal (log-space) parameter the design is optimized *at* (local OED; ``NUDGE-LIM-023``);
    ``param_names`` labels its entries. ``sigma`` is the (iid Gaussian) observation noise in
    the observation space. ``phi_bounds`` = ``(t_min, t_max)`` clips the design to the
    physically measurable window. ``meta`` carries model provenance (e.g. the true kinetics)
    for tests / the notebook.
    """

    observe: Callable[[Array, Array], Array]
    theta0: np.ndarray
    param_names: tuple[str, ...]
    sigma: float
    phi_bounds: tuple[float, float]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_params(self) -> int:
        return int(np.asarray(self.theta0).shape[0])

    def param_index(self, name: str) -> int:
        """Positional index of a named parameter (for a targeted-CRLB objective)."""
        return self.param_names.index(name)


# --------------------------------------------------------------------------- #
# self-contained differentiable integrator + continuous-time observation
# --------------------------------------------------------------------------- #
def _rk4_fine(
    rhs: Callable[[Array], Array], x0: Array, grid_t: Array, dt: float
) -> Array:
    """Integrate ``dx/dt = rhs(x)`` on the fine grid (RK4 ``lax.scan``) → full trajectory.

    Returns the ``(G+1, S)`` trajectory (index 0 = ``t0``) on the whole fine grid so it can
    be **interpolated at continuous observation times** (the design ``φ``). ``rhs`` closes
    over the (traced) kinetics so the trajectory is differentiable w.r.t. them.
    """
    n_steps = int(grid_t.shape[0]) - 1

    def step(x: Array, _dummy: Array) -> tuple[Array, Array]:
        k1 = rhs(x)
        k2 = rhs(x + 0.5 * dt * k1)
        k3 = rhs(x + 0.5 * dt * k2)
        k4 = rhs(x + dt * k3)
        x_next = jnp.clip(x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0, _X_CAP)
        return x_next, x_next

    _final, traj = jax.lax.scan(step, x0, None, length=n_steps)
    return jnp.concatenate([x0[None, :], traj], axis=0)


def _observe_at(traj: Array, grid_t: Array, phi: Array) -> Array:
    """Interpolate a fine trajectory ``(G+1, S)`` at continuous times ``phi`` and log it.

    Linear interpolation (``jnp.interp``) is differentiable w.r.t. **both** the query times
    ``phi`` (the design gradient) and the trajectory values (the FIM's ``∂/∂θ``), so the
    whole ``criterion(φ)`` is autodiff-smooth. Returns the stacked log-abundance vector
    ``(m·S,)``.
    """
    n_species = int(traj.shape[1])
    cols = [
        jnp.interp(phi, grid_t, traj[:, s]) for s in range(n_species)
    ]  # each (m,)
    obs = jnp.stack(cols, axis=1)  # (m, S)
    return jnp.log(jnp.clip(obs, 0.0, _X_CAP) + _LOG_OFFSET).reshape(-1)


# --------------------------------------------------------------------------- #
# Fisher information + the differentiable design criteria
# --------------------------------------------------------------------------- #
def _jacobian(problem: DesignProblem, phi: Array, theta: Array) -> Array:
    """``J = ∂observe/∂θ`` at ``(theta, phi)`` — the sensitivity matrix ``(n_obs, p)``."""
    return jax.jacobian(lambda th: problem.observe(th, phi))(theta)


def fisher_information(
    problem: DesignProblem,
    phi: Array | np.ndarray | Sequence[float],
    theta: Array | np.ndarray | None = None,
) -> np.ndarray:
    """The Fisher information ``FIM(φ) = Jᵀ J / σ²`` at design ``φ`` (numpy ``float64``).

    ``J = ∂observe/∂θ`` is the autodiff Jacobian through the ODE solve; for iid Gaussian
    observation noise ``σ`` the FIM of the (log-)parameters is ``Jᵀ J / σ²`` — the curvature
    of the Gaussian NLL, exactly as in :func:`nudge.inference.sloppiness.fisher_information`.
    Assembled in numpy ``float64`` (from the Jacobian) so the reported information geometry
    is precision-safe even when the global ``jax_enable_x64`` flag is off. ``theta`` defaults
    to ``problem.theta0``.
    """
    th = problem.theta0 if theta is None else theta
    th_j = jnp.asarray(np.asarray(th, dtype=np.float64))
    phi_j = jnp.asarray(np.asarray(phi, dtype=np.float64))
    jac = np.asarray(_jacobian(problem, phi_j, th_j), dtype=np.float64)
    return (jac.T @ jac) / (float(problem.sigma) ** 2)


def _ridged(fim: Array, ridge: float) -> Array:
    """``fim + ridge·λ̄·I`` — a *relative* Tikhonov ridge (scale-free), so a near-singular
    FIM stays invertible and its flat direction gets a large-but-finite variance."""
    p = fim.shape[0]
    scale = jnp.maximum(jnp.trace(fim) / p, 1e-30)
    return fim + ridge * scale * jnp.eye(p, dtype=fim.dtype)


def d_optimality(fim: Array, *, ridge: float = 1e-6) -> Array:
    """**D-optimality**: ``log det FIM`` — maximize the total information (overall
    identifiability). Ridge-stabilized ``slogdet`` so a degenerate design has a finite,
    differentiable score. Differentiable in ``φ`` when ``fim = fim(φ)``."""
    sign, logabsdet = jnp.linalg.slogdet(_ridged(fim, ridge))
    return jnp.asarray(jnp.where(sign > 0, logabsdet, -jnp.inf))


def a_optimality(fim: Array, *, ridge: float = 1e-6) -> Array:
    """**A-optimality**: ``−tr(FIM⁻¹)`` — maximize (least negative) minimizes the *total*
    parameter variance (the sum of the CRLBs). Ridge-guarded inverse."""
    return -jnp.trace(jnp.linalg.inv(_ridged(fim, ridge)))


def e_optimality(fim: Array, *, ridge: float = 1e-6) -> Array:
    """**E-optimality**: the **smallest eigenvalue** of the FIM — maximize it to fight the
    *worst* (sloppiest) direction directly. This is the criterion that targets a degenerate
    direction without naming a parameter (the smallest eigenvalue *is* the sloppy mode's
    information). ``eigvalsh`` is differentiable; the ridge keeps it strictly positive."""
    return jnp.linalg.eigvalsh(_ridged(fim, ridge))[0]


def neg_log_crlb(fim: Array, index: int, *, ridge: float = 1e-6) -> Array:
    """**Targeted reciprocal-CRLB**: ``−log([FIM⁻¹]_{ii})`` for parameter ``index`` —
    maximize it to minimize the Cramér–Rao lower bound (the best achievable variance) of
    that *specific* sloppy parameter. This is the OED objective for *"resolve THIS
    degenerate parameter"* (e.g. the gLV growth ``α`` confounded with self-interaction
    ``βᵢᵢ``). Smooth in ``φ`` (matrix inverse + log), no eigendecomposition needed."""
    var = jnp.linalg.inv(_ridged(fim, ridge))[index, index]
    return -jnp.log(jnp.clip(var, 1e-300, None))


# --- numpy reporting helpers (float64, no autodiff) ------------------------- #
def crlb(fim: np.ndarray, *, ridge: float = 1e-8) -> np.ndarray:
    """Per-parameter Cramér–Rao lower bound ``diag(FIM⁻¹)`` (numpy ``float64``, guarded).

    The measured variance floor for each parameter under this design. Ridge-guarded so a
    near-singular FIM gives a large-but-finite (honest, over-cautious) bound — never a NaN
    and never the pseudo-inverse's *zeroed* flat direction (which would understate it)."""
    fim = np.asarray(fim, dtype=np.float64)
    p = fim.shape[0]
    scale = max(float(np.trace(fim)) / p, 1e-30)
    inv = np.linalg.inv(fim + ridge * scale * np.eye(p))
    return np.clip(np.diag(inv), 0.0, None)


def min_eigenvalue(fim: np.ndarray) -> float:
    """The smallest eigenvalue of the FIM (numpy ``float64``) — the information along the
    sloppiest direction; ~0 signals a (near-)degenerate design."""
    return float(np.linalg.eigvalsh(np.asarray(fim, dtype=np.float64))[0])


def _criterion_fn(
    objective: str, target_index: int, ridge: float
) -> Callable[[Array], Array]:
    """Return a ``fim -> scalar`` criterion (phrased as maximize) for ``objective``."""
    if objective == "d_opt":
        return lambda fim: d_optimality(fim, ridge=ridge)
    if objective == "a_opt":
        return lambda fim: a_optimality(fim, ridge=ridge)
    if objective == "e_opt":
        return lambda fim: e_optimality(fim, ridge=ridge)
    if objective == "crlb":
        return lambda fim: neg_log_crlb(fim, target_index, ridge=ridge)
    raise ValueError(f"unknown objective {objective!r}; expected one of {_OBJECTIVES}")


def criterion_value(
    problem: DesignProblem,
    phi: Array | np.ndarray | Sequence[float],
    *,
    objective: str = "crlb",
    target: str | int = 0,
    ridge: float = 1e-6,
    theta: Array | np.ndarray | None = None,
) -> float:
    """The (scalar) design criterion at ``φ`` — the number a black box would grid-search.

    ``objective`` ∈ ``{"d_opt", "a_opt", "e_opt", "crlb"}``; ``target`` (name or index)
    selects the parameter for the ``crlb`` objective. Higher is always better (each is
    phrased to maximize). Computed via the float64 FIM (:func:`fisher_information`)."""
    idx = target if isinstance(target, int) else problem.param_index(target)
    fim = jnp.asarray(fisher_information(problem, phi, theta))
    return float(_criterion_fn(objective, idx, ridge)(fim))


def design_gradient(
    problem: DesignProblem,
    phi: Array | np.ndarray | Sequence[float],
    *,
    objective: str = "crlb",
    target: str | int = 0,
    ridge: float = 1e-6,
) -> np.ndarray:
    """``∂criterion/∂φ`` — the white-box design gradient a black-box solver cannot form.

    Autodiff of the criterion **through the FIM assembly and the ODE solve**
    (``jax.grad`` of a function that itself calls ``jax.jacobian``). This vector is the
    entire moat: it points to the exact design change that most improves identifiability,
    in one pass regardless of the design dimension. Returned as numpy."""
    idx = target if isinstance(target, int) else problem.param_index(target)
    crit = _criterion_fn(objective, idx, ridge)
    theta_j = jnp.asarray(np.asarray(problem.theta0, dtype=np.float64))

    def obj(phi_v: Array) -> Array:
        jac = _jacobian(problem, phi_v, theta_j)
        fim = (jac.T @ jac) / (float(problem.sigma) ** 2)
        return crit(fim)

    grad = jax.grad(obj)(jnp.asarray(np.asarray(phi, dtype=np.float64)))
    return np.asarray(grad, dtype=np.float64)


# --------------------------------------------------------------------------- #
# result containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OEDResult:
    """A gradient-optimized design + the MEASURED identifiability gain over the start.

    ``phi_opt`` is the optimal design ``φ*``; ``criterion_*`` the objective value at the
    initial vs optimal design (higher = better). ``crlb_init`` / ``crlb_opt`` are the
    per-parameter Cramér–Rao bounds; ``target_crlb_*`` the targeted parameter's; and
    ``crlb_improvement`` = ``crlb_init/crlb_opt`` for that parameter — *the headline number*
    (the factor by which the previously-sloppy parameter is resolved). ``min_eig_*`` track
    the FIM's smallest eigenvalue (the sloppy-direction information).
    """

    objective: str
    target_name: str
    target_index: int
    phi_init: np.ndarray
    phi_opt: np.ndarray
    criterion_init: float
    criterion_opt: float
    fim_init: np.ndarray
    fim_opt: np.ndarray
    crlb_init: np.ndarray
    crlb_opt: np.ndarray
    target_crlb_init: float
    target_crlb_opt: float
    crlb_improvement: float
    min_eig_init: float
    min_eig_opt: float
    min_eig_improvement: float
    history: np.ndarray
    n_steps: int


@dataclass(frozen=True)
class GridSearchResult:
    """The best design a **black-box grid search** finds, and how many evaluations it cost.

    The honest baseline the white-box gradient is contrasted with: a black box has no
    ``∂criterion/∂φ`` so it must *evaluate* candidate designs. ``n_evaluations`` is how many
    forward-model FIM assemblies it took; ``best_criterion`` / ``best_phi`` the winner.
    """

    objective: str
    best_phi: np.ndarray
    best_criterion: float
    n_evaluations: int
    all_criteria: np.ndarray


# --------------------------------------------------------------------------- #
# the optimizer (projected gradient ASCENT on φ) + the grid-search baseline
# --------------------------------------------------------------------------- #
def optimize_design(
    problem: DesignProblem,
    phi_init: Array | np.ndarray | Sequence[float],
    *,
    objective: str = "crlb",
    target: str | int = 0,
    steps: int = 400,
    learning_rate: float = 0.1,
    ridge: float = 1e-6,
    seed: int = 0,
) -> OEDResult:
    """Gradient-**ascend** the design ``φ`` to the optimum ``φ*`` (projected Adam).

    Maximizes the chosen ``objective`` of ``FIM(φ)`` by Adam on ``∂criterion/∂φ`` (the
    white-box gradient), **projecting** ``φ`` back into ``problem.phi_bounds`` after each
    step (so the design stays physically measurable). The optimum resolves the sloppy /
    degenerate parameter; the returned :class:`OEDResult` reports the *measured* gain
    (``crlb_improvement`` / ``min_eig_improvement``) vs the starting design — nothing
    asserted. ``target`` (name/index) selects the parameter for the ``crlb`` objective;
    it is also the parameter whose CRLB improvement is headlined for the other objectives.
    """
    if objective not in _OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; expected one of {_OBJECTIVES}")
    idx = target if isinstance(target, int) else problem.param_index(target)
    lo, hi = problem.phi_bounds
    crit = _criterion_fn(objective, idx, ridge)
    theta_j = jnp.asarray(np.asarray(problem.theta0, dtype=np.float64))
    sigma2 = float(problem.sigma) ** 2

    def neg_obj(phi_v: Array) -> Array:
        jac = _jacobian(problem, phi_v, theta_j)
        fim = (jac.T @ jac) / sigma2
        return -crit(fim)  # ascend the criterion == descend its negation

    phi0 = jnp.asarray(np.asarray(phi_init, dtype=np.float64))
    phi = phi0
    opt = optax.adam(learning_rate)
    state = opt.init(phi)

    def project(v: Array) -> Array:
        return jnp.clip(v, lo, hi)

    @jax.jit
    def step_fn(p: Array, st: optax.OptState) -> tuple[Array, Any, Array]:
        val, grad = jax.value_and_grad(neg_obj)(p)
        updates, st = opt.update(grad, st)
        p_new = project(jnp.asarray(optax.apply_updates(p, updates)))
        return p_new, st, val

    history = np.empty(steps, dtype=np.float64)
    for i in range(steps):
        phi, state, val = step_fn(phi, state)
        history[i] = -float(val)  # store the (maximized) criterion

    phi_init_np = np.asarray(phi0, dtype=np.float64)
    phi_opt_np = np.asarray(phi, dtype=np.float64)
    fim_init = fisher_information(problem, phi_init_np)
    fim_opt = fisher_information(problem, phi_opt_np)
    crlb_init = crlb(fim_init)
    crlb_opt = crlb(fim_opt)
    tci, tco = float(crlb_init[idx]), float(crlb_opt[idx])
    me_i, me_o = min_eigenvalue(fim_init), min_eigenvalue(fim_opt)

    return OEDResult(
        objective=objective,
        target_name=problem.param_names[idx],
        target_index=idx,
        phi_init=phi_init_np,
        phi_opt=phi_opt_np,
        criterion_init=float(_criterion_fn(objective, idx, ridge)(jnp.asarray(fim_init))),
        criterion_opt=float(_criterion_fn(objective, idx, ridge)(jnp.asarray(fim_opt))),
        fim_init=fim_init,
        fim_opt=fim_opt,
        crlb_init=crlb_init,
        crlb_opt=crlb_opt,
        target_crlb_init=tci,
        target_crlb_opt=tco,
        crlb_improvement=float(tci / tco) if tco > 0 else float("inf"),
        min_eig_init=me_i,
        min_eig_opt=me_o,
        min_eig_improvement=float(me_o / me_i) if me_i > 0 else float("inf"),
        history=history,
        n_steps=steps,
    )


def grid_search_design(
    problem: DesignProblem,
    candidates: Sequence[np.ndarray | Sequence[float]],
    *,
    objective: str = "crlb",
    target: str | int = 0,
    ridge: float = 1e-6,
) -> GridSearchResult:
    """The **black-box baseline**: evaluate the criterion on each candidate design, keep best.

    A legacy solver without ``∂criterion/∂φ`` can only *sample* the design space. This
    evaluates every candidate's FIM criterion (no gradient) and returns the winner + the
    number of forward-model evaluations it cost — the honest cost the white-box gradient is
    contrasted against (gradient: one pass, all design dims jointly; grid: exponential in the
    design dimension). Use it to *measure* that the gradient optimum meets or beats the best
    affordable grid design."""
    idx = target if isinstance(target, int) else problem.param_index(target)
    crit = _criterion_fn(objective, idx, ridge)
    scores = np.array(
        [float(crit(jnp.asarray(fisher_information(problem, c)))) for c in candidates],
        dtype=np.float64,
    )
    best = int(np.argmax(scores))
    return GridSearchResult(
        objective=objective,
        best_phi=np.asarray(candidates[best], dtype=np.float64),
        best_criterion=float(scores[best]),
        n_evaluations=len(candidates),
        all_criteria=scores,
    )


# --------------------------------------------------------------------------- #
# builders — the sloppy / degenerate showcase models
# --------------------------------------------------------------------------- #
def make_logistic_design_problem(
    *,
    alpha: float = 0.8,
    beta: float = -0.4,
    x0: float = 0.05,
    t_max: float = 12.0,
    dt: float = 0.02,
    sigma: float = 0.05,
    t_min: float = 0.02,
) -> DesignProblem:
    """A single-species **logistic** growth model ``dx/dt = x·(α + β·x)`` (``β < 0``).

    The canonical growth ⇄ carrying-capacity degeneracy, made analytically transparent: the
    steady state is ``K = −α/β``, so **measuring only the equilibrium leaves ``α`` and
    ``β`` confounded** (any ``(α, β)`` with the same ``K`` fits) — while the early
    **transient** climbs at rate ≈ ``α`` and *separates* them. Parameters are in **log
    magnitude** ``θ = (log α, log |β|)`` (scale-free, matching the sloppiness convention);
    the design ``φ`` is the vector of measurement times in ``[t_min, t_max]``. This is the
    exact showcase for *"sample the transient to break the α⇄βᵢᵢ tie"* — the OED gradient
    says by how much and where."""
    grid_np = (np.arange(int(round(t_max / dt)) + 1) * dt).astype(np.float64)
    x0_np = np.array([x0], dtype=np.float64)

    def observe(theta: Array, phi: Array) -> Array:
        a = jnp.exp(theta[0])
        b = -jnp.exp(theta[1])
        grid_t = jnp.asarray(grid_np, theta.dtype)
        x0_j = jnp.asarray(x0_np, theta.dtype)
        traj = _rk4_fine(lambda x: x * (a + b * x), x0_j, grid_t, dt)
        return _observe_at(traj, grid_t, phi)

    theta0 = np.array([np.log(alpha), np.log(-beta)], dtype=np.float64)
    return DesignProblem(
        observe=observe,
        theta0=theta0,
        param_names=("log_alpha", "log_abs_beta"),
        sigma=sigma,
        phi_bounds=(t_min, t_max),
        meta={"model": "logistic", "alpha": alpha, "beta": beta, "K": -alpha / beta,
              "x0": x0, "t_max": t_max, "dt": dt},
    )


def make_glv_design_problem(
    *,
    n_species: int = 3,
    target: int = 0,
    t_max: float = 12.0,
    dt: float = 0.02,
    sigma: float = 0.05,
    t_min: float = 0.02,
    seed: int = 0,
) -> DesignProblem:
    """A multi-species **generalized Lotka–Volterra** community — the degeneracy in an
    ecological context (matching :mod:`nudge.inference.lotka_volterra`).

    ``dxᵢ/dt = xᵢ·(αᵢ + Σⱼ βᵢⱼ xⱼ)`` with strong self-limitation and weak cross-coupling
    (diagonally dominant → bounded orbits). The two free parameters are the **target taxon's
    growth ``α_t`` and self-interaction ``|β_tt|``** (in log magnitude) — the pair that is
    degenerate near equilibrium (``K_t = −α_t/β_tt``) — with every other kinetic held fixed
    at truth. The design ``φ`` is the community's measurement times. Shows the OED gradient
    generalizes past the single-species logistic to a compositional network."""
    rng = np.random.default_rng(seed)
    s = n_species
    alpha = rng.uniform(0.6, 1.0, size=s)
    beta = np.zeros((s, s))
    for i in range(s):
        beta[i, i] = -rng.uniform(0.8, 1.2)
        for j in range(s):
            if i != j:
                beta[i, j] = rng.uniform(-0.12, 0.12)
    x0 = rng.uniform(0.1, 0.3, size=s)

    grid_np = (np.arange(int(round(t_max / dt)) + 1) * dt).astype(np.float64)
    alpha_t = float(alpha[target])
    beta_tt = float(beta[target, target])

    def observe(theta: Array, phi: Array) -> Array:
        grid_t = jnp.asarray(grid_np, theta.dtype)
        a = jnp.asarray(alpha, theta.dtype).at[target].set(jnp.exp(theta[0]))
        b = jnp.asarray(beta, theta.dtype).at[target, target].set(-jnp.exp(theta[1]))
        x0_j = jnp.asarray(x0, theta.dtype)

        def rhs(x: Array) -> Array:
            return x * (a + b @ x)

        traj = _rk4_fine(rhs, x0_j, grid_t, dt)
        return _observe_at(traj, grid_t, phi)

    theta0 = np.array([np.log(alpha_t), np.log(-beta_tt)], dtype=np.float64)
    return DesignProblem(
        observe=observe,
        theta0=theta0,
        param_names=("log_alpha_t", "log_abs_beta_tt"),
        sigma=sigma,
        phi_bounds=(t_min, t_max),
        meta={"model": "glv", "n_species": s, "target": target, "alpha": alpha,
              "beta": beta, "x0": x0, "K_t": -alpha_t / beta_tt, "t_max": t_max, "dt": dt},
    )
