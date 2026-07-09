"""The ``Circuit`` — NUDGE's compositional gene-regulatory model + deterministic solve.

A circuit is species (each with an integrator) plus regulatory edges. Its numerical
core is a **self-contained, differentiable JAX vector field** parameterized by a
parameter pytree ``θ`` (*not* baked constants), so we can ``vmap`` over per-cell
parameter draws (the population model) and differentiate for the fit.

**Architecture note.** The plan's ``steady_state(θ)`` / ``solve_population`` signatures
take parameters as *traced arguments*; MADDENING's ``GraphManager`` bakes node
parameters as compile-time constants (DESIGN.md §10), and its differentiable-param
path is a separate "calibrate" wrapper planned for MADDENING Phase 4. So the
per-cell-varying solve lives here in JAX and reuses MADDENING *primitives*
(``ift_linear_solve`` for the zero-order integrator, later) rather than routing
through ``GraphManager``. Generation integrates to convergence (robust near a
bifurcation); the near-critical adjoint fragility is a fit-time concern.

Combination of multiple regulators into a species' drive is **additive** for now
(each edge's cooperativity lives in its own Hill ``n``); multiplicative / AND-gate
combination is a later extension.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental import enable_x64

from nudge.mechanisms.integrators.saturating import saturating_production
from nudge.mechanisms.regulatory import (
    hill_activation,
    hill_repression,
    linear_effect,
)

__all__ = ["Circuit", "EdgeDef", "Params", "SpeciesDef"]

#: The parameter pytree: ``{"species": {...}, "edges": {...}}`` of arrays.
Params = dict[str, dict[str, Array]]


@dataclass(frozen=True)
class SpeciesDef:
    """A species node and its integrator + (true) kinetic parameters."""

    name: str
    integrator: str = "linear"  # "linear" | "saturating"
    basal: float = 0.0
    decay: float = 1.0
    vmax: float = 10.0  # saturating only (its own ceiling)
    km: float = 1.0  # saturating only


@dataclass(frozen=True)
class EdgeDef:
    """A regulatory edge ``source → target`` and its effect + (true) parameters."""

    source: int
    target: int
    effect: str = "hill_activation"  # linear | hill_activation | hill_repression
    K: float = 1.0  # threshold
    n: float = 1.0  # gain
    vmax: float = 1.0  # ceiling
    weight: float = 1.0  # linear only


class Circuit:
    """A differentiable gene-regulatory circuit with a deterministic solve."""

    def __init__(
        self, species: Sequence[SpeciesDef], edges: Sequence[EdgeDef]
    ) -> None:
        self.species = tuple(species)
        self.edges = tuple(edges)
        self.names = tuple(s.name for s in self.species)
        self._is_saturating = jnp.array(
            [s.integrator == "saturating" for s in self.species], dtype=bool
        )

    @property
    def n_species(self) -> int:
        return len(self.species)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def index(self, name: str) -> int:
        """Index of a species by name."""
        return self.names.index(name)

    def base_params(self) -> Params:
        """The circuit's declared ("true") parameters as a pytree of arrays."""
        sp, ed = self.species, self.edges
        return {
            "species": {
                "basal": jnp.array([s.basal for s in sp]),
                "decay": jnp.array([s.decay for s in sp]),
                "vmax": jnp.array([s.vmax for s in sp]),
                "km": jnp.array([s.km for s in sp]),
            },
            "edges": {
                key: jnp.array([getattr(e, key) for e in ed]) if ed else jnp.zeros(0)
                for key in ("K", "n", "vmax", "weight")
            },
        }

    def production(self, x: Array, params: Params) -> Array:
        """Per-species production (drive for linear; MM(drive) for saturating)."""
        sp, ep = params["species"], params["edges"]
        drive = sp["basal"]
        for i, e in enumerate(self.edges):
            xs = x[e.source]
            if e.effect == "hill_activation":
                r = hill_activation(xs, ep["K"][i], ep["n"][i], ep["vmax"][i])
            elif e.effect == "hill_repression":
                r = hill_repression(xs, ep["K"][i], ep["n"][i], ep["vmax"][i])
            else:  # linear
                r = linear_effect(xs, ep["weight"][i])
            drive = drive.at[e.target].add(r)
        mm = saturating_production(drive, sp["vmax"], sp["km"])
        return jnp.where(self._is_saturating, mm, drive)

    def vector_field(self, x: Array, params: Params) -> Array:
        """Time derivative ``dx/dt = production(x) − decay · x``."""
        return self.production(x, params) - params["species"]["decay"] * x

    def steady_state(
        self, params: Params, x0: Array, *, dt: float = 0.1, n_steps: int = 500
    ) -> Array:
        """Integrate to steady state from ``x0``.

        Semi-implicit decay (``x' = (x + dt·prod) / (1 + dt·decay)``) is stable for
        any ``dt``, so this converges robustly even for stiff / near-critical circuits.
        """
        decay = params["species"]["decay"]

        def step(x: Array, _: None) -> tuple[Array, None]:
            x_new = (x + dt * self.production(x, params)) / (1.0 + dt * decay)
            return jnp.maximum(x_new, 0.0), None  # activities are non-negative

        x_final, _ = jax.lax.scan(step, x0, None, length=n_steps)
        return x_final

    def solve_population(
        self, params: Params, x0: Array, *, dt: float = 0.1, n_steps: int = 500
    ) -> Array:
        """vmap the steady-state solve over per-cell params + initial states.

        ``params`` leaves carry a leading cell axis; ``x0`` is ``(n_cells, n_species)``.
        """

        def solve(cell_params: Params, cell_x0: Array) -> Array:
            return self.steady_state(cell_params, cell_x0, dt=dt, n_steps=n_steps)

        return jax.vmap(solve)(params, x0)

    def linear_baseline(self) -> Circuit:
        """The same topology with every edge swapped to ``LinearEffect``."""
        edges = tuple(replace(e, effect="linear") for e in self.edges)
        return Circuit(self.species, edges)

    def fixed_points(self) -> list[tuple[np.ndarray, str]] | None:
        """Steady-state fixed points + stability labels, or ``None`` if unsupported.

        Returns ``[(state_vector (n_species,) float32, label), ...]`` where label is
        ``stable`` / ``saddle-index1`` / ``source`` / ``other``. Decouples the topology-
        specific saddle math from the fit (which needs the unstable index-1 fixed point
        but should not itself know how to find one). Dispatch:

        - **1-species self-activation Hill switch**: the exact 1-D grid+bisection roots
          (unchanged math), ordered ``[low, saddle, high]`` (bistable) or ``[low]``
          (monostable) and labelled by that order;
        - **N-species (>=2)**: multi-start Newton + Jacobian-eigenvalue index
          classification (``_nd_fixed_points``, under a local x64 context). The heavy
          Newton/dedupe/eigenvalue kernel is **jitted and cached per topology** (base
          kinetics enter as a traced argument), so recomputing it every optimizer step
          costs ~1 ms — it traces once and only *executes* thereafter;
        - **any other 1-species topology**: ``None`` (caller abstains).

        Never raises (``None`` on any numerical failure) — safe in an optimizer step.
        """
        if self.n_species == 1:
            if self.n_edges != 1:
                return None
            e, s = self.edges[0], self.species[0]
            if e.source != 0 or e.target != 0 or e.effect != "hill_activation":
                return None
            try:
                roots = _self_activation_roots(
                    basal=s.basal, decay=s.decay, K=e.K, n=e.n, vmax=e.vmax
                )
            except Exception:
                return None
            labels = (
                ["stable", "saddle-index1", "stable"]
                if len(roots) == 3
                else ["stable"] * len(roots)
            )
            return [
                (np.asarray([r], dtype=np.float32), lab)
                for r, lab in zip(roots, labels, strict=True)
            ]
        try:
            return _nd_fixed_points(self)
        except Exception:
            return None

    def transition_state(self) -> np.ndarray | None:
        """The index-1 saddle state (n_species,) when bistable, else ``None``.

        Where graded cells pile up when a switch loses cooperativity (a gain reduction),
        so it centres the fit's transition mixture mode. ``None`` (monostable or an
        unsupported topology) means "no transition mode" — the fit collapses gracefully
        and the gain gate abstains. Generalizes the old 1-D scalar to an N-D vector (a
        length-1 array for 1 species — same saddle location, now a vector).
        """
        fps = self.fixed_points()
        if fps is None:
            return None
        saddles = [state for state, label in fps if label == "saddle-index1"]
        return saddles[0] if saddles else None

    def mode_covariances(self) -> list[tuple[np.ndarray, np.ndarray]] | None:
        """Per-stable-mode linear-noise covariance: ``[(mean, cov), ...]`` or ``None``.

        For each **stable** fixed point (a lobe of the stationary distribution), the
        local Gaussian covariance from the **linear-noise / Lyapunov equation**
        ``A Σ + Σ Aᵀ + D = 0`` — ``A`` the drift Jacobian at the mode (autodiff), ``D =
        diag(2·decay·μ)`` the birth-death diffusion (at a fixed point production =
        decay·μ, so birth + death = 2·decay·μ). Returns ``[(mean (n,) float32, cov
        (n,n) float32), ...]`` for the stable modes (mean-sorted, matching
        ``fixed_points``); ``None`` when there are no fixed points (unsupported
        topology / numerical failure).

        These are the mode means + shapes the covariance-structured Gaussian-mixture
        attribution loss fits — the channel that carries the gain/threshold/ceiling
        information (the Fisher-information analysis; ``design/`` +
        ``scripts/vv/fisher_sloppiness.py``). It is an LNA approximation: local to each
        stable mode, and degrades near a bifurcation / at low copy number, so callers
        should treat it as unreliable there. Never raises (``None`` on any failure).
        """
        fps = self.fixed_points()
        if fps is None:
            return None
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for state, label in fps:
            if label != "stable":
                continue
            try:
                cov = _lna_covariance(self, state)
            except Exception:
                return None
            out.append((np.asarray(state, dtype=np.float32), cov))
        return out or None


def _lna_covariance(circuit: Circuit, mu_np: np.ndarray) -> np.ndarray:
    """Linear-noise covariance at fixed point ``mu``: solve ``A Σ + Σ Aᵀ + D = 0``.

    ``A`` = the drift Jacobian at ``mu`` (``production − decay·x``, autodiff), ``D`` =
    ``diag(2·decay·μ)`` the birth-death diffusion. A Kronecker solve of the Lyapunov
    equation, under a **local x64 context** (the Jacobian is ill-conditioned near a
    saddle-node), cast to float32. Symmetrized to kill round-off asymmetry.
    """
    n = circuit.n_species
    with enable_x64():
        base = circuit.base_params()
        decay = base["species"]["decay"]

        def drift(x: Array) -> Array:
            return circuit.production(jnp.maximum(x, 0.0), base) - decay * x

        mu = jnp.asarray(mu_np, dtype=jnp.float64)
        jac = jax.jacfwd(drift)(mu)
        diff = jnp.diag(2.0 * decay * jnp.clip(mu, 1e-9))
        kron = jnp.kron(jnp.eye(n), jac) + jnp.kron(jac, jnp.eye(n))
        sig = jnp.linalg.solve(kron, -diff.reshape(-1)).reshape(n, n)
        sig = 0.5 * (sig + sig.T)
    return np.asarray(sig, dtype=np.float32)


def _self_activation_roots(
    *, basal: float, decay: float, K: float, n: float, vmax: float, n_grid: int = 2000
) -> list[float]:
    """Real roots of ``f(x) = basal + vmax·x^n/(K^n+x^n) − decay·x`` on ``x ≥ 0``.

    Grid sign-change detection + bisection (real roots only — no complex/duplicate
    artifacts), deduped. Returns a sorted list (1 root monostable, 3 bistable).
    """

    def f(x: np.ndarray) -> np.ndarray:
        return basal + vmax * x**n / (K**n + x**n) - decay * x

    hi = 1.3 * (basal + vmax) / decay
    xs = np.linspace(1e-6, hi, n_grid)
    fx = f(xs)
    roots: list[float] = []
    for i in range(len(xs) - 1):
        if fx[i] == 0.0:
            roots.append(float(xs[i]))
        elif fx[i] * fx[i + 1] < 0.0:
            a, b, fa = xs[i], xs[i + 1], fx[i]
            for _ in range(60):
                m = 0.5 * (a + b)
                fm = float(f(np.array([m]))[0])
                if fa * fm <= 0.0:
                    b = m
                else:
                    a, fa = m, fm
            roots.append(0.5 * (a + b))
    deduped: list[float] = []
    for r in sorted(roots):
        if not deduped or abs(r - deduped[-1]) > 1e-4:
            deduped.append(float(r))
    return deduped


def _nd_box_hi(circuit: Circuit) -> float:
    """Upper corner of the start box: ``1.2·max((basal + Σvmax)/decay)`` (+ ε)."""
    base = circuit.base_params()
    basal = np.asarray(base["species"]["basal"], dtype=float)
    dec = np.asarray(base["species"]["decay"], dtype=float)
    vmax_sum = (
        float(np.sum(np.asarray(base["edges"]["vmax"], dtype=float)))
        if circuit.n_edges
        else 0.0
    )
    return float(np.max((basal + vmax_sum) / dec)) * 1.2 + 1e-3


#: Per-topology cache of the jitted Newton/dedupe/eigenvalue kernel. Keyed on the
#: circuit's *structure* (species count, integrators, edge source/target/effect) — not
#: its kinetic values, which enter the kernel as a traced ``base`` argument. So one
#: compiled program serves every optimizer step (any kinetics, same topology).
_NDKernel = Callable[[Array, Params], tuple[Array, Array, Array]]
_ND_KERNEL_CACHE: dict[tuple, _NDKernel] = {}


def _nd_topology_key(circuit: Circuit) -> tuple:
    """Structural signature that determines ``Circuit.production``'s traced graph."""
    return (
        circuit.n_species,
        tuple(s.integrator for s in circuit.species),
        tuple((e.source, e.target, e.effect) for e in circuit.edges),
    )


def _nd_kernel(
    circuit: Circuit,
    *,
    tol: float = 1e-5,
    reg: float = 1e-6,
    dedup_tol: float = 1e-3,
    max_iter: int = 100,
    box_hi: float = 1e4,
) -> _NDKernel:
    """The jitted N-D root kernel for ``circuit``'s topology (built once, then cached).

    Returns a jitted ``(starts, base) → (roots, keep, eig_real)`` where the kinetics
    ``base`` is a **traced argument**, not a baked constant — so it compiles a single
    XLA program that every optimizer step reuses (the ~1 ms/step path; the un-jitted
    version re-traced the ``vmap(jacfwd-Newton)`` in Python each call, ~0.3 s).
    ``roots``/``keep`` are static ``[n_starts, n]`` / ``[n_starts]`` (no dynamic root
    lists in XLA); ``eig_real`` is the Jacobian eigenvalue real parts at every
    candidate, so the caller classifies eagerly with no per-root host round-trip.
    """
    key = _nd_topology_key(circuit)
    cached = _ND_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    n = circuit.n_species
    eye = jnp.eye(n)

    @jax.jit
    def kernel(starts: Array, base: Params) -> tuple[Array, Array, Array]:
        decay = base["species"]["decay"]

        def field(state: Array) -> Array:
            # clip >=0 inside production so a Newton overshoot never hits x^frac < 0
            return circuit.production(jnp.maximum(state, 0.0), base) - decay * state

        jac = jax.jacfwd(field)

        def newton(x0: Array) -> tuple[Array, Array]:
            def cond(c: tuple[Array, Array, Array]) -> Array:
                _, i, done = c
                return jnp.logical_and(i < max_iter, jnp.logical_not(done))

            def body(c: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
                x, i, _ = c
                fx, jx = field(x), jac(x)
                delta = jnp.linalg.solve(jx.T @ jx + reg * eye, -(jx.T @ fx))
                sn = jnp.linalg.norm(delta)
                x_new = x + jnp.minimum(1.0, 5.0 / (sn + 1e-30)) * delta
                x_new = jnp.clip(x_new, 0.0, box_hi)
                x_new = jnp.where(jnp.isfinite(x_new), x_new, box_hi)
                return x_new, i + 1, jnp.linalg.norm(field(x_new)) < tol

            init = (x0, jnp.array(0), jnp.array(False))
            xf, _, done = jax.lax.while_loop(cond, body, init)
            resid_ok = jnp.linalg.norm(field(xf)) < tol * 10
            ok = done & resid_ok & jnp.all(jnp.isfinite(xf))
            return xf, ok

        roots, oks = jax.vmap(newton)(starts)  # (S, n), (S,) static shapes

        # masked-distance dedupe: keep a converged root iff no earlier converged root
        # lies within dedup_tol (static (S, S) matrix; entirely on-device).
        diff = roots[:, None, :] - roots[None, :, :]
        dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-30)
        s = roots.shape[0]
        earlier = jnp.tril(jnp.ones((s, s), dtype=bool), k=-1)
        is_dup = jnp.any((dist < dedup_tol) & earlier & oks[None, :], axis=1)
        keep = oks & jnp.logical_not(is_dup)
        eig_real = jax.vmap(lambda r: jnp.real(jnp.linalg.eigvals(jac(r))))(roots)
        return roots, keep, eig_real

    _ND_KERNEL_CACHE[key] = kernel
    return kernel


def _nd_fixed_points(
    circuit: Circuit,
    *,
    n_grid: int = 9,
    n_rand: int = 200,
    seed: int = 0,
    eig_tol: float = 1e-4,
) -> list[tuple[np.ndarray, str]]:
    """Enumerate + classify the steady states of an N-species circuit.

    Recipe (validated in a spike; endorsed by the saddle literature review): fixed-point
    enumeration via vmap'd multi-start Newton, masked-distance dedupe, then a Jacobian-
    eigenvalue index classification (index-1 saddle = exactly one eigenvalue with a
    positive real part). Energy-landscape path methods are wrong here (GRN ODEs are
    non-gradient). The vector field reuses ``Circuit.production`` (any topology).

    Numerics: run under a **local x64 context** (f32 Newton cancels catastrophically on
    an ill-conditioned Jacobian near a saddle-node), casting to f32 on return. The heavy
    Newton/dedupe/eigenvalue work is the **jitted, per-topology-cached** kernel
    (:func:`_nd_kernel`) — kinetics enter as a traced arg, so per-step cost is ~1 ms.
    Only the small unique set + its precomputed eigenvalues cross to numpy for eager
    labelling. Returns ``[(state, label), ...]`` sorted lexicographically for a
    deterministic slot order.
    """
    n = circuit.n_species
    per_dim = n_grid if n_grid**n <= 1000 else max(2, round(1000 ** (1.0 / n)))
    with enable_x64():
        base = circuit.base_params()
        hi = _nd_box_hi(circuit)
        g = np.linspace(1e-3, hi, per_dim)
        grid = np.stack([a.ravel() for a in np.meshgrid(*([g] * n))], axis=-1)
        rand = np.random.default_rng(seed).uniform(0.0, hi, size=(n_rand, n))
        starts = jnp.asarray(np.vstack([grid, rand]))
        roots_j, keep_j, eig_j = _nd_kernel(circuit)(starts, base)
        roots = np.asarray(roots_j)
        keep = np.asarray(keep_j)
        eig = np.asarray(eig_j)

    out: list[tuple[np.ndarray, str]] = []
    for r, re in zip(roots[keep], eig[keep], strict=True):
        n_pos = int(np.sum(re > eig_tol))
        n_neg = int(np.sum(re < -eig_tol))
        if n_neg == n:
            label = "stable"
        elif n_pos == 1 and n_neg == n - 1:
            label = "saddle-index1"
        elif n_pos == n:
            label = "source"
        else:
            label = "other"
        out.append((np.asarray(r, dtype=np.float32), label))
    out.sort(key=lambda sl: tuple(float(v) for v in sl[0]))
    return out
