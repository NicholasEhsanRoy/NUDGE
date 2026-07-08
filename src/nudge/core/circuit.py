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

from collections.abc import Sequence
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

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

    def fixed_points(self) -> list[float] | None:
        """Steady-state activity fixed points — **only** for a 1-species self-switch.

        Decouples the topology-specific saddle math from the fit (the fit's transition
        mode needs the intermediate/unstable fixed point, but should not itself know
        how to find one — see ``design/STATE.md`` §6). For a single self-activating gene
        this returns the sorted real roots of ``basal + vmax·x^n/(K^n+x^n) − decay·x``:
        ``[low]`` when monostable, ``[low, saddle, high]`` when bistable. For any other
        topology (N-species, non-self, non-Hill) it returns ``None`` — there is no
        general N-D saddle finder yet, so the caller falls back to safe abstention
        rather than fabricating a fixed point. Never raises (returns ``None`` on any
        numerical failure), so it is safe to call inside an optimizer step.
        """
        if self.n_species != 1 or self.n_edges != 1:
            return None
        e, s = self.edges[0], self.species[0]
        if e.source != 0 or e.target != 0 or e.effect != "hill_activation":
            return None
        try:
            return _self_activation_roots(
                basal=s.basal, decay=s.decay, K=e.K, n=e.n, vmax=e.vmax
            )
        except Exception:
            return None

    def transition_state(self) -> float | None:
        """The intermediate (unstable saddle) activity when bistable, else ``None``.

        ``fixed_points()[1]`` iff there are exactly three roots. This is where graded
        cells pile up when a switch loses cooperativity (a gain reduction), so it is the
        centre of the fit's transition mixture mode. ``None`` (monostable or N-species)
        means "no transition mode" — the fit collapses gracefully and the gain gate
        abstains.
        """
        roots = self.fixed_points()
        if roots is None or len(roots) != 3:
            return None
        return roots[1]


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
