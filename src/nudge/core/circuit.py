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
