"""The fit engine.

``fit(adata, circuit)`` (the public verb, completed once ``classify`` lands)
recovers the circuit's kinetic parameters from single-cell counts and attributes
each perturbation's mechanism. This module provides the core optimizer,
``fit_parameters`` — a minibatch optax loop that recovers a chosen set of circuit
parameters by matching the simulated single-cell distribution to the observed one.

**Differentiability.** ``params → per-cell θ → activity → Λ`` is fully
differentiable (θ via reparameterized draws). The final NB count *sample* is
discrete and non-differentiable, so the forward model used for the loss replaces
it with a **reparameterized, moment-matched continuous observation**
``μ + √(μ + φ·μ²)·ζ`` (``ζ ~ N(0,1)``) — a Gaussian relaxation of the NB matching
its first two moments. Gradients then flow cleanly to the parameters. (The
relaxation is loose at very low counts; adequate for a switch reporter.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from nudge.core.results import MechanismMap
from nudge.data.ingest import check_counts
from nudge.data.synthetic import _per_cell_params
from nudge.inference.losses import energy_distance
from nudge.mechanisms.readout import Readout

if TYPE_CHECKING:
    from nudge.core.circuit import Circuit

__all__ = ["FreeParam", "fit", "fit_parameters"]

#: A parameter to optimize: ``(scope, index, name)`` — e.g. ``("edge", 0, "K")``.
FreeParam = tuple[str, int, str]


def _param_value(circuit: Circuit, free: FreeParam) -> float:
    scope, index, name = free
    obj = circuit.edges[index] if scope == "edge" else circuit.species[index]
    return float(getattr(obj, name))


def _override(params: dict, free: FreeParam, value: Array) -> None:
    scope, index, name = free
    collection = "edges" if scope == "edge" else "species"
    params[collection][name] = params[collection][name].at[:, index].set(value)


def _simulate(
    circuit: Circuit,
    params: dict,
    readout: Readout,
    key: Array,
    *,
    dispersion: float,
    library_sigma: float,
) -> Array:
    """Reparameterized, moment-matched continuous observation (differentiable)."""
    n_cells = params["species"]["basal"].shape[0]
    x0 = jnp.zeros((n_cells, circuit.n_species))
    activity = circuit.solve_population(params, x0)
    expression = readout.expression(activity)
    k_lib, k_obs = jax.random.split(key)
    library = jnp.exp(library_sigma * jax.random.normal(k_lib, (n_cells, 1)))
    mean = library * expression
    nb_var = mean + dispersion * mean**2
    noise = jnp.sqrt(nb_var + 1e-8) * jax.random.normal(k_obs, mean.shape)
    return jnp.maximum(mean + noise, 0.0)


def fit_parameters(
    adata: Any,
    circuit: Circuit,
    free: list[FreeParam],
    *,
    condition: str = "WT",
    readout: Readout | None = None,
    n_cells: int = 256,
    steps: int = 300,
    learning_rate: float = 0.05,
    extrinsic_sigma: float = 0.3,
    dispersion: float = 0.1,
    library_sigma: float = 0.2,
    seed: int = 0,
) -> tuple[dict[FreeParam, float], list[float]]:
    """Recover the ``free`` circuit parameters from one condition's counts.

    Optimizes in log-space (parameters are positive) with Adam and a fresh
    minibatch of simulated + observed cells each step. Returns the recovered
    values and the loss history.
    """
    check_counts(adata)  # the bouncer — raw integer counts only
    if readout is None:
        readout = Readout.identity(circuit.n_species)

    mask = np.asarray(adata.obs["condition"] == condition)
    observed = jnp.asarray(np.asarray(adata.X, dtype=np.float32)[mask])
    log_theta = jnp.log(jnp.array([_param_value(circuit, f) for f in free]))

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(log_theta)

    def loss_fn(log_vals: Array, key: Array) -> Array:
        k_ext, k_pick, k_sim = jax.random.split(key, 3)
        params = _per_cell_params(circuit, k_ext, n_cells, extrinsic_sigma)
        for f, value in zip(free, jnp.exp(log_vals), strict=True):
            _override(params, f, value)
        sim = _simulate(
            circuit, params, readout, k_sim,
            dispersion=dispersion, library_sigma=library_sigma,
        )
        idx = jax.random.choice(k_pick, observed.shape[0], (n_cells,), replace=False)
        return energy_distance(sim, observed[idx])

    @jax.jit
    def step(
        log_vals: Array, state: optax.OptState, key: Array
    ) -> tuple[Array, optax.OptState, Array]:
        loss, grad = jax.value_and_grad(loss_fn)(log_vals, key)
        updates, state = optimizer.update(grad, state)
        new_log = jnp.asarray(optax.apply_updates(log_vals, updates))
        return new_log, state, loss

    key = jax.random.key(seed)
    history: list[float] = []
    for _ in range(steps):
        key, sub = jax.random.split(key)
        log_theta, opt_state, loss = step(log_theta, opt_state, sub)
        history.append(float(loss))

    recovered = {f: float(v) for f, v in zip(free, jnp.exp(log_theta), strict=True)}
    return recovered, history


def fit(adata: Any, circuit: Circuit) -> MechanismMap:
    """Fit ``circuit`` to ``adata`` (raw-count Perturb-seq) → a ``MechanismMap``.

    Completed once the abstention classifier (``inference.classify``) lands; the
    optimizer engine is ``fit_parameters``.
    """
    raise NotImplementedError("fit — completed with inference.classify (Phase 2)")
