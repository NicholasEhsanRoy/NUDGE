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

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from nudge.core.circuit import Circuit
from nudge.core.results import MechanismCall, MechanismMap
from nudge.core.vocabulary import POSITIVE_CLASSES, MechanismClass
from nudge.data.ingest import check_counts
from nudge.data.synthetic import _per_cell_params
from nudge.inference.classify import decide, decide_with_transition, switch_detected
from nudge.inference.losses import energy_distance, energy_distance_weighted
from nudge.mechanisms.readout import Readout

__all__ = [
    "FreeParam",
    "fit",
    "fit_multibasin",
    "fit_multibasin_parameters",
    "fit_parameters",
    "fit_transition_parameters",
]

#: A parameter to optimize: ``(scope, index, name)`` — e.g. ``("edge", 0, "K")``.
FreeParam = tuple[str, int, str]


def _apply_transform(x: Array, transform: str) -> Array:
    """Optional shape-sensitizing transform applied to counts before the distance."""
    if transform == "log1p":
        return jnp.log1p(x)
    if transform == "none":
        return x
    raise ValueError(f"unknown transform {transform!r}")


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
    transform: str = "log1p",
    seed: int = 0,
) -> tuple[dict[FreeParam, float], list[float]]:
    """Recover the ``free`` circuit parameters from one condition's counts.

    Optimizes in log-space (parameters are positive) with Adam and a fresh
    minibatch of simulated + observed cells each step. ``transform="log1p"``
    compares distributions in log space, which is scale-robust and far more
    sensitive to distribution *shape* (bimodality) than raw counts. Returns the
    recovered values and the loss history.
    """
    check_counts(adata)  # the bouncer — raw integer counts only
    if readout is None:
        readout = Readout.identity(circuit.n_species)

    mask = np.asarray(adata.obs["condition"] == condition)
    raw = jnp.asarray(np.asarray(adata.X, dtype=np.float32)[mask])
    observed = _apply_transform(raw, transform)
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
        return energy_distance(_apply_transform(sim, transform), observed[idx])

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


def _simulate_basin(
    circuit: Circuit,
    params: dict,
    readout: Readout,
    key: Array,
    x0: Array,
    *,
    dispersion: float,
    library_sigma: float,
) -> Array:
    """Moment-matched continuous observation, solved from a given ``x0`` (basin seed).

    Identical to ``_simulate`` but with the initial condition exposed, so the
    multi-basin model can solve the same population from the LOW (``x0 = 0``) and
    HIGH (``x0 = high_ic``) basins of a bistable circuit.
    """
    activity = circuit.solve_population(params, x0)
    expression = readout.expression(activity)
    n_cells = params["species"]["basal"].shape[0]
    k_lib, k_obs = jax.random.split(key)
    library = jnp.exp(library_sigma * jax.random.normal(k_lib, (n_cells, 1)))
    mean = library * expression
    nb_var = mean + dispersion * mean**2
    noise = jnp.sqrt(nb_var + 1e-8) * jax.random.normal(k_obs, mean.shape)
    return jnp.maximum(mean + noise, 0.0)


def fit_multibasin_parameters(
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
    transform: str = "log1p",
    high_ic: float = 10.0,
    p_init: float = 0.5,
    fixed_p: float | None = None,
    seed: int = 0,
) -> tuple[dict[FreeParam, float], float, list[float]]:
    """Like ``fit_parameters`` but with a **basin-occupancy latent** ``p``.

    The forward model solves the population from BOTH basins — low (``x0 = 0``) and
    high (``x0 = high_ic``) — and forms the mixture ``(1−p)·low + p·high`` as a
    weighted empirical distribution (the ``p``-weighted energy distance). This lets
    the deterministic fit *represent* an emergent-bistable population that a single
    ``x0 = 0`` solve cannot (the Tier-0.5 gap; an autonomous R&D spike showed ``p`` is
    recoverable because the modes are pinned to the ODE fixed points). ``p`` is fit
    jointly with the kinetics via an unconstrained logit ``p = sigmoid(p_raw)``.

    ``fixed_p`` pins the occupancy (skips optimizing it) — used for attribution, where
    letting every restricted fit re-optimize ``p`` lets the occupancy latent absorb the
    shape signal that should discriminate the mechanisms (``FINDINGS.md`` §T0.5-4). Pin
    ``p`` to a per-condition estimate so the kinetic params compete on residual shape.

    Returns the recovered kinetics, the recovered ``p`` (fraction in the HIGH basin),
    and the loss history.
    """
    check_counts(adata)
    if readout is None:
        readout = Readout.identity(circuit.n_species)

    mask = np.asarray(adata.obs["condition"] == condition)
    raw = jnp.asarray(np.asarray(adata.X, dtype=np.float32)[mask])
    observed = _apply_transform(raw, transform)
    # Optimize [log kinetics..., p_raw] jointly. p_raw is linear (a logit), kinetics
    # log-space (positive); Adam's per-coordinate scaling handles the mixed geometry.
    # With fixed_p, p is held constant and only the kinetics are optimized.
    fit_p = fixed_p is None
    log_kin = jnp.log(jnp.array([_param_value(circuit, f) for f in free]))
    p_raw0 = jnp.asarray(float(np.log(p_init / (1.0 - p_init))))
    theta = jnp.concatenate([log_kin, p_raw0[None]]) if fit_p else log_kin
    p_const = jnp.asarray(0.5 if fit_p else float(fixed_p))

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(theta)
    n_free = len(free)

    def loss_fn(vec: Array, key: Array) -> Array:
        log_vals = vec[:n_free]
        p = jax.nn.sigmoid(vec[n_free]) if fit_p else p_const
        k_ext, k_pick, k_low, k_high = jax.random.split(key, 4)
        params = _per_cell_params(circuit, k_ext, n_cells, extrinsic_sigma)
        for f, value in zip(free, jnp.exp(log_vals), strict=True):
            _override(params, f, value)
        x0_low = jnp.zeros((n_cells, circuit.n_species))
        x0_high = jnp.full((n_cells, circuit.n_species), high_ic)
        sim_low = _simulate_basin(
            circuit, params, readout, k_low, x0_low,
            dispersion=dispersion, library_sigma=library_sigma,
        )
        sim_high = _simulate_basin(
            circuit, params, readout, k_high, x0_high,
            dispersion=dispersion, library_sigma=library_sigma,
        )
        sim = jnp.concatenate(
            [
                _apply_transform(sim_low, transform),
                _apply_transform(sim_high, transform),
            ],
            axis=0,
        )
        weights = jnp.concatenate(
            [jnp.full((n_cells,), 1.0 - p), jnp.full((n_cells,), p)]
        )
        idx = jax.random.choice(k_pick, observed.shape[0], (n_cells,), replace=False)
        return energy_distance_weighted(sim, weights, observed[idx])

    @jax.jit
    def step(
        vec: Array, state: optax.OptState, key: Array
    ) -> tuple[Array, optax.OptState, Array]:
        loss, grad = jax.value_and_grad(loss_fn)(vec, key)
        updates, state = optimizer.update(grad, state)
        new_vec = jnp.asarray(optax.apply_updates(vec, updates))
        return new_vec, state, loss

    key = jax.random.key(seed)
    history: list[float] = []
    for _ in range(steps):
        key, sub = jax.random.split(key)
        theta, opt_state, loss = step(theta, opt_state, sub)
        history.append(float(loss))

    recovered = {
        f: float(v) for f, v in zip(free, jnp.exp(theta[:n_free]), strict=True)
    }
    p_hat = float(jax.nn.sigmoid(theta[n_free])) if fit_p else float(fixed_p)
    return recovered, p_hat, history


def _sim_transition(
    readout: Readout,
    key: Array,
    center: Array,
    log_width: Array,
    n_cells: int,
    n_species: int,
    *,
    dispersion: float,
    library_sigma: float,
) -> Array:
    """Transition-mode sample: activity spread lognormally about the saddle ``center``.

    ``center`` is an ``(n_species,)`` vector (length-1 for a 1-species switch; the
    saddle of an N-node toggle otherwise); it broadcasts over cells. A strictly-positive
    lognormal width (``exp(log_width)``) means there is no covariance matrix to collapse
    near the bifurcation (the FM1 NaN risk). The centre is a per-step stop-grad const.
    """
    k_a, k_lib, k_obs = jax.random.split(key, 3)
    width = jnp.exp(log_width)
    act = center * jnp.exp(width * jax.random.normal(k_a, (n_cells, n_species)))
    expression = readout.expression(act)
    library = jnp.exp(library_sigma * jax.random.normal(k_lib, (n_cells, 1)))
    mean = library * expression
    nb_var = mean + dispersion * mean**2
    noise = jnp.sqrt(nb_var + 1e-8) * jax.random.normal(k_obs, mean.shape)
    return jnp.maximum(mean + noise, 0.0)


def fit_transition_parameters(
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
    transform: str = "log1p",
    high_ic: float = 10.0,
    width_init: float = 0.35,
    seed: int = 0,
) -> tuple[dict[FreeParam, float], float, list[float]]:
    """Three-mode fit: LOW + HIGH basins + a **TRANSITION mode at the ODE saddle**.

    Extends the two-basin mixture with a third component centred on the unstable
    intermediate fixed point (``circuit.transition_state``), recomputed from the
    current kinetics each step and passed in as a stop-gradient constant. Mixture
    weights are a softmax over three logits; the transition width is fitted in
    log-space. Returns the recovered kinetics, the **effective transition weight
    ``w_trans``** (the gain-gate probe), and the loss history.

    Graceful at the bifurcation (FM1): when the circuit is monostable
    (``transition_state`` is ``None``) the transition weight is **masked to zero** and
    the mix reduces to the two basins — no fabricated saddle ever enters a gradient.
    When ``circuit`` has no 1-D saddle at all (N-species; ``fixed_points`` is ``None``),
    ``w_trans`` stays ~0 throughout and the gain gate abstains (FM2).
    """
    check_counts(adata)
    if readout is None:
        readout = Readout.identity(circuit.n_species)

    mask = np.asarray(adata.obs["condition"] == condition)
    raw = jnp.asarray(np.asarray(adata.X, dtype=np.float32)[mask])
    observed = _apply_transform(raw, transform)
    n_species = circuit.n_species
    n_free = len(free)

    log_kin0 = jnp.log(jnp.array([_param_value(circuit, f) for f in free]))
    logits0 = jnp.zeros(3)  # equal thirds
    log_width0 = jnp.asarray(float(np.log(width_init)))
    theta = jnp.concatenate([log_kin0, logits0, log_width0[None]])

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(theta)

    def loss_fn(vec: Array, key: Array, center: Array, valid: Array) -> Array:
        # The saddle center/valid are per-step constants from the eager finder — never
        # differentiate through the root-finder (XLA would trace its backward pass).
        center = jax.lax.stop_gradient(center)
        valid = jax.lax.stop_gradient(valid)
        log_vals = vec[:n_free]
        logits = vec[n_free : n_free + 3]
        log_width = vec[n_free + 3]
        w = jax.nn.softmax(logits)
        # Mask the transition weight to 0 when no saddle exists, then renormalize.
        w_trans = w[2] * valid
        total = w[0] + w[1] + w_trans + 1e-8
        k_ext, k_pick, k_low, k_high, k_tr = jax.random.split(key, 5)
        params = _per_cell_params(circuit, k_ext, n_cells, extrinsic_sigma)
        for f, value in zip(free, jnp.exp(log_vals), strict=True):
            _override(params, f, value)
        x0_low = jnp.zeros((n_cells, n_species))
        x0_high = jnp.full((n_cells, n_species), high_ic)
        sim_low = _simulate_basin(
            circuit, params, readout, k_low, x0_low,
            dispersion=dispersion, library_sigma=library_sigma,
        )
        sim_high = _simulate_basin(
            circuit, params, readout, k_high, x0_high,
            dispersion=dispersion, library_sigma=library_sigma,
        )
        sim_tr = _sim_transition(
            readout, k_tr, center, log_width, n_cells, n_species,
            dispersion=dispersion, library_sigma=library_sigma,
        )
        sim = jnp.concatenate(
            [
                _apply_transform(sim_low, transform),
                _apply_transform(sim_high, transform),
                _apply_transform(sim_tr, transform),
            ],
            axis=0,
        )
        weights = jnp.concatenate(
            [
                jnp.full((n_cells,), w[0] / total),
                jnp.full((n_cells,), w[1] / total),
                jnp.full((n_cells,), w_trans / total),
            ]
        )
        idx = jax.random.choice(k_pick, observed.shape[0], (n_cells,), replace=False)
        return energy_distance_weighted(sim, weights, observed[idx])

    @jax.jit
    def step(
        vec: Array, state: optax.OptState, key: Array, center: Array, valid: Array
    ) -> tuple[Array, optax.OptState, Array]:
        loss, grad = jax.value_and_grad(loss_fn)(vec, key, center, valid)
        updates, state = optimizer.update(grad, state)
        new_vec = jnp.asarray(optax.apply_updates(vec, updates))
        return new_vec, state, loss

    def _saddle(vec: Array) -> tuple[Array, Array]:
        # Recompute the saddle from the current concrete kinetics per step (eager, then
        # fed into the jitted step as a stop-gradient constant). `transition_state()`
        # returns an (n_species,) vector (length-1 for a 1-species switch) or None.
        vals = np.exp(np.asarray(vec[:n_free]))
        cur = _updated_circuit(
            circuit, {f: float(v) for f, v in zip(free, vals, strict=True)}
        )
        state = cur.transition_state()
        if state is None:
            # monostable/no saddle → safe finite centre, transition masked off
            return jnp.ones((n_species,)), jnp.asarray(0.0)
        return jnp.asarray(state), jnp.asarray(1.0)

    key = jax.random.key(seed)
    history: list[float] = []
    for _ in range(steps):
        key, sub = jax.random.split(key)
        center, valid = _saddle(theta)
        theta, opt_state, loss = step(theta, opt_state, sub, center, valid)
        history.append(float(loss))

    recovered = {
        f: float(v) for f, v in zip(free, jnp.exp(theta[:n_free]), strict=True)
    }
    w_final = np.asarray(jax.nn.softmax(theta[n_free : n_free + 3]))
    _, valid_final = _saddle(theta)
    w_trans_eff = float(w_final[2]) * float(valid_final)
    total = float(w_final[0]) + float(w_final[1]) + w_trans_eff + 1e-8
    return recovered, w_trans_eff / total, history


def _condition_counts(adata: Any, condition: str) -> Array:
    mask = np.asarray(adata.obs["condition"] == condition)
    return jnp.asarray(np.asarray(adata.X, dtype=np.float32)[mask])


def _updated_circuit(circuit: Circuit, updates: dict[FreeParam, float]) -> Circuit:
    """Return a copy of ``circuit`` with the given parameters replaced."""
    species = list(circuit.species)
    edges = list(circuit.edges)
    for (scope, index, name), value in updates.items():
        if scope == "edge":
            edges[index] = replace(edges[index], **{name: value})
        else:
            species[index] = replace(species[index], **{name: value})
    return Circuit(species, edges)


def _loss_stats(history: list[float], tail: int = 40) -> tuple[float, float]:
    """Converged loss (mean) and its noise floor (std) from the loss-history tail."""
    arr = np.asarray(history[-tail:])
    return float(arr.mean()), float(arr.std())


def _subsample(counts: Array, n: int, key: Array) -> Array:
    take = min(n, int(counts.shape[0]))
    idx = jax.random.choice(key, counts.shape[0], (take,), replace=False)
    return counts[idx]


def _self_distance(
    counts: Array, n: int, key: Array, reps: int = 10
) -> tuple[float, float]:
    """Irreducible finite-sample loss floor (log1p space) and its std, from bootstrap.

    A perfect fit's loss ≈ the energy distance between two ``n``-cell subsamples of
    the same distribution; its std is the loss noise floor the gates compare against.
    """
    dists = []
    for _ in range(reps):
        key, k1, k2 = jax.random.split(key, 3)
        a = _apply_transform(_subsample(counts, n, k1), "log1p")
        b = _apply_transform(_subsample(counts, n, k2), "log1p")
        dists.append(float(energy_distance(a, b)))
    arr = np.asarray(dists)
    return float(arr.mean()), float(arr.std())


def _log1p_distance(a: Array, b: Array, n: int, key: Array) -> float:
    k1, k2 = jax.random.split(key)
    return float(
        energy_distance(
            _apply_transform(_subsample(a, n, k1), "log1p"),
            _apply_transform(_subsample(b, n, k2), "log1p"),
        )
    )


def fit(
    adata: Any,
    circuit: Circuit,
    *,
    target_edge: int = 0,
    wt_condition: str = "WT",
    conditions: list[str] | None = None,
    readout: Readout | None = None,
    steps: int = 300,
    n_cells: int = 256,
    margin_k: float = 1.7,
    off_model_k: float = 5.0,
    seed: int = 0,
) -> MechanismMap:
    """Fit ``circuit`` to ``adata`` (raw-count Perturb-seq) → a ``MechanismMap``.

    (1) Fit the WT mechanistic circuit and the WT linear baseline; the
    linear-baseline parsimony gate (``switch_detected``) decides whether a switch
    exists at all — if not, every perturbation is ``off-model``. (2) Given a switch,
    for each perturbation fit the three restricted mechanistic models (free K / n /
    vmax of ``target_edge``) and route them through ``classify.decide``. The gate
    noise floor is the WT self-distance bootstrap (the irreducible loss scale).

    ``margin_k`` scales that noise floor. The default 1.7 is calibrated
    (``scripts/vv/``): across 300 synthetic linear datasets it holds the
    false-positive rate < 2%, and across 120 switch datasets the misclassification
    rate is 0% (the tool abstains, never calls the wrong mechanism) — at any
    ``margin_k``. Lower it toward 1.0 for more sensitivity (≈88% correct at ≈8%
    false positives); raise it for stricter specificity.
    """
    check_counts(adata)
    if readout is None:
        readout = Readout.identity(circuit.n_species)
    labels = list(dict.fromkeys(np.asarray(adata.obs["condition"]).tolist()))
    if conditions is None:
        conditions = [c for c in labels if c != wt_condition]

    common: dict[str, Any] = {"readout": readout, "steps": steps, "n_cells": n_cells}
    wt_counts = _condition_counts(adata, wt_condition)
    floor_key = jax.random.key(seed + 50)
    floor_mean, floor_std = _self_distance(wt_counts, n_cells, floor_key)
    noise_margin = margin_k * floor_std
    effect_margin = floor_mean + 3.0 * floor_std
    off_model_loss = off_model_k * floor_mean

    # 1. WT: mechanistic (free K, n, vmax) vs linear baseline (free weight).
    wt_free: list[FreeParam] = [("edge", target_edge, p) for p in ("K", "n", "vmax")]
    wt_values, wt_mech_history = fit_parameters(
        adata, circuit, wt_free, condition=wt_condition, seed=seed, **common
    )
    wt_circuit = _updated_circuit(circuit, wt_values)
    _, wt_lin_history = fit_parameters(
        adata, wt_circuit.linear_baseline(), [("edge", target_edge, "weight")],
        condition=wt_condition, seed=seed + 1, **common,
    )
    wt_mech_loss, _ = _loss_stats(wt_mech_history)
    wt_lin_loss, _ = _loss_stats(wt_lin_history)

    # 2. The linear-baseline parsimony gate — is there a switch to attribute?
    if not switch_detected(wt_mech_loss, wt_lin_loss, noise_margin=noise_margin):
        calls = [
            MechanismCall(
                perturbation=c,
                mechanism=MechanismClass.OFF_MODEL,
                confidence=0.0,
                rationale="no switch detected — linear baseline explains WT",
            )
            for c in conditions
        ]
        return MechanismMap(
            calls=calls,
            beats_linear_baseline=False,
            provenance={"conditions": ",".join(conditions), "seed": str(seed)},
        )

    # 3. A switch exists — attribute each perturbation via restricted fits.
    key = jax.random.key(seed + 100)
    calls = []
    for i, condition in enumerate(conditions):
        key, k_dist = jax.random.split(key)
        perturbed = _condition_counts(adata, condition)
        wt_distance = _log1p_distance(perturbed, wt_counts, n_cells, k_dist)
        param_losses: dict[str, float] = {}
        for j, param in enumerate(("K", "n", "vmax")):
            _, history = fit_parameters(
                adata, wt_circuit, [("edge", target_edge, param)],
                condition=condition, seed=seed + 10 * (i + 1) + j, **common,
            )
            param_losses[param] = _loss_stats(history)[0]
        calls.append(
            decide(
                condition, param_losses, wt_distance,
                noise_margin=noise_margin,
                effect_margin=effect_margin,
                off_model_loss=off_model_loss,
            )
        )

    beats = any(c.mechanism in POSITIVE_CLASSES for c in calls)
    return MechanismMap(
        calls=calls,
        beats_linear_baseline=beats,
        provenance={"conditions": ",".join(conditions), "seed": str(seed)},
    )


def fit_multibasin(
    adata: Any,
    circuit: Circuit,
    *,
    target_edge: int = 0,
    wt_condition: str = "WT",
    conditions: list[str] | None = None,
    readout: Readout | None = None,
    steps: int = 300,
    n_cells: int = 256,
    margin_k: float = 1.7,
    off_model_k: float = 5.0,
    high_ic: float = 10.0,
    transition_mode: bool = False,
    gain_wtrans_tau: float = 0.5,
    seed: int = 0,
) -> MechanismMap:
    """``fit`` with a **basin-occupancy latent** — for emergent-bistable circuits.

    Orchestration mirrors :func:`fit` (WT parsimony gate → per-perturbation restricted
    fits → classify), but the mechanistic fits solve from BOTH basins (low ``x0 = 0``
    and high ``x0 = high_ic``), so the deterministic model can *represent* an emergent-
    bistable population a single ``x0 = 0`` solve cannot. The linear baseline stays
    single-basin (a linear circuit is monostable). Built alongside :func:`fit`, which
    is unchanged.

    **``transition_mode`` — the saddle gain gate (the fail-safe fix).** With the plain
    two-basin mixture, attribution *degenerates*: a gain reduction makes the switch
    graded — intermediate cells the two modes can't hold — so the model conflates it
    with a ceiling reduction and can be confidently wrong (``FINDINGS.md`` §T0.5-4).
    ``transition_mode=True`` adds a third mixture mode at the ODE's unstable saddle
    (:func:`fit_transition_parameters`, via ``circuit.transition_state``): the
    transition weight a restricted free-``n`` fit is *forced* to spend is a clean,
    seed-robust gain detector (``w_trans`` ≈ 0.9 gain vs ≈ 0.01 else), gated in
    ``classify.decide_with_transition``. This **fixes the gain/ceiling degeneracy for
    1-species self-activation switches while staying never-wrong** (§T0.5-5). It is
    isolated by ``circuit.n_species == 1``: N-species circuits have no saddle finder,
    so the gate defers to honest abstention (FM2). Without ``transition_mode`` the
    two-basin path is EXPERIMENTAL / not-fail-safe — prefer ``transition_mode=True``
    for 1-species emergent-bistable workloads, or single-basin :func:`fit` otherwise.
    ``gain_wtrans_tau`` (default 0.5) sits in a wide verified margin (0.12 ↔ 0.87).
    """
    check_counts(adata)
    if readout is None:
        readout = Readout.identity(circuit.n_species)
    labels = list(dict.fromkeys(np.asarray(adata.obs["condition"]).tolist()))
    if conditions is None:
        conditions = [c for c in labels if c != wt_condition]

    mb: dict[str, Any] = {
        "readout": readout, "steps": steps, "n_cells": n_cells, "high_ic": high_ic,
    }
    lin: dict[str, Any] = {"readout": readout, "steps": steps, "n_cells": n_cells}
    wt_counts = _condition_counts(adata, wt_condition)
    floor_key = jax.random.key(seed + 50)
    floor_mean, floor_std = _self_distance(wt_counts, n_cells, floor_key)
    noise_margin = margin_k * floor_std
    effect_margin = floor_mean + 3.0 * floor_std
    off_model_loss = off_model_k * floor_mean

    # 1. WT: multi-basin mechanistic (free K, n, vmax + p) vs single-basin linear.
    wt_free: list[FreeParam] = [("edge", target_edge, p) for p in ("K", "n", "vmax")]
    wt_values, _wt_p, wt_mech_history = fit_multibasin_parameters(
        adata, circuit, wt_free, condition=wt_condition, seed=seed, **mb
    )
    wt_circuit = _updated_circuit(circuit, wt_values)
    _, wt_lin_history = fit_parameters(
        adata, wt_circuit.linear_baseline(), [("edge", target_edge, "weight")],
        condition=wt_condition, seed=seed + 1, **lin,
    )
    wt_mech_loss, _ = _loss_stats(wt_mech_history)
    wt_lin_loss, _ = _loss_stats(wt_lin_history)

    # 2. The linear-baseline parsimony gate — is there a switch to attribute?
    if not switch_detected(wt_mech_loss, wt_lin_loss, noise_margin=noise_margin):
        calls = [
            MechanismCall(
                perturbation=c,
                mechanism=MechanismClass.OFF_MODEL,
                confidence=0.0,
                rationale="no switch detected — linear baseline explains WT",
            )
            for c in conditions
        ]
        return MechanismMap(
            calls=calls,
            beats_linear_baseline=False,
            provenance={
                "conditions": ",".join(conditions), "seed": str(seed),
                "model": "multibasin",
            },
        )

    # 3. A switch exists — attribute each perturbation via restricted fits.
    mb_t: dict[str, Any] = {
        "readout": readout, "steps": steps, "n_cells": n_cells, "high_ic": high_ic,
    }
    key = jax.random.key(seed + 100)
    calls = []
    for i, condition in enumerate(conditions):
        key, k_dist = jax.random.split(key)
        perturbed = _condition_counts(adata, condition)
        wt_distance = _log1p_distance(perturbed, wt_counts, n_cells, k_dist)
        param_losses: dict[str, float] = {}
        if transition_mode:
            # Three-mode restricted fits: each frees one kinetic + the 3 mixture
            # weights. The free-n fit's transition weight is the gain-gate probe.
            # Start from the NOMINAL circuit, not the WT-recovered one: the multibasin
            # WT fit distorts the kinetics (n inflates as the 2 basins compensate),
            # which shifts the saddle and corrupts the w_trans signal. The saddle
            # reference must be the un-distorted nominal switch.
            w_trans_n: float | None = None
            for j, param in enumerate(("K", "n", "vmax")):
                _, w_tr, history = fit_transition_parameters(
                    adata, circuit, [("edge", target_edge, param)],
                    condition=condition, seed=seed + 10 * (i + 1) + j, **mb_t,
                )
                param_losses[param] = _loss_stats(history)[0]
                if param == "n":
                    w_trans_n = w_tr
            calls.append(
                decide_with_transition(
                    condition, param_losses, wt_distance,
                    noise_margin=noise_margin,
                    effect_margin=effect_margin,
                    off_model_loss=off_model_loss,
                    transition_weight=w_trans_n,
                    n_species=circuit.n_species,
                    gain_wtrans_tau=gain_wtrans_tau,
                )
            )
            continue
        # Two-basin path: estimate occupancy p* once (all kinetics free), then PIN it
        # for the restricted fits so kinetics discriminate on residual shape rather than
        # re-absorbing the occupancy change (the T0.5-4 degeneracy — EXPERIMENTAL).
        _, p_star, _ = fit_multibasin_parameters(
            adata, wt_circuit, wt_free,
            condition=condition, seed=seed + 10 * (i + 1), **mb,
        )
        for j, param in enumerate(("K", "n", "vmax")):
            _, _p, history = fit_multibasin_parameters(
                adata, wt_circuit, [("edge", target_edge, param)],
                condition=condition, seed=seed + 10 * (i + 1) + j + 1,
                fixed_p=p_star, **mb,
            )
            param_losses[param] = _loss_stats(history)[0]
        calls.append(
            decide(
                condition, param_losses, wt_distance,
                noise_margin=noise_margin,
                effect_margin=effect_margin,
                off_model_loss=off_model_loss,
            )
        )

    beats = any(c.mechanism in POSITIVE_CLASSES for c in calls)
    return MechanismMap(
        calls=calls,
        beats_linear_baseline=beats,
        provenance={
            "conditions": ",".join(conditions), "seed": str(seed),
            "model": "multibasin",
        },
    )
