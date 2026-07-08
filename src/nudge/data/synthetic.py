"""Synthetic Perturb-seq generator — the Tier-0 CI backbone.

``generate_synthetic_perturbseq(...)`` runs a NUDGE ``Circuit`` as a *generative*
model (``design/GENERATOR_DESIGN.md``): per-cell parameters (extrinsic noise) →
vmapped deterministic steady-state solve → ``Readout`` link → negative-binomial /
Poisson counts, returning an ``AnnData`` with the ground-truth parameters and
mechanism labels in ``.uns['ground_truth']``. Because the circuit *is* the
generator, ground truth is exact — the controllable, designed-in mode split is a
feature (the deterministic transfer-function route, Ochab-Marcinek & Tabaka 2010).

A perturbation moves one circuit parameter, and *which* parameter fixes the true
mechanism: ``K`` → threshold, ``n`` → gain, ``vmax`` → ceiling.

Tier 0 is inverse-crime-prone by construction; the honest robustness test is an
independent stochastic Tier-0.5 simulator (deferred). Level-3 misspecification is
likewise deferred to Phase 3.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import anndata as ad
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from nudge.core.circuit import Circuit, Params
from nudge.core.vocabulary import MechanismClass
from nudge.data.noise import sample_counts, sample_library_sizes
from nudge.mechanisms.readout import Readout

__all__ = ["PerturbationSpec", "generate_synthetic_perturbseq"]

# Which parameter a perturbation moves fixes the ground-truth mechanism.
_PARAM_MECHANISM = {
    "K": MechanismClass.THRESHOLD,
    "n": MechanismClass.GAIN,
    "vmax": MechanismClass.CEILING,
}

# Extrinsic (global, per-cell) noise scales expression-level knobs — NOT the
# switch *shape* params (K, n), which are a shared circuit property.
_EXTRINSIC_PARAMS = ("basal", "decay")

# realism_level → (dispersion, library_sigma). Extrinsic noise is separate
# (always on: it is the population model, not optional noise).
_NOISE_PRESETS: dict[int, tuple[float, float]] = {
    0: (0.0, 0.0),  # Poisson, fixed library size — the easy, exact-ish case
    1: (0.10, 0.20),  # + NB overdispersion + mild library variation
    2: (0.15, 0.35),  # + more library variation
    3: (0.15, 0.35),  # + misspecification (Phase 3; currently == level 2)
}


@dataclass(frozen=True)
class PerturbationSpec:
    """A ground-truth perturbation: multiply one circuit parameter by ``factor``.

    ``scope`` is ``"edge"`` or ``"species"``; ``index`` selects which; ``param`` is
    the parameter name. ``mechanism`` is derived from ``param``.
    """

    name: str
    scope: str
    index: int
    param: str
    factor: float

    @property
    def mechanism(self) -> MechanismClass:
        return _PARAM_MECHANISM.get(self.param, MechanismClass.NO_EFFECT)


def _per_cell_params(
    circuit: Circuit, key: Array, n_cells: int, extrinsic_sigma: float
) -> Params:
    """Broadcast the circuit's base params to ``n_cells`` with extrinsic noise."""
    base = circuit.base_params()

    def tile(a: Array) -> Array:
        return jnp.broadcast_to(a, (n_cells, *a.shape))

    params: Params = {
        "species": {k: tile(v) for k, v in base["species"].items()},
        "edges": {k: tile(v) for k, v in base["edges"].items()},
    }
    if extrinsic_sigma > 0:
        sub_keys = jax.random.split(key, len(_EXTRINSIC_PARAMS))
        for sub_key, name in zip(sub_keys, _EXTRINSIC_PARAMS, strict=True):
            factor = jnp.exp(extrinsic_sigma * jax.random.normal(sub_key, (n_cells, 1)))
            params["species"][name] = params["species"][name] * factor
    return params


def _apply_perturbation(params: Params, pert: PerturbationSpec) -> Params:
    """Return a copy of ``params`` with the targeted parameter scaled by ``factor``."""
    collection = "edges" if pert.scope == "edge" else "species"
    scaled = params[collection][pert.param].at[:, pert.index].multiply(pert.factor)
    out: Params = {k: dict(v) for k, v in params.items()}
    out[collection][pert.param] = scaled
    return out


def generate_synthetic_perturbseq(
    circuit: Circuit,
    perturbations: Sequence[PerturbationSpec] = (),
    readout: Readout | None = None,
    *,
    n_cells_per_condition: int = 1000,
    realism_level: int = 1,
    extrinsic_sigma: float = 0.3,
    seed: int = 0,
    gene_names: Sequence[str] | None = None,
) -> ad.AnnData:
    """Generate a Tier-0 synthetic Perturb-seq ``AnnData`` (ground truth in ``.uns``).

    Conditions are a WT control plus one per ``PerturbationSpec``. Each cell's
    ``.obs`` carries its ``condition`` and ``true_mechanism``; ``.uns['ground_truth']``
    carries the per-condition specs and the circuit's true parameters.
    """
    if readout is None:
        readout = Readout.identity(circuit.n_species)
    dispersion, library_sigma = _NOISE_PRESETS[realism_level]
    n_cells = n_cells_per_condition
    conditions: list[PerturbationSpec | None] = [None, *perturbations]

    key = jax.random.key(seed)
    count_blocks: list[np.ndarray] = []
    obs_condition: list[str] = []
    obs_mechanism: list[str] = []
    ground_truth_conditions: list[dict[str, object]] = []

    for cond in conditions:
        key, k_params, k_lib, k_counts = jax.random.split(key, 4)
        params = _per_cell_params(circuit, k_params, n_cells, extrinsic_sigma)
        if cond is not None:
            params = _apply_perturbation(params, cond)
        x0 = jnp.zeros((n_cells, circuit.n_species))
        activity = circuit.solve_population(params, x0)
        expression = readout.expression(activity)
        library = sample_library_sizes(k_lib, n_cells, log_sd=library_sigma)
        counts = sample_counts(k_counts, expression, library, dispersion=dispersion)

        count_blocks.append(np.asarray(counts))
        name = "WT" if cond is None else cond.name
        mechanism = MechanismClass.NO_EFFECT if cond is None else cond.mechanism
        obs_condition.extend([name] * n_cells)
        obs_mechanism.extend([mechanism.value] * n_cells)
        if cond is None:
            ground_truth_conditions.append({"name": "WT", "mechanism": mechanism.value})
        else:
            ground_truth_conditions.append(
                {
                    "name": cond.name,
                    "mechanism": cond.mechanism.value,
                    "scope": cond.scope,
                    "index": cond.index,
                    "param": cond.param,
                    "factor": cond.factor,
                }
            )

    counts_matrix = np.concatenate(count_blocks, axis=0)
    n_genes = counts_matrix.shape[1]
    if gene_names is None:
        gene_names = (
            list(circuit.names)
            if n_genes == circuit.n_species
            else [f"gene_{i}" for i in range(n_genes)]
        )

    obs = pd.DataFrame(
        {"condition": obs_condition, "true_mechanism": obs_mechanism},
        index=[f"cell_{i}" for i in range(counts_matrix.shape[0])],
    )
    var = pd.DataFrame(index=list(gene_names))
    adata = ad.AnnData(X=counts_matrix, obs=obs, var=var)
    base = circuit.base_params()
    adata.uns["ground_truth"] = {
        "conditions": ground_truth_conditions,
        "species": list(circuit.names),
        "seed": int(seed),
        "realism_level": int(realism_level),
        "true_params": {
            group: {k: np.asarray(v) for k, v in sub.items()}
            for group, sub in base.items()
        },
    }
    return adata
