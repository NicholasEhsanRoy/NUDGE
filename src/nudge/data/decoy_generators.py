"""Generators for decoy-battery cases whose bimodality is *not* a switch.

Each returns raw-count AnnData a naive bimodality detector would call a hit; the
battery (``data/decoys.py`` + ``tests/decoys/``) asserts NUDGE declines. These are
simpler than the stochastic simulators — direct sampling through the same negative-
binomial observation layer (``data/noise.py``) — because the *point* is that no
dynamical switch is present at all.
"""

from __future__ import annotations

import anndata as ad
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef
from nudge.core.vocabulary import MechanismClass
from nudge.data.noise import sample_counts, sample_library_sizes
from nudge.data.synthetic import _per_cell_params

__all__ = [
    "generate_dropout_decoy",
    "generate_mixture_decoy",
    "generate_readout_nonlinearity_decoy",
]

# The identity-readout map NUDGE's fit assumes: Λ = base + scale·activity.
_SCALE, _BASE = 5.0, 0.2


def _assemble(count_blocks: list[np.ndarray], obs_condition: list[str]) -> ad.AnnData:
    counts = np.concatenate(count_blocks, axis=0)
    obs = pd.DataFrame(
        {
            "condition": obs_condition,
            "true_mechanism": [MechanismClass.OFF_MODEL.value] * len(obs_condition),
        },
        index=pd.Index([f"cell_{i}" for i in range(counts.shape[0])]),
    )
    var = pd.DataFrame(index=pd.Index(["SW"]))
    return ad.AnnData(X=counts, obs=obs, var=var)


def generate_mixture_decoy(
    *,
    n_cells_per_condition: int = 1000,
    low: float = 0.05,
    high: float = 2.0,
    p_high_wt: float = 0.5,
    p_high_pert: float = 0.3,
    dispersion: float = 0.1,
    library_sigma: float = 0.2,
    seed: int = 0,
) -> ad.AnnData:
    """Two-population (cell-type / doublet) mixture — bimodality with no dynamics.

    A fraction ``p_high`` of cells sit at a fixed HIGH expression level and the rest at
    a fixed LOW level (each monostable, constant). The aggregate is bimodal (an OFF-like
    spike + an ON-like mode) that a switch-detector would misread — but it is a *static
    population mixture*, not an ultrasensitive response. NUDGE must decline (off-model).
    """
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)
    blocks: list[np.ndarray] = []
    obs_condition: list[str] = []
    for name, p_high in (("WT", p_high_wt), ("pert", p_high_pert)):
        n = n_cells_per_condition
        is_high = rng.uniform(size=n) < p_high
        activity = np.where(is_high, high, low)
        expression = jnp.asarray(_BASE + _SCALE * activity)[:, None]
        key, k_lib, k_counts = jax.random.split(key, 3)
        library = sample_library_sizes(k_lib, n, log_sd=library_sigma)
        counts = sample_counts(k_counts, expression, library, dispersion=dispersion)
        blocks.append(np.asarray(counts))
        obs_condition.extend([name] * n)
    return _assemble(blocks, obs_condition)


def generate_dropout_decoy(
    *,
    n_cells_per_condition: int = 1000,
    activity: float = 1.2,
    low_depth_frac_wt: float = 0.5,
    low_depth_frac_pert: float = 0.3,
    low_depth: float = 0.04,
    dispersion: float = 0.1,
    library_sigma: float = 0.2,
    seed: int = 0,
) -> ad.AnnData:
    """Technical dropout zero-peak — a MONOSTABLE population made to look bimodal.

    Biology is a single constant expression level (no switch). But per-cell library
    depth is bimodal: a fraction of cells are captured at very low depth (``low_depth``)
    and read out as near-all-zeros, the rest at normal depth show the expression mode.
    The resulting zero-peak + expression-peak mimics an OFF/ON switch, but it is a pure
    *measurement* artifact — NUDGE must decline (off-model).
    """
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)
    expression = jnp.full((n_cells_per_condition, 1), _BASE + _SCALE * activity)
    blocks: list[np.ndarray] = []
    obs_condition: list[str] = []
    for name, frac in (("WT", low_depth_frac_wt), ("pert", low_depth_frac_pert)):
        n = n_cells_per_condition
        is_low = rng.uniform(size=n) < frac
        depth = np.where(is_low, low_depth, 1.0)
        jitter = np.exp(library_sigma * rng.standard_normal(n))
        library = jnp.asarray(depth * jitter)
        key, k_counts = jax.random.split(key)
        counts = sample_counts(k_counts, expression, library, dispersion=dispersion)
        blocks.append(np.asarray(counts))
        obs_condition.extend([name] * n)
    return _assemble(blocks, obs_condition)


def generate_readout_nonlinearity_decoy(
    *,
    n_cells_per_condition: int = 1500,
    weight_wt: float = 1.0,
    weight_pert: float = 3.0,
    readout_h: float = 6.0,
    readout_km: float = 1.488,
    readout_vmax: float = 5.0,
    readout_base: float = 1.0,
    extrinsic_sigma: float = 0.55,
    dispersion: float = 0.1,
    library_sigma: float = 0.2,
    seed: int = 1,
) -> ad.AnnData:
    """A LINEAR circuit through a NONLINEAR (Hill) readout — a KNOWN blind spot.

    This is the witness for ``NUDGE-LIM-006``, **not** a passing decoy: the ground
    truth is a linear (non-switch) circuit ``IN → SW``, but the sigmoidal reporter
    ``Λ = base + Vmax·a^h/(Km^h + a^h)`` on the broad activity distribution produces a
    skewed/pseudo-bimodal count distribution that NUDGE — which assumes an *affine*
    readout — can misattribute to a circuit switch. NUDGE *should* return off-model
    here; at strong readout nonlinearity + strong perturbation it is fooled into a
    confident positive (see ``FINDINGS.md``). Kept as an executable witness.
    """
    blocks: list[np.ndarray] = []
    obs_condition: list[str] = []
    for si, (name, w) in enumerate((("WT", weight_wt), ("pert", weight_pert))):
        circ = Circuit(
            [
                SpeciesDef("IN", basal=1.0, decay=1.0),
                SpeciesDef("SW", basal=0.5, decay=1.0),
            ],
            [EdgeDef(0, 1, "linear", weight=w)],
        )
        k_ext, k_lib, k_counts = jax.random.split(jax.random.key(seed + si), 3)
        params = _per_cell_params(circ, k_ext, n_cells_per_condition, extrinsic_sigma)
        activity = circ.solve_population(
            params, jnp.zeros((n_cells_per_condition, circ.n_species))
        )
        a = activity[:, 1]  # SW
        lam = readout_base + readout_vmax * a**readout_h / (
            readout_km**readout_h + a**readout_h
        )
        library = sample_library_sizes(
            k_lib, n_cells_per_condition, log_sd=library_sigma
        )
        counts = sample_counts(k_counts, lam[:, None], library, dispersion=dispersion)
        blocks.append(np.asarray(counts))
        obs_condition.extend([name] * n_cells_per_condition)
    return _assemble(blocks, obs_condition)
