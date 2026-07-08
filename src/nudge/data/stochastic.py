"""Tier-0.5 independent stochastic simulator — the inverse-crime guard.

Tier-0 (``data/synthetic.py``) generates a population by ``vmap``-ing a
*deterministic* steady-state solve over per-cell parameter draws: bimodality is
**designed-in** by the extrinsic parameter distribution crossing a transfer
function (the Ochab-Marcinek & Tabaka 2010 route). The fitter shares that exact
model, so any V&V on Tier-0 is an **inverse crime** (``design/STATE.md`` §6).

Tier-0.5 breaks the crime while changing *only the dynamics*. It runs a
self-contained **tau-leaping stochastic simulation** (SSA) of a self-activating
gene — genuine birth/death Poisson reactions with cooperative positive feedback —
so bimodality is **emergent**: modes sit at the deterministic fixed points but are
populated by intrinsic noise, not by a parameter distribution we chose. If NUDGE's
deterministic fit still attributes mechanism (and, crucially, is *never wrong*) on
this independent process, the approach generalises; if it breaks we learn the
failure mode cheaply (Kepler & Elston 2001; To & Maheshri 2010).

The observation layer is reused **verbatim**: molecule counts ``X`` → concentration
``X/Ω`` → ``Readout.expression`` (``Λ``) → ``sample_counts`` (NB Poisson-Gamma) —
the same path Tier-0 takes. So ``fit()`` — which reads only ``.obs['condition']``
and the count matrix, never ``.uns`` — consumes the result identically, and any
difference in NUDGE's verdict is attributable to the *dynamics*, not the
measurement model.

**System size ``Ω``.** The SSA works in molecule numbers; concentration is
``c = X/Ω``. Production propensity is ``Ω·(basal + vmax·c^n/(K^n+c^n))`` and
degradation ``decay·X``, so the concentration fixed points match the deterministic
circuit while molecule counts land in the tens-to-hundreds — the regime where
Poisson noise gives cleanly separated modes rather than swamping them. ``Ω`` also
sets the barrier height between basins (smaller ``Ω`` → more noise-induced hopping).
"""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from nudge.core.circuit import Circuit
from nudge.core.vocabulary import MechanismClass
from nudge.data.noise import sample_counts, sample_library_sizes
from nudge.data.synthetic import PerturbationSpec
from nudge.mechanisms.readout import Readout

__all__ = ["generate_stochastic_perturbseq"]

# Extrinsic (per-cell) variation scales the expression-level rate constants only,
# NOT the switch shape params (K, n) — matching Tier-0's population model so the
# controlled difference vs Tier-0 is purely deterministic-vs-stochastic dynamics.
_EXTRINSIC_PARAMS = ("basal", "decay")


def _read_kinetics(circuit: Circuit) -> dict[str, float]:
    """Scalar kinetics of the self-activation switch: species 0 + self-edge 0."""
    if circuit.n_species < 1 or circuit.n_edges < 1:
        raise ValueError(
            "generate_stochastic_perturbseq needs a self-activation circuit: "
            "one species with a self-edge (e.g. EdgeDef(0, 0, 'hill_activation'))."
        )
    base = circuit.base_params()
    return {
        "basal": float(base["species"]["basal"][0]),
        "decay": float(base["species"]["decay"][0]),
        "K": float(base["edges"]["K"][0]),
        "n": float(base["edges"]["n"][0]),
        "vmax": float(base["edges"]["vmax"][0]),
    }


def _apply_perturbation(
    kin: dict[str, float], pert: PerturbationSpec
) -> dict[str, float]:
    """Scale the targeted propensity parameter by ``factor`` (ground-truth mover)."""
    out = dict(kin)
    out[pert.param] = out[pert.param] * pert.factor
    return out


def _simulate_condition(
    rng: np.random.Generator,
    kin: dict[str, float],
    *,
    n_cells: int,
    omega: float,
    dt: float,
    n_steps: int,
    extrinsic_sigma: float,
) -> np.ndarray:
    """Tau-leap the self-activating gene to steady state → concentration ``X/Ω``.

    Returns per-cell concentrations (shape ``(n_cells,)``), the analogue of the
    Tier-0 deterministic activity that the ``Readout`` then maps to expression.
    """
    basal = np.full(n_cells, kin["basal"])
    decay = np.full(n_cells, kin["decay"])
    if extrinsic_sigma > 0:
        basal = basal * np.exp(extrinsic_sigma * rng.standard_normal(n_cells))
        decay = decay * np.exp(extrinsic_sigma * rng.standard_normal(n_cells))
    n, k, vmax = kin["n"], kin["K"], kin["vmax"]

    # Seed both basins: cover [0, 2·high-fixed-point] in molecule units so the
    # snapshot samples both attractors rather than one initial condition.
    hi = 2.0 * omega * (kin["basal"] + kin["vmax"]) / kin["decay"]
    x = rng.uniform(0.0, hi, size=n_cells)  # molecule counts

    for _ in range(n_steps):
        c = x / omega  # concentration — Hill lives in concentration units
        c_n = c**n
        hill = vmax * c_n / (k**n + c_n)
        prod_rate = omega * (basal + hill)  # molecules / time
        deg_rate = decay * x  # molecules / time
        births = rng.poisson(prod_rate * dt)
        deaths = rng.poisson(deg_rate * dt)
        x = np.maximum(x + births - deaths, 0.0)

    return x / omega  # → concentration, same scale as deterministic activity


def generate_stochastic_perturbseq(
    circuit: Circuit,
    perturbations: Sequence[PerturbationSpec] = (),
    readout: Readout | None = None,
    *,
    n_cells_per_condition: int = 1000,
    omega: float = 50.0,
    dt: float = 0.05,
    n_steps: int = 2000,
    dispersion: float = 0.1,
    library_sigma: float = 0.2,
    extrinsic_sigma: float = 0.1,
    seed: int = 0,
    gene_names: Sequence[str] | None = None,
) -> ad.AnnData:
    """Generate a **Tier-0.5** stochastic Perturb-seq ``AnnData`` (emergent bimodality).

    ``circuit`` must be a self-activation switch (species 0 with self-edge 0); its
    ``K`` / ``n`` / ``vmax`` / ``basal`` / ``decay`` seed the SSA propensities.
    Passing the *same* ``Circuit`` object the fitter later uses keeps the mechanistic
    hypothesis matched to the generating **topology** while the generation *process*
    is fully independent (Poisson dynamics, not a deterministic solve) — the clean
    inverse-crime break.

    A ``PerturbationSpec`` scales one propensity parameter, and *which* fixes the
    true mechanism (``K`` → threshold, ``n`` → gain, ``vmax`` → ceiling). Output
    schema is identical to :func:`nudge.data.synthetic.generate_synthetic_perturbseq`
    (counts in ``.X``; ``condition`` / ``true_mechanism`` in ``.obs``; ground truth
    in ``.uns['ground_truth']``, tagged ``tier='0.5-stochastic'``).
    """
    if readout is None:
        readout = Readout.identity(circuit.n_species)
    kin = _read_kinetics(circuit)
    n_cells = n_cells_per_condition
    conditions: list[PerturbationSpec | None] = [None, *perturbations]

    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)
    count_blocks: list[np.ndarray] = []
    obs_condition: list[str] = []
    obs_mechanism: list[str] = []
    ground_truth_conditions: list[dict[str, object]] = []

    for cond in conditions:
        cond_kin = kin if cond is None else _apply_perturbation(kin, cond)
        activity = _simulate_condition(
            rng, cond_kin,
            n_cells=n_cells, omega=omega, dt=dt, n_steps=n_steps,
            extrinsic_sigma=extrinsic_sigma,
        )
        expression = readout.expression(jnp.asarray(activity[:, None]))
        key, k_lib, k_counts = jax.random.split(key, 3)
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
        index=pd.Index([f"cell_{i}" for i in range(counts_matrix.shape[0])]),
    )
    var = pd.DataFrame(index=pd.Index(list(gene_names)))
    adata = ad.AnnData(X=counts_matrix, obs=obs, var=var)
    adata.uns["ground_truth"] = {
        "tier": "0.5-stochastic",
        "conditions": ground_truth_conditions,
        "species": list(circuit.names),
        "seed": int(seed),
        "omega": float(omega),
        "kinetics": {k: float(v) for k, v in kin.items()},
    }
    return adata
