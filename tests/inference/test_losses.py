"""Distributional losses: metric properties + separating real generated conditions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef
from nudge.data.synthetic import PerturbationSpec, generate_synthetic_perturbseq
from nudge.inference.losses import energy_distance, rbf_mmd


def test_energy_distance_zero_for_identical() -> None:
    x = jnp.asarray(np.random.default_rng(0).normal(size=(50, 2)))
    assert abs(float(energy_distance(x, x))) < 1e-4


def test_energy_distance_symmetric_and_nonnegative() -> None:
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.normal(size=(40, 2)))
    y = jnp.asarray(rng.normal(size=(60, 2)) + 3.0)
    ed_xy, ed_yx = float(energy_distance(x, y)), float(energy_distance(y, x))
    assert ed_xy == pytest.approx(ed_yx, abs=1e-5)
    assert ed_xy > 0


def test_energy_distance_differentiable() -> None:
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.normal(size=(30, 2)))
    y = jnp.asarray(rng.normal(size=(30, 2)) + 1.0)
    grad = jax.grad(lambda a: energy_distance(a, y))(x)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_rbf_mmd_zero_for_identical_and_nonnegative() -> None:
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(size=(50, 2)))
    y = jnp.asarray(rng.normal(size=(50, 2)) + 3.0)
    assert abs(float(rbf_mmd(x, x))) < 1e-5
    assert float(rbf_mmd(x, y)) > 0


@pytest.fixture(scope="module")
def condition_counts() -> dict[str, jnp.ndarray]:
    circuit = Circuit(
        [
            SpeciesDef("IN", basal=1.0, decay=1.0),
            SpeciesDef("SW", basal=0.05, decay=1.0),
        ],
        [EdgeDef(0, 1, "hill_activation", K=1.0, n=6.0, vmax=2.0)],
    )
    perts = [PerturbationSpec("KD_thresh", "edge", 0, "K", 2.0)]
    adata = generate_synthetic_perturbseq(
        circuit, perts, n_cells_per_condition=400, realism_level=1, seed=0
    )
    out = {}
    for cond in ("WT", "KD_thresh"):
        block = np.asarray(adata[adata.obs.condition == cond].X, dtype=np.float32)
        out[cond] = jnp.asarray(block)
    return out


def test_energy_distance_separates_conditions(condition_counts) -> None:
    # Two halves of WT are the same distribution; the threshold mover is different.
    wt = condition_counts["WT"]
    wt_a, wt_b = wt[:200], wt[200:]
    within = float(energy_distance(wt_a, wt_b))
    between = float(energy_distance(wt_a, condition_counts["KD_thresh"][:200]))
    assert between > within


def test_rbf_mmd_separates_conditions(condition_counts) -> None:
    wt = condition_counts["WT"]
    wt_a, wt_b = wt[:200], wt[200:]
    within = float(rbf_mmd(wt_a, wt_b, sigma=5.0))
    between = float(rbf_mmd(wt_a, condition_counts["KD_thresh"][:200], sigma=5.0))
    assert between > within
