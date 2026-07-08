"""The Tier-0 generator: valid AnnData + threshold/gain/ceiling are distinguishable.

The switch testbed is the validated deterministic transfer-function route
(Ochab-Marcinek & Tabaka 2010): a monostable input ``IN`` with per-cell extrinsic
variation drives a steep Hill into a switch ``SW``, giving a bimodal ``SW``. Each
perturbation moves one parameter and must shift the population the *right* way.
"""

from __future__ import annotations

import numpy as np
import pytest

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef
from nudge.core.vocabulary import MechanismClass
from nudge.data.synthetic import PerturbationSpec, generate_synthetic_perturbseq

_N = 2000
_OFF, _ON = 3, 6  # count thresholds for the OFF / ON modes of the SW reporter


def _switch_circuit() -> Circuit:
    return Circuit(
        [
            SpeciesDef("IN", basal=1.0, decay=1.0),
            SpeciesDef("SW", basal=0.05, decay=1.0),
        ],
        [EdgeDef(0, 1, "hill_activation", K=1.0, n=6.0, vmax=2.0)],
    )


_PERTS = [
    PerturbationSpec("KD_thresh", "edge", 0, "K", 2.0),  # ↑ threshold
    PerturbationSpec("KD_gain", "edge", 0, "n", 0.34),  # ↓ gain (less steep)
    PerturbationSpec("KD_ceil", "edge", 0, "vmax", 0.5),  # ↓ ceiling
]


@pytest.fixture(scope="module")
def adata():
    return generate_synthetic_perturbseq(
        _switch_circuit(), _PERTS, n_cells_per_condition=_N, realism_level=1, seed=0
    )


def _sw(adata, condition: str) -> np.ndarray:
    return np.asarray(adata[adata.obs.condition == condition, "SW"].X).ravel()


# ── Output validity ──────────────────────────────────────────────────────────
def test_shape_and_dtype(adata) -> None:
    assert adata.shape == (4 * _N, 2)  # WT + 3 perturbations
    assert list(adata.var_names) == ["IN", "SW"]
    assert np.issubdtype(adata.X.dtype, np.integer)
    assert adata.X.min() >= 0


def test_obs_and_ground_truth_labels(adata) -> None:
    assert set(adata.obs["condition"]) == {"WT", "KD_thresh", "KD_gain", "KD_ceil"}
    gt = {c["name"]: c["mechanism"] for c in adata.uns["ground_truth"]["conditions"]}
    assert gt["WT"] == MechanismClass.NO_EFFECT.value
    assert gt["KD_thresh"] == MechanismClass.THRESHOLD.value
    assert gt["KD_gain"] == MechanismClass.GAIN.value
    assert gt["KD_ceil"] == MechanismClass.CEILING.value
    # per-cell mechanism label agrees with the condition
    row = adata.obs[adata.obs.condition == "KD_thresh"].iloc[0]
    assert row["true_mechanism"] == MechanismClass.THRESHOLD.value


def test_determinism(adata) -> None:
    again = generate_synthetic_perturbseq(
        _switch_circuit(), _PERTS, n_cells_per_condition=_N, realism_level=1, seed=0
    )
    assert np.array_equal(adata.X, again.X)


# ── The science: each mechanism moves the population the right way ────────────
def test_wt_is_bimodal(adata) -> None:
    sw = _sw(adata, "WT")
    assert (sw < _OFF).mean() > 0.2  # a substantial OFF mode
    assert (sw > _ON).mean() > 0.2  # and a substantial ON mode


def test_threshold_mover_reduces_on_fraction(adata) -> None:
    on_wt = (_sw(adata, "WT") > _ON).mean()
    on_kd = (_sw(adata, "KD_thresh") > _ON).mean()
    assert on_kd < 0.2
    assert on_kd < on_wt - 0.15  # far fewer cells activate


def test_ceiling_mover_lowers_on_level(adata) -> None:
    # The ON-state expression level drops: the upper tail shrinks.
    p90_ceil = np.percentile(_sw(adata, "KD_ceil"), 90)
    p90_wt = np.percentile(_sw(adata, "WT"), 90)
    assert p90_ceil < p90_wt


def test_gain_mover_makes_response_more_graded(adata) -> None:
    def mid_fraction(v: np.ndarray) -> float:
        return float(((v >= _OFF) & (v <= _ON)).mean())

    # Lower Hill n ⇒ a less sharp switch ⇒ more cells in the intermediate range.
    assert mid_fraction(_sw(adata, "KD_gain")) > mid_fraction(_sw(adata, "WT")) + 0.05


# ── Misc ─────────────────────────────────────────────────────────────────────
def test_perturbation_mechanism_derivation() -> None:
    cases = [
        ("K", MechanismClass.THRESHOLD),
        ("n", MechanismClass.GAIN),
        ("vmax", MechanismClass.CEILING),
        ("basal", MechanismClass.NO_EFFECT),
    ]
    for param, mechanism in cases:
        assert PerturbationSpec("p", "edge", 0, param, 0.5).mechanism is mechanism


def test_realism_level_zero_is_poisson_integer(adata) -> None:
    poisson = generate_synthetic_perturbseq(
        _switch_circuit(), _PERTS, n_cells_per_condition=200, realism_level=0, seed=1
    )
    assert np.issubdtype(poisson.X.dtype, np.integer)
    assert poisson.X.min() >= 0
