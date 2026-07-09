"""Tests for bifurcation / tipping-point proximity — the robustness dial.

The synthetic **parameter-sweep ground truth** is the load-bearing validation: the
self-activation switch has a known saddle-node fold in its cooperativity ``n`` (and in
``K``), so we sweep toward it and assert the three channels move **monotonically** as
the fold nears, the fused dial *ranks* proximity correctly, and ``one_sided`` sets near
the fold. Because we know where the fold is, this is a clean ground-truth test, not a
vibe check. The four verdicts (near-fold / robust / unresolved / not-bistable) are
pinned by the regime tests, and the honesty crux (a one-sided lower bound near the fold;
an abstention on the deep-basin side) by the near-fold + deep-basin tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nudge.circuits import ras_switch_1node
from nudge.inference.bifurcation import (
    BifurcationScore,
    attribute_bifurcation,
    bifurcation_proximity,
    classify_robustness,
)


def _score(**kw) -> BifurcationScore | None:
    return bifurcation_proximity(ras_switch_1node(**kw))


# --------------------------------------------------------------------------- #
# The four verdicts (regime tests)
# --------------------------------------------------------------------------- #
def test_near_fold_is_a_one_sided_lower_bound() -> None:
    """A switch pushed toward its cooperativity fold reads near-fold + one_sided."""
    score = _score(n=2.0)
    assert score is not None
    call, reason = classify_robustness(score)
    assert call == "near-fold"
    assert score.one_sided is True  # the noise lobes overlap → the LNA is breaking down
    assert score.lna_lobe_ratio >= 1.0
    assert "LOWER BOUND" in reason and "NUDGE-LIM-012" in reason


def test_deep_basin_abstains_with_no_precise_far_number() -> None:
    """A very cooperative (deep-basin) switch abstains — no false-precise 'far'."""
    score = _score(n=10.0)
    assert score is not None
    call, reason = classify_robustness(score)
    assert call == "unresolved"
    assert score.proximity < 0.05
    # the abstention must NOT emit a precise distance; it explains the floored signal.
    assert "ABSTAINS" in reason and "beyond the dial's resolution" in reason


def test_well_buffered_switch_is_robust() -> None:
    """The nominal (well-buffered) self-activation switch reads robust."""
    score = _score(n=6.0)
    assert score is not None
    call, _reason = classify_robustness(score)
    assert call == "robust"
    assert 0.05 <= score.proximity < 0.55


def test_monostable_is_not_bistable() -> None:
    """A non-cooperative switch is monostable → score None → not-bistable."""
    score = _score(n=1.0)
    assert score is None
    call, reason = classify_robustness(score)
    assert call == "not-bistable"
    assert "fewer than two stable modes" in reason


# --------------------------------------------------------------------------- #
# The parameter-sweep monotonicity ground truth (the load-bearing validation)
# --------------------------------------------------------------------------- #
def test_channels_move_monotonically_toward_the_known_fold() -> None:
    """Sweeping n toward the self-activation fold moves all three channels correctly.

    Ground truth: as cooperativity ``n`` drops toward the saddle-node, min|Re λ| → 0,
    node→saddle → 0, the lobe ratio → 1(+), and the fused dial RANKS proximity
    (monotonically increasing). A clean analytic fold, so the direction is known.
    """
    ladder = [6.0, 4.0, 3.0, 2.5, 2.2]  # decreasing n → approaching the fold
    scores = [_score(n=n) for n in ladder]
    assert all(s is not None for s in scores)
    min_re = [s.min_re_lambda for s in scores]  # type: ignore[union-attr]
    nsd = [s.node_saddle_distance for s in scores]  # type: ignore[union-attr]
    lobe = [s.lna_lobe_ratio for s in scores]  # type: ignore[union-attr]
    prox = [s.proximity for s in scores]  # type: ignore[union-attr]

    assert all(a > b for a, b in zip(min_re, min_re[1:], strict=False))  # → 0
    assert all(a > b for a, b in zip(nsd, nsd[1:], strict=False))  # → 0
    assert all(a < b for a, b in zip(lobe, lobe[1:], strict=False))  # → 1(+)
    assert all(a < b for a, b in zip(prox, prox[1:], strict=False))  # dial ranks it
    # and the dial ends near the fold (one_sided set) but starts robust.
    assert scores[0].one_sided is False  # type: ignore[union-attr]
    assert scores[-1].proximity >= scores[0].proximity  # type: ignore[union-attr]


def test_k_sweep_also_ranks_proximity() -> None:
    """The fold in K: raising the threshold toward its saddle-node ranks the dial."""
    ladder = [1.0, 1.1, 1.2, 1.3]  # rising K → approaching the high-K fold
    prox = [s.proximity for k in ladder if (s := _score(K=k)) is not None]
    assert len(prox) == len(ladder)
    assert all(a < b for a, b in zip(prox, prox[1:], strict=False))
    # past the fold the switch is monostable (score None).
    assert _score(K=1.4) is None


# --------------------------------------------------------------------------- #
# The raw-metadata channels (for the demo / notebook)
# --------------------------------------------------------------------------- #
def test_metadata_channels_are_populated_and_json_serialisable() -> None:
    """The channels dict carries the raw per-mode values, JSON-serialisable."""
    import json

    score = _score(n=3.0)
    assert score is not None
    ch = score.channels
    for key in (
        "per_mode_min_abs_re_lambda",
        "per_mode_eig_real",
        "per_mode_lobe_std",
        "stable_nodes",
        "separation",
        "decay_ref",
        "channel_proximities",
    ):
        assert key in ch, key
    assert len(ch["per_mode_min_abs_re_lambda"]) == score.n_stable_modes
    assert len(ch["stable_nodes"]) == score.n_stable_modes
    assert set(ch["channel_proximities"]) == {
        "critical_slowing",
        "basin_collapse",
        "lobe_overlap",
    }
    json.dumps(ch)  # must not raise


# --------------------------------------------------------------------------- #
# The data-driven entry (attribute_bifurcation)
# --------------------------------------------------------------------------- #
def test_attribute_bifurcation_from_activity_data_scores_and_gates_depth() -> None:
    """attribute_bifurcation scores the circuit and calibrates the depth from data."""
    import jax

    from nudge.inference.lyapunov import sample_lna_mixture

    circuit = ras_switch_1node(n=2.2)  # near-ish the fold
    data = sample_lna_mixture(circuit, 400, jax.random.PRNGKey(0), scale=40.0)
    res = attribute_bifurcation(data, circuit, free=None)
    assert res.score is not None
    assert res.call in {"near-fold", "robust"}
    assert np.isfinite(res.scale) and res.scale > 0
    assert isinstance(res.lna_reason, str) and res.lna_reason


# --------------------------------------------------------------------------- #
# Real-data lock-in (skip-if-absent — the synthetic sweep is the load-bearing test)
# --------------------------------------------------------------------------- #
_DRIVE = Path("/media/nick/Seagate Hub/bifurcation")


@pytest.mark.needs_data
@pytest.mark.validation
def test_real_data_dose_ladder_dial_rises() -> None:
    """A bistable dose ladder approaching a fold: the dial should RISE along the ladder.

    Follow-up lock-in (deferred — the synthetic parameter sweep is the load-bearing
    validation). Expects a small processed CSV at ``$DRIVE/dose_ladder.csv`` with a
    ``rung`` column (ordered toward the fold) + per-rung activity cols; skips if absent
    CI never depends on the download (toggle+hysteresis Zenodo 11817798 or a morphogen
    top rung GSE233574; ground the reading in critical-slowing-down theory).
    """
    csv = _DRIVE / "dose_ladder.csv"
    if not csv.exists():
        pytest.skip(f"no processed dose ladder at {csv} (deferred follow-up)")
    import pandas as pd

    df = pd.read_csv(csv)
    rungs = sorted(df["rung"].unique())
    prox: list[float] = []
    for rung in rungs:
        activity = df[df["rung"] == rung].drop(columns=["rung"]).to_numpy(dtype=float)
        res = attribute_bifurcation(activity, ras_switch_1node(), free=None)
        assert res.score is not None
        prox.append(res.score.proximity)
    assert prox[-1] > prox[0]  # the dial rises toward the fold rung
