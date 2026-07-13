"""Regression lock for NUDGE-LIM-025 — the multi-point breaker's identifiability gate.

The M3 breaker (``attribute_lyapunov_multi``) resolved a THRESHOLD-DOMINATED large-gain
perturbation to a CONFIDENT WRONG ``threshold`` (Hill n 4→1.5): the true knob is gain, but
the perturbed condition slides through the fold (monostable at one operating point, at the
fold at the other) so the second operating point never breaks the gain⇄threshold
degeneracy, yet the pure NLL-gap test resolves anyway (gap ≈1.7 ≫ 0.03).

The fix (additive in ``inference/lyapunov.py``) adds an IDENTIFIABILITY GATE: after the
breaker resolves a mechanism X, it fits the joint (X, runner-up Y) model and reads the
Laplace posterior; if a runner-up is identifiable and displaced beyond the MEASURED cut
(``_CONTAM_MARGIN``=0.5 log-units; genuine ≤0.12 vs the hole ≈1.0), NUDGE abstains. Plus a
graceful "bistability lost" degradation. These tests lock: the confident-wrong now abstains
(0 confident-wrong), the positive controls still resolve / abstain (no over-abstention), and
a monostable operating point degrades gracefully.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef
from nudge.inference.lyapunov import (
    OperatingPoint,
    attribute_lyapunov_multi,
    calibrate_from_wt,
    sample_lna_mixture,
)

SCALE, OBS_SD = 20.0, 0.5


def _toggle_bB(bB: float) -> Circuit:
    return Circuit(
        [SpeciesDef("A", basal=0.05, decay=1.0), SpeciesDef("B", basal=bB, decay=1)],
        [
            EdgeDef(1, 0, "hill_repression", K=1.0, n=4.0, vmax=2.0),
            EdgeDef(0, 1, "hill_repression", K=1.0, n=4.0, vmax=2.0),
        ],
    )


def _operating_point(bB: float, param: str, val: float, seed: int) -> OperatingPoint:
    # Matches tests/inference/test_lyapunov.py: WT key 500(+seed), cond key seed.
    wt = _toggle_bB(bB)
    wt_data = sample_lna_mixture(
        wt, 3000, jax.random.PRNGKey(500 + seed), scale=SCALE, obs_sd=OBS_SD
    )
    scale, obs = calibrate_from_wt(wt_data, wt)
    cond = sample_lna_mixture(
        wt, 3000, jax.random.PRNGKey(seed),
        free=[("edge", 0, param)], vals=np.array([val]), scale=SCALE, obs_sd=OBS_SD,
    )
    return OperatingPoint(data=cond, circuit=wt, scale=scale, obs_sd=obs)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1])
def test_threshold_dominated_large_gain_abstains(seed: int) -> None:
    # THE HOLE, LOCKED: a true GAIN knockdown (n 4→1.5) that is threshold-dominated in the
    # moments used to resolve a CONFIDENT WRONG 'threshold' (gap ≈1.7). The identifiability
    # gate now abstains — never the wrong (or any) bare mechanism.
    op1 = _operating_point(0.05, "n", 1.5, seed)
    op2 = _operating_point(0.30, "n", 1.5, seed)
    label, _nlls = attribute_lyapunov_multi([op1, op2], target_edge=0, steps=200, seed=0)
    assert label == "unresolved", label
    assert label not in ("gain", "threshold", "ceiling")  # never confidently wrong


@pytest.mark.slow
def test_genuine_threshold_still_resolves() -> None:
    # POSITIVE CONTROL (no over-abstention): a genuine, resolvable threshold shift (K=2.0)
    # still resolves 'threshold' through the gate — the joint (threshold, runner-up) fit
    # leaves the runner-up a free nuisance / barely displaced (contam ≈0.095 < 0.5). Uses
    # the exact seed config of test_second_operating_point_breaks_confound.
    op1 = _operating_point(0.05, "K", 2.0, 0)
    op2 = _operating_point(0.30, "K", 2.0, 0)
    label, nlls = attribute_lyapunov_multi([op1, op2], target_edge=0, steps=200, seed=0)
    assert label == "threshold", (label, nlls)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1])
def test_confounded_gain_still_abstains(seed: int) -> None:
    # A weak gain shift (n=2.4) already abstains at the NLL-gap step (gap ≈0.005 < 0.03) —
    # unchanged by the gate, still 'unresolved', never a bare mechanism.
    op1 = _operating_point(0.05, "n", 2.4, seed)
    op2 = _operating_point(0.30, "n", 2.4, seed)
    label, _nlls = attribute_lyapunov_multi([op1, op2], target_edge=0, steps=200, seed=0)
    assert label == "unresolved", label


def test_monostable_operating_point_degrades_gracefully() -> None:
    # GRACEFUL DEGRADATION (fast — abstains before any fit): a monostable operating-point
    # circuit (bB=0.5 toggle, one stable mode) returns ('unresolved', {}) with no ValueError.
    mono = _toggle_bB(0.5)
    assert mono.mode_covariances() is None or len(mono.mode_covariances()) < 2
    data = np.zeros((50, 2), dtype=np.float32)
    op = OperatingPoint(data=data, circuit=mono, scale=20.0, obs_sd=0.5)
    label, nlls = attribute_lyapunov_multi([op, op], target_edge=0, steps=5, seed=0)
    assert label == "unresolved" and nlls == {}


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="NUDGE-LIM-025 residual BOUND: a threshold-dominated large-gain perturbation "
    "whose perturbed condition has slid through the fold is fundamentally non-separable "
    "from a threshold shift with these two operating points, so NUDGE ABSTAINS "
    "(fail-safe over-abstention) rather than positively resolving 'gain'. Recovering a "
    "positive gain call would need a better-buffered operating point. This decoy locks the "
    "honest bound: if it ever XPASSES (n=1.5 resolves 'gain'), the bound was overcome and "
    "this record must be updated.",
)
def test_large_gain_cannot_be_positively_resolved() -> None:
    # The bound: we cannot POSITIVELY resolve gain for the degenerate n=1.5 case — we abstain.
    op1 = _operating_point(0.05, "n", 1.5, 0)
    op2 = _operating_point(0.30, "n", 1.5, 0)
    label, _nlls = attribute_lyapunov_multi([op1, op2], target_edge=0, steps=200, seed=0)
    assert label == "gain", label  # XFAIL: it abstains ('unresolved'), the honest outcome
