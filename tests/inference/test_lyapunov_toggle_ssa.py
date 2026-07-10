"""Toggle covariance attribution on INDEPENDENT stochastic (SSA) ground truth.

`test_lyapunov.py` validates the Lyapunov path on *inverse-crime* data (cells sampled
from the very LNA Gaussian mixture the fitter maximizes). This module is the harder,
honest test: data from the **independent tau-leaping SSA** (the true stochastic
stationary distribution, `generate_toggle_perturbseq`, not a Gaussian the fitter drew),
bridged to activity as the real-data path does (`inference.bridge.counts_to_activity`).

Two measured outcomes (`scripts/vv/toggle_lyapunov_ssa.py`; FINDINGS "Covariance
attribution — independent-SSA validation"):

- **The single snapshot DEGENERATES.** The inverse-crime result that "ceiling is the
  identifiable one" does NOT survive: on independent SSA the free-vmax fit becomes the
  *worst* explanation of a true ceiling, so the single-condition call mis-narrows a true
  ceiling to ``gain_or_threshold`` and abstains (``unresolved``) on gain/threshold. It
  never emits a bare gain/threshold/ceiling — an abstention, never a confident wrong.
- **The two-operating-point breaker RECOVERS — fail-safe.** Adding a second operating
  point (a basal-B shift) lets the shared-parameter joint fit resolve **threshold** and
  **ceiling** correctly (3/3 across seeds, margins 0.16–0.35 nats) while honestly
  **abstaining on gain** (the residual gain⇄threshold confound). **0 confident-wrong
  calls** across all mechanisms × seeds — the non-negotiable fail-safe holds on
  independent stochastic data.

The budget (n_cells=3000, steps=200) matches the FINDINGS measurement; the tests are
`slow` (SSA generation + N-D-finder fits).
"""

from __future__ import annotations

import numpy as np
import pytest

from nudge.circuits import toggle
from nudge.data.stochastic import generate_toggle_perturbseq
from nudge.data.synthetic import PerturbationSpec
from nudge.inference.bridge import counts_to_activity
from nudge.inference.lyapunov import (
    OperatingPoint,
    attribute_lyapunov_multi,
    attribute_lyapunov_single,
    calibrate_from_wt,
)
from nudge.mechanisms.readout import Readout

pytestmark = pytest.mark.verification

# One marker gene per species (identity readout → gene name == species name).
_MARKERS = {"A": ["A"], "B": ["B"]}
# Deep-sequencing readout (Lambda = 0.2 + 15*activity): the high mode reads ~30 counts,
# so scale*peak clears lna_reliable's >=15 depth guard (the default scale=5 trips it, at
# which depth NUDGE correctly abstains before any fit). Adequate depth is the regime
# where the covariance signature can be *tested* rather than guard-abstained.
_DEEP = Readout.identity(2, scale=15.0)
# Factors kept mild enough that BOTH attractors stay populated (the LNA needs 2 modes):
# gain n:4→2.4, threshold K:1→1.6, ceiling vmax:2→1.2.
_FACTOR = {"n": 0.6, "K": 1.6, "vmax": 0.6}

# The three bare positive labels a wrong call would take (the fail-safe forbids these
# unless they match ground truth).
_BARE = frozenset({"gain", "threshold", "ceiling"})
_TRUE_LABEL = {"n": "gain", "K": "threshold", "vmax": "ceiling"}


def _activity(adata: object, condition: str, circ: object) -> np.ndarray:
    mask = np.asarray(adata.obs["condition"] == condition)  # type: ignore[attr-defined]
    return counts_to_activity(adata[mask], circ, _MARKERS)  # type: ignore[index]


def _ssa(basal: float, param: str, seed: int):
    """Independent-SSA toggle AnnData (WT + one perturbed condition) + its circuit."""
    circ = toggle(basal=basal)
    adata = generate_toggle_perturbseq(
        circ,
        [PerturbationSpec("cond", "edge", 0, param, _FACTOR[param])],
        readout=_DEEP,
        n_cells_per_condition=3000,
        seed=seed,
    )
    return adata, circ


def _operating_point(basal: float, param: str, seed: int) -> OperatingPoint:
    adata, circ = _ssa(basal, param, seed)
    wt = _activity(adata, "WT", circ)
    cond = _activity(adata, "cond", circ)
    scale, obs = calibrate_from_wt(wt, circ)
    return OperatingPoint(data=cond, circuit=circ, scale=scale, obs_sd=obs)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("param", "expected"),
    [
        ("K", "threshold"),  # recovered by the second operating point
        ("vmax", "ceiling"),  # recovered by the second operating point
        ("n", "unresolved"),  # honest abstention (residual gain⇄threshold confound)
    ],
)
def test_multi_operating_point_recovers_or_abstains(param: str, expected: str) -> None:
    # The two-operating-point covariance breaker on INDEPENDENT SSA data. It resolves
    # threshold and ceiling and abstains on gain — and, whatever it returns, it is
    # NEVER a bare mechanism that contradicts the ground truth (the fail-safe).
    op1 = _operating_point(0.05, param, seed=0)
    op2 = _operating_point(0.30, param, seed=0)
    label, nlls = attribute_lyapunov_multi([op1, op2], target_edge=0, steps=200, seed=0)

    # Fail-safe (non-negotiable): never a confident WRONG bare mechanism.
    if label in _BARE:
        assert label == _TRUE_LABEL[param], (
            f"toggle multi emitted WRONG bare mechanism {label!r} for true "
            f"{_TRUE_LABEL[param]!r} — fail-safe violated"
        )
    # Measured recover-or-abstain outcome (FINDINGS): threshold/ceiling resolve, gain
    # abstains.
    assert label == expected, f"{param}: got {label!r} (nlls={nlls})"


@pytest.mark.slow
@pytest.mark.decoy
def test_near_fold_third_point_never_confident_wrong() -> None:
    # NUDGE-LIM-017 decoy (red-team Hole 1 + the round-2 knife-edge). A TRUE ceiling
    # (vmax×0.6) seen at three operating points where the 3rd is near the toggle's fold but
    # STILL PASSES lna_reliable. Before the fix, adding it flipped the correct 2-point
    # 'ceiling' to a CONFIDENT WRONG 'threshold' (gap ≫ resolve_margin). A hard proximity
    # margin proved a knife-edge (the corruption onset is non-monotonic and only ~0.007 above
    # the useful operating point), so the fix is graded down-weighting + best-buffered-pair
    # CORROBORATION. The fail-safe assertion: the 3rd point must NEVER yield the wrong
    # 'threshold'/'gain' — ceiling (a far point down-weighted out) or unresolved (a marginal
    # point failing corroboration) are both correct-or-abstain outcomes.
    op_clean1 = _operating_point(0.05, "vmax", seed=0)
    op_clean2 = _operating_point(0.30, "vmax", seed=0)  # proximity 0.112 — the useful 2nd pt
    op_nearfold = _operating_point(0.60, "vmax", seed=0)  # proximity 0.231 — far near-fold

    # POSITIVE CONTROL: the two clean, well-buffered points resolve the true ceiling.
    clean, _ = attribute_lyapunov_multi(
        [op_clean1, op_clean2], target_edge=0, steps=200, seed=0
    )
    assert clean == "ceiling", f"clean 2-point control regressed: got {clean!r}"

    # THE DECOY: adding the near-fold 3rd point must NEVER produce a confident WRONG mechanism.
    label, _ = attribute_lyapunov_multi(
        [op_clean1, op_clean2, op_nearfold], target_edge=0, steps=200, seed=0
    )
    assert label not in ("threshold", "gain"), (
        f"near-fold 3rd point produced a confident WRONG mechanism {label!r} for a true "
        "ceiling — NUDGE-LIM-017 fail-safe violated"
    )
    assert label in ("ceiling", "unresolved"), (
        f"expected correct-or-abstain (ceiling/unresolved), got {label!r}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("param", ["n", "K", "vmax"])
def test_single_snapshot_never_confidently_wrong(param: str) -> None:
    # The single-snapshot degeneracy, guarded as fail-safe: on independent SSA the
    # covariance signature cannot resolve mechanism from ONE operating point — gain/
    # threshold abstain (unresolved), a true ceiling mis-narrows to gain_or_threshold.
    # The load-bearing assertion: it NEVER emits a bare gain/threshold/ceiling from one
    # snapshot (never a confident wrong call). Both are abstention-class labels.
    adata, circ = _ssa(0.05, param, seed=0)
    wt = _activity(adata, "WT", circ)
    cond = _activity(adata, "cond", circ)
    label, _nlls = attribute_lyapunov_single(
        cond, circ, wt_data=wt, target_edge=0, steps=200, seed=0
    )
    assert label not in _BARE, (
        f"single snapshot emitted bare mechanism {label!r} for true "
        f"{_TRUE_LABEL[param]!r} — a single toggle snapshot must abstain"
    )
    assert label in ("unresolved", "gain_or_threshold")
