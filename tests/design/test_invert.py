"""Tests for ``design()`` — inverse / intervention design (the flagship).

Six ground-truth checks mirror the plan's Definition of Done:

1. **known-intervention recovery** — perturb a monostable switch off-target on a single
   knob, then ``design()`` recovers the inverse Δ (factor ≈ the known change, loss ≈ 0);
2. **integrity gate** — an *unreliable* attribution (``unresolved``) → an
   ``AbstentionResult`` (NUDGE never designs off a fit it does not trust);
3. **reachability** — an impossible target → ``AbstentionResult`` (no Δ reaches it);
4. **safety gate** — a flip-ON intervention that drives a bistable switch over its fold
   sets ``crosses_fold`` / ``high_risk_of_instability`` (reuses the Cap-5 dial);
5. **curve-level** — invert a synthetic ``DoseResponseFit`` to the dose achieving a
   target response (round-trips the Hill), and a reachability abstain out of range;
6. **needs_data** — invert the REAL OCT4 dose-response fit to a target self-renewal.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from nudge.circuits import ras_switch_1node
from nudge.design.invert import (
    AbstentionResult,
    CircuitFit,
    InterventionPlan,
    design,
    flip_target,
)
from nudge.inference.dose_response import (
    DoseResponseResult,
    fit_dose_response,
)
from nudge.mechanisms.readout import Readout

_OCT4_NANOG_H5AD = "/media/nick/Seagate Hub/oct4_nanog/ESC.h5ad"


def _hill_repress(d: np.ndarray, floor: float, amp: float, k: float, n: float):
    return floor + amp * k**n / (k**n + np.maximum(d, 1e-9) ** n)


# --------------------------------------------------------------------------- #
# (1) known-intervention recovery — the loss ≈ 0 ground truth.
# --------------------------------------------------------------------------- #
def test_known_intervention_recovery_hits_zero_loss() -> None:
    """A monostable switch perturbed by a known ×2 on v_max: design recovers the Δ."""
    readout = Readout.identity(1)
    base = ras_switch_1node(n=1.0, K=1.0, vmax=2.0, basal=0.2)  # monostable
    known = ras_switch_1node(n=1.0, K=1.0, vmax=4.0, basal=0.2)  # a known ×2 on v_max
    # Target = the readout of the known circuit's (unique) steady state.
    from nudge.design.invert import _resolve_x0

    x0 = _resolve_x0(known, "low")
    target = np.asarray(readout.expression(known.steady_state(known.base_params(), x0)))

    plan = design(
        CircuitFit(circuit=base, free=[("edge", 0, "vmax")]),
        target,
        readout=readout,
        steps=600,
        l1=0.0,
    )
    assert isinstance(plan, InterventionPlan)
    assert plan.mode == "circuit"
    assert plan.achieved_loss < 1e-3  # closed essentially all of the gap
    assert len(plan.deltas) == 1
    (_scope, _idx, name), _log_delta, factor = plan.deltas[0]
    assert name == "vmax"
    assert factor == pytest.approx(2.0, rel=0.05)  # recovered the known ×2


# --------------------------------------------------------------------------- #
# (2) integrity gate — never design off an unreliable attribution.
# --------------------------------------------------------------------------- #
def test_integrity_gate_refuses_unreliable_attribution() -> None:
    """An ``unresolved`` dose-response is not reliable → design abstains immediately."""
    d = np.linspace(0.1, 5.0, 8)
    y = 0.5 + 0.01 * d  # a flat-ish curve
    fit = fit_dose_response(d, y, direction="repress", n_boot=50, seed=0)
    unreliable = DoseResponseResult(fit=fit, call="unresolved", reason="not Hill-like")
    assert unreliable.is_reliable is False

    out = design(unreliable, 0.4)
    assert isinstance(out, AbstentionResult)
    assert "integrity gate" in out.reason
    assert "not Hill-like" in out.reason


# --------------------------------------------------------------------------- #
# (3) reachability — an impossible target abstains (no false extrapolation).
# --------------------------------------------------------------------------- #
def test_reachability_abstains_on_impossible_target() -> None:
    """A readout target far above any reachable state → reachability abstention."""
    readout = Readout.identity(1)
    base = ras_switch_1node(n=6.0, K=1.0, vmax=2.0, basal=0.05)
    out = design(
        CircuitFit(circuit=base, free=[("edge", 0, "K")]),
        np.asarray([1e6]),  # unreachable
        readout=readout,
        steps=300,
    )
    assert isinstance(out, AbstentionResult)
    assert "reachability abstention" in out.reason
    assert "NUDGE-LIM-013" in out.reason


# --------------------------------------------------------------------------- #
# (4) safety gate — a flip-ON intervention that crosses the fold is flagged.
# --------------------------------------------------------------------------- #
def test_safety_gate_flags_a_fold_crossing_flip() -> None:
    """Flipping a bistable switch ON by raising basal crosses the fold → HIGH RISK."""
    readout = Readout.identity(1)
    bistable = ras_switch_1node(n=6.0, K=1.0, vmax=2.0, basal=0.05)
    target_high = flip_target(bistable, to="high", readout=readout)
    fit = CircuitFit(
        circuit=bistable,
        free=[("species", 0, "basal"), ("edge", 0, "K"), ("edge", 0, "vmax")],
    )
    plan = design(fit, target_high, readout=readout, steps=600, start="low")
    assert isinstance(plan, InterventionPlan)
    assert plan.achieved_loss < 1e-2  # it did reach the ON state
    assert plan.safety is not None
    assert plan.safety.crosses_fold is True
    assert plan.safety.high_risk_of_instability is True
    assert plan.safety.proximity_before is not None  # the base WAS a bistable switch
    assert "HIGH RISK OF INSTABILITY" in plan.reason


def test_safe_intervention_stays_away_from_the_fold() -> None:
    """Nudging the ON level from the high basin stays bistable → not high-risk."""
    import jax.numpy as jnp

    from nudge.design.invert import _stable_states

    readout = Readout.identity(1)
    bistable = ras_switch_1node(n=6.0, K=1.0, vmax=2.0, basal=0.05)
    high = np.asarray(_stable_states(bistable)[-1], np.float32)
    target = np.asarray(readout.expression(jnp.asarray(high))) * 1.05  # +5% ON level
    plan = design(
        CircuitFit(circuit=bistable, free=[("edge", 0, "vmax")]),
        target,
        readout=readout,
        steps=600,
        start="high",
    )
    assert isinstance(plan, InterventionPlan)
    assert plan.safety is not None
    assert plan.safety.crosses_fold is False
    assert plan.safety.high_risk_of_instability is False


# --------------------------------------------------------------------------- #
# (4b) REGRESSION LOCK (NUDGE-LIM-013 / FAILSAFE_REDTEAM_3 HOLE 3) — the safety gate
# must fire on an ABSOLUTE near-fold landing, not only a relative delta > margin.
# Red-team repro: scripts/redteam/design_safety_gate_absolute_proximity.py.
# --------------------------------------------------------------------------- #
def test_safety_gate_flags_sub_margin_push_across_near_fold() -> None:
    """A sub-margin proximity rise that lands the switch AT/ABOVE ``NEAR_FOLD`` must be
    flagged high-risk (absolute check) — never "OK, stays away from the fold" — and must
    AGREE with ``classify_robustness`` on the identical intervened circuit.

    The deterministic red-team construction: base ``ras_switch_1node(n=2, vmax=3, K=1.5)``
    (proximity ~0.500, robust); the reachable inversion scales K to ~1.0 to hit the ON
    level, landing at proximity ~0.589 (near-fold) — a rise of only ~0.089 < margin 0.15.
    """
    from nudge.design.invert import _rebuild
    from nudge.inference.bifurcation import (
        NEAR_FOLD,
        bifurcation_proximity,
        classify_robustness,
    )

    base = ras_switch_1node(n=2.0, vmax=3.0, K=1.5, basal=0.05)
    near_variant = ras_switch_1node(n=2.0, vmax=3.0, K=1.0, basal=0.05)
    target = flip_target(near_variant, to="high")

    fit = CircuitFit(circuit=base, free=[("edge", 0, "K")], is_reliable=True)
    plan = design(
        fit, target, free=[("edge", 0, "K")], start="high", steps=400,
        l1=1e-3, tol=0.3, margin=0.15,
    )
    assert isinstance(plan, InterventionPlan)
    assert plan.safety is not None
    # The delta stayed BELOW the relative margin (that is the whole point of the hole).
    assert plan.safety.delta is not None and plan.safety.delta < 0.15
    # ...yet the ABSOLUTE landing is in the near-fold regime → high risk, NOT "safe".
    assert plan.safety.proximity_after is not None
    assert plan.safety.proximity_after >= NEAR_FOLD
    assert plan.safety.near_fold is True
    assert plan.safety.high_risk_of_instability is True
    assert "HIGH RISK OF INSTABILITY" in plan.reason
    assert "near-fold" in plan.reason
    assert "stays away from the fold" not in plan.reason

    # The safety gate now AGREES with classify_robustness on the same circuit.
    vals = np.array([1.5 * plan.deltas[0][2]]) if plan.deltas else np.array([1.5])
    after_circuit = _rebuild(base, [("edge", 0, "K")], vals)
    after_call, _ = classify_robustness(bifurcation_proximity(after_circuit))
    assert after_call == "near-fold"


def test_positive_control_robust_intervention_below_near_fold_stays_ok() -> None:
    """POSITIVE CONTROL (no over-abstention): a reachable intervention whose intervened
    proximity stays BELOW ``NEAR_FOLD`` must still be cleared "OK, stays away from the
    fold" — the absolute gate must not fire on a genuinely-robust switch.

    Same base (K=1.5, proximity ~0.500); target the ON level of the K=1.2 variant
    (proximity ~0.498 < 0.55) — a robust-side move near, but below, the near-fold cut.
    """
    from nudge.inference.bifurcation import NEAR_FOLD

    base = ras_switch_1node(n=2.0, vmax=3.0, K=1.5, basal=0.05)
    robust_variant = ras_switch_1node(n=2.0, vmax=3.0, K=1.2, basal=0.05)
    target = flip_target(robust_variant, to="high")

    fit = CircuitFit(circuit=base, free=[("edge", 0, "K")], is_reliable=True)
    plan = design(
        fit, target, free=[("edge", 0, "K")], start="high", steps=400,
        l1=1e-3, tol=0.3, margin=0.15,
    )
    assert isinstance(plan, InterventionPlan)
    assert plan.safety is not None
    assert plan.safety.proximity_after is not None
    assert plan.safety.proximity_after < NEAR_FOLD  # genuinely robust
    assert plan.safety.near_fold is False
    assert plan.safety.high_risk_of_instability is False
    assert plan.safety.crosses_fold is False
    assert "stays away from the fold" in plan.reason


def test_safe_branch_carries_one_sided_lower_bound_caveat() -> None:
    """The "OK" reason must carry the one-sided-lower-bound caveat whenever
    ``proximity_after`` is a lower bound (``one_sided``) — the aggravating factor from
    HOLE 3: the falsely-reassuring number must not be presented as a point estimate.

    Unit-level: build a SAFE ``SafetyReport`` (below near-fold, not high-risk) that IS
    one-sided and confirm ``_circuit_reason`` hedges it (NUDGE-LIM-012).
    """
    from nudge.design.invert import SafetyReport, _circuit_reason

    safe_one_sided = SafetyReport(
        proximity_before=0.20,
        proximity_after=0.40,
        delta=0.20,
        one_sided=True,
        high_risk_of_instability=False,
        crosses_fold=False,
        near_fold=False,
    )
    reason = _circuit_reason([(("edge", 0, "vmax"), 0.1, 1.1)], 0.0, safe_one_sided)
    assert "stays away from the fold" in reason
    assert "one-sided LOWER bound" in reason
    assert "may be higher" in reason


# --------------------------------------------------------------------------- #
# (5) curve-level — closed-form Hill inversion + reachability abstain.
# --------------------------------------------------------------------------- #
def test_curve_inversion_round_trips_to_a_dose() -> None:
    """Invert a synthetic switch curve to the dose achieving y = floor + amp/2 (≈ K)."""
    k, n, amp, floor = 1.5, 4.0, 0.8, 0.1
    d = np.linspace(0.1, 5.0, 12)
    y = _hill_repress(d, floor, amp, k, n)
    fit = fit_dose_response(d, y, direction="repress", n_boot=100, seed=0)
    res = DoseResponseResult(fit=fit, call="switch", reason="synthetic switch")
    assert res.is_reliable is True

    y_target = fit.floor + 0.5 * fit.amp
    plan = design(res, y_target)
    assert isinstance(plan, InterventionPlan)
    assert plan.mode == "dose"
    assert plan.safety is None  # curve mode has NO safety gate (stated)
    # At the returned dose the Hill predicts y_target back (a genuine round trip).
    y_back = _hill_repress(
        np.asarray([plan.dose]), fit.floor, fit.amp, fit.k_threshold, fit.n
    )[0]
    assert y_back == pytest.approx(y_target, rel=1e-3)
    # y = floor + amp/2 lands at the threshold K.
    assert plan.dose == pytest.approx(fit.k_threshold, rel=0.05)


def test_curve_inversion_abstains_out_of_range() -> None:
    """A target below the floor is unreachable → reachability abstention (no dose)."""
    k, n, amp, floor = 1.5, 4.0, 0.8, 0.1
    d = np.linspace(0.1, 5.0, 12)
    y = _hill_repress(d, floor, amp, k, n)
    fit = fit_dose_response(d, y, direction="repress", n_boot=100, seed=0)
    res = DoseResponseResult(fit=fit, call="switch", reason="synthetic switch")
    out = design(res, fit.floor - 0.1)  # below the achievable floor
    assert isinstance(out, AbstentionResult)
    assert "outside the curve's achievable range" in out.reason


# --------------------------------------------------------------------------- #
# (6) needs_data — invert the REAL OCT4 dose-response fit.
# --------------------------------------------------------------------------- #
@pytest.mark.needs_data
@pytest.mark.skipif(
    not os.path.exists(_OCT4_NANOG_H5AD), reason="OCT4/NANOG ESC.h5ad not present"
)
def test_design_inverts_real_oct4_fit() -> None:
    """Lock-in: OCT4's real switch fit inverts to a knockdown dose within its range."""
    import anndata as ad

    from nudge.inference.bridge import knockdown_dose_response

    adata = ad.read_h5ad(_OCT4_NANOG_H5AD)
    sig = ["SOX2", "LIN28A", "UTF1", "DNMT3B", "TDGF1", "ZFP42", "SALL4"]
    present = [g for g in sig if g in set(map(str, adata.var_names))]
    dose, resp = knockdown_dose_response(
        adata, target_gene="POU5F1", signature=present, group_prefix="OCT4"
    )
    fit = fit_dose_response(dose, resp, direction="repress", n_boot=200, seed=0)
    res = DoseResponseResult(fit=fit, call="switch", reason="real OCT4 switch")
    assert res.is_reliable is True

    # Target: a 25%-of-range drop in self-renewal — squarely reachable.
    y_target = fit.floor + 0.75 * fit.amp
    plan = design(res, y_target)
    assert isinstance(plan, InterventionPlan)
    assert plan.mode == "dose"
    assert plan.dose > 0.0  # a positive knockdown fraction

    # And an out-of-range target (below the fully-silenced floor) abstains.
    out = design(res, fit.floor - 0.2)
    assert isinstance(out, AbstentionResult)
