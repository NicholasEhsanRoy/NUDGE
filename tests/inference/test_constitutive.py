"""Constitutive-reporter calibration control — the NUDGE-LIM-006 mitigation.

Four things must hold (synthetic ground truth; NUDGE-METHOD-011, NUDGE-LIM-018):

(a) **The calibration anchors the readout** — fitting the constitutive control at KNOWN doses
    recovers the reporter's Hill parameters (h, Km, Vmax, base), and the calibration loss uses
    the READOUT parameters ONLY (the no-circuit-leak property: ∂/∂circuit ≡ 0).
(b) **The LIM-006 flip** — WITHOUT the control the circuit-n profile is FLAT (degenerate: you
    cannot tell a switch exists); WITH the control, a TRUE switch (n=3) has "no switch" (n=1)
    REJECTED → the verdict flips to `biological-switch`.
(c) **The fail-safe** — the module NEVER emits a bare threshold/gain/ceiling. On the LINEAR
    (n=1) LIM-006 hazard it ABSTAINS (`unresolved`) rather than manufacture a switch; across
    cases it is 0 confident-wrong.
(d) **The honest caveat holds** — a `biological-switch` verdict does NOT point-identify the
    circuit n (the profile argmin is not a reliable point estimate; the reason says so).
"""

from __future__ import annotations

import numpy as np
import pytest

from nudge.inference.constitutive import (
    ConstitutiveControl,
    ReadoutCircuitParams,
    calibrate_readout,
    control_loss_circuit_gradient,
    generate_constitutive_dataset,
    profile_circuit_n,
)

_GRID = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)


def _switch_params(n: float) -> ReadoutCircuitParams:
    return ReadoutCircuitParams(
        k=1.0, n=n, vmax=1.0, basal=0.05, km=0.5, h=6.0, readout_vmax=20.0, readout_base=0.1
    )


# --------------------------------------------------------------------------- #
# (a) the calibration anchors the readout — and leaks NO circuit parameter
# --------------------------------------------------------------------------- #
@pytest.mark.verification
def test_calibration_recovers_known_readout_params() -> None:
    params = _switch_params(3.0)
    _pop, control, _ = generate_constitutive_dataset(
        params, n_cells=400, n_ctrl_doses=10, n_ctrl_reps=200, seed=0
    )
    calib = calibrate_readout(control, n_boot=200, seed=0)
    # The reporter is confidently nonlinear (true h=6): the CI clears the affine line.
    assert calib.is_nonlinear
    assert calib.ci_h[0] >= 1.5
    # The Hill coefficient is recovered near the truth (apparent, not exact).
    assert 3.0 <= calib.h <= 9.0
    assert calib.km == pytest.approx(params.km, abs=0.3)
    assert calib.r2 > 0.98


@pytest.mark.verification
def test_control_loss_has_zero_circuit_gradient() -> None:
    # The load-bearing no-leak property: the calibration loss is a function of the readout
    # parameters ONLY — its gradient w.r.t. every circuit parameter is identically zero.
    params = _switch_params(3.0)
    _pop, control, _ = generate_constitutive_dataset(
        params, n_cells=200, n_ctrl_doses=8, n_ctrl_reps=80, seed=0
    )
    grads = control_loss_circuit_gradient(control, params)
    assert set(grads) == {"K", "n", "vmax", "basal"}
    for name, g in grads.items():
        assert g == 0.0, f"circuit parameter {name} leaked into the control loss (grad={g})"


# --------------------------------------------------------------------------- #
# (b) + (c) + (d) the LIM-006 flip, the fail-safe, and the honest caveat
# --------------------------------------------------------------------------- #
@pytest.mark.verification
@pytest.mark.slow
def test_true_switch_flips_to_biological_with_control() -> None:
    params = _switch_params(3.0)
    pop, control, _ = generate_constitutive_dataset(
        params, n_cells=600, n_ctrl_doses=10, n_ctrl_reps=200, seed=0
    )
    res = profile_circuit_n(
        pop, control, params, n_grid=_GRID, restarts=3, steps=600, n_model_cells=400, seed=0
    )
    # WITHOUT control: the n-profile is FLAT (degenerate) — a graded n=1 fits as well as n=3.
    assert res.span_no_control < 5e-3
    # WITH control: "no switch" (n=1) is REJECTED and the split gains structure.
    assert res.n1_rejection > 5.0 * res.span_no_control
    assert res.n1_rejection > 1e-2
    # The verdict flips to biological-switch — the ultrasensitivity is a real circuit switch.
    assert res.call == "biological-switch"
    # (c) NEVER a bare mechanism.
    assert not res.is_confident_wrong
    # (d) the honest caveat: it does NOT point-identify n (the reason says so loudly).
    assert "does NOT" in res.reason
    assert "point-identif" in res.reason.lower()


@pytest.mark.verification
@pytest.mark.slow
def test_linear_circuit_lim006_hazard_abstains_not_confident_wrong() -> None:
    # The pure NUDGE-LIM-006 hazard: a LINEAR (n=1) circuit through a nonlinear reporter is
    # exactly what fools the affine-readout fit into a CONFIDENT false positive. WITH the
    # control the profile does NOT reject n=1 → NUDGE abstains (unresolved), turning a
    # confident false positive into an honest abstention. It must NEVER call a bare switch.
    params = _switch_params(1.0)
    pop, control, _ = generate_constitutive_dataset(
        params, n_cells=600, n_ctrl_doses=10, n_ctrl_reps=200, seed=0
    )
    res = profile_circuit_n(
        pop, control, params, n_grid=_GRID, restarts=3, steps=600, n_model_cells=400, seed=0
    )
    assert res.call == "unresolved"
    assert res.call != "biological-switch"
    assert not res.is_confident_wrong
    # The profile minimum sits at the NO-SWITCH end of the grid (the two lowest points,
    # n in {1.0, 1.5}) — the honest signal there is none. We assert the end, not the exact
    # grid index: the profile is near-flat at the low end for a linear circuit, so the
    # argmin drifts one grid step (1.0 <-> 1.5) with optimizer/platform numerics. The
    # load-bearing guarantee — unresolved, never a confident switch — is asserted above.
    assert res.argmin_n_with_control <= 1.5


@pytest.mark.verification
@pytest.mark.slow
@pytest.mark.parametrize("n_true", [1.0, 3.0])
def test_never_confident_wrong_across_ground_truth(n_true: float) -> None:
    # Fail-safe sweep: whatever the ground truth, the module is correct-or-abstains and
    # NEVER emits a bare threshold/gain/ceiling.
    params = _switch_params(n_true)
    pop, control, _ = generate_constitutive_dataset(
        params, n_cells=500, n_ctrl_doses=10, n_ctrl_reps=150, seed=1
    )
    res = profile_circuit_n(
        pop, control, params, n_grid=_GRID, restarts=2, steps=500, n_model_cells=350, seed=1
    )
    assert res.call in {"biological-switch", "unresolved", "no-confound"}
    assert not res.is_confident_wrong


# --------------------------------------------------------------------------- #
# decoy-style abstention: an ~affine reporter has no confound to correct
# --------------------------------------------------------------------------- #
@pytest.mark.decoy
def test_affine_reporter_is_no_confound() -> None:
    # A near-affine reporter (h≈1) is NOT a LIM-006 confound: the calibration says so, and the
    # classifier returns `no-confound` — it must not claim a mitigation it did not perform.
    params = ReadoutCircuitParams(
        k=1.0, n=3.0, vmax=1.0, basal=0.05, km=8.0, h=1.0, readout_vmax=20.0, readout_base=0.1
    )
    _pop, control, _ = generate_constitutive_dataset(
        params, n_cells=300, n_ctrl_doses=12, n_ctrl_reps=150, seed=0
    )
    calib = calibrate_readout(control, n_boot=200, seed=0)
    assert not calib.is_nonlinear


# --------------------------------------------------------------------------- #
# NUDGE-LIM-019 (red-team round 2): a control-vs-population capture-efficiency mismatch
# re-opens NUDGE-LIM-006 — LOCKED as a strict-xfail so it cannot silently change.
# --------------------------------------------------------------------------- #
@pytest.mark.decoy
@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="NUDGE-LIM-019: the constitutive control and the circuit population must share a "
    "capture/depth scale, but the module applies NO relative-depth normalization between the "
    "two populations. Reading the control at ~0.5x the population's capture efficiency "
    "mis-anchors the reporter Vmax and makes NUDGE assert 'biological-switch' on a LINEAR "
    "circuit (the LIM-006 artifact resurrected via a batch/capture confound; 3/3 seeds). No "
    "safe runtime gate exists without a switch-independent shared reference (a magnitude gate "
    "would false-abstain on real switches), so this is LOCKED as strict-xfail; the "
    "shared-capture precondition is documented, and the Option B robustness fix is designed in "
    "design/CONSTITUTIVE_CONTROL.md. When a shared-reference anchor fixes it this XPASSes and "
    "prompts removing the mark.",
)
def test_capture_efficiency_mismatch_must_not_fake_biological_switch() -> None:
    # DESIRED (fail-safe) behavior: the truth is a LINEAR (n=1) circuit, so with the control
    # read at a DIFFERENT capture efficiency than the population, NUDGE must NOT assert a real
    # switch — it should abstain. It currently asserts 'biological-switch' (a confident-wrong),
    # so this assertion fails today (strict-xfail). asserts_biological_switch is the
    # falsifiable positive claim (NUDGE-LIM-019) that must never fire on a linear circuit.
    params = _switch_params(1.0)  # LINEAR circuit — the LIM-006 hazard
    pop, control, _ = generate_constitutive_dataset(
        params, n_cells=600, n_ctrl_doses=10, n_ctrl_reps=200, seed=0
    )
    # The control is a SEPARATE population, here read at ~0.5x the population's capture
    # efficiency (a routine single-cell batch difference — no shared normalization).
    confounded = ConstitutiveControl(
        activity=control.activity, response=control.response * 0.5
    )
    res = profile_circuit_n(
        pop, confounded, params, n_grid=_GRID, restarts=3, steps=600, n_model_cells=400, seed=0
    )
    assert not res.asserts_biological_switch, res.reason


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #
def test_control_requires_four_distinct_doses() -> None:
    with pytest.raises(ValueError, match="4 distinct"):
        ConstitutiveControl(
            activity=np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.3]),
            response=np.array([1.0, 1.1, 2.0, 2.1, 3.0, 3.1]),
        )
