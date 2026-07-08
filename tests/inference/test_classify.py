"""Abstention gate logic: the circuit-level linear-baseline gate + per-perturbation."""

from __future__ import annotations

from nudge.core.vocabulary import MechanismClass
from nudge.inference.classify import decide, switch_detected


def _decide(
    losses: dict[str, float], wt_distance: float = 1.0, off_model_loss: float = 10.0
):
    return decide(
        "p", losses, wt_distance,
        noise_margin=0.05, effect_margin=0.02, off_model_loss=off_model_loss,
    )


# ── The linear-baseline / parsimony gate (circuit level) ─────────────────────
def test_switch_detected_when_mechanistic_beats_linear() -> None:
    assert switch_detected(0.10, 0.40, noise_margin=0.05) is True


def test_no_switch_when_linear_ties() -> None:
    # mechanistic (0.38) does not beat linear (0.40) beyond the noise floor (0.05)
    assert switch_detected(0.38, 0.40, noise_margin=0.05) is False


# ── Per-perturbation attribution + abstention ────────────────────────────────
def test_threshold_wins() -> None:
    call = _decide({"K": 0.10, "n": 0.30, "vmax": 0.35})
    assert call.mechanism is MechanismClass.THRESHOLD
    assert call.confidence > 0


def test_gain_wins() -> None:
    call = _decide({"K": 0.30, "n": 0.10, "vmax": 0.35})
    assert call.mechanism is MechanismClass.GAIN


def test_ceiling_wins() -> None:
    call = _decide({"K": 0.35, "n": 0.30, "vmax": 0.10})
    assert call.mechanism is MechanismClass.CEILING


def test_no_effect_gate() -> None:
    call = _decide({"K": 0.10, "n": 0.30, "vmax": 0.35}, wt_distance=0.01)
    assert call.mechanism is MechanismClass.NO_EFFECT


def test_off_model_gate_on_poor_absolute_fit() -> None:
    call = _decide({"K": 5.0, "n": 6.0, "vmax": 7.0}, off_model_loss=1.0)
    assert call.mechanism is MechanismClass.OFF_MODEL


def test_unresolved_gate() -> None:
    call = _decide({"K": 0.10, "n": 0.12, "vmax": 0.35})
    assert call.mechanism is MechanismClass.UNRESOLVED
