"""The abstention gate logic — including the linear-baseline parsimony gate."""

from __future__ import annotations

from nudge.core.vocabulary import MechanismClass
from nudge.inference.classify import decide


def _decide(losses: dict[str, float], linear_loss: float, wt_distance: float = 1.0):
    return decide(
        "p", losses, linear_loss, wt_distance,
        noise_margin=0.05, effect_margin=0.02,
    )


def test_threshold_wins() -> None:
    call = _decide({"K": 0.10, "n": 0.30, "vmax": 0.35}, linear_loss=0.40)
    assert call.mechanism is MechanismClass.THRESHOLD
    assert call.confidence > 0


def test_gain_wins() -> None:
    call = _decide({"K": 0.30, "n": 0.10, "vmax": 0.35}, linear_loss=0.40)
    assert call.mechanism is MechanismClass.GAIN


def test_ceiling_wins() -> None:
    call = _decide({"K": 0.35, "n": 0.30, "vmax": 0.10}, linear_loss=0.40)
    assert call.mechanism is MechanismClass.CEILING


def test_no_effect_gate() -> None:
    losses = {"K": 0.1, "n": 0.3, "vmax": 0.35}
    call = _decide(losses, linear_loss=0.4, wt_distance=0.01)
    assert call.mechanism is MechanismClass.NO_EFFECT


def test_linear_baseline_gate_gives_off_model() -> None:
    # Best mechanistic (0.38) does NOT beat linear (0.40) beyond the noise floor (0.05).
    call = _decide({"K": 0.38, "n": 0.42, "vmax": 0.45}, linear_loss=0.40)
    assert call.mechanism is MechanismClass.OFF_MODEL
    assert "linear baseline" in call.rationale


def test_unresolved_gate() -> None:
    # K and n are within the noise floor of each other → cannot resolve.
    call = _decide({"K": 0.10, "n": 0.12, "vmax": 0.35}, linear_loss=0.40)
    assert call.mechanism is MechanismClass.UNRESOLVED
