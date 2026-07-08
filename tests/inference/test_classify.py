"""Abstention gate logic: the circuit-level linear-baseline gate + per-perturbation."""

from __future__ import annotations

from nudge.core.vocabulary import MechanismClass
from nudge.inference.classify import (
    decide,
    decide_with_transition,
    switch_detected,
)


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


# ── The saddle transition-mode gain gate (multi-basin path) ──────────────────
def _decide_t(
    losses: dict[str, float],
    *,
    w_trans: float | None,
    n_species: int = 1,
    wt_distance: float = 1.0,
    off_model_loss: float = 10.0,
):
    return decide_with_transition(
        "p", losses, wt_distance,
        noise_margin=0.05, effect_margin=0.02, off_model_loss=off_model_loss,
        transition_weight=w_trans, n_species=n_species, gain_wtrans_tau=0.5,
    )


def test_gain_gate_promotes_thin_margin_to_gain() -> None:
    # Loss argmin ties thin (n barely < vmax) → decide() would say UNRESOLVED; the
    # decisive transition weight promotes it to GAIN.
    call = _decide_t({"K": 0.06, "n": 0.0090, "vmax": 0.0091}, w_trans=0.87)
    assert call.mechanism is MechanismClass.GAIN


def test_gain_gate_silent_on_low_wtrans() -> None:
    # Ceiling-like: low transition weight → the gate stays out of the way.
    call = _decide_t({"K": 0.35, "n": 0.30, "vmax": 0.10}, w_trans=0.01)
    assert call.mechanism is MechanismClass.CEILING


def test_gain_gate_isolated_to_one_species() -> None:
    # N-species: no saddle finder → gate never fires even with high w_trans (FM2).
    call = _decide_t({"K": 0.06, "n": 0.0090, "vmax": 0.0091}, w_trans=0.9, n_species=2)
    assert call.mechanism is not MechanismClass.GAIN


def test_gain_gate_never_overrides_no_effect_or_off_model() -> None:
    # Even a high w_trans cannot manufacture a positive from an abstaining base call.
    ne = _decide_t({"K": 0.1, "n": 0.1, "vmax": 0.3}, w_trans=0.9, wt_distance=0.01)
    assert ne.mechanism is MechanismClass.NO_EFFECT
    om = _decide_t({"K": 5.0, "n": 6.0, "vmax": 7.0}, w_trans=0.9, off_model_loss=1.0)
    assert om.mechanism is MechanismClass.OFF_MODEL


def test_gain_gate_none_weight_falls_through() -> None:
    call = _decide_t({"K": 0.35, "n": 0.30, "vmax": 0.10}, w_trans=None)
    assert call.mechanism is MechanismClass.CEILING
