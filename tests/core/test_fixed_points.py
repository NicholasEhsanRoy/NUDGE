"""Circuit.fixed_points / transition_state — the decoupled saddle finder (1-D + N-D).

Underpins the multi-basin transition mode (the gain-gate probe). The safety contract
matters as much as the numbers: monostable and unsupported topologies must return
gracefully (no exception, no fabricated saddle) so the fit falls back to abstention
rather than feeding a bogus root into a gradient step. fixed_points returns
``[(state_vector, label), ...]``; transition_state returns the index-1 saddle vector.
"""

from __future__ import annotations

import pytest

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef


def _switch(*, K: float = 1.0, n: float = 6.0, vmax: float = 2.0) -> Circuit:
    return Circuit(
        [SpeciesDef("SW", basal=0.05, decay=1.0)],
        [EdgeDef(0, 0, "hill_activation", K=K, n=n, vmax=vmax)],
    )


def _toggle(
    *, n: float = 4.0, vmax: float = 2.0, basal_b: float = 0.05, vmax_b: float = 2.0
) -> Circuit:
    """2-node mutual-inhibition (Gardner-Collins) toggle switch."""
    return Circuit(
        [
            SpeciesDef("A", basal=0.05, decay=1.0),
            SpeciesDef("B", basal=basal_b, decay=1.0),
        ],
        [
            EdgeDef(1, 0, "hill_repression", K=1.0, n=n, vmax=vmax),
            EdgeDef(0, 1, "hill_repression", K=1.0, n=n, vmax=vmax_b),
        ],
    )


# --- 1-species self-activation: unchanged math, now a vector return -----------
def test_bistable_switch_three_fixed_points() -> None:
    fps = _switch().fixed_points()
    assert fps is not None and len(fps) == 3
    states = [float(s[0]) for s, _ in fps]  # length-1 state vectors
    labels = [lab for _, lab in fps]
    assert states[0] == pytest.approx(0.05, abs=0.02)
    assert states[1] == pytest.approx(0.975, abs=0.05)
    assert states[2] == pytest.approx(2.02, abs=0.05)
    assert labels == ["stable", "saddle-index1", "stable"]
    sad = _switch().transition_state()
    assert sad is not None and sad.shape == (1,)
    assert float(sad[0]) == pytest.approx(states[1], abs=1e-6)


def test_gain_collapse_monostable_no_saddle() -> None:
    fps = _switch(n=1.2).fixed_points()
    assert fps is not None and len(fps) == 1
    assert float(fps[0][0][0]) == pytest.approx(1.1, abs=0.15)
    assert _switch(n=1.2).transition_state() is None


def test_threshold_ceiling_collapse_low_monostable() -> None:
    for circ in (_switch(K=3.0), _switch(vmax=0.6)):
        fps = circ.fixed_points()
        assert fps is not None and len(fps) == 1
        assert float(fps[0][0][0]) == pytest.approx(0.05, abs=0.03)
        assert circ.transition_state() is None


# --- N-D toggle switch: the generalization ------------------------------------
def test_symmetric_toggle_two_stable_one_saddle() -> None:
    fps = _toggle().fixed_points()
    assert fps is not None
    labels = [lab for _, lab in fps]
    assert labels.count("stable") == 2
    assert labels.count("saddle-index1") == 1
    sad = _toggle().transition_state()
    assert sad is not None and sad.shape == (2,)
    # symmetric → the saddle sits on the diagonal separatrix, near [1, 1]
    assert float(sad[0]) == pytest.approx(float(sad[1]), abs=0.05)
    assert float(sad[0]) == pytest.approx(1.0, abs=0.2)


def test_asymmetric_toggle_saddle_moves_off_diagonal() -> None:
    sad = _toggle(basal_b=0.15, vmax_b=1.6).transition_state()
    assert sad is not None and sad.shape == (2,)
    assert abs(float(sad[0]) - float(sad[1])) > 0.05  # off the diagonal


def test_monostable_toggle_no_saddle() -> None:
    # Low Hill coefficient destroys bistability → no saddle (graceful).
    assert _toggle(n=1.0).transition_state() is None


def test_unsupported_topologies_return_gracefully() -> None:
    # 1-species linear (non-Hill) self-edge is out of the 1-D scope → None.
    linear = Circuit(
        [SpeciesDef("SW", basal=0.05, decay=1.0)],
        [EdgeDef(0, 0, "linear", weight=1.0)],
    )
    assert linear.fixed_points() is None
    # A 2-species feedforward switch is monostable: the finder runs (not None) but
    # finds no saddle — so the transition mode / gain gate abstain, never crash.
    feedforward = Circuit(
        [
            SpeciesDef("IN", basal=1.0, decay=1.0),
            SpeciesDef("SW", basal=0.05, decay=1.0),
        ],
        [EdgeDef(0, 1, "hill_activation", K=1.0, n=6.0, vmax=2.0)],
    )
    assert feedforward.transition_state() is None
