"""Circuit.fixed_points / transition_state — the decoupled saddle finder.

These underpin the multi-basin transition mode (the gain-gate probe). The safety
contract matters as much as the numbers: monostable and N-species circuits must
return gracefully (no exception, no fabricated saddle), so the fit falls back to
abstention rather than feeding a bogus root into a gradient step.
"""

from __future__ import annotations

import pytest

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef


def _switch(*, K: float = 1.0, n: float = 6.0, vmax: float = 2.0) -> Circuit:
    return Circuit(
        [SpeciesDef("SW", basal=0.05, decay=1.0)],
        [EdgeDef(0, 0, "hill_activation", K=K, n=n, vmax=vmax)],
    )


def test_bistable_switch_has_three_fixed_points() -> None:
    roots = _switch().fixed_points()
    assert roots is not None and len(roots) == 3
    low, saddle, high = roots
    assert low == pytest.approx(0.05, abs=0.02)
    assert saddle == pytest.approx(0.975, abs=0.05)
    assert high == pytest.approx(2.02, abs=0.05)
    # transition_state is the middle (unstable) root.
    assert _switch().transition_state() == pytest.approx(saddle, abs=1e-6)


def test_gain_collapse_is_monostable_intermediate() -> None:
    # A gain reduction (n: 6 -> 1.2) destroys bistability: one intermediate root,
    # right where the WT saddle sat (~1.1). transition_state must be None (no saddle).
    roots = _switch(n=1.2).fixed_points()
    assert roots is not None and len(roots) == 1
    assert roots[0] == pytest.approx(1.1, abs=0.15)
    assert _switch(n=1.2).transition_state() is None


def test_threshold_and_ceiling_collapse_low_and_monostable() -> None:
    # Threshold (K*3) and ceiling (vmax*0.3) both drop to a single low fixed point.
    for circ in (_switch(K=3.0), _switch(vmax=0.6)):
        roots = circ.fixed_points()
        assert roots is not None and len(roots) == 1
        assert roots[0] == pytest.approx(0.05, abs=0.03)
        assert circ.transition_state() is None


def test_n_species_and_non_self_return_none_safely() -> None:
    # No general N-D saddle finder → None (caller abstains), never an exception.
    feedforward = Circuit(
        [
            SpeciesDef("IN", basal=1.0, decay=1.0),
            SpeciesDef("SW", basal=0.05, decay=1.0),
        ],
        [EdgeDef(0, 1, "hill_activation", K=1.0, n=6.0, vmax=2.0)],
    )
    assert feedforward.fixed_points() is None
    assert feedforward.transition_state() is None
    # A 1-species linear (non-Hill) self-edge is also out of scope → None.
    linear = Circuit(
        [SpeciesDef("SW", basal=0.05, decay=1.0)],
        [EdgeDef(0, 0, "linear", weight=1.0)],
    )
    assert linear.fixed_points() is None
