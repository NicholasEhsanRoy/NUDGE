"""Multi-reporter joint attribution: the degeneracy break, the guard, the fail-safe.

Three things must hold. (a) **The degeneracy break** — on synthetic ground truth where
ONE latent switch carries a known threshold- / gain- / ceiling-only perturbation, seen
through M reporters of heterogeneous gain, the JOINT multi-reporter fit RESOLVES the
mechanism where a SINGLE reporter ABSTAINS (`unresolved`) — the measured K⇄v_max
degeneracy (FINDINGS §2) broken by the panel. (b) **The consistency guard** — a panel a
single latent CANNOT explain (a reporter secretly reads a different latent) is refused
`off-model` (NUDGE-LIM-014), never averaged into a call. (c) **Fail-safe** — across a
sweep of mechanisms / factors / noise / seeds, the joint fit NEVER emits a confident
wrong mechanism; it is correct or it abstains.
"""

from __future__ import annotations

import pytest

from nudge.inference.multi_reporter import (
    attribute_multi_reporter,
    fit_multi_reporter,
    simulate_reporter_panel,
)

_MECHS = ("threshold", "gain", "ceiling")


# --------------------------------------------------------------------------- #
# (a) the headline: joint resolves where a single reporter abstains
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mechanism", _MECHS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_joint_resolves_where_single_abstains(mechanism: str, seed: int) -> None:
    panel = simulate_reporter_panel(
        mechanism=mechanism, n_reporters=4, factor=3.0, seed=seed
    )
    joint = attribute_multi_reporter(panel, n_boot=80, seed=seed)
    single = attribute_multi_reporter(panel[:1], n_boot=80, seed=seed)

    # The JOINT panel recovers the ground-truth mechanism ...
    assert joint.call == mechanism, (mechanism, joint.call, joint.reason)
    # ... while the SINGLE reporter cannot separate the knobs (the degeneracy).
    assert single.call == "unresolved", (mechanism, single.call, single.reason)
    # And the resolved joint call is trustworthy enough to invert.
    assert joint.is_reliable
    assert not single.is_reliable


def test_single_reporter_reason_names_the_degeneracy() -> None:
    panel = simulate_reporter_panel(mechanism="ceiling", n_reporters=1, seed=0)
    res = attribute_multi_reporter(panel, n_boot=60, seed=0)
    assert res.call == "unresolved"
    assert "single reporter" in res.reason.lower()
    assert not res.fit.pinned_affine


# --------------------------------------------------------------------------- #
# (b) the consistency guard: an inconsistent panel abstains off-model
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hidden", [0, 2, 3])
def test_inconsistent_panel_is_off_model(hidden: int) -> None:
    # One reporter secretly reads a DIFFERENT latent (a shifted K) — the panel cannot be
    # explained by a single shared latent, so NUDGE must decline rather than average.
    panel = simulate_reporter_panel(
        mechanism="ceiling", n_reporters=4, hidden_latent_reporter=hidden, seed=0
    )
    res = attribute_multi_reporter(panel, n_boot=60, seed=0)
    assert res.call == "off-model", res.reason
    assert "single latent" in res.reason.lower()
    assert not res.is_reliable


# --------------------------------------------------------------------------- #
# (c) no-effect, and the fail-safe never-wrong sweep
# --------------------------------------------------------------------------- #
def test_no_effect_reads_no_effect() -> None:
    panel = simulate_reporter_panel(mechanism="none", n_reporters=4, seed=0)
    res = attribute_multi_reporter(panel, n_boot=80, seed=0)
    assert res.call in ("no-effect", "unresolved"), res.reason
    assert res.call not in _MECHS  # never a mechanism when nothing moved


def test_fail_safe_never_a_confident_wrong_call() -> None:
    """The crown-jewel guarantee: 0 confident WRONG mechanism calls across the sweep."""
    wrong = 0
    resolved = 0
    for mechanism in _MECHS:
        for factor in (2.5, 3.0, 4.0):
            for noise in (0.03, 0.06):
                for seed in range(3):
                    panel = simulate_reporter_panel(
                        mechanism=mechanism,
                        n_reporters=4,
                        factor=factor,
                        noise=noise,
                        seed=seed,
                    )
                    res = attribute_multi_reporter(panel, n_boot=40, seed=seed)
                    if res.call in _MECHS:
                        resolved += 1
                        if res.call != mechanism:
                            wrong += 1
    assert wrong == 0, f"{wrong} confident-wrong calls"
    # And the panel does not merely abstain on everything — it resolves the clear cases.
    assert resolved >= 40


def test_bad_input_raises() -> None:
    panel = simulate_reporter_panel(mechanism="gain", n_reporters=2, seed=0)
    # too few dose points
    short = [type(panel[0])(name="x", dose=[0, 1, 2], control=[0, 1, 2],
                            perturbed=[0, 1, 2])]
    with pytest.raises(ValueError):
        fit_multi_reporter(short, n_boot=0)
    with pytest.raises(ValueError):
        fit_multi_reporter([], n_boot=0)
