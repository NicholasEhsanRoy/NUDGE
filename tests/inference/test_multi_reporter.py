"""Multi-reporter joint attribution: the degeneracy break, the guard, the fail-safe.

Three things must hold. (a) **The degeneracy break** — on synthetic ground truth where
ONE latent switch carries a known threshold- / gain- / ceiling-only perturbation, seen
through M reporters of heterogeneous gain, the JOINT multi-reporter fit RESOLVES the
mechanism where a SINGLE reporter ABSTAINS (`unresolved`) — the measured K⇄v_max
degeneracy (FINDINGS §2) broken by the panel. (b) **The consistency guard** — a panel a
single latent CANNOT explain (a reporter secretly reads a different latent) is refused
`off-model` (NUDGE-LIM-014), never averaged into a call. (c) **Fail-safe** — across a
sweep of mechanisms / factors / noise / seeds, the joint fit NEVER emits a confident
wrong mechanism; it is correct or it abstains. (d) **The floor-consistency gate**
(NUDGE-LIM-014, P2) — a per-condition multiplicative batch/depth scale on the perturbed
panel aliases 1:1 onto a `ceiling`; the OFF baseline separates a genuine ceiling (floor
fixed) from a batch (floor rescaled with the ON scale), and where the floors are
unmeasurable NUDGE abstains rather than confidently mis-call `ceiling`.
"""

from __future__ import annotations

import numpy as np
import pytest

from nudge.inference.multi_reporter import (
    ReporterObservation,
    attribute_multi_reporter,
    fit_multi_reporter,
    simulate_reporter_panel,
)

_MECHS = ("threshold", "gain", "ceiling")


def _batch_scaled_no_effect(
    seed: int, factor: float, floor_range: tuple[float, float] = (0.0, 0.02)
) -> list[ReporterObservation]:
    """A ``mechanism="none"`` panel whose PERTURBED curves are all scaled by one factor.

    The red-team confound (``scripts/redteam/multi_reporter_batch_confound.py``): a
    per-condition batch/depth/instrument-gain difference between the control-condition and
    perturbed-condition measurement, consistent across the panel. Truth = no-effect; it
    aliases onto a shared latent-ceiling change ``A = factor``. NUDGE must NOT call
    ``ceiling`` (NUDGE-LIM-014).
    """
    panel = simulate_reporter_panel(
        mechanism="none", n_reporters=5, k_wt=20.0, n_wt=4.0,
        gain_range=(0.5, 3.0), floor_range=floor_range, noise=0.02, seed=seed,
    )
    return [
        ReporterObservation(
            name=o.name, dose=o.dose, control=o.control,
            perturbed=np.asarray(o.perturbed, dtype=float) * factor,
        )
        for o in panel
    ]


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


# --------------------------------------------------------------------------- #
# (d) the floor-consistency gate: a per-condition batch scale is NOT a ceiling
#     (the P2 red-team regression lock — NUDGE-LIM-014)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("factor", [0.5, 0.6, 0.75])
def test_batch_scale_is_not_confidently_a_ceiling(seed: int, factor: float) -> None:
    """DECOY (P2): a uniform batch scale on the perturbed panel must NOT be `ceiling`.

    truth = no-effect; the confound aliases onto a ceiling change `A = factor` with margins
    that crush every prior gate. The floor-consistency gate catches it — either the OFF
    baseline is rescaled with the ON scale (a batch) or the tiny floors are unmeasurable —
    so NUDGE abstains. A resolved MECHANISM here is the confident-wrong hole.
    """
    panel = _batch_scaled_no_effect(seed, factor)
    res = attribute_multi_reporter(panel, n_boot=120, seed=seed)
    assert res.call not in _MECHS, (factor, res.call, res.reason)
    assert not res.is_reliable
    # and it is caught by the floor-consistency gate (not some unrelated abstention)
    assert "NUDGE-LIM-014" in res.reason


def test_batch_scale_at_realistic_floors_is_not_a_ceiling() -> None:
    """The confound is NOT a tiny-floor artifact: measurable floors → caught by coupling."""
    for seed in range(2):
        for factor in (0.5, 0.75):
            panel = _batch_scaled_no_effect(seed, factor, floor_range=(0.2, 0.6))
            res = attribute_multi_reporter(panel, n_boot=120, seed=seed)
            assert res.call not in _MECHS, (seed, factor, res.call, res.reason)
            # measurable floors → the OFF-baseline moved WITH the ON scale (batch signature)
            assert res.fit.floor_measurability > 0.6
            assert res.fit.off_on_coupling > 0.5


def test_genuine_ceiling_resolves_with_off_baseline_fixed() -> None:
    """No over-abstention: a REAL ceiling (floor fixed) still resolves at realistic floors.

    The positive control the floor-consistency gate must NOT block — a genuine latent-ceiling
    change leaves each reporter's OFF baseline unchanged (`off_on_coupling` ≈ 0).
    """
    for seed in range(3):
        panel = simulate_reporter_panel(mechanism="ceiling", n_reporters=4,
                                        factor=3.0, seed=seed)
        res = attribute_multi_reporter(panel, n_boot=80, seed=seed)
        assert res.call == "ceiling", (seed, res.call, res.reason)
        assert res.fit.floor_measurability > 0.6
        assert res.fit.off_on_coupling < 0.5


def test_floorless_ceiling_abstains_the_documented_bound() -> None:
    """The residual BOUND (NUDGE-LIM-014): on a (near-)zero-floor panel a genuine ceiling
    is INSEPARABLE from a per-condition batch scale, so NUDGE abstains on BOTH — the honest
    conservative outcome (never a confident-wrong), locked so it cannot silently start
    resolving without an independent depth anchor.
    """
    for seed in range(3):
        panel = simulate_reporter_panel(mechanism="ceiling", n_reporters=5, factor=3.0,
                                        floor_range=(0.0, 0.0), noise=0.02, seed=seed)
        res = attribute_multi_reporter(panel, n_boot=60, seed=seed)
        assert res.call == "unresolved", (seed, res.call, res.reason)
        assert res.fit.floor_measurability < 0.6  # unmeasurable → no depth anchor


@pytest.mark.xfail(strict=True, reason="BOUND (NUDGE-LIM-014): a per-condition scale on a "
                   "(near-)zero-floor panel is indistinguishable from a genuine ceiling "
                   "without an independent depth anchor (spike-in / housekeeping / "
                   "no-response reporter), which NUDGE does not yet ingest. If a depth "
                   "anchor is added and a floorless ceiling becomes resolvable, this XPASSes "
                   "— update NUDGE-LIM-014.")
def test_floorless_genuine_ceiling_cannot_be_resolved_bound() -> None:
    """Marks the irreducible residual: ideally a floorless genuine ceiling would resolve,
    but with no measurable floor it cannot be told from a batch — so it does not."""
    panel = simulate_reporter_panel(mechanism="ceiling", n_reporters=5, factor=3.0,
                                    floor_range=(0.0, 0.0), noise=0.02, seed=0)
    res = attribute_multi_reporter(panel, n_boot=60, seed=0)
    assert res.call == "ceiling"


def test_bad_input_raises() -> None:
    panel = simulate_reporter_panel(mechanism="gain", n_reporters=2, seed=0)
    # too few dose points
    short = [type(panel[0])(name="x", dose=[0, 1, 2], control=[0, 1, 2],
                            perturbed=[0, 1, 2])]
    with pytest.raises(ValueError):
        fit_multi_reporter(short, n_boot=0)
    with pytest.raises(ValueError):
        fit_multi_reporter([], n_boot=0)
