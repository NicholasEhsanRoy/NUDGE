"""Comparative / differential attribution — WHICH knob differs between two contexts.

Two layers:

- **fast, no-fit** tests of :func:`classify_differential` on hand-built
  :class:`DifferentialFit`s — the fail-safe gate logic (no-difference / the Δ-model tie /
  the ceiling-depth confound / underpowered), instant.
- **slow, real-fit** tests on synthetic ground truth (a KNOWN single-knob difference
  between two contexts, drawn from the shipped LNA Gaussian mixture): BIC recovers WHICH
  knob differs (or abstains), the depth-aligned-with-context confound abstains, and 0
  confident-wrong across a mechanism / seed sweep.
"""

from __future__ import annotations

import pytest

from nudge.circuits import ras_switch_1node
from nudge.inference.differential import (
    DifferentialFit,
    attribute_differential,
    classify_differential,
    simulate_context_pair,
)

# The synthetic switch + regime the ground-truth pairs use (a resolvable OFF mode so the
# confound guard is not blind; a scale that clears the LNA depth guard).
CIRC = ras_switch_1node(n=6.0, vmax=2.5, K=1.0, basal=0.2)
SCALE, OBS_SD, NCELLS = 25.0, 0.5, 2000
POSITIVE = {"threshold-diff", "gain-diff", "ceiling-diff"}


# --------------------------------------------------------------------------- #
# fast: the classifier gate logic (no LNA fit)
# --------------------------------------------------------------------------- #
def _fit(
    bic: dict[str, float],
    *,
    n_cells: int = 2000,
    lna_ok: bool = True,
    depth_ratio: float = 1.02,
    off_shift_ratio: float = 1.0,
    best_diff: str | None = None,
) -> DifferentialFit:
    """A minimal ``DifferentialFit`` carrying just what ``classify_differential`` reads."""
    if best_diff is None:
        best_diff = min(("n", "K", "vmax"), key=lambda m: bic[m])
    est = {m: {"n": 6.0, "K": 1.0, "vmax": 2.5} for m in ("shared", "n", "K", "vmax")}
    est_b = {m: dict(v) for m, v in est.items()}
    est_b["vmax"] = {"n": 6.0, "K": 1.0, "vmax": 3.2}
    est_b["n"] = {"n": 3.5, "K": 1.0, "vmax": 2.5}
    return DifferentialFit(
        target_edge=0, n_species=1, k_modes=2, n_cells_a=n_cells, n_cells_b=n_cells,
        scale_a=SCALE, obs_sd_a=OBS_SD, scale_b=SCALE * depth_ratio, obs_sd_b=OBS_SD,
        lna_ok_a=lna_ok, lna_ok_b=lna_ok, lna_reason_a="ok", lna_reason_b="ok",
        bic=bic, nll={m: 0.0 for m in bic}, n_params={m: 3 if m == "shared" else 4 for m in bic},
        est_a=est, est_b=est_b, selected=min(bic, key=lambda m: bic[m]), best_diff=best_diff,
        depth_ratio=depth_ratio, off_shift_a=1.0, off_shift_b=off_shift_ratio,
        off_shift_ratio=off_shift_ratio,
    )


def test_classify_no_difference_when_shared_wins() -> None:
    # No Δ model earns its parameter over shared → no-difference.
    fit = _fit({"shared": 100.0, "n": 103.0, "K": 104.0, "vmax": 105.0})
    call, _ = classify_differential(fit)
    assert call == "no-difference"


def test_classify_gain_difference_clean() -> None:
    # Δn clearly beats shared and the other Δ models, OFF fixed → gain-diff.
    fit = _fit({"shared": 200.0, "n": 100.0, "K": 130.0, "vmax": 160.0})
    call, _ = classify_differential(fit)
    assert call == "gain-diff"


def test_classify_tie_between_deltas_abstains() -> None:
    # Two Δ models within resolve_margin → unresolved (the gain⇄threshold confound).
    fit = _fit({"shared": 200.0, "n": 100.0, "K": 102.0, "vmax": 160.0})
    call, reason = classify_differential(fit)
    assert call == "unresolved" and "unidentifiable" in reason


def test_classify_ceiling_clean_is_called() -> None:
    # v_max wins with matched per-context depth → ceiling-diff.
    bic = {"shared": 400.0, "n": 300.0, "K": 250.0, "vmax": 100.0}
    fit = _fit(bic, depth_ratio=1.03, best_diff="vmax")
    call, _ = classify_differential(fit)
    assert call == "ceiling-diff"


def test_classify_depth_confound_abstains() -> None:
    # A depth difference aligned with the context axis + a ceiling winner → unresolved
    # (depth ⇄ ceiling degenerate, NUDGE-LIM-016), never a spurious ceiling-diff.
    bic = {"shared": 400.0, "n": 300.0, "K": 250.0, "vmax": 100.0}
    fit = _fit(bic, depth_ratio=1.9, best_diff="vmax")
    call, reason = classify_differential(fit)
    assert call == "unresolved" and "depth" in reason.lower() and "NUDGE-LIM-016" in reason


def test_classify_depth_confound_no_difference_also_abstains() -> None:
    # A depth difference with no clear mechanism (shared wins) → unresolved: NUDGE cannot
    # certify a masked ceiling isn't hiding in the depth (the confounded ground truth).
    bic = {"shared": 100.0, "n": 103.0, "K": 104.0, "vmax": 105.0}
    fit = _fit(bic, depth_ratio=1.8)
    call, _ = classify_differential(fit)
    assert call == "unresolved"


def test_classify_clean_gain_survives_a_depth_difference() -> None:
    # The one exception: a cleanly-resolved GAIN difference reshapes the distribution
    # (orthogonal to a global scale), so it is still callable despite a depth difference.
    bic = {"shared": 400.0, "n": 100.0, "K": 250.0, "vmax": 300.0}
    fit = _fit(bic, depth_ratio=1.9, best_diff="n")
    call, _ = classify_differential(fit)
    assert call == "gain-diff"


def test_classify_underpowered_abstains() -> None:
    fit = _fit({"shared": 400.0, "n": 100.0, "K": 250.0, "vmax": 300.0}, n_cells=120)
    call, reason = classify_differential(fit)
    assert call == "unresolved" and "underpowered" in reason


def test_classify_lna_untrustworthy_abstains() -> None:
    fit = _fit({"shared": 400.0, "n": 100.0, "K": 250.0, "vmax": 300.0}, lna_ok=False)
    call, reason = classify_differential(fit)
    assert call == "unresolved" and "untrustworthy" in reason


# --------------------------------------------------------------------------- #
# slow: BIC recovery on synthetic ground truth
# --------------------------------------------------------------------------- #
def _pair(mechanism: str, factor: float, *, scale_b: float = SCALE, seed: int = 1):
    return simulate_context_pair(
        CIRC, mechanism=mechanism, factor=factor,
        n_cells=NCELLS, scale_a=SCALE, scale_b=scale_b, obs_sd=OBS_SD, seed=seed,
    )


@pytest.mark.slow
def test_recovers_ceiling_difference() -> None:
    # A raised ceiling in context B (the drug-resistance story) → ceiling-diff.
    a, b = _pair("ceiling", 1.4, seed=1)
    res = attribute_differential(a, b, CIRC, steps=250, seed=0)
    assert res.call == "ceiling-diff"
    assert res.is_reliable


@pytest.mark.slow
def test_recovers_gain_difference() -> None:
    # A rewired gain in context B → gain-diff (a different drug class, not just more dose).
    a, b = _pair("gain", 0.55, seed=1)
    res = attribute_differential(a, b, CIRC, steps=250, seed=0)
    assert res.call == "gain-diff"


@pytest.mark.slow
def test_no_difference_reads_no_difference() -> None:
    # The SAME perturbation, same mechanism in both contexts → no-difference.
    a, b = _pair("none", 1.0, seed=1)
    res = attribute_differential(a, b, CIRC, steps=250, seed=0)
    assert res.call == "no-difference"


@pytest.mark.slow
def test_threshold_recovers_or_abstains() -> None:
    # A threshold shift is the HARDEST to resolve from a bistable snapshot (the stable
    # modes barely move with K — the measured hierarchy gain > ceiling > threshold). NUDGE
    # must recover threshold-diff OR abstain — but NEVER call the wrong mechanism.
    a, b = _pair("threshold", 1.4, seed=1)
    res = attribute_differential(a, b, CIRC, steps=250, seed=0)
    assert res.call in {"threshold-diff", "no-difference", "unresolved"}
    assert res.call not in {"gain-diff", "ceiling-diff"}


@pytest.mark.slow
def test_confound_depth_aligned_with_context_abstains() -> None:
    # THE load-bearing honesty test (NUDGE-LIM-016): a sequencing-depth / batch difference
    # ALIGNED WITH THE CONTEXT AXIS (context B — control AND perturbed — sequenced ~1.6x
    # deeper) but NO real mechanism difference. Depth is degenerate with the ceiling, so
    # NUDGE must ABSTAIN (unresolved), NEVER emit a spurious ceiling-diff (or any mech-diff).
    for seed in (1, 2, 3):
        a, b = _pair("none", 1.0, scale_b=SCALE * 1.6, seed=seed)
        res = attribute_differential(a, b, CIRC, steps=250, seed=0)
        assert res.call not in POSITIVE, f"seed {seed}: spurious {res.call} on a depth confound"
        assert res.call == "unresolved", f"seed {seed}: expected unresolved, got {res.call}"


@pytest.mark.slow
def test_underpowered_context_abstains() -> None:
    # Too few cells in a context → unresolved (cannot separate depth from mechanism).
    a, b = _pair("ceiling", 1.4, seed=1)
    a_small = type(a)(name=a.name, data=a.data[:150], control=a.control[:150])
    res = attribute_differential(a_small, b, CIRC, steps=200, seed=0)
    assert res.call == "unresolved"


@pytest.mark.slow
def test_fail_safe_never_confident_wrong() -> None:
    # The headline safety property: across mechanisms × seeds, NUDGE NEVER names the WRONG
    # mechanism-difference. A correct call or an abstention is fine; a confident-wrong call
    # is not. (Threshold is allowed to abstain; the confound must never read a mechanism.)
    truth = {
        "ceiling": ("ceiling", 1.4, SCALE, "ceiling-diff"),
        "gain": ("gain", 0.55, SCALE, "gain-diff"),
        "none": ("none", 1.0, SCALE, "no-difference"),
        "confound": ("none", 1.0, SCALE * 1.6, "unresolved"),
    }
    wrong = 0
    for _name, (mech, factor, scale_b, correct) in truth.items():
        for seed in (1, 2):
            a, b = _pair(mech, factor, scale_b=scale_b, seed=seed)
            res = attribute_differential(a, b, CIRC, steps=250, seed=0)
            if res.call in POSITIVE and res.call != correct:
                wrong += 1
    assert wrong == 0, f"{wrong} confident-wrong mechanism-difference calls"
