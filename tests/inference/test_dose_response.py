"""Dose-response mechanism attribution — fit + classify, and the OCT4/NANOG lock-in.

The classifier must (a) call a genuine high-gain curve a ``switch``; (b) call an ``n≈1``
curve ``graded``; (c) **abstain** (``unresolved``) when the doses don't span the
inflection
(one arm of a sigmoid — gain unidentifiable) or when the ``n`` CI straddles the line;
and
(d) call a flat curve ``no-effect``. A dedicated regression pins the float32 bug: JAX is
float32, so a finite-difference Jacobian froze ``n`` at its init — the exact autodiff
Jacobian must let ``n`` move (else every high-gain curve mis-reads as ``n≈init``).

The real-data test (``needs_data``) locks in the adjudicated OCT4/NANOG result: OCT4 is
a
resolved ultrasensitive switch; NANOG is a *principled abstention* (its knockdown did
not
span its threshold — verified by an n-profile whose R² is flat within 0.075 across all
n).
This is the fail-safe catching a curve a naive Hill fit would over-call as ``graded``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from nudge.inference.dose_response import (
    attribute_dose_response,
    classify_dose_response,
    fit_dose_response,
)
from nudge.mechanisms.regulatory import hill_repression

_OCT4_NANOG_H5AD = "/media/nick/Seagate Hub/oct4_nanog/ESC.h5ad"


def _curve(k: float, n: float, *, floor: float = 0.2, amp: float = 0.8,
           lo: float = 0.0, hi: float = 1.0, n_pts: int = 24,
           noise: float = 0.02, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic repressive Hill curve from the real primitive + seeded noise."""
    rng = np.random.default_rng(seed)
    dose = np.linspace(lo, hi, n_pts)
    resp = floor + np.asarray(hill_repression(dose, k, n, amp))
    return dose, resp + rng.normal(0.0, noise, size=dose.shape)


def test_high_gain_curve_is_a_switch() -> None:
    dose, resp = _curve(k=0.5, n=6.0)
    res = attribute_dose_response(dose, resp, direction="repress", n_boot=200, seed=0)
    assert res.call == "switch", res.reason
    assert res.fit.ci_n[0] > 2.0  # CI clears the ultrasensitive line


def test_autodiff_jacobian_lets_n_move_off_its_seed() -> None:
    """Regression for the float32 finite-difference bug: n must NOT freeze at init.

    Every seed in the grid starts n at {1,2,4,8}; a broken (frozen) fit would return an
    n pinned at a seed with a degenerate CI. A true fit of n=7 data recovers n≈7.
    """
    dose, resp = _curve(k=0.5, n=7.0, noise=0.01, seed=1)
    fit = fit_dose_response(dose, resp, direction="repress", n_boot=100, seed=1)
    assert fit.n > 4.5, f"n did not climb to the true high gain (got {fit.n})"
    assert fit.ci_n[1] - fit.ci_n[0] > 0.05, "CI is degenerate — n frozen (f32 bug)"


def test_graded_curve_is_graded() -> None:
    dose, resp = _curve(k=0.4, n=1.0, noise=0.015, seed=2)
    res = attribute_dose_response(dose, resp, direction="repress", n_boot=200, seed=2)
    assert res.call == "graded", res.reason


def test_one_arm_curve_is_unresolved() -> None:
    """A NANOG-like shallow decline that stops before the inflection (K past max dose).

    The knockdown never reaches half-max, so a graded n≈1.5 (K just past the range) and
    a
    high-threshold switch fit comparably — gain is unidentifiable. This mirrors the real
    NANOG curve the classifier correctly abstains on.
    """
    dose, resp = _curve(k=0.9, n=1.5, lo=0.05, hi=0.5, n_pts=14, noise=0.01, seed=3)
    res = attribute_dose_response(dose, resp, direction="repress", n_boot=200, seed=3)
    assert res.call == "unresolved", res.reason
    assert not res.fit.spans_inflection


def test_flat_curve_is_no_effect() -> None:
    rng = np.random.default_rng(4)
    dose = np.linspace(0.0, 1.0, 24)
    resp = 0.7 + rng.normal(0.0, 0.02, size=dose.shape)  # inert: no dose dependence
    res = attribute_dose_response(dose, resp, direction="repress", n_boot=100, seed=4)
    assert res.call == "no-effect", res.reason


def test_activate_direction_switch() -> None:
    """The activation branch (response rises with dose) classifies symmetrically."""
    rng = np.random.default_rng(5)
    dose = np.linspace(0.0, 1.0, 24)
    from nudge.mechanisms.regulatory import hill_activation

    resp = 0.1 + np.asarray(hill_activation(dose, 0.5, 6.0, 0.8))
    resp = resp + rng.normal(0.0, 0.02, size=dose.shape)
    res = attribute_dose_response(dose, resp, direction="activate", n_boot=200, seed=5)
    assert res.call == "switch", res.reason


def test_too_few_points_raises() -> None:
    with pytest.raises(ValueError, match="4 dose points"):
        fit_dose_response([0.1, 0.5, 0.9], [1.0, 0.6, 0.2])


def test_classify_thresholds_are_tunable() -> None:
    """A borderline switch downgrades to graded when n_switch is raised past its CI."""
    dose, resp = _curve(k=0.5, n=3.0, noise=0.015, seed=6)
    fit = fit_dose_response(dose, resp, direction="repress", n_boot=200, seed=6)
    strict, _ = classify_dose_response(fit, n_switch=8.0)
    assert strict in {"graded", "unresolved"}


@pytest.mark.needs_data
@pytest.mark.skipif(
    not os.path.exists(_OCT4_NANOG_H5AD), reason="OCT4/NANOG ESC.h5ad not present"
)
def test_oct4_switch_nanog_unresolved_real_data() -> None:
    """Lock in the adjudicated flagship result on the real GSE283614 ESC data."""
    import anndata as ad

    from nudge.inference.bridge import knockdown_dose_response

    adata = ad.read_h5ad(_OCT4_NANOG_H5AD)
    sig = ["SOX2", "LIN28A", "UTF1", "DNMT3B", "TDGF1", "ZFP42", "SALL4"]

    d_oct4, r_oct4 = knockdown_dose_response(
        adata, target_gene="POU5F1", signature=sig, group_prefix="OCT4"
    )
    oct4 = attribute_dose_response(d_oct4, r_oct4, direction="repress", seed=0)
    assert oct4.call == "switch", oct4.reason
    assert oct4.fit.n > 4.0 and oct4.fit.r2 > 0.9

    d_nan, r_nan = knockdown_dose_response(
        adata, target_gene="NANOG", signature=sig, group_prefix="NANOG"
    )
    nanog = attribute_dose_response(d_nan, r_nan, direction="repress", seed=0)
    assert nanog.call == "unresolved", nanog.reason
