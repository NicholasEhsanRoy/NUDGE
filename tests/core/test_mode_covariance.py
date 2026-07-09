"""Circuit.mode_covariances — the linear-noise (Lyapunov) covariance primitive (M0).

Per-stable-mode Gaussian covariance from ``A Σ + Σ Aᵀ + D = 0``, the shape the
covariance-structured attribution loss fits. The load-bearing check is that it
reproduces the covariances the Fisher-information analysis was built on
(``scripts/vv/``), and that it degrades gracefully (monostable → one mode;
unsupported topology → ``None``).
"""

from __future__ import annotations

import numpy as np
import pytest

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef


def _switch(*, n: float = 6.0) -> Circuit:
    return Circuit(
        [SpeciesDef("SW", basal=0.05, decay=1.0)],
        [EdgeDef(0, 0, "hill_activation", K=1.0, n=n, vmax=2.0)],
    )


def _toggle(*, n: float = 4.0) -> Circuit:
    return Circuit(
        [SpeciesDef("A", basal=0.05, decay=1.0), SpeciesDef("B", basal=0.05, decay=1)],
        [
            EdgeDef(1, 0, "hill_repression", K=1.0, n=n, vmax=2.0),
            EdgeDef(0, 1, "hill_repression", K=1.0, n=n, vmax=2.0),
        ],
    )


def test_toggle_covariances_match_lna_reference() -> None:
    # Reproduces the FIM-analysis reference (scripts/vv/fisher_sloppiness.py):
    # each toggle lobe has mu=[0.157, 2.049], cov diag=[0.199, 2.055], corr=-0.324.
    modes = _toggle().mode_covariances()
    assert modes is not None and len(modes) == 2
    low = min(modes, key=lambda mc: float(mc[0][0]))  # the low-A lobe
    mean, cov = low
    assert cov.shape == (2, 2)
    assert np.allclose(cov, cov.T, atol=1e-5)  # symmetric
    assert (np.linalg.eigvalsh(cov) > 0).all()  # positive-definite
    assert float(mean[0]) < float(mean[1])  # low-A, high-B lobe
    assert cov[0, 0] == pytest.approx(0.199, abs=0.02)
    assert cov[1, 1] == pytest.approx(2.055, abs=0.1)
    corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    assert corr == pytest.approx(-0.324, abs=0.03)


def test_switch_covariances_1d_positive() -> None:
    modes = _switch().mode_covariances()  # bistable 1-species → two stable modes
    assert modes is not None and len(modes) == 2
    for mean, cov in modes:
        assert mean.shape == (1,) and cov.shape == (1, 1)
        assert cov[0, 0] > 0  # a positive scalar variance at each stable FP


def test_monostable_single_mode() -> None:
    modes = _switch(n=1.2).mode_covariances()  # collapsed → one stable mode
    assert modes is not None and len(modes) == 1
    assert modes[0][1][0, 0] > 0


def test_unsupported_topology_returns_none() -> None:
    linear = Circuit(
        [SpeciesDef("SW", basal=0.05, decay=1.0)],
        [EdgeDef(0, 0, "linear", weight=1.0)],
    )
    assert linear.mode_covariances() is None
