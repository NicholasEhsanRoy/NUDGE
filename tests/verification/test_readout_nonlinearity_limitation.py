"""NUDGE-LIM-006 witness — a nonlinear readout is misattributed as a circuit switch.

This documents a *known limitation* as an executable, strict-xfail test. NUDGE assumes
an affine reporter; a LINEAR (non-switch) circuit observed through a steep Hill readout
is misread as a circuit switch. Ground truth is no switch, so NUDGE *should* return
off-model — it currently emits a confident false positive (threshold), so the assertion
below fails and the test xfails.

``strict=True`` is deliberate: if a future change (a calibrated / non-affine Readout,
or a reporter-linearity control) fixes this, the assertion starts passing → XPASS →
strict turns it into a FAILURE, forcing us to remove the xfail and update the docs.
See ``scripts/vv/FINDINGS.md`` (NUDGE-LIM-006).
"""

from __future__ import annotations

import pytest

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef
from nudge.core.vocabulary import POSITIVE_CLASSES
from nudge.data.decoy_generators import generate_readout_nonlinearity_decoy
from nudge.inference.fit import fit

pytestmark = pytest.mark.verification


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="NUDGE-LIM-006: a nonlinear readout is misattributed as a circuit switch",
)
def test_nonlinear_readout_should_be_declined_but_is_not() -> None:
    # Linear circuit (no switch) through a steep Hill reporter + strong perturbation.
    adata = generate_readout_nonlinearity_decoy(seed=1)
    switch = Circuit(
        [SpeciesDef("SW", basal=0.05, decay=1.0)],
        [EdgeDef(0, 0, "hill_activation", K=1.0, n=6.0, vmax=2.0)],
    )
    mm = fit(adata, switch, n_cells=384, steps=400, margin_k=1.7, seed=0)
    # DESIRED behaviour (currently unmet — hence xfail): NUDGE declines rather than
    # inventing a mechanism from readout nonlinearity.
    assert all(c.mechanism not in POSITIVE_CLASSES for c in mm.calls)
