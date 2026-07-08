"""N-D safety: NUDGE must stay fail-safe on a multi-species (toggle) switch.

The N-D saddle finder + multi-basin representation land (M1/M2), but the w_trans gain
signature is 1-D-specific and does NOT extend to a toggle (measured — gain w_trans stays
below the calibrated tau and is seed-unreliable; FINDINGS "N-D saddle"). So the gain
gate stays guarded to n_species==1. This test locks in the consequence: on toggle data
NUDGE must **abstain, never emit a wrong positive** — fail-safe holds off 1-D too.
"""

from __future__ import annotations

import pytest

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef
from nudge.core.vocabulary import POSITIVE_CLASSES, MechanismClass
from nudge.data.stochastic import generate_toggle_perturbseq
from nudge.data.synthetic import PerturbationSpec
from nudge.inference.fit import fit_multibasin

pytestmark = pytest.mark.verification

_TRUE = {
    "thr": MechanismClass.THRESHOLD,
    "gai": MechanismClass.GAIN,
    "cei": MechanismClass.CEILING,
}


def _toggle() -> Circuit:
    return Circuit(
        [
            SpeciesDef("A", basal=0.05, decay=1.0),
            SpeciesDef("B", basal=0.05, decay=1.0),
        ],
        [
            EdgeDef(1, 0, "hill_repression", K=1.0, n=4.0, vmax=2.0),
            EdgeDef(0, 1, "hill_repression", K=1.0, n=4.0, vmax=2.0),
        ],
    )


@pytest.mark.slow
def test_toggle_transition_never_misclassifies() -> None:
    adata = generate_toggle_perturbseq(
        _toggle(),
        [
            PerturbationSpec("thr", "edge", 0, "K", 3.0),
            PerturbationSpec("gai", "edge", 0, "n", 0.3),
            PerturbationSpec("cei", "edge", 0, "vmax", 0.4),
        ],
        n_cells_per_condition=3000, seed=0,
    )
    mm = fit_multibasin(
        adata, _toggle(), n_cells=384, steps=400, margin_k=1.7,
        transition_mode=True, seed=0,
    )
    # Fail-safe on a multi-species switch: no perturbation is ever assigned the WRONG
    # positive mechanism (it abstains — off-model/unresolved/no-effect).
    for call in mm.calls:
        if call.mechanism in POSITIVE_CLASSES:
            assert call.mechanism is _TRUE.get(call.perturbation), (
                f"{call.perturbation}: N-D toggle emitted WRONG positive "
                f"{call.mechanism.value} — fail-safe violated"
            )
