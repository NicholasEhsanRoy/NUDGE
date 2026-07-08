"""Tier-0.5 inverse-crime guard — the fail-safe property under *independent*
stochastic dynamics.

Everything in Tier-0 is an inverse crime: the generator and the fitter share the
same deterministic model + noise. Tier-0.5 (``nudge.data.stochastic``) breaks that
by generating from a genuinely stochastic tau-leaping SSA of a self-activating gene,
where bimodality is **emergent** (noise-induced), not designed-in.

**What this locks in (and what it deliberately does not).** NUDGE's fit engine
solves each cell's steady state from a fixed ``x0 = 0`` (``inference/fit.py``
``_simulate``). For a self-activation *feedback* switch that only ever reaches the
LOW basin, so the transfer-function forward model cannot fully represent the emergent
HIGH mode. Across seeds NUDGE therefore either **abstains** (``unresolved`` /
``off-model``) or recovers only the most robust mechanism (**gain**) — and across
seeds 0–3 it emits **zero wrong positives** (measured; ``scripts/vv/FINDINGS.md``
§Tier-0.5). That is the property this test guards: **on independent emergent-bistable
data, NUDGE never emits a WRONG positive mechanism.** It does *not* require recovery —
abstention is honest, correct behaviour here. The complementary finding — that fitting
a *wrong topology* (feedforward) to this feedback data CAN produce a confident wrong
call, so the fail-safe guarantee is conditional on approximately-correct topology — is
documented in FINDINGS and not asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef
from nudge.core.vocabulary import POSITIVE_CLASSES, MechanismClass
from nudge.data.stochastic import generate_stochastic_perturbseq
from nudge.data.synthetic import PerturbationSpec
from nudge.inference.fit import fit, fit_multibasin

pytestmark = pytest.mark.verification

# Ground-truth mechanism of each mover (which propensity param it scales).
_TRUE = {
    "thr": MechanismClass.THRESHOLD,
    "gai": MechanismClass.GAIN,
    "cei": MechanismClass.CEILING,
}


def _self_activation_switch() -> Circuit:
    """A single self-activating gene — genuinely bistable (low ≈ 0.05, high ≈ 2)."""
    return Circuit(
        [SpeciesDef("SW", basal=0.05, decay=1.0)],
        [EdgeDef(0, 0, "hill_activation", K=1.0, n=6.0, vmax=2.0)],
    )


def _movers() -> list[PerturbationSpec]:
    return [
        PerturbationSpec("thr", "edge", 0, "K", 3.0),  # threshold
        PerturbationSpec("gai", "edge", 0, "n", 0.2),  # gain
        PerturbationSpec("cei", "edge", 0, "vmax", 0.3),  # ceiling
    ]


def test_stochastic_wt_is_emergently_bimodal() -> None:
    # The simulator itself: a WT-only snapshot must show two separated modes — a
    # low/OFF mode and a populated ON mode — arising from stochastic dynamics, not a
    # designed-in parameter distribution. Guards the generator before the fit test.
    adata = generate_stochastic_perturbseq(
        _self_activation_switch(), n_cells_per_condition=2000, seed=0
    )
    counts = np.asarray(adata.X).ravel()
    frac_low = float((counts <= 1).mean())
    frac_high = float((counts >= 8).mean())
    assert frac_low > 0.15, f"no populated OFF mode (frac_low={frac_low:.2f})"
    assert frac_high > 0.15, f"no populated ON mode (frac_high={frac_high:.2f})"


@pytest.mark.slow
def test_stochastic_fit_never_misclassifies() -> None:
    # The fail-safe property under model mismatch: fit the (matched-topology)
    # self-activation switch to independent stochastic data. NUDGE may abstain freely
    # (it cannot fully represent emergent feedback bistability from x0=0), but it must
    # NEVER assign a perturbation the WRONG positive mechanism. Seed 2 is a seed where
    # NUDGE *does* emit a positive (gain) — so the never-wrong assertion exercises a
    # real attribution path, not just trivial all-abstention.
    adata = generate_stochastic_perturbseq(
        _self_activation_switch(), _movers(), n_cells_per_condition=3000, seed=2
    )
    mm = fit(
        adata, _self_activation_switch(),
        n_cells=384, steps=400, margin_k=1.7, seed=0,
    )
    for call in mm.calls:
        if call.mechanism in POSITIVE_CLASSES:
            assert call.mechanism is _TRUE[call.perturbation], (
                f"{call.perturbation}: emitted WRONG positive {call.mechanism.value} "
                f"(true = {_TRUE[call.perturbation].value}) — fail-safe violated"
            )


@pytest.mark.slow
def test_saddle_transition_recovers_gain_never_wrong() -> None:
    # The saddle gain gate (FINDINGS §T0.5-5): fit_multibasin(transition_mode=True)
    # adds a transition mode at the ODE saddle, whose free-n weight is a fail-safe gain
    # detector. It must RECOVER gain on seed 2 — the exact seed where single-basin fit
    # abstains and the 2-basin model was confidently WRONG (gain→ceiling) — while never
    # assigning any perturbation the wrong positive mechanism.
    adata = generate_stochastic_perturbseq(
        _self_activation_switch(), _movers(), n_cells_per_condition=3000, seed=2
    )
    mm = fit_multibasin(
        adata, _self_activation_switch(),
        n_cells=384, steps=400, margin_k=1.7, transition_mode=True, seed=0,
    )
    calls = {c.perturbation: c.mechanism for c in mm.calls}
    # The headline: gain is recovered on the notorious seed.
    assert calls["gai"] is MechanismClass.GAIN
    # Fail-safe: nothing is ever assigned the wrong positive mechanism.
    for pert, mech in calls.items():
        if mech in POSITIVE_CLASSES:
            assert mech is _TRUE[pert], (
                f"{pert}: WRONG positive {mech.value} (true {_TRUE[pert].value})"
            )
