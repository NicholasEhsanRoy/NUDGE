"""The decoy battery — synthetic negatives engineered to look positive.

Each ``DecoyCase`` is a dataset a naive method would call a confident hit; the
pass condition is that NUDGE returns the correct *negative* / abstention verdict.
The registry lives here (in ``src``, not ``tests``) so Mechanism Cards and
``known_limitations.yaml`` can cross-reference decoy IDs. It accepts human- and
AI-authored generators (creative-AI idea 1) via ``authored_by`` / ``prompt_ref``.

Phase-0: the schema + an empty battery; the nine cases land in Phase 3.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef
from nudge.core.vocabulary import MechanismClass
from nudge.data.decoy_generators import (
    generate_dropout_decoy,
    generate_mixture_decoy,
)
from nudge.data.stochastic import generate_telegraph_perturbseq
from nudge.data.synthetic import PerturbationSpec, generate_synthetic_perturbseq


@dataclass(frozen=True)
class DecoyCase:
    """A single adversarial dataset and the verdict NUDGE must return."""

    decoy_id: str  # e.g. "NUDGE-DECOY-001"
    summary: str
    generate: Callable[[], Any]  # () -> AnnData
    expected_verdict: MechanismClass
    limitation_ref: str = ""  # the NUDGE-LIM-* it maps to
    authored_by: Literal["human", "ai"] = "human"
    prompt_ref: str = ""  # for AI-authored decoys: prompt/model hash (idea 1)
    #: The circuit hypothesis to fit (a naive method's switch model). ``None`` → the
    #: battery's default 1-species self-activation switch. Cases whose data is a real
    #: (detectable) switch carry their matching topology here so the fit dimensions line
    #: up and the *right* gate (e.g. no-effect) is exercised.
    hypothesis: Circuit | None = None


def _feedforward_switch(n: float = 6.0) -> Circuit:
    """A 2-species feedforward switch (IN → SW) — a reliably *detected* switch."""
    return Circuit(
        [
            SpeciesDef("IN", basal=1.0, decay=1.0),
            SpeciesDef("SW", basal=0.05, decay=1.0),
        ],
        [EdgeDef(0, 1, "hill_activation", K=1.0, n=n, vmax=2.0)],
    )


def _telegraph_decoy() -> Any:
    """Bimodal-but-monostable telegraph data (To & Maheshri 2010) — not a switch."""
    return generate_telegraph_perturbseq(n_cells_per_condition=2000, seed=0)


def _mixture_decoy() -> Any:
    """Two-population (cell-type / doublet) mixture faking bimodality — not a switch."""
    return generate_mixture_decoy(n_cells_per_condition=2000, seed=0)


def _dropout_decoy() -> Any:
    """Technical dropout zero-peak on a monostable population — not a switch."""
    return generate_dropout_decoy(n_cells_per_condition=2000, seed=0)


def _no_effect_decoy() -> Any:
    """A REAL switch with a NULL (factor-1.0 / dead-guide) perturbation → no-effect."""
    return generate_synthetic_perturbseq(
        _feedforward_switch(6.0),
        [PerturbationSpec("null", "edge", 0, "K", 1.0)],
        n_cells_per_condition=2000, seed=0,
    )


def _marginal_hill_decoy() -> Any:
    """A barely-nonlinear (n=1.2) circuit — a marginal Hill the gate must not call."""
    return generate_synthetic_perturbseq(
        _feedforward_switch(1.2),
        [PerturbationSpec("kd", "edge", 0, "vmax", 0.5)],
        n_cells_per_condition=2000, seed=0,
    )


#: The decoy battery. Grows with the mechanism library and the AI decoy generator.
DECOY_BATTERY: list[DecoyCase] = [
    DecoyCase(
        decoy_id="NUDGE-DECOY-001",
        summary=(
            "Noise-induced bimodality without bistability: a non-cooperative "
            "positive-feedback loop with slow promoter switching is deterministically "
            "monostable yet bimodal. NUDGE must return off-model (not-a-switch)."
        ),
        generate=_telegraph_decoy,
        expected_verdict=MechanismClass.OFF_MODEL,
        limitation_ref="NUDGE-LIM-001",
    ),
    DecoyCase(
        decoy_id="NUDGE-DECOY-002",
        summary=(
            "Two-population mixture (cell types / doublets): a static mix of a low- "
            "and a high-expressing subpopulation is bimodal but has no switch. "
            "NUDGE must return off-model, not attribute ultrasensitivity."
        ),
        generate=_mixture_decoy,
        expected_verdict=MechanismClass.OFF_MODEL,
        limitation_ref="NUDGE-LIM-002",
    ),
    DecoyCase(
        decoy_id="NUDGE-DECOY-003",
        summary=(
            "Dropout zero-peak: a monostable population read out at bimodal library "
            "depth (a fraction captured at very low depth → near-all-zeros) mimics an "
            "OFF/ON switch. The bimodality is a measurement artifact — off-model."
        ),
        generate=_dropout_decoy,
        expected_verdict=MechanismClass.OFF_MODEL,
        limitation_ref="NUDGE-LIM-003",
    ),
    DecoyCase(
        decoy_id="NUDGE-DECOY-004",
        summary=(
            "Dead guide / no-effect: a genuine switch with a NULL perturbation "
            "(targeted but no knockdown). NUDGE must return no-effect — not report a "
            "mechanism just because the WT is a switch. Exercises the no-effect gate."
        ),
        generate=_no_effect_decoy,
        expected_verdict=MechanismClass.NO_EFFECT,
        limitation_ref="NUDGE-LIM-004",
        hypothesis=_feedforward_switch(6.0),
    ),
    DecoyCase(
        decoy_id="NUDGE-DECOY-005",
        summary=(
            "Marginal-overfit Hill: data from a barely-nonlinear (n=1.2) circuit. The "
            "mechanistic model must not be over-called as a switch — the parsimony "
            "gate's noise margin must reject a nonlinearity within the loss floor."
        ),
        generate=_marginal_hill_decoy,
        expected_verdict=MechanismClass.OFF_MODEL,
        limitation_ref="NUDGE-LIM-005",
        hypothesis=_feedforward_switch(6.0),
    ),
]
