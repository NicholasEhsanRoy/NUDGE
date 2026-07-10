"""Tests for hidden-node ABSTENTION — the differential, and the honesty guarantee.

The load-bearing test is `test_never_emits_a_positive_hidden_node_claim`: NUDGE ships
ONLY the abstention half of the hidden-node problem, so no surface it produces may assert
a hidden node exists. The other tests pin the legible differential (an off-model case
enumerates all six candidate causes with their LIM/decoy refs) and the adequate case (no
inadequacy → no differential), plus the service round-trip (the CLI/MCP path).
"""

from __future__ import annotations

import re

from nudge.inference.hidden_node import (
    CandidateCause,
    InadequacyReport,
    diagnose_inadequacy,
)

# The six causes the differential must enumerate (a differential, not a verdict).
_EXPECTED_CAUSES = {
    "hidden node / unmeasured regulator",
    "off-target perturbation effect",
    "nonlinear measurement readout",
    "wrong / misspecified topology",
    "batch / depth confound",
    "genuinely not-a-switch (linear circuit)",
}


def _all_text(report: InadequacyReport) -> str:
    """Every human-facing string the report emits, concatenated (for scanning)."""
    parts = [report.verdict, report.reason]
    for c in report.causes:
        parts += [c.name, c.evidence, c.distinguishing_experiment]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# The off-model case → a legible differential
# --------------------------------------------------------------------------- #
def test_off_model_enumerates_the_full_differential() -> None:
    """An off-model verdict returns the full six-cause differential with LIM/decoy refs."""
    report = diagnose_inadequacy(off_model=True, neomorphic_ratio=2.5)
    assert report.is_adequate is False
    assert report.verdict == "off-model"
    names = {c.name for c in report.causes}
    assert names == _EXPECTED_CAUSES

    # Every cause is a real CandidateCause with an evidence string + a distinguishing test.
    for c in report.causes:
        assert isinstance(c, CandidateCause)
        assert c.evidence.strip()
        assert c.distinguishing_experiment.strip()
        assert c.qualitative_rank in {"leading", "plausible", "less-likely"}

    # The causes map to the documented limitations / decoys.
    lim_refs = {c.limitation_ref for c in report.causes if c.limitation_ref}
    assert "NUDGE-LIM-015" in lim_refs  # the hidden-node honesty cap
    assert "NUDGE-LIM-006" in lim_refs  # the nonlinear-readout confound
    decoy_refs = {c.decoy_ref for c in report.causes if c.decoy_ref}
    assert "NUDGE-DECOY-005" in decoy_refs  # the marginal-Hill / not-a-switch decoy


def test_ranking_never_makes_hidden_node_the_lone_leading_cause() -> None:
    """Even with a large off-axis residual, hidden-node is capped below the leading cause."""
    report = diagnose_inadequacy(off_model=True, neomorphic_ratio=9.0)
    ranked = report.ranked_causes()
    leading = [c for c in ranked if c.qualitative_rank == "leading"]
    # The not-a-switch reading (the gate working) leads on an off-model verdict.
    assert leading and leading[0].name == "genuinely not-a-switch (linear circuit)"
    hidden = next(c for c in ranked if c.name.startswith("hidden node"))
    assert hidden.qualitative_rank != "leading"


# --------------------------------------------------------------------------- #
# The adequate case → no differential
# --------------------------------------------------------------------------- #
def test_adequate_model_emits_no_differential() -> None:
    """No off-model verdict and no residual → is_adequate=True, no causes, honest reason."""
    report = diagnose_inadequacy(off_model=False)
    assert report.is_adequate is True
    assert report.verdict == "model-adequate"
    assert report.causes == []
    assert "adequate" in report.reason.lower()


def test_small_residual_alone_does_not_fire() -> None:
    """A below-threshold off-axis residual with no off-model verdict stays adequate."""
    report = diagnose_inadequacy(off_model=False, neomorphic_ratio=0.2)
    assert report.is_adequate is True
    assert report.causes == []


def test_diagnostic_residual_without_off_model_triggers_differential() -> None:
    """A fired residual (no off-model) still yields the differential, verdict labelled."""
    report = diagnose_inadequacy(off_model=False, neomorphic_ratio=3.0)
    assert report.is_adequate is False
    assert report.verdict == "diagnostic-residual"
    assert {c.name for c in report.causes} == _EXPECTED_CAUSES


# --------------------------------------------------------------------------- #
# The honesty guarantee — NEVER a bare positive "hidden node" claim
# --------------------------------------------------------------------------- #
def test_never_emits_a_positive_hidden_node_claim() -> None:
    """The strongest hidden-node statement is HEDGED — never a positive assertion.

    This is the abstention-half-only guarantee: no surface may assert a hidden node
    exists. We scan every emitted string for bare positive assertions, and confirm the
    hidden-node cause is explicitly hedged.
    """
    # The strongest evidence regime: a large off-axis residual (the most tempting to
    # over-read as a discovery).
    report = diagnose_inadequacy(off_model=True, neomorphic_ratio=12.0)
    text = _all_text(report).lower()

    # No bare positive hidden-node assertion anywhere in the report.
    forbidden = [
        r"hidden node detected",
        r"there is a hidden node",
        r"a hidden node is present",
        r"hidden node found",
        r"we (?:have )?identif\w* a hidden node",
        r"proves? (?:the existence of )?a hidden node",
        r"confirm\w* a hidden node",
    ]
    for pat in forbidden:
        assert not re.search(pat, text), f"forbidden positive claim matched: {pat!r}"

    # The hidden-node cause IS enumerated, and its evidence is explicitly hedged.
    hidden = next(c for c in report.causes if c.name.startswith("hidden node"))
    ev = hidden.evidence.lower()
    assert "consistent with" in ev
    assert "does not prove" in ev or "not prove" in ev
    # It points at a positive measurement as the only way to establish one.
    assert "measure" in hidden.distinguishing_experiment.lower()


# --------------------------------------------------------------------------- #
# Service-wiring round-trip (the CLI / MCP path)
# --------------------------------------------------------------------------- #
def test_service_round_trip() -> None:
    """service.diagnose_abstention serialises the differential + the honesty guarantee."""
    from nudge.service import diagnose_abstention

    out = diagnose_abstention(off_model=True, neomorphic_ratio=2.0, readout_flag=True)
    assert out["is_adequate"] is False
    assert out["verdict"] == "off-model"
    assert out["hidden_node_claim"] is False
    names = {c["name"] for c in out["causes"]}
    assert names == _EXPECTED_CAUSES
    # The knowledge backbone enriched a cause with its limitation title.
    titles = {c["limitation_title"] for c in out["causes"] if c["limitation_title"]}
    assert any(t for t in titles)
    # The readout flag promoted the nonlinear-readout cause to leading.
    readout = next(c for c in out["causes"] if c["name"] == "nonlinear measurement readout")
    assert readout["qualitative_rank"] == "leading"


def test_service_adequate_round_trip() -> None:
    """An adequate model round-trips to is_adequate=True with no causes."""
    from nudge.service import diagnose_abstention

    out = diagnose_abstention(off_model=False)
    assert out["is_adequate"] is True
    assert out["causes"] == []
    assert out["hidden_node_claim"] is False
