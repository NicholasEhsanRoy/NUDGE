"""Tests for the shared knowledge base (nudge.knowledge)."""

from __future__ import annotations

from nudge.core.vocabulary import ABSTENTION_CLASSES, MechanismClass
from nudge.knowledge import (
    explain,
    get_mechanism_card,
    list_decoys,
    list_limitations,
    list_mechanisms,
)


def test_every_registered_mechanism_is_listed_with_metadata() -> None:
    rows = list_mechanisms()
    names = {m["registry_name"] for m in rows}
    # The registry must be fully populated (regression: LinearIntegrator was dropped).
    assert {"LinearEffect", "HillActivation", "LinearIntegrator", "Readout"} <= names
    for m in rows:
        assert m["algorithm_id"]
        assert m["role"]


def test_every_registered_mechanism_has_a_card() -> None:
    for m in list_mechanisms():
        assert m["card"] is not None, f"{m['registry_name']} has no Mechanism Card"
        assert get_mechanism_card(m["registry_name"]) is not None


def test_get_mechanism_card_accepts_stem_and_registry_name() -> None:
    by_stem = get_mechanism_card("hill_activation")
    by_registry = get_mechanism_card("HillActivation")
    assert by_stem is not None and by_registry is not None
    assert by_stem == by_registry
    assert get_mechanism_card("does-not-exist") is None


def test_explain_abstention_pulls_decoys_and_cards() -> None:
    result = explain("off-model")
    assert result["kind"] == "abstention"
    assert result["verdict"] == "off-model"
    # off-model decoys are exactly those with that expected verdict.
    ids = {d["decoy_id"] for d in result["decoys"]}
    assert {"NUDGE-DECOY-001", "NUDGE-DECOY-002"} <= ids
    assert all(d["expected_verdict"] == "off-model" for d in result["decoys"])


def test_explain_all_abstention_classes_resolve() -> None:
    for cls in ABSTENTION_CLASSES:
        result = explain(cls.value)
        assert result["kind"] == "abstention"
        assert result["meaning"]


def test_explain_decoy_id() -> None:
    result = explain("NUDGE-DECOY-001")
    assert result["kind"] == "decoy"
    assert result["decoy_id"] == "NUDGE-DECOY-001"
    assert result["expected_verdict"] == MechanismClass.OFF_MODEL.value


def test_explain_limitation_id() -> None:
    result = explain("NUDGE-LIM-006")
    assert result["kind"] == "limitation"
    assert result["anomaly_id"] == "NUDGE-LIM-006"
    assert "readout" in result.get("description", "").lower()


def test_explain_mechanism_name_returns_card() -> None:
    result = explain("hill_activation")
    assert result["kind"] == "mechanism_card"
    assert "Hill" in result["markdown"]


def test_explain_unknown_query_is_graceful() -> None:
    result = explain("not-a-real-thing")
    assert result["kind"] == "unknown"
    assert result["suggestions"]


def test_decoys_and_limitations_load() -> None:
    decoys = list_decoys()
    assert len(decoys) >= 5
    assert all("decoy_id" in d for d in decoys)
    lims = list_limitations()
    assert any(entry["anomaly_id"] == "NUDGE-LIM-001" for entry in lims)
