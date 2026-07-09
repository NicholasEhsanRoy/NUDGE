"""Tests for the shared application service (nudge.service)."""

from __future__ import annotations

import numpy as np
import pytest

from nudge.service import build_circuit, dose_response_file, parse_markers


def test_build_circuit_topologies() -> None:
    for topo in ("1node", "2node", "toggle"):
        c = build_circuit(topo)
        assert len(c.names) >= 1
    with pytest.raises(ValueError):
        build_circuit("nonsense")


def test_parse_markers_defaults_species_to_own_name() -> None:
    circuit = build_circuit("1node")
    markers = parse_markers(circuit, None)
    assert markers == {name: [name] for name in circuit.names}


def test_parse_markers_explicit_pairs() -> None:
    circuit = build_circuit("2node")
    markers = parse_markers(circuit, ["RasGTP=FOS,EGR1"])
    assert markers["RasGTP"] == ["FOS", "EGR1"]
    # unspecified species still get their default single-gene marker.
    for name in circuit.names:
        assert name in markers


def test_parse_markers_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        parse_markers(build_circuit("1node"), ["no-equals-sign"])


def test_dose_response_file_csv_wiring(tmp_path) -> None:
    """The CSV path the CLI/MCP share: a switch curve round-trips to a 'switch' dict."""
    from nudge.mechanisms.regulatory import hill_repression

    dose = np.linspace(0.0, 1.0, 22)
    resp = 0.2 + np.asarray(hill_repression(dose, 0.5, 6.0, 0.8))
    resp = resp + np.random.default_rng(0).normal(0.0, 0.02, dose.shape)
    csv = tmp_path / "curve.csv"
    csv.write_text(
        "dose,response\n"
        + "\n".join(f"{d},{r}" for d, r in zip(dose, resp, strict=True))
    )
    out = dose_response_file(str(csv), n_boot=200)
    assert out["call"] == "switch"
    assert out["n_apparent_gain"] > 4.0
    assert set(out) >= {"call", "reason", "ci_n", "spans_inflection", "direction"}


@pytest.mark.slow
def test_attribute_file_end_to_end_is_honest(tmp_path) -> None:
    """Attribution runs end-to-end and returns an AttributionReport (call or skip).

    We assert the *contract* (a structured, honest report), not a specific verdict:
    single-condition attribution may legitimately abstain or skip (low LNA depth),
    which is exactly the fail-safe behaviour under test.
    """
    from nudge.circuits import ras_switch_1node
    from nudge.data.synthetic import PerturbationSpec, generate_synthetic_perturbseq
    from nudge.service import attribute_file, report_to_dict

    adata = generate_synthetic_perturbseq(
        ras_switch_1node(),
        [PerturbationSpec("SOS1", "edge", 0, "n", 0.4)],
        n_cells_per_condition=250,
        seed=1,
    )
    adata.uns.clear()  # list-of-dicts ground_truth is not h5ad-serializable
    path = tmp_path / "switch.h5ad"
    adata.write_h5ad(path)

    report = attribute_file(str(path), "SOS1", topology="1node", steps=60)
    assert report.target == "SOS1"
    assert report.n_cells  # populated
    # every operating point is either attributed (single) or skipped, never dropped.
    for label in report.n_cells:
        assert label in report.single or label in report.skipped
    d = report_to_dict(report)
    assert isinstance(d, dict) and d["target"] == "SOS1"
    assert isinstance(np.asarray(list(report.n_cells.values())), np.ndarray)
