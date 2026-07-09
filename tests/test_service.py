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


def test_synergy_file_h5ad_wiring(tmp_path) -> None:
    """The .h5ad path the CLI/MCP share: a super-additive combo → a 'synergistic' dict.

    Builds a tiny 4-condition AnnData whose signature gene is clearly super-additive in
    A+B, writes it, and round-trips it through the shared ``synergy_file`` service.
    """
    import anndata as ad
    import pandas as pd

    from nudge.service import synergy_file

    rng = np.random.default_rng(0)
    genes = ["SIG", "G2", "G3", "G4", "G5"]
    # per-condition mean count of the signature gene: A+B (~200) >> additive (~40).
    levels = {"control": 10, "A": 25, "B": 25, "A+B": 200}
    rows, labels = [], []
    for cond, sig_mean in levels.items():
        for _ in range(220):
            row = rng.poisson([sig_mean, 30, 30, 30, 30]).astype(float)
            rows.append(row)
            labels.append(cond)
    x = np.vstack(rows)
    obs = pd.DataFrame(
        {"condition": labels, "total_counts": x.sum(axis=1)},
        index=[f"c{i}" for i in range(len(labels))],
    )
    adata = ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=genes))
    path = tmp_path / "combo.h5ad"
    adata.write_h5ad(path)

    out = synergy_file(
        str(path),
        control_label="control",
        a_label="A",
        b_label="B",
        ab_label="A+B",
        signature=["SIG"],
        n_boot=300,
    )
    assert out["call"] == "synergistic", out["reason"]
    assert out["interaction"] > 0.0
    assert set(out) >= {
        "call", "reason", "interaction", "ci_interaction", "n_cells", "effect_space"
    }
    assert out["n_cells"]["A+B"] == 220


def test_robustness_circuit_wiring() -> None:
    """The parametric robustness dial the CLI/MCP share round-trips to a dict."""
    from nudge.service import ROBUSTNESS_TOPOLOGIES, robustness_circuit

    assert set(ROBUSTNESS_TOPOLOGIES) == {"1node", "2node", "toggle"}

    near = robustness_circuit("1node", n=2.0)
    assert near["call"] == "near-fold"
    assert near["one_sided"] is True
    assert 0.0 <= near["proximity"] <= 1.0
    assert set(near) >= {
        "call", "reason", "proximity", "one_sided", "n_stable_modes",
        "min_re_lambda", "node_saddle_distance", "lna_lobe_ratio", "channels",
    }
    assert "channel_proximities" in near["channels"]

    # monostable → not-bistable, with a null proximity (never a forced number).
    mono = robustness_circuit("1node", n=1.0)
    assert mono["call"] == "not-bistable"
    assert mono["proximity"] is None


def test_bifurcation_file_npy_wiring(tmp_path) -> None:
    """The activity-file path the CLI/MCP share: score + depth-calibrated LNA reason."""
    import jax

    from nudge.inference.lyapunov import sample_lna_mixture
    from nudge.service import bifurcation_file

    circuit = build_circuit("1node")
    data = sample_lna_mixture(circuit, 300, jax.random.PRNGKey(0), scale=40.0)
    npy = tmp_path / "activity.npy"
    np.save(npy, np.asarray(data))

    out = bifurcation_file(str(npy), topology="1node")
    assert out["call"] in {"robust", "unresolved", "near-fold"}
    assert "lna_reason" in out and isinstance(out["lna_reason"], str)
    assert out["scale"] > 0.0
