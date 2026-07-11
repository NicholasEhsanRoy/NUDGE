"""Tests for the `nudge` typer CLI (thin layer over the tested API)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from nudge.cli import app

runner = CliRunner()


def _counts_adata(path, *, integer: bool = True):
    rng = np.random.default_rng(0)
    x = rng.poisson(3.0, size=(20, 3)).astype("float32")
    if not integer:
        x = x + 0.5  # non-integer → looks normalized
    obs = pd.DataFrame(
        {"condition": (["WT"] * 10) + (["SOS1"] * 10)},
        index=[f"c{i}" for i in range(20)],
    )
    adata = ad.AnnData(X=x, obs=obs)
    adata.var_names = ["Activation", "g1", "g2"]
    adata.write_h5ad(path)
    return path


def test_mechanisms_lists_library() -> None:
    result = runner.invoke(app, ["mechanisms"])
    assert result.exit_code == 0
    assert "HillActivation" in result.output
    assert "LinearIntegrator" in result.output  # regression: was dropped from registry


def test_explain_abstention() -> None:
    result = runner.invoke(app, ["explain", "unresolved"])
    assert result.exit_code == 0
    assert "abstention" in result.output
    assert "operating point" in result.output


def test_explain_decoy() -> None:
    result = runner.invoke(app, ["explain", "NUDGE-DECOY-001"])
    assert result.exit_code == 0
    assert "off-model" in result.output


def test_explain_unknown_exits_nonzero() -> None:
    result = runner.invoke(app, ["explain", "totally-bogus"])
    assert result.exit_code == 1
    assert "unrecognised" in result.output


def test_check_data_accepts_integer_counts(tmp_path) -> None:
    path = _counts_adata(tmp_path / "ok.h5ad", integer=True)
    result = runner.invoke(app, ["check-data", str(path)])
    assert result.exit_code == 0
    assert "OK" in result.output


def test_check_data_rejects_non_integer(tmp_path) -> None:
    path = _counts_adata(tmp_path / "bad.h5ad", integer=False)
    result = runner.invoke(app, ["check-data", str(path)])
    assert result.exit_code == 1
    assert "REJECTED" in result.output


def test_load_summarises_conditions(tmp_path) -> None:
    path = _counts_adata(tmp_path / "load.h5ad", integer=True)
    result = runner.invoke(app, ["load", str(path)])
    assert result.exit_code == 0
    assert "conditions" in result.output
    assert "SOS1" in result.output and "WT" in result.output


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    # no_args_is_help → exit code 0 or 2 depending on click version; help text present.
    assert "attribute" in result.output and "mechanisms" in result.output


@pytest.mark.slow
def test_attribute_runs_and_reports(tmp_path) -> None:
    from nudge.circuits import ras_switch_1node
    from nudge.data.synthetic import PerturbationSpec, generate_synthetic_perturbseq

    adata = generate_synthetic_perturbseq(
        ras_switch_1node(),
        [PerturbationSpec("SOS1", "edge", 0, "n", 0.4)],
        n_cells_per_condition=250,
        seed=1,
    )
    adata.uns.clear()
    path = tmp_path / "switch.h5ad"
    adata.write_h5ad(path)
    result = runner.invoke(
        app, ["attribute", str(path), "--target", "SOS1", "--steps", "60"]
    )
    assert result.exit_code == 0
    assert "target: SOS1" in result.output


# --------------------------------------------------------------------------- #
# nudge viz — the figure-rendering verb (reuses the service.render_result seam)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["identifiability", "robustness", "epistasis"])
def test_viz_demo_writes_png(kind: str, tmp_path) -> None:
    """`nudge viz KIND --demo --out ...` writes a non-empty PNG + succeeds (fast kinds)."""
    pytest.importorskip("matplotlib")
    out = tmp_path / f"{kind}.png"
    result = runner.invoke(
        app, ["viz", kind, "--demo", "--out", str(out), "--no-fig-code"]
    )
    assert result.exit_code == 0, result.output
    assert out.exists() and out.stat().st_size > 0
    assert "abstained:" in result.output


def test_viz_bad_kind_lists_and_exits_2() -> None:
    result = runner.invoke(app, ["viz", "not-a-kind"])
    assert result.exit_code == 2
    assert "unknown figure kind" in result.output
    assert "dose_response" in result.output  # the list of known kinds is printed


def test_viz_json_replay_roundtrips(tmp_path) -> None:
    """`--json` replays a saved figure-data dict with no re-fit."""
    pytest.importorskip("matplotlib")
    import json

    from nudge.viz.demo import demo_result

    data = demo_result("differential")
    jpath = tmp_path / "diff.json"
    jpath.write_text(json.dumps(data))
    out = tmp_path / "diff.png"
    result = runner.invoke(
        app, ["viz", "differential", "--json", str(jpath), "--out", str(out),
              "--no-fig-code"]
    )
    assert result.exit_code == 0, result.output
    assert out.exists() and out.stat().st_size > 0
