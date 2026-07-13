"""Tests for the general ``identifiability`` + ``oed`` MCP tools (over the model registry).

The load-bearing checks: the tools run the REAL analysis on ≥2 DIFFERENT registered models
(proving generality, not a demo hardcode), return honest verdicts including abstentions, carry
the inline figure + fig.py provenance, and work through job_submit/job_status. Two decoys keep
the tool honest: a well-constrained model must NOT be flagged sloppy; a structurally
rank-deficient one MUST abstain.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp")

from nudge.mcp.server import _unwrap, build_server  # noqa: E402


def _call(server, name, args):
    return _unwrap(asyncio.run(server.call_tool(name, args)))


# --------------------------------------------------------------------------- #
# the registry (the generality proof)
# --------------------------------------------------------------------------- #
def test_list_models_spans_multiple_domains_and_tools() -> None:
    server = build_server()
    models = _call(server, "list_models", {})
    names = {m["name"] for m in models}
    # ≥3 models per tool, across domains — "works on any differentiable ODE", not "the demo".
    ident = {m["name"] for m in models if m["supports_identifiability"]}
    oed = {m["name"] for m in models if m["supports_oed"]}
    assert len(ident) >= 3 and len(oed) >= 3
    assert {"glv", "ad_qsp", "logistic"} <= names
    assert len({m["domain"] for m in models}) >= 3  # multiple scientific domains


# --------------------------------------------------------------------------- #
# identifiability — generality + honesty
# --------------------------------------------------------------------------- #
def test_identifiability_result_shape_two_models() -> None:
    """The tool runs the real diagnostic on ≥2 different models and returns a full verdict."""
    server = build_server()
    for model in ("linear_pathway", "logistic"):
        out = _call(server, "identifiability", {"model": model, "with_figure": False})
        assert out["tool"] == "identifiability"
        assert out["model"] == model
        assert out["verdict"] in {
            "well-constrained", "sloppy-but-predictive", "unidentifiable"
        }
        assert out["limitation"] == "NUDGE-LIM-023"
        assert isinstance(out["param_names"], list) and out["n_params"] >= 1
        assert "cond_number" in out and "null_directions" in out


def test_identifiability_carries_inline_figure_with_provenance(monkeypatch) -> None:
    """The FIM-spectrum figure rides back inline (base64) with the fig.py + data sidecar."""
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("NUDGE_ENV", "cloud")
    server = build_server()
    out = _call(server, "identifiability", {"model": "linear_pathway"})
    fig = out["figure"]
    assert fig["transport"] == "inline"
    assert fig["kind"] == "identifiability"
    assert fig["image_base64"] and len(fig["image_base64"]) <= 1_500_000
    assert fig["code"] and "viz.render" in fig["code"]  # the regenerating fig.py
    assert fig["data"] and "identifiability" in fig["data"]  # the sidecar


def test_identifiability_decoy_well_conditioned_not_flagged_sloppy() -> None:
    """DECOY: a well-posed model must be well-constrained — never mislabelled sloppy/unident."""
    server = build_server()
    out = _call(server, "identifiability", {"model": "well_conditioned", "with_figure": False})
    assert out["verdict"] == "well-constrained"
    assert out["is_sloppy"] is False
    assert out["abstained"] is False


def test_identifiability_decoy_rank_deficient_abstains() -> None:
    """DECOY: a structural null (A·e^{-(k1+k2)t}) must abstain + NAME the null direction."""
    server = build_server()
    out = _call(server, "identifiability",
                {"model": "redundant_exponential", "with_figure": False})
    assert out["verdict"] == "unidentifiable"
    assert out["abstained"] is True
    assert out["n_null_dims"] >= 1
    assert out["null_directions"] and out["null_directions"][0]["hint"]


def test_identifiability_free_subset_restricts_params() -> None:
    """The free-parameter selection knob restricts the FIM to the named subset."""
    server = build_server()
    out = _call(server, "identifiability",
                {"model": "glv", "free": "alpha[0],beta[0][0]", "with_figure": False})
    assert out["param_names"] == ["alpha[0]", "beta[0][0]"]
    assert out["n_params"] == 2


def test_identifiability_unknown_model_errors_cleanly() -> None:
    server = build_server()
    out = _call(server, "identifiability", {"model": "not_a_model", "with_figure": False})
    assert "error" in out


# --------------------------------------------------------------------------- #
# oed — measured gain + the inline GIF
# --------------------------------------------------------------------------- #
def test_oed_measures_improvement_and_returns_gif(monkeypatch) -> None:
    """OED returns the MEASURED CRLB / eigenvalue lift + the ellipse-collapse GIF inline."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    monkeypatch.setenv("NUDGE_ENV", "cloud")
    server = build_server()
    out = _call(server, "oed", {"model": "logistic", "steps": 120, "n_frames": 8})
    assert out["tool"] == "oed"
    assert out["limitation"] == "NUDGE-LIM-024"
    # MEASURED, not asserted — the naive design is confounded, the optimum resolves it.
    assert out["crlb_improvement"] > 1.0
    assert out["min_eig_improvement"] > 1.0
    fig = out["figure"]
    assert fig["transport"] == "inline"
    assert fig["mime_type"] == "image/gif"
    # inlined (small) or a static preview above the cap — never over the cap, never dropped.
    assert fig["image_base64"] is None or len(fig["image_base64"]) <= 1_500_000
    assert fig["code"]  # the regenerating fig.py provenance


def test_oed_second_model_also_resolves() -> None:
    """Generality: a DIFFERENT OED model (gLV) also produces a measured improvement."""
    server = build_server()
    out = _call(server, "oed", {"model": "glv", "steps": 120, "with_figure": False})
    assert out["model"] == "glv"
    assert out["crlb_improvement"] > 1.0


def test_oed_unsupported_model_errors_cleanly() -> None:
    server = build_server()
    out = _call(server, "oed", {"model": "sum_of_exponentials", "with_figure": False})
    assert "error" in out


# --------------------------------------------------------------------------- #
# async job pattern (both tools can be slow at scale)
# --------------------------------------------------------------------------- #
def test_identifiability_via_job_submit() -> None:
    import time

    server = build_server()
    sub = _call(server, "job_submit",
                {"tool": "identifiability",
                 "args_json": '{"model": "well_conditioned", "with_figure": false}'})
    assert sub["status"] == "running" and sub["job_id"]
    st = {"status": "running"}
    for _ in range(100):
        st = _call(server, "job_status", {"job_id": sub["job_id"]})
        if st["status"] != "running":
            break
        time.sleep(0.2)
    assert st["status"] == "done", st
    assert st["result"]["verdict"] == "well-constrained"
