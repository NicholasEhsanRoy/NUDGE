"""Tests for the NUDGE MCP server (thin FastMCP adapter)."""

from __future__ import annotations

import asyncio

import pytest

mcp = pytest.importorskip("mcp")  # skip if the optional [mcp] extra is absent

from nudge.mcp.server import _unwrap, build_server  # noqa: E402

# ``_unwrap`` is imported from the server (single source of truth): it only unstrips a
# single-key ``{"result": …}`` envelope, so a tool whose OWN payload has a ``result`` key
# (e.g. ``job_status``) is not accidentally over-unwrapped.


def test_server_registers_the_expected_tools() -> None:
    server = build_server()
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "attribute",
        "dose_response",
        "synergy",
        "cross_modality",
        "robustness",
        "design",
        "multi_reporter",
        "differential",
        "differential_robust",
        "lotka",
        "fibrillization",
        "constitutive",
        "diagnose_abstention",
        "explain_abstention",
        "list_mechanisms",
        "get_mechanism_card",
        "render_figure",
        "identifiability",
        "oed",
        "list_models",
        "job_submit",
        "job_status",
    }


def test_explain_abstention_tool_matches_knowledge() -> None:
    server = build_server()
    call = server.call_tool("explain_abstention", {"context": "unresolved"})
    out = _unwrap(asyncio.run(call))
    assert out["kind"] == "abstention"
    assert out["verdict"] == "unresolved"


def test_list_mechanisms_tool() -> None:
    server = build_server()
    out = _unwrap(asyncio.run(server.call_tool("list_mechanisms", {})))
    assert len(out) >= 6
    assert any(m["registry_name"] == "HillActivation" for m in out)


def test_get_mechanism_card_tool() -> None:
    server = build_server()
    call = server.call_tool("get_mechanism_card", {"name": "hill_activation"})
    out = _unwrap(asyncio.run(call))
    text = out if isinstance(out, str) else str(out)
    assert "Hill" in text


def test_render_figure_tool_demo_path_transport(monkeypatch) -> None:
    """PATH transport (NUDGE_ENV unset): render_figure writes a file + inline provenance text."""
    pytest.importorskip("matplotlib")
    monkeypatch.delenv("NUDGE_ENV", raising=False)
    server = build_server()
    call = server.call_tool("render_figure", {"kind": "identifiability", "demo": True})
    out = _unwrap(asyncio.run(call))
    assert out["kind"] == "identifiability"
    assert out["transport"] == "path"
    assert out["abstained"] is False  # sloppy-but-predictive is usable, not an abstention
    import os

    assert out["image_path"] and os.path.getsize(out["image_path"]) > 0
    assert out["png_path"] == out["image_path"]  # back-compat alias
    assert out["image_base64"] is None
    assert out["code"] and "viz.render" in out["code"]  # the regenerating fig.py, inline
    assert out["data"] and "identifiability" in out["data"]  # the sidecar, inline


def test_render_figure_tool_inline_transport(monkeypatch) -> None:
    """INLINE transport (NUDGE_ENV=cloud): the image rides back as size-capped base64."""
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("NUDGE_ENV", "cloud")
    server = build_server()
    call = server.call_tool("render_figure", {"kind": "identifiability", "demo": True})
    out = _unwrap(asyncio.run(call))
    assert out["transport"] == "inline"
    assert out["mime_type"] == "image/png"
    assert out["image_base64"] and len(out["image_base64"]) <= 1_500_000
    assert out["image_path"] is None  # nothing to read from a path in the cloud sandbox
    assert out["code"] and out["data"]  # provenance travels inline as text


def test_render_figure_animate_inline_gif(monkeypatch) -> None:
    """An animated GIF inlines under the size discipline; an abstention survives transport."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    monkeypatch.setenv("NUDGE_ENV", "cloud")
    server = build_server()
    result_json = (
        '{"kind":"constitutive","call":"unresolved","reason":"flat","label":"linear",'
        '"n_grid":[1,2,3,4,5],"loss_no_control":[1,1,1,1,1],'
        '"loss_with_control":[1,1,1,1,1],"n1_rejection":0.0,'
        '"argmin_n_with_control":3.0,"calibration":{"h":1.0,"km":0.5,"vmax":10,'
        '"base":0.1,"r2":0.9,"is_nonlinear":false},"asserts_biological_switch":false}'
    )
    out = _unwrap(asyncio.run(server.call_tool(
        "render_figure",
        {"kind": "constitutive", "animate": True, "result_json": result_json},
    )))
    assert out["transport"] == "inline"
    assert out["abstained"] is True  # LOAD-BEARING: the overlay flag survives the transport
    # inlined (small) or fell back to a preview — never over the cap, never silently dropped
    assert out["image_base64"] is None or len(out["image_base64"]) <= 1_500_000


def test_render_figure_animate_unanimatable_kind() -> None:
    """Requesting an animation for a kind with no natural frame variable errors cleanly."""
    server = build_server()
    out = _unwrap(asyncio.run(server.call_tool(
        "render_figure", {"kind": "epistasis", "demo": True, "animate": True}
    )))
    assert "error" in out and "animatable_kinds" in out


def test_render_figure_tool_unknown_kind() -> None:
    server = build_server()
    out = _unwrap(asyncio.run(server.call_tool("render_figure", {"kind": "nope"})))
    assert "error" in out and "known_kinds" in out


def test_job_submit_status_round_trip() -> None:
    """job_submit returns a job_id fast; job_status polls to done with the real result."""
    import time

    server = build_server()
    sub = _unwrap(asyncio.run(server.call_tool(
        "job_submit", {"tool": "list_mechanisms", "args_json": "{}"}
    )))
    assert sub["status"] == "running" and sub["job_id"] and sub["tool"] == "list_mechanisms"
    job_id = sub["job_id"]
    for _ in range(50):
        st = _unwrap(asyncio.run(server.call_tool("job_status", {"job_id": job_id})))
        if st["status"] != "running":
            break
        time.sleep(0.1)
    assert st["status"] == "done", st
    # the job result equals calling the tool directly
    assert isinstance(st["result"], list)
    assert any(m["registry_name"] == "HillActivation" for m in st["result"])
    assert "elapsed_s" in st


def test_job_submit_surfaces_tool_error() -> None:
    """A tool that fails inside a job is reported as status='error', not a crash."""
    import time

    server = build_server()
    sub = _unwrap(asyncio.run(server.call_tool(
        "job_submit", {"tool": "fibrillization", "args_json": "{}"}  # missing required path
    )))
    job_id = sub["job_id"]
    for _ in range(50):
        st = _unwrap(asyncio.run(server.call_tool("job_status", {"job_id": job_id})))
        if st["status"] != "running":
            break
        time.sleep(0.1)
    assert st["status"] == "error" and "error" in st


def test_job_submit_rejects_recursive_and_unknown_job() -> None:
    server = build_server()
    out = _unwrap(asyncio.run(server.call_tool(
        "job_submit", {"tool": "job_status", "args_json": "{}"}
    )))
    assert "error" in out
    miss = _unwrap(asyncio.run(server.call_tool("job_status", {"job_id": "nope"})))
    assert "error" in miss
