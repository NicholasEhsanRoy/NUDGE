"""Tests for the NUDGE MCP server (thin FastMCP adapter)."""

from __future__ import annotations

import asyncio

import pytest

mcp = pytest.importorskip("mcp")  # skip if the optional [mcp] extra is absent

from nudge.mcp.server import build_server  # noqa: E402


def _unwrap(result):
    """FastMCP call_tool returns (content, structured) across versions; take that."""
    structured = result[1] if isinstance(result, tuple) else result
    if isinstance(structured, dict) and "result" in structured:
        return structured["result"]
    return structured


def test_server_registers_the_expected_tools() -> None:
    server = build_server()
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "attribute",
        "dose_response",
        "synergy",
        "explain_abstention",
        "list_mechanisms",
        "get_mechanism_card",
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
