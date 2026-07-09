"""NUDGE MCP server — exposes NUDGE to a Claude client as tools.

A thin FastMCP (``stdio``) adapter over the tested engine + the Mechanism-Card
knowledge base, so Claude (Claude Code / Desktop / the Claude Science workbench)
gets exactly the honest, abstaining output a human gets from the ``nudge`` CLI —
including *why* NUDGE abstained. See ``design/INTEGRATION_FEASIBILITY.md`` for the
verified connection recipes.

Tools:
- ``attribute(h5ad_path, target, ...)`` — run covariance attribution at one
  operating point; returns the call(s) + honest skip/abstention reasons.
- ``explain_abstention(context)`` — pull the Mechanism Card + decoy/limitation that
  explains a verdict (``off-model`` / ``unresolved`` / a decoy or limitation id).
- ``list_mechanisms()`` — the registered mechanism library.
- ``get_mechanism_card(name)`` — the full Markdown card for a mechanism.

Run: ``uv run nudge-mcp`` (or ``python -m nudge.mcp.server``). The ``mcp`` SDK is an
optional dependency: install ``nudge-bio[mcp]``.

Register (Claude Code):  ``claude mcp add --scope project nudge -- uv run nudge-mcp``
"""

from __future__ import annotations

from typing import Any


def _require_mcp() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SystemExit(
            "The MCP SDK is not installed. Install the optional extra:\n"
            "    uv pip install -e '.[mcp]'   (or:  pip install mcp)\n"
            f"(import error: {exc})"
        ) from exc
    return FastMCP


def build_server() -> Any:
    """Construct and return the FastMCP server with NUDGE's tools registered."""
    FastMCP = _require_mcp()
    mcp = FastMCP("nudge")

    @mcp.tool()
    def list_mechanisms() -> list[dict[str, Any]]:
        """List NUDGE's registered mechanism library (id, role, summary, card)."""
        from nudge.knowledge import list_mechanisms as _list

        return _list()

    @mcp.tool()
    def get_mechanism_card(name: str) -> str:
        """Return the Markdown Mechanism Card for ``name`` (e.g. 'hill_activation').

        ``name`` may be a card stem, a registry key (``HillActivation``), or a
        card ``name:``. Returns a not-found message if there is no such card.
        """
        from nudge.knowledge import get_mechanism_card as _card

        card = _card(name)
        return card if card is not None else f"No Mechanism Card found for {name!r}."

    @mcp.tool()
    def explain_abstention(context: str) -> dict[str, Any]:
        """Explain *why* NUDGE returned a given verdict.

        ``context`` may be an abstention class (``off-model``, ``unresolved``,
        ``no-effect``, ``technical-artifact``), a decoy id (``NUDGE-DECOY-001``), a
        limitation id (``NUDGE-LIM-006``), or a mechanism name. Returns the
        matching decoy(s), documented limitation(s), and Mechanism Card(s) to read.
        """
        from nudge.knowledge import explain as _explain

        return _explain(context)

    @mcp.tool()
    def attribute(
        h5ad_path: str,
        target: str,
        topology: str = "1node",
        control: str = "WT",
        steps: int = 200,
        min_cells: int = 200,
        preset: str = "native",
    ) -> dict[str, Any]:
        """Attribute a perturbation's mechanism at one operating point.

        Fits the circuit hypothesis (``topology``: ``1node`` / ``2node`` / ``toggle``)
        and runs covariance attribution for ``target`` vs ``control``. Single-condition
        attribution is *expected* to abstain between gain and threshold (the measured
        Fisher degeneracy); the breaker needs a second operating point. Returns the
        call(s) plus honest ``skipped`` reasons — it never forces a confident guess.
        Use ``explain_abstention`` on any non-positive verdict.
        """
        from nudge.service import attribute_file, report_to_dict

        report = attribute_file(
            h5ad_path,
            target,
            topology=topology,
            control=control,
            steps=steps,
            min_cells=min_cells,
            preset=preset,
        )
        return report_to_dict(report)

    return mcp


def main() -> None:
    """Console-script entry point (``nudge-mcp``): run the server over stdio."""
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
