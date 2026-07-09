"""Application service — the one place the CLI and the MCP server share.

Thin orchestration over the tested engine: build the circuit hypothesis, parse
markers, load the file, and run
:func:`nudge.inference.pipeline.attribute_across_operating_points`. No modelling
logic of its own; it exists so the ``nudge`` CLI and the MCP server give byte-for-
byte identical attribution results and can be tested once, here.
"""

from __future__ import annotations

from typing import Any

TOPOLOGIES = ("1node", "2node", "toggle")


def build_circuit(topology: str) -> Any:
    """Return a named circuit motif (``1node`` / ``2node`` / ``toggle``)."""
    if topology not in TOPOLOGIES:
        raise ValueError(f"topology must be one of {TOPOLOGIES}, got {topology!r}")
    from nudge.circuits import ras_switch_1node, ras_switch_2node, toggle

    return {"1node": ras_switch_1node, "2node": ras_switch_2node, "toggle": toggle}[
        topology
    ]()


def parse_markers(circuit: Any, marker: list[str] | None) -> dict[str, list[str]]:
    """Parse ``SPECIES=GENE1,GENE2`` strings; default each species to its own name.

    The default (species name == marker gene) is what the synthetic generator emits
    (one gene per species), so attribution Just Works on synthetic data.
    """
    markers: dict[str, list[str]] = {}
    for spec in marker or []:
        if "=" not in spec:
            raise ValueError(f"marker {spec!r} must be SPECIES=GENE1,GENE2")
        species, _, genes = spec.partition("=")
        markers[species.strip()] = [g.strip() for g in genes.split(",") if g.strip()]
    for name in circuit.names:
        markers.setdefault(name, [name])
    return markers


def _load(path: str, target: str, preset: str) -> Any:
    if preset == "gladstone":
        from nudge.data.loaders.tier2 import load_gladstone

        return load_gladstone(path, target_genes=(target,))
    import anndata as ad

    return ad.read_h5ad(path, backed=None)


def attribute_file(
    path: str,
    target: str,
    *,
    topology: str = "1node",
    markers: list[str] | None = None,
    control: str = "WT",
    steps: int = 200,
    min_cells: int = 200,
    preset: str = "native",
) -> Any:
    """Attribute ``target`` at one operating point; returns an ``AttributionReport``.

    Single-condition attribution is expected to abstain between gain and threshold
    (the measured Fisher degeneracy); skips (too few cells / unreliable LNA) are
    recorded in the report, not hidden.
    """
    from nudge.inference.pipeline import attribute_across_operating_points

    circuit = build_circuit(topology)
    adata = _load(path, target, preset)
    marker_map = parse_markers(circuit, markers)
    return attribute_across_operating_points(
        {"op": adata},
        circuit,
        marker_map,
        target,
        wt_condition=control,
        min_cells=min_cells,
        steps=steps,
    )


def report_to_dict(report: Any) -> dict[str, Any]:
    """Serialise an ``AttributionReport`` to plain JSON-able dicts (for MCP)."""
    return {
        "target": report.target,
        "n_cells": dict(report.n_cells),
        "single": {
            label: {"call": call, "nlls": dict(nlls)}
            for label, (call, nlls) in report.single.items()
        },
        "multi": (
            {"call": report.multi[0], "nlls": dict(report.multi[1])}
            if report.multi is not None
            else None
        ),
        "skipped": dict(report.skipped),
    }
