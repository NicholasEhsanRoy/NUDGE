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


# --------------------------------------------------------------------------- #
# dose-response attribution (the same K/n/v_max vocabulary, a dose axis instead
# of single cells — a second measurement of one circuit; see inference.dose_response)
# --------------------------------------------------------------------------- #
def _dose_points(
    path: str,
    *,
    dose_col: str,
    response_col: str,
    target: str | None,
    target_gene: str | None,
    signature: list[str] | None,
    group_col: str,
    control: str,
    min_cells: int,
) -> Any:
    """Read ``(dose, response)`` from a two-column CSV/TSV, or extract it from an
    ``.h5ad`` knockdown screen (per-guide fractional knockdown → dose, signature →
    response).
    """
    if path.endswith(".h5ad"):
        if not (target and target_gene and signature):
            raise ValueError(
                ".h5ad input needs --target, --target-gene, and --signature"
            )
        import anndata as ad

        from nudge.inference.bridge import knockdown_dose_response

        adata = ad.read_h5ad(path, backed=None)
        return knockdown_dose_response(
            adata,
            target_gene=target_gene,
            signature=signature,
            group_prefix=target,
            group_col=group_col,
            control_label=control,
            min_cells_per_group=min_cells,
        )
    import pandas as pd

    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep)
    for col in (dose_col, response_col):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not in {list(df.columns)}")
    return df[dose_col].to_numpy(dtype=float), df[response_col].to_numpy(dtype=float)


def dose_response_to_dict(res: Any) -> dict[str, Any]:
    """Serialise a ``DoseResponseResult`` to a plain JSON-able dict (for CLI / MCP)."""
    f = res.fit
    return {
        "call": res.call,
        "reason": res.reason,
        "n_apparent_gain": f.n,
        "ci_n": list(f.ci_n),
        "K_threshold": f.k_threshold,
        "ci_K": list(f.ci_k),
        "amp": f.amp,
        "floor": f.floor,
        "r2": f.r2,
        "graded_r2": f.graded_r2,
        "bic_switch": f.bic_switch,
        "bic_graded": f.bic_graded,
        "delta_bic_graded_minus_switch": f.bic_graded - f.bic_switch,
        "n_points": f.n_points,
        "n_boot": f.n_boot,
        "dose_range": [f.dose_min, f.dose_max],
        "spans_inflection": f.spans_inflection,
        "direction": f.direction,
    }


def dose_response_file(
    path: str,
    *,
    direction: str = "repress",
    dose_col: str = "dose",
    response_col: str = "response",
    target: str | None = None,
    target_gene: str | None = None,
    signature: list[str] | None = None,
    group_col: str = "guide",
    control: str = "WT",
    min_cells: int = 15,
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit + classify a dose-response curve from a CSV/TSV or an ``.h5ad`` screen.

    Returns the verdict (``switch`` / ``graded`` / ``no-effect`` / ``unresolved``) with
    the apparent gain ``n`` + CI and the honest abstention reason — never a forced call.
    The reported ``n`` is an **apparent population gain**, not molecular cooperativity.
    """
    from nudge.inference.dose_response import attribute_dose_response

    dose, response = _dose_points(
        path,
        dose_col=dose_col,
        response_col=response_col,
        target=target,
        target_gene=target_gene,
        signature=signature,
        group_col=group_col,
        control=control,
        min_cells=min_cells,
    )
    res = attribute_dose_response(
        dose, response, direction=direction, n_boot=n_boot, seed=seed
    )
    return dose_response_to_dict(res)


# --------------------------------------------------------------------------- #
# synergy / epistasis attribution (A / B / A+B as three operating points; the
# additive Bliss null vs a non-additive combo — see inference.epistasis)
# --------------------------------------------------------------------------- #
def synergy_to_dict(res: Any) -> dict[str, Any]:
    """Serialise an ``EpistasisResult`` to a plain JSON-able dict (for CLI / MCP)."""
    f = res.fit
    return {
        "call": res.call,
        "reason": res.reason,
        "effect_a": f.effect_a,
        "effect_b": f.effect_b,
        "effect_ab": f.effect_ab,
        "ci_a": list(f.ci_a),
        "ci_b": list(f.ci_b),
        "ci_ab": list(f.ci_ab),
        "additive_pred": f.additive_pred,
        "interaction": f.interaction,
        "ci_interaction": list(f.ci_interaction),
        "bic_additive": f.bic_additive,
        "bic_free": f.bic_free,
        "delta_bic_additive_minus_free": f.bic_additive - f.bic_free,
        "n_cells": {
            "control": f.n_control,
            "A": f.n_a,
            "B": f.n_b,
            "A+B": f.n_ab,
        },
        "n_boot": f.n_boot,
        "effect_space": f.effect_space,
    }


def synergy_file(
    path: str,
    *,
    control_label: str = "control",
    a_label: str,
    b_label: str,
    ab_label: str,
    condition_col: str = "condition",
    signature: list[str] | None = None,
    library_col: str | None = "total_counts",
    n_top_genes: int = 2000,
    n_boot: int = 1000,
    seed: int = 0,
    bic_margin: float = 2.0,
    min_cells: int = 30,
    rel_width: float = 0.5,
) -> dict[str, Any]:
    """Classify a two-perturbation combination from an ``.h5ad`` — the CLI/MCP entry.

    Reads the {control, A, B, A+B} conditions of ``path`` (by ``condition_col`` labels),
    reduces each to a per-cell **effect score** (projection onto the additive axis fixed
    by the singles, or a fixed ``signature``; both depth-normalized, log-fold-change
    space), and returns the verdict (``additive`` / ``synergistic`` / ``buffering`` /
    ``no-effect`` / ``unresolved``) with the interaction + its bootstrap CI and the
    honest abstention reason — never a forced call.
    """
    import anndata as ad

    from nudge.inference.bridge import combo_effect_scores
    from nudge.inference.epistasis import attribute_synergy

    adata = ad.read_h5ad(path, backed=None)
    control, a, b, ab = combo_effect_scores(
        adata,
        control_label=control_label,
        a_label=a_label,
        b_label=b_label,
        ab_label=ab_label,
        condition_col=condition_col,
        library_col=library_col,
        signature=signature,
        n_top_genes=n_top_genes,
    )
    res = attribute_synergy(
        control,
        a,
        b,
        ab,
        n_boot=n_boot,
        seed=seed,
        bic_margin=bic_margin,
        min_cells=min_cells,
        rel_width=rel_width,
    )
    return synergy_to_dict(res)
