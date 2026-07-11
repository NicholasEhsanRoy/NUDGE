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


def render_result(
    kind: str,
    result_or_dict: Any,
    *,
    out: str | None,
    emit_code: bool = True,
    theme: str = "auto",
    self_contained: bool = False,
    animate: bool = False,
    inline_png: bool = False,
    cli_call: str | None = None,
    **ctx: Any,
) -> dict[str, Any]:
    """Render a NUDGE result to a figure — the one place CLI + MCP share the figure path.

    Lazy-imports the opt-in :mod:`nudge.viz` (raising the friendly ``[viz]``-extra install
    message if absent), dispatches on ``kind``, and returns the ``FigureResult`` as a
    plain dict (paths + honest caption + ``abstained`` flag + optional size-capped inline
    PNG). ``ctx`` carries any renderer inputs the serialized dict lacks — e.g. the raw
    ``dose`` / ``response`` points for the dose-response scatter. The abstention overlay
    is applied by :mod:`nudge.viz` off the result's own verdict; this seam never re-fits.
    """
    import nudge.viz as viz

    fr = viz.render(
        result_or_dict,
        out,
        kind=kind,
        emit_code=emit_code,
        theme=theme,
        self_contained=self_contained,
        animate=animate,
        inline_png=inline_png,
        cli_call=cli_call,
        **ctx,
    )
    return {
        "png_path": fr.path,
        "code_path": fr.code_path,
        "data_path": fr.data_path,
        "png_base64": fr.png_base64,
        "png_base64_omitted_reason": (
            None
            if (fr.png_base64 is not None or not inline_png)
            else "exceeds inline cap; read png_path"
        ),
        "caption": fr.caption,
        "abstained": fr.abstained,
        "kind": fr.kind,
    }


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
    fig_out: str | None = None,
    fig_code: bool = True,
    fig_theme: str = "auto",
    fig_self_contained: bool = False,
    fig_label: str | None = None,
    cli_call: str | None = None,
) -> dict[str, Any]:
    """Fit + classify a dose-response curve from a CSV/TSV or an ``.h5ad`` screen.

    Returns the verdict (``switch`` / ``graded`` / ``no-effect`` / ``unresolved``) with
    the apparent gain ``n`` + CI and the honest abstention reason — never a forced call.
    The reported ``n`` is an **apparent population gain**, not molecular cooperativity.

    When ``fig_out`` is given, an opt-in figure of the fit is written (PNG + a
    regenerating ``fig.py`` + data sidecar unless ``fig_code=False``) via the shared
    :func:`render_result` seam, and the returned dict gains a ``"figure"`` key with the
    written paths + honest caption. The default (no ``fig_out``) behaviour is unchanged.
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
    out = dose_response_to_dict(res)
    if fig_out is not None:
        out["figure"] = render_result(
            "dose_response",
            res,
            out=fig_out,
            emit_code=fig_code,
            theme=fig_theme,
            self_contained=fig_self_contained,
            dose=dose,
            response=response,
            label=fig_label or (target or "dose-response"),
            cli_call=cli_call,
        )
    return out


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
        "off_axis_residual": f.off_axis_residual,
        "neomorphic_ratio": f.neomorphic_ratio,
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
    control, a, b, ab, geometry = combo_effect_scores(
        adata,
        control_label=control_label,
        a_label=a_label,
        b_label=b_label,
        ab_label=ab_label,
        condition_col=condition_col,
        library_col=library_col,
        signature=signature,
        n_top_genes=n_top_genes,
        return_geometry=True,
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
        geometry=geometry,
    )
    return synergy_to_dict(res)


# --------------------------------------------------------------------------- #
# cross-modality readout attribution (the same K/n/v_max, read from a CONTINUOUS
# single channel — fluorescence / activity / fold-change — not counts; a panel of
# variants localized to threshold / gain / ceiling vs a control — see
# inference.cross_modality). The Chure-2019 LacI benchmark's engine.
# --------------------------------------------------------------------------- #
def variant_attribution_to_dict(v: Any) -> dict[str, Any]:
    """Serialise one ``VariantAttribution`` to a plain JSON-able dict (CLI / MCP)."""
    return {
        "variant": v.variant,
        "class_label": v.class_label,
        "call": v.call,
        "reason": v.reason,
        "knob": v.knob,
        "knob_reason": v.knob_reason,
        "K_threshold": v.k_threshold,
        "ci_K": list(v.ci_k),
        "n_apparent_gain": v.n,
        "ci_n": list(v.ci_n),
        "amp": v.amp,
        "ci_amp": list(v.ci_amp),
        "floor": v.floor,
        "ci_floor": list(v.ci_floor),
        "r2": v.r2,
        "n_points": v.n_points,
        "log2_K_ratio_vs_control": v.log2_k_ratio,
        "delta_floor_vs_control": v.delta_floor,
        "delta_n_vs_control": v.delta_n,
    }


def _coerce(val: str) -> Any:
    """Coerce a CLI ``key=value`` filter value to float when it looks numeric."""
    try:
        return float(val)
    except ValueError:
        return val


def cross_modality_panel_file(
    path: str,
    *,
    dose_col: str,
    response_col: str,
    variant_col: str,
    control_variant: str,
    class_col: str | None = None,
    variants: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    modality: str = "fluorescence",
    direction: str = "activate",
    autofluor: float = 0.0,
    agg: str = "mean",
    n_boot: int = 400,
    seed: int = 0,
) -> dict[str, Any]:
    """Attribute a panel of **continuous-readout** dose-responses — the CLI/MCP entry.

    Reads a tidy CSV/TSV of a continuous single-channel readout (fluorescence /
    activity / fold-change, declared by ``modality`` — NUDGE never guesses it and the
    bouncer refuses log-normalized or raw counts, NUDGE-LIM-008), extracts each
    variant's ``(dose, response)`` curve, fits + classifies it with the shipped
    dose-response path, and localizes each variant's effect to one knob (**threshold** /
    **gain** / **ceiling**) vs ``control_variant`` — or abstains (**non-responsive** /
    **inconclusive**). Returns the per-variant table (with the author/ground-truth
    ``class_col`` label carried through, if given) — never a forced call. This is the
    Chure-2019 LacI benchmark's engine: DNA-binding mutants localize to
    ceiling/leakiness, inducer-binding mutants to threshold.
    """
    import pandas as pd

    from nudge.inference.cross_modality import attribute_variant_panel

    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep, comment="#")
    coerced = {
        k: _coerce(v) if isinstance(v, str) else v for k, v in (filters or {}).items()
    }
    panel = attribute_variant_panel(
        df,
        dose_col=dose_col,
        response_col=response_col,
        variant_col=variant_col,
        control_variant=control_variant,
        variants=variants,
        class_col=class_col,
        filters=coerced or None,
        modality=modality,
        direction=direction,
        autofluor=autofluor,
        agg=agg,
        n_boot=n_boot,
        seed=seed,
    )
    return {
        "modality": modality,
        "direction": direction,
        "control_variant": control_variant,
        "variants": [variant_attribution_to_dict(v) for v in panel],
    }


# --------------------------------------------------------------------------- #
# bifurcation / tipping-point proximity (the "robustness dial": how close is a
# bistable switch to LOSING bistability — a saddle-node fold? — see
# inference.bifurcation). A one-sided lower bound near the fold, because the
# linear-noise Gaussian breaks down there (NUDGE-LIM-012).
# --------------------------------------------------------------------------- #
ROBUSTNESS_TOPOLOGIES = ("1node", "2node", "toggle")


def _build_named_circuit(
    topology: str, *, n: float, k: float, vmax: float, basal: float
) -> Any:
    """Build a named bistable motif with explicit switch kinetics (the what-if dial)."""
    if topology not in ROBUSTNESS_TOPOLOGIES:
        raise ValueError(
            f"topology must be one of {ROBUSTNESS_TOPOLOGIES}, got {topology!r}"
        )
    from nudge.circuits import ras_switch_1node, ras_switch_2node, toggle

    builders = {
        "1node": ras_switch_1node,
        "2node": ras_switch_2node,
        "toggle": toggle,
    }
    return builders[topology](n=n, K=k, vmax=vmax, basal=basal)


def bifurcation_to_dict(score: Any, call: str, reason: str) -> dict[str, Any]:
    """Serialise a ``BifurcationScore`` + its verdict to a plain JSON-able dict."""
    out: dict[str, Any] = {"call": call, "reason": reason}
    if score is None:
        out.update(
            {
                "proximity": None,
                "one_sided": None,
                "n_stable_modes": 0,
                "min_re_lambda": None,
                "node_saddle_distance": None,
                "lna_lobe_ratio": None,
                "channels": {},
            }
        )
    else:
        out.update(
            {
                "proximity": score.proximity,
                "one_sided": score.one_sided,
                "n_stable_modes": score.n_stable_modes,
                "min_re_lambda": score.min_re_lambda,
                "node_saddle_distance": score.node_saddle_distance,
                "lna_lobe_ratio": score.lna_lobe_ratio,
                "channels": dict(score.channels),
            }
        )
    return out


def robustness_circuit(
    topology: str = "1node",
    *,
    n: float = 6.0,
    k: float = 1.0,
    vmax: float = 2.0,
    basal: float = 0.05,
) -> dict[str, Any]:
    """Score a **parametric** bistable switch's fold proximity — the what-if entry.

    Builds the named motif at the given switch kinetics and returns the robustness dial:
    the fused 0..1 ``proximity`` + the three raw channels (critical slowing, basin
    collapse, LNA lobe swell) + the honest ``call`` (``near-fold`` / ``robust`` /
    ``unresolved`` / ``not-bistable``). Near the fold the number is a **one-sided lower
    bound** (``one_sided``): the LNA Gaussian breaks down there (NUDGE-LIM-012).
    """
    from nudge.inference.bifurcation import (
        bifurcation_proximity,
        classify_robustness,
    )

    circuit = _build_named_circuit(topology, n=n, k=k, vmax=vmax, basal=basal)
    score = bifurcation_proximity(circuit)
    call, reason = classify_robustness(score)
    out = bifurcation_to_dict(score, call, reason)
    out["topology"] = topology
    out["kinetics"] = {"n": n, "K": k, "vmax": vmax, "basal": basal}
    return out


def _read_activity(path: str) -> Any:
    """Read an ``(n_cells, n_species)`` activity array from ``.npy`` / CSV / TSV."""
    import numpy as np

    if path.endswith(".npy"):
        return np.load(path)
    import pandas as pd

    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    return pd.read_csv(path, sep=sep, comment="#").to_numpy(dtype=float)


def bifurcation_file(
    path: str,
    *,
    topology: str = "1node",
    n: float = 6.0,
    k: float = 1.0,
    vmax: float = 2.0,
    basal: float = 0.05,
    steps: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Score a switch's fold proximity from an **activity data file** + a topology.

    Reads an ``(n_cells, n_species)`` array (``.npy`` / CSV / TSV), builds the named
    circuit hypothesis at the given kinetics, and runs the data-driven attribution
    (:func:`nudge.inference.bifurcation.attribute_bifurcation`): it scores the circuit,
    and calibrates the sequencing **depth** from the data so the LNA lobe channel's
    reliability is honestly reported (``lna_reason``). The proximity is a property of
    the circuit; the data pins the depth context. Near the fold the number is a
    **one-sided lower bound** (NUDGE-LIM-012).
    """
    from nudge.inference.bifurcation import attribute_bifurcation

    arr = _read_activity(path)
    circuit = _build_named_circuit(topology, n=n, k=k, vmax=vmax, basal=basal)
    res = attribute_bifurcation(arr, circuit, free=None, steps=steps, seed=seed)
    out = bifurcation_to_dict(res.score, res.call, res.reason)
    out["topology"] = topology
    out["scale"] = res.scale
    out["lna_reason"] = res.lna_reason
    out["recovered"] = dict(res.recovered)
    return out


# --------------------------------------------------------------------------- #
# inverse / intervention design (the flagship — turn a diagnosis into a
# prescription: invert a RELIABLE attribution to propose an untested intervention,
# behind an integrity gate + a bifurcation safety gate — see design.invert).
# --------------------------------------------------------------------------- #
def _safety_to_dict(safety: Any) -> dict[str, Any] | None:
    """Serialise a ``SafetyReport`` (the Cap-5 before/after dial) to a dict."""
    if safety is None:
        return None
    return {
        "proximity_before": safety.proximity_before,
        "proximity_after": safety.proximity_after,
        "delta": safety.delta,
        "one_sided": safety.one_sided,
        "high_risk_of_instability": safety.high_risk_of_instability,
        "crosses_fold": safety.crosses_fold,
        "channels_before": dict(safety.channels_before),
        "channels_after": dict(safety.channels_after),
    }


def design_to_dict(plan: Any) -> dict[str, Any]:
    """Serialise an ``InterventionPlan`` or ``AbstentionResult`` to a JSON-able dict."""
    from nudge.design.invert import AbstentionResult

    if isinstance(plan, AbstentionResult):
        return {
            "kind": "abstention",
            "verdict": plan.verdict.value,
            "reason": plan.reason,
        }
    out: dict[str, Any] = {
        "kind": "intervention",
        "mode": plan.mode,
        "reason": plan.reason,
        "achieved_loss": plan.achieved_loss,
    }
    if plan.mode == "dose":
        out["dose"] = plan.dose
        out["predicted_response"] = plan.predicted_response
        out["safety"] = None
    else:
        out["deltas"] = [
            {
                "param": {"scope": s, "index": i, "name": n},
                "log_delta": ld,
                "factor": fac,
            }
            for (s, i, n), ld, fac in plan.deltas
        ]
        out["predicted_state"] = (
            list(plan.predicted_state) if plan.predicted_state is not None else None
        )
        out["safety"] = _safety_to_dict(plan.safety)
    return out


def design_circuit(
    topology: str = "1node",
    *,
    n: float = 6.0,
    k: float = 1.0,
    vmax: float = 2.0,
    basal: float = 0.05,
    to: str = "high",
    start: str = "low",
    free: list[str] | None = None,
    steps: int = 400,
    l1: float = 1e-2,
    tol: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Invert a named bistable circuit to flip it ``to`` a basin (circuit-mode entry).

    Builds the named motif at the given switch kinetics, targets the high/low stable
    state (``to``), and runs :func:`nudge.design.invert.design` over the addressable
    kinetic knobs (``free`` names like ``edge0.K`` / ``species0.basal``; default = the
    full set), starting from the ``start`` basin. Returns the proposed Δ + the Cap-5
    **safety** verdict (does the intervention push the switch toward / over its fold?),
    or an abstention if the target is unreachable within the fitted region.
    """
    from nudge.design.invert import CircuitFit, design, flip_target

    circuit = _build_named_circuit(topology, n=n, k=k, vmax=vmax, basal=basal)
    knobs = _parse_free(free) if free else None
    target = flip_target(circuit, to=to)
    plan = design(
        CircuitFit(circuit=circuit, free=knobs),
        target,
        steps=steps,
        l1=l1,
        tol=tol,
        seed=seed,
        start=start,
    )
    out = design_to_dict(plan)
    out["topology"] = topology
    out["kinetics"] = {"n": n, "K": k, "vmax": vmax, "basal": basal}
    out["target_basin"] = to
    return out


def _parse_free(free: list[str]) -> list[tuple[str, int, str]]:
    """Parse ``edge0.K`` / ``species1.basal`` knob strings into ``FreeParam`` tuples."""
    out: list[tuple[str, int, str]] = []
    for spec in free:
        head, _, name = spec.partition(".")
        if not name:
            raise ValueError(f"knob {spec!r} must look like 'edge0.K'/'species0.basal'")
        if head.startswith("edge"):
            out.append(("edge", int(head[len("edge") :]), name))
        elif head.startswith("species"):
            out.append(("species", int(head[len("species") :]), name))
        else:
            raise ValueError(f"knob {spec!r} must start with 'edge' or 'species'")
    return out


def design_file(
    path: str,
    *,
    target_response: float,
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
    """Invert a **real-data dose-response** fit to a dose achieving ``target_response``.

    Fits the dose-response curve from a CSV/TSV or an ``.h5ad`` screen (the same
    inputs as :func:`dose_response_file`), then :func:`nudge.design.invert.design`
    invert it: what dose achieves a target readout ``y``? Behind the **integrity gate**
    (refuses to invert an ``unresolved`` / ``no-effect`` fit) and with an **honest
    reachability abstention** when ``y`` is outside the curve's achievable range. Curve
    mode carries **no** bifurcation safety gate (no circuit/fold), stated in the
    plan's ``reason``.
    """
    from nudge.design.invert import design
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
    plan = design(res, target_response)
    out = design_to_dict(plan)
    out["attribution_call"] = res.call
    out["target_response"] = target_response
    return out


# --------------------------------------------------------------------------- #
# multi-reporter joint attribution (several downstream reporters of ONE latent
# switch, fit jointly to break the K⇄v_max degeneracy — see
# inference.multi_reporter). The panel over-determines the latent, so the joint
# fit RESOLVES threshold/gain/ceiling where a single reporter ABSTAINS; a panel
# that cannot be explained by one shared latent abstains off-model (NUDGE-LIM-014).
# --------------------------------------------------------------------------- #
def multi_reporter_to_dict(res: Any) -> dict[str, Any]:
    """Serialise a ``MultiReporterResult`` to a plain JSON-able dict (CLI / MCP)."""
    f = res.fit
    return {
        "call": res.call,
        "reason": res.reason,
        "direction": f.direction,
        "n_reporters": f.n_reporters,
        "pinned_affine": f.pinned_affine,
        "winner": f.winner,
        "knob_margin": f.knob_margin,
        "effect_margin": f.effect_margin,
        "k_wt": f.k_wt,
        "n_wt": f.n_wt,
        "k_ratio": f.k_ratio,
        "n_ratio": f.n_ratio,
        "ceiling_ratio": f.ceiling_ratio,
        "ci_log2_k": list(f.ci_log2_k),
        "ci_log2_n": list(f.ci_log2_n),
        "ci_log2_ceiling": list(f.ci_log2_ceiling),
        "losses": {
            "no_effect": f.loss_no_effect,
            "threshold": f.loss_threshold,
            "gain": f.loss_gain,
            "ceiling": f.loss_ceiling,
            "full": f.loss_full,
        },
        "panel_r2": f.panel_r2,
        "worst_reporter_r2": f.worst_reporter_r2,
        "consistency_ratio": f.consistency_ratio,
        "n_points_total": f.n_points_total,
        "reporters": [
            {
                "name": r.name,
                "floor": r.floor,
                "gain": r.gain,
                "r2_shared": r.r2_shared,
                "r2_independent": r.r2_independent,
            }
            for r in f.reporters
        ],
    }


def _read_reporter_panel(
    path: str,
    *,
    dose_col: str,
    reporter_col: str,
    control_col: str,
    perturbed_col: str,
) -> list[Any]:
    """Read a tidy long CSV/TSV into a list of ``ReporterObservation``.

    Expected columns: a reporter label, a dose, and the reporter's control and
    perturbed responses (one row per reporter × dose). Rows are grouped by reporter and
    dose-sorted.
    """
    import numpy as np
    import pandas as pd

    from nudge.inference.multi_reporter import ReporterObservation

    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep, comment="#")
    for col in (dose_col, reporter_col, control_col, perturbed_col):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not in {list(df.columns)}")
    out: list[Any] = []
    for name, rows in df.groupby(reporter_col):
        rows = rows.sort_values(dose_col)
        out.append(
            ReporterObservation(
                name=str(name),
                dose=rows[dose_col].to_numpy(dtype=float),
                control=rows[control_col].to_numpy(dtype=float),
                perturbed=np.asarray(rows[perturbed_col], dtype=float),
            )
        )
    return out


def multi_reporter_file(
    path: str,
    *,
    dose_col: str = "dose",
    reporter_col: str = "reporter",
    control_col: str = "control",
    perturbed_col: str = "perturbed",
    direction: str = "activate",
    n_boot: int = 200,
    seed: int = 0,
    knob_margin: float = 1.5,
    effect_margin: float = 1.4,
    min_panel_r2: float = 0.5,
) -> dict[str, Any]:
    """Jointly attribute a multi-reporter panel from a tidy CSV/TSV — the CLI/MCP entry.

    Reads a long table (one row per reporter × dose, with control and perturbed response
    columns) of several reporters of ONE latent switch, fits them jointly, and localizes
    the perturbation to a single shared knob — **threshold** / **gain** / **ceiling** —
    or abstains (**no-effect** / **unresolved** / **off-model**). The joint panel
    resolves the mechanism where a single reporter is degenerate (the K⇄v_max
    degeneracy); a panel that cannot be explained by one shared latent abstains
    ``off-model`` (NUDGE-LIM-014) rather than average it into a call.
    """
    from nudge.inference.multi_reporter import attribute_multi_reporter

    reporters = _read_reporter_panel(
        path,
        dose_col=dose_col,
        reporter_col=reporter_col,
        control_col=control_col,
        perturbed_col=perturbed_col,
    )
    res = attribute_multi_reporter(
        reporters,
        direction=direction,
        n_boot=n_boot,
        seed=seed,
        knob_margin=knob_margin,
        effect_margin=effect_margin,
        min_panel_r2=min_panel_r2,
    )
    return multi_reporter_to_dict(res)


# --------------------------------------------------------------------------- #
# hidden-node ABSTENTION — turn a bare off-model verdict into a legible
# differential of candidate causes (nudge.inference.hidden_node,
# NUDGE-METHOD-009). The abstention half ONLY: it NEVER asserts a hidden node
# (NUDGE-LIM-015); it consumes verdicts, never touches the fit.
# --------------------------------------------------------------------------- #
def inadequacy_to_dict(report: Any) -> dict[str, Any]:
    """Serialise an ``InadequacyReport`` + its differential to a JSON-able dict.

    Enriches each candidate cause with the *title* of its documented limitation via the
    read-only :func:`nudge.knowledge.explain` backbone, so a caller (CLI / MCP / Claude)
    can render the legible differential without re-reading the YAML. Purely additive — the
    underlying :class:`~nudge.inference.hidden_node.InadequacyReport` is unchanged.
    """
    from nudge.knowledge import explain as _explain

    def _lim_title(ref: str) -> str:
        if not ref:
            return ""
        info = _explain(ref)
        return str(info.get("title", "")) if info.get("kind") == "limitation" else ""

    causes = [
        {
            "name": c.name,
            "qualitative_rank": c.qualitative_rank,
            "evidence": c.evidence,
            "distinguishing_experiment": c.distinguishing_experiment,
            "limitation_ref": c.limitation_ref,
            "limitation_title": _lim_title(c.limitation_ref),
            "decoy_ref": c.decoy_ref,
        }
        for c in report.ranked_causes()
    ]
    return {
        "is_adequate": report.is_adequate,
        "verdict": report.verdict,
        "reason": report.reason,
        "causes": causes,
        # A loud, machine-readable restatement of the abstention-half-only guarantee.
        "hidden_node_claim": False,
        "honesty_note": (
            "This is a DIFFERENTIAL of candidate causes, not a verdict. NUDGE never "
            "positively asserts a hidden node from an off-model result — the causes are "
            "observationally overlapping (NUDGE-LIM-015)."
        ),
    }


def diagnose_abstention(
    *,
    off_model: bool,
    neomorphic_ratio: float | None = None,
    readout_flag: bool = False,
    perturbation_residual: float | None = None,
    topology_uncertain: bool = False,
    depth_confounded: bool = False,
    neomorphic_ratio_threshold: float = 1.0,
) -> dict[str, Any]:
    """Diagnose *why* a NUDGE attribution is inadequate — the CLI / MCP entry point.

    Consumes the evidence an attribution already produced (the ``off_model`` parsimony
    verdict + optional diagnostic signals) and returns the rank-ordered differential of
    candidate causes (:func:`nudge.inference.hidden_node.diagnose_inadequacy`). When the
    model is adequate it returns ``is_adequate=True`` with no causes. It **never** emits a
    positive hidden-node claim — the strongest it says is that an off-axis residual is
    *consistent with, does not prove* an unmeasured regulator (NUDGE-LIM-015).
    """
    from nudge.inference.hidden_node import diagnose_inadequacy

    report = diagnose_inadequacy(
        off_model=off_model,
        neomorphic_ratio=neomorphic_ratio,
        readout_flag=readout_flag or None,
        perturbation_residual=perturbation_residual,
        topology_uncertain=topology_uncertain or None,
        depth_confounded=depth_confounded or None,
        neomorphic_ratio_threshold=neomorphic_ratio_threshold,
    )
    return inadequacy_to_dict(report)


# --------------------------------------------------------------------------- #
# comparative / differential attribution — the SAME perturbation in two contexts
# (resistant vs sensitive line; donor A vs B; disease vs healthy): isolate whether
# the mechanistic difference is in K (threshold), n (gain), or v_max (ceiling) — a
# call linear differential expression structurally cannot make — or abstain. Reuses
# the shipped LNA multi-operating-point machinery + BIC parsimony (inference.
# differential, NUDGE-METHOD-010). Confound guard: per-context depth pinning + the
# OFF-baseline ceiling/depth guard (NUDGE-LIM-016).
# --------------------------------------------------------------------------- #
def differential_to_dict(res: Any) -> dict[str, Any]:
    """Serialise a ``DifferentialResult`` to a plain JSON-able dict (CLI / MCP)."""
    f = res.fit
    return {
        "call": res.call,
        "reason": res.reason,
        "is_reliable": res.is_reliable,
        "selected_model": f.selected,
        "best_diff_model": f.best_diff,
        "target_edge": f.target_edge,
        "bic": dict(f.bic),
        "n_params": dict(f.n_params),
        "log2_ratio": f.log2_ratio,
        "ci_log2": list(f.ci_log2),
        "n_cells": {"a": f.n_cells_a, "b": f.n_cells_b},
        "depth": {
            "scale_a": f.scale_a,
            "scale_b": f.scale_b,
            "depth_ratio": f.depth_ratio,
        },
        "off_baseline_shift": {
            "a": f.off_shift_a,
            "b": f.off_shift_b,
            "ratio": f.off_shift_ratio,
        },
        "lna_ok": {"a": f.lna_ok_a, "b": f.lna_ok_b},
        "estimates": {
            "a": {m: dict(f.est_a[m]) for m in f.est_a},
            "b": {m: dict(f.est_b[m]) for m in f.est_b},
        },
    }


def _switch_circuit(
    circuit: str = "ras_switch_1node",
    *,
    n: float = 6.0,
    vmax: float = 2.5,
    k: float = 1.0,
    basal: float = 0.2,
) -> Any:
    """Build a named bistable switch motif for differential attribution (shared topology)."""
    from nudge import circuits as _circuits

    factory = getattr(_circuits, circuit, None)
    if factory is None:
        raise ValueError(f"unknown circuit {circuit!r} (see nudge.circuits)")
    return factory(n=n, vmax=vmax, K=k, basal=basal)


def differential_arrays(
    data_a: Any,
    control_a: Any,
    data_b: Any,
    control_b: Any,
    *,
    circuit: str = "ras_switch_1node",
    n: float = 6.0,
    vmax: float = 2.5,
    k: float = 1.0,
    basal: float = 0.2,
    target_edge: int = 0,
    steps: int = 250,
    n_boot: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """Differential attribution from four activity arrays — the programmatic entry point.

    ``data_x`` / ``control_x`` are ``(n_cells, n_species)`` activity-space arrays for each
    context's perturbed cells and its OWN control. Fits the shared switch topology
    (``circuit`` + kinetics) jointly and BIC-selects which single knob — **threshold**
    (K), **gain** (n), or **ceiling** (v_max) — differs between the contexts, or abstains
    (``no-difference`` / ``unresolved``). The confound guard pins depth per context from
    each control and abstains on a ceiling call corrupted by a depth/batch shift
    (NUDGE-LIM-016).
    """
    import numpy as np

    from nudge.inference.differential import Context, attribute_differential

    circ = _switch_circuit(circuit, n=n, vmax=vmax, k=k, basal=basal)
    ctx_a = Context(
        name="A", data=np.asarray(data_a, dtype=float), control=np.asarray(control_a, float)
    )
    ctx_b = Context(
        name="B", data=np.asarray(data_b, dtype=float), control=np.asarray(control_b, float)
    )
    res = attribute_differential(
        ctx_a, ctx_b, circ,
        target_edge=target_edge, steps=steps, n_boot=n_boot, seed=seed,
    )
    return differential_to_dict(res)


def differential_file(
    path: str,
    *,
    circuit: str = "ras_switch_1node",
    n: float = 6.0,
    vmax: float = 2.5,
    k: float = 1.0,
    basal: float = 0.2,
    target_edge: int = 0,
    steps: int = 250,
    n_boot: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """Differential attribution from a ``.npz`` of two contexts — the CLI / MCP entry.

    The ``.npz`` holds four ``(n_cells, n_species)`` **activity-space** arrays:
    ``data_a`` / ``control_a`` (context A's perturbed cells + its own control) and
    ``data_b`` / ``control_b`` (context B). Attributes whether the SAME perturbation
    differs between the two contexts in its switch's **threshold** / **gain** / **ceiling**
    — or abstains. Returns the verdict, the per-model BIC, the winning knob's Δ estimate +
    (optional) bootstrap CI, the per-context depth, and the confound diagnostics.
    """
    import numpy as np

    with np.load(path) as npz:
        missing = [k2 for k2 in ("data_a", "control_a", "data_b", "control_b")
                   if k2 not in npz]
        if missing:
            raise KeyError(f"{path} is missing array(s) {missing}; need data_a/control_a"
                           "/data_b/control_b")
        arrays = {k2: np.asarray(npz[k2], dtype=float) for k2 in
                  ("data_a", "control_a", "data_b", "control_b")}
    return differential_arrays(
        arrays["data_a"], arrays["control_a"], arrays["data_b"], arrays["control_b"],
        circuit=circuit, n=n, vmax=vmax, k=k, basal=basal,
        target_edge=target_edge, steps=steps, n_boot=n_boot, seed=seed,
    )


def _jsonsafe(x: float) -> float | None:
    """NaN / ±inf → None so the verdict serializes cleanly over MCP/JSON."""
    import math

    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None


def differential_robust_arrays(
    data_a: Any,
    control_a: Any,
    data_b: Any,
    control_b: Any,
    *,
    circuit: str = "ras_switch_1node",
    n: float = 6.0,
    vmax: float = 2.5,
    k: float = 1.0,
    basal: float = 0.2,
    k_modes: int = 2,
    steps: int = 150,
    earn_margin: float = 6.0,
    cond_max: float = 100.0,
    check_both: bool = True,
) -> dict[str, Any]:
    """ROBUST differential attribution — the affine-nuisance **Earn-Guard** (opt-in fail-safe).

    Same four-array contract as :func:`differential_arrays`, but instead of the shipped
    per-confound OFF-cluster bands — which have **measured blind spots** across the affine
    confound family (``NUDGE-LIM-016``: a per-context multiplicative scale can land on the
    threshold/gain channel or between the ceiling band's calibrated cuts, and slip) — this
    uses the Earn-Guard (:func:`nudge.inference._proto_nuisance.guard_b_classify`). It re-fits
    each context's apparent knob difference against a **free per-context affine nuisance**
    ``(s, o)`` and returns a positive ``*-diff`` ONLY if the biological knob **earns** its BIC
    parameter over that affine null, in BOTH directions. Because the whole per-condition affine
    confound family lies inside the free-affine null's span, this abstains on it **continuously**
    — one measured statistic, no calibrated bands, no blind gaps (proven **0/24 confident-wrong**
    on the exact red-team P1/P4/P5 repros; ``scripts/vv/FINDINGS.md`` §EG). Slower than the banded
    path (it fits a reference + two augmented models per direction); use it when robustness to a
    perturbed-side technical confound matters more than latency.
    """
    import numpy as np

    from nudge.inference._proto_nuisance import guard_b_classify
    from nudge.inference.differential import Context

    circ = _switch_circuit(circuit, n=n, vmax=vmax, k=k, basal=basal)
    ctx_a = Context(
        name="A", data=np.asarray(data_a, dtype=float), control=np.asarray(control_a, float)
    )
    ctx_b = Context(
        name="B", data=np.asarray(data_b, dtype=float), control=np.asarray(control_b, float)
    )
    res = guard_b_classify(
        ctx_a, ctx_b, circ, k_modes=k_modes, steps=steps,
        earn_margin=earn_margin, cond_max=cond_max, check_both=check_both,
    )
    return {
        "call": res.call,
        "reason": res.reason,
        "knob": res.knob,
        "is_reliable": res.is_reliable,
        "earn_bic": _jsonsafe(res.earn_bic),
        "cond_number": _jsonsafe(res.cond_number),
        "knob_identifiable": res.knob_identifiable,
        "s_hat": _jsonsafe(res.s_hat),
        "o_hat": _jsonsafe(res.o_hat),
        "guard": "earn-guard (affine-nuisance robust; NUDGE-LIM-016 / "
        "design/PERTURBED_CONFOUND_STRATEGY.md)",
    }


def differential_robust_file(
    path: str,
    *,
    circuit: str = "ras_switch_1node",
    n: float = 6.0,
    vmax: float = 2.5,
    k: float = 1.0,
    basal: float = 0.2,
    k_modes: int = 2,
    steps: int = 150,
    earn_margin: float = 6.0,
    cond_max: float = 100.0,
    check_both: bool = True,
) -> dict[str, Any]:
    """Robust (Earn-Guard) differential attribution from a ``.npz`` — the CLI / MCP entry.

    Same ``.npz`` contract as :func:`differential_file` (``data_a`` / ``control_a`` /
    ``data_b`` / ``control_b``). Uses the affine-nuisance Earn-Guard instead of the banded
    default — abstains continuously over the per-condition affine confound family.
    """
    import numpy as np

    with np.load(path) as npz:
        missing = [k2 for k2 in ("data_a", "control_a", "data_b", "control_b")
                   if k2 not in npz]
        if missing:
            raise KeyError(f"{path} is missing array(s) {missing}; need data_a/control_a"
                           "/data_b/control_b")
        arrays = {k2: np.asarray(npz[k2], dtype=float) for k2 in
                  ("data_a", "control_a", "data_b", "control_b")}
    return differential_robust_arrays(
        arrays["data_a"], arrays["control_a"], arrays["data_b"], arrays["control_b"],
        circuit=circuit, n=n, vmax=vmax, k=k, basal=basal,
        k_modes=k_modes, steps=steps, earn_margin=earn_margin, cond_max=cond_max,
        check_both=check_both,
    )


# --------------------------------------------------------------------------- #
# constitutive-reporter calibration control — the NUDGE-LIM-006 mitigation
# (nudge.inference.constitutive, NUDGE-METHOD-011). A constitutive control drives
# the reporter at KNOWN activity doses, bypassing the circuit, so it anchors the
# readout (READOUT params only — no circuit leak); a profile over circuit n then
# tests whether the observed ultrasensitivity is BIOLOGICAL (reject "no switch") or
# lives in the measurement. Fail-safe: never a bare mechanism (NUDGE-LIM-018).
# --------------------------------------------------------------------------- #
def constitutive_to_dict(res: Any) -> dict[str, Any]:
    """Serialise a ``ConstitutiveResult`` to a plain JSON-able dict (CLI / MCP)."""
    c = res.calibration
    return {
        "call": res.call,
        "reason": res.reason,
        "confident_wrong": res.is_confident_wrong,  # bare-knob only (structurally False)
        # The falsifiable POSITIVE claim, bounded by the shared-capture precondition
        # (NUDGE-LIM-019). Surfaced so a caller never reads biological-switch as an
        # unconditional certainty.
        "asserts_biological_switch": res.asserts_biological_switch,
        "calibration": {
            "reporter_hill_h": c.h,
            "ci_h": list(c.ci_h),
            "km": c.km,
            "vmax": c.vmax,
            "base": c.base,
            "r2": c.r2,
            "is_nonlinear": c.is_nonlinear,
        },
        "n_grid": list(res.n_grid),
        "loss_no_control": list(res.loss_no_control),
        "loss_with_control": list(res.loss_with_control),
        "span_no_control": res.span_no_control,
        "span_with_control": res.span_with_control,
        "n1_rejection": res.n1_rejection,
        "argmin_n_with_control": res.argmin_n_with_control,
        "floor_mean": res.floor_mean,
        "floor_std": res.floor_std,
    }


def constitutive_arrays(
    population: Any,
    control_activity: Any,
    control_response: Any,
    *,
    circuit_n: float = 3.0,
    k: float = 1.0,
    vmax: float = 1.0,
    basal: float = 0.05,
    km: float = 0.5,
    h: float = 6.0,
    readout_vmax: float = 20.0,
    readout_base: float = 0.1,
    dispersion: float = 0.1,
    mu_log: float = 0.0,
    sd_log: float = 0.6,
    steps: int = 600,
    restarts: int = 3,
    n_model_cells: int = 400,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the constitutive-control analysis from arrays — the programmatic entry point.

    ``population`` is the observed circuit-population counts (1-D; the reporter read of the
    circuit). ``control_activity`` / ``control_response`` are the constitutive control's KNOWN
    driven activity doses + measured reporter output (paired 1-D, ≥4 distinct doses). The
    remaining kinetics supply the KNOWN floors + count model + latent-input assumption
    (``basal`` / ``readout_base`` / ``dispersion`` / ``mu_log`` / ``sd_log``); the circuit
    ``n`` is what NUDGE profiles (``circuit_n`` only seeds the ground-truth container). Returns
    the verdict (``biological-switch`` / ``unresolved`` / ``no-confound``) with both
    ``n``-profiles — never a bare mechanism (NUDGE-LIM-018).
    """
    import numpy as np

    from nudge.inference.constitutive import (
        ConstitutiveControl,
        ReadoutCircuitParams,
        profile_circuit_n,
    )

    control = ConstitutiveControl(
        activity=np.asarray(control_activity, dtype=float),
        response=np.asarray(control_response, dtype=float),
    )
    params = ReadoutCircuitParams(
        k=k, n=circuit_n, vmax=vmax, basal=basal, km=km, h=h,
        readout_vmax=readout_vmax, readout_base=readout_base,
    )
    res = profile_circuit_n(
        np.asarray(population, dtype=float), control, params,
        dispersion=dispersion, mu_log=mu_log, sd_log=sd_log,
        steps=steps, restarts=restarts, n_model_cells=n_model_cells, seed=seed,
    )
    return constitutive_to_dict(res)


def constitutive_file(
    path: str,
    *,
    circuit_n: float = 3.0,
    k: float = 1.0,
    vmax: float = 1.0,
    basal: float = 0.05,
    km: float = 0.5,
    h: float = 6.0,
    readout_vmax: float = 20.0,
    readout_base: float = 0.1,
    dispersion: float = 0.1,
    steps: int = 600,
    restarts: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the constitutive-control analysis from a ``.npz`` — the CLI / MCP entry point.

    The ``.npz`` holds ``population`` (1-D circuit-population counts), ``control_activity`` and
    ``control_response`` (the constitutive calibration's KNOWN doses + measured reporter). See
    :func:`constitutive_arrays` for the kinetics / count-model knobs. Returns the fail-safe
    verdict (``biological-switch`` = reject the readout-only explanation / ``unresolved`` =
    honest abstention / ``no-confound``) — never a bare threshold/gain/ceiling.
    """
    import numpy as np

    with np.load(path) as npz:
        missing = [key for key in ("population", "control_activity", "control_response")
                   if key not in npz]
        if missing:
            raise KeyError(
                f"{path} is missing array(s) {missing}; need population/control_activity"
                "/control_response"
            )
        pop = np.asarray(npz["population"], dtype=float)
        act = np.asarray(npz["control_activity"], dtype=float)
        resp = np.asarray(npz["control_response"], dtype=float)
    return constitutive_arrays(
        pop, act, resp, circuit_n=circuit_n, k=k, vmax=vmax, basal=basal, km=km, h=h,
        readout_vmax=readout_vmax, readout_base=readout_base, dispersion=dispersion,
        steps=steps, restarts=restarts, seed=seed,
    )


def constitutive_demo(
    *,
    circuit_n: float = 3.0,
    readout_h: float = 6.0,
    n_cells: int = 600,
    n_ctrl_doses: int = 10,
    n_ctrl_reps: int = 200,
    steps: int = 600,
    restarts: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """Synthesize a matched population + constitutive control and run the analysis (no data).

    The zero-setup demo of the NUDGE-LIM-006 mitigation: a nonlinear (``readout_h``) reporter
    over a circuit of true Hill ``circuit_n``. Set ``circuit_n=1`` for the LIM-006
    false-positive HAZARD (a linear circuit whose apparent ultrasensitivity lives in the
    reporter → NUDGE abstains) or ``circuit_n>1`` for a genuine biological switch (→ NUDGE
    rejects "no switch"). Returns the verdict + both ``n``-profiles.
    """
    from nudge.inference.constitutive import (
        ReadoutCircuitParams,
        generate_constitutive_dataset,
        profile_circuit_n,
    )

    params = ReadoutCircuitParams(
        k=1.0, n=circuit_n, vmax=1.0, basal=0.05, km=0.5, h=readout_h,
        readout_vmax=20.0, readout_base=0.1,
    )
    pop, control, _ = generate_constitutive_dataset(
        params, n_cells=n_cells, n_ctrl_doses=n_ctrl_doses, n_ctrl_reps=n_ctrl_reps, seed=seed
    )
    res = profile_circuit_n(
        pop, control, params, steps=steps, restarts=restarts, seed=seed
    )
    out = constitutive_to_dict(res)
    out["ground_truth"] = {"circuit_n": circuit_n, "reporter_h": readout_h}
    return out


def lotka_demo(
    *,
    mechanism: str = "susceptibility",
    dense_transient: bool = True,
    n_species: int = 3,
    n_replicates: int = 60,
    steps: int = 250,
    n_sim: int = 30,
    seed: int = 0,
) -> dict[str, Any]:
    """Synthesize a gLV community pair + attribute which knob moved (no data file).

    The zero-setup demo of the temporal / Lotka–Volterra capability
    (``NUDGE-METHOD-012``). ``mechanism`` ∈ ``{"growth", "interaction",
    "susceptibility", "none"}`` sets the KNOWN single-knob perturbation; the antibiotic
    susceptibility (``susceptibility``) axis is the identifiable positive, while a
    near-equilibrium growth change (``dense_transient=False``) is the degenerate α⇄βᵢᵢ
    case NUDGE must abstain on (``NUDGE-LIM-020``). Returns the verdict + per-model BIC +
    the measured degeneracy.
    """
    from nudge.inference.lotka_volterra import attribute_glv, simulate_glv_perturbseq

    ds = simulate_glv_perturbseq(
        n_species=n_species, n_replicates=n_replicates, mechanism=mechanism,
        dense_transient=dense_transient, seed=seed,
    )
    res = attribute_glv(ds, steps=steps, n_sim=n_sim, seed=seed)
    f = res.fit
    return {
        "call": res.call,
        "reason": res.reason,
        "is_reliable": res.is_reliable,
        "ground_truth": dict(ds.ground_truth),
        "selected_knob": f.selected,
        "bic": dict(f.bic),
        "delta": dict(f.delta),
        "identifiability": {
            "cond_number": f.cond_number,
            "abs_corr_alpha_beta": f.corr_alpha_beta,
            "degenerate": f.degenerate,
            "reason": f.identifiability_reason,
        },
        "n_replicates": f.n_replicates,
        "n_timepoints": f.n_timepoints,
    }
