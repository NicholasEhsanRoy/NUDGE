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


#: Cap on the inline provenance ``data`` text (sidecar JSON). Animation sidecars carry a
#: whole frame sequence, so cap the inlined copy; the full sidecar is always on disk.
_DATA_TEXT_CAP = 200_000


def _read_text_capped(path: str | None, cap: int) -> tuple[str | None, bool]:
    """Read a text file, capped. Returns ``(text_or_None, truncated)``."""
    if not path:
        return None, False
    try:
        from pathlib import Path

        text = Path(path).read_text(encoding="utf-8")
    except OSError:  # pragma: no cover - defensive
        return None, False
    if len(text) > cap:
        return text[:cap] + f"\n… (truncated at {cap:,} chars; full sidecar on disk)", True
    return text, False


def _resolve_transport(transport: str | None) -> str:
    """Resolve the figure transport: explicit override, else the ``NUDGE_ENV`` toggle.

    ``NUDGE_ENV=cloud`` → ``inline`` (the Claude Science reality: the connector can only
    deliver a figure as inline base64); anything else → ``path`` (write a file the local
    client can read). Default (env unset) is ``path`` — safe for a local host / the CLI.
    """
    if transport in ("inline", "path"):
        return transport
    import os

    return "inline" if os.environ.get("NUDGE_ENV", "").strip().lower() == "cloud" else "path"


def _artifact_dir(out: str | None) -> str:
    """The directory to write a figure into for PATH transport.

    ``NUDGE_ARTIFACT_DIR`` (if set) wins — it lets a host pin a client-visible directory;
    otherwise the caller's ``out`` directory; otherwise the system temp dir.
    """
    import os
    import tempfile

    env_dir = os.environ.get("NUDGE_ARTIFACT_DIR", "").strip()
    if env_dir:
        return env_dir
    if out:
        d = os.path.dirname(out)
        if d:
            return d
    return tempfile.gettempdir()


def render_result(
    kind: str,
    result_or_dict: Any,
    *,
    out: str | None,
    emit_code: bool = True,
    theme: str = "auto",
    self_contained: bool = False,
    animate: bool = False,
    inline_png: bool = False,  # deprecated; transport now governs inlining (kept for compat)
    cli_call: str | None = None,
    transport: str | None = None,
    **ctx: Any,
) -> dict[str, Any]:
    """Render a NUDGE result to a figure — the one place CLI + MCP share the figure path.

    Lazy-imports the opt-in :mod:`nudge.viz` (raising the friendly ``[viz]``-extra install
    message if absent), dispatches on ``kind``, and returns a transport-aware dict. Two
    transports (``NUDGE_ENV=cloud`` → ``inline``, else ``path``; override with
    ``transport=``):

    - **inline** (Claude Science): the image travels as size-disciplined **base64**
      (``image_base64`` + ``mime_type``) — GIFs downscaled / frame-limited / never-inflated
      and capped, falling back to a static final-frame preview or a ``too large`` note.
      Provenance (``code`` = the regenerating ``fig.py``; ``data`` = the sidecar, capped)
      always rides along inline.
    - **path** (local hosts / the CLI): the figure is written to ``NUDGE_ARTIFACT_DIR``
      (fallback: the caller's dir, then the system temp dir) and its ``image_path`` /
      ``code_path`` / ``data_path`` are returned.

    ``ctx`` carries any renderer inputs the serialized dict lacks (e.g. the raw ``dose`` /
    ``response`` points). The abstention overlay is applied by :mod:`nudge.viz` off the
    result's own verdict; this seam never re-fits.
    """
    import os

    import nudge.viz as viz

    resolved = _resolve_transport(transport)
    ext = ".gif" if animate else ".png"

    if resolved == "inline":
        import tempfile

        write_dir = tempfile.mkdtemp(prefix="nudge_viz_")  # staging only (base64'd, not returned)
        write_path = os.path.join(write_dir, f"{kind}{ext}")
    else:
        write_dir = _artifact_dir(out)
        os.makedirs(write_dir, exist_ok=True)
        basename = os.path.basename(out) if out else f"{kind}{ext}"
        write_path = os.path.join(write_dir, basename)

    fr = viz.render(
        result_or_dict,
        write_path,
        kind=kind,
        emit_code=emit_code,
        theme=theme,
        self_contained=self_contained,
        animate=animate,
        inline_png=False,
        cli_call=cli_call,
        **ctx,
    )

    mime_type = "image/gif" if (fr.path or write_path).endswith(".gif") else "image/png"
    code_text, _ = _read_text_capped(fr.code_path, _DATA_TEXT_CAP)
    data_text, data_trunc = _read_text_capped(fr.data_path, _DATA_TEXT_CAP)

    common: dict[str, Any] = {
        "transport": resolved,
        "kind": fr.kind,
        "caption": fr.caption,
        "abstained": fr.abstained,
        "mime_type": mime_type,
        "code": code_text,
        "data": data_text,
        "data_truncated": data_trunc,
    }

    if resolved == "inline":
        from nudge.viz.inline import prepare_inline_image

        image_bytes = b""
        if fr.path:
            with open(fr.path, "rb") as fh:
                image_bytes = fh.read()
        common.update(prepare_inline_image(image_bytes, mime_type))
        common["image_path"] = None  # the staging file is invisible to the client
        common["png_path"] = None
        return common

    # PATH transport: return the written paths (png_path kept as a back-compat alias).
    common.update({
        "image_path": fr.path,
        "png_path": fr.path,
        "code_path": fr.code_path,
        "data_path": fr.data_path,
        "image_base64": None,
        "image_base64_omitted_reason": f"path transport; read image_path ({fr.path})",
    })
    return common


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


def _attribution_demo_op(n_wt: int, n_target: int, *, basal: float, seed: int) -> Any:
    """One synthetic operating point (WT + target cells) for :func:`attribution_demo`.

    Mirrors ``tests/inference/test_pipeline._op_adata``: IEG counts track a bimodal
    activation over the 1-node RAS switch, with a depth-only non-marker gene.
    """
    import anndata as ad
    import jax
    import numpy as np
    import pandas as pd

    from nudge.circuits import ras_switch_1node
    from nudge.inference.lyapunov import sample_lna_mixture

    genes = ["IL2", "CD69", "EGR1", "ACTB"]
    kw, kt = jax.random.split(jax.random.PRNGKey(seed))
    wt = sample_lna_mixture(ras_switch_1node(basal=basal), n_wt, kw, scale=1.0)
    tg = sample_lna_mixture(ras_switch_1node(basal=basal, n=3.0), n_target, kt, scale=1.0)
    act = np.clip(np.concatenate([wt, tg])[:, 0], 0, None)
    rng = np.random.default_rng(seed)
    lib = rng.integers(1500, 3000, size=act.size).astype(float)
    x = np.zeros((act.size, len(genes)), dtype=np.int32)
    for g in range(3):
        x[:, g] = rng.poisson(act * lib / 80.0)
    x[:, 3] = rng.poisson(lib / 100.0)
    cond = ["WT"] * n_wt + ["SOS1"] * n_target
    obs = pd.DataFrame(
        {"condition": pd.Categorical(cond), "total_counts": lib},
        index=pd.Index([f"c{i}" for i in range(act.size)]),
    )
    return ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=pd.Index(genes)))


def attribution_demo(*, steps: int = 150, min_cells: int = 200, seed: int = 0) -> dict[str, Any]:
    """Synthesize a multi-operating-point screen + run the capstone attribution (no data file).

    The zero-setup demo of the core attribution pipeline
    (:func:`nudge.inference.pipeline.attribute_across_operating_points`): two usable
    operating points (which the joint breaker can use to resolve a single knob) plus a
    third with too few target cells (recorded as a **skip**, not dropped). Returns the
    ``report_to_dict`` form — the exact honest report the CLI / MCP already surface — so
    the ``attribution`` renderer draws the per-op verdict chips + the joint restricted-NLL
    profile from genuine (synthetic) numbers, not a fabricated layout.
    """
    from nudge.circuits import ras_switch_1node
    from nudge.inference.pipeline import attribute_across_operating_points

    ops = {
        "Stim8hr": _attribution_demo_op(1500, 1500, basal=0.05, seed=seed),
        "Stim48hr": _attribution_demo_op(1500, 1500, basal=0.15, seed=seed + 1),
        "Rest": _attribution_demo_op(1500, 40, basal=0.05, seed=seed + 2),  # too few → skip
    }
    report = attribute_across_operating_points(
        ops, ras_switch_1node(), {"Activation": ("IL2", "CD69", "EGR1")}, "SOS1",
        steps=steps, min_cells=min_cells, seed=seed,
    )
    return report_to_dict(report)


def identifiability_demo(*, case: str = "sloppy", sigma: float = 0.01) -> dict[str, Any]:
    """Diagnose a canonical model's identifiability + return the figure-data (no data file).

    The zero-setup demo of the sloppiness diagnostic
    (:func:`nudge.inference.sloppiness.analyze_model`), on the canonical models that
    separate the three verdicts a condition-number-only test conflates:

    - ``"sloppy"`` — a sum-of-exponentials: a Fisher spectrum spanning many decades
      (individual parameters loose) yet tight predictions ⇒ **sloppy-but-predictive**
      (NUDGE must NOT abstain);
    - ``"unidentifiable"`` — ``A·e^{-(k1+k2)t}``: an exact structural null (only ``k1+k2``
      enters) ⇒ **unidentifiable** (NUDGE abstains, naming the null direction);
    - ``"well-constrained"`` — a linear model: a narrow spectrum ⇒ **well-constrained**.

    Returns a figure-data dict the ``identifiability`` renderer consumes (the FIM spectrum
    + the naive-vs-measured verdict), with every number MEASURED from the sensitivity
    matrix, never asserted.
    """
    import jax
    import numpy as np

    from nudge.inference.sloppiness import (
        analyze_model,
        redundant_exponential_predict,
        sum_of_exponentials_predict,
        well_conditioned_predict,
    )

    t = np.linspace(0.05, 6.0, 60)
    if case == "unidentifiable":
        predict = redundant_exponential_predict(amp=1.0, k1=0.7, k2=0.9, t=t)
        label = "A·e^{-(k1+k2)t}"
    elif case in ("well-constrained", "well_constrained"):
        predict, label = well_conditioned_predict(slope=2.0, offset=1.0, t=t), "linear model"
    else:
        predict = sum_of_exponentials_predict(
            rates=[0.5, 1.3, 2.5, 4.5], amps=[1.0, 1.0, 1.0, 1.0], t=t
        )
        label = "sum-of-exponentials"
    # The sloppiness diagnostic needs float64 to resolve the smallest FIM eigenvalues
    # (float32 truncates the sloppy end); enable it locally and restore.
    _prev_x64 = bool(getattr(jax.config, "jax_enable_x64", False))
    jax.config.update("jax_enable_x64", True)
    try:
        report = analyze_model(predict, sigma=sigma)
    finally:
        jax.config.update("jax_enable_x64", _prev_x64)
    nulls = report.null_directions
    return {
        "kind": "identifiability",
        "label": label,
        "call": report.label,
        "verdict": report.label,
        "reason": report.reason,
        "param_names": list(report.param_names),
        "fim_eigenvalues": [float(v) for v in np.asarray(report.fim_eigenvalues).ravel()],
        "cond_number": _jsonsafe(report.cond_number),
        "span_decades": _jsonsafe(report.spectral_span_decades),
        "smallest_eigenvalue": float(report.smallest_eigenvalue),
        "largest_eigenvalue": float(report.largest_eigenvalue),
        "n_sloppy_dims": int(report.n_sloppy_dims),
        "n_null_dims": int(report.n_null_dims),
        "is_sloppy": bool(report.is_sloppy),
        "predictive": bool(report.predictive),
        "relative_prediction_std": _jsonsafe(report.relative_prediction_std),
        "pred_rel_tol": 0.05,
        "naive_verdict": report.naive_verdict,
        "naive_is_wrong": bool(report.naive_is_wrong),
        "sloppy_decade_threshold": 3.0,
        "null_hint": (nulls[0].hint if nulls else ""),
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
            transport="path",  # the CLI --fig flag writes a local file
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


def _augment_flip_design(
    out: dict[str, Any], circuit: Any, base: Any, start: str, to: str
) -> None:
    """Make a circuit-mode flip result RELIABLE + unambiguous (in place).

    Two failure modes an inversion demo surfaced (FINDINGS §DES): (1) ``predicted_state`` is in
    **readout** space (``Λ = base + scale·activity``, e.g. ``Readout.identity`` = ``0.2 + 5·act``),
    easy to misread as raw activity exceeding a physical ceiling; (2) the ranked ``deltas``
    **overshoot** — ``design()`` chases the ORIGINAL opposite fixed point, but the intervention
    MOVES the fixed points, so it lands deep in the target basin rather than at the minimal fold.
    This adds the validated **activity-space** fixed point and the **minimal fold-crossing** factor
    so both are explicit.
    """
    import jax.numpy as jnp
    import numpy as np

    from nudge.design.invert import _apply_delta, _base_value, _resolve_x0, _stable_states

    out["predicted_state_space"] = (
        "readout Λ = base + scale·activity (Readout.identity = 0.2 + 5·activity) — NOT raw activity"
    )
    deltas = out.get("deltas") or []
    states = _stable_states(circuit)
    if len(states) < 2 or not deltas:
        return
    start_state = states[-1] if start == "high" else states[0]
    target_state = states[0] if to == "low" else states[-1]
    x0 = _resolve_x0(circuit, start)
    moved = [((d["param"]["scope"], d["param"]["index"], d["param"]["name"]), d["factor"])
             for d in deltas]
    free_list = [fp for fp, _ in moved]
    vals = jnp.asarray([_base_value(base, fp) * fac for fp, fac in moved], jnp.float32)
    params = _apply_delta(base, free_list, vals)
    act = np.asarray(circuit.steady_state(params, x0, n_steps=2000), dtype=float)
    prod = np.asarray(circuit.production(jnp.asarray(act, jnp.float32), params), dtype=float)
    decay = np.asarray(base["species"]["decay"], dtype=float)
    out["predicted_activity"] = [round(float(v), 4) for v in act]
    out["predicted_is_fixed_point"] = bool(np.max(np.abs(prod - decay * act)) < 1e-2)

    # The MINIMAL fold-crossing factor for a single-knob flip (bisection along the knob).
    if len(moved) == 1 and out.get("safety", {}).get("crosses_fold"):
        fp, design_factor = moved[0]
        base_val = _base_value(base, fp)

        def flips(f: float) -> bool:
            p = _apply_delta(base, [fp], jnp.asarray([base_val * f], jnp.float32))
            ss = np.asarray(circuit.steady_state(p, x0, n_steps=1500), dtype=float)
            return bool(np.linalg.norm(ss - target_state) < np.linalg.norm(ss - start_state))

        if flips(design_factor):
            lo, hi = design_factor, 1.0  # lo flips; hi (no change) does not
            for _ in range(44):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if flips(mid) else (lo, mid)
            out["minimal_flip"] = {
                "param": {"scope": fp[0], "index": fp[1], "name": fp[2]},
                "factor": round(lo, 4),
                "percent_change": round((lo - 1.0) * 100.0, 1),
                "note": "the MINIMAL knob change that crosses the fold (destabilises the current "
                "state so it collapses to the other basin). The ranked `deltas` OVERSHOOT this — "
                "they land robustly deep in the target basin; for a 'minimum intervention' answer "
                "use `minimal_flip`.",
            }


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
    if out.get("kind") == "intervention" and out.get("mode") == "circuit":
        _augment_flip_design(out, circuit, circuit.base_params(), start, to)
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
        "off_cluster_scale": {"a": f.off_scale_a, "b": f.off_scale_b},
        "nuisance_earn": {
            # gate 4d (NUDGE-LIM-016 P5): the profiled ΔBIC by which the winning knob
            # out-explains a FREE per-condition affine (s, o) on the perturbed context (min
            # over both directions). nan = not computed (only candidate positives pay). < the
            # earn margin ⇒ the apparent difference is absorbable by a technical affine ⇒ abstain.
            "earn": f.earn,
            "side": f.earn_side,
            "s_hat": f.nuisance_s,
            "o_hat": f.nuisance_o,
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
    steps: int = 250,  # >=~180 required: at ~150 the affine null under-fits and a multiplicative
    # confound can spuriously "earn" a knob at some noise seeds (a MEASURED false positive that
    # vanishes once the null is optimized; FINDINGS §EG). 250 matches the banded `differential`.
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
    — one measured statistic, no calibrated bands (proven **0/24 confident-wrong** at adequate
    optimizer steps on the red-team P1/P4/P5 repros; ``scripts/vv/FINDINGS.md`` §EG). **Numerical
    caveat (MEASURED):** the abstention relies on the affine null being *optimized* — at too few
    ``steps`` (≲180) the null under-fits and a multiplicative confound can spuriously "earn" a knob
    at some noise seeds; the default ``steps=250`` clears this (a false positive at 150 vanishes at
    ≥180). Slower than the banded path (it fits a reference + two augmented models per direction);
    use it when robustness to a perturbed-side technical confound matters more than latency.
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
    steps: int = 250,  # >=~180 required: at ~150 the affine null under-fits and a multiplicative
    # confound can spuriously "earn" a knob at some noise seeds (a MEASURED false positive that
    # vanishes once the null is optimized; FINDINGS §EG). 250 matches the banded `differential`.
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


def temporal_animation_demo(
    *, mechanism: str = "susceptibility", n_species: int = 3, n_replicates: int = 40,
    steps: int = 250, n_sim: int = 30, seed: int = 0,
) -> dict[str, Any]:
    """Temporal / gLV demo enriched for the ANIMATION: the community integrating under the
    antibiotic pulse, perturbed vs reference DIVERGING (``NUDGE-METHOD-012``).

    Simulates a reference vs perturbed gLV community, returns the per-timepoint mean
    trajectories + the pulse window + the honest attribution verdict, and the animator
    (viz/temporal.py) sweeps a time cursor so the perturbed community visibly diverges from
    the reference as the drug pulse hits (susceptibility → the identifiable positive). Only
    READS the simulated trajectories + the fit's verdict; near-equilibrium growth is the
    degenerate α⇄βᵢᵢ case NUDGE abstains on (``NUDGE-LIM-020``).
    """
    import numpy as np

    from nudge.inference.lotka_volterra import attribute_glv, simulate_glv_perturbseq

    ds = simulate_glv_perturbseq(
        n_species=n_species, n_replicates=n_replicates, mechanism=mechanism,
        dense_transient=True, seed=seed,
    )
    res = attribute_glv(ds, steps=steps, n_sim=n_sim, seed=seed)
    ref = np.asarray(ds.reference, dtype=float).mean(axis=0)   # (T, S)
    pert = np.asarray(ds.perturbed, dtype=float).mean(axis=0)  # (T, S)
    t = np.asarray(ds.t_obs, dtype=float)
    gt = dict(ds.ground_truth)
    pulse = gt.get("pulse_window", (float("nan"), float("nan")))
    target = int(gt.get("target", 0))

    return {
        "kind": "temporal",
        "label": "gLV community under an antibiotic pulse",
        "call": res.call,
        "reason": res.reason,
        "selected_knob": res.fit.selected,
        "identifiability": {
            "cond_number": _jsonsafe(res.fit.cond_number),
            "degenerate": bool(res.fit.degenerate),
        },
        "animation": {
            "t": [float(v) for v in t],
            "reference": [[float(v) for v in row] for row in ref],
            "perturbed": [[float(v) for v in row] for row in pert],
            "pulse_window": [float(pulse[0]), float(pulse[1])],
            "target": target,
            "species_labels": [f"taxon {i}" + (" (target)" if i == target else "")
                               for i in range(ref.shape[1])],
        },
    }


def lotka_file(
    path: str, *, target: int | None = None, steps: int = 300, n_sim: int = 30, seed: int = 0
) -> dict[str, Any]:
    """Fit + classify a gLV perturbation from a ``.npz`` of observable trajectories (CLI/MCP).

    The ``.npz`` holds ``reference`` / ``perturbed`` ``(R, T, S)`` replicate ensembles + the
    sampling arrays (``t_obs`` / ``u_grid`` / ``obs_idx`` / ``dt``) — NO kinetics/ground truth.
    NUDGE re-fits the baseline internally, attributes which knob (**growth α / interaction β /
    susceptibility ε**) the perturbation moved, and — critically for a "just fit it and give me the
    parameters" request — reports the **identifiability** of the α⇄βᵢᵢ pair (Laplace condition
    number, |corr|, whether it is DEGENERATE) plus the null-space **degeneracy_direction** + a
    plain-language hint. Fail-safe: on a sloppy / near-equilibrium community it returns
    ``unresolved`` with the measured degeneracy rather than a confident (unidentifiable) parameter
    estimate (``NUDGE-LIM-020``).
    """
    import numpy as np

    from nudge.inference.lotka_volterra import GLVDataset, _default_baseline, attribute_glv

    with np.load(path) as z:
        ref = np.asarray(z["reference"], dtype=float)
        pert = np.asarray(z["perturbed"], dtype=float)
        ds = GLVDataset(
            reference=ref, perturbed=pert, t_obs=np.asarray(z["t_obs"], dtype=float),
            u_grid=np.asarray(z["u_grid"], dtype=float), obs_idx=np.asarray(z["obs_idx"]),
            dt=float(z["dt"]),
            baseline=_default_baseline(ref.shape[2], np.random.default_rng(seed)),
            ground_truth={},
        )
    res = attribute_glv(ds, target=target, steps=steps, n_sim=n_sim, seed=seed)
    f = res.fit
    dd = res.degeneracy_direction
    return {
        "call": res.call, "reason": res.reason, "is_reliable": res.is_reliable,
        "status": getattr(res, "status", None),
        "selected_knob": f.selected,
        "bic": {k: _jsonsafe(v) for k, v in dict(f.bic).items()},
        "fitted_delta": {k: _jsonsafe(v) for k, v in dict(f.delta).items()},
        "identifiability": {
            "cond_number": _jsonsafe(f.cond_number),
            "abs_corr_alpha_beta": _jsonsafe(f.corr_alpha_beta),
            "degenerate": bool(f.degenerate), "reason": f.identifiability_reason,
        },
        "degeneracy_direction": None if dd is None else [float(x) for x in np.asarray(dd)],
        "human_readable_hint": res.human_readable_hint,
    }


def oed_demo(
    *,
    model: str = "logistic",
    objective: str = "crlb",
    n_obs: int = 8,
    steps: int = 400,
    learning_rate: float = 0.2,
    seed: int = 0,
) -> dict[str, Any]:
    """Gradient-optimize an experimental design + report the MEASURED identifiability gain.

    The zero-setup demo of the optimal-experimental-design capability (``NUDGE-METHOD-014``,
    the differentiability moat). Starting from a naive **near-equilibrium** measurement
    schedule — where the growth ⇄ self-limitation (α⇄βᵢᵢ) pair is degenerate
    (``Kᵢ=−αᵢ/βᵢᵢ``) — it gradient-ascends the measurement times to the optimal design and
    reports the factor by which the growth parameter's Cramér–Rao bound (and the FIM's
    smallest eigenvalue) improves. ``model`` ∈ ``{"logistic", "glv"}``; ``objective`` ∈
    ``{"d_opt", "a_opt", "e_opt", "crlb"}``. Everything returned is *measured* at the nominal
    θ₀ (local OED; ``NUDGE-LIM-024``), never asserted.
    """
    import numpy as np

    from nudge.inference.oed import (
        make_glv_design_problem,
        make_logistic_design_problem,
        optimize_design,
    )

    if model == "logistic":
        prob = make_logistic_design_problem()
        target = "log_alpha"
    elif model == "glv":
        prob = make_glv_design_problem(seed=seed)
        target = "log_alpha_t"
    else:
        raise ValueError(f"unknown model {model!r}; expected 'logistic' or 'glv'")

    lo, hi = prob.phi_bounds
    naive = np.linspace(0.6 * hi, hi, n_obs)  # the "measure at steady state" default
    res = optimize_design(
        prob, naive, objective=objective, target=target, steps=steps,
        learning_rate=learning_rate, seed=seed,
    )
    return {
        "model": model,
        "objective": objective,
        "target_parameter": res.target_name,
        "phi_init": [float(x) for x in np.sort(res.phi_init)],
        "phi_opt": [float(x) for x in np.sort(res.phi_opt)],
        "criterion_init": _jsonsafe(res.criterion_init),
        "criterion_opt": _jsonsafe(res.criterion_opt),
        "target_crlb_init": _jsonsafe(res.target_crlb_init),
        "target_crlb_opt": _jsonsafe(res.target_crlb_opt),
        "crlb_improvement": _jsonsafe(res.crlb_improvement),
        "min_eig_init": _jsonsafe(res.min_eig_init),
        "min_eig_opt": _jsonsafe(res.min_eig_opt),
        "min_eig_improvement": _jsonsafe(res.min_eig_improvement),
        "caveat": (
            "local OED: the optimal design and the reported gains are MEASURED at the "
            "nominal parameter θ₀, not extrapolated to far-from-θ₀ truths (NUDGE-LIM-024)."
        ),
    }


def oed_animation_demo(
    *, objective: str = "crlb", n_obs: int = 8, steps: int = 300, n_frames: int = 24,
    learning_rate: float = 0.2, seed: int = 0,
) -> dict[str, Any]:
    """OED demo enriched for the ANIMATION: the design φ trajectory + the (α,β) 95%
    confidence-ellipse collapse over the gradient steps.

    Runs the logistic-growth OED (``optimize_design(..., capture_phi=True)``) and, for
    ``n_frames`` checkpoints along the gradient trajectory, computes the measurement times
    ``φ`` and the 2×2 parameter covariance ``FIM(φ)⁻¹`` → the 95%-confidence ellipse in
    ``(log α, log|β|)`` space. The animator (viz/oed.py) then shows the naive
    near-equilibrium samples sliding into the informative transient while the ellipse
    collapses — the differentiability moat, in motion. Everything is MEASURED at θ₀ (local
    OED, ``NUDGE-LIM-024``); this only READS the fit's output for drawing.
    """
    import numpy as np

    from nudge.inference.oed import (
        fisher_information,
        make_logistic_design_problem,
        optimize_design,
    )

    prob = make_logistic_design_problem()
    lo, hi = prob.phi_bounds
    naive = np.linspace(0.6 * hi, hi, n_obs)
    res = optimize_design(
        prob, naive, objective=objective, target="log_alpha", steps=steps,
        learning_rate=learning_rate, seed=seed, capture_phi=True,
    )

    # χ²(2 dof, 95%) = 5.991 — the 95%-confidence ellipse scale for a 2-parameter covariance.
    chi2_95 = 5.991
    idx = np.unique(np.linspace(0, steps - 1, num=min(n_frames, steps)).round().astype(int))
    frames: list[dict[str, Any]] = []
    for i in idx:
        phi = res.phi_history[int(i)]
        fim = fisher_information(prob, phi)
        p = fim.shape[0]
        scale = max(float(np.trace(fim)) / p, 1e-30)
        cov = np.linalg.inv(fim + 1e-8 * scale * np.eye(p))
        evals, evecs = np.linalg.eigh(cov)
        evals = np.clip(evals, 0.0, None)
        width = float(2.0 * np.sqrt(chi2_95 * evals[1]))   # full major axis
        height = float(2.0 * np.sqrt(chi2_95 * evals[0]))  # full minor axis
        angle = float(np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1])))
        frames.append({
            "step": int(i),
            "phi": [float(x) for x in np.sort(phi)],
            "ellipse": {"width": width, "height": height, "angle": angle},
            "target_crlb": _jsonsafe(float(np.diag(cov)[res.target_index])),
        })

    # The transient backdrop (analytic logistic solution x(t)) the samples slide over.
    meta = prob.meta
    alpha, k_cap, x0 = float(meta["alpha"]), float(meta["K"]), float(meta["x0"])
    tgrid = np.linspace(0.0, float(meta["t_max"]), 120)
    xt = k_cap / (1.0 + (k_cap / x0 - 1.0) * np.exp(-alpha * tgrid))

    return {
        "kind": "oed",
        "label": "logistic growth OED",
        "model": "logistic",
        "objective": objective,
        "target_parameter": res.target_name,
        "call": "",
        "reason": "",
        "crlb_improvement": _jsonsafe(res.crlb_improvement),
        "min_eig_improvement": _jsonsafe(res.min_eig_improvement),
        "animation": {
            "param_labels": ["log α (growth)", "log |β| (self-limitation)"],
            "theta0": [float(np.log(alpha)), float(np.log(-float(meta["beta"])))],
            "t_bounds": [float(lo), float(hi)],
            "traj_t": [float(x) for x in tgrid],
            "traj_x": [float(x) for x in xt],
            "frames": frames,
        },
        "caveat": (
            "local OED: the optimal design + the ellipse collapse are MEASURED at θ₀, not "
            "extrapolated (NUDGE-LIM-024)."
        ),
    }


# --------------------------------------------------------------------------- #
# GENERAL identifiability + OED tools (over the model registry — nudge.inference
# .model_registry). Thin orchestration: build a problem BY NAME, run the REAL
# matrix-free FIM diagnostic / gradient OED, return whatever it measures (incl.
# honest abstentions) + the provenance-carrying figure. NOT demo-specific: any
# registered differentiable model, ≥3 across domains, works identically.
# --------------------------------------------------------------------------- #
def _sloppiness_figuredata(report: Any, label: str) -> dict[str, Any]:
    """Serialise a ``SloppinessReport`` to the ``identifiability`` renderer's figure-data dict.

    The same shape :func:`identifiability_demo` emits, so the shared ``identifiability``
    renderer draws the FIM spectrum + naive-vs-measured verdict from MEASURED numbers.
    """
    import numpy as np

    nulls = report.null_directions
    # The renderer wants real floats (it draws inf/nan as "∞ (structural null)"), NOT None —
    # so keep inf/nan here rather than route through _jsonsafe.
    return {
        "kind": "identifiability",
        "label": label,
        "model_label": label,
        "call": report.label,
        "verdict": report.label,
        "reason": report.reason,
        "param_names": list(report.param_names),
        "fim_eigenvalues": [float(v) for v in np.asarray(report.fim_eigenvalues).ravel()],
        "cond_number": float(report.cond_number),
        "span_decades": float(report.spectral_span_decades),
        "spectral_span_decades": float(report.spectral_span_decades),
        "smallest_eigenvalue": float(report.smallest_eigenvalue),
        "largest_eigenvalue": float(report.largest_eigenvalue),
        "n_sloppy_dims": int(report.n_sloppy_dims),
        "n_null_dims": int(report.n_null_dims),
        "is_sloppy": bool(report.is_sloppy),
        "predictive": bool(report.predictive),
        "relative_prediction_std": _jsonsafe(report.relative_prediction_std),
        "pred_rel_tol": 0.05,
        "naive_verdict": report.naive_verdict,
        "naive_is_wrong": bool(report.naive_is_wrong),
        "sloppy_decade_threshold": 3.0,
        "null_hint": (nulls[0].hint if nulls else ""),
    }


def _null_directions_to_list(report: Any) -> list[dict[str, Any]]:
    """The named null (unrecoverable) directions of an ``unidentifiable`` verdict — the
    ACTIONABLE half: which parameter combination the data cannot pin, in plain language."""
    out: list[dict[str, Any]] = []
    for nd in report.null_directions:
        out.append({
            "param_loadings": {k: round(float(v), 4) for k, v in nd.param_loadings.items()},
            "prediction_sensitivity": _jsonsafe(nd.prediction_sensitivity),
            "hint": nd.hint,
        })
    return out


def identifiability_tool(
    model: str,
    *,
    free: list[str] | None = None,
    n_free: int = 0,
    method: str = "auto",
    sigma: float | None = None,
    seed: int = 0,
    with_figure: bool = True,
    fig_theme: str = "auto",
    fig_self_contained: bool = False,
    transport: str | None = None,
) -> dict[str, Any]:
    """Which parameters of a differentiable ODE model are identifiable / sloppy / unrecoverable?

    The general identifiability verb over the model registry
    (:mod:`nudge.inference.model_registry`): builds the model BY NAME, runs the REAL
    matrix-free Fisher-information diagnostic
    (:func:`nudge.inference.sloppiness.sloppiness_diagnostic_matrixfree`), and returns the
    verdict (``well-constrained`` / ``sloppy-but-predictive`` / ``unidentifiable``), the named
    null directions, the honest fail-safe bound (``NUDGE-LIM-023``: the matrix-free path never
    asserts an identifiability it cannot certify — it abstains instead), and — unless
    ``with_figure=False`` — the FIM-spectrum figure via the shared render seam (inline base64 +
    the regenerating ``fig.py`` + data sidecar when ``NUDGE_ENV=cloud``).

    ``free`` restricts to a named parameter subset; ``n_free`` is the population-/dimension-scale
    knob (``glv`` / ``linear_pathway`` / ``ad_qsp``); ``method`` ∈ ``auto`` / ``dense`` /
    ``iterative``; ``sigma`` overrides the observation noise. Everything is MEASURED at θ₀; the
    registry scope is ``NUDGE-LIM-027``.
    """
    import jax

    from nudge.inference.model_registry import build_identifiability_problem, get_model
    from nudge.inference.sloppiness import sloppiness_diagnostic_matrixfree

    try:
        entry = get_model(model)
    except KeyError as exc:
        return {"error": str(exc),
                "registered": [m["name"] for m in _registry_models()]}
    if not entry.supports_identifiability:
        supported = [m["name"] for m in _registry_models() if m["supports_identifiability"]]
        return {"error": f"model {model!r} does not support the identifiability tool",
                "supported": supported}
    # The FIM's smallest eigenvalues need float64 (float32 truncates the sloppy end); the ODE
    # builders also allocate their trajectories at build time, so enable x64 around BOTH.
    _prev = bool(getattr(jax.config, "jax_enable_x64", False))
    jax.config.update("jax_enable_x64", True)
    try:
        problem = build_identifiability_problem(
            model, free=free, n_free=n_free, sigma=sigma, seed=seed
        )
        report = sloppiness_diagnostic_matrixfree(
            problem.predict_fn, problem.theta0, problem.sigma,
            problem.param_names, method=method,
        )
    finally:
        jax.config.update("jax_enable_x64", _prev)

    abstained = report.label == "unidentifiable"
    label = f"{model} ({entry.domain})"
    out: dict[str, Any] = {
        "tool": "identifiability",
        "model": model,
        "domain": entry.domain,
        "n_params": problem.n_params,
        "param_names": list(problem.param_names),
        "method": method,
        "sigma": float(problem.sigma),
        "verdict": report.label,
        "abstained": abstained,
        "reason": report.reason,
        "cond_number": _jsonsafe(report.cond_number),
        "spectral_span_decades": _jsonsafe(report.spectral_span_decades),
        "smallest_eigenvalue": _jsonsafe(report.smallest_eigenvalue),
        "largest_eigenvalue": _jsonsafe(report.largest_eigenvalue),
        "n_sloppy_dims": int(report.n_sloppy_dims),
        "n_null_dims": int(report.n_null_dims),
        "is_sloppy": bool(report.is_sloppy),
        "predictive": bool(report.predictive),
        "relative_prediction_std": _jsonsafe(report.relative_prediction_std),
        "naive_verdict": report.naive_verdict,
        "naive_is_wrong": bool(report.naive_is_wrong),
        "null_directions": _null_directions_to_list(report),
        "fim_greedy_warning": report.fim_greedy_warning,
        "limitation": "NUDGE-LIM-023",
        "registry_scope": "NUDGE-LIM-027",
        "meta": {k: _jsonsafe(v) if isinstance(v, float) else v
                 for k, v in problem.meta.items()},
    }
    if with_figure:
        out["figure"] = render_result(
            "identifiability",
            _sloppiness_figuredata(report, label),
            out=None,
            emit_code=True,
            theme=fig_theme,
            self_contained=fig_self_contained,
            transport=transport,
            cli_call=f"identifiability(model={model!r})",
        )
    return out


def _registry_models() -> list[dict[str, Any]]:
    from nudge.inference.model_registry import list_models

    return list_models()


def _naive_oed_schedule(t_min: float, t_max: float, n_obs: int) -> Any:
    """A realistic sparse 'baseline + end of study' schedule — clusters at the two ends and
    MISSES the informative transient, so the confounded parameter pair is near-singular. This
    is the naive design the gradient OED is measured *against* (not a hand-picked degenerate)."""
    import numpy as np

    h = n_obs // 2
    span = t_max - t_min
    return np.concatenate([
        np.linspace(t_min, t_min + 0.05 * span, h),
        np.linspace(t_max - 0.05 * span, t_max, n_obs - h),
    ])


def _oed_animation_block(problem: Any, res: Any, *, t_bounds: tuple[float, float],
                         n_frames: int) -> dict[str, Any]:
    """The ``oed`` animator's frame sequence: the design-φ trajectory + the 2×2 confidence
    ellipse collapse over the gradient steps, for ANY 2-parameter :class:`DesignProblem`.

    Generalises :func:`oed_animation_demo` off the model registry: it re-instantiates nothing,
    only READS the captured φ-history (``optimize_design(..., capture_phi=True)``) and, per
    checkpoint, recomputes ``FIM(φ)⁻¹`` → the 95% ellipse. A representative observed-biomarker
    trajectory is drawn as the backdrop the measurement times slide over.
    """
    import jax.numpy as jnp
    import numpy as np

    from nudge.inference.oed import fisher_information

    chi2_95 = 5.991
    steps = int(res.n_steps)
    idx = np.unique(np.linspace(0, steps - 1, num=min(n_frames, steps)).round().astype(int))
    frames: list[dict[str, Any]] = []
    for i in idx:
        phi = res.phi_history[int(i)]
        fim = fisher_information(problem, phi)
        p = fim.shape[0]
        scale = max(float(np.trace(fim)) / p, 1e-30)
        cov = np.linalg.inv(fim + 1e-8 * scale * np.eye(p))
        evals, evecs = np.linalg.eigh(cov)
        evals = np.clip(evals, 0.0, None)
        frames.append({
            "step": int(i),
            "phi": [float(x) for x in np.sort(phi)],
            "ellipse": {
                "width": float(2.0 * np.sqrt(chi2_95 * evals[1])),
                "height": float(2.0 * np.sqrt(chi2_95 * evals[0])),
                "angle": float(np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))),
            },
            "target_crlb": _jsonsafe(float(np.diag(cov)[res.target_index])),
        })
    # a representative observed-biomarker trajectory the samples slide over (first output).
    # Use the observe map's own working dtype (no float64 override) so it doesn't warn when
    # x64 is off — the FIM numbers above are assembled precisely in numpy float64 regardless.
    lo, hi = t_bounds
    theta0_j = jnp.asarray(np.asarray(problem.theta0, dtype=np.float64))
    fine = jnp.asarray(np.linspace(lo, hi, 120), dtype=theta0_j.dtype)
    n_out = int(np.asarray(problem.observe(theta0_j, fine[:1])).shape[0])
    obs = np.asarray(problem.observe(theta0_j, fine)).reshape(len(fine), n_out)
    traj_x = np.exp(obs[:, 0])  # undo the log-observation transform for the backdrop
    return {
        "param_labels": list(problem.param_names),
        "theta0": [float(x) for x in np.asarray(problem.theta0, dtype=np.float64)],
        "t_bounds": [float(lo), float(hi)],
        "traj_t": [float(x) for x in np.asarray(fine)],
        "traj_x": [float(x) for x in traj_x],
        "frames": frames,
    }


def oed_tool(
    model: str,
    *,
    target: str | None = None,
    objective: str = "d_opt",
    n_obs: int = 8,
    steps: int = 400,
    learning_rate: float = 0.2,
    sigma: float | None = None,
    naive: list[float] | None = None,
    seed: int = 0,
    with_figure: bool = True,
    n_frames: int = 24,
    fig_theme: str = "auto",
    fig_self_contained: bool = False,
    transport: str | None = None,
) -> dict[str, Any]:
    """Design the experiment that best resolves a confounded parameter of an ODE model.

    The general OED verb over the model registry: builds a differentiable
    :class:`~nudge.inference.oed.DesignProblem` BY NAME, gradient-ascends the measurement
    schedule to the optimal design (:func:`nudge.inference.oed.optimize_design`), and returns
    the optimal design + the **MEASURED** identifiability gain (the target parameter's CRLB
    factor and the FIM smallest-eigenvalue lift — never asserted), the local-OED caveat
    (``NUDGE-LIM-024``), and — unless ``with_figure=False`` — the 95%-ellipse-collapse GIF via
    the shared render seam (inline base64 + the regenerating ``fig.py`` + sidecar when
    ``NUDGE_ENV=cloud``).

    ``target`` selects the parameter to resolve (default: the model's ``default_oed_target``);
    ``objective`` ∈ ``d_opt`` / ``a_opt`` / ``e_opt`` / ``crlb``; ``naive`` overrides the naive
    'baseline+end' schedule; ``sigma`` overrides the observation noise.
    """
    import numpy as np

    from nudge.inference.model_registry import build_oed_problem, get_model
    from nudge.inference.oed import optimize_design

    try:
        entry = get_model(model)
    except KeyError as exc:
        return {"error": str(exc),
                "registered": [m["name"] for m in _registry_models()]}
    if not entry.supports_oed:
        return {"error": f"model {model!r} does not support the OED tool",
                "supported": [m["name"] for m in _registry_models() if m["supports_oed"]]}
    problem = build_oed_problem(model, target=target, sigma=sigma, seed=seed)
    tgt = target or entry.default_oed_target or problem.param_names[0]
    if tgt not in problem.param_names:
        return {"error": f"unknown target {tgt!r} for model {model!r}",
                "param_names": list(problem.param_names)}
    lo, hi = problem.phi_bounds
    naive_design = (
        np.asarray(naive, dtype=float) if naive
        else _naive_oed_schedule(lo, hi, n_obs)
    )
    res = optimize_design(
        problem, naive_design, objective=objective, target=tgt, steps=steps,
        learning_rate=learning_rate, seed=seed, capture_phi=with_figure,
    )
    fim_init, fim_opt = res.fim_init, res.fim_opt
    corr_init = _fim_corr(fim_init)
    out: dict[str, Any] = {
        "tool": "oed",
        "model": model,
        "domain": entry.domain,
        "objective": objective,
        "target_parameter": res.target_name,
        "param_names": list(problem.param_names),
        "phi_init": [float(x) for x in np.sort(res.phi_init)],
        "phi_opt": [float(x) for x in np.sort(res.phi_opt)],
        "criterion_init": _jsonsafe(res.criterion_init),
        "criterion_opt": _jsonsafe(res.criterion_opt),
        "target_crlb_init": _jsonsafe(res.target_crlb_init),
        "target_crlb_opt": _jsonsafe(res.target_crlb_opt),
        "crlb_improvement": _jsonsafe(res.crlb_improvement),
        "min_eig_init": _jsonsafe(res.min_eig_init),
        "min_eig_opt": _jsonsafe(res.min_eig_opt),
        "min_eig_improvement": _jsonsafe(res.min_eig_improvement),
        "cond_init": _jsonsafe(float(np.linalg.cond(fim_init))),
        "cond_opt": _jsonsafe(float(np.linalg.cond(fim_opt))),
        "naive_correlation": _jsonsafe(corr_init),
        "measured_note": (
            "MEASURED at the nominal θ₀ — the optimal design and the CRLB / smallest-eigenvalue "
            "gains are computed, not asserted; a design recommendation, never a mechanism call."
        ),
        "limitation": "NUDGE-LIM-024",
        "registry_scope": "NUDGE-LIM-027",
    }
    if with_figure:
        anim_block = _oed_animation_block(
            problem, res, t_bounds=(lo, hi), n_frames=n_frames
        )
        fig_data = {
            "kind": "oed",
            "label": f"{model} OED",
            "model": model,
            "objective": objective,
            "target_parameter": res.target_name,
            "call": "",
            "reason": "",
            "phi_init": out["phi_init"],
            "phi_opt": out["phi_opt"],
            "target_crlb_init": out["target_crlb_init"],
            "target_crlb_opt": out["target_crlb_opt"],
            "crlb_improvement": out["crlb_improvement"],
            "min_eig_init": out["min_eig_init"],
            "min_eig_opt": out["min_eig_opt"],
            "min_eig_improvement": out["min_eig_improvement"],
            "animation": anim_block,
        }
        out["figure"] = render_result(
            "oed", fig_data, out=None, emit_code=True, theme=fig_theme,
            self_contained=fig_self_contained, animate=True, transport=transport,
            anim_frames=n_frames, cli_call=f"oed(model={model!r})",
        )
    return out


def _fim_corr(fim: Any) -> float:
    """The off-diagonal correlation of a 2×2 FIM (the confound strength), else NaN."""
    import numpy as np

    f = np.asarray(fim, dtype=float)
    if f.shape != (2, 2):
        return float("nan")
    denom = float(np.sqrt(f[0, 0] * f[1, 1]))
    return float(f[0, 1] / denom) if denom > 0 else float("nan")


def robustness_animation_demo(
    *, k: float = 1.0, vmax: float = 2.0, basal: float = 0.05,
    n_hi: float = 6.0, n_lo: float = 1.5, n_frames: int = 24,
) -> dict[str, Any]:
    """Robustness demo enriched for the ANIMATION: a 1-node switch swept TOWARD its fold.

    Sweeps the self-activation Hill ``n`` from robust (deep two-basin bistability) down
    toward and past the saddle-node fold, and per frame returns the fused proximity dial +
    the three channel proximities (all 0..1) + the honest ``call`` (``robust`` → ``near-fold``
    → ``unresolved``/``not-bistable``) + the **potential well** ``U(x)`` (two basins → one).
    The animator (viz/robustness.py) shows the dial climbing while the well flattens — the
    tipping point, in motion. It only READS ``bifurcation_proximity`` / ``classify_robustness``
    (no fit); near the fold the dial is a ONE-SIDED lower bound (``NUDGE-LIM-012``).
    """
    import jax.numpy as jnp
    import numpy as np

    from nudge.inference.bifurcation import bifurcation_proximity, classify_robustness

    # x-grid from the robust (n_hi) switch's high basin, so the well morph shares one axis.
    hi_circ = _build_named_circuit("1node", n=n_hi, k=k, vmax=vmax, basal=basal)
    hi_fps = hi_circ.fixed_points() or []
    x_hi = max((float(s[0]) for s, _ in hi_fps), default=float(vmax))
    x_max = max(1.25 * x_hi, 1.0)
    xg = np.linspace(0.0, x_max, 160)

    def potential(circ: Any) -> np.ndarray:
        params = circ.base_params()
        f = np.asarray(
            jax.vmap(lambda x: circ.vector_field(jnp.array([x]), params)[0])(
                jnp.asarray(xg)
            ),
            dtype=float,
        )
        # U(x) = -∫₀ˣ f  (trapezoid); wells at stable FPs, barrier at the saddle.
        u = -np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(xg))])
        return u - float(np.min(u))

    import jax

    frames: list[dict[str, Any]] = []
    u_max = 0.0
    for nv in np.linspace(n_hi, n_lo, n_frames):
        circ = _build_named_circuit("1node", n=float(nv), k=k, vmax=vmax, basal=basal)
        score = bifurcation_proximity(circ)
        call, reason = classify_robustness(score)
        u = potential(circ)
        u_max = max(u_max, float(np.max(u)))
        fps = circ.fixed_points() or []
        chan = (score.channels.get("channel_proximities", {}) if score else {})
        frames.append({
            "n": float(nv),
            "U": [float(v) for v in u],
            "proximity": _jsonsafe(score.proximity if score else float("nan")),
            "one_sided": bool(score.one_sided) if score else False,
            "channel_proximities": {k2: _jsonsafe(v) for k2, v in chan.items()},
            "call": call,
            "reason": reason,
            "fixed_points": [[float(s[0]), lab] for s, lab in fps],
        })

    return {
        "kind": "robustness",
        "label": "1-node switch → the fold",
        "call": frames[-1]["call"],
        "reason": frames[-1]["reason"],
        "proximity": frames[-1]["proximity"],
        "one_sided": frames[-1]["one_sided"],
        "channels": frames[-1]["channel_proximities"],
        "animation": {
            "x": [float(v) for v in xg],
            "u_max": float(u_max) or 1.0,
            "frames": frames,
        },
    }


def fibrillization_animation_demo(*, n_frames: int = 28, seed: int = 0) -> dict[str, Any]:
    """Aggregation demo enriched for the ANIMATION: the **gauge orbit** — the honesty visual.

    A single amyloid aggregation curve is PROVABLY non-identifiable in the three microscopic
    constants: the exact continuous gauge ``(k_n, k_+, k_2) → (k_n/α, α·k_+, k_2/α)`` leaves
    the mass-fraction curve — and the two identifiable composites κ, λ — UNCHANGED. This
    builds that orbit: a sinusoidal sweep of α so the three constants swing (a seamless loop)
    while the curve and κ, λ stay put. It re-simulates a gauged triple to MEASURE the curve
    invariance (``gauge_check`` = max|Δ mass-fraction| ≈ 0). No fit needed — the gauge is an
    exact analytic symmetry (Meisl 2016 / Michaels 2020), so this is fast; the animator only
    DRAWS it. Individual constants are NOT identifiable → the constants panel abstains
    (``NUDGE-LIM-021``).
    """
    import numpy as np

    from nudge.mechanisms.fibrillization import (
        _BALANCED_TRUTH,
        AggregationParams,
        composite_lambda_kappa,
        simulate_aggregation_curve,
    )

    truth = _BALANCED_TRUTH
    m_tot = 1.0
    curve = simulate_aggregation_curve(params=truth, m_tot=m_tot, obs_noise=0.0, seed=seed)
    t = np.asarray(curve.t_obs, dtype=float)
    m = np.clip(np.asarray(curve.signal, dtype=float).mean(axis=0), 0.0, 1.0)
    lam, kap = composite_lambda_kappa(truth, m_tot)

    # a sinusoidal α(frame) so the orbit loops seamlessly (α=1 at both ends).
    log_amp = float(np.log(2.8))
    alphas = np.exp(log_amp * np.sin(2.0 * np.pi * np.arange(n_frames) / n_frames))
    orbit = [
        {"alpha": float(a), "k_n": float(truth.k_n / a),
         "k_plus": float(truth.k_plus * a), "k_2": float(truth.k_2 / a)}
        for a in alphas
    ]

    # MEASURE the curve invariance under the gauge (a strong-α gauged triple).
    a_check = 2.5
    gauged = AggregationParams(k_n=truth.k_n / a_check, k_plus=truth.k_plus * a_check,
                               k_2=truth.k_2 / a_check, n_c=truth.n_c, n_2=truth.n_2)
    m_g = np.clip(
        np.asarray(simulate_aggregation_curve(params=gauged, m_tot=m_tot, obs_noise=0.0,
                                              seed=seed).signal, dtype=float).mean(axis=0),
        0.0, 1.0,
    )
    gauge_check = float(np.max(np.abs(m - m_g)))

    return {
        "kind": "aggregation",
        "call": "composites-identified",
        "reason": ("the mass-fraction curve fixes only the composites κ, λ; the three "
                   "microscopic constants are gauge-degenerate (NUDGE-LIM-021)"),
        "label": "amyloid aggregation — the gauge orbit",
        "kappa": _jsonsafe(kap),
        "lambda": _jsonsafe(lam),
        "individual_k_identifiable": False,
        "null_direction": [0.5773502691896258, -0.5773502691896258, 0.5773502691896258],
        "gauge_check": gauge_check,
        "animation": {
            "t": [float(v) for v in t],
            "m": [float(v) for v in m],
            "kappa": _jsonsafe(kap),
            "lambda": _jsonsafe(lam),
            "k_labels": ["kₙ (primary)", "k₊ (elongation)", "k₂ (secondary)"],
            "truth": {"k_n": float(truth.k_n), "k_plus": float(truth.k_plus),
                      "k_2": float(truth.k_2)},
            "gauge_check": gauge_check,
            "orbit": orbit,
        },
    }


def fibrillization_demo(
    *,
    mode: str = "single",
    inhibitor_target: str = "secondary_nucleation",
    steps: int = 600,
    seed: int = 0,
) -> dict[str, Any]:
    """Synthesize an amyloid aggregation curve + run NUDGE on it in one call (no data file).

    The zero-setup efficiency demo of the fibrillization capability (``NUDGE-METHOD-013``).

    - ``mode="single"`` — one aggregation curve → the identifiable composites (κ, λ) + the
      MEASURED non-identifiability of the three microscopic constants (the exact gauge null,
      ``NUDGE-LIM-021``); the honest answer a control LLM agent took 12.2 min / 28 turns / 6
      scripts to hand-derive, in one call.
    - ``mode="inhibitor"`` — a control vs inhibited curve pair → which microscopic step
      (primary / elongation / secondary nucleation) the inhibitor lowered, or abstain.
    - ``mode="series"`` — a concentration series + a seeded anchor → resolve the three
      individual constants (and show a series WITHOUT the anchor stays degenerate).
    """
    from nudge.mechanisms.fibrillization import (
        _BALANCED_TRUTH,
        attribute_aggregation,
        attribute_inhibitor,
        resolve_series,
        simulate_aggregation_curve,
        simulate_concentration_series,
        simulate_inhibitor_pair,
    )

    if mode == "single":
        curve = simulate_aggregation_curve(seed=seed)
        res = attribute_aggregation(curve, steps=steps, seed=seed)
        ident = res.identifiability
        return {
            "mode": "single",
            "call": res.call,
            "reason": res.reason,
            "guidance": res.guidance,
            "kappa": _jsonsafe(res.kappa),
            "lambda": _jsonsafe(res.lam),
            "kappa_ci": [_jsonsafe(x) for x in res.fit.kappa_ci],
            "lambda_ci": [_jsonsafe(x) for x in res.fit.lam_ci],
            "individual_k_identifiable": bool(res.individual_k_identifiable),
            "cond_number": _jsonsafe(ident.cond_number),
            "null_direction": [float(x) for x in ident.null_direction],
            "unidentifiable": list(ident.unidentifiable),
            "gauge_check": _jsonsafe(ident.gauge_check),
            "ground_truth": dict(curve.ground_truth),
        }
    if mode == "inhibitor":
        ctrl, inhib, gt = simulate_inhibitor_pair(target=inhibitor_target, seed=seed)
        res_i = attribute_inhibitor(ctrl, inhib, steps=steps, seed=seed)
        return {
            "mode": "inhibitor",
            "call": res_i.call,
            "reason": res_i.reason,
            "is_reliable": bool(res_i.is_reliable),
            "r_lambda": _jsonsafe(res_i.r_lambda),
            "r_kappa": _jsonsafe(res_i.r_kappa),
            "ground_truth": gt,
        }
    if mode == "series":
        series = simulate_concentration_series(with_anchor=True, seed=seed)
        with_anchor = resolve_series(series, use_anchor=True, steps=max(steps * 2, 1200))
        no_anchor = resolve_series(series, use_anchor=False, steps=max(steps * 2, 1200))
        truth = {"k_n": _BALANCED_TRUTH.k_n, "k_plus": _BALANCED_TRUTH.k_plus,
                 "k_2": _BALANCED_TRUTH.k_2}
        return {
            "mode": "series",
            "with_anchor": {
                "identifiable": bool(with_anchor.identifiable),
                "cond_number": _jsonsafe(with_anchor.cond_number),
                "k_n": _jsonsafe(with_anchor.k_n), "k_plus": _jsonsafe(with_anchor.k_plus),
                "k_2": _jsonsafe(with_anchor.k_2), "reason": with_anchor.reason,
            },
            "without_anchor": {
                "identifiable": bool(no_anchor.identifiable),
                "cond_number": _jsonsafe(no_anchor.cond_number),
                "reason": no_anchor.reason,
            },
            "ground_truth": truth,
        }
    raise ValueError(f"unknown mode {mode!r}; expected single | inhibitor | series")


def fibrillization_file(
    path: str,
    *,
    time_col: str | None = None,
    value_col: str | None = None,
    m_tot: float = 1.0,
    steps: int = 600,
    seed: int = 0,
) -> dict[str, Any]:
    """Attribute a SINGLE aggregation curve from a CSV/TSV (``time`` + ``mass_fraction``).

    The file-input twin of :func:`fibrillization_demo` (``mode="single"``): reads a
    polymerization / aggregation curve (mass fraction ∈ [0, 1] vs time), fits the microscopic
    filament-assembly moment model, and returns the identifiable composites ``κ`` / ``λ`` +
    the MEASURED non-identifiability of the three microscopic rate constants (the gauge null,
    ``NUDGE-LIM-021``) — the honest answer an unaided agent hand-derives in ~12 min, in one call.
    ``time_col`` / ``value_col`` default to the first two columns; ``m_tot`` is the (normalized)
    initial monomer concentration (1.0 for a mass-fraction curve).
    """
    import numpy as np
    import pandas as pd

    from nudge.mechanisms.fibrillization import (
        AggregationCurve,
        _grid,
        attribute_aggregation,
    )

    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep)
    cols = list(df.columns)
    tcol = time_col or cols[0]
    vcol = value_col or cols[1]
    t = df[tcol].to_numpy(dtype=float)
    y = df[vcol].to_numpy(dtype=float)
    order = np.argsort(t)
    t, y = t[order], y[order]
    t_max = float(t[-1])
    dt = t_max / 2000.0
    n_steps, obs_idx = _grid(t_max, dt, t)
    curve = AggregationCurve(
        signal=y[None, :], t_obs=t, m_tot=m_tot, dt=dt, n_steps=n_steps,
        obs_idx=obs_idx, p0=0.0, ground_truth={},
    )
    res = attribute_aggregation(curve, steps=steps, seed=seed)
    ident = res.identifiability
    return {
        "call": res.call, "reason": res.reason, "guidance": res.guidance,
        "kappa": _jsonsafe(res.kappa), "lambda": _jsonsafe(res.lam),
        "kappa_ci": [_jsonsafe(x) for x in res.fit.kappa_ci],
        "lambda_ci": [_jsonsafe(x) for x in res.fit.lam_ci],
        "individual_k_identifiable": bool(res.individual_k_identifiable),
        "cond_number": _jsonsafe(ident.cond_number),
        "null_direction": [float(x) for x in ident.null_direction],
    }
