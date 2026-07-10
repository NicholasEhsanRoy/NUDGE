"""The ``nudge`` command-line interface — a thin, honest layer over the tested API.

Verbs:

- ``nudge load FILE`` — backed-load a Perturb-seq ``.h5ad`` and summarise it.
- ``nudge check-data FILE`` — run the raw-count ingestion guardrail (fails loudly).
- ``nudge attribute FILE --target GENE`` — run covariance attribution and print the
  call + the honest abstention / skip reasons.
- ``nudge mechanisms`` — list the mechanism library (registry + Mechanism Cards).
- ``nudge explain QUERY`` — explain an abstention, decoy, limitation, or mechanism —
  including *which* documented failure mode / decoy explains a given abstention.

This module owns **no** modelling logic: attribution is
:func:`nudge.inference.pipeline.attribute_across_operating_points`; the knowledge
lookups are :mod:`nudge.knowledge`; the ingest guard is
:func:`nudge.data.ingest.check_counts`.
"""

from __future__ import annotations

from typing import Any

import typer

app = typer.Typer(
    name="nudge",
    help="Mechanism attribution for Perturb-seq screens (fails safely and loudly).",
    no_args_is_help=True,
    add_completion=False,
)

_TOPOLOGIES = ("1node", "2node", "toggle")


def _echo(msg: str = "") -> None:
    typer.echo(msg)


def _read_adata(path: str, *, backed: bool) -> Any:
    import anndata as ad

    return ad.read_h5ad(path, backed="r" if backed else None)


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
@app.command()
def load(
    path: str = typer.Argument(..., help="a Perturb-seq .h5ad file"),
    preset: str = typer.Option(
        "native",
        help="'gladstone' to use the Gladstone T-cell loader; 'native' for a "
        "NUDGE-schema file (obs['condition'] present).",
    ),
    target_genes: str = typer.Option(
        "",
        "--target-genes",
        help="comma-separated perturbation gene symbols to subset (gladstone preset).",
    ),
) -> None:
    """Backed-load a dataset and print a summary (conditions, cells, genes)."""
    genes = tuple(g for g in target_genes.split(",") if g)
    if preset == "gladstone":
        from nudge.data.loaders.tier2 import load_gladstone

        adata = load_gladstone(path, target_genes=genes or None)
    else:
        adata = _read_adata(path, backed=True)

    n_obs, n_vars = adata.shape
    _echo(f"file: {path}")
    _echo(f"cells: {n_obs:,}   genes: {n_vars:,}")
    obs = getattr(adata, "obs", None)
    if obs is not None and "condition" in obs:
        counts = obs["condition"].value_counts()
        _echo(f"conditions ({len(counts)}):")
        for name, n in counts.items():
            _echo(f"  {name:>16}  {int(n):>8,} cells")
    else:
        _echo("no obs['condition'] — pass --preset gladstone or a NUDGE-schema file.")
    if n_vars <= 24:
        _echo(f"genes: {', '.join(map(str, adata.var_names))}")


# --------------------------------------------------------------------------- #
# check-data
# --------------------------------------------------------------------------- #
@app.command("check-data")
def check_data(
    path: str = typer.Argument(..., help="a Perturb-seq .h5ad file"),
    readout_genes: str = typer.Option(
        "", help="comma-separated readout genes that must be present."
    ),
) -> None:
    """Run the raw-count ingestion guardrail. Exit 1 (loudly) on a hard violation."""
    import warnings

    from nudge.data.ingest import IngestError, check_counts

    adata = _read_adata(path, backed=False)
    genes = tuple(g for g in readout_genes.split(",") if g)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            check_counts(adata, readout_genes=genes)
        except IngestError as exc:
            _echo(f"REJECTED: {exc}")
            raise typer.Exit(code=1) from None
    _echo("OK: .X looks like raw integer counts.")
    for w in caught:
        _echo(f"WARNING: {w.message}")


# --------------------------------------------------------------------------- #
# attribute
# --------------------------------------------------------------------------- #
@app.command()
def attribute(
    path: str = typer.Argument(..., help="a Perturb-seq .h5ad (one operating point)"),
    target: str = typer.Option(..., "--target", "-t", help="perturbation gene"),
    topology: str = typer.Option(
        "1node", help=f"circuit hypothesis: {', '.join(_TOPOLOGIES)}"
    ),
    marker: list[str] = typer.Option(
        [], "--marker", help="SPECIES=GENE1,GENE2 (repeatable); defaults to species"
    ),
    control: str = typer.Option("WT", help="the control/WT condition label"),
    steps: int = typer.Option(200, help="fit steps (lower = faster, less precise)"),
    min_cells: int = typer.Option(200, help="min target cells to attempt attribution"),
    preset: str = typer.Option("native", help="'gladstone' or 'native'"),
) -> None:
    """Attribute a perturbation's mechanism at one operating point — honestly.

    Single-condition attribution is *expected* to abstain between gain and
    threshold (the measured Fisher degeneracy); the breaker needs a second
    operating point. Skips (too few cells / unreliable LNA) are reported, not hidden.
    """
    from nudge.service import TOPOLOGIES, attribute_file

    if topology not in TOPOLOGIES:
        raise typer.BadParameter(f"topology must be one of {TOPOLOGIES}")

    _echo(f"attributing {target}  (topology={topology}, control={control})")
    report = attribute_file(
        path,
        target,
        topology=topology,
        markers=list(marker),
        control=control,
        steps=steps,
        min_cells=min_cells,
        preset=preset,
    )
    _print_report(report)


def _print_report(report: Any) -> None:
    _echo(f"\ntarget: {report.target}")
    for label, n in report.n_cells.items():
        if label in report.single:
            call, nlls = report.single[label]
            prof = "  ".join(f"{k}={v:.3f}" for k, v in nlls.items())
            _echo(f"  {label}: n={n:,}  → {call}   [{prof}]")
        else:
            _echo(f"  {label}: n={n:,}  SKIPPED: {report.skipped.get(label, '?')}")
    if report.multi is not None:
        call, nlls = report.multi
        prof = "  ".join(f"{k}={v:.3f}" for k, v in nlls.items())
        _echo(f"  → BREAKER (joint): {call}   [{prof}]")
    else:
        _echo("  → breaker needs ≥2 operating points; single-condition may abstain.")
    _echo("\nWhy an abstention?  run:  nudge explain <verdict>   (e.g. unresolved)")


# --------------------------------------------------------------------------- #
# warmup
# --------------------------------------------------------------------------- #
@app.command()
def warmup() -> None:
    """Pre-compile the hot JAX paths (dummy data) so the first real fit is fast.

    Useful before a live demo / in a long-lived session — the dose-response model and
    the circuit fixed-point kernel compile once (~0.5 s) then run ~50–250× faster. A
    notebook's first cell can call ``nudge.warmup()``; the MCP server warms on startup.
    """
    from nudge.warmup import warmup as _warmup

    _echo("warming JAX compile caches …")
    dt = _warmup(quiet=True)
    _echo(f"done in {dt:.2f}s — the next attribution / dose-response fit is now fast.")


# --------------------------------------------------------------------------- #
# dose-response
# --------------------------------------------------------------------------- #
@app.command("dose-response")
def dose_response(
    path: str = typer.Argument(
        ..., help="a 2-column CSV/TSV (dose,response) OR an .h5ad knockdown screen"
    ),
    direction: str = typer.Option(
        "repress", help="'repress' (response falls with dose) or 'activate' (rises)"
    ),
    dose_col: str = typer.Option("dose", help="CSV dose column"),
    response_col: str = typer.Option("response", help="CSV response column"),
    target: str = typer.Option(
        "", "--target", "-t", help="h5ad: guide-group prefix (e.g. OCT4)"
    ),
    target_gene: str = typer.Option(
        "", help="h5ad: the gene whose knockdown is the dose axis (e.g. POU5F1)"
    ),
    signature: str = typer.Option(
        "", help="h5ad: comma-separated readout genes whose mean is the response"
    ),
    group_col: str = typer.Option("guide", help="h5ad: obs column of guide IDs"),
    control: str = typer.Option("WT", help="h5ad: control/WT condition label"),
    min_cells: int = typer.Option(15, help="h5ad: min cells/guide for a dose point"),
    n_boot: int = typer.Option(500, help="bootstrap resamples for the n/K CIs"),
    seed: int = typer.Option(0, help="bootstrap RNG seed"),
    fig_out: str = typer.Option(
        "", "--fig-out", help="opt-in: also write a PNG of the fit here (text unchanged)"
    ),
    fig_code: bool = typer.Option(
        True, "--fig-code/--no-fig-code",
        help="with --fig-out, also emit the regenerating fig.py + data sidecar",
    ),
    fig_theme: str = typer.Option(
        "auto", "--fig-theme", help="figure theme: auto|light|dark"
    ),
    fig_self_contained: bool = typer.Option(
        False, "--fig-self-contained",
        help="inline the data inside fig.py (one portable file; for Artifacts)",
    ),
) -> None:
    """Attribute a mechanism from a dose-response curve — switch vs graded, or abstain.

    The same K (threshold) / n (gain) / v_max (ceiling) vocabulary as single-cell
    attribution, read from a dose axis instead (two measurements of one circuit).
    Reports
    ``n`` as an **apparent population gain** with a CI, and abstains (``unresolved`` /
    ``no-effect``) rather than over-call an unidentifiable curve — e.g. when the doses
    do
    not span the inflection. Pass ``--fig-out fig.png`` to also write an honest figure
    of the fit (the abstention is drawn as an abstention); text output is unchanged.
    """
    from nudge.service import dose_response_file

    sig = [g.strip() for g in signature.split(",") if g.strip()]
    out = dose_response_file(
        path,
        direction=direction,
        dose_col=dose_col,
        response_col=response_col,
        target=target or None,
        target_gene=target_gene or None,
        signature=sig or None,
        group_col=group_col,
        control=control,
        min_cells=min_cells,
        n_boot=n_boot,
        seed=seed,
        fig_out=fig_out or None,
        fig_code=fig_code,
        fig_theme=fig_theme,
        fig_self_contained=fig_self_contained,
        fig_label=target or None,
        cli_call=f"nudge dose-response {path} --fig-out {fig_out}" if fig_out else None,
    )
    lo, hi = out["ci_n"]
    _echo(f"dose-response  ({out['n_points']} points, direction={out['direction']})")
    _echo(
        f"  apparent gain n = {out['n_apparent_gain']:.2f}  95% CI [{lo:.2f}, {hi:.2f}]"
        f"   K = {out['K_threshold']:.3f}   R² = {out['r2']:.3f}"
    )
    _echo(
        f"  ΔBIC(graded−switch) = {out['delta_bic_graded_minus_switch']:+.1f}   "
        f"spans_inflection = {out['spans_inflection']}   "
        f"dose∈[{out['dose_range'][0]:.3f}, {out['dose_range'][1]:.3f}]"
    )
    _echo(f"\n  → CALL: {out['call'].upper()}")
    _echo(f"     {out['reason']}")
    _echo(
        "\n  note: n is an APPARENT population gain + CI, not molecular cooperativity."
    )
    fig = out.get("figure")
    if fig:
        _echo(f"\n  wrote {fig['png_path']}")
        if fig.get("code_path"):
            _echo(f"  wrote {fig['code_path']}  (re-runs to regenerate the figure)")
        if fig.get("data_path"):
            _echo(f"  wrote {fig['data_path']}")
        if fig.get("abstained"):
            _echo("  (the figure draws the abstention AS an abstention)")


# --------------------------------------------------------------------------- #
# synergy / epistasis
# --------------------------------------------------------------------------- #
@app.command()
def synergy(
    path: str = typer.Argument(..., help="an .h5ad with control/A/B/A+B conditions"),
    a_label: str = typer.Option(..., "--a", help="the A condition label (obs)"),
    b_label: str = typer.Option(..., "--b", help="the B condition label (obs)"),
    ab_label: str = typer.Option(..., "--ab", help="the A+B combination label (obs)"),
    control_label: str = typer.Option(
        "control", "--control", help="the control/WT condition label"
    ),
    condition_col: str = typer.Option("condition", help="obs condition-label column"),
    signature: str = typer.Option(
        "", help="comma-separated signature genes; default projects onto the "
        "additive axis fixed by the singles (recommended)"
    ),
    n_top_genes: int = typer.Option(2000, help="top-variable genes for the projection"),
    n_boot: int = typer.Option(1000, help="bootstrap resamples for the interaction CI"),
    seed: int = typer.Option(0, help="bootstrap RNG seed"),
    min_cells: int = typer.Option(30, help="min cells/condition or the call abstains"),
) -> None:
    """Classify a two-perturbation combination — additive vs synergistic/buffering.

    Reads {control, A, B, A+B} as three operating points against a shared control,
    reduces each to a scalar **effect** (log-fold-change space; the additive null is
    Bliss), and reports the **interaction** (``effect(A+B) − [effect(A)+effect(B)]``)
    with a bootstrap CI. Calls ``additive`` / ``synergistic`` / ``buffering`` — or
    abstains (``no-effect`` / ``unresolved``) when an arm is underpowered or the CI is
    too wide. A super-additive residual is NOT a hidden-node claim (NUDGE-LIM-009).
    """
    from nudge.service import synergy_file

    sig = [g.strip() for g in signature.split(",") if g.strip()]
    out = synergy_file(
        path,
        control_label=control_label,
        a_label=a_label,
        b_label=b_label,
        ab_label=ab_label,
        condition_col=condition_col,
        signature=sig or None,
        n_top_genes=n_top_genes,
        n_boot=n_boot,
        seed=seed,
        min_cells=min_cells,
    )
    nc = out["n_cells"]
    ilo, ihi = out["ci_interaction"]
    _echo(
        f"synergy  A={a_label}  B={b_label}  A+B={ab_label}  "
        f"(effect space: {out['effect_space']})"
    )
    _echo(
        f"  cells: control={nc['control']:,}  A={nc['A']:,}  B={nc['B']:,}  "
        f"A+B={nc['A+B']:,}"
    )
    _echo(
        f"  effect A = {out['effect_a']:+.3f}   effect B = {out['effect_b']:+.3f}   "
        f"additive pred = {out['additive_pred']:+.3f}"
    )
    _echo(
        f"  observed A+B = {out['effect_ab']:+.3f}   "
        f"interaction = {out['interaction']:+.3f}  95% CI [{ilo:+.3f}, {ihi:+.3f}]"
    )
    _echo(f"  ΔBIC(additive−free) = {out['delta_bic_additive_minus_free']:+.1f}")
    _echo(f"\n  → CALL: {out['call'].upper()}")
    _echo(f"     {out['reason']}")


# --------------------------------------------------------------------------- #
# cross-modality readout (fluorescence / activity / fold-change, not counts)
# --------------------------------------------------------------------------- #
@app.command("cross-modality")
def cross_modality(
    path: str = typer.Argument(
        ..., help="a tidy CSV/TSV of a CONTINUOUS readout (dose/response/variant cols)"
    ),
    dose_col: str = typer.Option(..., "--dose-col", help="dose column (e.g. IPTGuM)"),
    response_col: str = typer.Option(
        ..., "--response-col", help="continuous readout column (e.g. fold-change mean)"
    ),
    variant_col: str = typer.Option(
        ..., "--variant-col", help="column of variant/mutant labels"
    ),
    control: str = typer.Option(
        ..., "--control", help="the control/WT variant label to compare against"
    ),
    class_col: str = typer.Option(
        "", "--class-col", help="optional ground-truth class column (carried through)"
    ),
    modality: str = typer.Option(
        "fluorescence", help="continuous modality: fluorescence | activity | foldchange"
    ),
    direction: str = typer.Option(
        "activate", help="'activate' (readout rises with dose, e.g. induction) or "
        "'repress'"
    ),
    filt: list[str] = typer.Option(
        [], "--filter", help="KEY=VALUE to pin another axis (repeatable, e.g. "
        "operator=O2)"
    ),
    n_boot: int = typer.Option(400, help="bootstrap resamples for the K/n/amp CIs"),
    seed: int = typer.Option(0, help="bootstrap RNG seed"),
) -> None:
    """Attribute a panel of CONTINUOUS-readout dose-responses (cross-modality adapter).

    Runs the *same* K (threshold) / n (gain) / v_max (ceiling) attribution NUDGE does on
    counts, but on a continuous single channel (flow fluorescence, an activity reporter,
    a fold-change summary). The modality is **declared, never guessed** — the bouncer
    refuses log-normalized or raw counts masquerading as fluorescence (NUDGE-LIM-008).
    Each variant is localized to one knob vs the control — or abstains
    (**non-responsive** / **inconclusive**). This is the Chure-2019 LacI benchmark:
    DNA-binding mutants localize to ceiling/leakiness, inducer-binding to threshold.
    """
    from nudge.service import cross_modality_panel_file

    filters: dict[str, str] = {}
    for spec in filt:
        if "=" not in spec:
            raise typer.BadParameter(f"--filter {spec!r} must be KEY=VALUE")
        k, _, v = spec.partition("=")
        filters[k.strip()] = v.strip()

    out = cross_modality_panel_file(
        path,
        dose_col=dose_col,
        response_col=response_col,
        variant_col=variant_col,
        control_variant=control,
        class_col=class_col or None,
        filters=filters or None,
        modality=modality,
        direction=direction,
        n_boot=n_boot,
        seed=seed,
    )
    _echo(
        f"cross-modality  ({out['modality']}, direction={out['direction']}, "
        f"control={out['control_variant']})"
    )
    header = f"  {'variant':10} {'class':6} {'knob':14} {'call':11} K        n    log2K"
    _echo(header)
    for v in out["variants"]:
        cls = v["class_label"] or "—"
        k = v["K_threshold"]
        ks = f"{k:8.2f}" if k == k else "     nan"  # nan-safe
        lr = v["log2_K_ratio_vs_control"]
        lrs = f"{lr:+6.2f}" if lr == lr else "   —  "
        _echo(
            f"  {v['variant']:10} {cls:6} {v['knob']:14} {v['call']:11} {ks} "
            f"{v['n_apparent_gain']:5.2f} {lrs}"
        )
    _echo(
        "\n  knob: threshold=EC50 shift · ceiling=leakiness/range · gain=Hill n · "
        "\n  non-responsive/inconclusive = honest abstention (never a forced call)."
    )


# --------------------------------------------------------------------------- #
# robustness (bifurcation / tipping-point proximity — the "robustness dial")
# --------------------------------------------------------------------------- #
@app.command()
def robustness(
    path: str = typer.Argument(
        "", help="optional (n_cells, n_species) activity file (.npy/CSV/TSV); omit to "
        "score a parametric circuit from the kinetics below"
    ),
    topology: str = typer.Option(
        "1node", help=f"bistable motif: {', '.join(_TOPOLOGIES)}"
    ),
    n: float = typer.Option(6.0, "--n", help="switch cooperativity (Hill n)"),
    k: float = typer.Option(1.0, "--k", help="switch threshold (K)"),
    vmax: float = typer.Option(2.0, "--vmax", help="switch ceiling (v_max)"),
    basal: float = typer.Option(0.05, "--basal", help="basal production"),
    steps: int = typer.Option(200, help="LNA-depth calibration steps (data path only)"),
    seed: int = typer.Option(0, help="RNG seed (data path only)"),
) -> None:
    """How close is a bistable switch to LOSING bistability (a saddle-node fold)?

    Reports a **robustness dial**: the fused 0..1 ``proximity`` + three raw channels —
    **critical slowing** (min|Reλ|→0), **basin collapse** (node→saddle→0), and **LNA
    lobe swell** (lobe ratio→1). The call is ``near-fold`` / ``robust`` / ``unresolved``
    (deep basin, abstains) / ``not-bistable``. Near the fold the number is a **ONE-SIDED
    LOWER BOUND** — the LNA Gaussian breaks down precisely at the fold, so it is least
    reliable exactly there (NUDGE-LIM-012). With a data file, the sequencing depth
    is calibrated from the data to gate the LNA lobe channel's reliability.
    """
    from nudge.service import (
        ROBUSTNESS_TOPOLOGIES,
        bifurcation_file,
        robustness_circuit,
    )

    if topology not in ROBUSTNESS_TOPOLOGIES:
        raise typer.BadParameter(f"topology must be one of {ROBUSTNESS_TOPOLOGIES}")

    if path:
        out = bifurcation_file(
            path, topology=topology, n=n, k=k, vmax=vmax, basal=basal,
            steps=steps, seed=seed,
        )
    else:
        out = robustness_circuit(topology, n=n, k=k, vmax=vmax, basal=basal)

    _echo(f"robustness dial  (topology={topology}, n={n:g}, K={k:g}, v_max={vmax:g})")
    if out["proximity"] is None:
        _echo(f"\n  → CALL: {out['call'].upper()}")
        _echo(f"     {out['reason']}")
        return
    cp = out["channels"].get("channel_proximities", {})
    _echo(
        f"  proximity dial = {out['proximity']:.3f} / 1.0   "
        f"(one-sided lower bound: {out['one_sided']})   modes={out['n_stable_modes']}"
    )
    _echo(
        f"  channels:  critical-slowing min|Reλ| = {out['min_re_lambda']:.3f} (→0)   "
        f"node→saddle = {out['node_saddle_distance']:.3f} (→0)   "
        f"LNA lobe ratio = {out['lna_lobe_ratio']:.3f} (→1)"
    )
    if cp:
        _echo(
            "  per-channel proximity:  "
            + "  ".join(f"{key}={val:.3f}" for key, val in cp.items())
        )
    if "lna_reason" in out:
        _echo(f"  LNA reliability at this depth (lobe channel): {out['lna_reason']}")
    _echo(f"\n  → CALL: {out['call'].upper()}")
    _echo(f"     {out['reason']}")
    _echo(
        "\n  note: near the fold this is a ONE-SIDED LOWER BOUND — the noise model is "
        "\n  weakest exactly at the fold; NUDGE abstains on the deep-basin side "
        "(NUDGE-LIM-012)."
    )


# --------------------------------------------------------------------------- #
# design (inverse / intervention — turn a diagnosis into a prescription)
# --------------------------------------------------------------------------- #
@app.command()
def design(
    path: str = typer.Argument(
        "", help="a dose-response CSV/TSV or .h5ad screen (curve mode); omit for a "
        "parametric circuit (--topology)"
    ),
    target_response: float = typer.Option(
        float("nan"), "--target-response", "-y",
        help="curve mode: the target readout y to reach (invert the Hill to a dose)",
    ),
    topology: str = typer.Option(
        "", help=f"circuit mode: a bistable motif ({', '.join(_TOPOLOGIES)}) to flip"
    ),
    to: str = typer.Option("high", help="circuit mode: target basin ('high' | 'low')"),
    start: str = typer.Option("low", help="circuit mode: start basin ('low' | 'high')"),
    knob: list[str] = typer.Option(
        [], "--knob", help="circuit: an addressable knob to move (repeatable, e.g. "
        "species0.basal, edge0.K); default = the full set"
    ),
    n: float = typer.Option(6.0, "--n", help="circuit: switch cooperativity (Hill n)"),
    k: float = typer.Option(1.0, "--k", help="circuit mode: switch threshold (K)"),
    vmax: float = typer.Option(2.0, "--vmax", help="circuit: switch ceiling (v_max)"),
    basal: float = typer.Option(0.05, "--basal", help="circuit mode: basal production"),
    direction: str = typer.Option(
        "repress", help="curve mode: 'repress' (response falls with dose) or 'activate'"
    ),
    dose_col: str = typer.Option("dose", help="curve CSV dose column"),
    response_col: str = typer.Option("response", help="curve CSV response column"),
    target: str = typer.Option("", "--target", "-t", help="h5ad: guide-group prefix"),
    target_gene: str = typer.Option("", help="h5ad: gene whose knockdown is the dose"),
    signature: str = typer.Option("", help="h5ad: comma-separated readout genes"),
    steps: int = typer.Option(400, help="circuit mode: inversion steps"),
) -> None:
    """Propose an untested intervention that reaches a target — the inverse verb.

    Two modes. **Curve mode** (real data): give a dose-response ``path`` + a
    ``--target-response y`` — NUDGE inverts the fitted Hill to the dose achieving ``y``,
    behind the integrity gate (won't invert an abstained fit) with a reachability
    abstention when ``y`` is out of range (no safety gate — no circuit/fold). **Circuit
    mode**: give ``--topology`` — NUDGE gradient-inverts the bistable circuit to flip it
    to a basin and runs the Cap-5 **safety gate**, flagging an intervention that pushes
    the switch toward / over its fold (HIGH RISK OF INSTABILITY). Never designs off an
    unreliable fit; proposals are valid only within the fitted region (NUDGE-LIM-013).
    """
    from nudge.service import ROBUSTNESS_TOPOLOGIES, design_circuit, design_file

    if topology:
        if topology not in ROBUSTNESS_TOPOLOGIES:
            raise typer.BadParameter(f"topology must be one of {ROBUSTNESS_TOPOLOGIES}")
        out = design_circuit(
            topology, n=n, k=k, vmax=vmax, basal=basal, to=to, start=start,
            free=list(knob) or None, steps=steps,
        )
    elif path and target_response == target_response:  # a real (non-NaN) target
        sig = [g.strip() for g in signature.split(",") if g.strip()]
        out = design_file(
            path, target_response=target_response, direction=direction,
            dose_col=dose_col, response_col=response_col, target=target or None,
            target_gene=target_gene or None, signature=sig or None,
        )
    else:
        raise typer.BadParameter(
            "give either --topology (circuit mode) or a PATH + --target-response "
            "(curve mode)"
        )

    if out["kind"] == "abstention":
        _echo(f"design → ABSTAIN ({out['verdict']})")
        _echo(f"  {out['reason']}")
        return
    if out["mode"] == "dose":
        _echo(f"design (curve mode)  attribution={out.get('attribution_call', '?')}")
        _echo(
            f"  → DOSE = {out['dose']:.3g}  to reach response "
            f"y = {out['target_response']:.3g}"
        )
        _echo(f"  {out['reason']}")
        return
    _echo(f"design (circuit)  topology={out['topology']}  flip->{out['target_basin']}")
    _echo("  proposed intervention (ranked knobs):")
    for d in out["deltas"]:
        p = d["param"]
        _echo(f"    {p['scope']}[{p['index']}].{p['name']}  ×{d['factor']:.3f}")
    s = out["safety"]
    if s is None:
        _echo("  safety: n/a")
    elif s["crosses_fold"]:
        _echo("  -> SAFETY: HIGH RISK OF INSTABILITY — CROSSES THE FOLD "
              "(the switch loses bistability; NUDGE-LIM-013)")
    elif s["high_risk_of_instability"]:
        bound = " (one-sided LOWER bound)" if s["one_sided"] else ""
        _echo(
            f"  → SAFETY: HIGH RISK — pushes toward the fold "
            f"(prox {s['proximity_before']:.2f}->{s['proximity_after']:.2f}{bound}; "
            "NUDGE-LIM-013)"
        )
    else:
        _echo(
            f"  → SAFETY: OK — stays away from the fold "
            f"(proximity {s['proximity_before']:.2f}→{s['proximity_after']:.2f})"
        )
    _echo(f"\n  {out['reason']}")


# --------------------------------------------------------------------------- #
# mechanisms
# --------------------------------------------------------------------------- #
@app.command()
def mechanisms() -> None:
    """List the registered mechanism library and whether each has a Mechanism Card."""
    from nudge.knowledge import list_mechanisms

    rows = list_mechanisms()
    _echo(f"{len(rows)} registered mechanisms:\n")
    _echo(f"  {'ID':<14} {'REGISTRY':<18} {'ROLE':<16} CARD")
    for m in rows:
        card = m["card"] or "—"
        _echo(
            f"  {m['algorithm_id'] or '—':<14} {m['registry_name']:<18} "
            f"{m['role'] or '—':<16} {card}"
        )
    _echo("\nSee a card:  nudge explain <mechanism>   (e.g. hill_activation)")


# --------------------------------------------------------------------------- #
# explain
# --------------------------------------------------------------------------- #
@app.command()
def explain(
    query: str = typer.Argument(
        ...,
        help="an abstention (off-model, unresolved, no-effect, technical-artifact), "
        "a decoy id (NUDGE-DECOY-001), a limitation id (NUDGE-LIM-006), or a "
        "mechanism name (hill_activation).",
    ),
) -> None:
    """Explain a verdict, decoy, limitation, or mechanism (the 'why abstain?' verb)."""
    from nudge.knowledge import explain as _explain

    result = _explain(query)
    kind = result["kind"]

    if kind == "abstention":
        _echo(f"abstention: {result['verdict']}\n")
        _echo(result["meaning"])
        if result["decoys"]:
            _echo("\ndecoys that pin this failure mode:")
            for d in result["decoys"]:
                ref = d["limitation_ref"] or "—"
                _echo(f"  {d['decoy_id']}  ({ref})  {d['summary']}")
        if result["limitations"]:
            _echo("\ndocumented limitations:")
            for lim in result["limitations"]:
                _echo(f"  {lim['anomaly_id']}: {lim.get('title', '')}")
        if result["cards"]:
            _echo(f"\nread the mechanism card(s): {', '.join(result['cards'])}")
    elif kind == "attribution":
        _echo(f"attribution: {result['verdict']}\n{result['meaning']}")
    elif kind == "decoy":
        _echo(f"{result['decoy_id']}  (expected: {result['expected_verdict']})\n")
        _echo(result["summary"])
        lim = result.get("limitation")
        if lim:
            _echo(f"\nlimitation {lim['anomaly_id']}: {lim.get('title', '')}")
    elif kind == "limitation":
        _echo(f"{result['anomaly_id']}: {result.get('title', '')}\n")
        _echo(str(result.get("description", "")).strip())
        _echo(f"\nseverity: {result.get('severity', '?')}")
    elif kind == "mechanism_card":
        _echo(result["markdown"])
    else:  # unknown
        _echo(f"unrecognised query: {result['query']}")
        _echo("try one of:")
        for s in result.get("suggestions", []):
            _echo(f"  {s}")
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# multi-reporter (several reporters of ONE latent switch, fit jointly)
# --------------------------------------------------------------------------- #
@app.command("multi-reporter")
def multi_reporter(
    path: str = typer.Argument(
        ..., help="a tidy long CSV/TSV: reporter, dose, control, perturbed columns"
    ),
    dose_col: str = typer.Option("dose", "--dose-col", help="dose column"),
    reporter_col: str = typer.Option(
        "reporter", "--reporter-col", help="reporter-label column"
    ),
    control_col: str = typer.Option(
        "control", "--control-col", help="control/WT response column"
    ),
    perturbed_col: str = typer.Option(
        "perturbed", "--perturbed-col", help="perturbed response column"
    ),
    direction: str = typer.Option(
        "activate", help="'activate' (readout rises with dose) or 'repress'"
    ),
    n_boot: int = typer.Option(200, help="bootstrap resamples for the shared-knob CIs"),
    seed: int = typer.Option(0, help="bootstrap RNG seed"),
) -> None:
    """Jointly attribute a panel of reporters of ONE latent switch — break K⇄v_max.

    Several downstream reporters of the *same* latent switch, each with its own gain /
    offset, are fit **jointly**: the panel over-determines the latent, so it resolves
    **threshold** (K) vs **gain** (n) vs **ceiling** (v_max) where a single reporter is
    degenerate and abstains (the measured K⇄v_max degeneracy, FINDINGS §2). Abstains
    **off-model** when the reporters cannot be explained by one shared latent (a
    reporter reads a different latent — NUDGE-LIM-014), not average it into a call.
    """
    from nudge.service import multi_reporter_file

    out = multi_reporter_file(
        path,
        dose_col=dose_col,
        reporter_col=reporter_col,
        control_col=control_col,
        perturbed_col=perturbed_col,
        direction=direction,
        n_boot=n_boot,
        seed=seed,
    )
    klo, khi = out["ci_log2_k"]
    _echo(
        f"multi-reporter  ({out['n_reporters']} reporters, dir={out['direction']}, "
        f"panel R²={out['panel_r2']:.2f})"
    )
    _echo(f"  shared latent (WT):  K = {out['k_wt']:.3g}   n = {out['n_wt']:.2f}")
    _echo(
        f"  restricted losses:  threshold={out['losses']['threshold']:.3g}  "
        f"gain={out['losses']['gain']:.3g}  ceiling={out['losses']['ceiling']:.3g}  "
        f"(no-effect={out['losses']['no_effect']:.3g})"
    )
    _echo(
        f"  winner={out['winner']}  knob margin ×{out['knob_margin']:.2f}  "
        f"effect margin ×{out['effect_margin']:.2f}  "
        f"consistency ratio {out['consistency_ratio']:.1f}"
    )
    _echo("  per-reporter (gain, shared-R², own-R²):")
    for r in out["reporters"]:
        _echo(
            f"    {r['name']:6} gain={r['gain']:6.2f}  shared R²={r['r2_shared']:.2f}  "
            f"own R²={r['r2_independent']:.2f}"
        )
    _echo(f"\n  → CALL: {out['call'].upper()}")
    _echo(f"     {out['reason']}")
    _echo(
        "\n  note: a single reporter of one latent is degenerate (K⇄v_max); the JOINT "
        "\n  panel resolves it, and abstains OFF-MODEL on an inconsistent panel "
        "(NUDGE-LIM-014)."
    )


# --------------------------------------------------------------------------- #
# diagnose-abstention (hidden-node ABSTENTION — why is the model inadequate?)
# --------------------------------------------------------------------------- #
@app.command("diagnose-abstention")
def diagnose_abstention(
    off_model: bool = typer.Option(
        True, "--off-model/--adequate",
        help="the parsimony gate returned off-model (default) vs the model is adequate",
    ),
    neomorphic_ratio: float = typer.Option(
        float("nan"), "--neomorphic-ratio",
        help="the off-axis / on-axis residual ratio (ComboGeometry); omit if unmeasured",
    ),
    readout_flag: bool = typer.Option(
        False, "--readout-flag", help="the reporter may be nonlinear (NUDGE-LIM-006)"
    ),
    perturbation_residual: float = typer.Option(
        float("nan"), "--perturbation-residual",
        help="the best restricted-fit absolute residual (off-target evidence)",
    ),
    topology_uncertain: bool = typer.Option(
        False, "--topology-uncertain", help="the fitted topology may be wrong (T0.5-2)"
    ),
    depth_confounded: bool = typer.Option(
        False, "--depth-confounded", help="a depth/batch difference aligns with the condition"
    ),
) -> None:
    """Diagnose *why* an attribution is inadequate — the honest hidden-node abstention.

    Turns a bare **off-model** verdict (or a fired diagnostic residual) into a legible
    **differential diagnosis**: it ENUMERATES the candidate causes — not-a-switch,
    nonlinear readout, off-target, wrong topology, batch/depth confound, and a hidden
    node — each with its evidence, documented limitation, and the experiment that would
    distinguish it. **It never asserts a hidden node** (the causes are observationally
    overlapping): the strongest it says is that an off-axis residual is *consistent with,
    does not prove* an unmeasured regulator (NUDGE-LIM-015). Abstention half ONLY.
    """
    import math

    from nudge.service import diagnose_abstention as _diagnose

    out = _diagnose(
        off_model=off_model,
        neomorphic_ratio=None if math.isnan(neomorphic_ratio) else neomorphic_ratio,
        readout_flag=readout_flag,
        perturbation_residual=(
            None if math.isnan(perturbation_residual) else perturbation_residual
        ),
        topology_uncertain=topology_uncertain,
        depth_confounded=depth_confounded,
    )
    if out["is_adequate"]:
        _echo("model adequate — no inadequacy to explain, no differential emitted.")
        _echo(f"  {out['reason']}")
        return
    _echo(f"model INADEQUATE  (verdict: {out['verdict']})\n")
    _echo(out["reason"])
    _echo(f"\ndifferential — {len(out['causes'])} candidate causes (ranked hypotheses):")
    for i, c in enumerate(out["causes"], 1):
        refs = "  ".join(x for x in (c["limitation_ref"], c["decoy_ref"]) if x) or "—"
        _echo(f"\n  {i}. [{c['qualitative_rank']}] {c['name']}   ({refs})")
        _echo(f"     evidence:   {c['evidence']}")
        _echo(f"     distinguish: {c['distinguishing_experiment']}")
    _echo(
        "\n  note: this is a DIFFERENTIAL, not a verdict. NUDGE ships only the abstention "
        "\n  half — it NEVER positively asserts a hidden node (NUDGE-LIM-015). Resolve any "
        "\n  NUDGE-LIM-* / NUDGE-DECOY-* above with:  nudge explain <id>"
    )


# --------------------------------------------------------------------------- #
# differential (the SAME perturbation in two contexts — which knob differs?)
# --------------------------------------------------------------------------- #
@app.command("differential")
def differential(
    path: str = typer.Argument(
        ..., help="a .npz with data_a/control_a/data_b/control_b activity arrays"
    ),
    circuit: str = typer.Option(
        "ras_switch_1node", help="the shared switch motif (nudge.circuits factory)"
    ),
    n: float = typer.Option(6.0, help="nominal Hill n (gain) of the switch"),
    vmax: float = typer.Option(2.5, help="nominal v_max (ceiling)"),
    k: float = typer.Option(1.0, "--k", help="nominal K (threshold)"),
    basal: float = typer.Option(0.2, help="nominal basal (OFF level)"),
    target_edge: int = typer.Option(0, help="the attributable edge index"),
    steps: int = typer.Option(250, help="optimizer steps per nested-model fit"),
    n_boot: int = typer.Option(0, help="bootstrap resamples for the winning-knob CI"),
    seed: int = typer.Option(0, help="fit RNG seed"),
) -> None:
    """Isolate WHICH knob differs for the SAME perturbation across two contexts.

    Given the same perturbation in two **contexts** (resistant vs sensitive line; donor A
    vs B; disease vs healthy) as four activity arrays, fits the shared switch **jointly**
    and BIC-selects which single knob must differ — **threshold** (K) / **gain** (n) /
    **ceiling** (v_max) — or abstains (**no-difference** / **unresolved**). A raised
    *ceiling* means more dose of the SAME drug; a rewired *gain/threshold* means a
    DIFFERENT class — a call linear differential expression cannot make. **Confound
    guard:** depth is pinned per context from each control, and a ceiling call corrupted
    by a depth/batch shift aligned with the context axis abstains (NUDGE-LIM-016).
    """
    from nudge.service import differential_file

    out = differential_file(
        path,
        circuit=circuit,
        n=n,
        vmax=vmax,
        k=k,
        basal=basal,
        target_edge=target_edge,
        steps=steps,
        n_boot=n_boot,
        seed=seed,
    )
    bic = out["bic"]
    best_fin = min(v for v in bic.values() if v == v and v != float("inf"))
    _echo(
        f"differential  (edge {out['target_edge']}, "
        f"n_a={out['n_cells']['a']}, n_b={out['n_cells']['b']})"
    )
    _echo("  ΔBIC vs best (lower = more parsimonious):")
    labels = {"shared": "shared (no diff)", "K": "ΔK (threshold)",
              "n": "Δn (gain)", "vmax": "Δv_max (ceiling)"}
    for m in ("shared", "n", "K", "vmax"):
        v = bic[m]
        shown = "inf" if (v != v or v == float("inf")) else f"{v - best_fin:+8.1f}"
        _echo(f"    {labels[m]:20} {shown}")
    d = out["depth"]
    o = out["off_baseline_shift"]
    _echo(
        f"  per-context depth: scale_a={d['scale_a']:.1f} scale_b={d['scale_b']:.1f} "
        f"(ratio {d['depth_ratio']:.2f})   OFF-shift ratio {o['ratio']:.2f}"
    )
    _echo(f"\n  → CALL: {out['call'].upper()}")
    _echo(f"     {out['reason']}")
    _echo(
        "\n  note: a depth/batch shift aligned with the context axis mimics a CEILING "
        "\n  difference; NUDGE pins depth per context + guards the ceiling call "
        "(NUDGE-LIM-016)."
    )


@app.command()
def constitutive(
    path: str = typer.Argument(
        "",
        help="a .npz with population/control_activity/control_response arrays "
        "(omit with --demo to synthesize)",
    ),
    demo: bool = typer.Option(
        False, "--demo", help="synthesize a matched population + control (no data file)"
    ),
    circuit_n: float = typer.Option(
        3.0, "--circuit-n", help="demo/seed ground-truth circuit Hill n (1 = LIM-006 hazard)"
    ),
    readout_h: float = typer.Option(6.0, "--readout-h", help="demo reporter Hill h"),
    steps: int = typer.Option(600, help="optimizer steps per profile point"),
    restarts: int = typer.Option(3, help="restarts per profile point"),
    seed: int = typer.Option(0, help="fit RNG seed"),
) -> None:
    """Separate CIRCUIT ultrasensitivity from a NONLINEAR READOUT (the NUDGE-LIM-006 fix).

    A nonlinear reporter over a linear circuit fools the affine-readout fit into a
    CONFIDENT false switch (NUDGE-LIM-006). A **constitutive-reporter control** — the
    reporter driven at KNOWN activity doses, bypassing the circuit — anchors the readout
    (using READOUT parameters ONLY, no circuit leak). NUDGE then profiles the circuit
    Hill ``n``: WITHOUT the control the profile is FLAT (you cannot tell a switch exists);
    WITH it, "no switch" (n=1) is REJECTED for a genuine switch → **biological-switch**, or
    the profile stays flat for a linear circuit → **unresolved** (honest abstention).
    **Fail-safe:** it NEVER emits a bare threshold/gain/ceiling, and it does NOT
    point-identify n (needs a second anchor; NUDGE-LIM-018). Use ``--demo`` for a
    zero-setup run.
    """
    from nudge.service import constitutive_demo, constitutive_file

    if demo or not path:
        out = constitutive_demo(
            circuit_n=circuit_n, readout_h=readout_h, steps=steps, restarts=restarts, seed=seed
        )
        gt = out["ground_truth"]
        _echo(
            f"constitutive (DEMO: true circuit n={gt['circuit_n']:g}, "
            f"reporter h={gt['reporter_h']:g})"
        )
    else:
        out = constitutive_file(
            path, circuit_n=circuit_n, h=readout_h, steps=steps, restarts=restarts, seed=seed
        )
        _echo("constitutive")
    cal = out["calibration"]
    _echo(
        f"  calibrated reporter Hill h = {cal['reporter_hill_h']:.2f} "
        f"(95% CI {cal['ci_h'][0]:.2f}-{cal['ci_h'][1]:.2f}, "
        f"nonlinear={cal['is_nonlinear']})"
    )
    _echo(
        f"  WITHOUT control: n-profile span = {out['span_no_control']:.5f}  (flat => degenerate)"
    )
    _echo(
        f"  WITH    control: n=1 rejection  = {out['n1_rejection']:.5f}  "
        f"(argmin n≈{out['argmin_n_with_control']:g})"
    )
    _echo(f"\n  → CALL: {out['call'].upper()}   (confident-wrong={out['confident_wrong']})")
    _echo(f"     {out['reason']}")


@app.command("lotka")
def lotka(
    mechanism: str = typer.Option(
        "susceptibility",
        help="the KNOWN perturbation: growth | interaction | susceptibility | none",
    ),
    near_equilibrium: bool = typer.Option(
        False, "--near-equilibrium/--dense-transient",
        help="sample near equilibrium (the degenerate α⇄βᵢᵢ regime) vs the dense transient",
    ),
    n_species: int = typer.Option(3, help="number of taxa in the community"),
    n_replicates: int = typer.Option(60, help="replicate communities per group"),
    steps: int = typer.Option(250, help="optimizer steps per restricted-model fit"),
    seed: int = typer.Option(0, help="fit / simulation RNG seed"),
) -> None:
    """Temporal / gLV attribution — which knob did a community perturbation move?

    A synthetic, zero-setup demo of NUDGE pointed at a NEW dynamical-systems domain
    (microbiome ecology; ``NUDGE-METHOD-012``). Simulates a reference vs perturbed
    community under an antibiotic pulse with a KNOWN single-knob change, then attributes
    it to **growth (α)** / **interaction (β)** / **susceptibility (ε)** — or abstains.
    The **ε** axis is the identifiable positive; a near-equilibrium **growth** change is
    the degenerate **α⇄βᵢᵢ** case NUDGE must abstain on, with the degeneracy MEASURED by
    the Laplace curvature (``NUDGE-LIM-020``). Fail-safe: recover-or-abstain, never a
    confident wrong knob.
    """
    from nudge.service import lotka_demo

    out = lotka_demo(
        mechanism=mechanism,
        dense_transient=not near_equilibrium,
        n_species=n_species,
        n_replicates=n_replicates,
        steps=steps,
        seed=seed,
    )
    gt = out["ground_truth"]
    ident = out["identifiability"]
    _echo(
        f"gLV attribution  (taxa={n_species}, replicates={out['n_replicates']}, "
        f"timepoints={out['n_timepoints']})"
    )
    _echo(
        f"  ground truth: moved '{gt['mechanism']}' on taxon {gt['target']} "
        f"(Δ={gt['delta']:+.2f}); sampling="
        f"{'dense-transient' if gt['dense_transient'] else 'near-equilibrium'}"
    )
    bic = out["bic"]
    best = min(bic.values())
    _echo("  ΔBIC vs best (lower = more parsimonious):")
    for m in ("null", "growth", "interaction", "susceptibility"):
        _echo(f"    {m:16} {bic[m] - best:+8.1f}")
    _echo(
        f"  α⇄βᵢᵢ identifiability: condition number {ident['cond_number']:.0f}, "
        f"|corr|={ident['abs_corr_alpha_beta']:.3f}, degenerate={ident['degenerate']}"
    )
    _echo(f"\n  → CALL: {out['call'].upper()}")
    _echo(f"     {out['reason']}")
    _echo(
        "\n  note: gLV inference is ill-posed — abstaining is on-thesis. The ε axis is the "
        "\n  identifiable one; α vs βᵢᵢ (self-limitation) is degenerate near equilibrium "
        "(NUDGE-LIM-020)."
    )


def main() -> None:
    """Console-script entry point (kept for parity; ``app`` is the real entry)."""
    app()


if __name__ == "__main__":
    app()
