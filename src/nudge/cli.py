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
) -> None:
    """Attribute a mechanism from a dose-response curve — switch vs graded, or abstain.

    The same K (threshold) / n (gain) / v_max (ceiling) vocabulary as single-cell
    attribution, read from a dose axis instead (two measurements of one circuit).
    Reports
    ``n`` as an **apparent population gain** with a CI, and abstains (``unresolved`` /
    ``no-effect``) rather than over-call an unidentifiable curve — e.g. when the doses
    do
    not span the inflection.
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


def main() -> None:
    """Console-script entry point (kept for parity; ``app`` is the real entry)."""
    app()


if __name__ == "__main__":
    app()
