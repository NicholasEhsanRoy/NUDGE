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

    @mcp.tool()
    def dose_response(
        path: str,
        direction: str = "repress",
        dose_col: str = "dose",
        response_col: str = "response",
        target: str = "",
        target_gene: str = "",
        signature: str = "",
        group_col: str = "guide",
        control: str = "WT",
        min_cells: int = 15,
        n_boot: int = 500,
    ) -> dict[str, Any]:
        """Attribute a mechanism from a dose-response curve: switch/graded or abstain.

        The same K/n/v_max vocabulary as single-cell ``attribute``, read from a dose
        axis (two measurements of one circuit). ``path`` is a 2-column CSV/TSV
        (``dose_col`` / ``response_col``) or an ``.h5ad`` knockdown screen — for an
        ``.h5ad`` give ``target`` (guide-group prefix, e.g. ``OCT4``), ``target_gene``
        (the gene whose knockdown is the dose, e.g. ``POU5F1``), and ``signature``
        (comma-separated readout genes). ``direction`` is ``repress`` when the response
        falls with dose. Returns the verdict (``switch`` / ``graded`` / ``no-effect`` /
        ``unresolved``) with the apparent population gain ``n`` + CI and the honest
        reason — it abstains on an unidentifiable curve (e.g. doses not spanning the
        inflection) rather than force a call. ``n`` is an apparent gain, not molecular
        cooperativity.
        """
        from nudge.service import dose_response_file

        sig = [g.strip() for g in signature.split(",") if g.strip()]
        return dose_response_file(
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
        )

    @mcp.tool()
    def synergy(
        path: str,
        a_label: str,
        b_label: str,
        ab_label: str,
        control_label: str = "control",
        condition_col: str = "condition",
        signature: str = "",
        n_top_genes: int = 2000,
        n_boot: int = 1000,
        min_cells: int = 30,
    ) -> dict[str, Any]:
        """Classify a two-perturbation combination: additive vs synergistic/buffering.

        Reads {control, A, B, A+B} from an ``.h5ad`` (by ``condition_col`` labels
        ``control_label`` / ``a_label`` / ``b_label`` / ``ab_label``) as three operating
        points against a shared control, reduces each to a scalar **effect** in
        log-fold-change space (the additive null is Bliss independence), and returns the
        **interaction** ``effect(A+B) − [effect(A)+effect(B)]`` with a bootstrap CI. By
        default the per-cell score projects onto the additive axis fixed by the two
        single arms (direction-safe; pass ``signature`` for a fixed gene set instead).
        Returns the verdict (``additive`` / ``synergistic`` / ``buffering`` /
        ``no-effect`` / ``unresolved``) with the honest reason — it abstains when an arm
        is underpowered or the CI is too wide rather than force a call. A super-additive
        residual is NOT by itself a hidden-node claim (NUDGE-LIM-009).
        """
        from nudge.service import synergy_file

        sig = [g.strip() for g in signature.split(",") if g.strip()]
        return synergy_file(
            path,
            control_label=control_label,
            a_label=a_label,
            b_label=b_label,
            ab_label=ab_label,
            condition_col=condition_col,
            signature=sig or None,
            n_top_genes=n_top_genes,
            n_boot=n_boot,
            min_cells=min_cells,
        )

    @mcp.tool()
    def cross_modality(
        path: str,
        dose_col: str,
        response_col: str,
        variant_col: str,
        control_variant: str,
        class_col: str = "",
        modality: str = "fluorescence",
        direction: str = "activate",
        filters: dict[str, Any] | None = None,
        n_boot: int = 400,
    ) -> dict[str, Any]:
        """Attribute a panel of CONTINUOUS-readout dose-responses (cross-modality tool).

        Runs the *same* K (threshold) / n (gain) / v_max (ceiling) attribution NUDGE
        does on counts, but on a **continuous single channel** — flow fluorescence,
        an activity reporter, or a fold-change summary — read from a tidy CSV/TSV
        (``dose_col`` / ``response_col`` / ``variant_col``). The ``modality``
        (``fluorescence`` / ``activity`` / ``foldchange``) is **declared, never
        guessed**: the bouncer refuses log-normalized or raw counts masquerading as
        fluorescence (NUDGE-LIM-008). Each variant's curve is fit + classified with the
        shipped dose-response path and localized to one knob vs ``control_variant`` —
        **threshold** (dose-EC50 shift) / **gain** (Hill steepness) / **ceiling**
        (leakiness / dynamic range) — or abstains (**non-responsive** /
        **inconclusive**). ``filters`` (e.g. ``{"operator": "O2"}``) pins other axes;
        ``class_col`` carries a ground-truth label. ``direction`` is ``activate`` when
        the readout rises with dose (induction). This is the Chure-2019 LacI benchmark:
        DNA-binding-domain mutants localize to ceiling/leakiness, inducer-binding-domain
        mutants to threshold. Returns the honest per-variant table.
        """
        from nudge.service import cross_modality_panel_file

        return cross_modality_panel_file(
            path,
            dose_col=dose_col,
            response_col=response_col,
            variant_col=variant_col,
            control_variant=control_variant,
            class_col=class_col or None,
            filters=filters,
            modality=modality,
            direction=direction,
            n_boot=n_boot,
        )

    @mcp.tool()
    def robustness(
        topology: str = "1node",
        n: float = 6.0,
        K: float = 1.0,
        vmax: float = 2.0,
        basal: float = 0.05,
        path: str = "",
    ) -> dict[str, Any]:
        """How close is a bistable switch to LOSING bistability (a saddle-node fold)?

        Returns a **robustness dial** for a bistable motif (``topology``: ``1node`` /
        ``2node`` / ``toggle``) at the given switch kinetics (``n`` gain, ``K``
        threshold, ``vmax`` ceiling, ``basal``): the fused 0..1 ``proximity`` plus three
        raw channels — **critical slowing** (``min|Re λ|`` → 0), **basin collapse**
        (node→saddle → 0), and **LNA lobe swell** (lobe ratio → 1). The ``call`` is
        ``near-fold`` / ``robust`` / ``unresolved`` (deep-basin abstention) /
        ``not-bistable``. **Honesty (load-bearing):** near the fold the number is a
        **ONE-SIDED LOWER BOUND** (``one_sided``) — the linear-noise Gaussian breaks
        down precisely at the fold (a mode's variance
        diverges), so it is least reliable exactly there (NUDGE-LIM-012); NUDGE abstains
        (``unresolved``) on the deep-basin side rather than emit a false-precise "far"
        number. Give ``path`` (an ``(n_cells, n_species)`` activity ``.npy``/CSV/TSV) to
        also calibrate the sequencing depth from data and report the LNA lobe channel's
        reliability at that depth. This is the same score the future ``design()`` safety
        gate uses to flag an intervention that pushes a switch toward a tipping point.
        """
        from nudge.service import bifurcation_file, robustness_circuit

        if path:
            return bifurcation_file(
                path, topology=topology, n=n, k=K, vmax=vmax, basal=basal
            )
        return robustness_circuit(topology, n=n, k=K, vmax=vmax, basal=basal)

    return mcp


def main() -> None:
    """Console-script entry point (``nudge-mcp``): run the server over stdio.

    Warms the JAX compile caches before serving so the first ``attribute`` /
    ``dose_response`` tool call a client makes is already fast (the server is a
    long-lived process — see :func:`nudge.warmup.warmup`).
    """
    from nudge.warmup import warmup

    warmup()
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
