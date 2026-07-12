"""NUDGE MCP server — exposes NUDGE to a Claude client as tools.

A thin FastMCP (``stdio``) adapter over the tested engine + the Mechanism-Card
knowledge base, so Claude (Claude Code / Desktop / the Claude Science workbench)
gets exactly the honest, abstaining output a human gets from the ``nudge`` CLI —
including *why* NUDGE abstained. See ``design/INTEGRATION_FEASIBILITY.md`` for the
verified connection recipes.

Tools:
- ``attribute(h5ad_path, target, ...)`` — run covariance attribution at one
  operating point; returns the call(s) + honest skip/abstention reasons.
- ``design(...)`` — invert a reliable attribution to PROPOSE an intervention (a dose,
  or a kinetic Δ) reaching a target, behind an integrity + a bifurcation safety gate.
- ``explain_abstention(context)`` — pull the Mechanism Card + decoy/limitation that
  explains a verdict (``off-model`` / ``unresolved`` / a decoy or limitation id).
- ``diagnose_abstention(off_model, ...)`` — turn a bare off-model verdict into a legible
  DIFFERENTIAL of candidate causes (never a positive hidden-node claim; NUDGE-LIM-015).
- ``constitutive(...)`` — separate CIRCUIT ultrasensitivity from a NONLINEAR READOUT with a
  constitutive-reporter control (the NUDGE-LIM-006 mitigation); reject "no switch" or abstain.
- ``render_figure(kind, ...)`` — render any NUDGE result to an honest figure (the abstention
  overlay is stamped off the result's own verdict); returns paths + a size-capped inline PNG.
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

    @mcp.tool()
    def design(
        path: str = "",
        target_response: float = float("nan"),
        topology: str = "",
        to: str = "high",
        start: str = "low",
        n: float = 6.0,
        K: float = 1.0,
        vmax: float = 2.0,
        basal: float = 0.05,
        free: str = "",
        direction: str = "repress",
        dose_col: str = "dose",
        response_col: str = "response",
        target: str = "",
        target_gene: str = "",
        signature: str = "",
    ) -> dict[str, Any]:
        """Propose an untested INTERVENTION that reaches a target — the inverse verb.

        NUDGE's headline: turn a *diagnosis* (which knob a switch turns) into a
        *prescription* by running the fit **backwards**, behind two honesty
        gates. Two modes:

        - **Curve mode** (real data): a dose-response ``path`` (a 2-column CSV/TSV or
          an ``.h5ad`` screen — for ``.h5ad`` set ``target`` / ``target_gene`` /
          ``signature`` as in ``dose_response``) plus ``target_response`` (the readout
          ``y`` to reach). NUDGE inverts the fitted Hill to the dose achieving ``y``,
          behind the **integrity gate** (it refuses to invert an ``unresolved`` /
          ``no-effect`` fit) with an honest **reachability abstention** when ``y`` is
          outside the curve's achievable ``(floor, floor+amp)`` range. Curve mode has
          **no** bifurcation safety gate — there is no circuit/fold (stated honestly).
        - **Circuit mode**: give ``topology`` (``1node`` / ``2node`` / ``toggle``) + the
          switch kinetics (``n`` gain, ``K`` threshold, ``vmax`` ceiling, ``basal``).
          NUDGE gradient-inverts the circuit to flip it ``to`` a basin (from the
          ``start`` basin) over its addressable kinetic knobs — restrict which knobs it
          may move with ``free`` (comma-separated ``edge0.K`` / ``species0.basal`` names;
          e.g. ``free="species0.basal"`` asks ONLY "what change to Gene A's basal rate
          flips it?"; each returned Δ carries a multiplicative ``factor``, so e.g. a
          ``factor`` of 0.5 = a 50% reduction, 1.5 = a 50% increase). Then runs the **Cap-5
          bifurcation safety gate**: it flags an intervention that pushes the switch
          toward / over its fold (``crosses_fold`` / ``high_risk_of_instability`` — the
          proximity is a ONE-SIDED LOWER BOUND near the fold; ``NUDGE-LIM-012``). It
          (reachability) if no intervention reaches the target within the fitted region.

        Returns the ``InterventionPlan`` (mode, ranked ``deltas`` or ``dose``, the
        ``safety`` report) or an ``abstention`` + reason. Every proposal is valid
        only within the fit's identifiable region — extrapolation is flagged
        (``NUDGE-LIM-013``). Never designs off a fit it does not trust.
        """
        from nudge.service import design_circuit, design_file

        if topology:
            knobs = [s.strip() for s in free.split(",") if s.strip()] or None
            return design_circuit(
                topology, n=n, k=K, vmax=vmax, basal=basal, to=to, start=start, free=knobs
            )
        sig = [g.strip() for g in signature.split(",") if g.strip()]
        return design_file(
            path,
            target_response=target_response,
            direction=direction,
            dose_col=dose_col,
            response_col=response_col,
            target=target or None,
            target_gene=target_gene or None,
            signature=sig or None,
        )

    @mcp.tool()
    def multi_reporter(
        path: str,
        dose_col: str = "dose",
        reporter_col: str = "reporter",
        control_col: str = "control",
        perturbed_col: str = "perturbed",
        direction: str = "activate",
        n_boot: int = 200,
    ) -> dict[str, Any]:
        """Jointly attribute a panel of reporters of ONE latent switch — break K⇄v_max.

        NUDGE's dominant reason to abstain is the **K⇄v_max / gain⇄threshold
        degeneracy**: a *single* reporter of one latent switch under-determines the
        mechanism (``FINDINGS`` §2). The fix is to fit **several downstream reporters of
        the SAME latent switch jointly** — each an affine readout ``y_j = base_j +
        gain_j·activity`` with its own heterogeneous gain. Because a threshold shift
        (moves the inflection identically across reporters) and a ceiling change (scales
        every reporter's ON amplitude by the same fraction) project DIFFERENTLY onto a
        panel of heterogeneous gains, the joint fit is over-determined and **resolves**
        threshold / gain / ceiling where a single reporter abstains.

        ``path`` is a tidy long CSV/TSV — one row per reporter × dose — with
        ``reporter_col`` / ``dose_col`` / ``control_col`` (WT response) /
        ``perturbed_col`` (perturbed response). ``direction`` is ``activate`` when the
        readout rises with dose. Returns the verdict (``threshold`` / ``gain`` /
        ``ceiling`` / ``no-effect`` / ``unresolved`` / ``off-model``) with the shared
        latent's WT ``K`` / ``n``, the shared perturbation ratios + bootstrap CIs, the
        per-reporter fits, and the honest reason. **Fail-safe:** a spurious mechanism
        must be consistent across ALL reporters; a panel that cannot be explained by one
        shared latent (a reporter reads a *different* latent) abstains ``off-model``
        (NUDGE-LIM-014) rather than being averaged into a confident call.
        """
        from nudge.service import multi_reporter_file

        return multi_reporter_file(
            path,
            dose_col=dose_col,
            reporter_col=reporter_col,
            control_col=control_col,
            perturbed_col=perturbed_col,
            direction=direction,
            n_boot=n_boot,
        )

    @mcp.tool()
    def diagnose_abstention(
        off_model: bool = True,
        neomorphic_ratio: float = float("nan"),
        readout_flag: bool = False,
        perturbation_residual: float = float("nan"),
        topology_uncertain: bool = False,
        depth_confounded: bool = False,
    ) -> dict[str, Any]:
        """Diagnose WHY a NUDGE attribution is inadequate — the honest hidden-node abstention.

        Turns a bare ``off-model`` verdict (or a fired diagnostic residual) into a legible
        **differential diagnosis** that ENUMERATES the candidate causes — genuinely
        not-a-switch, a nonlinear readout (``NUDGE-LIM-006``), an off-target perturbation,
        a wrong/misspecified topology, a batch/depth confound (``NUDGE-LIM-003``), and a
        hidden node / unmeasured regulator (``NUDGE-LIM-009``) — each with its evidence,
        the documented limitation it maps to, and the experiment that would distinguish it.

        **Abstention half ONLY (load-bearing honesty, ``NUDGE-LIM-015``).** It NEVER
        positively asserts a hidden node: the causes are observationally overlapping, so
        the strongest it says is that an off-axis residual is *consistent with — but does
        not prove —* an unmeasured regulator. It consumes verdicts/evidence and never
        touches the fit. Pass the evidence you have: ``off_model`` (the parsimony gate),
        the ``neomorphic_ratio`` (off-axis residual, omit as NaN if unmeasured), a
        ``readout_flag``, a ``perturbation_residual``, ``topology_uncertain`` /
        ``depth_confounded``. With ``off_model=False`` and no residual it reports the model
        adequate and emits no differential.
        """
        import math

        from nudge.service import diagnose_abstention as _diagnose

        return _diagnose(
            off_model=off_model,
            neomorphic_ratio=None if math.isnan(neomorphic_ratio) else neomorphic_ratio,
            readout_flag=readout_flag,
            perturbation_residual=(
                None if math.isnan(perturbation_residual) else perturbation_residual
            ),
            topology_uncertain=topology_uncertain,
            depth_confounded=depth_confounded,
        )

    @mcp.tool()
    def differential(
        path: str,
        circuit: str = "ras_switch_1node",
        n: float = 6.0,
        vmax: float = 2.5,
        k: float = 1.0,
        basal: float = 0.2,
        target_edge: int = 0,
        steps: int = 250,
        n_boot: int = 0,
    ) -> dict[str, Any]:
        """Comparative attribution — WHICH knob differs for the SAME perturbation in TWO contexts.

        Given the SAME perturbation run in two **contexts** — a drug-resistant vs sensitive
        line, donor A vs B, disease vs healthy — isolate whether the mechanistic difference
        is in the switch's **threshold** (`K`), **gain** (`n`), or **ceiling** (`v_max`), a
        distinction linear differential expression structurally **cannot** make. A resistant
        line with a raised *ceiling* needs more dose of the SAME drug; one with a rewired
        *gain / threshold* needs a DIFFERENT drug class. Fits the shared switch **jointly**
        with a shared-vs-per-context parameter structure and **BIC-selects** which single
        knob must differ (`shared` / `ΔK` / `Δn` / `Δv_max`), or abstains.

        `path` is a `.npz` with four `(n_cells, n_species)` **activity-space** arrays:
        `data_a` / `control_a` (context A's perturbed cells + its own control) and
        `data_b` / `control_b`. The switch topology is `circuit` (a `nudge.circuits`
        factory) at nominal `n` / `vmax` / `k` / `basal`. Returns the verdict
        (`threshold-diff` / `gain-diff` / `ceiling-diff` / `no-difference` / `unresolved`),
        the per-model BIC, the winning knob's Δ estimate, per-context depth, and the
        confound diagnostics. **Fail-safe (`NUDGE-LIM-016`):** depth is pinned PER CONTEXT
        from each control, and a ceiling call corrupted by a depth/batch shift aligned with
        the context axis (the OFF baseline moved) abstains `unresolved` rather than emit a
        spurious ceiling difference; an underpowered / untrustworthy context abstains too.
        """
        from nudge.service import differential_file

        return differential_file(
            path,
            circuit=circuit,
            n=n,
            vmax=vmax,
            k=k,
            basal=basal,
            target_edge=target_edge,
            steps=steps,
            n_boot=n_boot,
        )

    @mcp.tool()
    def differential_robust(
        path: str,
        circuit: str = "ras_switch_1node",
        n: float = 6.0,
        vmax: float = 2.5,
        k: float = 1.0,
        basal: float = 0.2,
        steps: int = 250,
    ) -> dict[str, Any]:
        """ROBUST differential attribution — hardened against a per-condition technical confound.

        Same four-array `.npz` contract as `differential` (`data_a` / `control_a` / `data_b` /
        `control_b`, activity space), but hardened against the systemic failure mode the banded
        `differential` guard has **measured blind spots** for: a per-context **affine technical
        nuisance** (a scale/offset on ONE context's *perturbed* cells only — a batch / depth /
        capture-efficiency difference, its control clean) that **aliases onto a mechanism** (a
        scale looks like a raised ceiling `v_max`; an offset shifts the modes → threshold/gain).
        A naive differential-expression read — and the banded `differential` at some confound
        magnitudes — calls a confident mechanism difference where the truth is **no-difference**.

        This uses the **Earn-Guard**: it re-fits each context's apparent knob difference against a
        FREE per-context affine `(s, o)` and returns a positive `*-diff` ONLY if the biological
        knob **earns** its BIC parameter over that affine null, in both directions — otherwise it
        abstains (`no-difference` / `unresolved`). Because the whole affine confound family lies
        inside the free-affine null's span, it abstains on it **continuously** (no calibrated
        bands; proven 0/24 confident-wrong on the red-team P1/P4/P5 repros AT ADEQUATE `steps`).
        The abstention needs the null to be *optimized*: at too few `steps` (≲180) it can spuriously
        "earn" a knob on a multiplicative confound at some seeds — the default `steps=250` clears
        this. Slower than `differential` (fits a reference + two augmented models per direction).
        Returns the verdict, the knob it screened, the `earn` (profiled ΔBIC), and the nuisance
        `(s, o)`.
        """
        from nudge.service import differential_robust_file

        return differential_robust_file(
            path, circuit=circuit, n=n, vmax=vmax, k=k, basal=basal, steps=steps,
        )

    @mcp.tool()
    def lotka(path: str, target: int = -1, steps: int = 300) -> dict[str, Any]:
        """Fit a generalized-Lotka–Volterra community + report which knob moved AND identifiability.

        For time-series abundance data of an ecological community (or any gLV system)
        `dxᵢ/dt = xᵢ(αᵢ + Σⱼ βᵢⱼxⱼ + εᵢ·u(t))`, `path` is an `.npz` with `reference` /
        `perturbed` `(R, T, S)` replicate ensembles + `t_obs` / `u_grid` / `obs_idx` / `dt`.
        NUDGE re-fits the community and attributes which single knob a perturbation moved —
        **growth (α) / interaction (β) / susceptibility (ε)** — OR abstains.

        **The point for a naive "just fit it and give me the interaction parameters β" request:**
        gLV parameters are a canonical **sloppy / near-equilibrium-degenerate** problem — intrinsic
        growth α and self-limitation βᵢᵢ trade off along `Kᵢ = −αᵢ/βᵢᵢ`, so a least-squares / ridge
        fit returns a CONFIDENT but UNIDENTIFIABLE parameter estimate. NUDGE measures the α⇄βᵢᵢ
        Laplace curvature and, when the pair is degenerate, returns `unresolved` with the condition
        number, whether it is `degenerate`, the null-space **`degeneracy_direction`**, and a plain
        hint — *the honest "these parameters are not separately identifiable; here is the exact
        combination the data cannot pin, and the experiment that would"* — instead of a fabricated
        point estimate (`NUDGE-LIM-020`). Pass `target` = a species index for a specific taxon, or
        leave it at -1 to let NUDGE screen.
        """
        from nudge.service import lotka_file

        return lotka_file(path, target=None if target < 0 else target, steps=steps)

    @mcp.tool()
    def fibrillization(path: str, m_tot: float = 1.0) -> dict[str, Any]:
        """Fit a protein-aggregation / polymerization curve + report the rate + its identifiability.

        For a single sigmoidal aggregation (amyloid-type filament assembly / nucleated
        polymerization) curve, `path` is a CSV/TSV of mass-fraction (∈[0,1]) vs time (first two
        columns). NUDGE fits the microscopic filament-assembly moment model — PRIMARY NUCLEATION
        (k_n), ELONGATION (k_+), SECONDARY surface-catalysed NUCLEATION (k_2) — and returns the two
        composite rate parameters the curve actually determines (λ=√(2·k_+·k_n·…), κ=√(2·k_+·k_2·…))
        with CIs, PLUS the honest identifiability of the THREE individual constants.

        **The point (Meisl 2016 / Michaels 2020):** a single curve is PROVABLY non-identifiable in
        the individual rate constants — there is an exact gauge (k_n,k_+,k_2)→(k_n/α, α·k_+, k_2/α)
        that leaves the mass-fraction curve unchanged — so a least-squares fit that reports three
        confident k's is over-fitting. NUDGE measures the Fisher/Laplace curvature and, when the
        individual constants are degenerate, returns their identifiable composites + the null
        direction + "need a concentration series AND a seeded/elongation anchor" (`NUDGE-LIM-021`),
        rather than a fabricated point estimate. `m_tot` is the initial monomer concentration
        (1.0 for a normalized mass-fraction curve).
        """
        from nudge.service import fibrillization_file

        return fibrillization_file(path, m_tot=m_tot)

    @mcp.tool()
    def constitutive(
        path: str = "",
        demo: bool = False,
        circuit_n: float = 3.0,
        readout_h: float = 6.0,
        steps: int = 600,
        restarts: int = 3,
    ) -> dict[str, Any]:
        """Separate CIRCUIT ultrasensitivity from a NONLINEAR READOUT — the NUDGE-LIM-006 fix.

        NUDGE assumes an AFFINE reporter. A *nonlinear* reporter (saturating / sigmoidal Hill)
        over a *linear* circuit produces a pseudo-bimodal count distribution the affine-readout
        switch model can only explain by bending the circuit — a CONFIDENT FALSE POSITIVE
        (``NUDGE-LIM-006``, the sharpest bound on the fail-safe guarantee). Only the composition
        readout∘circuit is observed, so from one population you cannot factor it.

        The fix is a **constitutive-reporter control**: a calibration population whose reporter
        is driven at KNOWN activity doses, BYPASSING the circuit — it measures the reporter's
        own transfer function directly and anchors the readout parameters ONLY (no circuit
        leak). NUDGE then runs a **profile likelihood over the circuit Hill n**: WITHOUT the
        control the profile is FLAT (a graded n=1 fits as well as a real switch — you cannot
        even tell a switch exists); WITH the control, "no switch" (n=1) is REJECTED for a
        genuine circuit switch (Δloss ≫ the flat span) → the ultrasensitivity is BIOLOGICAL.

        ``path`` is a ``.npz`` with ``population`` (1-D circuit-population counts) +
        ``control_activity`` / ``control_response`` (the calibration's known doses + measured
        reporter). Set ``demo=True`` (or omit ``path``) to synthesize a matched case: a
        nonlinear (``readout_h``) reporter over a circuit of true Hill ``circuit_n`` — use
        ``circuit_n=1`` for the LIM-006 false-positive HAZARD (→ NUDGE ABSTAINS) or
        ``circuit_n>1`` for a real switch (→ NUDGE rejects "no switch").

        Returns the verdict (``biological-switch`` / ``unresolved`` / ``no-confound``) with the
        calibrated reporter Hill h (+ CI), both n-profiles, and the n=1-rejection metric.
        **Fail-safe (``NUDGE-LIM-018``):** it NEVER emits a bare threshold/gain/ceiling — it
        turns the LIM-006 confident false positive into a correct BIOLOGICAL call or an honest
        abstention — and it does NOT point-identify the exact n (that needs a second anchor: an
        input titration / circuit dose-response).
        """
        from nudge.service import constitutive_demo, constitutive_file

        if demo or not path:
            return constitutive_demo(
                circuit_n=circuit_n, readout_h=readout_h, steps=steps, restarts=restarts
            )
        return constitutive_file(
            path, circuit_n=circuit_n, h=readout_h, steps=steps, restarts=restarts
        )

    @mcp.tool()
    def render_figure(
        kind: str,
        result_json: str = "",
        demo: bool = False,
        out_dir: str = "",
        self_contained: bool = False,
        animate: bool = False,
        theme: str = "auto",
    ) -> dict[str, Any]:
        """Render a NUDGE result to an HONEST figure (the ``nudge.viz`` battery).

        Draws any registered result ``kind`` (``dose_response`` / ``attribution`` /
        ``identifiability`` / ``robustness`` / ``epistasis`` / ``differential`` /
        ``multi_reporter`` / ``temporal`` / ``aggregation`` / ``constitutive`` /
        ``diagnose`` / ``design`` / ``cross_modality`` / ``oed``) from a frozen result — it
        NEVER re-fits and NEVER re-attributes. The **abstention overlay is stamped off the
        result's own verdict**, so an abstention is drawn as an abstention: the picture can
        never claim more than the text.

        Provide EITHER ``result_json`` (a prior ``*_to_dict()`` / figure-data JSON string,
        e.g. the output of the ``attribute`` / ``dose_response`` / ``differential`` tools) OR
        ``demo=True`` for a zero-setup synthetic example of that kind. Writes to ``out_dir``
        (a server-side path; a temp dir if empty) and returns the written paths + the honest
        caption + the ``abstained`` flag + a size-capped inline ``png_base64`` (omitted, with
        a reason, above the cap; GIFs are always path-only). Also returns ``code_path`` — the
        standalone regenerating ``fig.py`` (the Claude Science provenance grain).
        """
        import json as _json
        import tempfile

        from nudge.service import render_result
        from nudge.viz import _RENDERERS
        from nudge.viz.demo import demo_result

        if kind not in _RENDERERS:
            return {"error": f"unknown figure kind {kind!r}", "known_kinds": sorted(_RENDERERS)}
        if result_json:
            result: Any = _json.loads(result_json)
        elif demo:
            result = demo_result(kind)
        else:
            return {"error": "provide result_json (a saved result) or demo=True"}

        out_base = out_dir or tempfile.mkdtemp(prefix="nudge_viz_")
        ext = ".gif" if animate else ".png"
        out_path = f"{out_base.rstrip('/')}/{kind}{ext}"
        return render_result(
            kind, result, out=out_path, emit_code=True, theme=theme,
            animate=animate, self_contained=self_contained, inline_png=not animate,
            cli_call=f"render_figure({kind})",
        )

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
