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
- ``identifiability(model, ...)`` — which parameters of a differentiable ODE model (BY NAME
  from a general registry) are identifiable / sloppy / unrecoverable — the matrix-free FIM
  diagnostic + the honest abstention (``NUDGE-LIM-023``) + the FIM-spectrum figure.
- ``oed(model, target, ...)`` — gradient-design the experiment that best resolves a confounded
  parameter of a registered ODE model; the MEASURED CRLB / eigenvalue lift + the ellipse GIF.
- ``list_models()`` — the general model registry the two tools above operate over.
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


# --------------------------------------------------------------------------- #
# async job pattern — the ~60s connector cap breaker
#
# The Claude Science connector kills any tool call exceeding ~60s, but several NUDGE tools
# (a fit, an OED optimisation, the constitutive demo ~64s) legitimately exceed it. So the
# heavy tools can be run as a JOB: ``job_submit(tool, args_json)`` returns a ``job_id`` in
# <1s and runs the real tool in a background thread; ``job_status(job_id)`` polls it. JAX
# releases the GIL during XLA, so the worker thread does not block the event loop, and each
# individual call stays well under the cap while the real compute spans however long it takes.
# --------------------------------------------------------------------------- #

#: job_id → {future, tool, submitted}. Mutated only from the tool-dispatch thread; the
#: worker thread only touches the (thread-safe) Future.
_JOBS: dict[str, dict[str, Any]] = {}
_JOB_EXECUTOR: Any = None  # a lazily-created ThreadPoolExecutor

#: Tools that can exceed the ~60s cap — clients should prefer job_submit for these. (Purely
#: advisory; job_submit accepts any tool except itself and job_status.)
HEAVY_TOOLS = frozenset({
    "attribute", "fibrillization", "constitutive", "differential", "differential_robust",
    "multi_reporter", "lotka", "design", "synergy", "cross_modality", "render_figure",
    "identifiability", "oed",
})


def _executor() -> Any:
    """The shared job thread-pool (created on first use)."""
    global _JOB_EXECUTOR
    if _JOB_EXECUTOR is None:
        from concurrent.futures import ThreadPoolExecutor

        _JOB_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="nudge-job")
    return _JOB_EXECUTOR


def _unwrap(result: Any) -> Any:
    """FastMCP ``call_tool`` returns ``(content, structured)`` across versions — take the
    structured payload (and unwrap a single ``{"result": …}`` envelope)."""
    structured = result[1] if isinstance(result, tuple) else result
    if isinstance(structured, dict) and "result" in structured and len(structured) == 1:
        return structured["result"]
    return structured


def _register_job_tools(mcp: Any) -> None:
    """Register ``job_submit`` / ``job_status`` on ``mcp`` (the async-job pattern)."""

    @mcp.tool()
    def job_submit(tool: str, args_json: str = "{}") -> dict[str, Any]:
        """Run a (possibly slow) NUDGE tool as a BACKGROUND JOB — returns a ``job_id`` in <1s.

        The connector kills any single tool call over ~60s, but a fit / OED optimisation /
        the constitutive demo can exceed that. ``job_submit`` starts the real ``tool`` (any
        NUDGE tool, e.g. ``attribute`` / ``fibrillization`` / ``constitutive`` / ``lotka`` /
        ``design`` / ``multi_reporter`` / ``differential`` / ``render_figure``) in a
        background thread with ``args_json`` (a JSON object of that tool's arguments) and
        returns immediately with a ``job_id``. Poll :func:`job_status` until it is ``done``
        (carrying the real ``result``) or ``error``. Fast tools (``list_mechanisms`` /
        ``dose_response`` / ``explain_abstention`` / ``get_mechanism_card`` /
        ``diagnose_abstention``) can just be called directly.
        """
        import uuid

        if tool in ("job_submit", "job_status"):
            return {"error": f"{tool!r} cannot itself be submitted as a job; call it directly"}
        try:
            import json as _json

            args = _json.loads(args_json) if args_json.strip() else {}
        except ValueError as exc:
            return {"error": f"args_json is not valid JSON: {exc}"}
        if not isinstance(args, dict):
            return {"error": "args_json must be a JSON object of the tool's arguments"}

        import asyncio
        import time

        job_id = uuid.uuid4().hex[:12]
        submitted = time.monotonic()

        def _run() -> Any:
            # A fresh event loop in the worker thread drives the real tool coroutine; the
            # heavy (JAX) compute runs here, off the server's event loop.
            return _unwrap(asyncio.run(mcp.call_tool(tool, args)))

        future = _executor().submit(_run)
        _JOBS[job_id] = {"future": future, "tool": tool, "submitted": submitted}
        return {
            "job_id": job_id, "tool": tool, "status": "running",
            "note": "poll job_status(job_id) until status is 'done' or 'error'",
        }

    @mcp.tool()
    def job_status(job_id: str) -> dict[str, Any]:
        """Poll a background job started by :func:`job_submit`.

        Returns ``status`` ``running`` (with ``elapsed_s``), ``done`` (with the tool's real
        ``result``), or ``error`` (with the exception message) — so a client can start a slow
        job, keep each poll well under the ~60s cap, and collect the result when it lands.
        """
        import time

        job = _JOBS.get(job_id)
        if job is None:
            return {"error": f"unknown job_id {job_id!r}"}
        future = job["future"]
        elapsed = round(time.monotonic() - job["submitted"], 2)
        if not future.done():
            return {"job_id": job_id, "tool": job["tool"], "status": "running",
                    "elapsed_s": elapsed}
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001 - surface any tool failure honestly
            return {"job_id": job_id, "tool": job["tool"], "status": "error",
                    "elapsed_s": elapsed, "error": f"{type(exc).__name__}: {exc}"}
        return {"job_id": job_id, "tool": job["tool"], "status": "done",
                "elapsed_s": elapsed, "result": result}


#: Connector usage note delivered to the client (Claude Science / Desktop) as the server's
#: MCP ``instructions`` — the runtime agent's "how to drive NUDGE well" (docs/user_guide/
#: claude_science.md is the human version).
NUDGE_MCP_INSTRUCTIONS = (
    "NUDGE attributes perturbation mechanisms and ABSTAINS loudly when it can't tell — a "
    "confident-wrong answer is the only hard failure, so present every abstention AS an "
    "abstention.\n\n"
    "Figures: render_figure returns an image. With NUDGE_ENV=cloud it comes back INLINE as "
    "`image_base64` + `mime_type` (a file path is impossible in this sandbox); decode and "
    "display it immediately and DON'T echo the base64 blob. The `code` (a standalone fig.py) "
    "and `data` (the sidecar) fields regenerate it with no re-fit — attach them as the "
    "artifact's provenance. GIFs are size-disciplined and fall back to a static preview above "
    "the inline cap (never truncated). MCP resource URIs are not dereferenceable here.\n\n"
    "Long calls: the host kills any single tool call over ~60s. Run heavy tools (attribute, "
    "fibrillization, constitutive, differential, multi_reporter, lotka, design, identifiability, "
    "oed, and a slow render_figure demo) as a background job: job_submit(tool, args_json) "
    "returns a job_id in "
    "<1s; poll job_status(job_id) until 'done' (with the real result) or 'error'. Fast tools "
    "(list_mechanisms, dose_response, explain_abstention, get_mechanism_card, "
    "diagnose_abstention) can be called directly."
)


def build_server() -> Any:
    """Construct and return the FastMCP server with NUDGE's tools registered."""
    FastMCP = _require_mcp()
    mcp = FastMCP("nudge", instructions=NUDGE_MCP_INSTRUCTIONS)

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
        never claim more than the text. ``animate=True`` renders an animated GIF for the
        subset with a natural frame variable (``constitutive`` / ``oed`` / ``robustness`` /
        ``aggregation`` / ``temporal`` / ``multi_reporter`` / ``identifiability`` /
        ``design`` / ``dose_response`` / ``cross_modality``).

        Provide EITHER ``result_json`` (a prior ``*_to_dict()`` / figure-data JSON string,
        e.g. the output of the ``attribute`` / ``dose_response`` / ``differential`` tools) OR
        ``demo=True`` for a zero-setup synthetic example of that kind.

        **Transport (see ``docs/user_guide/claude_science.md``).** The figure is delivered
        per the ``NUDGE_ENV`` toggle. ``NUDGE_ENV=cloud`` (the Claude Science reality — its
        connector cannot hand back a readable file path) → the image rides back **inline** as
        ``image_base64`` + ``mime_type`` (a GIF is downscaled / frame-limited / never-inflated
        and capped, falling back to a static final-frame preview above the cap — never a
        silent truncation), with the regenerating ``fig.py`` (``code``) and the sidecar
        (``data``) inline as text so the client's artifact-provenance system can ingest them.
        Otherwise → the figure is **written** (to ``out_dir`` / ``NUDGE_ARTIFACT_DIR`` /
        temp) and ``image_path`` / ``code_path`` / ``data_path`` are returned. Every response
        carries ``caption`` + ``abstained``. For a slow ``demo`` build (a real synthetic
        analysis, e.g. ``constitutive`` / ``oed`` / ``temporal`` / ``aggregation``), submit
        via ``job_submit("render_figure", …)`` so the call returns under the connector's ~60s
        cap.
        """
        import json as _json

        from nudge.service import render_result
        from nudge.viz import _RENDERERS
        from nudge.viz.animate import _ANIMATORS
        from nudge.viz.demo import demo_result

        if kind not in _RENDERERS:
            return {"error": f"unknown figure kind {kind!r}", "known_kinds": sorted(_RENDERERS)}
        if animate and kind not in _ANIMATORS:
            return {
                "error": f"kind {kind!r} has no animation (no natural frame variable)",
                "animatable_kinds": sorted(_ANIMATORS),
            }
        if result_json:
            result: Any = _json.loads(result_json)
        elif demo:
            result = demo_result(kind, animate=animate)
        else:
            return {"error": "provide result_json (a saved result) or demo=True"}

        ext = ".gif" if animate else ".png"
        out_path = f"{out_dir.rstrip('/')}/{kind}{ext}" if out_dir else None
        return render_result(
            kind, result, out=out_path, emit_code=True, theme=theme,
            animate=animate, self_contained=self_contained,
            cli_call=f"render_figure({kind})",
        )

    @mcp.tool()
    def list_models() -> list[dict[str, Any]]:
        """List the differentiable models the ``identifiability`` / ``oed`` tools can analyse.

        The general model registry (:mod:`nudge.inference.model_registry`): each entry is a
        differentiable ODE / forward model across a domain (microbiome ecology, reaction
        kinetics, clinical pharmacology, population dynamics, canonical sloppiness toys) with
        which tools it supports. Both tools take a model **by name** from this list and run the
        REAL analysis — not a hardcoded answer. Register your own in a few lines
        (:func:`nudge.inference.model_registry.register_model`); arbitrary models remain a
        plain ``import nudge`` library path (``NUDGE-LIM-027``).
        """
        from nudge.inference.model_registry import list_models as _list

        return _list()

    @mcp.tool()
    def identifiability(
        model: str,
        free: str = "",
        n_free: int = 0,
        method: str = "auto",
        sigma: float = float("nan"),
        with_figure: bool = True,
    ) -> dict[str, Any]:
        """Which parameters of a differentiable ODE model are identifiable / sloppy / unrecoverable?

        A GENERAL identifiability tool: it takes a model **by reference** (a name from
        ``list_models`` — e.g. ``glv`` / ``linear_pathway`` / ``ad_qsp`` / ``logistic``, plus
        the canonical ``sum_of_exponentials`` / ``redundant_exponential`` / ``well_conditioned``
        toys), runs NUDGE's REAL **matrix-free Fisher-information** diagnostic
        (:func:`nudge.inference.sloppiness.sloppiness_diagnostic_matrixfree`), and returns
        whatever it MEASURES — never a hardcoded answer.

        Returns the verdict — ``well-constrained`` (every parameter individually identifiable),
        ``sloppy-but-predictive`` (loose parameters but tight predictions — NUDGE must NOT
        abstain), or ``unidentifiable`` (a structural null / rank-deficiency — NUDGE **abstains**
        and NAMES the unrecoverable parameter combination) — the FIM spectrum (condition number,
        spectral span, smallest/largest eigenvalue), the named ``null_directions``, and the
        honest fail-safe bound (``NUDGE-LIM-023``: the matrix-free path never asserts an
        identifiability it cannot certify — it abstains instead).

        ``free`` restricts to a comma-separated parameter subset (the rest held at nominal);
        ``n_free`` is the population-/dimension-scale knob (``glv`` / ``linear_pathway`` /
        ``ad_qsp`` — how many parameters are jointly estimated, which drives a sparse-data model
        into honest rank-deficiency); ``method`` ∈ ``auto`` / ``dense`` / ``iterative``;
        ``sigma`` overrides the observation noise (omit / NaN → the model default). With
        ``with_figure`` the FIM-spectrum figure rides back via the render seam (inline base64 +
        the regenerating ``fig.py`` + data sidecar under ``NUDGE_ENV=cloud``). Can be slow at
        scale → prefer ``job_submit("identifiability", …)``.
        """
        import math

        from nudge.service import identifiability_tool as _run

        free_list = [s.strip() for s in free.split(",") if s.strip()]
        return _run(
            model,
            free=free_list or None,
            n_free=n_free,
            method=method,
            sigma=None if math.isnan(sigma) else sigma,
            with_figure=with_figure,
        )

    @mcp.tool()
    def oed(
        model: str,
        target: str = "",
        objective: str = "d_opt",
        n_obs: int = 8,
        steps: int = 400,
        sigma: float = float("nan"),
        naive: str = "",
        with_figure: bool = True,
    ) -> dict[str, Any]:
        """Design the experiment that best resolves a confounded parameter of an ODE model.

        A GENERAL gradient optimal-experimental-design tool: it takes a model **by reference**
        (a name from ``list_models`` supporting OED — ``logistic`` / ``glv`` / ``ad_qsp``),
        builds a differentiable :class:`~nudge.inference.oed.DesignProblem`, and
        gradient-ascends the measurement schedule to the design that best resolves the target
        parameter (:func:`nudge.inference.oed.optimize_design`) — the white-box advantage a
        black-box solver can't offer (``∂criterion/∂φ`` by autodiff through the ODE solve).

        Returns the optimal measurement schedule and the **MEASURED** identifiability gain — the
        target parameter's Cramér–Rao-bound factor (``crlb_improvement``) and the FIM
        smallest-eigenvalue lift (``min_eig_improvement``), plus the naive-design confound
        correlation and the condition-number before/after — **computed, never asserted**. It is
        a design *recommendation*, not an attribution verdict, so it can't emit a confident-wrong
        mechanism call. Local OED: valid near the nominal θ₀ (``NUDGE-LIM-024``).

        ``target`` selects the parameter to resolve (default: the model's confounded target,
        e.g. ``log_k_on`` for ``ad_qsp``); ``objective`` ∈ ``d_opt`` / ``a_opt`` / ``e_opt`` /
        ``crlb``; ``n_obs`` the number of measurement times; ``naive`` overrides the naive
        'baseline+end' schedule (comma-separated times); ``sigma`` overrides the observation
        noise. With ``with_figure`` the 95%-confidence-ellipse-collapse GIF rides back inline
        (base64 + ``fig.py`` + sidecar under ``NUDGE_ENV=cloud``). Slow → prefer
        ``job_submit("oed", …)``.
        """
        import math

        from nudge.service import oed_tool as _run

        naive_list = [float(s) for s in naive.split(",") if s.strip()]
        return _run(
            model,
            target=target or None,
            objective=objective,
            n_obs=n_obs,
            steps=steps,
            sigma=None if math.isnan(sigma) else sigma,
            naive=naive_list or None,
            with_figure=with_figure,
        )

    _register_job_tools(mcp)
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
