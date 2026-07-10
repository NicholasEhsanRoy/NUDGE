"""Hidden-node **abstention** — turn a bare ``off-model`` into a legible differential.

**This module ships ONLY the abstention half of the hidden-node problem, never a
positive identification.** When NUDGE's switch model is inadequate — the circuit-level
parsimony gate returns ``off-model`` (:func:`nudge.inference.classify.switch_detected`),
or a diagnostic residual fires (the off-axis / neomorphic ratio,
:class:`nudge.inference.epistasis.ComboGeometry`) — a user is left with a one-word verdict
and no idea *why*. This packages the evidence into a **differential diagnosis**: it
enumerates the *candidate causes* of the inadequacy, each with its evidence, the
documented limitation / decoy it maps to, and the experiment that would distinguish it.

**The load-bearing honesty rule (NUDGE-LIM-015).** Positive hidden-node identification is
**not identifiable** from an off-model verdict: an unmeasured regulator, an off-target
effect, a nonlinear readout, a misspecified topology, a batch/depth confound, and a
genuinely-linear (not-a-switch) circuit are **observationally overlapping** — they all
present as "the affine switch model does not fit." So this module NEVER asserts "there is
a hidden node here." The strongest thing it will ever say about a hidden node is that an
off-axis residual is *consistent with — but does not prove —* an unmeasured regulator, and
*here is the experiment* that would separate it from the other causes. The differential is
a **hypothesis-ranking aid, not a discovery.**

It is a pure **packaging / knowledge** layer: it consumes verdicts and diagnostic scalars
(the ``off-model`` boolean, the neomorphic ratio, a readout-linearity flag, …) and returns
the ranked differential. It has **zero import of** :mod:`nudge.inference.fit` — it never
touches the fit; it only reads what the fit already decided.

The six enumerated causes (a differential, not a verdict):

1. **hidden node / unmeasured regulator** — the neomorphic off-axis residual
   (``NUDGE-LIM-009``); hedged, never asserted (``NUDGE-LIM-015``).
2. **off-target perturbation effect** — the guide hits something other than its target.
3. **nonlinear measurement readout** — a saturating/sigmoidal reporter fakes
   ultrasensitivity (``NUDGE-LIM-006``).
4. **wrong / misspecified topology** — the fitted circuit is the wrong shape (Tier-0.5
   T0.5-2 boundary).
5. **batch / depth confound** — a sequencing-depth or batch difference aligned with the
   condition masquerades as structure (``NUDGE-LIM-003`` / ``NUDGE-LIM-009``).
6. **genuinely not-a-switch (linear circuit)** — the parsimony gate rejecting the switch
   model is the gate *working* (``NUDGE-LIM-005`` / ``NUDGE-DECOY-005``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "CandidateCause",
    "InadequacyReport",
    "diagnose_inadequacy",
]

#: The qualitative ranks a candidate cause can carry (a differential, not a probability).
#: Deliberately coarse — the evidence does not support a calibrated posterior over causes
#: (that is the whole point of ``NUDGE-LIM-015``), only a legible ordering of hypotheses.
RANK_ORDER: dict[str, int] = {"leading": 0, "plausible": 1, "less-likely": 2}


@dataclass(frozen=True)
class CandidateCause:
    """One entry in the inadequacy differential — a *hypothesis*, with how to test it.

    Every field is descriptive; nothing here is a claim that this cause *is* the truth.
    ``evidence`` states what (if anything) in the data points at this cause;
    ``distinguishing_experiment`` is the concrete measurement that would separate it from
    the others; ``limitation_ref`` / ``decoy_ref`` cross-reference the documented failure
    mode (``NUDGE-LIM-*``) and the decoy (``NUDGE-DECOY-*``) that pin it — resolvable via
    :func:`nudge.knowledge.explain`. ``qualitative_rank`` is a coarse ordering token
    (``leading`` / ``plausible`` / ``less-likely``), **not** a probability.
    """

    name: str
    evidence: str
    distinguishing_experiment: str
    limitation_ref: str = ""
    decoy_ref: str = ""
    qualitative_rank: str = "plausible"


@dataclass(frozen=True)
class InadequacyReport:
    """The differential diagnosis for an inadequate (or adequate) attribution.

    When the model is **adequate** (no off-model verdict, no diagnostic residual fired),
    ``is_adequate`` is ``True``, ``causes`` is empty, and ``reason`` says so. When it is
    **inadequate**, ``is_adequate`` is ``False`` and ``causes`` is the rank-ordered
    differential. ``verdict`` is the trigger (``off-model`` / ``diagnostic-residual`` /
    ``model-adequate``).

    **Honesty invariant.** No field of this report — and no ``CandidateCause`` it carries —
    ever asserts a hidden node exists. The hidden-node cause is always phrased as
    *consistent with, does not prove* (``NUDGE-LIM-015``).
    """

    is_adequate: bool
    verdict: str
    causes: list[CandidateCause] = field(default_factory=list)
    reason: str = ""

    def ranked_causes(self) -> list[CandidateCause]:
        """The causes sorted by qualitative rank (leading first), ties in insertion order."""
        return sorted(
            self.causes, key=lambda c: RANK_ORDER.get(c.qualitative_rank, 99)
        )


# --------------------------------------------------------------------------- #
# The six candidate causes. Each builder returns a CandidateCause at a rank that
# depends on the evidence supplied. Hidden-node's rank is CAPPED at "plausible" —
# it can never be the single "leading" answer, by construction (NUDGE-LIM-015).
# --------------------------------------------------------------------------- #
def _cause_hidden_node(neomorphic_ratio: float | None, threshold: float) -> CandidateCause:
    has = neomorphic_ratio is not None and neomorphic_ratio == neomorphic_ratio  # not nan
    fired = has and neomorphic_ratio is not None and neomorphic_ratio >= threshold
    if fired and neomorphic_ratio is not None:
        evidence = (
            f"an off-axis / neomorphic residual (ratio {neomorphic_ratio:.2f} ≥ "
            f"{threshold:g}) is CONSISTENT WITH — but does NOT prove — an unmeasured "
            "regulator / emergent state the measured axes do not span. It is "
            "observationally indistinguishable from the other causes below; NUDGE does "
            "NOT assert a hidden node (NUDGE-LIM-015, NUDGE-LIM-009)"
        )
        rank = "plausible"
    elif has and neomorphic_ratio is not None:
        evidence = (
            f"the off-axis / neomorphic residual is small (ratio {neomorphic_ratio:.2f} < "
            f"{threshold:g}) — no positive signal for an unmeasured regulator, and it "
            "could never prove one anyway (NUDGE-LIM-015, NUDGE-LIM-009)"
        )
        rank = "less-likely"
    else:
        evidence = (
            "no off-axis diagnostic was supplied — a hidden node cannot be assessed, and "
            "even a large off-axis residual would only be CONSISTENT WITH one, never prove "
            "it (NUDGE-LIM-015, NUDGE-LIM-009)"
        )
        rank = "less-likely"
    return CandidateCause(
        name="hidden node / unmeasured regulator",
        evidence=evidence,
        distinguishing_experiment=(
            "measure the candidate regulator directly (add it to the panel / a multiome or "
            "multi-reporter readout) so the off-axis dimension becomes an ON-axis measured "
            "one — only a positive measurement, not a residual, can establish a hidden node"
        ),
        limitation_ref="NUDGE-LIM-015",
        decoy_ref="",
        qualitative_rank=rank,
    )


def _cause_off_target(perturbation_residual: float | None) -> CandidateCause:
    finite = (
        perturbation_residual is not None
        and perturbation_residual == perturbation_residual  # not nan
    )
    if finite and perturbation_residual is not None:
        evidence = (
            f"even the best restricted mechanistic fit leaves a large absolute residual "
            f"({perturbation_residual:.3g}) — the perturbation may act OFF-TARGET (a knob "
            "the fitted circuit does not contain), the off-model branch of "
            "nudge.inference.classify.decide"
        )
    else:
        evidence = (
            "the perturbation may act OFF-TARGET — hitting a gene other than its nominal "
            "target — which the fitted circuit cannot represent (the off-model residual "
            "branch of nudge.inference.classify.decide)"
        )
    return CandidateCause(
        name="off-target perturbation effect",
        evidence=evidence,
        distinguishing_experiment=(
            "a SECOND, independent guide against the same target (a different seed "
            "sequence): a concordant call across guides argues against an off-target "
            "effect; a discordant one implicates it"
        ),
        limitation_ref="NUDGE-LIM-004",
        decoy_ref="NUDGE-DECOY-004",
        qualitative_rank="plausible",
    )


def _cause_nonlinear_readout(readout_flag: bool | None) -> CandidateCause:
    if readout_flag:
        evidence = (
            "a readout-linearity flag is SET — a nonlinear (saturating / sigmoidal) "
            "reporter can manufacture apparent ultrasensitivity the affine-readout switch "
            "model cannot separate from a circuit switch (NUDGE-LIM-006)"
        )
        rank = "leading"
    else:
        evidence = (
            "a nonlinear (saturating / sigmoidal) reporter can manufacture apparent "
            "ultrasensitivity; NUDGE assumes an AFFINE readout and cannot structurally "
            "separate readout- from circuit-ultrasensitivity from one population "
            "(NUDGE-LIM-006)"
        )
        rank = "plausible"
    return CandidateCause(
        name="nonlinear measurement readout",
        evidence=evidence,
        distinguishing_experiment=(
            "a CONSTITUTIVE-reporter calibration control (drive the reporter at known "
            "doses, bypassing the circuit): VALIDATED to break the readout-vs-circuit "
            "degeneracy (FINDINGS 'NUDGE-LIM-006 mitigation')"
        ),
        limitation_ref="NUDGE-LIM-006",
        decoy_ref="",
        qualitative_rank=rank,
    )


def _cause_wrong_topology(topology_uncertain: bool | None) -> CandidateCause:
    rank = "plausible" if topology_uncertain else "less-likely"
    evidence = (
        "the fitted circuit may be the WRONG SHAPE — 'the edge's K/n/vmax' means "
        "different things in different topologies, and a confidently-wrong call can slip "
        "the gates under topology misspecification (Tier-0.5 boundary T0.5-2, FINDINGS)"
    )
    if topology_uncertain:
        evidence += " — and the caller flagged the topology as uncertain"
    return CandidateCause(
        name="wrong / misspecified topology",
        evidence=evidence,
        distinguishing_experiment=(
            "refit alternative topologies (1-node / 2-node / toggle) and compare parsimony "
            "(a topology-adequacy check); multi-basin IC seeding — recovering the "
            "attribution still needs a topology prior (T0.5-3/4)"
        ),
        limitation_ref="",
        decoy_ref="",
        qualitative_rank=rank,
    )


def _cause_batch_depth(depth_confounded: bool | None) -> CandidateCause:
    rank = "plausible" if depth_confounded else "less-likely"
    evidence = (
        "a sequencing-DEPTH or BATCH difference aligned with the condition can masquerade "
        "as structure NUDGE reads as a switch / interaction (dropout zero-peak, "
        "NUDGE-LIM-003; a depth/batch confound, NUDGE-LIM-009)"
    )
    if depth_confounded:
        evidence += " — and the caller flagged a depth/batch imbalance"
    return CandidateCause(
        name="batch / depth confound",
        evidence=evidence,
        distinguishing_experiment=(
            "pin sequencing depth per condition (size-factor / calibrate_from_wt), add a "
            "batch covariate, or rebalance depth — a confound perfectly aligned with the "
            "condition is not separable and NUDGE abstains rather than guess"
        ),
        limitation_ref="NUDGE-LIM-003",
        decoy_ref="NUDGE-DECOY-003",
        qualitative_rank=rank,
    )


def _cause_not_a_switch(off_model: bool) -> CandidateCause:
    # When the parsimony gate itself rejected the switch, "not a switch" is the MOST
    # parsimonious reading (the gate working) — it leads. Absent an off-model verdict
    # (a residual-only trigger) it is merely one hypothesis among the six.
    rank = "leading" if off_model else "plausible"
    return CandidateCause(
        name="genuinely not-a-switch (linear circuit)",
        evidence=(
            "the mechanistic switch model did not beat a linear baseline beyond the loss "
            "noise floor — the data may simply hold NO switch (the parsimony gate working "
            "as designed; this is NUDGE's crown-jewel fail-safe, not an error). A "
            "bimodal snapshot alone is not a switch (telegraph noise, NUDGE-LIM-001; a "
            "cell-type mixture, NUDGE-LIM-002; a marginal Hill, NUDGE-LIM-005)"
        ),
        distinguishing_experiment=(
            "none needed for adequacy — abstaining IS the honest call here. If the data is "
            "merely underpowered, more cells / a wider dose ladder around the inflection "
            "would let a real switch (if any) clear the gate"
        ),
        limitation_ref="NUDGE-LIM-005",
        decoy_ref="NUDGE-DECOY-005",
        qualitative_rank=rank,
    )


def diagnose_inadequacy(
    *,
    off_model: bool,
    neomorphic_ratio: float | None = None,
    readout_flag: bool | None = None,
    perturbation_residual: float | None = None,
    topology_uncertain: bool | None = None,
    depth_confounded: bool | None = None,
    neomorphic_ratio_threshold: float = 1.0,
) -> InadequacyReport:
    """Package NUDGE's model-inadequacy evidence into a legible **differential diagnosis**.

    Takes the *evidence* an attribution already produced — the ``off_model`` parsimony
    verdict (:func:`nudge.inference.classify.switch_detected`) and optional diagnostic
    scalars/flags — and returns the rank-ordered differential of *candidate causes* for the
    inadequacy. It is a pure packaging step: it consumes verdicts, never runs a fit.

    Triggers (``is_adequate`` becomes ``False``): the ``off_model`` verdict, **or** a
    diagnostic residual firing (``neomorphic_ratio ≥ neomorphic_ratio_threshold``, a set
    ``readout_flag``, a finite ``perturbation_residual``, a set ``topology_uncertain`` /
    ``depth_confounded``). When none fire, the model is reported **adequate** with no
    causes.

    **What it will NOT do (NUDGE-LIM-015).** It never returns a positive "hidden node
    detected" claim. The hidden-node cause is always phrased *consistent with — does not
    prove*, and its rank is capped at ``plausible`` so it is never the lone leading answer;
    the parsimony-gate's own reading ("not a switch") and the readout confound lead when the
    evidence warrants. The differential is a hypothesis-ranking aid, not a discovery.

    Args:
        off_model: the circuit-level parsimony gate rejected the switch model.
        neomorphic_ratio: the off-axis / on-axis residual ratio
            (:attr:`nudge.inference.epistasis.ComboGeometry.neomorphic_ratio`); ``None`` if
            not measured.
        readout_flag: the reporter may be nonlinear (a readout-linearity control failed /
            is absent).
        perturbation_residual: the best restricted-fit absolute residual, if available.
        topology_uncertain: the caller is unsure the fitted topology is correct.
        depth_confounded: a depth/batch difference is aligned with the condition.
        neomorphic_ratio_threshold: the ratio at/above which the off-axis residual counts
            as a fired diagnostic (default ``1.0``, matching
            :func:`nudge.inference.epistasis.classify_synergy`).
    """
    neo = neomorphic_ratio
    neo_fired = (
        neo is not None and neo == neo and neo >= neomorphic_ratio_threshold  # not nan
    )
    residual_fired = (
        perturbation_residual is not None
        and perturbation_residual == perturbation_residual  # not nan
    )
    any_diagnostic = bool(
        neo_fired or readout_flag or residual_fired or topology_uncertain
        or depth_confounded
    )

    if not off_model and not any_diagnostic:
        return InadequacyReport(
            is_adequate=True,
            verdict="model-adequate",
            causes=[],
            reason=(
                "no off-model verdict and no diagnostic residual fired — the switch model "
                "is adequate for this attribution; there is no inadequacy to explain, so "
                "no differential is emitted (and, per NUDGE-LIM-015, none is invented)"
            ),
        )

    verdict = "off-model" if off_model else "diagnostic-residual"
    causes = [
        _cause_not_a_switch(off_model),
        _cause_nonlinear_readout(readout_flag),
        _cause_off_target(perturbation_residual),
        _cause_wrong_topology(topology_uncertain),
        _cause_batch_depth(depth_confounded),
        _cause_hidden_node(neo, neomorphic_ratio_threshold),
    ]
    causes.sort(key=lambda c: RANK_ORDER.get(c.qualitative_rank, 99))

    trigger = (
        "the circuit-level parsimony gate returned off-model"
        if off_model
        else "a diagnostic residual fired (no off-model verdict)"
    )
    reason = (
        f"the switch model is INADEQUATE here: {trigger}. Below is a DIFFERENTIAL of "
        f"{len(causes)} candidate causes (ranked hypotheses, each with the experiment that "
        "would distinguish it) — NOT a verdict on which one it is. Most sharply, an "
        "off-axis residual is consistent with, but does NOT prove, a hidden node: NUDGE "
        "ships only this abstention, never a positive hidden-node claim (NUDGE-LIM-015)."
    )
    return InadequacyReport(
        is_adequate=False, verdict=verdict, causes=causes, reason=reason
    )
