"""Map a fitted circuit to a ``MechanismClass``, with the abstention gates.

The gates, in order — fail-loud by construction:

1. **no-effect** — the perturbed distribution is ≈ WT (distance below the effect
   noise floor).
2. **off-model — the linear-baseline / parsimony gate.** The mechanistic (switch)
   model is *not warranted* unless it beats the linear baseline by **more than the
   loss noise floor**. Because ``LinearEffect`` is a limiting case of Hill, the
   more-flexible mechanistic model almost always fits ≥ linear, so without this
   gate false positives are structural. This is what stops NUDGE inventing a
   nonlinear switch when linear regression explains the data just as well —
   catching the "linear circuit + saturating readout" and "marginal-overfit Hill"
   decoys. Broadened to also cover poor absolute fit (off-target). Distinguished
   by ``rationale``.
3. **unresolved** — the best two switch hypotheses (threshold / gain / ceiling)
   are within the loss noise floor: can't tell which.
4. **threshold / gain / ceiling** — a clear winner.
"""

from __future__ import annotations

from nudge.core.results import MechanismCall
from nudge.core.vocabulary import MechanismClass

__all__ = ["decide"]

_PARAM_MECHANISM = {
    "K": MechanismClass.THRESHOLD,
    "n": MechanismClass.GAIN,
    "vmax": MechanismClass.CEILING,
}


def decide(
    perturbation: str,
    param_losses: dict[str, float],
    linear_loss: float,
    wt_distance: float,
    *,
    noise_margin: float,
    effect_margin: float,
) -> MechanismCall:
    """Apply the abstention gates to fitted losses → a ``MechanismCall``.

    ``param_losses`` maps each candidate parameter (``K``/``n``/``vmax``) to its
    restricted-mechanistic fit loss; ``linear_loss`` is the linear-baseline fit;
    ``wt_distance`` is the perturbed-vs-WT distributional distance; ``noise_margin``
    is the loss noise floor (bootstrap std) and ``effect_margin`` the no-effect
    floor.
    """
    best_param = min(param_losses, key=param_losses.__getitem__)
    best = param_losses[best_param]
    ordered = sorted(param_losses.values())
    runner_up = ordered[1] if len(ordered) > 1 else float("inf")

    def call(
        mechanism: MechanismClass, confidence: float, rationale: str
    ) -> MechanismCall:
        return MechanismCall(
            perturbation=perturbation,
            mechanism=mechanism,
            confidence=confidence,
            rationale=rationale,
        )

    # 1. no-effect — the perturbation barely moved the distribution.
    if wt_distance < effect_margin:
        return call(MechanismClass.NO_EFFECT, 0.0, "distribution ~ WT (no effect)")

    # 2. off-model — the linear-baseline / parsimony gate.
    if best >= linear_loss - noise_margin:
        return call(
            MechanismClass.OFF_MODEL,
            0.0,
            "mechanistic fit does not beat the linear baseline beyond the noise "
            "floor — no switch mechanism warranted",
        )

    # 3. unresolved — the top two switch hypotheses are within the noise floor.
    if runner_up - best < noise_margin:
        return call(
            MechanismClass.UNRESOLVED,
            0.0,
            "top two mechanisms within the loss noise floor — cannot resolve",
        )

    # 4. a clear switch mechanism.
    gap_vs_linear = (linear_loss - best) / max(linear_loss, 1e-9)
    gap_vs_runner = (runner_up - best) / max(runner_up, 1e-9)
    confidence = float(min(1.0, 2.0 * min(gap_vs_linear, gap_vs_runner)))
    return call(
        _PARAM_MECHANISM[best_param],
        confidence,
        f"{best_param} beats the linear baseline and the runner-up beyond the floor",
    )
