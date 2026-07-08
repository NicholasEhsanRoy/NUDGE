"""Map fitted losses to a ``MechanismClass``, with the abstention gates.

Two levels, both fail-loud:

**Circuit level — the linear-baseline / parsimony gate** (``switch_detected``).
Before attributing anything, ask whether the WT data contains a switch at all: the
mechanistic model must beat the linear baseline on WT by **more than the loss noise
floor**. Because ``LinearEffect`` is a limiting case of Hill, the more-flexible
mechanistic model almost always fits ≥ linear, so without this gate false positives
are structural. If it fails, there is no switch to attribute — every perturbation
is ``off-model``. This catches the "linear circuit + saturating readout" and
"marginal-overfit Hill" decoys at the right level (the whole dataset), and — unlike
a per-perturbation version — does **not** misfire on genuine gain/ceiling reductions
that make a *perturbed* condition look linear.

**Per-perturbation** (``decide``), given a switch exists:
1. **no-effect** — the perturbed distribution ≈ WT.
2. **off-model** — even the best restricted fit leaves a large *absolute* residual
   (off-target / wrong circuit).
3. **unresolved** — the best two switch hypotheses are within the noise floor.
4. **threshold / gain / ceiling** — the clear winner among the restricted fits.

Design credit: the linear-baseline-as-primary-gate idea is the user's; the noise
floor is the loss's own finite-sample scale (== "the margin survives uncertainty").
"""

from __future__ import annotations

from nudge.core.results import MechanismCall
from nudge.core.vocabulary import MechanismClass

__all__ = ["decide", "switch_detected"]

_PARAM_MECHANISM = {
    "K": MechanismClass.THRESHOLD,
    "n": MechanismClass.GAIN,
    "vmax": MechanismClass.CEILING,
}


def switch_detected(
    wt_mechanistic_loss: float, wt_linear_loss: float, *, noise_margin: float
) -> bool:
    """Linear-baseline parsimony gate (circuit level): does WT contain a switch?

    ``True`` iff the mechanistic model beats the linear baseline on WT by more than
    the loss noise floor — i.e. a nonlinear switch is warranted over linear regression.
    """
    return wt_mechanistic_loss < wt_linear_loss - noise_margin


def decide(
    perturbation: str,
    param_losses: dict[str, float],
    wt_distance: float,
    *,
    noise_margin: float,
    effect_margin: float,
    off_model_loss: float,
) -> MechanismCall:
    """Attribute one perturbation to a ``MechanismClass`` (a switch is assumed present).

    ``param_losses`` maps each candidate parameter (``K``/``n``/``vmax``) to its
    restricted-mechanistic fit loss; ``wt_distance`` is the perturbed-vs-WT
    distance; ``noise_margin`` the loss noise floor; ``effect_margin`` the no-effect
    floor; ``off_model_loss`` the absolute residual above which even the best fit is
    deemed off-model.
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

    if wt_distance < effect_margin:
        return call(MechanismClass.NO_EFFECT, 0.0, "distribution ~ WT (no effect)")
    if best > off_model_loss:
        return call(
            MechanismClass.OFF_MODEL,
            0.0,
            "even the best mechanistic fit leaves a large residual (off-target)",
        )
    if runner_up - best < noise_margin:
        return call(
            MechanismClass.UNRESOLVED,
            0.0,
            "top two mechanisms within the loss noise floor — cannot resolve",
        )
    confidence = float(min(1.0, (runner_up - best) / max(runner_up, 1e-9)))
    return call(
        _PARAM_MECHANISM[best_param],
        confidence,
        f"{best_param} beats the runner-up beyond the loss noise floor",
    )
