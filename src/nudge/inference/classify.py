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

__all__ = ["decide", "decide_with_transition", "switch_detected"]

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


def decide_with_transition(
    perturbation: str,
    param_losses: dict[str, float],
    wt_distance: float,
    *,
    noise_margin: float,
    effect_margin: float,
    off_model_loss: float,
    transition_weight: float | None,
    n_species: int,
    gain_wtrans_tau: float = 0.5,
) -> MechanismCall:
    """``decide`` plus the **saddle transition-mode gain gate** (multi-basin path).

    A gain reduction destroys a switch's cooperativity, collapsing it toward a single
    *intermediate* fixed point — graded cells the two basins cannot hold but the
    transition-at-saddle mode can. So when a restricted free-``n`` fit is *forced* to
    spend a large transition weight, that is a specific, seed-robust gain signature
    (measured: ``w_trans`` ≈ 0.9 for gain vs ≈ 0.01 for threshold/ceiling/no-effect;
    ``FINDINGS.md`` §T0.5-5) — and it is decisive where the raw loss argmin ties
    thin (the gain/ceiling degeneracy of the 2-mode model).

    Fail-safe by construction:
    - Runs **after** the no-effect and off-model gates (via ``decide``), so it can never
      promote a WT-like or badly-fit condition to GAIN.
    - Fires **only** for a genuine 1-species saddle (``n_species == 1`` and a real
      ``transition_weight``). For N-species there is no saddle finder, so it defers to
      the honest single-basin abstention — the degeneracy is isolated to the case we
      have proven (FM2).
    - ``gain_wtrans_tau`` sits in a wide verified margin (0.12 ↔ 0.87); tunable.
    """
    base = decide(
        perturbation, param_losses, wt_distance,
        noise_margin=noise_margin,
        effect_margin=effect_margin,
        off_model_loss=off_model_loss,
    )
    abstained_low = base.mechanism in (
        MechanismClass.NO_EFFECT,
        MechanismClass.OFF_MODEL,
    )
    gate_fires = (
        n_species == 1
        and transition_weight is not None
        and transition_weight > gain_wtrans_tau
        and not abstained_low
    )
    if gate_fires:
        assert transition_weight is not None
        return MechanismCall(
            perturbation=perturbation,
            mechanism=MechanismClass.GAIN,
            confidence=float(min(1.0, transition_weight)),
            rationale=(
                f"graded transition-mode signature at the saddle "
                f"(w_trans={transition_weight:.2f} > {gain_wtrans_tau}) → gain"
            ),
        )
    return base
