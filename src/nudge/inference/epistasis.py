"""Synergy / epistasis attribution for a two-perturbation combination (A / B / A+B).

For a combination of two perturbations, is the joint effect **additive** (the two act
on the same knob — just more of it) or **non-additive** (the combo deviates from the
additive prediction: **synergistic** super-additivity, or **buffering** / epistatic
sub-additivity)? This is the question thousands of combination-therapy and
genetic-interaction labs ask, and — like the switch-vs-threshold distinction — linear
screen analysis structurally cannot make it.

NUDGE reads A, B and A+B as three **operating points** measured against a shared control
and reduces each to a scalar **effect** — a response-magnitude shift of a chosen
signature vs control, depth-normalized like
:func:`~nudge.inference.bridge.knockdown_dose_response`. The **additive null** is
Bliss/HSA-style: ``predicted effect(A+B) = effect(A) + effect(B)``. The **interaction**
is ``effect(A+B) − additive prediction``, carried with a **bootstrap CI over cells**.

**Effect space (state it, because the null depends on it).** Effects are measured in
**log-fold-change space** (a shift of a log-normalized signature score vs control), so
the additive null ``e(A)+e(B)`` is **Bliss independence** — multiplicative in linear
expression space. This is the standard, defensible genetic-interaction null; a different
effect space (raw counts, HSA) would move the additive baseline, so the choice is
reported alongside every call (``EpistasisFit.effect_space``). Because the extractor
that supplies the per-cell scores projects each cell onto the **additive direction fixed
by the two single arms** (see :func:`~nudge.inference.bridge.combo_effect_scores`), a
positive interaction is unambiguously super-additive *along the axis the singles push*
and a negative one is sub-additive — the synergistic/buffering labels are direction-safe
by construction.

**Honesty (the load-bearing rule).** A combination attribution **inherits its weakest
single arm**: we cannot call an interaction whose components we cannot call. So the
classifier gates, most-conservative first, and **abstains** rather than guess:

- ``"unresolved"`` — a condition is **underpowered** (too few cells), or the interaction
  CI is **too wide** to separate additive from synergistic (it straddles 0 but is not
  tight enough to *rule out* synergy), or a clearly-nonzero interaction fails the BIC
  parsimony gate (the CI says non-additive but a free A+B model does not *earn* its
  parameter over the additive null);
- ``"no-effect"`` — **neither** single arm moves the signature above noise (both single
  CIs straddle 0), so there is no interaction to attribute;
- ``"additive"`` — the interaction CI **straddles 0** *and* is tight (a free A+B model
  does not beat the additive null by the BIC margin): same knob, more of it;
- ``"synergistic"`` — the interaction CI is **clearly > 0** *and* a free A+B model beats
  the additive null by the BIC margin: super-additive;
- ``"buffering"`` — the interaction CI is **clearly < 0** *and* the BIC margin clears:
  sub-additive / epistatic.

A super-additive residual is emphatically **not** by itself a hidden-node claim
(NUDGE-LIM-009); and the additive null presumes both single arms are correctly measured
and the readout is approximately affine (NUDGE-LIM-006). These are stated with the call,
never assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "ComboGeometry",
    "EpistasisFit",
    "EpistasisResult",
    "combo_geometry",
    "fit_synergy",
    "classify_synergy",
    "attribute_synergy",
]


def _bic(rss: float, n_obs: int, k: float) -> float:
    """BIC for a Gaussian-residual least-squares fit with unknown variance.

    ``-2 log L = N(ln 2π + ln(RSS/N) + 1)``; ``k`` counts the free params **plus** the
    residual variance σ². Lower is better (Schwarz 1978) — the *same* penalty
    :func:`nudge.inference.dose_response._bic` and
    :mod:`nudge.inference.model_select` use, so every parsimony gate stays consistent.
    """
    rss = max(rss, 1e-12)
    n_obs = max(n_obs, 1)
    return k * np.log(n_obs) + n_obs * (np.log(2 * np.pi) + np.log(rss / n_obs) + 1.0)


@dataclass(frozen=True)
class ComboGeometry:
    """The off-axis (**possible-neomorphic**) diagnostic for a combination.

    NUDGE's scalar interaction is exactly the **on-axis projection** of the combo's
    full interaction-residual vector ``r = v_AB − v_A − v_B`` onto the additive axis
    ``u = (v_A + v_B)/‖v_A + v_B‖`` (fixed by the singles only). This dataclass carries
    the part the scalar *cannot* see: the component of ``r`` **orthogonal** to ``u``.

    - ``on_axis_interaction`` — ``r·u``; equals :attr:`EpistasisFit.interaction` (the
      scalar NUDGE reports), given here so the ratio is self-contained.
    - ``off_axis_residual`` — ``‖r − (r·u)·u‖``: the magnitude of the interaction
      residual orthogonal to the additive axis, in the same 2000-HVG ``log1p`` space.
    - ``neomorphic_ratio`` — ``off_axis_residual / max(|on_axis_interaction|, ε)``: how
      large the invisible/emergent component is relative to the on-axis interaction.

    A large ratio is an **under-count warning** — NUDGE's scalar is a direction-correct
    but magnitude-incomplete slice of a larger, possibly-emergent state (Norman 2019's
    *neomorphic* dimension). It is **NOT** a hidden-node claim or a positive discovery
    of a new mechanism (NUDGE-LIM-009). Only defined for the projection extractor;
    ``None`` in the fixed-signature scoring mode (there is no additive axis to project).
    """

    on_axis_interaction: float
    off_axis_residual: float
    neomorphic_ratio: float
    n_top_genes: int | None = None


@dataclass(frozen=True)
class EpistasisFit:
    """A two-perturbation interaction fit + everything :func:`classify_synergy` needs.

    Effects are shifts of a signature score vs control (``effect_space``), so the
    additive null ``additive_pred = effect_a + effect_b`` is Bliss-style.
    ``interaction`` is ``effect_ab − additive_pred``; ``ci_interaction`` is a percentile
    CI from a
    bootstrap over cells. ``bic_additive`` / ``bic_free`` are the BIC of the A+B cells
    under the fixed additive prediction vs a free A+B mean — the parsimony gate for
    whether the combo *earns* a distinct level.
    """

    effect_a: float
    effect_b: float
    effect_ab: float
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]
    ci_ab: tuple[float, float]
    additive_pred: float
    interaction: float
    ci_interaction: tuple[float, float]
    bic_additive: float
    bic_free: float
    n_control: int
    n_a: int
    n_b: int
    n_ab: int
    n_boot: int
    effect_space: str
    boot_interaction: tuple[float, ...] = ()
    off_axis_residual: float | None = None
    neomorphic_ratio: float | None = None


@dataclass(frozen=True)
class EpistasisResult:
    """A fit plus its conservative verdict and the human-readable reason."""

    fit: EpistasisFit
    call: str
    reason: str

    @property
    def is_reliable(self) -> bool:
        """Trustworthy enough to invert (``design()``'s integrity gate).

        A resolved interaction call (``additive`` / ``synergistic`` / ``buffering``) is
        reliable; the abstentions (``unresolved`` / ``no-effect``) are not; satisfies
        :class:`~nudge.design.invert.AttributionResult` protocol additively.
        """
        return self.call not in {"unresolved", "no-effect"}


def _percentile_ci(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def combo_geometry(
    v_a: Any, v_b: Any, v_ab: Any, *, n_top_genes: int | None = None, eps: float = 1e-9
) -> ComboGeometry:
    """Decompose a combination's interaction residual into on- vs off-axis parts.

    ``v_a`` / ``v_b`` / ``v_ab`` are the single/combo **mean-shift vectors** vs control
    (``mean(X) − mean(control)``) in some feature space (the 2000-HVG ``log1p`` space
    when supplied by :func:`~nudge.inference.bridge.combo_effect_scores`). The additive
    axis ``u = (v_a + v_b)/‖v_a + v_b‖`` is fixed by the singles only. Returns the
    :class:`ComboGeometry`: the on-axis interaction ``r·u`` (the scalar NUDGE reports),
    the **off-axis residual** ``‖r − (r·u)·u‖`` (the emergent piece the scalar discards,
    ``r = v_ab − v_a − v_b``), and their ratio. Raises ``ValueError`` if the singles
    produce no net shift (the additive axis is undefined).
    """
    v_a = np.asarray(v_a, dtype=float).ravel()
    v_b = np.asarray(v_b, dtype=float).ravel()
    v_ab = np.asarray(v_ab, dtype=float).ravel()
    u = v_a + v_b
    nu = float(np.linalg.norm(u))
    if nu <= 1e-12:
        raise ValueError(
            "the two single arms produce no net shift (additive axis undefined) — "
            "cannot compute the off-axis diagnostic"
        )
    u = u / nu
    r = v_ab - v_a - v_b
    on_axis = float(r @ u)
    off_axis = float(np.linalg.norm(r - on_axis * u))
    ratio = off_axis / max(abs(on_axis), eps)
    return ComboGeometry(
        on_axis_interaction=on_axis,
        off_axis_residual=off_axis,
        neomorphic_ratio=ratio,
        n_top_genes=n_top_genes,
    )


def fit_synergy(
    control: Any,
    a: Any,
    b: Any,
    ab: Any,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    effect_space: str = "log-fold-change",
    geometry: ComboGeometry | None = None,
) -> EpistasisFit:
    """Fit the additive null + the observed interaction for a combination A / B / A+B.

    ``control`` / ``a`` / ``b`` / ``ab`` are 1-D arrays of per-cell **scalar scores**
    (one value per cell) — a signature's log-normalized magnitude, produced by
    :func:`~nudge.inference.bridge.combo_effect_scores` (which projects each cell onto
    the additive direction fixed by the two single arms, so a positive interaction is
    super-additive *along the axis the singles push*). Effects are control-referenced
    shifts (``effect_x = mean(x) − mean(control)``); the additive prediction is
    ``effect_a + effect_b``; the interaction is ``effect_ab − additive_pred``. CIs come
    from a seeded bootstrap resampling cells **within each condition**. Raises
    ``ValueError`` if any condition is empty.

    ``geometry`` is an optional :class:`ComboGeometry` (from
    :func:`~nudge.inference.bridge.combo_effect_scores`): if given, its
    ``off_axis_residual`` / ``neomorphic_ratio`` are attached to the fit as the
    **possible-neomorphic off-axis diagnostic**. This is purely additive — the scalar
    fit is identical whether or not it is supplied, so the synthetic scalar-array path
    (no HVG vectors) is unchanged.
    """
    control = np.asarray(control, dtype=float).ravel()
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    ab = np.asarray(ab, dtype=float).ravel()
    for name, arr in (("control", control), ("A", a), ("B", b), ("A+B", ab)):
        if arr.size == 0:
            raise ValueError(f"condition {name!r} has no cells")

    c0 = float(control.mean())
    effect_a = float(a.mean()) - c0
    effect_b = float(b.mean()) - c0
    effect_ab = float(ab.mean()) - c0
    additive_pred = effect_a + effect_b
    interaction = effect_ab - additive_pred

    # BIC parsimony gate on the A+B cells: does a FREE A+B level beat the fixed additive
    # prediction? The additive model has no free parameter for the combo (its level is
    # pinned by control + the two singles); the free model spends one on the A+B mean.
    pred_add = c0 + additive_pred
    rss_add = float(np.sum((ab - pred_add) ** 2))
    rss_free = float(np.sum((ab - ab.mean()) ** 2))
    n_ab = int(ab.size)
    bic_additive = _bic(rss_add, n_ab, 0 + 1)  # variance only
    bic_free = _bic(rss_free, n_ab, 1 + 1)  # free A+B mean + variance

    rng = np.random.default_rng(seed)
    boot_i: list[float] = []
    boot_a: list[float] = []
    boot_b: list[float] = []
    boot_ab: list[float] = []
    for _ in range(max(n_boot, 0)):
        cs = control[rng.integers(0, control.size, control.size)].mean()
        as_ = a[rng.integers(0, a.size, a.size)].mean()
        bs = b[rng.integers(0, b.size, b.size)].mean()
        abs_ = ab[rng.integers(0, ab.size, ab.size)].mean()
        ea, eb, eab = as_ - cs, bs - cs, abs_ - cs
        boot_a.append(float(ea))
        boot_b.append(float(eb))
        boot_ab.append(float(eab))
        boot_i.append(float(eab - (ea + eb)))

    return EpistasisFit(
        effect_a=effect_a,
        effect_b=effect_b,
        effect_ab=effect_ab,
        ci_a=_percentile_ci(boot_a),
        ci_b=_percentile_ci(boot_b),
        ci_ab=_percentile_ci(boot_ab),
        additive_pred=additive_pred,
        interaction=interaction,
        ci_interaction=_percentile_ci(boot_i),
        bic_additive=bic_additive,
        bic_free=bic_free,
        n_control=int(control.size),
        n_a=int(a.size),
        n_b=int(b.size),
        n_ab=n_ab,
        n_boot=len(boot_i),
        effect_space=effect_space,
        boot_interaction=tuple(boot_i),
        off_axis_residual=(
            None if geometry is None else geometry.off_axis_residual
        ),
        neomorphic_ratio=(
            None if geometry is None else geometry.neomorphic_ratio
        ),
    )


def _straddles_zero(ci: tuple[float, float]) -> bool:
    lo, hi = ci
    return bool(lo <= 0.0 <= hi)


def _neomorphic_note(fit: EpistasisFit, threshold: float) -> str:
    """A possible-neomorphic off-axis warning to append to a non-additive reason.

    Returns ``""`` unless the off-axis diagnostic is available *and* its
    ``neomorphic_ratio`` clears ``threshold`` (off-axis residual ≥ ``threshold`` × the
    on-axis interaction). Never changes the call — it flags that the scalar may
    **under-count** an off-axis component, and — load-bearing honesty — that a large
    off-axis residual is EQUALLY consistent with a batch / ambient / technical artifact
    aligned with the A+B condition, so it must NOT be read as corroboration of a real
    interaction (NUDGE-LIM-009 red-team Hole 2: an additive ambient offset on A+B fakes
    synergy *and* raises this ratio — the flag would misdirect toward "emergent biology"
    on a boring batch artifact). It is a caveat, never a discovery or a hidden-node claim.
    """
    ratio = fit.neomorphic_ratio
    off = fit.off_axis_residual
    if ratio is None or off is None or not np.isfinite(ratio) or ratio < threshold:
        return ""
    return (
        f" | large off-axis residual (‖ = {off:.2f} ≈ {ratio:.1f}× the on-axis "
        f"interaction): the scalar may UNDER-count an emergent (possible-neomorphic) "
        f"component — BUT an off-axis residual is EQUALLY consistent with a batch / "
        f"ambient / technical artifact aligned with the A+B condition (NUDGE-LIM-009), "
        f"is NOT by itself evidence of emergent biology, and must not be read as "
        f"corroboration of the interaction"
    )


def classify_synergy(
    fit: EpistasisFit,
    *,
    bic_margin: float = 2.0,
    min_cells: int = 30,
    rel_width: float = 0.5,
    neomorphic_ratio_threshold: float = 1.0,
) -> tuple[str, str]:
    """Turn a fit into a conservative verdict + reason (the fail-safe classifier).

    Gates, most-conservative first: **unresolved** if any condition has fewer than
    ``min_cells`` cells (an underpowered arm we cannot trust) or the interaction CI is
    undefined; **no-effect** if *both* single-arm CIs straddle 0 (neither perturbation
    moved the signature — nothing to attribute); **additive** if the interaction CI
    straddles 0 *and* is tight (half-width ≤ ``rel_width`` × the single-arm scale) and a
    free A+B level does not beat the additive null by ``bic_margin``;
    **synergistic** / **buffering** only when the interaction CI is entirely above /
    below 0 *and* the free model beats the additive null by ``bic_margin`` (both the CI
    *and* parsimony must agree); otherwise **unresolved** — a wide CI that cannot rule
    out synergy, or a nonzero CI the BIC cannot justify. Returns ``(call, reason)``.

    **Off-axis diagnostic (additive; never changes the call).** When the call is
    ``synergistic`` / ``buffering`` and the fit carries the off-axis diagnostic (from
    :func:`~nudge.inference.bridge.combo_effect_scores`) with ``neomorphic_ratio ≥
    neomorphic_ratio_threshold`` (default ``1.0``: the off-axis residual is at least as
    large as the on-axis interaction NUDGE reports), a **possible-neomorphic** warning
    is appended to the reason — the scalar may UNDER-count an emergent/off-axis
    component (NUDGE-LIM-009). The threshold defaults to 1.0 because at that point the
    invisible
    (off-axis) piece equals or exceeds the reported (on-axis) one, so the scalar is at
    best half the story; below it the scalar captures the majority and the warning would
    over-fire. It is a warning, not a discovery — never a hidden-node claim.
    """
    n_min = min(fit.n_control, fit.n_a, fit.n_b, fit.n_ab)
    if n_min < min_cells:
        return "unresolved", (
            f"underpowered — the smallest condition has {n_min} cells "
            f"(< {min_cells}); an interaction inherits its weakest single arm, so the "
            "combo cannot be attributed (NUDGE-LIM-009)"
        )
    lo, hi = fit.ci_interaction
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "unresolved", (
            "the bootstrap CI on the interaction is undefined (an unstable resample)"
        )

    if _straddles_zero(fit.ci_a) and _straddles_zero(fit.ci_b):
        return "no-effect", (
            f"neither single arm moves the signature above noise (effect A="
            f"{fit.effect_a:+.3g} CI [{fit.ci_a[0]:+.3g}, {fit.ci_a[1]:+.3g}]; "
            f"B={fit.effect_b:+.3g} CI [{fit.ci_b[0]:+.3g}, {fit.ci_b[1]:+.3g}]) — "
            "there is no interaction to attribute"
        )

    effect_scale = abs(fit.effect_a) + abs(fit.effect_b)
    half_width = 0.5 * (hi - lo)
    beats_additive = fit.bic_free < fit.bic_additive - bic_margin
    d_bic = fit.bic_additive - fit.bic_free

    if _straddles_zero(fit.ci_interaction):
        if half_width <= rel_width * max(effect_scale, 1e-9) and not beats_additive:
            return "additive", (
                f"interaction {fit.interaction:+.3g} (95% CI [{lo:+.3g}, {hi:+.3g}]) "
                f"straddles 0 and a free A+B level does not beat the additive null by "
                f"ΔBIC>{bic_margin:g} (ΔBIC={d_bic:.1f}) — the combo is the additive "
                f"sum of A and B ({fit.effect_space} space; same knob, more of it)"
            )
        return "unresolved", (
            f"interaction {fit.interaction:+.3g} (95% CI [{lo:+.3g}, {hi:+.3g}]) "
            f"straddles 0 but the CI half-width {half_width:.3g} exceeds "
            f"{rel_width:g}× the single-arm effect scale {effect_scale:.3g} — the "
            "combo is underpowered to separate additive from synergistic"
        )

    if lo > 0.0:
        if beats_additive:
            return "synergistic", (
                f"interaction {fit.interaction:+.3g} (95% CI [{lo:+.3g}, {hi:+.3g}]) "
                f"is clearly > 0 and a free A+B level beats the additive null by "
                f"ΔBIC={d_bic:.1f} — SUPER-ADDITIVE / synergistic ({fit.effect_space} "
                "space). NOT a hidden-node claim (NUDGE-LIM-009); presumes both single "
                "arms are correctly measured and an approximately-affine readout "
                "(NUDGE-LIM-006)" + _neomorphic_note(fit, neomorphic_ratio_threshold)
            )
        return "unresolved", (
            f"interaction CI [{lo:+.3g}, {hi:+.3g}] is above 0 but a free A+B level "
            f"does not earn its parameter over the additive null (ΔBIC={d_bic:.1f} < "
            f"{bic_margin:g}) — the parsimony gate declines the synergy call"
        )
    # hi < 0
    if beats_additive:
        return "buffering", (
            f"interaction {fit.interaction:+.3g} (95% CI [{lo:+.3g}, {hi:+.3g}]) is "
            f"clearly < 0 and a free A+B level beats the additive null by "
            f"ΔBIC={d_bic:.1f} — SUB-ADDITIVE / buffering / epistatic "
            f"({fit.effect_space} space). NOT a hidden-node claim (NUDGE-LIM-009); "
            "presumes both single arms are correctly measured and an "
            "approximately-affine readout (NUDGE-LIM-006)"
            + _neomorphic_note(fit, neomorphic_ratio_threshold)
        )
    return "unresolved", (
        f"interaction CI [{lo:+.3g}, {hi:+.3g}] is below 0 but a free A+B level does "
        f"not earn its parameter over the additive null (ΔBIC={d_bic:.1f} < "
        f"{bic_margin:g}) — the parsimony gate declines the buffering call"
    )


def attribute_synergy(
    control: Any,
    a: Any,
    b: Any,
    ab: Any,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    effect_space: str = "log-fold-change",
    bic_margin: float = 2.0,
    min_cells: int = 30,
    rel_width: float = 0.5,
    neomorphic_ratio_threshold: float = 1.0,
    geometry: ComboGeometry | None = None,
) -> EpistasisResult:
    """Fit + classify a combination in one call — the CLI / MCP entry point.

    Pass ``geometry`` (from :func:`~nudge.inference.bridge.combo_effect_scores` with
    ``return_geometry=True``) to attach the possible-neomorphic off-axis diagnostic and,
    for a non-additive call above ``neomorphic_ratio_threshold``, append its warning to
    the reason. Omitting it leaves the scalar fit unchanged (the synthetic path).
    """
    fit = fit_synergy(
        control,
        a,
        b,
        ab,
        n_boot=n_boot,
        seed=seed,
        effect_space=effect_space,
        geometry=geometry,
    )
    call, reason = classify_synergy(
        fit,
        bic_margin=bic_margin,
        min_cells=min_cells,
        rel_width=rel_width,
        neomorphic_ratio_threshold=neomorphic_ratio_threshold,
    )
    return EpistasisResult(fit=fit, call=call, reason=reason)
