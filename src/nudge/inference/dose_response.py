"""Mechanism attribution from a **dose-response curve** — one circuit, another view.

Single-cell bimodality (the Lyapunov path) and bulk / pseudobulk dose-response
ultrasensitivity are *two measurements of one Hill circuit*: both are read out in the
same **K (threshold) / n (gain) / v_max (ceiling)** vocabulary. Where a single-cell
distribution is not bimodal — so :mod:`nudge.inference.lyapunov` correctly abstains —
the ultrasensitivity can still live in the **dose-response**: a readout's response to a
graded perturbation dose. This module fits that curve with the *same* Hill primitive the
circuit vector field uses (:func:`~nudge.mechanisms.regulatory.hill_activation` /
:func:`~nudge.mechanisms.regulatory.hill_repression`) and gates the call with the *same*
parsimony / fail-safe discipline as :mod:`nudge.inference.model_select`.

In NUDGE's terms a dose axis is a set of operating points (cf.
:class:`~nudge.inference.lyapunov.OperatingPoint`): the Fisher result that a *second*
operating point breaks the gain⇄threshold degeneracy (``scripts/vv/FINDINGS.md``) is
exactly why a dose *series* can attribute a mechanism a single snapshot cannot.

**Honesty (the load-bearing rule).** The fitted ``n`` is an **apparent population gain**
with a CI — **not** molecular cooperativity. Pseudobulk conflates within-cell
cooperativity with a *spread of single-cell thresholds*, and a nonlinear readout can
manufacture apparent ultrasensitivity (NUDGE-LIM-006). So we report ``n`` + CI, classify
conservatively, and **abstain** rather than guess:

- ``"no-effect"`` — the response is flat within noise (an inert perturbation);
- ``"unresolved"`` — the curve is not Hill-like (low R² / too few points), *or* the
  doses do not span the inflection (``K`` outside the dose range → one arm of a sigmoid,
  where gain is unidentifiable), *or* the ``n`` CI straddles the switch/graded line;
- ``"graded"`` — a free ``n`` does not beat the ``n = 1`` model by the BIC margin, or
  the whole ``n`` CI sits at/below the ultrasensitive threshold;
- ``"switch"`` — free ``n`` **earns** its parameter over graded (ΔBIC) *and* the ``n``
  CI clears the ultrasensitive line (a conservative, two-condition call).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from nudge.mechanisms.regulatory import hill_activation, hill_repression

__all__ = [
    "DoseResponseFit",
    "DoseResponseResult",
    "fit_dose_response",
    "classify_dose_response",
    "attribute_dose_response",
]

Direction = str  # "repress" (response falls with dose) | "activate" (rises with dose)


def _prim(direction: str) -> Callable[..., Any]:
    if direction == "repress":
        return hill_repression
    if direction == "activate":
        return hill_activation
    raise ValueError(f"direction must be 'repress' or 'activate', got {direction!r}")


@lru_cache(maxsize=8)
def _jax_model(
    direction: str, free_n: bool
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Jitted ``predict`` + its exact param-Jacobian, per (direction, free_n).

    We reuse the circuit Hill primitive and differentiate it with JAX autodiff, then
    hand the analytic Jacobian to ``curve_fit``. Not incidental: JAX defaults to
    float32, so a finite-difference Jacobian (curve_fit's default) underflows in the
    ``n`` direction and the optimizer never moves the gain — an exact autodiff Jacobian
    is precise at float32 and keeps the fit reusing NUDGE's differentiable engine.
    """
    import jax
    import jax.numpy as jnp

    prim = _prim(direction)

    def predict(params: Any, dose: Any) -> Any:
        d = jnp.maximum(dose, 0.0)
        floor, amp, k = params[0], params[1], params[2]
        n = params[3] if free_n else 1.0
        return floor + prim(d, k, n, amp)

    return jax.jit(predict), jax.jit(jax.jacfwd(predict, argnums=0))


_K_QUANTILES = (0.2, 0.4, 0.6, 0.8)  # seed K across the dose range …
_N_SEEDS = (1.0, 2.0, 4.0, 8.0)  # … and n from graded to steeply switch-like


def _fit(
    direction: str,
    dose: np.ndarray,
    y: np.ndarray,
    *,
    free_n: bool,
    seeds: Sequence[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Robust least-squares Hill fit via multi-start. Returns ``(params, yhat, rss)``.

    ``params`` = ``[floor, amp, K, n]`` (n = 1.0 for the graded fit). The model is
    ``y = floor + hill(dose, K, n, amp)`` (the exact circuit Hill primitive) with the
    **exact autodiff Jacobian** from :func:`_jax_model` handed to ``curve_fit`` — needed
    because JAX is float32 and a finite-difference Jacobian underflows to zero in the
    ``n`` direction (the optimizer would leave the gain pinned at its init). On top of
    that we multi-start from a small grid of ``(K, n)`` seeds and keep the
    lowest-RSS one, so a genuine local optimum can't hand back a confident-wrong ``n``.
    ``seeds`` overrides the grid (used to warm-start the bootstrap from the MLE). Raises
    ``RuntimeError`` if every start fails to converge.
    """
    from scipy.optimize import curve_fit

    pred_fn, jac_fn = _jax_model(direction, free_n)
    dose = np.asarray(dose, dtype=float)
    y = np.asarray(y, dtype=float)
    yr = max(float(y.max() - y.min()), 1e-9)
    dmax = max(float(np.max(dose)), 1e-9)
    dpos = dose[dose > 0]

    def model(d: Any, *p: float) -> Any:
        pa, da = np.asarray(p, dtype=float), np.asarray(d, dtype=float)
        return np.asarray(pred_fn(pa, da))

    def jac(d: Any, *p: float) -> Any:
        pa, da = np.asarray(p, dtype=float), np.asarray(d, dtype=float)
        return np.asarray(jac_fn(pa, da))

    if free_n:
        lb = [float(y.min()) - yr, 0.0, 1e-3 * dmax, 0.3]
        ub = [float(y.max()) + yr, 3.0 * yr, 3.0 * dmax, 12.0]
    else:
        lb = [float(y.min()) - yr, 0.0, 1e-3 * dmax]
        ub = [float(y.max()) + yr, 3.0 * yr, 3.0 * dmax]

    if seeds is None:
        ks = (
            np.quantile(dpos, _K_QUANTILES)
            if dpos.size
            else np.array([0.5 * dmax])
        )
        ns = _N_SEEDS if free_n else (1.0,)
        seeds = [(float(k), float(n)) for k in ks for n in ns]

    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for k0, n0 in seeds:
        k0 = float(np.clip(k0, lb[2], ub[2]))
        p0 = [float(y.min()), yr, k0]
        if free_n:
            p0.append(float(np.clip(n0, 0.3, 12.0)))
        try:
            popt, _pcov = curve_fit(
                model, dose, y, p0=p0, bounds=(lb, ub), jac=jac, maxfev=20000
            )
        except Exception:
            continue
        yhat = np.asarray(model(dose, *popt), dtype=float)
        rss = float(np.sum((y - yhat) ** 2))
        params = np.asarray(popt) if free_n else np.array([*popt, 1.0], dtype=float)
        if best is None or rss < best[2]:
            best = (params, yhat, rss)
    if best is None:
        raise RuntimeError("Hill fit failed to converge from any start")
    return best


def _bic(rss: float, n_obs: int, k: float) -> float:
    """BIC for a Gaussian-residual least-squares fit with unknown variance.

    ``-2 log L = N(ln 2π + ln(RSS/N) + 1)``; ``k`` counts the free params **plus** the
    residual variance σ². Lower is better (Schwarz 1978) — the same penalty
    :mod:`nudge.inference.model_select` uses so the two paths gate consistently.
    """
    rss = max(rss, 1e-12)
    return k * np.log(n_obs) + n_obs * (np.log(2 * np.pi) + np.log(rss / n_obs) + 1.0)


@dataclass(frozen=True)
class DoseResponseFit:
    """A Hill dose-response fit + everything :func:`classify_dose_response` needs.

    ``n`` is an **apparent population gain** (with ``ci_n``), not molecular
    cooperativity.
    ``bic_switch`` / ``bic_graded`` are the free-n vs ``n = 1`` BIC;
    ``spans_inflection`` is
    whether ``K`` lies within the observed dose range (else gain is unidentifiable).
    """

    direction: str
    n: float
    k_threshold: float
    amp: float
    floor: float
    r2: float
    graded_r2: float
    bic_switch: float
    bic_graded: float
    ci_n: tuple[float, float]
    ci_k: tuple[float, float]
    n_points: int
    n_boot: int
    dose_min: float
    dose_max: float
    spans_inflection: bool
    noise: float
    boot_n: tuple[float, ...] = ()
    #: Bootstrap CIs on the response span (``amp``) and baseline (``floor``) — the
    #: **ceiling / leakiness** axis, used by cross-modality knob attribution
    #: (:mod:`nudge.inference.cross_modality`); ``(nan, nan)`` if bootstrap is empty.
    ci_amp: tuple[float, float] = (float("nan"), float("nan"))
    ci_floor: tuple[float, float] = (float("nan"), float("nan"))


@dataclass(frozen=True)
class DoseResponseResult:
    """A fit plus its conservative verdict and the human-readable reason."""

    fit: DoseResponseFit
    call: str
    reason: str


def fit_dose_response(
    dose: Any,
    response: Any,
    *,
    direction: str = "repress",
    n_boot: int = 500,
    seed: int = 0,
) -> DoseResponseFit:
    """Fit ``response = floor + hill(dose, K, n, amp)`` + its ``n = 1`` graded sibling.

    ``dose`` / ``response`` are paired 1-D arrays (one point per dose level / guide /
    condition). ``direction`` is ``"repress"`` when the readout *falls* with dose
    (e.g. a self-renewal signature vs knockdown) or ``"activate"`` when it rises.
    CIs on ``n`` and ``K`` come from a seeded bootstrap over the observations; the fit
    reuses the exact circuit Hill primitive. Raises on fewer than 4 points (a Hill curve
    has 4 parameters — abstaining there is the caller's job via a small-``n_points``
    check).
    """
    dose = np.asarray(dose, dtype=float).ravel()
    response = np.asarray(response, dtype=float).ravel()
    if dose.shape != response.shape:
        raise ValueError("dose and response must be the same length")
    if dose.size < 4:
        raise ValueError(f"need >= 4 dose points to fit a Hill curve, got {dose.size}")

    params, _yhat, rss = _fit(direction, dose, response, free_n=True)
    floor, amp, k_threshold, n = (float(v) for v in params)
    _gp, _gyhat, grss = _fit(direction, dose, response, free_n=False)

    n_obs = int(dose.size)
    tss = max(float(np.sum((response - response.mean()) ** 2)), 1e-12)
    r2 = 1.0 - rss / tss
    graded_r2 = 1.0 - grss / tss
    bic_switch = _bic(rss, n_obs, 4 + 1)  # floor, amp, K, n, sigma
    bic_graded = _bic(grss, n_obs, 3 + 1)  # floor, amp, K, sigma
    noise = float(np.sqrt(rss / max(n_obs - 4, 1)))

    # Bootstrap warm-started from the MLE (plus graded/steep alternatives) so each
    # resample fit is fast and stable, without re-scanning the full seed grid.
    boot_seeds = [(k_threshold, n), (k_threshold, 1.0), (k_threshold, 4.0)]
    rng = np.random.default_rng(seed)
    boot_n: list[float] = []
    boot_k: list[float] = []
    boot_amp: list[float] = []
    boot_floor: list[float] = []
    for _ in range(max(n_boot, 0)):
        idx = rng.integers(0, n_obs, n_obs)
        try:
            bp, _by, _brss = _fit(
                direction, dose[idx], response[idx], free_n=True, seeds=boot_seeds
            )
        except Exception:
            continue
        boot_n.append(float(bp[3]))
        boot_k.append(float(bp[2]))
        boot_floor.append(float(bp[0]))
        boot_amp.append(float(bp[1]))

    def _ci(samples: list[float]) -> tuple[float, float]:
        if not samples:
            return (float("nan"), float("nan"))
        return (
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        )

    ci_n = _ci(boot_n)
    ci_k = _ci(boot_k)
    ci_amp = _ci(boot_amp)
    ci_floor = _ci(boot_floor)

    dose_min, dose_max = float(dose.min()), float(dose.max())
    spans = bool(dose_min <= k_threshold <= dose_max)
    return DoseResponseFit(
        direction=direction,
        n=n,
        k_threshold=k_threshold,
        amp=float(amp),
        floor=float(floor),
        r2=r2,
        graded_r2=graded_r2,
        bic_switch=bic_switch,
        bic_graded=bic_graded,
        ci_n=ci_n,
        ci_k=ci_k,
        n_points=n_obs,
        n_boot=len(boot_n),
        dose_min=dose_min,
        dose_max=dose_max,
        spans_inflection=spans,
        noise=noise,
        boot_n=tuple(boot_n),
        ci_amp=ci_amp,
        ci_floor=ci_floor,
    )


def classify_dose_response(
    fit: DoseResponseFit,
    *,
    n_switch: float = 2.0,
    min_r2: float = 0.5,
    bic_margin: float = 2.0,
    noise_amp_ratio: float = 2.0,
) -> tuple[str, str]:
    """Turn a fit into a conservative verdict: switch / graded / no-effect / unresolved.

    The gates, in order (fail-safe first): **no-effect** if the fitted amplitude is
    within
    ``noise_amp_ratio`` × residual noise; **unresolved** if even the better of the two
    models has ``R² < min_r2``, if the bootstrap ``n`` CI is undefined, or if the doses
    do
    not span the inflection (``K`` outside the dose range — one arm of a sigmoid);
    **switch** only if free-n beats graded by ``bic_margin`` *and* the whole ``n`` CI
    clears
    ``n_switch``; **graded** if the ``n`` CI sits at/below ``n_switch`` or free-n is not
    justified; otherwise **unresolved** (the CI straddles the line). Returns
    ``(call, reason)``.
    """
    if fit.amp < noise_amp_ratio * fit.noise:
        return "no-effect", (
            f"response amplitude {fit.amp:.3g} is within ~{noise_amp_ratio:g}x the "
            f"residual noise ({fit.noise:.3g}) — an inert / flat curve; nothing to fit"
        )

    best_r2 = max(fit.r2, fit.graded_r2)
    if best_r2 < min_r2:
        return "unresolved", (
            f"best fit R²={best_r2:.2f} < {min_r2} — the curve is not Hill-like "
            "(too noisy or too few dose points); gain/threshold are unidentifiable"
        )
    lo, hi = fit.ci_n
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "unresolved", (
            "the bootstrap CI on n is undefined (the fit is unstable under resampling)"
        )
    if not fit.spans_inflection:
        return "unresolved", (
            f"K={fit.k_threshold:.3g} lies outside the dose range "
            f"[{fit.dose_min:.3g}, {fit.dose_max:.3g}] — the doses do not span the "
            "inflection, so a Hill fit sees only one arm and gain is unidentifiable"
        )

    beats_graded = fit.bic_switch < fit.bic_graded - bic_margin
    d_bic = fit.bic_graded - fit.bic_switch
    if beats_graded and lo > n_switch:
        return "switch", (
            f"apparent gain n={fit.n:.2f} (95% CI {lo:.2f}–{hi:.2f}); CI clears the "
            f"ultrasensitive line n>{n_switch:g} and free-n beats graded by "
            f"ΔBIC={d_bic:.1f} — ultrasensitive/switch-like (APPARENT population gain, "
            "not molecular cooperativity: pseudobulk conflates cooperativity with a "
            "spread of single-cell thresholds)"
        )
    if hi <= n_switch or not beats_graded:
        return "graded", (
            f"apparent gain n={fit.n:.2f} (95% CI {lo:.2f}–{hi:.2f}) is consistent "
            f"with graded (n≈1) — free-n does not beat n=1 by ΔBIC>{bic_margin:g} "
            f"(ΔBIC={d_bic:.1f}); a poor switch fit IS the graded signature"
        )
    return "unresolved", (
        f"apparent gain n={fit.n:.2f} (95% CI {lo:.2f}–{hi:.2f}) straddles the "
        f"switch/graded line (n={n_switch:g}) — the data cannot resolve which"
    )


def attribute_dose_response(
    dose: Any,
    response: Any,
    *,
    direction: str = "repress",
    n_boot: int = 500,
    seed: int = 0,
    n_switch: float = 2.0,
    min_r2: float = 0.5,
    bic_margin: float = 2.0,
    noise_amp_ratio: float = 2.0,
) -> DoseResponseResult:
    """Fit + classify in one call — the convenience entry point for the CLI / MCP."""
    fit = fit_dose_response(
        dose, response, direction=direction, n_boot=n_boot, seed=seed
    )
    call, reason = classify_dose_response(
        fit,
        n_switch=n_switch,
        min_r2=min_r2,
        bic_margin=bic_margin,
        noise_amp_ratio=noise_amp_ratio,
    )
    return DoseResponseResult(fit=fit, call=call, reason=reason)
