"""Sloppiness diagnostic — separate *sloppy-but-predictive* from *unidentifiable*.

**The subtlety a 0-shot identifiability test gets wrong.** The obvious way to ask "are my
parameters identifiable?" is to look at the Fisher/Hessian **eigenvalue spectrum** and
declare the model unidentifiable when the condition number is huge or the spectrum spans
many decades. That test is **unreliable** — and for *sloppy* models it gives the WRONG
answer (``design/DEEP_RESEARCH_drug_discovery_directions.md`` findings 6–7; Transtrum
2014; Chis 2011). A **sloppy** model has a Fisher spectrum spanning many orders of
magnitude — its individual parameters are poorly constrained — yet it is often perfectly
**structurally/practically identifiable** and its **predictions are tightly constrained**.
Calling it "unidentifiable" from the eigenvalue gap alone would make NUDGE **over-abstain**
on a model it can actually use, and would tempt it to recommend a Fisher-greedy experiment
that *destroys* predictivity (finding 6).

**What this module measures instead.** Sloppiness ≠ non-identifiability, so we separate the
two questions the naive test conflates:

1. **Is a parameter direction genuinely unrecoverable?** — a *structural* null is an
   **exact** functional redundancy: moving along it changes **no** model output. We detect
   it from the **rank of the sensitivity matrix** (an SVD; Chis 2011) — a right-singular
   vector whose **prediction sensitivity ``‖J·v‖`` is ~0**. That is a true null (an
   eigenvalue at the machine-rank floor), categorically different from a *sloppy* direction
   whose eigenvalue is tiny-but-**finite** (its ``‖J·v‖`` is small but clearly non-zero —
   the data *does* constrain it, weakly).
2. **Are the predictions constrained?** — propagate the (guarded) parameter covariance
   ``Σ = FIM⁻¹`` through the prediction map ``J_pred`` and read the **prediction**
   uncertainty. Sloppy directions blow up ``Σ`` but map to ~0 prediction change, so a
   sloppy model stays **predictive**; a model whose loose directions *do* move the
   prediction is genuinely not usable.

The verdict is one of three (the calibrated-abstention target):

- **``unidentifiable``** — a true null/near-null direction (parameters not recoverable),
  or the loose directions also make predictions loose. NUDGE **abstains**, and names the
  unrecoverable parameter combination.
- **``sloppy-but-predictive``** — a wide Fisher spectrum (loose individual parameters) but
  **tight predictions** and no structural null. NUDGE should **not** abstain on this — its
  predictions/attributions are reliable; it just cannot pin every parameter, and a
  Fisher-greedy experiment targeting the sloppy directions is flagged as low/negative value
  (``fim_greedy_warning``; finding 6).
- **``well-constrained``** — a narrow spectrum; every parameter individually identifiable.

Each report also carries the ``naive_verdict`` (what a condition-number-only test would
say) and ``naive_is_wrong`` — the concrete demonstration that the eigenvalue gap alone
misclassifies a sloppy model. Validated on a canonical **sum-of-exponentials** (sloppy but
predictive) vs a **structurally-redundant** ``A·e^{-(k₁+k₂)t}`` (a true null) in
``scripts/vv/sloppiness_validation.py`` / ``tests/inference/test_sloppiness.py``.

Additive / opt-in: this reads a caller-supplied prediction function; it never touches the
frozen ``fit.py`` or ``core/``. It complements :mod:`nudge.inference.uncertainty` (the
local-curvature CIs) with the sloppiness-vs-identifiability distinction.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

__all__ = [
    "NullDirection",
    "SloppinessReport",
    "relative_sensitivity_jacobian",
    "fisher_information",
    "sloppiness_diagnostic",
    "analyze_model",
    "sum_of_exponentials_predict",
    "redundant_exponential_predict",
    "well_conditioned_predict",
]


@dataclass(frozen=True)
class NullDirection:
    """A parameter combination the data cannot recover — the actionable half of an
    ``unidentifiable`` verdict.

    ``vector`` is the unit right-singular vector (in **log-parameter** space) whose
    prediction sensitivity ``‖J·v‖`` is at the rank floor; ``param_loadings`` maps each
    parameter name to its component; ``prediction_sensitivity`` is the (near-zero)
    ``‖J·v‖`` that *earns* the "structural null" label; ``hint`` is the plain-language
    reading (e.g. *"k₁ and k₂ enter only through their sum"*).
    """

    vector: np.ndarray
    param_loadings: dict[str, float]
    prediction_sensitivity: float
    hint: str


@dataclass(frozen=True)
class SloppinessReport:
    """The sloppiness / identifiability verdict + the measurements that earned it."""

    label: str  # "well-constrained" | "sloppy-but-predictive" | "unidentifiable"
    param_names: tuple[str, ...]
    fim_eigenvalues: np.ndarray  # ascending (log-parameter Fisher)
    cond_number: float
    spectral_span_decades: float
    smallest_eigenvalue: float
    largest_eigenvalue: float
    n_sloppy_dims: int
    n_null_dims: int
    is_sloppy: bool
    predictive: bool
    max_prediction_std: float
    relative_prediction_std: float
    sloppy_prediction_variance_fraction: float
    naive_verdict: str
    naive_is_wrong: bool
    null_directions: tuple[NullDirection, ...]
    reason: str
    fim_greedy_warning: str | None


# --------------------------------------------------------------------------- #
# canonical validation models (predict_fn: theta -> output vector)
# --------------------------------------------------------------------------- #
def sum_of_exponentials_predict(
    rates: Sequence[float], amps: Sequence[float], t: np.ndarray
) -> Callable[[Array], Array]:
    """A **sum of exponentials** ``y(t) = Σ_m A_m e^{-k_m t}`` — the canonical *sloppy*
    model (Transtrum/Sethna). Distinct rates → structurally identifiable, but the Fisher
    spectrum spans many decades (individual ``A_m, k_m`` poorly constrained) while the
    fitted curve is tight. ``theta = [A_1, k_1, A_2, k_2, …]`` (positive)."""
    t_j = jnp.asarray(np.asarray(t))
    m = len(rates)

    def predict(theta: Array) -> Array:
        a = theta[0::2]
        k = theta[1::2]
        return jnp.sum(a[:, None] * jnp.exp(-k[:, None] * t_j[None, :]), axis=0)

    predict.theta0 = np.array(  # type: ignore[attr-defined]
        [v for pair in zip(amps, rates, strict=True) for v in pair], dtype=np.float64
    )
    predict.names = tuple(  # type: ignore[attr-defined]
        n for i in range(m) for n in (f"A{i + 1}", f"k{i + 1}")
    )
    return predict


def redundant_exponential_predict(
    amp: float, k1: float, k2: float, t: np.ndarray
) -> Callable[[Array], Array]:
    """A **structurally unidentifiable** model ``y(t) = A e^{-(k₁+k₂) t}``: only the SUM
    ``k₁+k₂`` enters, so the ``(k₁, k₂)`` anti-diagonal is an **exact null direction** —
    the data cannot recover ``k₁`` and ``k₂`` separately no matter how much of it there is.
    ``theta = [A, k_1, k_2]``."""
    t_j = jnp.asarray(np.asarray(t))

    def predict(theta: Array) -> Array:
        a, kk1, kk2 = theta[0], theta[1], theta[2]
        return a * jnp.exp(-(kk1 + kk2) * t_j)

    predict.theta0 = np.array([amp, k1, k2], dtype=np.float64)  # type: ignore[attr-defined]
    predict.names = ("A", "k1", "k2")  # type: ignore[attr-defined]
    return predict


def well_conditioned_predict(
    slope: float, offset: float, t: np.ndarray
) -> Callable[[Array], Array]:
    """A **well-conditioned** control ``y(t) = offset + slope·t`` sampled over a spread of
    ``t`` — both parameters individually identifiable (a narrow Fisher spectrum).
    ``theta = [offset, slope]``."""
    t_j = jnp.asarray(np.asarray(t))

    def predict(theta: Array) -> Array:
        return theta[0] + theta[1] * t_j

    predict.theta0 = np.array([offset, slope], dtype=np.float64)  # type: ignore[attr-defined]
    predict.names = ("offset", "slope")  # type: ignore[attr-defined]
    return predict


# --------------------------------------------------------------------------- #
# the sensitivity matrix + Fisher information (log-parameter space)
# --------------------------------------------------------------------------- #
def relative_sensitivity_jacobian(
    predict_fn: Callable[[Array], Array], theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The **relative** (log-parameter) sensitivity matrix ``J_ij = ∂yᵢ/∂ log θⱼ`` + ``y``.

    Sloppy-model analysis is conventionally done in **log parameters** (dimensionless,
    scale-free), where ``∂y/∂ log θ_j = θ_j · ∂y/∂θ_j``. Returns ``(J (n_obs, n_theta), y
    (n_obs,))``, computed by forward-mode autodiff of ``predict_fn``.
    """
    theta_j = jnp.asarray(theta, jnp.float64) if theta.dtype == np.float64 else jnp.asarray(
        theta, jnp.float32
    )
    y = predict_fn(theta_j)
    jac = jax.jacfwd(predict_fn)(theta_j)  # (n_obs, n_theta) = ∂y/∂θ
    jac_log = np.asarray(jac) * np.asarray(theta)[None, :]  # ∂y/∂ log θ
    return np.asarray(jac_log, dtype=np.float64), np.asarray(y, dtype=np.float64)


def fisher_information(
    jac_log: np.ndarray, sigma: float
) -> np.ndarray:
    """The Fisher information ``FIM = Jᵀ J / σ²`` for iid Gaussian noise ``σ`` (log-params).

    For a least-squares fit with observation noise ``σ`` the observed Fisher information of
    the (log-)parameters is ``Jᵀ J / σ²`` — the curvature of the Gaussian NLL. Its
    eigenvalues are the sloppy spectrum.
    """
    return (jac_log.T @ jac_log) / (float(sigma) ** 2)


# --------------------------------------------------------------------------- #
# the diagnostic
# --------------------------------------------------------------------------- #
def _guarded_inverse(fim: np.ndarray, ridge: float) -> np.ndarray:
    """A ridge-floored eigen-inverse of the FIM — a flat/near-null direction gets a
    large-but-**finite** variance (never a NaN, never the pseudo-inverse mistake of
    *zeroing* the flat direction). Mirrors ``nudge.inference.uncertainty.laplace_posterior``."""
    evals, evecs = np.linalg.eigh(0.5 * (fim + fim.T))
    lam_max = float(evals[-1]) if evals[-1] > 0.0 else 1.0
    floor = ridge * lam_max
    inv = 1.0 / np.maximum(evals, floor)
    return (evecs * inv) @ evecs.T


def sloppiness_diagnostic(
    jac_log: np.ndarray,
    y: np.ndarray,
    sigma: float,
    param_names: Sequence[str] | None = None,
    *,
    jac_pred_log: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    rank_rtol: float = 1e-7,
    sloppy_decade_threshold: float = 3.0,
    pred_rel_tol: float = 0.05,
    naive_cond_threshold: float = 1e6,
    ridge: float = 1e-10,
) -> SloppinessReport:
    """Classify a model as ``well-constrained`` / ``sloppy-but-predictive`` /
    ``unidentifiable`` from its sensitivity matrix.

    ``jac_log`` is the fit observables' relative-sensitivity matrix (builds the FIM);
    ``jac_pred_log`` / ``y_pred`` are the **prediction of interest**'s sensitivity + value
    (default: the fit observables themselves — the self-consistent case). Thresholds:

    - ``rank_rtol`` — a singular value ``σ_j/σ_1 < rank_rtol`` is a **structural null**
      (a true redundancy; its prediction sensitivity ``‖J·v‖`` is at the rank floor). This
      cleanly separates a machine-rank null (~1e-15 relative) from a *sloppy* small
      singular value (typically ≳1e-4 relative), so it is NOT the naive eigenvalue-gap test.
    - ``sloppy_decade_threshold`` — the FIM spectrum spans more than this many decades → the
      model is *sloppy* (a diagnostic flag, not a verdict).
    ``param_names`` labels the parameters (defaults to ``θ0…θ{n-1}`` when omitted).

    - ``pred_rel_tol`` — predictions are "tight" when the max propagated prediction std is
      below this fraction of the prediction RMS.
    - ``naive_cond_threshold`` — the strawman: a condition-number-only test calls the model
      unidentifiable above this. ``naive_is_wrong`` records where it disagrees with the
      measured verdict.
    """
    p = jac_log.shape[1]
    names = tuple(param_names) if param_names is not None else tuple(f"θ{i}" for i in range(p))
    fim = fisher_information(jac_log, sigma)
    evals = np.linalg.eigvalsh(0.5 * (fim + fim.T))
    evals = np.clip(evals, 0.0, None)
    evals_sorted = np.sort(evals)  # ascending
    lam_min = float(evals_sorted[0])
    lam_max = float(evals_sorted[-1])
    cond = np.inf if lam_min <= 0.0 else lam_max / lam_min
    span_decades = float(np.log10(lam_max / lam_min)) if lam_min > 0.0 else np.inf

    # --- structural nulls: rank of the sensitivity matrix (SVD; Chis 2011) --------- #
    # right-singular vectors with sv/sv_max < rank_rtol are exact redundancies.
    _u, svals, vt = np.linalg.svd(jac_log, full_matrices=False)
    sv_max = float(svals[0]) if svals.size and svals[0] > 0.0 else 1.0
    null_mask = svals < rank_rtol * sv_max
    # pad for the underdetermined case (fewer obs than params -> extra exact nulls).
    n_null = int(null_mask.sum()) + max(p - svals.size, 0)

    null_dirs: list[NullDirection] = []
    for j in np.where(null_mask)[0]:
        v = np.asarray(vt[j], dtype=np.float64)
        dom = int(np.argmax(np.abs(v)))
        if v[dom] < 0.0:
            v = -v
        loadings = {names[i]: float(v[i]) for i in range(p)}
        null_dirs.append(
            NullDirection(
                vector=v,
                param_loadings=loadings,
                prediction_sensitivity=float(svals[j]),
                hint=_null_hint(loadings),
            )
        )

    # --- prediction uncertainty: propagate Σ = FIM⁻¹ through the prediction map ----- #
    jpred = jac_log if jac_pred_log is None else jac_pred_log
    ypred = y if y_pred is None else y_pred
    cov = _guarded_inverse(fim, ridge)
    pred_cov = jpred @ cov @ jpred.T
    pred_var = np.clip(np.diag(pred_cov), 0.0, None)
    max_pred_std = float(np.sqrt(pred_var.max())) if pred_var.size else 0.0
    pred_rms = float(np.sqrt(np.mean(np.asarray(ypred) ** 2))) + 1e-12
    rel_pred_std = max_pred_std / pred_rms
    predictive = rel_pred_std < pred_rel_tol

    # fraction of prediction variance carried by the SLOPPY (small-eigenvalue) directions:
    # if tiny, a Fisher-greedy experiment that constrains them buys ~no predictive gain.
    frac = _sloppy_prediction_fraction(fim, jpred, ridge, sloppy_decade_threshold)

    is_sloppy = span_decades > sloppy_decade_threshold
    n_sloppy = int(
        np.sum(evals_sorted < (lam_max * 10.0 ** (-sloppy_decade_threshold)))
    )

    # --- the verdict ---------------------------------------------------------------- #
    fim_greedy_warning: str | None = None
    if n_null > 0:
        label = "unidentifiable"
        reason = (
            f"STRUCTURAL non-identifiability: {n_null} sensitivity-matrix null "
            f"direction(s) (singular value/σ_max < {rank_rtol:g}) — a parameter "
            "combination the data cannot recover no matter how much is collected. "
            + (null_dirs[0].hint if null_dirs else "")
        )
    elif is_sloppy and predictive:
        label = "sloppy-but-predictive"
        reason = (
            f"SLOPPY but PREDICTIVE: the Fisher spectrum spans {span_decades:.1f} decades "
            f"(cond {cond:.2e}) — individual parameters are poorly constrained — yet the "
            f"prediction is tight (max relative prediction std {rel_pred_std:.2%} < "
            f"{pred_rel_tol:.0%}) and there is NO structural null. The model is usable: "
            "do not abstain, and do not naively Fisher-optimize the sloppy directions."
        )
        fim_greedy_warning = (
            f"Only {frac:.1%} of the prediction variance comes from the sloppy directions, "
            "so a Fisher-greedy experiment that constrains them adds ~no predictive value "
            "and risks destroying predictivity by over-fitting previously-loose parameters "
            "where model discrepancy lives (deep-research finding 6)."
        )
    elif is_sloppy and not predictive:
        label = "unidentifiable"
        reason = (
            f"PRACTICALLY non-identifiable: the Fisher spectrum spans {span_decades:.1f} "
            f"decades AND the loose directions make predictions loose (max relative "
            f"prediction std {rel_pred_std:.2%} ≥ {pred_rel_tol:.0%}) — parameters not "
            "recoverable and predictions not trustworthy. Abstain."
        )
    else:
        label = "well-constrained"
        reason = (
            f"WELL-CONSTRAINED: the Fisher spectrum spans only {span_decades:.1f} decades "
            f"(cond {cond:.2e}); every parameter is individually identifiable and the "
            f"prediction is tight (max relative prediction std {rel_pred_std:.2%})."
        )

    naive_verdict = (
        "unidentifiable"
        if (not np.isfinite(cond)) or cond > naive_cond_threshold
        else "identifiable"
    )
    # the naive cond-number test is WRONG when it calls a usable model unidentifiable.
    naive_is_wrong = naive_verdict == "unidentifiable" and label in (
        "sloppy-but-predictive",
        "well-constrained",
    )

    return SloppinessReport(
        label=label,
        param_names=names,
        fim_eigenvalues=evals_sorted,
        cond_number=float(cond),
        spectral_span_decades=span_decades,
        smallest_eigenvalue=lam_min,
        largest_eigenvalue=lam_max,
        n_sloppy_dims=n_sloppy,
        n_null_dims=n_null,
        is_sloppy=is_sloppy,
        predictive=predictive,
        max_prediction_std=max_pred_std,
        relative_prediction_std=rel_pred_std,
        sloppy_prediction_variance_fraction=frac,
        naive_verdict=naive_verdict,
        naive_is_wrong=naive_is_wrong,
        null_directions=tuple(null_dirs),
        reason=reason,
        fim_greedy_warning=fim_greedy_warning,
    )


def _sloppy_prediction_fraction(
    fim: np.ndarray, jpred: np.ndarray, ridge: float, decade_threshold: float
) -> float:
    """Fraction of total prediction variance contributed by the sloppy (small-eigenvalue)
    FIM eigendirections — a measured proxy for how much a Fisher-greedy experiment could
    even help predictions."""
    evals, evecs = np.linalg.eigh(0.5 * (fim + fim.T))
    lam_max = float(evals[-1]) if evals[-1] > 0.0 else 1.0
    floor = ridge * lam_max
    inv = 1.0 / np.maximum(evals, floor)
    # per-eigendirection prediction variance contribution: inv_i * ||jpred @ v_i||^2
    contrib = inv * np.sum((jpred @ evecs) ** 2, axis=0)
    total = float(contrib.sum())
    if total <= 0.0:
        return 0.0
    sloppy = evals < (lam_max * 10.0 ** (-decade_threshold))
    return float(contrib[sloppy].sum() / total)


def _null_hint(loadings: dict[str, float]) -> str:
    """Read a null-direction's loadings into a plain-language redundancy statement."""
    items = sorted(loadings.items(), key=lambda kv: -abs(kv[1]))
    big = [(n, v) for n, v in items if abs(v) >= 0.3]
    if len(big) == 2:
        (n1, v1), (n2, v2) = big
        rel = "sum" if v1 * v2 < 0 else "difference"
        # anti-correlated loadings (opposite signs) mean the two enter only via their sum.
        return (
            f"{n1} and {n2} are unrecoverable individually — the flat direction moves them "
            f"in {'opposite' if v1 * v2 < 0 else 'the same'} directions, so only their "
            f"{rel} is constrained by the data."
        )
    if big:
        return (
            f"{big[0][0]} is unrecoverable — it barely affects the model output along this "
            "direction."
        )
    return "a multi-parameter combination is unrecoverable (see param_loadings)."


def analyze_model(
    predict_fn: Callable[[Array], Array],
    theta: np.ndarray | Sequence[float] | None = None,
    *,
    sigma: float,
    param_names: Sequence[str] | None = None,
    predict_of_interest: Callable[[Array], Array] | None = None,
    rank_rtol: float = 1e-7,
    sloppy_decade_threshold: float = 3.0,
    pred_rel_tol: float = 0.05,
    naive_cond_threshold: float = 1e6,
) -> SloppinessReport:
    """End-to-end: build the sensitivity matrix + FIM from ``predict_fn`` and classify.

    ``predict_fn(theta) -> outputs`` is the fit model; ``theta`` its parameters (defaults to
    a ``predict_fn.theta0`` attribute if present, as the canonical models set); ``sigma`` the
    observation noise. ``predict_of_interest`` is an optional downstream prediction map whose
    reliability is judged separately (defaults to the fit observables). Returns a
    :class:`SloppinessReport`.
    """
    if theta is None:
        theta = getattr(predict_fn, "theta0", None)
        if theta is None:
            raise ValueError("theta not given and predict_fn has no .theta0")
    theta_arr = np.asarray(theta, dtype=np.float64)
    if param_names is None:
        param_names = getattr(predict_fn, "names", None) or [
            f"theta{i}" for i in range(theta_arr.shape[0])
        ]

    jac_log, y = relative_sensitivity_jacobian(predict_fn, theta_arr)
    if predict_of_interest is not None:
        jac_pred_log, y_pred = relative_sensitivity_jacobian(predict_of_interest, theta_arr)
    else:
        jac_pred_log, y_pred = None, None

    return sloppiness_diagnostic(
        jac_log, y, sigma, param_names,
        jac_pred_log=jac_pred_log, y_pred=y_pred,
        rank_rtol=rank_rtol, sloppy_decade_threshold=sloppy_decade_threshold,
        pred_rel_tol=pred_rel_tol, naive_cond_threshold=naive_cond_threshold,
    )
