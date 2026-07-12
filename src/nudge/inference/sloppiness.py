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
from typing import Any

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
    "sloppiness_diagnostic_matrixfree",
    "analyze_model",
    "analyze_model_matrixfree",
    "fim_matvec",
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


def _classify_verdict(
    *,
    n_null: int,
    is_sloppy: bool,
    predictive: bool,
    span_decades: float,
    cond: float,
    rel_pred_std: float,
    pred_rel_tol: float,
    frac: float,
    rank_rtol: float,
    null_hint: str,
) -> tuple[str, str, str | None]:
    """The three-way verdict tree, shared by the dense and matrix-free diagnostics.

    Pure function of the measured summary statistics — the *identical* logic and wording
    is used whether the FIM spectrum was obtained by a dense ``eigh`` or by a matrix-free
    iterative eigensolver, so the two paths return the same label/reason on the same model.
    """
    fim_greedy_warning: str | None = None
    if n_null > 0:
        label = "unidentifiable"
        reason = (
            f"STRUCTURAL non-identifiability: {n_null} sensitivity-matrix null "
            f"direction(s) (singular value/σ_max < {rank_rtol:g}) — a parameter "
            "combination the data cannot recover no matter how much is collected. "
            + (null_hint or "")
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
    return label, reason, fim_greedy_warning


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
    label, reason, fim_greedy_warning = _classify_verdict(
        n_null=n_null,
        is_sloppy=is_sloppy,
        predictive=predictive,
        span_decades=span_decades,
        cond=cond,
        rel_pred_std=rel_pred_std,
        pred_rel_tol=pred_rel_tol,
        frac=frac,
        rank_rtol=rank_rtol,
        null_hint=(null_dirs[0].hint if null_dirs else ""),
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


# --------------------------------------------------------------------------- #
# the MATRIX-FREE path — the SAME verdict without ever materializing J
# --------------------------------------------------------------------------- #
# The dense path (above) builds ``J = ∂(observables)/∂θ`` with ``jax.jacfwd`` — an
# O(n_obs · n_params) array whose forward-mode intermediates also scale with n_params. For a
# large mechanistic ODE network that array (and jacfwd's tangent fan-out) OOMs (measured:
# dense/jacfwd SIGKILL'd at ~6000 params on a 62 GB box). The matrix-free path below computes
# the SAME Fisher-information (FIM = JᵀJ/σ²) eigenspectrum + verdict using ONLY matrix-vector
# products ``JᵀJ·v`` — one ``jax.jvp`` (a single forward tangent, ``J_log·v``) composed with
# one ``jax.vjp`` (a single reverse cotangent, ``J_logᵀ·w``) — never forming J. Peak memory is
# O(n_params + n_obs + reverse-mode tape), independent of the *product* n_obs·n_params.


def _as_working(theta: np.ndarray | Sequence[float]) -> Array:
    """A device array in the working precision — float64 iff the input is float64 (mirrors
    :func:`relative_sensitivity_jacobian`). NOTE: float64 requires ``jax_enable_x64``; without
    it JAX silently downcasts to float32 and the smallest eigenvalues lose resolution."""
    arr = np.asarray(theta)
    if arr.dtype == np.float64:
        return jnp.asarray(arr, jnp.float64)
    return jnp.asarray(arr, jnp.float32)


def _build_matvecs(
    predict_fn: Callable[[Array], Array], theta_j: Array, sigma: float
) -> tuple[Callable[[Array], Array], Callable[[Array], Array]]:
    """Return ``(fim_mv, jvp_log)`` as jitted JAX closures on **log-parameter** vectors.

    ``fim_mv(v) = (JᵀJ/σ²)·v`` (the FIM matvec); ``jvp_log(v) = J_log·v`` (a forward tangent,
    the prediction-space image of a parameter direction). Both use a single ``jvp`` / ``vjp``,
    so their memory is independent of the parameter count.
    """
    inv_s2 = 1.0 / (float(sigma) ** 2)

    @jax.jit
    def jvp_log(v: Array) -> Array:
        # J_log·v = (∂y/∂θ)·(θ ⊙ v) — one forward tangent.
        return jax.jvp(predict_fn, (theta_j,), (theta_j * v,))[1]

    @jax.jit
    def fim_mv(v: Array) -> Array:
        _y, jv = jax.jvp(predict_fn, (theta_j,), (theta_j * v,))  # J_log·v  (n_obs,)
        _y2, vjp_fn = jax.vjp(predict_fn, theta_j)
        (jt,) = vjp_fn(jv)  # Jᵀ·(J_log·v)  (n_params,)
        return (theta_j * jt) * inv_s2  # J_logᵀ·(J_log·v)/σ²

    return fim_mv, jvp_log


def fim_matvec(
    predict_fn: Callable[[Array], Array], theta: np.ndarray | Sequence[float], sigma: float
) -> Callable[[np.ndarray], np.ndarray]:
    """A matrix-free FIM matvec ``v -> (JᵀJ/σ²)·v`` (log-parameter space), numpy-in/numpy-out.

    Never forms J: one ``jax.jvp`` (``J_log·v``) composed with one ``jax.vjp`` (``J_logᵀ·w``).
    The returned callable is exactly what drives an iterative eigensolver
    (``scipy.sparse.linalg.eigsh`` over a ``LinearOperator``) — see
    :func:`sloppiness_diagnostic_matrixfree`.
    """
    theta_j = _as_working(theta)
    fim_mv, _ = _build_matvecs(predict_fn, theta_j, sigma)
    out_dtype = np.float64 if theta_j.dtype == jnp.float64 else np.float32

    def matvec(v: np.ndarray) -> np.ndarray:
        return np.asarray(fim_mv(jnp.asarray(v, theta_j.dtype)), dtype=out_dtype)

    return matvec


def _dense_spectrum_from_matvec(
    fim_mv: Callable[[Array], Array], n: int, dtype: Any
) -> tuple[np.ndarray, np.ndarray]:
    """FULL spectrum via ``n`` matvecs (reconstruct the n×n FIM column-by-column, then
    ``eigh``). Uses ONLY the matvec — never forms J. For small ``n`` this is exact and makes
    the matrix-free verdict match the dense one bit-for-bit; O(n²) memory bounds it to small n.
    """
    cols = np.asarray(jax.vmap(fim_mv)(jnp.eye(n, dtype=dtype)))  # (n, n): column i = FIM·e_i
    fim = 0.5 * (cols + cols.T)
    evals, evecs = np.linalg.eigh(fim)
    return np.asarray(evals, np.float64), np.asarray(evecs, np.float64)


def _topk_eigsh(
    matvec_np: Callable[[np.ndarray], np.ndarray], n: int, k: int, dtype: Any
) -> tuple[np.ndarray, np.ndarray]:
    """The **top-``k``** FIM eigenpairs (``which='LM'``) via Lanczos over a matrix-free
    ``LinearOperator`` — never forms the FIM. Lanczos is fast and reliable for the LARGEST
    eigenvalues, so this pins ``λ_max`` (and the stiff directions) accurately."""
    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

    op = LinearOperator((n, n), matvec=matvec_np, dtype=dtype)  # type: ignore  # noqa: PGH003
    k = int(np.clip(k, 1, n - 2))
    ncv = int(min(n, max(2 * k + 1, 20)))
    try:
        w, v = eigsh(op, k=k, which="LM", ncv=ncv, maxiter=n * 20, tol=0)
    except ArpackNoConvergence as exc:  # keep whatever converged
        w, v = exc.eigenvalues, exc.eigenvectors
    order = np.argsort(w)
    return np.asarray(w[order], np.float64), np.asarray(v[:, order], np.float64)


def _verified_smallest_eigsh(
    matvec_np: Callable[[np.ndarray], np.ndarray],
    n: int,
    k: int,
    lam_max: float,
    dtype: Any,
    res_tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Best-effort **smallest-``k``** FIM eigenpairs (``which='SA'``), each **verified** by its
    Rayleigh residual ``‖FIM·v − λv‖/λ_max``.

    Lanczos is UNRELIABLE for the smallest eigenvalues of an ill-conditioned FIM (measured: it
    can return a large eigenvalue as "smallest" and mislabel a rank-deficient model
    well-constrained). So every returned pair is checked against the matvec itself; only pairs
    whose residual is below ``res_tol`` are trusted. Returns ``(evals, evecs, converged)`` for
    the trusted pairs; ``converged=False`` means the smallest end could NOT be certified (the
    caller must then fail safe — never assert identifiability it did not verify).

    **The Rayleigh check verifies eigenpair-ness, NOT smallest-ness** (this is the load-bearing
    caveat behind ``NUDGE-LIM-023`` / the P6 finding). ``eigsh(which='SA')`` uses a polynomial
    filter that de-emphasises the smallest end; on an ISOLATED near-null in an otherwise
    well-conditioned spectrum it converges to the well-conditioned cluster and MISSES the null,
    and those returned pairs are *genuine* eigenpairs so they pass this residual check. A
    ``converged=True`` here therefore does **not** license a ``well-constrained`` verdict: the
    caller must ALSO run :func:`_smallest_eig_null_probe` (inverse iteration), which reliably
    catches the missed null, and must abstain when neither route certifies the smallest end.
    """
    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

    op = LinearOperator((n, n), matvec=matvec_np, dtype=dtype)  # type: ignore  # noqa: PGH003
    k = int(np.clip(k, 1, n - 2))
    ncv = int(min(n, max(2 * k + 1, 20)))
    try:
        w, v = eigsh(op, k=k, which="SA", ncv=ncv, maxiter=n * 40, tol=0)
    except ArpackNoConvergence as exc:
        w, v = exc.eigenvalues, exc.eigenvectors
    if w is None or np.size(w) == 0:
        return np.empty(0), np.empty((n, 0)), False
    w = np.atleast_1d(np.asarray(w, np.float64))
    v = np.asarray(v, np.float64).reshape(n, -1)
    scale = max(lam_max, 1e-300)
    keep_w, keep_v = [], []
    for i in range(w.shape[0]):
        vi = v[:, i]
        nv = float(np.linalg.norm(vi))
        if nv == 0.0:
            continue
        vi = vi / nv
        resid = float(np.linalg.norm(matvec_np(vi) - w[i] * vi)) / scale
        if resid < res_tol:
            keep_w.append(max(w[i], 0.0))
            keep_v.append(vi)
    if not keep_w:
        return np.empty(0), np.empty((n, 0)), False
    order = np.argsort(keep_w)
    kw = np.asarray(keep_w, np.float64)[order]
    kv = np.stack(keep_v, axis=1)[:, order]
    return kw, kv, True


def _smallest_eig_null_probe(
    matvec_np: Callable[[np.ndarray], np.ndarray],
    n: int,
    lam_max: float,
    dtype: Any,
    floor_ratio: float,
    *,
    eps_rel: float = 1e-3,
    n_iter: int = 40,
    cg_rtol: float = 1e-8,
    res_tol: float = 1e-3,
    seed: int = 0,
) -> tuple[np.ndarray, float, bool]:
    """Detect an ISOLATED (near-)null FIM eigenpair by **inverse iteration** (shift-invert via
    CG) — the reliable complement to ``eigsh(which='SA')``, and the fix for the P6 /
    ``NUDGE-LIM-023`` hole.

    ``eigsh(which='SA')`` uses a polynomial filter that de-emphasises the smallest end and can
    MISS an isolated near-zero eigenvalue, converging to the well-conditioned cluster and
    mislabelling a rank-deficient model ``well-constrained``. Inverse iteration on
    ``(FIM + eps·I)`` instead AMPLIFIES the FIM's smallest eigenvalue — the null is the strictly
    dominant eigenpair of ``(FIM + eps·I)⁻¹`` (eigenvalue ``1/eps`` vs the next ``1/(λ₂+eps)``),
    and power/inverse iteration reliably converges to a well-separated dominant eigenpair, so an
    isolated null is caught. Matrix-free: each step is a CG solve driven by the FIM matvec, no
    factorisation — cost ``O(n_params + n_obs)`` per matvec (MEASURED: ~150–800 matvecs, <0.3 s
    to catch the P6 null that ``eigsh('SA')`` misses).

    Returns ``(vector, rayleigh_quotient, found_null)``. ``found_null`` is True **only** when the
    converged direction has Rayleigh quotient ``vᵀ·FIM·v ≤ floor_ratio·λ_max`` (the rank floor)
    AND a small Rayleigh residual — a genuine, residual-verified near-null.

    **One-sided certificate (the honest bound).** A low Rayleigh quotient PROVES a small
    eigenvalue exists (⇒ a null / unidentifiable — safe). A large quotient does NOT prove
    identifiability: on a dense *sloppy* small-end, inverse iteration overestimates ``λ_min``
    (MEASURED: RQ ≈ 5 for a true ``λ_min ≈ 1.7e-6``), so its value is never trusted as an
    accurate ``λ_min`` for a ``well-constrained`` / ``sloppy-but-predictive`` verdict. When no
    null is found the caller must ABSTAIN, not assert identifiability.
    """
    from scipy.sparse.linalg import LinearOperator, cg

    scale = max(float(lam_max), 1e-300)
    eps = eps_rel * scale

    def _shifted(x: np.ndarray) -> np.ndarray:  # (FIM + eps·I)·x, matrix-free
        return matvec_np(x) + eps * x

    shift = LinearOperator((n, n), matvec=_shifted, dtype=dtype)  # type: ignore  # noqa: PGH003
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n).astype(np.float64)
    v /= np.linalg.norm(v)
    maxiter = 10 * n
    floor = floor_ratio * scale
    rq = float("inf")  # Rayleigh quotient vᵀ·FIM·v — a MONOTONE-decreasing smallest-eig estimate
    resid = 1.0
    for _ in range(n_iter):
        x, _info = cg(shift, v, rtol=cg_rtol, maxiter=maxiter)
        nx = float(np.linalg.norm(x))
        if nx == 0.0:
            break
        v = x / nx
        av = matvec_np(v)
        rq_prev = rq
        rq = max(float(v @ av), 0.0)
        resid = float(np.linalg.norm(av - rq * v)) / scale
        # Stop only once the Rayleigh QUOTIENT (not just the eigen-residual) has settled: for a
        # true null the quotient keeps shrinking geometrically PAST the point the residual is
        # small (contamination b²·λ_cluster is still ≫ floor), so an early residual-only break
        # would return a spuriously-large quotient and MISS the null. Break when the residual is
        # small AND either the quotient reached the rank floor (a verified near-null) or it has
        # plateaued (converged to a finite, non-null smallest eigenvalue — for a true null the
        # per-step drop is many orders of magnitude, never a mere plateau).
        if resid < res_tol and (rq <= floor or rq > 0.5 * rq_prev):
            break
    found_null = (resid < res_tol) and (rq <= floor)
    return v.astype(np.float64), rq, found_null


def sloppiness_diagnostic_matrixfree(
    predict_fn: Callable[[Array], Array],
    theta: np.ndarray | Sequence[float],
    sigma: float,
    param_names: Sequence[str] | None = None,
    *,
    predict_of_interest: Callable[[Array], Array] | None = None,
    n_eigs: int = 6,
    rank_rtol: float = 1e-7,
    sloppy_decade_threshold: float = 3.0,
    pred_rel_tol: float = 0.05,
    naive_cond_threshold: float = 1e6,
    ridge: float = 1e-10,
    method: str = "auto",
    dense_below: int = 2048,
) -> SloppinessReport:
    """Matrix-free twin of :func:`sloppiness_diagnostic`: the SAME :class:`SloppinessReport`
    (label / null direction / spectrum / verdict) from a differentiable ``predict_fn`` —
    **without ever materializing the sensitivity matrix J** (so it does not OOM the way the
    dense ``jacfwd`` path does on a large ODE network). The FIM (``JᵀJ/σ²``) is touched only
    through matvecs (:func:`fim_matvec`: one ``jvp`` + one ``vjp``).

    Two matrix-free routes, chosen by ``method``:

    - ``"dense"`` (and ``"auto"`` when ``n_params ≤ dense_below``) — reconstruct the exact
      ``n_params × n_params`` FIM from ``n_params`` matvecs and ``eigh`` it. **Exact** full
      spectrum, so it matches :func:`sloppiness_diagnostic` bit-for-bit; O(n_params²) memory,
      but it *still* avoids the dense-J OOM (the matvec never forms J or the ``jacfwd`` tangent
      fan-out). The recommended path whenever ``n_params`` is not enormous. ``dense_below``
      defaults to **2048** — MEASURED affordable (``scripts/vv/sloppiness_scaling.py`` /
      FINDINGS §P6: exact reconstruct+``eigh`` ≈ 18 s / 0.7 GB peak at n=2048, recovering the
      exact null every time), well past the old 256, so the whole realistic regime gets the
      exact verdict. Raise it if you have the memory/time (n=4096 ≈ 42 s / 2 GB).
    - ``"iterative"`` (and ``"auto"`` when ``n_params > dense_below``) — an iterative
      eigensolver (Lanczos ``eigsh``) over the matvec ``LinearOperator``, O(n_params + n_obs)
      memory. Lanczos is reliable for the LARGEST eigenvalues (``λ_max`` + stiff directions).
      The **smallest** eigenvalue of an ill-conditioned FIM is NOT reliably reachable by
      ``eigsh(which='SA')`` — its polynomial filter de-emphasises the smallest end and can MISS
      an isolated near-null, returning genuine-but-not-smallest pairs that pass a Rayleigh check,
      which would mislabel a rank-deficient model well-constrained (``NUDGE-LIM-023`` / the P6
      finding). So the smallest end is handled fail-safe along three lines, and the iterative
      path **never** emits a ``well-constrained`` / ``sloppy-but-predictive`` verdict it cannot
      certify:

        * ``n_params > n_obs`` ⇒ the FIM is rank-deficient **by shape** (rank ≤ n_obs) ⇒
          ``unidentifiable``, certified without any smallest-eigenvalue solve;
        * otherwise the true smallest eigenvalue is probed by **inverse iteration** (shift-invert
          via CG, :func:`_smallest_eig_null_probe`), which AMPLIFIES the smallest eigenvalue and
          reliably CATCHES an isolated near-null that ``eigsh('SA')`` misses (MEASURED, FINDINGS
          §P6). A residual-verified near-null ⇒ ``unidentifiable`` (naming the null direction);
        * if no null is found, the full smallest spectrum is still not certifiable matrix-free
          (the probe is one-sided — a low Rayleigh quotient PROVES a null, but a large one does
          NOT prove identifiability), so the diagnostic **abstains** (``unidentifiable`` with an
          explicit "cannot certify the smallest eigenvalue" reason) rather than assert
          identifiability it cannot verify.

    **Honest bounds of the iterative path.** ``λ_max`` / the stiff spectrum are accurate; a
    structural null is caught (via shape or the inverse-iteration probe), but the iterative path
    **cannot** return a positive ``well-constrained`` / ``sloppy-but-predictive`` verdict — that
    requires the exact smallest spectrum, so it abstains instead. The prediction-variance
    propagation sums over the computed eigendirections only. ``fim_eigenvalues`` holds the
    computed subset, not the full spectrum, in the iterative case. For a definitive
    positive verdict on a moderate ``n_params`` use ``method="dense"`` (exact) or keep
    ``n_params ≤ dense_below`` where ``"auto"`` already routes to the exact path.
    """
    theta_arr = np.asarray(theta, dtype=np.float64)
    n_theta = int(theta_arr.shape[0])
    names = (
        tuple(param_names) if param_names is not None else tuple(f"θ{i}" for i in range(n_theta))
    )
    theta_j = _as_working(theta_arr)
    out_dtype = np.float64 if theta_j.dtype == jnp.float64 else np.float32

    fim_mv, jvp_log = _build_matvecs(predict_fn, theta_j, sigma)
    y = np.asarray(predict_fn(theta_j), dtype=np.float64)
    n_obs = int(y.size)

    poi = predict_fn if predict_of_interest is None else predict_of_interest
    if predict_of_interest is None:
        jvp_pred, y_pred = jvp_log, y
    else:
        _, jvp_pred = _build_matvecs(poi, theta_j, sigma)
        y_pred = np.asarray(poi(theta_j), dtype=np.float64)

    # --- the FIM spectrum, matrix-free ------------------------------------------------ #
    shape_null = max(n_theta - n_obs, 0)  # n_params > n_obs ⇒ rank ≤ n_obs ⇒ exact nulls
    use_iterative = method == "iterative" or (method == "auto" and n_theta > dense_below)
    if use_iterative and n_theta <= 2 * n_eigs + 2:
        use_iterative = False  # too few params for a meaningful extremal split → exact
    if method == "dense":
        use_iterative = False

    smallest_certified = True
    if use_iterative:
        # RELIABLE end: top-k via Lanczos 'LM' (anchors λ_max + the stiff directions).
        matvec_np = fim_matvec(predict_fn, theta_arr, sigma)
        w_top, v_top = _topk_eigsh(matvec_np, n_theta, n_eigs, out_dtype)
        lam_max = float(w_top[-1]) if w_top.size else 0.0
        if shape_null > 0:
            # n_params > n_obs: the FIM is rank-deficient by SHAPE — unidentifiable, exactly,
            # WITHOUT the (slow, unreliable) smallest-eigenvalue solve. λ_min ≡ 0.
            evals = np.concatenate([[0.0], w_top])
            evecs = np.concatenate([np.zeros((n_theta, 1)), v_top], axis=1)
        else:
            # HARD end: the smallest eigenvalue. eigsh(which='SA') CANNOT certify smallest-ness
            # — on an ISOLATED near-null it converges to the well-conditioned cluster and MISSES
            # the null, and the pairs it returns pass the Rayleigh check (they are genuine
            # eigenpairs, just not the smallest), so trusting it mislabels a rank-deficient model
            # well-constrained (the P6 / NUDGE-LIM-023 hole). Instead probe the true smallest by
            # INVERSE ITERATION (shift-invert via CG), which amplifies the FIM's smallest
            # eigenvalue and reliably catches an isolated null 'SA' misses (MEASURED).
            null_vec, null_lam, null_found = _smallest_eig_null_probe(
                matvec_np, n_theta, lam_max, out_dtype, floor_ratio=rank_rtol**2
            )
            if null_found:
                # a residual-verified near-null (structural non-identifiability) — inject it with
                # its concrete direction so the verdict is unidentifiable + names the null combo.
                evals = np.concatenate([[max(null_lam, 0.0)], w_top])
                evecs = np.concatenate([null_vec[:, None], v_top], axis=1)
            else:
                # No isolated null found. The FULL smallest spectrum is NOT certifiable
                # matrix-free (eigsh('SA') is unreliable there; the probe is one-sided), so we
                # cannot distinguish well-constrained / sloppy-but-predictive — gather the
                # best-effort verified small pairs for the report and ABSTAIN (never assert
                # identifiability we did not verify). smallest_certified drives the abstention.
                w_sa, v_sa, _sa_ok = _verified_smallest_eigsh(
                    matvec_np, n_theta, n_eigs, lam_max, out_dtype
                )
                smallest_certified = False
                evals = np.concatenate([w_sa, w_top]) if w_sa.size else w_top
                evecs = np.concatenate([v_sa, v_top], axis=1) if w_sa.size else v_top
        full_spectrum = False
    else:
        evals, evecs = _dense_spectrum_from_matvec(fim_mv, n_theta, theta_j.dtype)
        full_spectrum = True

    evals = np.clip(evals, 0.0, None)
    order = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]
    lam_min = float(evals[0])
    lam_max = float(evals[-1])
    cond = np.inf if lam_min <= 0.0 else lam_max / lam_min
    span_decades = float(np.log10(lam_max / lam_min)) if lam_min > 0.0 else np.inf

    # --- structural nulls ------------------------------------------------------------- #
    # FIM eigenvalue λ = (singular value)²/σ², so the dense ``sv/sv_max < rank_rtol`` test is
    # ``λ/λ_max < rank_rtol²``. Plus the shape deficiency (n_params > n_obs).
    lam_floor_ratio = rank_rtol**2
    computed_null = int(np.sum(evals < lam_floor_ratio * lam_max)) if lam_max > 0.0 else evals.size
    n_null = shape_null if shape_null > 0 else computed_null

    # null directions: only from VERIFIED near-null eigenvectors (residual-checked in the
    # iterative path; exact in the dense path). A shape-null (n_params > n_obs) is certain but
    # we don't fabricate a specific vector for its huge null space — the verdict + count carry it.
    null_idx = list(np.where(evals < lam_floor_ratio * lam_max)[0]) if lam_max > 0.0 else []
    sigma_f = float(sigma)
    null_dirs: list[NullDirection] = []
    for j in null_idx[:n_eigs]:
        v = np.asarray(evecs[:, j], dtype=np.float64)
        if float(np.linalg.norm(v)) == 0.0:  # the shape-null placeholder slot
            continue
        dom = int(np.argmax(np.abs(v)))
        if v[dom] < 0.0:
            v = -v
        loadings = {names[i]: float(v[i]) for i in range(n_theta)}
        null_dirs.append(
            NullDirection(
                vector=v,
                param_loadings=loadings,
                prediction_sensitivity=float(sigma_f * np.sqrt(max(float(evals[j]), 0.0))),
                hint=_null_hint(loadings),
            )
        )

    # --- prediction uncertainty (matrix-free covariance propagation) ------------------ #
    # pred_var[o] = Σ_i (1/max(λ_i, floor)) · (J_pred·v_i)²[o]; dominated by small λ_i.
    floor = ridge * lam_max if lam_max > 0.0 else ridge
    pred_var = np.zeros(y_pred.size, dtype=np.float64)
    contrib = np.zeros(evals.size, dtype=np.float64)  # per-direction total pred variance
    for i in range(evals.size):
        vi = jnp.asarray(evecs[:, i], theta_j.dtype)
        jvi = np.asarray(jvp_pred(vi), dtype=np.float64)  # J_pred·v_i in prediction space
        inv_i = 1.0 / max(float(evals[i]), floor)
        pred_var += inv_i * jvi**2
        contrib[i] = inv_i * float(np.sum(jvi**2))
    max_pred_std = float(np.sqrt(pred_var.max())) if pred_var.size else 0.0
    pred_rms = float(np.sqrt(np.mean(y_pred**2))) + 1e-12
    rel_pred_std = max_pred_std / pred_rms
    predictive = rel_pred_std < pred_rel_tol

    total_contrib = float(contrib.sum())
    sloppy_mask = evals < (lam_max * 10.0 ** (-sloppy_decade_threshold))
    frac = float(contrib[sloppy_mask].sum() / total_contrib) if total_contrib > 0.0 else 0.0

    is_sloppy = span_decades > sloppy_decade_threshold
    n_sloppy = int(np.sum(sloppy_mask)) + (0 if full_spectrum else shape_null)

    # --- the verdict (shared with the dense path) ------------------------------------- #
    if not smallest_certified:
        # FAIL-SAFE: the smallest end could NOT be certified matrix-free. eigsh(which='SA') is
        # unreliable at the smallest end of an ill-conditioned FIM (it can miss an isolated null),
        # and the inverse-iteration null probe found NO verified near-null — but a one-sided probe
        # that fails to find a null does NOT prove identifiability, and 'SA' alone cannot pin the
        # true λ_min. So we CANNOT distinguish well-constrained / sloppy-but-predictive here and
        # must NOT assert identifiability we could not verify — abstain instead.
        label = "unidentifiable"
        reason = (
            "MATRIX-FREE ABSTENTION: the smallest FIM eigenvalue could NOT be certified at "
            f"n_params={n_theta} (eigsh(which='SA') is unreliable at the smallest end of an "
            "ill-conditioned matrix, and the inverse-iteration null probe found no verified "
            "near-null — a one-sided check that cannot, by itself, prove identifiability). NUDGE "
            "will not claim identifiability it has not verified — rerun with method='dense' for "
            "the exact spectrum (feasible per-matvec; O(n_params²) memory), raise dense_below, or "
            "add observations."
        )
        fim_greedy_warning = None
    else:
        label, reason, fim_greedy_warning = _classify_verdict(
            n_null=n_null,
            is_sloppy=is_sloppy,
            predictive=predictive,
            span_decades=span_decades,
            cond=cond,
            rel_pred_std=rel_pred_std,
            pred_rel_tol=pred_rel_tol,
            frac=frac,
            rank_rtol=rank_rtol,
            null_hint=(null_dirs[0].hint if null_dirs else ""),
        )

    naive_verdict = (
        "unidentifiable"
        if (not np.isfinite(cond)) or cond > naive_cond_threshold
        else "identifiable"
    )
    naive_is_wrong = naive_verdict == "unidentifiable" and label in (
        "sloppy-but-predictive",
        "well-constrained",
    )

    return SloppinessReport(
        label=label,
        param_names=names,
        fim_eigenvalues=evals,
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


def analyze_model_matrixfree(
    predict_fn: Callable[[Array], Array],
    theta: np.ndarray | Sequence[float] | None = None,
    *,
    sigma: float,
    param_names: Sequence[str] | None = None,
    predict_of_interest: Callable[[Array], Array] | None = None,
    n_eigs: int = 6,
    rank_rtol: float = 1e-7,
    sloppy_decade_threshold: float = 3.0,
    pred_rel_tol: float = 0.05,
    naive_cond_threshold: float = 1e6,
    method: str = "auto",
    dense_below: int = 2048,
) -> SloppinessReport:
    """End-to-end matrix-free classify (twin of :func:`analyze_model`): read ``theta`` /
    ``param_names`` from the model (``predict_fn.theta0`` / ``.names`` when present) and run
    the matrix-free diagnostic. Scales to large ODE networks that OOM the dense jacfwd path.
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
    return sloppiness_diagnostic_matrixfree(
        predict_fn, theta_arr, sigma, param_names,
        predict_of_interest=predict_of_interest, n_eigs=n_eigs,
        rank_rtol=rank_rtol, sloppy_decade_threshold=sloppy_decade_threshold,
        pred_rel_tol=pred_rel_tol, naive_cond_threshold=naive_cond_threshold,
        method=method, dense_below=dense_below,
    )
