"""Laplace (curvature) posterior uncertainty for the differentiable circuit fit.

Turns a fit's point estimate ``theta*`` (the log-space kinetics the fit recovers)
into **curvature-based error bars**. At the optimum the loss Hessian
``H = grad^2 L(theta*)`` is the local precision of a Gaussian posterior
``theta ~ N(theta*, H^-1)`` (Laplace's approximation). For a per-cell **mean NLL**
loss -- the deterministic Lyapunov Gaussian-mixture likelihood
(:func:`lyapunov_nll_loss`), *not* the stochastic energy distance whose
minibatch-noisy Hessian is not a likelihood curvature -- this Hessian is the
observed Fisher information, so ``H^-1 / N`` is the asymptotic covariance of the
recovered kinetics. From it:

- **(a) marginal CIs** on ``K`` / ``n`` / ``v_max`` in natural units. The posterior
  is Gaussian in log-space (the fit's parameterization), so each natural-unit
  marginal is lognormal and the CI is ``exp(theta*_i +/- z*sigma_i)`` -- the delta
  method for a log transform, done exactly and always positive.
- **(b) the correlation structure.** The measured **K / v_max (gain / threshold)
  degeneracy** shows up as a near-singular Hessian (a high condition number and a
  strong off-diagonal correlation), reproducing the Fisher result
  (``scripts/vv/fisher_sloppiness.py``; ``FINDINGS.md`` section 2: cond ~ 210 -> 22
  with a second operating point).
- **(c) a mechanism-call confidence** that **abstains** -- marks a knob
  *unidentifiable* and its CI *unbounded* -- when the relevant curvature is flat.

**Fail-safe first (the load-bearing honesty point).** The Laplace Gaussian is a
*local* second-order approximation and is worst exactly at degeneracies and near
bifurcations -- precisely where a large perturbation pushes the system. So this
layer is engineered to **widen and abstain** rather than report a false-precise
interval: a flat / near-null Hessian direction (an eigenvalue ~ 0, a huge condition
number -- the unidentifiable case, e.g. gain / threshold from a single operating
point) sets ``degenerate=True`` and marks every knob loading on that direction as
**unidentifiable / CI unbounded**. The covariance is a **guarded, ridge-regularized
inverse** -- never a plain pseudo-inverse, which would *zero* the flat direction's
variance and *understate* the uncertainty (the opposite of safe) -- so it is finite
and PSD and never NaNs.

Additive / opt-in: this module computes over a loss the caller supplies; it does not
touch the ``fit()`` default output contract or the decoy battery.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from nudge.core.circuit import Circuit
from nudge.inference.fit import FreeParam
from nudge.inference.lyapunov import (
    _apply_free,
    _mode_cov,
    _mode_mean,
    _mvn_logpdf,
)

__all__ = [
    "LaplacePosterior",
    "ParamCI",
    "laplace_posterior",
    "lyapunov_nll_loss",
    "mechanism_confidence",
]

#: z for a two-sided 95% interval (a standard-normal quantile).
_Z95 = 1.959963984540054


@dataclass(frozen=True)
class ParamCI:
    """Natural-unit marginal credible interval for one free kinetic parameter.

    ``point`` is ``exp(theta*)``; ``lo``/``hi`` the ``z``-sigma lognormal interval.
    When the knob is **unidentifiable** (loads on a flat Hessian direction) it is
    reported honestly as *unbounded*: ``identifiable=False``, ``hi=inf``,
    ``log_sd=inf`` -- never a false-precise interval.
    """

    name: str
    point: float
    lo: float
    hi: float
    log_sd: float
    identifiable: bool


@dataclass(frozen=True)
class LaplacePosterior:
    """A local Gaussian posterior ``N(theta*, cov)`` from the curvature at ``theta*``.

    All matrices are **log-space** (the fit's parameterization). ``cond_number`` and
    ``correlation`` expose the identifiability geometry; ``degenerate`` + ``reason``
    carry the fail-safe verdict; ``marginal_ci`` gives per-parameter natural-unit
    intervals.
    """

    theta_opt: np.ndarray
    cov: np.ndarray
    hessian: np.ndarray
    eigenvalues: np.ndarray
    cond_number: float
    correlation: np.ndarray
    marginal_ci: tuple[ParamCI, ...]
    degenerate: bool
    reason: str


def laplace_posterior(
    loss_fn: Callable[[Array], Array],
    theta_opt: Array | np.ndarray | Sequence[float],
    *,
    names: Sequence[str] | None = None,
    n_data: int = 1,
    z: float = _Z95,
    ridge: float = 1e-6,
    cond_max: float = 100.0,
    flat_rel_tol: float = 1e-2,
    load_tol: float = 0.3,
) -> LaplacePosterior:
    """Laplace posterior from ``H = grad^2 loss_fn(theta*)``: CIs + a degeneracy guard.

    ``loss_fn`` maps a **log-space** parameter vector to a scalar loss (a *mean* NLL,
    e.g. :func:`lyapunov_nll_loss`); ``theta_opt`` is the log-space optimum
    ``theta*``. ``n_data`` is the number of cells the mean loss averaged over: the MLE
    covariance of a mean NLL is ``H^-1 / N`` (the observed Fisher of ``N`` cells is
    ``N*H``), so pass it for calibrated CIs (leave ``1`` if ``loss_fn`` is already a
    summed / neg-log-posterior curvature).

    **Guarded inversion.** ``H`` is symmetrized and eigendecomposed; the inverse
    floors each eigenvalue at ``ridge*lambda_max`` (a relative Tikhonov ridge, i.e. a
    weak Gaussian prior) so a flat / negative direction yields a
    **large-but-finite, PSD** variance -- never a NaN and never the
    plain-pseudo-inverse mistake of *zeroing* the flat direction's variance.

    **Fail-safe verdict.** ``degenerate=True`` when the condition number exceeds
    ``cond_max`` or ``H`` is not positive definite. Any knob loading
    (``|eigvec| >= load_tol``) on a **flat direction**
    (eigenvalue ``<= flat_rel_tol*lambda_max``) is marked *unidentifiable* and its CI
    unbounded. Defaults are conservative (abstain sooner): ``cond_max=100`` /
    ``flat_rel_tol=1e-2`` place the single-operating-point toggle degeneracy
    (cond ~ 210) on the *degenerate* side and the two-operating-point fit (cond ~ 22)
    on the *resolved* side, mirroring the measured Fisher result.
    """
    theta = np.asarray(jnp.asarray(theta_opt), dtype=np.float64)
    p = int(theta.shape[0])
    if names is None:
        names = [f"theta[{i}]" for i in range(p)]
    if len(names) != p:
        raise ValueError(f"names has {len(names)} entries, expected {p}")

    hess = np.asarray(jax.hessian(loss_fn)(jnp.asarray(theta_opt)), dtype=np.float64)
    hess = 0.5 * (hess + hess.T)
    evals, evecs = np.linalg.eigh(hess)  # ascending; evecs columns are eigenvectors
    lam_min, lam_max = float(evals[0]), float(evals[-1])

    lam_ref = lam_max if lam_max > 0.0 else 1.0
    cond = np.inf if lam_min <= 0.0 else lam_max / lam_min

    # Guarded (ridge-regularized) inverse: floor the eigenvalues, then rebuild. A flat
    # or negative direction gets a large-but-finite variance (~ 1/floor), never a NaN.
    floor = ridge * lam_ref
    inv_evals = 1.0 / np.maximum(evals, floor)
    cov = (evecs * inv_evals) @ evecs.T / float(n_data)
    cov = 0.5 * (cov + cov.T)

    diag = np.clip(np.diag(cov), 0.0, None)
    sd = np.sqrt(diag)
    dnorm = np.sqrt(np.clip(diag, 1e-300, None))
    correlation = cov / np.outer(dnorm, dnorm)

    # Which parameters load on a flat (unidentifiable) eigen-direction?
    flat = evals <= (flat_rel_tol * lam_ref)
    unident = np.zeros(p, dtype=bool)
    if flat.any():
        loadings = np.abs(evecs[:, flat])  # (p, n_flat)
        unident |= (loadings >= load_tol).any(axis=1)

    degenerate = bool(
        (not np.isfinite(cond)) or (cond > cond_max) or unident.any()
    )

    cis: list[ParamCI] = []
    for i in range(p):
        pt = float(np.exp(theta[i]))
        if (not unident[i]) and np.isfinite(sd[i]):
            cis.append(
                ParamCI(
                    name=names[i],
                    point=pt,
                    lo=float(np.exp(theta[i] - z * sd[i])),
                    hi=float(np.exp(theta[i] + z * sd[i])),
                    log_sd=float(sd[i]),
                    identifiable=True,
                )
            )
        else:  # unidentifiable -> honestly unbounded, never a false-precise interval
            cis.append(
                ParamCI(
                    name=names[i],
                    point=pt,
                    lo=0.0,
                    hi=float("inf"),
                    log_sd=float("inf"),
                    identifiable=False,
                )
            )

    bad = [names[i] for i in range(p) if unident[i]]
    if not np.isfinite(cond):
        reason = "Hessian not positive definite (theta* is not a local min): abstain"
    elif degenerate:
        why = f"condition number {cond:.1f} > {cond_max:g}"
        if bad:
            why += f"; unidentifiable: {', '.join(bad)}"
        reason = f"{why} -> local Laplace Gaussian unreliable, abstain"
    else:
        reason = f"well-conditioned (condition number {cond:.1f})"

    return LaplacePosterior(
        theta_opt=theta,
        cov=cov,
        hessian=hess,
        eigenvalues=evals,
        cond_number=float(cond),
        correlation=correlation,
        marginal_ci=tuple(cis),
        degenerate=degenerate,
        reason=reason,
    )


def mechanism_confidence(
    posterior: LaplacePosterior, free: Sequence[FreeParam]
) -> dict[str, Any]:
    """Per-knob CI + an ``unidentifiable`` flag and a scalar ``confidence``.

    Re-keys ``posterior.marginal_ci`` by each free parameter's kinetic name
    (``K`` / ``n`` / ``vmax``) -- ``free`` must match the ordering ``theta*`` was
    built with. The overall call is ``unidentifiable`` when the posterior is
    degenerate or any knob is flat; ``confidence`` then **abstains to 0.0**.
    Otherwise ``confidence = exp(-max log_sd)`` -- 1.0 for a perfectly pinned knob,
    shrinking toward 0 as the worst marginal widens -- a monotone, bounded summary of
    how sharply the curvature constrains the knobs.
    """
    if len(free) != len(posterior.marginal_ci):
        raise ValueError(
            f"free has {len(free)} entries, posterior has "
            f"{len(posterior.marginal_ci)} parameters"
        )
    knobs: dict[str, dict[str, Any]] = {}
    for f, ci in zip(free, posterior.marginal_ci, strict=True):
        knobs[f[2]] = {
            "point": ci.point,
            "ci": (ci.lo, ci.hi),
            "log_sd": ci.log_sd,
            "identifiable": ci.identifiable,
        }
    unident_knobs = [name for name, v in knobs.items() if not v["identifiable"]]
    unidentifiable = posterior.degenerate or bool(unident_knobs)
    if unidentifiable:
        confidence = 0.0
    else:
        worst = max(float(v["log_sd"]) for v in knobs.values())
        confidence = float(np.exp(-worst))
    return {
        "knobs": knobs,
        "confidence": confidence,
        "unidentifiable": unidentifiable,
        "unidentifiable_knobs": unident_knobs,
        "cond_number": posterior.cond_number,
        "reason": posterior.reason,
    }


def lyapunov_nll_loss(
    data: np.ndarray,
    circuit: Circuit,
    free: list[FreeParam],
    *,
    roots: np.ndarray,
    scale: float,
    obs_sd: float,
    log_weights: np.ndarray | None = None,
) -> Callable[[Array], Array]:
    """A deterministic **mean NLL** of the LNA Gaussian mixture over free kinetics.

    The smooth, likelihood-based Hessian target for :func:`laplace_posterior` (in
    contrast to the stochastic energy distance). Returns ``loss(log_theta)`` -- a
    scalar per-cell mean negative log-likelihood as a function of the **log** kinetic
    values -- with every nuisance **pinned**: the mode seeds ``roots``
    (``(k_modes, n_species)`` stable fixed points, held as ``stop_gradient`` constants
    inside ``_mode_mean``), the depth ``scale``, the observation floor ``obs_sd``, and
    the mixture ``log_weights`` (uniform if ``None``). Pinning the nuisances is what
    makes the Hessian the **observed Fisher information of the kinetics** -- so
    ``H^-1/N`` is their covariance. Mirrors ``fit_lyapunov_parameters``' NLL (means
    ``scale*mu_k``, covariances ``scale^2*Sigma_k + obs_sd^2*I``), reusing the same
    differentiable primitives.
    """
    y = jnp.asarray(np.asarray(data, dtype=np.float32))
    base = circuit.base_params()
    roots_j = jnp.asarray(np.asarray(roots, dtype=np.float32))
    k_modes = int(roots_j.shape[0])
    eye = jnp.eye(circuit.n_species)
    obs_var = float(obs_sd) ** 2
    if log_weights is None:
        log_w = jnp.log(jnp.full((k_modes,), 1.0 / k_modes))
    else:
        log_w = jnp.log(jnp.clip(jnp.asarray(np.asarray(log_weights)), 1e-12))

    def loss(log_theta: Array) -> Array:
        vals = jnp.exp(log_theta)
        params = _apply_free(base, free, vals)
        comps = []
        for k in range(k_modes):
            mu = _mode_mean(circuit, params, roots_j[k])
            cov = _mode_cov(circuit, params, mu)
            mean_obs = scale * mu
            cov_obs = scale**2 * cov + obs_var * eye
            comps.append(log_w[k] + _mvn_logpdf(y, mean_obs, cov_obs))
        ll = jax.scipy.special.logsumexp(jnp.stack(comps), axis=0)
        return -jnp.mean(ll)

    return loss
