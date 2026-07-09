"""Topology model-selection — let the data choose the circuit, don't presume it.

NUDGE is an inference tool, so it must not assume the wiring. This generalizes
the linear-baseline parsimony gate (``classify.switch_detected``, which asks *is there a
switch at all?*) to *which switch?* — scoring candidate circuits (a no-switch null
+ mechanistic switches like ``ras_switch_1node`` / ``ras_switch_2node``) on the same
observable and picking the most parsimonious.

**The complexity penalty is BIC**, ``BIC = k·ln(N) − 2·log L`` (Schwarz 1978): one
with more free parameters ``k`` must *earn* them with a higher likelihood ``L``,
or the simpler model wins. This is what stops NUDGE over-fitting the richer 2-node model
unless the data demands it. ``k`` is counted honestly per model:

- **no-switch** (a single free Gaussian — the "one blob, no bistability" null):
  ``k = D + D(D+1)/2`` (a free mean + covariance over the ``D`` observable dims);
- **a switch circuit** (a 2-mode linear-noise mixture, modes *pinned* to the ODE fixed
  points): ``k = (#free kinetics) + 1 (mixture weight) + 2 (scale, obs)``. The mode
  means/covariances are *derived* from the kinetics, not free — that is the parsimony.

**Honest bound.** Candidates are only BIC-comparable on the *same* observable. A 1-node
(1 species) and a 2-node (2 species) circuit are comparable only when both predict
the same-dimensional data; with a purely 1-D readout (e.g. an IEG activation score) the
2-node's extra species is *unobserved*, so parsimony cannot — and should not — select it
over the 1-node. Distinguishing them needs a genuinely 2-D observable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nudge.core.circuit import Circuit
from nudge.inference.fit import FreeParam
from nudge.inference.lyapunov import calibrate_scale, fit_lyapunov_parameters

__all__ = ["Candidate", "TopologyResult", "select_topology"]


@dataclass(frozen=True)
class Candidate:
    """A candidate topology: a name, a bistable ``Circuit``, and the kinetics to fit."""

    name: str
    circuit: Circuit
    free: list[FreeParam]


@dataclass(frozen=True)
class TopologyResult:
    """The outcome: the winner + per-candidate BIC / log-likelihood / ``k``."""

    selected: str
    is_switch: bool
    bic: dict[str, float]
    loglik: dict[str, float]
    n_params: dict[str, int]
    detail: dict[str, str] = field(default_factory=dict)


def _gaussian_loglik(data: np.ndarray) -> float:
    """Total log-lik of ``data`` under its MLE single Gaussian (the no-switch null)."""
    n, d = data.shape
    mu = data.mean(axis=0)
    cov = np.atleast_2d(np.cov(data, rowvar=False)).reshape(d, d) + 1e-6 * np.eye(d)
    diff = data - mu
    _sign, logdet = np.linalg.slogdet(cov)
    sol = np.linalg.solve(cov, diff.T).T
    quad = np.sum(diff * sol, axis=1)
    return float(np.sum(-0.5 * (quad + logdet + d * np.log(2 * np.pi))))


def select_topology(
    data: np.ndarray,
    candidates: list[Candidate],
    *,
    steps: int = 200,
    seed: int = 0,
    include_no_switch: bool = True,
) -> TopologyResult:
    """Pick the most parsimonious circuit for ``data`` by BIC (lower is better).

    ``data`` is an ``(n_cells, D)`` activity array; every candidate circuit must have
    ``n_species == D`` (same observable). Fits each switch candidate's kinetics via the
    Lyapunov mixture, adds the no-switch single-Gaussian null (unless off), and returns
    the min-BIC choice. ``is_switch`` is ``False`` when the null wins — the fail-safe
    "no switch detected" stance.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be (n_cells, D)")
    n, d = data.shape
    log_n = float(np.log(n))
    bic: dict[str, float] = {}
    loglik: dict[str, float] = {}
    n_params: dict[str, int] = {}
    detail: dict[str, str] = {}

    if include_no_switch:
        ll = _gaussian_loglik(data)
        k = d + d * (d + 1) // 2
        loglik["no-switch"] = ll
        n_params["no-switch"] = k
        bic["no-switch"] = k * log_n - 2 * ll

    for cand in candidates:
        if cand.circuit.n_species != d:
            raise ValueError(
                f"candidate {cand.name!r} has {cand.circuit.n_species} species "
                f"but data is {d}-dimensional"
            )
        try:
            scale = calibrate_scale(data, cand.circuit)
            _rec, _aux, hist = fit_lyapunov_parameters(
                data, cand.circuit, cand.free, k_modes=2, steps=steps, seed=seed,
                scale_init=scale, fit_scale=True, fit_obs=True,
            )
        except Exception as exc:  # e.g. the candidate is not bistable at nominal
            detail[cand.name] = f"unfittable: {exc}"
            continue
        ll = -n * float(np.mean(hist[-20:]))
        if not np.isfinite(ll):  # a candidate that won't fit stably is not a winner
            detail[cand.name] = "non-finite likelihood (unstable fit)"
            continue
        k = len(cand.free) + 1 + 2  # kinetics + mixture weight + (scale, obs)
        loglik[cand.name] = ll
        n_params[cand.name] = k
        bic[cand.name] = k * log_n - 2 * ll

    if not bic:
        raise ValueError("no candidate could be scored")
    selected = min(bic, key=lambda name: bic[name])
    return TopologyResult(
        selected=selected,
        is_switch=selected != "no-switch",
        bic=bic,
        loglik=loglik,
        n_params=n_params,
        detail=detail,
    )
