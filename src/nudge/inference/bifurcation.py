"""Bifurcation / tipping-point proximity — the **robustness dial**.

NUDGE's other capabilities ask *which knob* a perturbation moves. This one asks a
different, high-value question: **how close is a bistable switch to *losing*
bistability** — a saddle-node / fold? The answer is a scalar **robustness dial**: a
hair-trigger cliff vs a well-buffered switch. Audience: resilience / critical-transition
biology (aging, disease progression, cell-fate commitment) and engineered-circuit
robustness QA. It is also the hard dependency for the future ``design()`` safety gate:
an intervention that pushes a switch toward a tipping point must be flagged, which needs
a real, validated proximity score.

The proximity signal is *already computed but buried* inside the circuit engine — no
public accessor returns it as a number. This module lifts it out **properly**, with
**three complementary channels** (a proper build uses all three, not just one), each of
which has a known analytic limit at the fold:

1. **Primary — critical slowing.** ``min|Re λ|`` of the drift Jacobian at each stable
   fixed point → 0 at the fold (the slowest relaxation mode stalls — critical slowing
   down; Scheffer 2009). Recomputed from :meth:`Circuit.vector_field` at each stable
   state.
2. **Secondary — basin collapse.** The state-space distance from a stable node to the
   index-1 saddle → 0 at the fold (the basin flattens). From
   :meth:`Circuit.fixed_points` + :meth:`Circuit.transition_state`.
3. **Tertiary — LNA lobe swell.** ``√λ_max(Σ) / min‖μᵢ−μⱼ‖`` → 1 at the fold (the noise
   lobes merge). This is exactly the internal in
   :func:`nudge.inference.lyapunov.lna_reliable`; it reuses
   :meth:`Circuit.mode_covariances`.

These fuse into a normalized **0..1 proximity dial** keeping all three raw channels.

**Honesty (the crux — the capability lives or dies here).** The linear-noise (LNA)
Gaussian **breaks down *precisely at the fold***: a mode's variance diverges as its
Jacobian eigenvalue → 0, so the Gaussian lobe is *least* trustworthy exactly where it
matters most. Therefore the dial is reported as a **one-sided LOWER BOUND** near the
fold — "the switch is *at least* this close" — never a point estimate
(:attr:`BifurcationScore.one_sided`). And :func:`classify_robustness` **abstains**
(``unresolved``) on the deep-basin far side, where every channel has floored and the
Gaussian lobe carries *no* fold information, rather than emit a precise "far" number it
cannot support. See ``NUDGE-LIM-012``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import enable_x64

from nudge.core.circuit import Circuit, Params
from nudge.inference.fit import FreeParam

__all__ = [
    "BifurcationResult",
    "BifurcationScore",
    "attribute_bifurcation",
    "bifurcation_proximity",
    "classify_robustness",
]

#: Proximity at/above which the switch is called **near-fold** (one-sided lower bound).
NEAR_FOLD = 0.55
#: Proximity below which every channel has floored — abstain (``unresolved``) rather
#: than emit a precise "far" number the saturated signal cannot support.
DEEP_BASIN = 0.05
#: Lobe ratio at/above which the LNA Gaussian lobes overlap — the noise model is
#: breaking down, so the score becomes a one-sided lower bound.
LOBE_OVERLAP = 1.0


@dataclass(frozen=True)
class BifurcationScore:
    """A bistable switch's proximity to a saddle-node fold, as three channels + a dial.

    All fields are honest measurements of one circuit's steady-state geometry:

    - ``min_re_lambda`` — the smallest ``|Re λ|`` over the stable modes' drift Jacobians
      (→ 0 at the fold; critical slowing down). The **primary** channel.
    - ``node_saddle_distance`` — the smallest stable-node → index-1-saddle distance
      (→ 0 at the fold; basin collapse). ``nan`` if no saddle was located.
    - ``lna_lobe_ratio`` — ``√λ_max(Σ) / min‖μᵢ−μⱼ‖`` (→ 1 at the fold; lobes merge).
    - ``proximity`` — the fused **0..1 dial** (``max`` of the per-channel proximities: a
      fail-safe "report the most alarming channel" fusion).
    - ``one_sided`` — the noise lobes overlap (``lna_lobe_ratio ≥``
      :data:`LOBE_OVERLAP`), so the LNA Gaussian is breaking down and the score is a
      **lower bound**, not a point estimate.
    - ``n_stable_modes`` — the count of stable fixed points (≥ 2 for a bistable switch).
    - ``channels`` — the **raw per-mode values** the three channels are computed from
      (JSON-serialisable), so a demo / notebook can plot them without recomputation.
    """

    min_re_lambda: float
    node_saddle_distance: float
    lna_lobe_ratio: float
    proximity: float
    one_sided: bool
    n_stable_modes: int
    channels: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BifurcationResult:
    """A data-driven proximity attribution: score, call, and the depth context."""

    score: BifurcationScore | None
    call: str
    reason: str
    scale: float  # the data-calibrated depth (only the LNA lobe channel depends on it)
    lna_reason: str  # the LNA-reliability verdict at this depth (a lobe-channel caveat)
    recovered: Mapping[str, float]  # fitted kinetics (empty if ``free`` was not given)


def _f64_params(circuit: Circuit) -> Params:
    """The circuit's base params cast to float64 (for a well-conditioned Jacobian)."""
    base = circuit.base_params()
    return {
        "species": {k: jnp.asarray(v, jnp.float64) for k, v in base["species"].items()},
        "edges": {k: jnp.asarray(v, jnp.float64) for k, v in base["edges"].items()},
    }


def _stable_eig_reals(
    circuit: Circuit, states: Sequence[np.ndarray]
) -> list[np.ndarray]:
    """Real parts of the drift-Jacobian eigenvalues at each ``state`` (x64 context).

    ``A = ∂/∂x [production(x) − decay·x]`` by autodiff — the drift the LNA covariance
    linearizes. Under a **local x64 context**, because the Jacobian is ill-conditioned
    near a saddle-node (an eigenvalue → 0). Reuses :meth:`Circuit.production` (any
    topology); never differentiates the fit.
    """
    with enable_x64():
        base = _f64_params(circuit)
        decay = base["species"]["decay"]

        def drift(x: jax.Array) -> jax.Array:
            return circuit.production(jnp.maximum(x, 0.0), base) - decay * x

        jac = jax.jacfwd(drift)
        out: list[np.ndarray] = []
        for s in states:
            m = jnp.asarray(np.asarray(s, dtype=np.float64))
            re = jnp.real(jnp.linalg.eigvals(jac(m)))
            out.append(np.asarray(re, dtype=np.float64))
    return out


def bifurcation_proximity(circuit: Circuit) -> BifurcationScore | None:
    """Score how close ``circuit``'s bistable switch is to a saddle-node fold.

    Returns a :class:`BifurcationScore` with three channels + the fused 0..1 dial, or
    ``None`` when the circuit has **fewer than two stable modes** (no switch to be near
    a fold — the caller treats it as ``not-bistable``). Never raises (``None`` on
    any numerical failure), so it is safe inside a ``design()`` safety gate.

    The three channels, each with a known limit at the fold:

    - **critical slowing** ``p_slow = 1 − min|Re λ| / min(decay)`` — the slowest
      relaxation rate collapses toward 0 as the fold nears (Scheffer 2009);
    - **basin collapse** ``p_basin = 1 − node→saddle / (½·node-sep)`` — the saddle
      merges into a node;
    - **lobe overlap** ``p_lobe = clip(lna_lobe_ratio − 1, 0, 1)`` — the LNA noise lobes
      overlap (contributes only once they *genuinely* overlap; below overlap the LNA
      lobe carries no fold information — see the deep-basin abstention).

    The dial is ``proximity = max(½·(p_slow + p_basin), p_lobe)`` — the two
    deterministic, depth-independent channels averaged, then ``max``'d with the LNA
    overlap so the noise channel can only *raise* the alarm, never lower it (fail-safe).
    """
    modes = circuit.mode_covariances()
    if modes is None or len(modes) < 2:
        return None
    means = [np.asarray(m, dtype=float) for m, _ in modes]
    covs = [np.asarray(c, dtype=float) for _, c in modes]

    # channel 1 — critical slowing: min |Re λ| over the stable modes.
    try:
        eig_reals = _stable_eig_reals(circuit, means)
    except Exception:
        return None
    per_mode_min = [float(np.min(np.abs(re))) for re in eig_reals]
    min_re = float(min(per_mode_min))
    dref = max(
        float(np.min(np.asarray(circuit.base_params()["species"]["decay"]))), 1e-9
    )
    p_slow = float(np.clip(1.0 - min_re / dref, 0.0, 1.0))

    # channel 2 — basin collapse: nearest stable-node → index-1-saddle distance.
    saddle = circuit.transition_state()
    if saddle is not None:
        saddle_np = np.asarray(saddle, dtype=float)
        nsd = float(min(float(np.linalg.norm(m - saddle_np)) for m in means))
    else:
        saddle_np = None
        nsd = float("nan")

    # inter-mode separation (the denominator shared by channels 2 and 3).
    sep = float(
        min(
            float(np.linalg.norm(means[i] - means[j]))
            for i in range(len(means))
            for j in range(i + 1, len(means))
        )
    )
    if saddle_np is not None and sep > 0.0 and np.isfinite(nsd):
        p_basin = float(np.clip(1.0 - nsd / (0.5 * sep), 0.0, 1.0))
    else:
        p_basin = 0.0

    # channel 3 — LNA lobe swell (the exact lna_reliable internal, reused).
    lobe_std = [
        float(np.sqrt(max(float(np.max(np.linalg.eigvalsh(c))), 0.0))) for c in covs
    ]
    lobe_ratio = float(max(lobe_std) / sep) if sep > 0.0 else float("inf")
    p_lobe = float(np.clip(lobe_ratio - 1.0, 0.0, 1.0))

    det = 0.5 * (p_slow + p_basin)
    proximity = float(max(det, p_lobe))
    one_sided = bool(np.isfinite(lobe_ratio) and lobe_ratio >= LOBE_OVERLAP)

    channels: dict[str, Any] = {
        "per_mode_min_abs_re_lambda": [float(v) for v in per_mode_min],
        "per_mode_eig_real": [[float(x) for x in re] for re in eig_reals],
        "per_mode_lobe_std": [float(v) for v in lobe_std],
        "stable_nodes": [[float(x) for x in m] for m in means],
        "saddle": None if saddle_np is None else [float(x) for x in saddle_np],
        "separation": sep,
        "decay_ref": dref,
        "channel_proximities": {
            "critical_slowing": p_slow,
            "basin_collapse": p_basin,
            "lobe_overlap": p_lobe,
        },
    }
    return BifurcationScore(
        min_re_lambda=min_re,
        node_saddle_distance=nsd,
        lna_lobe_ratio=lobe_ratio,
        proximity=proximity,
        one_sided=one_sided,
        n_stable_modes=len(means),
        channels=channels,
    )


def classify_robustness(
    score: BifurcationScore | None,
    *,
    near: float = NEAR_FOLD,
    deep: float = DEEP_BASIN,
) -> tuple[str, str]:
    """Turn a score into an honest verdict — ``(call, reason)``.

    Four outcomes, fail-safe first:

    - **not-bistable** — ``score is None`` (< 2 stable modes): there is no switch to be
      near a fold.
    - **near-fold** — ``proximity ≥ near``: the switch is close to losing bistability.
      Because the LNA Gaussian is breaking down here (``one_sided`` is set once the
      lobes overlap), the number is a **one-sided LOWER BOUND** — "at least this close"
      — never a point estimate (``NUDGE-LIM-012``).
    - **unresolved** — ``proximity < deep``: the deep-basin far side, where the slowest
      relaxation rate, the basin depth *and* the noise-lobe overlap have all floored.
      The Gaussian lobe carries **no** fold information at this depth, so NUDGE
      **abstains** rather than emit a precise "far" number it cannot support (do not
      manufacture false precision about a distance beyond the dial's resolution).
    - **robust** — ``deep ≤ proximity < near``: a well-buffered switch, comfortably away
      from the fold, with the proximity signal still responsive (a trustworthy call).
    """
    if score is None:
        return "not-bistable", (
            "fewer than two stable modes — the circuit is monostable (or the topology "
            "is unsupported), so there is no bistable switch that could be near a fold"
        )
    p = score.proximity
    ch = ", ".join(
        f"{k}={v:.3f}"
        for k, v in score.channels.get("channel_proximities", {}).items()
    )
    if p >= near:
        return "near-fold", (
            f"proximity={p:.3f} ≥ {near:g} — the switch is close to a saddle-node fold "
            f"(min|Reλ|={score.min_re_lambda:.3f}→0, node→saddle="
            f"{score.node_saddle_distance:.3f}→0, lobe ratio="
            f"{score.lna_lobe_ratio:.3f}→1; {ch}). This is a ONE-SIDED LOWER BOUND — "
            "'at least this close' — because the linear-noise Gaussian breaks down "
            "precisely at the fold (a mode's variance diverges), so the estimate is "
            "least reliable exactly here (NUDGE-LIM-012)"
        )
    if p < deep:
        return "unresolved", (
            f"proximity={p:.3f} < {deep:g} — deep in the basin, every channel has "
            f"floored (min|Reλ|={score.min_re_lambda:.3f}≈decay, lobe ratio="
            f"{score.lna_lobe_ratio:.3f}≪1): the dial has hit the end of its dynamic "
            "range and carries no fold info at this depth. NUDGE ABSTAINS rather "
            "than emit a precise 'far' number it cannot support (the switch is clearly "
            "not near a fold, but the distance is beyond the dial's resolution)"
        )
    bound = (
        " (one-sided lower bound — the noise lobes overlap)" if score.one_sided else ""
    )
    return "robust", (
        f"proximity={p:.3f} ∈ [{deep:g}, {near:g}) — a well-buffered switch, "
        f"comfortably away from the fold{bound} "
        f"(min|Reλ|={score.min_re_lambda:.3f}, node→saddle="
        f"{score.node_saddle_distance:.3f}, lobe={score.lna_lobe_ratio:.3f}; {ch})"
    )


def _apply_kinetics(
    circuit: Circuit, free: list[FreeParam], vals: np.ndarray
) -> Circuit:
    """Rebuild ``circuit`` with the ``free`` kinetic params overridden by ``vals``."""
    species = list(circuit.species)
    edges = list(circuit.edges)
    for (scope, index, name), v in zip(free, vals, strict=True):
        if scope == "edge":
            edges[index] = replace(edges[index], **{name: float(v)})
        else:
            species[index] = replace(species[index], **{name: float(v)})
    return Circuit(species, edges)


def attribute_bifurcation(
    data: Any,
    circuit: Circuit,
    *,
    free: list[FreeParam] | None = None,
    k_modes: int = 2,
    steps: int = 200,
    seed: int = 0,
    near: float = NEAR_FOLD,
    deep: float = DEEP_BASIN,
) -> BifurcationResult:
    """Score a switch's fold proximity **from data** — fit (optional), score, classify.

    ``data`` is an ``(n_cells, n_species)`` activity-space array (the same space the
    Lyapunov path consumes). Steps:

    1. If ``free`` is given, fit those kinetics to ``data`` with the shipped LNA mixture
       fit (:func:`nudge.inference.lyapunov.fit_lyapunov_parameters`) and rebuild the
       circuit with the recovered values; otherwise score the circuit as given.
    2. Compute :func:`bifurcation_proximity` on the resulting circuit and
       :func:`classify_robustness`.
    3. Calibrate the sequencing **depth** ``scale`` from ``data``
       (:func:`nudge.inference.lyapunov.calibrate_scale`) and run
       :func:`nudge.inference.lyapunov.lna_reliable` at that depth — a caveat on the LNA
       lobe channel only (the two deterministic channels are depth-independent).

    Returns a :class:`BifurcationResult`. The proximity is a property of the *circuit*;
    the data pins the depth so the lobe-channel reliability is honestly reported.
    """
    from nudge.inference.lyapunov import (
        calibrate_scale,
        fit_lyapunov_parameters,
        lna_reliable,
    )

    arr = np.asarray(data, dtype=float)
    recovered: dict[str, float] = {}
    fitted = circuit
    if free:
        rec, _aux, _hist = fit_lyapunov_parameters(
            arr, circuit, free, k_modes=k_modes, steps=steps, seed=seed
        )
        vals = np.array([rec[f] for f in free], dtype=float)
        fitted = _apply_kinetics(circuit, free, vals)
        recovered = {f"{s}[{i}].{n}": float(rec[(s, i, n)]) for (s, i, n) in free}

    score = bifurcation_proximity(fitted)
    call, reason = classify_robustness(score, near=near, deep=deep)

    try:
        scale = calibrate_scale(arr, fitted)
    except Exception:
        scale = float("nan")
    if score is not None and np.isfinite(scale):
        _ok, lna_reason = lna_reliable(fitted, scale)
    else:
        lna_reason = "not applicable (monostable or depth uncalibrated)"

    return BifurcationResult(
        score=score,
        call=call,
        reason=reason,
        scale=scale,
        lna_reason=lna_reason,
        recovered=recovered,
    )
