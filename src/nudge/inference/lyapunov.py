"""Covariance-structured (linear-noise) Gaussian-mixture attribution — Lyapunov path.

An **additive, guarded** fit path — it never replaces the energy-distance default. It
fits a Gaussian mixture whose per-mode **means** are the circuit's stable fixed points
and per-mode **covariances** are the linear-noise (Lyapunov) covariances
``A Σ + Σ Aᵀ + D = 0`` — the channel the Fisher-information analysis showed carries the
gain/threshold/ceiling signal that basin *weights* do not
(``design/TOGGLE_ATTRIBUTION_RESEARCH.md``; ``scripts/vv/fisher_sloppiness.py``).

**Why it exists.** A single toggle snapshot confounds gain (Hill ``n``) with threshold
(``K``) — the snapshot constrains only ``n·ln(K/B)`` — so no single-condition fit can
separate them (M2 shows it *abstains*). The breaker is a **second operating point**
(``fit_lyapunov_multi``, M3): a shared kinetic parameter fit jointly across conditions.

**Honest bounds.** The LNA Gaussian is *local* to each stable mode and degrades near a
bifurcation and at low copy number — precisely where a large perturbation pushes the
system. Callers must guard on that (M4); here the forward model is deterministic-LNA.

Differentiability mirrors the transition fit: the fixed-point *locations* come from the
concrete finder each step (a ``stop_gradient`` seed), then the mode **means** are made
differentiable by one implicit-function-theorem Newton step and the **covariances** by
the Lyapunov solve — both differentiable in the free kinetics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from nudge.core.circuit import Circuit, Params
from nudge.inference.bifurcation import bifurcation_proximity
from nudge.inference.fit import FreeParam

__all__ = [
    "OperatingPoint",
    "attribute_lyapunov_multi",
    "attribute_lyapunov_single",
    "calibrate_from_wt",
    "calibrate_scale",
    "fit_lyapunov_multi",
    "fit_lyapunov_parameters",
    "lna_reliable",
    "sample_lna_mixture",
]


@dataclass(frozen=True)
class OperatingPoint:
    """One condition at one operating point, for a joint multi-condition fit (M3).

    ``data`` is the perturbed condition's cells at this operating point; ``circuit`` is
    the WT circuit *at that operating point* (e.g. a shifted basal from a second target,
    the synthetic stand-in for a second Gladstone perturbation); ``scale`` / ``obs_sd``
    are the depth/noise nuisances **pinned from this operating point's own WT**
    (``calibrate_from_wt``). A shared kinetic value is fit across a list of these.
    """

    data: np.ndarray
    circuit: Circuit
    scale: float
    obs_sd: float

#: Restricted-fit params, in a fixed order, and their mechanism names.
_ATTR_PARAMS: tuple[tuple[str, str], ...] = (
    ("n", "gain"),
    ("K", "threshold"),
    ("vmax", "ceiling"),
)


def _apply_free(base: Params, free: list[FreeParam], vals: Array) -> Params:
    """Functionally override the ``free`` params in ``base`` with ``vals`` (autodiff).

    ``base`` carries single (not per-cell) leaves — shape ``(n_species,)`` /
    ``(n_edges,)``
    — since the fixed points / covariances are a deterministic property of the circuit.
    """
    params: Params = {
        "species": dict(base["species"]),
        "edges": dict(base["edges"]),
    }
    for (scope, index, name), v in zip(free, vals, strict=True):
        coll = "edges" if scope == "edge" else "species"
        params[coll][name] = params[coll][name].at[index].set(v)
    return params


def _drift(circuit: Circuit, x: Array, params: Params) -> Array:
    decay = params["species"]["decay"]
    return circuit.production(jnp.maximum(x, 0.0), params) - decay * x


def _mode_mean(circuit: Circuit, params: Params, root: Array) -> Array:
    """IFT-differentiable fixed point: value ≈ root, grad = −A⁻¹ ∂f/∂θ."""
    x0 = jax.lax.stop_gradient(root)
    f = _drift(circuit, x0, params)
    jac = jax.jacobian(lambda x: _drift(circuit, x, params))(x0)
    return x0 - jnp.linalg.solve(jac, f)


def _mode_cov(circuit: Circuit, params: Params, mu: Array) -> Array:
    """LNA covariance at ``mu`` via the Lyapunov equation (Kronecker solve)."""
    jac = jax.jacobian(lambda x: _drift(circuit, x, params))(mu)
    decay = params["species"]["decay"]
    diff = jnp.diag(2.0 * decay * jnp.clip(mu, 1e-9))
    n = mu.shape[0]
    kron = jnp.kron(jnp.eye(n), jac) + jnp.kron(jac, jnp.eye(n))
    sig = jnp.linalg.solve(kron, -diff.reshape(-1)).reshape(n, n)
    return 0.5 * (sig + sig.T)


def _mvn_logpdf(y: Array, mean: Array, cov: Array) -> Array:
    """Log N(y; mean, cov) for a batch ``y`` (n_cells, G) and one component."""
    d = y - mean
    sol = jnp.linalg.solve(cov, d.T).T
    quad = jnp.sum(d * sol, axis=1)
    _, logdet = jnp.linalg.slogdet(cov)
    return -0.5 * (quad + logdet + mean.shape[0] * jnp.log(2 * jnp.pi))


def _updated_circuit(
    circuit: Circuit, free: list[FreeParam], vals: np.ndarray
) -> Circuit:
    species = list(circuit.species)
    edges = list(circuit.edges)
    for (scope, index, name), v in zip(free, vals, strict=True):
        if scope == "edge":
            edges[index] = replace(edges[index], **{name: float(v)})
        else:
            species[index] = replace(species[index], **{name: float(v)})
    return Circuit(species, edges)


def _stable_roots(
    circuit: Circuit, free: list[FreeParam], vals: np.ndarray
) -> list[np.ndarray]:
    """Concrete stable fixed points of the current circuit (mean-sorted), or ``[]``."""
    cur = _updated_circuit(circuit, free, vals)
    fps = cur.fixed_points()
    if fps is None:
        return []
    return [np.asarray(s, dtype=float) for s, lab in fps if lab == "stable"]


def sample_lna_mixture(
    circuit: Circuit,
    n_cells: int,
    key: Array,
    *,
    free: list[FreeParam] | None = None,
    vals: np.ndarray | None = None,
    scale: float = 1.0,
    obs_sd: float = 0.05,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Sample cells from the LNA Gaussian mixture — the inverse-crime data for M1.

    Each mode ``k`` contributes ``N(scale·μ_k, scale²·Σ_k + obs_sd²·I)`` with mixing
    ``weights`` (default uniform). ``μ_k`` / ``Σ_k`` are the stable fixed points + their
    Lyapunov covariances at the (optionally ``free``-overridden) kinetics.
    """
    free = free or []
    v = np.asarray(vals if vals is not None else [], dtype=float)
    roots = _stable_roots(circuit, free, v)
    if not roots:
        raise ValueError("no stable modes to sample from")
    base = _apply_free(circuit.base_params(), free, jnp.asarray(v))
    means, covs = [], []
    for r in roots:
        mu = _mode_mean(circuit, base, jnp.asarray(r))
        cov = _mode_cov(circuit, base, mu)
        means.append(scale * np.asarray(mu))
        covs.append(scale**2 * np.asarray(cov) + obs_sd**2 * np.eye(len(r)))
    k = len(roots)
    w = np.full(k, 1.0 / k) if weights is None else np.asarray(weights)
    key, ka = jax.random.split(key)
    assign = np.asarray(jax.random.choice(ka, k, (n_cells,), p=jnp.asarray(w)))
    out = np.empty((n_cells, circuit.n_species))
    for i in range(n_cells):
        key, ks = jax.random.split(key)
        c = int(assign[i])
        out[i] = np.asarray(
            jax.random.multivariate_normal(
                ks, jnp.asarray(means[c]), jnp.asarray(covs[c])
            )
        )
    return out


def calibrate_scale(
    data: np.ndarray,
    circuit: Circuit,
    *,
    weights: np.ndarray | None = None,
) -> float:
    """Pin the global scale (≈ sequencing depth) from ``data`` by a mean moment-match.

    ``scale = ⟨data_mean, model_mean⟩ / ‖model_mean‖²`` where ``model_mean`` is the
    weight-averaged stable-mode mean at the circuit's nominal kinetics (``scale = 1``).
    Because ``scale`` and ``vmax`` are degenerate (both multiply the mode means; see
    ``fit_lyapunov_parameters``), the depth nuisance must be pinned from data — the
    analogue of library-size normalization — so attribution turns on distribution
    *shape*, not overall magnitude. Uses the WT/nominal modes as the reference.
    """
    roots = _stable_roots(circuit, [], np.asarray([], dtype=float))
    if not roots:
        raise ValueError("no stable modes to calibrate against")
    base = circuit.base_params()
    means = [np.asarray(_mode_mean(circuit, base, jnp.asarray(r))) for r in roots]
    k = len(means)
    w = np.full(k, 1.0 / k) if weights is None else np.asarray(weights)
    model_mean = np.sum([wi * mi for wi, mi in zip(w, means, strict=True)], axis=0)
    data_mean = np.asarray(data, dtype=float).mean(axis=0)
    return float(data_mean @ model_mean / (model_mean @ model_mean + 1e-12))


def fit_lyapunov_parameters(
    data: np.ndarray,
    circuit: Circuit,
    free: list[FreeParam],
    *,
    k_modes: int = 2,
    steps: int = 300,
    learning_rate: float = 0.05,
    scale_init: float = 1.0,
    obs_sd_init: float = 0.1,
    fit_scale: bool = True,
    fit_obs: bool = True,
    seed: int = 0,
) -> tuple[dict[FreeParam, float], dict[str, Any], list[float]]:
    """Fit ``free`` kinetics by MAXIMIZING the LNA Gaussian-mixture NLL of ``data``.

    ``data`` is an ``(n_cells, n_species)`` activity-space array (identity readout). The
    mixture has ``k_modes`` components (the circuit's stable fixed points, recomputed
    step as ``stop_gradient`` seeds), with means ``scale·μ_k(θ)`` and covariances
    ``scale²·Σ_k(θ) + obs_var·I``; mixture weights, ``scale`` and ``obs_var`` are fitted
    nuisances. Returns ``(recovered kinetics, {weights, scale, obs_sd}, nll_history)``.

    Requires the WT circuit to present exactly ``k_modes`` stable fixed points; if a
    step finds a different count the previous roots are reused (a bifurcation).
    """
    y = jnp.asarray(np.asarray(data, dtype=np.float32))
    n_free = len(free)
    base = circuit.base_params()

    log_kin0 = jnp.log(jnp.array([_param_value(circuit, f) for f in free]))
    logits0 = jnp.zeros(k_modes)
    theta = jnp.concatenate(
        [log_kin0, logits0,
         jnp.array([np.log(scale_init), np.log(obs_sd_init)], dtype=log_kin0.dtype)]
    )

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(theta)
    g_dim = circuit.n_species
    eye = jnp.eye(g_dim)

    def nll(vec: Array, roots: Array) -> Array:
        roots = jax.lax.stop_gradient(roots)  # (k_modes, n_species) concrete seeds
        vals = jnp.exp(vec[:n_free])
        logits = vec[n_free : n_free + k_modes]
        # A free global scale is degenerate with vmax (both scale the mode means); a
        # free obs floor with the covariance. Freeze either via stop_gradient when the
        # caller knows it (inverse-crime / a calibrated normalization).
        s_raw = vec[n_free + k_modes]
        o_raw = vec[n_free + k_modes + 1]
        scale = jnp.exp(s_raw if fit_scale else jax.lax.stop_gradient(s_raw))
        obs_var = jnp.exp(o_raw if fit_obs else jax.lax.stop_gradient(o_raw)) ** 2
        params = _apply_free(base, free, vals)
        log_w = jax.nn.log_softmax(logits)
        comps = []
        for k in range(k_modes):
            mu = _mode_mean(circuit, params, roots[k])
            cov = _mode_cov(circuit, params, mu)
            mean_obs = scale * mu
            cov_obs = scale**2 * cov + obs_var * eye
            comps.append(log_w[k] + _mvn_logpdf(y, mean_obs, cov_obs))
        ll = jax.scipy.special.logsumexp(jnp.stack(comps), axis=0)  # (n_cells,)
        return -jnp.mean(ll)

    @jax.jit
    def step(
        vec: Array, state: optax.OptState, roots: Array
    ) -> tuple[Array, optax.OptState, Array]:
        loss, grad = jax.value_and_grad(nll)(vec, roots)
        updates, state = optimizer.update(grad, state)
        return jnp.asarray(optax.apply_updates(vec, updates)), state, loss

    def seeds(vec: Array) -> np.ndarray | None:
        roots = _stable_roots(circuit, free, np.exp(np.asarray(vec[:n_free])))
        roots = sorted(roots, key=lambda r: tuple(float(x) for x in r))
        if len(roots) != k_modes:
            return None
        return np.stack(roots)

    history: list[float] = []
    last_roots = seeds(theta)
    if last_roots is None:
        raise ValueError(f"WT circuit must present {k_modes} stable modes to fit")
    for _ in range(steps):
        cur = seeds(theta)
        if cur is not None:
            last_roots = cur
        theta, opt_state, loss = step(theta, opt_state, jnp.asarray(last_roots))
        history.append(float(loss))

    kin = np.exp(np.asarray(theta[:n_free]))
    recovered = {f: float(v) for f, v in zip(free, kin, strict=True)}
    aux = {
        "weights": np.asarray(jax.nn.softmax(theta[n_free : n_free + k_modes])),
        "scale": float(np.exp(theta[n_free + k_modes])),
        "obs_sd": float(np.exp(theta[n_free + k_modes + 1])),
    }
    return recovered, aux, history


def _param_value(circuit: Circuit, free: FreeParam) -> float:
    scope, index, name = free
    obj = circuit.edges[index] if scope == "edge" else circuit.species[index]
    return float(getattr(obj, name))


def calibrate_from_wt(
    wt_data: np.ndarray,
    circuit: Circuit,
    *,
    k_modes: int = 2,
    steps: int = 150,
    seed: int = 0,
) -> tuple[float, float]:
    """Pin the two depth/noise nuisances from the WT condition (nominal kinetics).

    Returns ``(scale, obs_sd)``. ``scale`` is the moment-match depth
    (``calibrate_scale``);
    ``obs_sd`` is fit on the WT mixture with ``scale`` pinned. Both come from WT — where
    the kinetics are nominal, so the data magnitude is *pure depth*, not a mechanism —
    are then held fixed for every perturbed condition. Calibrating the scale from a
    a *perturbed* condition's own magnitude would silently absorb a ceiling change into
    depth
    (``scale`` and ``vmax`` are degenerate), making ceiling unidentifiable. This is the
    linear-noise analogue of library-size normalization vs a housekeeping reference.
    """
    scale = calibrate_scale(wt_data, circuit)
    _rec, aux, _hist = fit_lyapunov_parameters(
        wt_data, circuit, [], k_modes=k_modes, steps=steps, seed=seed,
        scale_init=scale, fit_scale=False, fit_obs=True,
    )
    return scale, float(aux["obs_sd"])


def lna_reliable(
    circuit: Circuit,
    scale: float,
    *,
    min_count: float = 15.0,
    sep_ratio: float = 1.0,
) -> tuple[bool, str]:
    """Is the linear-noise Gaussian mixture trustworthy for ``circuit`` at this depth?

    The LNA Gaussian is *local and second-order*: it breaks down (a) **near a
    saddle-node bifurcation**, where a mode's covariance diverges (the Lyapunov solution
    Jacobian eigenvalue → 0) and the lobes stop being resolvable, and (b) **at low copy
    number**, where the discrete/skewed count distribution is not Gaussian. Attribution
    must **abstain loudly** in both regimes rather than trust a bad Gaussian. Returns
    ``(ok, reason)``:

    - **not bistable** — fewer than two stable modes (nothing to attribute);
    - **near bifurcation** — a lobe's std ``√λ_max(Σ)`` exceeds ``sep_ratio`` × the
      inter-mode separation ``min‖μ_i − μ_j‖`` (the lobes overlap → merging). Measured
      against the *separation*, not a lobe's own mean, so a near-zero OFF state (tiny
      ‖μ‖) is not mistaken for a bifurcation;
    - **insufficient depth** — the brightest state's expected counts ``scale·max|μ| <
      min_count`` (the Gaussian relaxation of the counts is untrustworthy).
    """
    modes = circuit.mode_covariances()
    if modes is None or len(modes) < 2:
        return False, "not bistable (need ≥2 stable modes)"
    means = [np.asarray(mean, dtype=float) for mean, _ in modes]
    peak = max(float(np.max(np.abs(mean))) for mean in means)
    if scale * peak < min_count:
        sp = scale * peak
        return False, f"insufficient depth (scale·peak={sp:.1f} < {min_count})"
    sep = min(
        float(np.linalg.norm(means[i] - means[j]))
        for i in range(len(means))
        for j in range(i + 1, len(means))
    )
    for _mean, cov in modes:
        spread = float(np.sqrt(np.max(np.linalg.eigvalsh(cov))))
        if spread > sep_ratio * sep:
            return False, f"near bifurcation (lobe std {spread:.2f} > {sep:.2f} sep)"
    return True, "ok"


def _decide_lyapunov(
    nlls: dict[str, float], ceiling_margin: float, confound_gap: float
) -> str:
    """The honest single-condition call from the restricted NLL profile.

    ``ceiling`` only when the free-vmax fit is the best AND clearly below the
    gain/threshold pair (ceiling is identifiable from a snapshot). ``gain_or_threshold``
    when gain/threshold are the best and ~indistinguishable (the measured confound — we
    abstain *between* them, not guess). Otherwise ``unresolved``. It never returns
    a bare ``gain`` or ``threshold``: a single snapshot cannot separate them.
    """
    n_gain, n_thr, n_cei = nlls["n"], nlls["K"], nlls["vmax"]
    best = min(nlls, key=lambda k: nlls[k])
    if best == "vmax" and min(n_gain, n_thr) - n_cei > ceiling_margin:
        return "ceiling"
    if best in ("n", "K") and abs(n_gain - n_thr) < confound_gap:
        return "gain_or_threshold"
    return "unresolved"


def attribute_lyapunov_single(
    cond_data: np.ndarray,
    circuit: Circuit,
    *,
    wt_data: np.ndarray | None = None,
    scale: float | None = None,
    obs_sd: float | None = None,
    target_edge: int = 0,
    k_modes: int = 2,
    steps: int = 200,
    ceiling_margin: float = 0.05,
    confound_gap: float = 0.05,
    seed: int = 0,
) -> tuple[str, dict[str, float]]:
    """Single-condition covariance attribution → ``(label, restricted NLLs)``.

    Runs the three restricted fits (free-``n`` / ``K`` / ``vmax`` of ``target_edge``,
    from WT) with ``scale`` / ``obs_sd`` **pinned** (from ``calibrate_from_wt`` if not
    given — a ceiling must not hide in the depth nuisance), then ``_decide_lyapunov``.
    Under the *inverse crime* (cells sampled from the LNA mixture) the single-snapshot
    outcome is **identify ceiling; abstain (gain_or_threshold) between gain and
    threshold**. On **independent stochastic (SSA) toggle data that identification does
    NOT survive**: the free-vmax fit becomes the *worst* explanation of a true ceiling,
    so a true ceiling mis-narrows to ``gain_or_threshold`` and gain/threshold abstain
    (``unresolved``) — a *single toggle snapshot degenerates* (FINDINGS "independent-SSA
    validation"). It still only ever returns an abstention-class label from one snapshot
    (``unresolved`` / ``gain_or_threshold``), never a bare gain/threshold/ceiling, so
    never confidently wrong; but **do not read a positive from one operating point on
    real data**. The *breaker* is a second operating point (M3, ``fit_lyapunov_multi``),
    which recovers threshold + ceiling on independent SSA and abstains on gain.
    """
    if scale is None or obs_sd is None:
        if wt_data is None:
            raise ValueError("provide wt_data, or both scale and obs_sd")
        scale, obs_sd = calibrate_from_wt(wt_data, circuit, k_modes=k_modes, seed=seed)
    ok, _reason = lna_reliable(circuit, scale)
    if not ok:
        return "unresolved", {}  # abstain loudly: the LNA Gaussian is untrustworthy
    nlls: dict[str, float] = {}
    for param, _name in _ATTR_PARAMS:
        _rec, _aux, hist = fit_lyapunov_parameters(
            cond_data, circuit, [("edge", target_edge, param)],
            k_modes=k_modes, steps=steps, seed=seed,
            scale_init=scale, obs_sd_init=obs_sd, fit_scale=False, fit_obs=False,
        )
        nlls[param] = float(np.mean(hist[-20:]))
    return _decide_lyapunov(nlls, ceiling_margin, confound_gap), nlls


def fit_lyapunov_multi(
    points: list[OperatingPoint],
    free: FreeParam,
    *,
    k_modes: int = 2,
    steps: int = 200,
    learning_rate: float = 0.05,
    weights: list[float] | None = None,
    seed: int = 0,
) -> tuple[float, float, list[float]]:
    """Fit ONE shared kinetic value jointly across a list of operating ``points``.

    The breaker (M3): a single value of ``free`` must explain *every* operating point at
    once. The true mechanism can (one gain change is one gain change everywhere); the
    confounder cannot — the ``K`` that mimics a gain change at one operating point
    differs at another (the snapshot constraint ``n·ln(K/B)`` moves with ``B``), so a
    shared ``K`` fits both poorly. Each point keeps its own weights + pinned scale/obs.

    ``weights`` (optional, one per point) scale each point's contribution to the joint
    loss — a **graded near-fold down-weighting** (NUDGE-LIM-017): a point approaching the
    fold contributes less, so it cannot dominate the shared-parameter argmin. ``None`` ⇒
    equal weights (the original behaviour). The combined NLL is the weight-normalized mean.

    Returns ``(recovered shared value, combined mean NLL, history)``.
    """
    n_pts = len(points)
    if n_pts == 0:
        raise ValueError("need at least one operating point")
    w_pt = (
        jnp.ones(n_pts) if weights is None
        else jnp.asarray(np.asarray(weights, dtype=float))
    )
    w_sum = jnp.maximum(jnp.sum(w_pt), 1e-9)
    bases = [p.circuit.base_params() for p in points]
    ys = [jnp.asarray(np.asarray(p.data, dtype=np.float32)) for p in points]
    eyes = [jnp.eye(p.circuit.n_species) for p in points]

    val0 = jnp.log(jnp.asarray(_param_value(points[0].circuit, free)))
    logits0 = jnp.zeros(n_pts * k_modes)
    theta = jnp.concatenate([val0[None], logits0])

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(theta)

    def nll(vec: Array, roots_all: Array) -> Array:
        roots_all = jax.lax.stop_gradient(roots_all)  # (n_pts, k_modes, n_species)
        val = jnp.exp(vec[0])
        total = jnp.asarray(0.0)
        for i, point in enumerate(points):
            params = _apply_free(bases[i], [free], val[None])
            w = jax.nn.log_softmax(vec[1 + i * k_modes : 1 + (i + 1) * k_modes])
            comps = []
            for k in range(k_modes):
                mu = _mode_mean(point.circuit, params, roots_all[i, k])
                cov = _mode_cov(point.circuit, params, mu)
                cov_obs = point.scale**2 * cov + point.obs_sd**2 * eyes[i]
                comps.append(w[k] + _mvn_logpdf(ys[i], point.scale * mu, cov_obs))
            ll = jax.scipy.special.logsumexp(jnp.stack(comps), axis=0)
            total = total - w_pt[i] * jnp.mean(ll)
        return total / w_sum

    @jax.jit
    def step(
        vec: Array, state: optax.OptState, roots_all: Array
    ) -> tuple[Array, optax.OptState, Array]:
        loss, grad = jax.value_and_grad(nll)(vec, roots_all)
        updates, state = optimizer.update(grad, state)
        return jnp.asarray(optax.apply_updates(vec, updates)), state, loss

    def seeds(vec: Array) -> np.ndarray | None:
        val = float(np.exp(np.asarray(vec[0])))
        allr = []
        for point in points:
            roots = _stable_roots(point.circuit, [free], np.asarray([val]))
            roots = sorted(roots, key=lambda r: tuple(float(x) for x in r))
            if len(roots) != k_modes:
                return None
            allr.append(np.stack(roots))
        return np.stack(allr)

    last = seeds(theta)
    if last is None:
        raise ValueError(f"every operating point must present {k_modes} stable modes")
    history: list[float] = []
    for _ in range(steps):
        cur = seeds(theta)
        if cur is not None:
            last = cur
        theta, opt_state, loss = step(theta, opt_state, jnp.asarray(last))
        history.append(float(loss))

    return float(np.exp(theta[0])), float(np.mean(history[-20:])), history


def _proximity_weight(prox: float, full_thresh: float, width: float = 0.04) -> float:
    """Graded near-fold down-weight for one operating point (NUDGE-LIM-017).

    A well-buffered point (``proximity ≤ full_thresh``) keeps full weight 1.0; past the
    threshold the weight decays as a Gaussian in the excess proximity, so a FAR near-fold
    point (e.g. the round-1 witness at proximity 0.231) contributes ~0 to the joint fit and
    cannot corrupt it. Points *just* past the threshold are only mildly down-weighted — the
    best-pair corroboration handles those, since proximity cannot separate a useful point
    (measured 0.112) from a corrupting one (0.119).
    """
    excess = max(prox - full_thresh, 0.0)
    return float(np.exp(-((excess / width) ** 2)))


#: Mechanism name → the kinetic parameter it frees (the inverse of ``_ATTR_PARAMS``).
_MECH_TO_PARAM: dict[str, str] = {name: param for param, name in _ATTR_PARAMS}

#: Identifiability-gate contamination cut, in **log-units** on a runner-up kinetic.
#: After the breaker resolves a single mechanism X, NUDGE fits the joint two-mechanism
#: model (X together with each runner-up Y) and measures Y's *identifiable* displacement
#: from its no-change (nominal) value. A genuine single-mechanism shift leaves every
#: runner-up either a free nuisance the joint fit cannot pin (unidentifiable → contributes
#: 0) or barely displaced — MEASURED ≤ 0.12 for the resolvable K=2.0 threshold and the
#: genuine ceiling. A confident-WRONG resolution of a threshold-DOMINATED large-gain
#: perturbation (Hill n 4→1.5) forces the runner-up gain ≈ 1.0 log-units off nominal,
#: because the second operating point did NOT break the gain⇄threshold degeneracy (one
#: point slid monostable, the other sits at the fold). The cut 0.5 sits in the measured gap
#: (≤ 0.12 genuine vs ≈ 1.0 confident-wrong) with ~0.4-log margin on each side — a MEASURED
#: separator, not a guessed constant (FINDINGS "P7"; NUDGE-LIM-025;
#: scripts/redteam/lyapunov_multi_gain_threshold_hole.py).
_CONTAM_MARGIN = 0.5


def _fixed_root_multi_loss(
    points: list[OperatingPoint], frees: list[FreeParam], k_modes: int
) -> Callable[[Array], Array]:
    """Smooth (pinned-mode-seed) multi-point mean NLL over the shared ``frees``.

    The multi-point analogue of :func:`nudge.inference.uncertainty.lyapunov_nll_loss`:
    each operating point's stable-mode seeds are pinned at its WT circuit's nominal fixed
    points (``stop_gradient`` constants inside :func:`_mode_mean`), so the Hessian in the
    shared log-kinetics is a clean observed-Fisher curvature that
    :func:`~nudge.inference.uncertainty.laplace_posterior` can read for identifiability.
    Used ONLY by the identifiability gate (:func:`_resolution_contamination`), never by the
    attribution fit itself (which re-seeds each step). Assumes every point is bistable at
    nominal kinetics (the caller guards on that).
    """
    bases = [p.circuit.base_params() for p in points]
    ys = [jnp.asarray(np.asarray(p.data, dtype=np.float32)) for p in points]
    eyes = [jnp.eye(p.circuit.n_species) for p in points]
    roots: list[Array] = []
    for p in points:
        r = _stable_roots(p.circuit, [], np.asarray([], dtype=float))
        r = sorted(r, key=lambda x: tuple(float(v) for v in x))
        roots.append(jnp.asarray(np.stack(r[:k_modes])))
    log_w = jnp.log(jnp.full((k_modes,), 1.0 / k_modes))

    def loss(log_theta: Array) -> Array:
        vals = jnp.exp(log_theta)
        total = jnp.asarray(0.0)
        for i, point in enumerate(points):
            params = _apply_free(bases[i], frees, vals)
            comps = []
            for k in range(k_modes):
                mu = _mode_mean(point.circuit, params, roots[i][k])
                cov = _mode_cov(point.circuit, params, mu)
                cov_obs = point.scale**2 * cov + point.obs_sd**2 * eyes[i]
                comps.append(log_w[k] + _mvn_logpdf(ys[i], point.scale * mu, cov_obs))
            ll = jax.scipy.special.logsumexp(jnp.stack(comps), axis=0)
            total = total - jnp.mean(ll)
        return total / len(points)

    return loss


def _fit_fixed_root(
    loss: Callable[[Array], Array], init: Array, *, steps: int = 600, lr: float = 0.03
) -> np.ndarray:
    """Adam-minimize the smooth fixed-root ``loss`` from ``init`` → the log-optimum θ*."""
    theta = jnp.asarray(init)
    opt = optax.adam(lr)
    state = opt.init(theta)

    @jax.jit
    def step(t: Array, s: optax.OptState) -> tuple[Array, optax.OptState]:
        _v, grad = jax.value_and_grad(loss)(t)
        updates, s = opt.update(grad, s)
        return jnp.asarray(optax.apply_updates(t, updates)), s

    for _ in range(steps):
        theta, state = step(theta, state)
    return np.asarray(theta)


def _resolution_contamination(
    points: list[OperatingPoint],
    winner: str,
    *,
    target_edge: int,
    k_modes: int,
) -> tuple[float, str]:
    """Does the data need a SECOND mechanism on top of the resolved ``winner``?

    The multi-point breaker resolves a single bare mechanism by an NLL gap — but a
    large-gain perturbation that is *threshold-dominated* in the LNA moments (its perturbed
    condition sliding to/through the saddle-node fold, so the second operating point never
    breaks the gain⇄threshold degeneracy) is resolved to ``threshold`` with a LARGE,
    confident gap even though the truth is gain (``scripts/redteam/
    lyapunov_multi_gain_threshold_hole.py``; NUDGE-LIM-025). The NLL gap cannot see this;
    the joint-fit **curvature** can. For each runner-up mechanism Y this fits the joint
    (winner, Y) two-mechanism model (:func:`_fixed_root_multi_loss`) and reads the Laplace
    posterior (:func:`~nudge.inference.uncertainty.laplace_posterior` — the SAME machinery
    the single-operating-point call abstains with): if Y is **identifiable** AND
    **displaced** from its no-change (nominal) value, the data demonstrably needs Y too and
    the single-mechanism resolution is contaminated. Returns ``(max_contamination,
    detail)`` — ``contamination = |log Y* − log Y_nom|`` when Y is identifiable, else 0,
    maxed over the runner-ups; the caller abstains above :data:`_CONTAM_MARGIN`.
    """
    from nudge.inference.uncertainty import laplace_posterior  # lazy: avoid import cycle

    n_data = int(sum(np.asarray(p.data).shape[0] for p in points))
    fw: FreeParam = ("edge", target_edge, _MECH_TO_PARAM[winner])
    win_nom = float(_param_value(points[0].circuit, fw))
    worst = 0.0
    details: list[str] = []
    for param, other in _ATTR_PARAMS:
        if other == winner:
            continue
        fy: FreeParam = ("edge", target_edge, param)
        y_nom = float(_param_value(points[0].circuit, fy))
        loss = _fixed_root_multi_loss(points, [fw, fy], k_modes)
        init = jnp.log(jnp.array([win_nom, y_nom]))
        theta = _fit_fixed_root(loss, init)
        post = laplace_posterior(loss, theta, names=[winner, other], n_data=n_data)
        y_ci = post.marginal_ci[1]
        disp = abs(float(theta[1]) - float(np.log(y_nom)))
        contam = disp if y_ci.identifiable else 0.0
        worst = max(worst, contam)
        details.append(
            f"{other}: id={y_ci.identifiable} disp={disp:.3f} contam={contam:.3f}"
        )
    return worst, "; ".join(details)


def attribute_lyapunov_multi(
    points_by_mech: dict[str, list[OperatingPoint]] | list[OperatingPoint],
    *,
    target_edge: int = 0,
    k_modes: int = 2,
    steps: int = 200,
    resolve_margin: float = 0.03,
    confound_gap: float = 0.05,
    well_buffered_margin: float = 0.15,
    seed: int = 0,
) -> tuple[str, dict[str, float]]:
    """Multi-operating-point attribution → ``(label, joint NLLs)``: breaks the confound.

    ``points`` is a list of :class:`OperatingPoint` (same perturbed condition at ≥2
    operating points). For each candidate mechanism a **shared** value is fit jointly
    (:func:`fit_lyapunov_multi`); the combined NLLs are compared. With ≥2 points
    gain and threshold separate, so — unlike the single-condition call — it can return a
    bare ``gain``/``threshold``/``ceiling`` when one clearly wins, else ``unresolved``.

    **Near-fold robustness (NUDGE-LIM-017).** The breaker assumes each operating point
    contributes *trustworthy* moments to the shared-parameter joint fit. But
    :func:`lna_reliable` — the per-point LNA gate — trips solely at lobe *overlap* (or low
    depth); a point still *approaching* the saddle-node fold, whose Lyapunov covariance is
    already biased but whose noise lobes have not yet merged, passes it and **poisons the
    joint argmin** (a near-fold 3rd toggle point flips a true ``ceiling`` to a confident
    ``threshold`` — ``scripts/redteam/nearfold_thirdpoint_hole.py``). A hard proximity gate
    is a **knife-edge** and does not close it: measurement (``well_buffered_margin`` probe)
    shows a *useful* second operating point (proximity 0.112) and a *corrupting* one (0.119)
    sit only 0.007 apart, so proximity cannot separate them and neither a threshold nor a
    pure proximity weighting suffices. Two mechanisms are used instead:

    1. **Graded down-weighting** (:func:`_proximity_weight`) — each point's contribution to
       the joint loss decays smoothly with its bifurcation proximity past
       ``well_buffered_margin``, so a *far* near-fold point (the round-1 witness at proximity
       0.231) is effectively ignored and cannot corrupt the fit.
    2. **Best-buffered-pair corroboration** — with >2 points, a resolved bare mechanism is
       accepted only if it AGREES with the call from the two *most-buffered* points; if a
       marginal point changed the answer, NUDGE **abstains**. Threshold-free, it closes the
       0.007 knife-edge that weighting cannot, while keeping a genuinely well-buffered
       multi-point set resolvable.

    **Identifiability gate (NUDGE-LIM-025).** The near-fold robustness above inspects the
    WT/CONTROL circuit at each operating point; it is blind to a *perturbed* condition that
    has itself slid to/through the fold. A large GAIN knockdown (Hill ``n`` 4→1.5) is
    **threshold-dominated** in the LNA moments (a ``ΔK`` mimics the dominant shift) and drives
    the perturbed condition monostable at one operating point / to the fold at the other — so
    the second operating point never breaks the gain⇄threshold degeneracy, yet the pure
    NLL-gap test resolves a **confident-wrong** ``threshold`` (gap ≈1.7 ≫ ``resolve_margin``;
    ``scripts/redteam/lyapunov_multi_gain_threshold_hole.py``). The NLL gap measures fit
    quality, not identifiability. So after a bare mechanism resolves, NUDGE fits the joint
    (winner, runner-up) two-mechanism model and reads its Laplace posterior
    (:func:`_resolution_contamination`, reusing :func:`~nudge.inference.uncertainty.
    laplace_posterior`): if a runner-up is **identifiable and displaced** from its no-change
    value beyond the MEASURED cut :data:`_CONTAM_MARGIN` (genuine resolutions ≤ 0.12 vs the
    hole ≈ 1.0), the data demonstrably needs a second mechanism and NUDGE **abstains**.

    **Graceful degradation (NUDGE-LIM-025).** An operating point whose circuit has lost
    bistability (monostable — :func:`~nudge.inference.bifurcation.bifurcation_proximity`
    ``None`` / < 2 stable modes) returns ``("unresolved", {})`` ("bistability lost") instead of
    raising deep in the k_modes fit; the joint-fit calls are wrapped so a mid-fit drift to
    monostability abstains rather than crashes.
    """
    points = (
        points_by_mech if isinstance(points_by_mech, list)
        else next(iter(points_by_mech.values()))
    )
    # Graceful degradation (NUDGE-LIM-025): an operating point whose circuit has LOST
    # bistability (monostable — bifurcation_proximity is None / < 2 stable modes) carries no
    # two-mode LNA moments, so the k_modes joint fit is meaningless and would otherwise
    # raise deep inside the fit ("must present 2 stable modes"). Abstain loudly
    # ("bistability lost") rather than crash. This is the perturbed-side analogue of the
    # lna_reliable guard below (which trips on the same, plus low depth / lobe overlap).
    if any(bifurcation_proximity(p.circuit) is None for p in points):
        return "unresolved", {}
    # Abstain loudly unless EVERY operating point's LNA is trustworthy (one bad Gaussian
    # corrupts the shared-parameter joint fit).
    if not all(lna_reliable(p.circuit, p.scale)[0] for p in points):
        return "unresolved", {}

    # Per-point bifurcation proximity (the deterministic dial lna_reliable ignores).
    prox = [
        (s.proximity if (s := bifurcation_proximity(p.circuit)) is not None else 0.0)
        for p in points
    ]

    def _resolve(idx: list[int]) -> tuple[str, dict[str, float]]:
        """Graded-weighted joint fit over the points ``idx`` → (call, NLLs)."""
        pts = [points[i] for i in idx]
        wts = [_proximity_weight(prox[i], well_buffered_margin) for i in idx]
        nlls: dict[str, float] = {}
        for param, name in _ATTR_PARAMS:
            _val, combined, _hist = fit_lyapunov_multi(
                pts, ("edge", target_edge, param),
                k_modes=k_modes, steps=steps, weights=wts, seed=seed,
            )
            nlls[name] = combined
        ordered = sorted(nlls, key=lambda k: nlls[k])
        best, second = ordered[0], ordered[1]
        if nlls[second] - nlls[best] > resolve_margin:
            return best, nlls
        return "unresolved", nlls

    try:
        call, nlls = _resolve(list(range(len(points))))
        # Best-buffered-pair corroboration: proximity cannot separate a useful 2nd point
        # (0.112) from a corrupting near-fold one (0.119), so require the resolved bare
        # mechanism to be confirmed by the two MOST-BUFFERED points. If a marginal point
        # changed the answer, abstain (fail-safe) — this, not the graded weighting, is what
        # closes the knife-edge.
        if call in ("gain", "threshold", "ceiling") and len(points) > 2:
            order = sorted(range(len(points)), key=lambda i: prox[i])
            pair_call, _ = _resolve(order[:2])
            if pair_call != call:
                return "unresolved", nlls
    except ValueError:
        # A shared parameter drifted an operating point monostable mid-fit (the k_modes
        # failure — "must present 2 stable modes"): bistability lost. Abstain gracefully
        # (NUDGE-LIM-025) instead of surfacing a raw stack trace.
        return "unresolved", {}

    # Identifiability gate (NUDGE-LIM-025): trust a resolved bare mechanism only if the
    # second operating point MEASURABLY broke the gain⇄threshold degeneracy — i.e. the data
    # does not ALSO demand a runner-up mechanism. A threshold-dominated large-gain
    # perturbation is resolved 'threshold' with a large NLL gap, yet the joint-fit curvature
    # shows the runner-up gain is identifiable and displaced ~1 log-unit from no-change; we
    # abstain there rather than emit a confident-wrong single-mechanism call. The NLL gap
    # (which resolves) is blind to this; the Laplace posterior of the joint fit is not.
    if call in ("gain", "threshold", "ceiling"):
        try:
            contam, _detail = _resolution_contamination(
                points, call, target_edge=target_edge, k_modes=k_modes
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return "unresolved", nlls  # an unreadable curvature ⇒ abstain (fail-safe)
        if contam > _CONTAM_MARGIN:
            return "unresolved", nlls
    return call, nlls
