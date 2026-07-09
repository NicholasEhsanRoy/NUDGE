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

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from nudge.core.circuit import Circuit, Params
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
    The honest single-snapshot outcome: **identify ceiling; abstain (gain_or_threshold)
    between gain and threshold** — the measured degeneracy. The *breaker* is a second
    operating point (M3, ``fit_lyapunov_multi``). Correct-or-abstain, never confidently
    wrong.
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
    seed: int = 0,
) -> tuple[float, float, list[float]]:
    """Fit ONE shared kinetic value jointly across a list of operating ``points``.

    The breaker (M3): a single value of ``free`` must explain *every* operating point at
    once. The true mechanism can (one gain change is one gain change everywhere); the
    confounder cannot — the ``K`` that mimics a gain change at one operating point
    differs at another (the snapshot constraint ``n·ln(K/B)`` moves with ``B``), so a
    shared ``K`` fits both poorly. Each point keeps its own weights + pinned scale/obs.

    Returns ``(recovered shared value, combined mean NLL, history)``.
    """
    n_pts = len(points)
    if n_pts == 0:
        raise ValueError("need at least one operating point")
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
            total = total - jnp.mean(ll)
        return total / n_pts

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


def attribute_lyapunov_multi(
    points_by_mech: dict[str, list[OperatingPoint]] | list[OperatingPoint],
    *,
    target_edge: int = 0,
    k_modes: int = 2,
    steps: int = 200,
    resolve_margin: float = 0.03,
    confound_gap: float = 0.05,
    seed: int = 0,
) -> tuple[str, dict[str, float]]:
    """Multi-operating-point attribution → ``(label, joint NLLs)``: breaks the confound.

    ``points`` is a list of :class:`OperatingPoint` (same perturbed condition at ≥2
    operating points). For each candidate mechanism a **shared** value is fit jointly
    (:func:`fit_lyapunov_multi`); the combined NLLs are compared. With ≥2 points
    gain and threshold separate, so — unlike the single-condition call — it can return a
    bare ``gain``/``threshold``/``ceiling`` when one clearly wins, else ``unresolved``.
    """
    points = (
        points_by_mech if isinstance(points_by_mech, list)
        else next(iter(points_by_mech.values()))
    )
    # Abstain loudly unless EVERY operating point's LNA is trustworthy (one bad Gaussian
    # corrupts the shared-parameter joint fit).
    if not all(lna_reliable(p.circuit, p.scale)[0] for p in points):
        return "unresolved", {}
    nlls: dict[str, float] = {}
    for param, name in _ATTR_PARAMS:
        _val, combined, _hist = fit_lyapunov_multi(
            points, ("edge", target_edge, param),
            k_modes=k_modes, steps=steps, seed=seed,
        )
        nlls[name] = combined
    ordered = sorted(nlls, key=lambda k: nlls[k])
    best, second = ordered[0], ordered[1]
    if nlls[second] - nlls[best] > resolve_margin:
        return best, nlls
    return "unresolved", nlls
