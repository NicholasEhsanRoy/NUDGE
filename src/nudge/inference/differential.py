"""Comparative / differential attribution — WHICH knob differs between two contexts.

The same perturbation, run in **two contexts** (drug-resistant vs sensitive line; donor
A vs B; disease vs healthy), can differ mechanistically in one of three ways NUDGE speaks
in — the switch's **threshold** (``K``), **gain** (Hill ``n``), or **ceiling**
(``v_max``). A resistant line that raised its *ceiling* needs more dose of the *same*
drug; one that rewired its *gain / threshold* needs a *different* drug class. Linear
differential expression cannot make that distinction — it sees only "the level moved".

This module fits the two contexts **jointly** with a **shared-vs-per-context** parameter
structure and **BIC-selects which SINGLE parameter must differ** to explain them:

- **shared** (no per-context difference) — one ``K``, ``n``, ``v_max`` explains both;
- **ΔK-only** / **Δn-only** / **Δv_max-only** — exactly one knob takes a per-context value.

The winner (the most parsimonious model that *earns* its extra per-context parameter over
the shared null) names the mechanistic difference — ``threshold-diff`` / ``gain-diff`` /
``ceiling-diff`` — or NUDGE abstains (``no-difference`` / ``unresolved``). It reuses the
shipped covariance (linear-noise) machinery verbatim — the differentiable LNA
Gaussian-mixture forward model (mode means + Lyapunov covariances) from
:mod:`nudge.inference.lyapunov`, the BIC parsimony pattern from
:mod:`nudge.inference.model_select`, and per-context depth pinning via
:func:`~nudge.inference.lyapunov.calibrate_from_wt`.

**Confound guard (``NUDGE-LIM-016``, the load-bearing honesty point).** A sequencing-depth
/ batch difference *aligned with the context axis* mimics a mechanism difference — most
sharply a **ceiling** difference, because depth (global ``scale``) and ``v_max`` both
multiply the mode means (they are degenerate; ``calibrate_from_wt``). NUDGE pins depth/noise
**per context from each context's OWN control** (the analogue of per-sample library-size
normalization), so a depth difference captured by the controls is calibrated out. And when
the two contexts' pinned depths **differ beyond a ratio** — a depth/batch difference aligned
with the context axis — NUDGE **abstains** (``unresolved``): it cannot certify that an
apparent ceiling / no-clear difference is not a masked depth artifact. The **one exception**
is a *cleanly-resolved threshold or gain* difference, which **reshapes** the distribution
(orthogonal to a global scale) and so survives a depth difference and is still callable —
only the ceiling channel is confounded with depth. NUDGE also abstains when either context
is underpowered or its LNA is untrustworthy
(:func:`~nudge.inference.lyapunov.lna_reliable`). (An OFF-baseline diagnostic —
``off_shift_ratio`` — is reported for transparency but is *not* load-bearing: the OFF mode's
linear-noise spread is too large to separate a genuine ceiling from an on-samples batch
reliably, which is exactly why the guard turns on the stable per-context depth ratio and why
NUDGE requires the per-context control to come from the same library as its perturbed cells.)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from nudge.core.circuit import Circuit
from nudge.inference.fit import FreeParam
from nudge.inference.lyapunov import (
    OperatingPoint,
    _apply_free,
    _mode_cov,
    _mode_mean,
    _mvn_logpdf,
    _param_value,
    _stable_roots,
    calibrate_from_wt,
    lna_reliable,
)

__all__ = [
    "Context",
    "DifferentialFit",
    "DifferentialResult",
    "attribute_differential",
    "classify_differential",
    "fit_differential",
    "simulate_context_pair",
]

#: The three attributable knobs on the target edge, in a fixed order.
_KNOBS: tuple[str, ...] = ("n", "K", "vmax")
#: model name → the knob that takes a per-context value (``"shared"`` → none).
_MODELS: tuple[str, ...] = ("shared", "n", "K", "vmax")
#: which per-context knob a positive call names.
_CALL_OF: dict[str, str] = {
    "n": "gain-diff",
    "K": "threshold-diff",
    "vmax": "ceiling-diff",
}


@dataclass(frozen=True)
class Context:
    """One context's cells for the SAME perturbation, plus its OWN control.

    ``data`` is the perturbed / characteristic cells of this context in **activity
    space** (``(n_cells, n_species)``, identity readout — the counts→activity bridge is
    the caller's job, as for the Lyapunov path). ``control`` is this context's own WT /
    reference cells, used to pin the depth/noise nuisances **per context**
    (:func:`~nudge.inference.lyapunov.calibrate_from_wt`) — the load-bearing confound
    guard (a depth/batch difference between contexts otherwise mimics a mechanism
    difference; ``NUDGE-LIM-016``).
    """

    name: str
    data: np.ndarray
    control: np.ndarray


@dataclass(frozen=True)
class DifferentialFit:
    """The joint two-context fit: per-model BIC + the Δ estimates + per-context depth.

    ``bic`` / ``nll`` / ``n_params`` are keyed by model (``shared`` / ``n`` / ``K`` /
    ``vmax``); lower BIC is more parsimonious. The ``est_a`` / ``est_b`` knob estimates
    come from each model's fit (the shared model reports one value in both slots).
    ``ci_log2`` is a bootstrap CI on the winning knob's ``log2(value_b / value_a)``
    (``(nan, nan)`` when ``n_boot == 0`` or the winner is ``shared``). ``scale_a`` /
    ``scale_b`` are the per-context depths pinned from each control; ``depth_ratio`` and
    ``off_shift_ratio`` (the differential OFF-baseline move vs each context's control)
    drive the confound guard.
    """

    target_edge: int
    n_species: int
    k_modes: int
    n_cells_a: int
    n_cells_b: int
    scale_a: float
    obs_sd_a: float
    scale_b: float
    obs_sd_b: float
    lna_ok_a: bool
    lna_ok_b: bool
    lna_reason_a: str
    lna_reason_b: str
    bic: dict[str, float]
    nll: dict[str, float]
    n_params: dict[str, int]
    # per-model per-context knob estimates (natural units).
    est_a: dict[str, dict[str, float]]
    est_b: dict[str, dict[str, float]]
    selected: str  # the min-BIC model name
    best_diff: str  # the min-BIC model among the three Δ models
    depth_ratio: float  # max(scale)/min(scale) from the controls (≥ 1)
    off_shift_a: float  # context A's OFF baseline in data vs its OWN control (≈1 = fixed)
    off_shift_b: float  # context B's OFF baseline in data vs its OWN control
    off_shift_ratio: float  # off_shift_b / off_shift_a — the differential OFF-mode move
    ci_log2: tuple[float, float] = (float("nan"), float("nan"))
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def log2_ratio(self) -> float:
        """``log2(value_b / value_a)`` of the best Δ model's differing knob."""
        knob = self.best_diff
        va = self.est_a[knob][knob]
        vb = self.est_b[knob][knob]
        if va <= 0 or vb <= 0:
            return float("nan")
        return float(np.log2(vb / va))


@dataclass(frozen=True)
class DifferentialResult:
    """A differential fit + its conservative verdict and the human-readable reason."""

    fit: DifferentialFit
    call: str  # threshold-diff | gain-diff | ceiling-diff | no-difference | unresolved
    reason: str

    @property
    def is_reliable(self) -> bool:
        """Trustworthy enough to invert (``design()``'s integrity gate).

        A resolved *difference* (``threshold-diff`` / ``gain-diff`` / ``ceiling-diff``) is
        a reliable finding to prescribe from (e.g. a raised ceiling → more of the same
        drug; a rewired gain → a different class). The abstentions (``no-difference`` /
        ``unresolved``) are not. Satisfies the
        :class:`~nudge.design.invert.AttributionResult` protocol additively.
        """
        return self.call in {"threshold-diff", "gain-diff", "ceiling-diff"}


# --------------------------------------------------------------------------- #
# the joint two-context LNA fit (shared-vs-per-context parameter structure)
# --------------------------------------------------------------------------- #
def _edge_free(target_edge: int) -> list[FreeParam]:
    return [("edge", target_edge, name) for name in _KNOBS]


def _slots(model: str) -> tuple[dict[str, list[int]], int]:
    """Index layout of the kinetic vector for ``model`` (which knob is per-context)."""
    slots: dict[str, list[int]] = {}
    idx = 0
    for p in _KNOBS:
        if p == model:
            slots[p] = [idx, idx + 1]
            idx += 2
        else:
            slots[p] = [idx]
            idx += 1
    return slots, idx


#: Max |log| excursion of any per-context knob from its nominal — a smooth ``tanh`` bound
#: (×e^±_BOUND ≈ ×[1/3, 3]). It keeps a free knob from running off to the LNA
#: variance-collapse degeneracy (n→∞ shrinks the Lyapunov covariance → an unbounded
#: likelihood spike); the differential only needs the modest per-context RATIO.
_BOUND = 1.1

#: Refresh the concrete root seeds every this-many optimizer steps (the roots drift slowly
#: under the bounded kinetics, so re-solving them every step is wasted numpy work).
_ROOT_REFRESH = 6


def _fit_model(
    op_a: OperatingPoint,
    op_b: OperatingPoint,
    model: str,
    *,
    target_edge: int,
    k_modes: int,
    steps: int,
    learning_rate: float,
) -> tuple[dict[str, float], dict[str, float], float, float, int]:
    """Fit one nested model jointly across the two contexts by MAXIMIZING the LNA NLL.

    ``model`` ∈ ``{"shared", "n", "K", "vmax"}`` picks which target-edge knob takes a
    per-context value (the others are shared). Each kinetic value is
    ``nominal · exp(_BOUND · tanh(θ))`` — a **smooth bound** around the nominal that stops
    a free knob from diverging to the LNA covariance-collapse likelihood spike. Returns
    ``(est_a, est_b, nll_a, nll_b, n_free)``: ``est_x`` maps each knob to its fitted value
    in context ``x`` (natural units), ``nll_x`` is that context's mean per-cell NLL, and
    ``n_free`` counts the free *kinetic* values (3 shared, 4 for a Δ model).
    """
    ops = [op_a, op_b]
    slots, n_kin = _slots(model)
    free_edge = _edge_free(target_edge)
    bases = [op.circuit.base_params() for op in ops]
    ys = [jnp.asarray(np.asarray(op.data, dtype=np.float32)) for op in ops]
    eyes = [jnp.eye(op.circuit.n_species) for op in ops]

    nominal = np.array([_param_value(op_a.circuit, f) for f in free_edge], dtype=float)
    log_nom = np.empty(n_kin, dtype=float)  # log nominal per kinetic slot
    for p in _KNOBS:
        v = float(nominal[_KNOBS.index(p)])
        for j in slots[p]:
            log_nom[j] = np.log(v)
    log_nom_j = jnp.asarray(log_nom, dtype=jnp.float32)
    theta = jnp.zeros(n_kin + len(ops) * k_modes, dtype=jnp.float32)  # 0 = nominal

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(theta)

    def kin_vals(vec: Array) -> Array:
        return jnp.exp(log_nom_j + _BOUND * jnp.tanh(vec[:n_kin]))  # bounded natural units

    def vals_for(kin: Array, i: int) -> Array:
        out = []
        for p in _KNOBS:
            sl = slots[p]
            out.append(kin[sl[i] if len(sl) == 2 else sl[0]])
        return jnp.stack(out)  # order: n, K, vmax

    def _nll_terms(vec: Array, roots_all: Array) -> list[Array]:
        kin = kin_vals(vec)
        terms = []
        for i, op in enumerate(ops):
            params = _apply_free(bases[i], free_edge, vals_for(kin, i))
            w = jax.nn.log_softmax(vec[n_kin + i * k_modes : n_kin + (i + 1) * k_modes])
            comps = []
            for k in range(k_modes):
                mu = _mode_mean(op.circuit, params, roots_all[i, k])
                cov = _mode_cov(op.circuit, params, mu)
                cov_obs = op.scale**2 * cov + op.obs_sd**2 * eyes[i]
                comps.append(w[k] + _mvn_logpdf(ys[i], op.scale * mu, cov_obs))
            ll = jax.scipy.special.logsumexp(jnp.stack(comps), axis=0)
            terms.append(-jnp.mean(ll))
        return terms

    def nll(vec: Array, roots_all: Array) -> Array:
        roots_all = jax.lax.stop_gradient(roots_all)  # (2, k_modes, n_species)
        return sum(_nll_terms(vec, roots_all), start=jnp.asarray(0.0))

    @jax.jit
    def step(
        vec: Array, state: optax.OptState, roots_all: Array
    ) -> tuple[Array, optax.OptState, Array]:
        loss, grad = jax.value_and_grad(nll)(vec, roots_all)
        updates, state = optimizer.update(grad, state)
        return jnp.asarray(optax.apply_updates(vec, updates)), state, loss

    @jax.jit
    def score(vec: Array, roots_all: Array) -> Array:
        return jnp.stack(_nll_terms(vec, jax.lax.stop_gradient(roots_all)))

    def kin_np(vec: Array) -> np.ndarray:
        return np.exp(log_nom + _BOUND * np.tanh(np.asarray(vec[:n_kin])))

    def vals_np(vec: Array, i: int) -> list[float]:
        kin = kin_np(vec)
        return [float(kin[slots[p][i] if len(slots[p]) == 2 else slots[p][0]])
                for p in _KNOBS]

    def seeds(vec: Array) -> np.ndarray | None:
        kin = kin_np(vec)
        allr = []
        for i, op in enumerate(ops):
            vals = np.asarray(
                [kin[slots[p][i] if len(slots[p]) == 2 else slots[p][0]] for p in _KNOBS]
            )
            roots = _stable_roots(op.circuit, free_edge, vals)
            roots = sorted(roots, key=lambda r: tuple(float(x) for x in r))
            if len(roots) != k_modes:
                return None
            allr.append(np.stack(roots))
        return np.stack(allr)

    last = seeds(theta)
    if last is None:
        raise ValueError(
            f"every context must present {k_modes} stable modes at nominal kinetics"
        )
    # Step while both contexts stay bistable. If a per-context knob wanders past the fold
    # (monostable → frozen roots → a garbage/non-finite loss), FREEZE at the last valid θ
    # rather than keep integrating into an invalid region — a diverged NaN must not poison
    # the argmin (this Δ model then simply fails to beat its nested shared parent).
    prev_theta = theta
    for s in range(steps):
        # Refresh the concrete root seeds periodically, not every step: with the bounded
        # (tanh) kinetics the stable fixed points drift slowly, so re-solving them every
        # `_ROOT_REFRESH` steps is byte-safe and cuts the per-step numpy Newton cost ~Nx.
        if s % _ROOT_REFRESH == 0:
            cur = seeds(theta)
            if cur is None:
                theta = prev_theta  # revert the excursion; keep the last bistable θ
                break
            last = cur
            prev_theta = theta
        theta, opt_state, _loss = step(theta, opt_state, jnp.asarray(last))
    roots = seeds(theta)
    if roots is None:
        theta = prev_theta
        roots = seeds(theta)
        assert roots is not None

    est_a = {p: v for p, v in zip(_KNOBS, vals_np(theta, 0), strict=True)}
    est_b = {p: v for p, v in zip(_KNOBS, vals_np(theta, 1), strict=True)}
    per = np.asarray(score(theta, jnp.asarray(roots)))
    return est_a, est_b, float(per[0]), float(per[1]), n_kin


#: Low activity quantile probing the OFF mode for the confound guard.
_Q_OFF = 0.3


def _off_baseline_shift(data: np.ndarray, control: np.ndarray) -> float:
    """How far a context's data OFF baseline has moved vs its OWN control (≈ 1 = fixed).

    A **reported diagnostic, NOT load-bearing.** The ``_Q_OFF`` quantile of each cell's
    total activity (row-sum, clipped ≥ 0 since counts cannot be negative) probes the OFF
    mode; the ratio in the perturbed data to the control cancels the context's depth. In
    principle a genuine ceiling change leaves the OFF baseline fixed (ratio ≈ 1) while a
    batch on the samples moves it — but the OFF mode's linear-noise spread is too large for
    this to separate a ceiling from an on-samples batch reliably (the genuine-ceiling and
    batch shifts overlap across seeds), so the confound guard turns on the **stable
    per-context depth ratio** instead (:func:`classify_differential`, ``NUDGE-LIM-016``).
    Reported for transparency (``off_shift_ratio``).
    """
    d = np.clip(np.asarray(data, dtype=float).sum(axis=1), 0.0, None)
    c = np.clip(np.asarray(control, dtype=float).sum(axis=1), 0.0, None)
    cq = float(np.quantile(c, _Q_OFF))
    return float(np.quantile(d, _Q_OFF)) / cq if cq > 1e-9 else float("nan")


def fit_differential(
    context_a: Context,
    context_b: Context,
    circuit: Circuit,
    *,
    target_edge: int = 0,
    k_modes: int = 2,
    steps: int = 200,
    learning_rate: float = 0.05,
    seed: int = 0,
    n_boot: int = 0,
) -> DifferentialFit:
    """Jointly fit the two contexts and BIC-score which SINGLE knob differs.

    Pins depth/noise **per context** from each context's own control
    (:func:`~nudge.inference.lyapunov.calibrate_from_wt`), fits the four nested models
    (``shared`` / ``ΔK`` / ``Δn`` / ``Δv_max``) by maximizing the LNA Gaussian-mixture
    NLL, and returns their BICs (``k·ln N − 2·log L``, ``N`` = total cells) plus the Δ
    estimates and the per-context depth. ``circuit`` is the shared switch topology (both
    contexts share it at nominal kinetics; the difference lives in the fitted per-context
    knob). ``n_boot`` > 0 bootstraps a CI on the winning Δ (over cells). Does **not**
    itself decide — see :func:`classify_differential` — but records the depth / global-
    rescale diagnostics the confound guard needs.
    """
    scale_a, obs_a = calibrate_from_wt(
        np.asarray(context_a.control, dtype=float), circuit, k_modes=k_modes, seed=seed
    )
    scale_b, obs_b = calibrate_from_wt(
        np.asarray(context_b.control, dtype=float), circuit, k_modes=k_modes, seed=seed
    )
    op_a = OperatingPoint(
        data=np.asarray(context_a.data, dtype=float),
        circuit=circuit,
        scale=scale_a,
        obs_sd=obs_a,
    )
    op_b = OperatingPoint(
        data=np.asarray(context_b.data, dtype=float),
        circuit=circuit,
        scale=scale_b,
        obs_sd=obs_b,
    )

    ok_a, why_a = lna_reliable(circuit, scale_a)
    ok_b, why_b = lna_reliable(circuit, scale_b)

    n_a = int(op_a.data.shape[0])
    n_b = int(op_b.data.shape[0])
    log_n = float(np.log(max(n_a + n_b, 1)))

    bic: dict[str, float] = {}
    nll: dict[str, float] = {}
    n_params: dict[str, int] = {}
    est_a: dict[str, dict[str, float]] = {}
    est_b: dict[str, dict[str, float]] = {}
    for model in _MODELS:
        ea, eb, nll_a, nll_b, n_free = _fit_model(
            op_a,
            op_b,
            model,
            target_edge=target_edge,
            k_modes=k_modes,
            steps=steps,
            learning_rate=learning_rate,
        )
        nll_sum = n_a * nll_a + n_b * nll_b  # total (not mean) NLL over all cells
        est_a[model] = ea
        est_b[model] = eb
        n_params[model] = int(n_free)
        # A diff model whose fit diverged (a per-context knob driven past the fold →
        # monostable → frozen roots → non-finite NLL) is UNFITTABLE, not a winner: give
        # it +inf so it can never be selected (mirrors model_select's finite-ll guard).
        # This is safety-critical — a NaN slipping into the argmin can mis-select a knob.
        if np.isfinite(nll_sum):
            nll[model] = float(nll_sum)
            bic[model] = float(n_free * log_n + 2.0 * nll_sum)
        else:
            nll[model] = float("inf")
            bic[model] = float("inf")

    selected = min(bic, key=lambda m: bic[m])
    best_diff = min(("n", "K", "vmax"), key=lambda m: bic[m])

    depth_ratio = float(max(scale_a, scale_b) / max(min(scale_a, scale_b), 1e-12))
    off_shift_a = _off_baseline_shift(context_a.data, context_a.control)
    off_shift_b = _off_baseline_shift(context_b.data, context_b.control)
    if np.isfinite(off_shift_a) and np.isfinite(off_shift_b) and off_shift_a > 0:
        off_shift_ratio = float(off_shift_b / off_shift_a)
    else:
        off_shift_ratio = float("nan")

    ci_log2 = (float("nan"), float("nan"))
    if n_boot > 0 and best_diff == selected != "shared":
        ci_log2 = _bootstrap_ci(
            op_a, op_b, best_diff, target_edge, k_modes, steps, learning_rate,
            seed=seed, n_boot=n_boot,
        )

    return DifferentialFit(
        target_edge=target_edge,
        n_species=circuit.n_species,
        k_modes=k_modes,
        n_cells_a=n_a,
        n_cells_b=n_b,
        scale_a=scale_a,
        obs_sd_a=obs_a,
        scale_b=scale_b,
        obs_sd_b=obs_b,
        lna_ok_a=ok_a,
        lna_ok_b=ok_b,
        lna_reason_a=why_a,
        lna_reason_b=why_b,
        bic=bic,
        nll=nll,
        n_params=n_params,
        est_a=est_a,
        est_b=est_b,
        selected=selected,
        best_diff=best_diff,
        depth_ratio=depth_ratio,
        off_shift_a=off_shift_a,
        off_shift_b=off_shift_b,
        off_shift_ratio=off_shift_ratio,
        ci_log2=ci_log2,
    )


def _bootstrap_ci(
    op_a: OperatingPoint,
    op_b: OperatingPoint,
    model: str,
    target_edge: int,
    k_modes: int,
    steps: int,
    learning_rate: float,
    *,
    seed: int,
    n_boot: int,
) -> tuple[float, float]:
    """Bootstrap CI on ``log2(value_b / value_a)`` of the winning Δ (resample cells)."""
    rng = np.random.default_rng(seed)
    da = np.asarray(op_a.data, dtype=float)
    db = np.asarray(op_b.data, dtype=float)
    vals: list[float] = []
    for _ in range(n_boot):
        ia = rng.integers(0, da.shape[0], da.shape[0])
        ib = rng.integers(0, db.shape[0], db.shape[0])
        try:
            ea, eb, _na, _nb, _nf = _fit_model(
                replace(op_a, data=da[ia]),
                replace(op_b, data=db[ib]),
                model,
                target_edge=target_edge,
                k_modes=k_modes,
                steps=steps,
                learning_rate=learning_rate,
            )
        except Exception:
            continue
        va, vb = ea[model], eb[model]
        if va > 0 and vb > 0:
            vals.append(float(np.log2(vb / va)))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# --------------------------------------------------------------------------- #
# the fail-safe classifier
# --------------------------------------------------------------------------- #
def classify_differential(
    fit: DifferentialFit,
    *,
    bic_margin: float = 6.0,
    resolve_margin: float = 6.0,
    min_cells: int = 300,
    depth_ratio_max: float = 1.5,
) -> tuple[str, str]:
    """Turn a joint fit into a conservative verdict — the fail-safe classifier.

    Gates, most-conservative first:

    1. **unresolved — underpowered / untrustworthy LNA.** Either context is below
       ``min_cells`` or its LNA is unreliable
       (:func:`~nudge.inference.lyapunov.lna_reliable`): the depth/batch nuisance cannot
       be pinned cleanly enough to separate it from a mechanism difference.
    2. **unresolved — the depth confound (``NUDGE-LIM-016``, the load-bearing guard).**
       The two contexts' per-context depths (pinned from their controls) differ by more
       than ``depth_ratio_max`` — a sequencing-depth / batch difference **aligned with the
       context axis**. Because a global depth ``scale`` and the ceiling ``v_max`` are
       degenerate (both multiply the mode means), NUDGE cannot certify that an apparent
       ceiling / no-clear difference isn't a masked depth artifact, so it **abstains** —
       *unless* the winner is a **cleanly-resolved threshold or gain** difference, which
       *reshapes* the distribution and is orthogonal to a global scale (so it survives a
       depth difference and is still callable).
    3. **no-difference.** No Δ model beats the shared null by ``bic_margin`` — the extra
       per-context parameter is not earned; one ``K`` / ``n`` / ``v_max`` explains both.
    4. **unresolved — the two Δ models tie.** The winning Δ model does not beat the
       runner-up Δ model by ``resolve_margin`` (the measured gain⇄threshold confound): the
       difference is real but WHICH knob moved is unidentifiable — abstain, don't guess.
    5. **threshold-diff / gain-diff / ceiling-diff.** The winning Δ model earns its
       parameter over the shared null AND beats the other Δ models. Returns
       ``(call, reason)``.
    """
    # 1. underpowered / untrustworthy depth pinning.
    if not fit.lna_ok_a:
        return "unresolved", (
            f"context A's LNA is untrustworthy ({fit.lna_reason_a}) — the depth/batch "
            "nuisance cannot be pinned cleanly enough to separate it from a mechanism "
            "difference; NUDGE abstains (NUDGE-LIM-016)"
        )
    if not fit.lna_ok_b:
        return "unresolved", (
            f"context B's LNA is untrustworthy ({fit.lna_reason_b}) — NUDGE abstains "
            "(NUDGE-LIM-016)"
        )
    if fit.n_cells_a < min_cells or fit.n_cells_b < min_cells:
        return "unresolved", (
            f"a context is underpowered (n_a={fit.n_cells_a}, n_b={fit.n_cells_b} < "
            f"{min_cells}) — too few cells to separate a depth/batch difference from a "
            "mechanism one; NUDGE abstains (NUDGE-LIM-016)"
        )

    d_shared = fit.bic["shared"] - fit.bic[fit.best_diff]
    others = sorted(m for m in ("n", "K", "vmax") if m != fit.best_diff)
    runner = min(others, key=lambda m: fit.bic[m])
    d_runner = fit.bic[runner] - fit.bic[fit.best_diff]
    call = _CALL_OF[fit.best_diff]
    clean_diff = d_shared >= bic_margin and d_runner >= resolve_margin

    # 2. the depth confound (NUDGE-LIM-016) — depth aligned with the context axis is
    # degenerate with the ceiling. Abstain UNLESS the winner is a clean, depth-robust
    # threshold / gain difference (those reshape the distribution, orthogonal to a scale).
    if fit.depth_ratio > depth_ratio_max and not (clean_diff and fit.best_diff in ("n", "K")):
        return "unresolved", (
            f"the two contexts' per-context sequencing depths differ (ratio "
            f"{fit.depth_ratio:.2f} > {depth_ratio_max:g}) — a depth/batch difference "
            "ALIGNED WITH THE CONTEXT AXIS. Depth (global scale) and the ceiling v_max are "
            "degenerate, so NUDGE cannot certify that an apparent ceiling / no-clear "
            "difference is not a masked depth artifact; it abstains rather than risk a "
            "spurious ceiling-diff (NUDGE-LIM-016). Only a cleanly-resolved threshold / "
            "gain difference — which reshapes the distribution, orthogonal to a scale — "
            "survives a depth difference"
        )

    # 3. no Δ model earns its per-context parameter → no difference.
    if d_shared < bic_margin:
        return "no-difference", (
            f"no single-knob difference earns its parameter over the shared null "
            f"(ΔBIC={d_shared:.1f} < {bic_margin:g}) — one K / n / v_max explains both "
            "contexts; the SAME perturbation behaves the same in both (no mechanistic "
            "difference to attribute)"
        )

    # 4. which knob moved must be identifiable (the gain⇄threshold confound).
    if d_runner < resolve_margin:
        return "unresolved", (
            f"a per-context difference is real (ΔBIC vs shared={d_shared:.1f}) but WHICH "
            f"knob moved is unidentifiable: {_CALL_OF[fit.best_diff]} beats "
            f"{_CALL_OF[runner]} by only ΔBIC={d_runner:.1f} < {resolve_margin:g} (the "
            "measured gain⇄threshold confound) — NUDGE abstains rather than guess"
        )

    # 5. a resolved mechanistic difference.
    lo, hi = fit.ci_log2
    ci_txt = (
        f", log2 ratio 95% CI [{lo:+.2f}, {hi:+.2f}]"
        if np.isfinite(lo) and np.isfinite(hi)
        else ""
    )
    knob = fit.best_diff
    va, vb = fit.est_a[knob][knob], fit.est_b[knob][knob]
    guidance = {
        "vmax": "a raised/lowered ceiling → the SAME intervention at a different dose",
        "K": "a shifted threshold → a re-tuned dose / a different setpoint",
        "n": "a rewired gain → likely a DIFFERENT intervention class, not just more dose",
    }[knob]
    return call, (
        f"{call}: the target edge's {knob} differs between contexts "
        f"({knob}_A={va:.3g} vs {knob}_B={vb:.3g}, log2 ratio {fit.log2_ratio:+.2f}"
        f"{ci_txt}). This model earns its per-context parameter over the shared null "
        f"(ΔBIC={d_shared:.1f}) and beats the runner-up {_CALL_OF[runner]} "
        f"(ΔBIC={d_runner:.1f}). Actionable read: {guidance}"
    )


def attribute_differential(
    context_a: Context,
    context_b: Context,
    circuit: Circuit,
    *,
    target_edge: int = 0,
    k_modes: int = 2,
    steps: int = 200,
    learning_rate: float = 0.05,
    seed: int = 0,
    n_boot: int = 0,
    bic_margin: float = 6.0,
    resolve_margin: float = 6.0,
    min_cells: int = 300,
    depth_ratio_max: float = 1.5,
) -> DifferentialResult:
    """Fit + classify a two-context differential in one call — the CLI / MCP entry point.

    Isolates whether the SAME perturbation differs between two contexts in its switch's
    **threshold** (``K``), **gain** (``n``), or **ceiling** (``v_max``) — a decision
    linear differential expression cannot make — or abstains (``no-difference`` /
    ``unresolved``). The depth/batch confound is guarded by per-context depth pinning +
    the global-rescale ceiling guard (``NUDGE-LIM-016``).
    """
    fit = fit_differential(
        context_a,
        context_b,
        circuit,
        target_edge=target_edge,
        k_modes=k_modes,
        steps=steps,
        learning_rate=learning_rate,
        seed=seed,
        n_boot=n_boot,
    )
    call, reason = classify_differential(
        fit,
        bic_margin=bic_margin,
        resolve_margin=resolve_margin,
        min_cells=min_cells,
        depth_ratio_max=depth_ratio_max,
    )
    return DifferentialResult(fit=fit, call=call, reason=reason)


# --------------------------------------------------------------------------- #
# synthetic ground truth — a KNOWN single-knob difference between two contexts
# --------------------------------------------------------------------------- #
def simulate_context_pair(
    circuit: Circuit,
    *,
    mechanism: str,
    factor: float = 1.6,
    target_edge: int = 0,
    n_cells: int = 3000,
    scale_a: float = 20.0,
    scale_b: float = 20.0,
    obs_sd: float = 0.5,
    seed: int = 0,
) -> tuple[Context, Context]:
    """Build two contexts of the SAME perturbation differing in ONE known knob.

    ``mechanism`` ∈ ``{"threshold", "gain", "ceiling", "none"}`` sets which target-edge
    knob differs between the contexts: context B's ``K`` (``threshold``) / ``n``
    (``gain``) / ``vmax`` (``ceiling``) is scaled by ``factor`` relative to A; ``none``
    makes them identical (the no-difference ground truth). Cells are drawn from the
    shipped LNA Gaussian mixture (:func:`~nudge.inference.lyapunov.sample_lna_mixture`) at
    the given per-context depth; **each context's control is drawn at its OWN depth**
    (``scale_a`` / ``scale_b``). Set ``scale_b != scale_a`` with ``mechanism="none"`` to
    build the **confounded** ground truth — a sequencing-depth / batch difference **aligned
    with the context axis** (both the context's control and its perturbed cells) but NO
    real mechanism difference, which the depth-confound guard must catch (``unresolved``),
    never a spurious ``ceiling-diff`` (``NUDGE-LIM-016``).
    """
    from nudge.inference.lyapunov import sample_lna_mixture

    if mechanism not in ("threshold", "gain", "ceiling", "none"):
        raise ValueError(f"unknown mechanism {mechanism!r}")
    knob = {"threshold": "K", "gain": "n", "ceiling": "vmax", "none": None}[mechanism]

    edge = circuit.edges[target_edge]
    free = [("edge", target_edge, knob)] if knob is not None else []
    if knob is not None:
        base_val = float(getattr(edge, knob))
        vals_a: np.ndarray | None = np.array([base_val], dtype=float)
        vals_b: np.ndarray | None = np.array([base_val * factor], dtype=float)
    else:
        vals_a = vals_b = None

    key = jax.random.PRNGKey(seed)
    key, k1, k2, k3, k4 = jax.random.split(key, 5)
    ctrl_a = sample_lna_mixture(circuit, n_cells, k1, scale=scale_a, obs_sd=obs_sd)
    ctrl_b = sample_lna_mixture(circuit, n_cells, k2, scale=scale_b, obs_sd=obs_sd)
    data_a = sample_lna_mixture(
        circuit, n_cells, k3, free=free, vals=vals_a, scale=scale_a, obs_sd=obs_sd
    )
    data_b = sample_lna_mixture(
        circuit, n_cells, k4, free=free, vals=vals_b, scale=scale_b, obs_sd=obs_sd
    )
    return (
        Context(name="A", data=np.asarray(data_a), control=np.asarray(ctrl_a)),
        Context(name="B", data=np.asarray(data_b), control=np.asarray(ctrl_b)),
    )
