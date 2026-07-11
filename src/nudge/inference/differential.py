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
(:func:`~nudge.inference.lyapunov.lna_reliable`).

**Second confound channel: an ADDITIVE perturbed-condition offset (``NUDGE-LIM-016`` P1).**
The depth guard above keys on the two *controls*, so it is structurally blind to a constant
additive / ambient offset applied to ONE context's **perturbed** cells only (its control
left clean) — the per-context ``depth_ratio`` stays ≈ 1. Such an offset nonetheless shifts
that context's perturbed modes and *compresses* their separation, which the joint BIC
misreads as reduced cooperativity — a confident spurious ``gain-diff`` where the truth is
no-difference. It leaves a fingerprint the depth ratio cannot: it TRANSLATES the perturbed
OFF baseline **up** relative to that context's own control (``off_shift`` ≫ 1), whereas a
genuine ``K`` / ``n`` / ``v_max`` difference leaves the OFF mode anchored near basal
(``off_shift`` ≈ 1). So before any positive call NUDGE **abstains** when either context's
perturbed OFF baseline is inflated above its own control beyond ``off_shift_max`` (a MEASURED
separator — every confident-wrong offset had ``off_shift`` ≥ 2.99, the strongest genuine
difference only ≤ 1.96; ``FINDINGS`` §P1). This is why the OFF-baseline shift, previously
reported only for transparency, is now **one-sided load-bearing**: the *inflation* direction
cleanly separates an additive offset from a mechanism difference. **Residual bound (honest):**
the guard is one-sided — a *deflating* perturbed-only offset (dropout-like, ``off_shift`` < 1)
aliases with a genuine knob *reduction* and is NOT separable, so it remains unguarded and
documented (``NUDGE-LIM-016``); and NUDGE still requires each context's control to come from
the same library as its perturbed cells.

**Third confound channel: a MULTIPLICATIVE perturbed-condition scale (``NUDGE-LIM-016`` P4).**
A constant *multiplicative* factor ``c`` on ONE context's **perturbed** cells only (its
control clean) is the sharpest confound of all: it aliases a genuine **ceiling** (``v_max``)
difference 1:1 (both multiply the ON mode), and it slips past BOTH earlier guards — the
control-keyed ``depth_ratio`` stays ≈ 1 (gate 2 blind), and a factor scales the near-zero OFF
*baseline* to near-zero so the additive ``off_shift`` stays ≈ 1 (gate 4b blind). It leaves ONE
fingerprint: it multiplies the OFF-cluster **spread** by ``c`` too (``off_scale`` ≈ ``c``),
whereas a genuine ``v_max`` difference leaves the OFF mode's spread anchored at basal
(``off_scale`` ≈ 1). So before a ``ceiling-diff`` call NUDGE **abstains** when the perturbed
OFF-cluster scale departs from its control beyond a MEASURED band (gate 4c). **INFLATION is
CLOSED** against a *uniform* or *smoothly content-dependent* inflating scale — a clean
measured gap (genuine ceiling ×1.4–×4 left ``off_scale`` ≤ 1.18; every such inflating confound
``c`` ≥ 1.5 drove it ≥ 1.43; ``FINDINGS`` §P4). (A pathological scale confined *strictly* to
above-median cells leaves the OFF cluster untouched and so evades this fingerprint — but it is
then observationally *identical* to a genuine ceiling change, i.e. not a distinguishable
confident-wrong, and has no plausible physical generator; red-team `runs/000000013`.)
**DEFLATION is BOUNDED, not
closed:** a genuine ceiling *reduction* collapses the switch toward monostable and shrinks the
OFF cluster into the same band as a deflating scale (both ``off_scale`` ≈ 0.5–0.75) — they are
INDISTINGUISHABLE, so the lower guard abstains on both, correctly killing the deflating confound
at the cost of no longer resolving a strong genuine ceiling *reduction* (the honest price; a
per-context multiplicative scale without an independent depth anchor is fundamentally degenerate
with a ceiling change). The guard is **ceiling-scoped** (only a ``v_max`` winner) since a global
scale is degenerate with ``v_max`` specifically — a genuine gain/threshold difference reshapes
the distribution and is untouched.
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
    # OFF-cluster SCALE (spread) of each context's perturbed data vs its own control (≈1 =
    # fixed): a per-context MULTIPLICATIVE measurement scale c multiplies this by c while a
    # genuine v_max difference leaves it ≈ 1. Drives the ceiling-channel gate 4c (P4).
    off_scale_a: float = 1.0
    off_scale_b: float = 1.0
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

#: Max inflation of EITHER context's perturbed OFF baseline above its OWN control
#: (``max(off_shift_a, off_shift_b)``) before the additive-perturbed-offset guard abstains
#: (``classify_differential`` gate 4b, ``NUDGE-LIM-016`` P1). A constant additive/ambient
#: offset on ONE context's perturbed cells (its control clean) TRANSLATES that context's OFF
#: baseline up relative to its control while a GENUINE knob difference leaves the OFF mode
#: anchored near basal. MEASURED separation (FINDINGS §P1, the default red-team regime):
#: every confident-wrong additive-offset call had ``off_shift ≥ 2.99``; the strongest
#: genuine gain/ceiling/threshold difference (up to ×3–4) inflated the OFF baseline to only
#: ``≤ 1.96``. This threshold sits in that gap with margin on both sides — it is a measured
#: separator, NOT an arbitrary cut. One-sided (inflation only): a genuine *reduction*
#: deflates the OFF baseline below 1 and never trips it (which is why a symmetric guard is
#: invalid — it would over-abstain on a genuine gain/ceiling reduction; see FINDINGS §P1).
_OFF_SHIFT_INFLATION_MAX = 2.5


def _off_baseline_shift(data: np.ndarray, control: np.ndarray) -> float:
    """How far a context's data OFF baseline has moved vs its OWN control (≈ 1 = fixed).

    **One-sided load-bearing (``NUDGE-LIM-016`` P1).** The ``_Q_OFF`` quantile of each cell's
    total activity (row-sum, clipped ≥ 0 since counts cannot be negative) probes the OFF
    mode; the ratio in the perturbed data to the control cancels the context's depth. A
    genuine ``K`` / ``n`` / ``v_max`` difference leaves the OFF baseline anchored near basal
    (ratio ≈ 1; measured ≤ 1.96 even for a ×3–4 difference), while a constant additive /
    ambient offset on the *perturbed* cells TRANSLATES it up (ratio ≫ 1; ≥ 2.99 for every
    confident-wrong offset, ``FINDINGS`` §P1). The **inflation** direction is therefore a
    clean, measured separator and drives the gate-4b abstention
    (:func:`classify_differential`, ``off_shift_max``). The **deflation** direction is NOT:
    a genuine reduction and a dropout-like offset both push the ratio below 1 and overlap
    (which is why the guard is one-sided and a deflating perturbed-only offset stays an
    unguarded residual, ``NUDGE-LIM-016``). Historically this diagnostic was reported for
    transparency only; P1's measurement made its inflation side load-bearing.
    """
    d = np.clip(np.asarray(data, dtype=float).sum(axis=1), 0.0, None)
    c = np.clip(np.asarray(control, dtype=float).sum(axis=1), 0.0, None)
    cq = float(np.quantile(c, _Q_OFF))
    return float(np.quantile(d, _Q_OFF)) / cq if cq > 1e-9 else float("nan")


#: OFF-cluster SCALE band for the ceiling channel (``NUDGE-LIM-016`` P4). A per-context
#: MULTIPLICATIVE measurement scale ``c`` on ONE context's PERTURBED cells (its control clean)
#: is degenerate with a genuine ceiling (``v_max``) difference — both multiply the ON mode — and
#: slips past gate 2 (its control-keyed ``depth_ratio`` stays ≈ 1) AND gate 4b (a factor scales
#: the near-zero OFF *baseline* to near-zero, so ``off_shift`` ≈ 1). Its ONE fingerprint: it
#: multiplies the OFF-cluster **spread** by ``c`` too, while a genuine ``v_max`` difference leaves
#: the OFF mode's spread anchored at basal. MEASURED separation (FINDINGS §P4, both the red-team
#: and test regimes, 3 seeds): a genuine ceiling INFLATION (×1.4–×4) left the perturbed/control
#: OFF-cluster scale ratio ≤ 1.18; every inflating multiplicative confound (``c`` ≥ 1.5) drove it
#: ≥ 1.43 — a gap no genuine ceiling crosses. 1.30 is that gap's midpoint: a measured separator,
#: NOT an arbitrary cut. INFLATION is therefore CLOSED against a uniform or smoothly
#: content-dependent inflating scale (a scale confined strictly to above-median cells evades
#: this OFF-cluster fingerprint, but is then indistinguishable from a genuine ceiling change;
#: red-team runs/000000013).
_OFF_SCALE_INFLATION_MAX = 1.30
#: DEFLATION is BOUNDED, not closed. A genuine ceiling *reduction* collapses the switch toward
#: monostable and shrinks the OFF cluster (ratio ≈ 0.64) into the same band as a DEFLATING scale
#: (``c`` = 0.5–0.7 → ratio 0.48–0.75) — the two are INDISTINGUISHABLE without an independent depth
#: anchor. So the lower guard is a *catch threshold*, not a clean separator: 0.80 abstains on every
#: demonstrated deflating confound (``c`` ≤ 0.7 → ratio ≤ 0.75, with margin) at the honest cost of
#: also abstaining on a strong genuine ceiling reduction (documented, ``NUDGE-LIM-016`` P4).
_OFF_SCALE_DEFLATION_MIN = 0.80


def _off_mode_scale_ratio(data: np.ndarray, control: np.ndarray) -> float:
    """OFF-cluster SPREAD in a context's perturbed data vs its OWN control (≈ 1 = fixed).

    **Load-bearing for the ceiling channel (``NUDGE-LIM-016`` P4).** A robust scale (median
    absolute deviation) of the LOW-activity cells — those whose total activity (row-sum, clipped
    ≥ 0) falls below the population median, i.e. the OFF cluster — measured in the perturbed data
    and in the context's OWN control; the ratio cancels the context's depth. A per-context
    **multiplicative** measurement scale ``c`` on the perturbed cells multiplies this OFF-cluster
    spread by ``c`` (ratio ≈ ``c``), the fingerprint that neither the control-keyed depth ratio
    (gate 2) nor the additive OFF-*baseline* shift (gate 4b, which a multiplicative factor leaves
    ≈ 1 because it scales near-zero to near-zero) can see. A genuine ``v_max`` (ceiling)
    difference leaves the OFF mode's spread anchored at basal (ratio ≈ 1 for inflation; measured
    ≤ 1.18 even for a ×4 difference — FINDINGS §P4). The **inflation** side is a clean measured
    separator; the **deflation** side is NOT (a genuine ceiling reduction shrinks the OFF cluster
    into the same band as a deflating scale), so gate 4c is CLOSED above and BOUNDED below.
    """
    def _low_scale(x: np.ndarray) -> float:
        # NOT clipped at 0: in activity space the near-zero OFF baseline has obs-noise
        # excursions below 0, and a multiplicative measurement scale dilates those too — the
        # very signal this estimator keys on. Clipping would collapse the OFF cluster to a
        # zero-spike (MAD → 0) and destroy the statistic (measured; the scale must be the raw
        # spread of the low-activity cells).
        s = np.asarray(x, dtype=float).sum(axis=1)
        if s.size == 0:
            return float("nan")
        low = s[s <= np.median(s)]
        if low.size == 0:
            return float("nan")
        return float(np.median(np.abs(low - np.median(low))))

    cs = _low_scale(control)
    ds = _low_scale(data)
    if np.isfinite(cs) and cs > 1e-9 and np.isfinite(ds):
        return ds / cs
    return float("nan")


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
    off_scale_a = _off_mode_scale_ratio(context_a.data, context_a.control)
    off_scale_b = _off_mode_scale_ratio(context_b.data, context_b.control)

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
        off_scale_a=off_scale_a,
        off_scale_b=off_scale_b,
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
    off_shift_max: float = _OFF_SHIFT_INFLATION_MAX,
    off_scale_inflation_max: float = _OFF_SCALE_INFLATION_MAX,
    off_scale_deflation_min: float = _OFF_SCALE_DEFLATION_MIN,
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
    4b. **unresolved — the additive perturbed-offset confound (``NUDGE-LIM-016``, P1).**
       Either context's perturbed OFF baseline is inflated above its OWN control beyond
       ``off_shift_max`` — the fingerprint of a constant additive / ambient offset on that
       context's *perturbed* cells only (its control clean), which the control-keyed
       ``depth_ratio`` (gate 2) is structurally blind to but which the joint BIC misreads as
       reduced cooperativity (a spurious ``gain-diff``). One-sided (inflation only): a
       genuine knob *reduction* deflates the OFF baseline and does not trip it.
    4c. **unresolved — the MULTIPLICATIVE perturbed-scale confound (``NUDGE-LIM-016``, P4).**
       Only on a **ceiling** (``v_max``) winner: a per-context multiplicative measurement scale
       ``c`` on one context's perturbed cells aliases a genuine ceiling difference 1:1 and slips
       past gates 2 and 4b, but multiplies that context's OFF-cluster **spread** by ``c`` too.
       When the perturbed OFF-cluster scale departs from its own control beyond the measured
       band ``[off_scale_deflation_min, off_scale_inflation_max]`` NUDGE abstains. INFLATION is
       CLOSED (a clean measured gap); DEFLATION is BOUNDED — a genuine ceiling reduction and a
       deflating scale both shrink the OFF cluster and are indistinguishable, so the lower guard
       abstains on both (sacrificing a strong genuine ceiling reduction, the honest price).
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

    # 4b. the additive perturbed-condition offset confound (NUDGE-LIM-016, P1). The
    # control-keyed depth guard (gate 2) is BLIND to a constant additive / ambient offset
    # applied to ONE context's PERTURBED cells only (its control clean): the per-context
    # depth_ratio is pinned from the two controls, so it stays ≈ 1 and gate 2 never
    # engages. But such an offset shifts that context's perturbed modes and COMPRESSES
    # their separation, which the joint LNA-BIC reads as reduced cooperativity — a confident
    # spurious gain-diff (verified, FINDINGS §P1). It leaves a MEASURABLE fingerprint the
    # depth_ratio cannot: it TRANSLATES the perturbed OFF baseline UP relative to that
    # context's own control (off_shift ≫ 1), whereas a genuine K/n/v_max difference leaves
    # the OFF mode anchored near basal (off_shift ≈ 1; measured ≤ 1.96 even for a ×3–4
    # difference, vs ≥ 2.99 for every confident-wrong offset — FINDINGS §P1). So before
    # emitting ANY positive *-diff, if EITHER context's perturbed OFF baseline is inflated
    # above its own control beyond _OFF_SHIFT_INFLATION_MAX — an additive/ambient artifact
    # on the perturbed condition, degenerate with a mechanism difference — NUDGE ABSTAINS.
    # One-sided by construction: a genuine reduction DEFLATES the OFF baseline (off_shift
    # < 1) and never trips this, so genuine gain/ceiling/threshold reductions still resolve
    # (a symmetric guard would over-abstain on them — FINDINGS §P1). Residual bound: a
    # DEFLATING perturbed-only offset (dropout-like) aliases with a genuine reduction and is
    # NOT caught here (NUDGE-LIM-016).
    finite_off = [v for v in (fit.off_shift_a, fit.off_shift_b) if np.isfinite(v)]
    off_infl = max(finite_off) if finite_off else float("nan")
    if np.isfinite(off_infl) and off_infl > off_shift_max:
        which = "A" if fit.off_shift_a == off_infl else "B"  # the inflated context
        return "unresolved", (
            f"context {which}'s perturbed OFF baseline is inflated {off_infl:.2f}× above "
            f"its OWN control (> {off_shift_max:g}) — the fingerprint of a constant "
            "additive / ambient offset on that context's PERTURBED cells (a batch / "
            "background artifact on the perturbed condition only). Its control-derived "
            f"depth ratio ({fit.depth_ratio:.2f}) stays ≈ 1, so the depth guard is blind to "
            "it, but such an offset shifts and compresses the perturbed modes — degenerate "
            f"with a reduced-cooperativity ({_CALL_OF[fit.best_diff]}) signal. NUDGE cannot "
            "certify the apparent difference is mechanistic rather than a masked "
            "perturbed-condition artifact, so it abstains (NUDGE-LIM-016). A genuine knob "
            "difference leaves the OFF mode anchored near basal (off_shift ≈ 1)"
        )

    # 4c. the MULTIPLICATIVE perturbed-condition scale confound (NUDGE-LIM-016, P4). Scoped to
    # the CEILING channel: a constant multiplicative factor c on ONE context's PERTURBED cells
    # (its control clean) is exactly degenerate with a genuine v_max difference (both multiply
    # the ON mode), and it slips past BOTH earlier guards — the control-keyed depth_ratio stays
    # ≈ 1 (gate 2 blind), and a factor scales the near-zero OFF baseline to near-zero so
    # off_shift ≈ 1 (gate 4b blind, which keys on TRANSLATION not scale). Its ONE fingerprint:
    # it multiplies that context's OFF-cluster SPREAD by c too, while a genuine ceiling
    # difference leaves the OFF mode's spread anchored at basal. So before a ceiling-diff call,
    # if EITHER context's perturbed OFF-cluster scale departs from its own control beyond the
    # MEASURED band [off_scale_deflation_min, off_scale_inflation_max], NUDGE ABSTAINS. Only the
    # ceiling channel is gated (a global scale is degenerate with v_max specifically; a genuine
    # gain/threshold difference reshapes the distribution and is untouched — no over-abstention
    # there). INFLATION is CLOSED (genuine ceiling ×1.4–×4 ≤ 1.18, every inflating confound
    # ≥ 1.43 — FINDINGS §P4). DEFLATION is BOUNDED: a genuine ceiling reduction shrinks the OFF
    # cluster into the same band as a deflating scale (indistinguishable), so the lower guard
    # abstains on both — killing the deflating confound at the cost of a strong genuine ceiling
    # reduction (the honest residual, NUDGE-LIM-016 P4).
    if fit.best_diff == "vmax":
        cand = [
            (v, ctx)
            for v, ctx in ((fit.off_scale_a, "A"), (fit.off_scale_b, "B"))
            if np.isfinite(v) and v > 0
        ]
        if cand:
            # the context whose OFF-cluster scale deviates most from 1 (up OR down).
            off_scale, which = max(cand, key=lambda t: abs(np.log(t[0])))
            if off_scale > off_scale_inflation_max or off_scale < off_scale_deflation_min:
                direction = "inflated" if off_scale > 1.0 else "deflated"
                extra = (
                    "this is the CLOSED (cleanly-separable) side: a genuine ceiling difference "
                    "leaves the OFF-cluster spread ≈ 1 (measured ≤ 1.18 even ×4)"
                    if direction == "inflated"
                    else "this is the BOUNDED side: a genuine ceiling REDUCTION also shrinks the "
                    "OFF cluster and is indistinguishable from a deflating scale, so NUDGE "
                    "abstains on both rather than risk a spurious ceiling-diff"
                )
                return "unresolved", (
                    f"context {which}'s perturbed OFF-cluster scale is {direction} "
                    f"{off_scale:.2f}× vs its OWN control (outside the measured band "
                    f"[{off_scale_deflation_min:g}, {off_scale_inflation_max:g}]) — the "
                    "fingerprint of a per-context MULTIPLICATIVE measurement scale on that "
                    "context's PERTURBED cells (a batch / library scale on the perturbed "
                    f"condition only). Its control-derived depth ratio ({fit.depth_ratio:.2f}) "
                    "stays ≈ 1 and its additive OFF-baseline shift is ≈ 1, so gates 2 and 4b are "
                    "blind to it, but a multiplicative factor is EXACTLY degenerate with a "
                    "ceiling (v_max) difference — both multiply the ON mode. NUDGE cannot certify "
                    "the apparent ceiling difference is mechanistic rather than a masked "
                    f"perturbed-condition scale, so it abstains (NUDGE-LIM-016 P4). ({extra}.)"
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
    off_shift_max: float = _OFF_SHIFT_INFLATION_MAX,
    off_scale_inflation_max: float = _OFF_SCALE_INFLATION_MAX,
    off_scale_deflation_min: float = _OFF_SCALE_DEFLATION_MIN,
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
        off_shift_max=off_shift_max,
        off_scale_inflation_max=off_scale_inflation_max,
        off_scale_deflation_min=off_scale_deflation_min,
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
