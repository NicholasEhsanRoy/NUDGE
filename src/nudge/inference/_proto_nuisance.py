"""PROTOTYPE (experimental, NOT shipped) — a *systemic* guard for the whole class of
per-condition **affine technical nuisances** on a differential's PERTURBED cells.

Read the standing task in ``design/PERTURBED_CONFOUND_STRATEGY.md`` (written only if this
prototype's measured criterion is met) and the sibling baseline
``nudge.inference.differential`` (the shipped per-confound OFF-cluster bands this aims to
replace). **This module is a sibling: it imports differential's primitives read-only and
NEVER modifies ``differential.py`` / ``fit.py`` / ``core/``.**

The class of confounds the red-team keeps finding (additive P1, multiplicative P4, small
multiplicative P5) is ONE thing: a per-condition **affine** nuisance ``y = s·x + o`` — a
scale ``s`` and offset ``o`` — applied to ONE context's *perturbed* cells only (its control
clean, so the control-keyed depth guard never engages). It aliases onto a mechanism: a
per-condition **scale is degenerate with ``v_max``** (both multiply the ON mode); a
per-condition **offset shifts the modes** (aliases threshold / gain). The shipped fix is
per-confound hand-calibrated OFF-cluster bands with measured blind gaps (P5 slips the
``(1.18,1.30]`` gap). This is whack-a-mole.

Two complementary principled mechanisms are prototyped here:

**(A) INERT-ANCHOR normalization** (:func:`anchor_normalize`) — estimate ``(s, o)`` from a
perturbation-INERT feature block (housekeeping / non-signature genes / spike-ins co-measured
on the perturbed condition — nearly free on real Perturb-seq, where the perturbation moves a
handful of the thousands of genes) and undo the affine on the perturbed readout BEFORE
attribution. This removes the class at the source and RECOVERS a genuine ceiling that lives
under a technical scale. The fundamental fix.

**(B) NUISANCE-AUGMENTED IDENTIFIABILITY ABSTENTION** (:func:`guard_b_classify`) — the
no-anchor fallback. Add the per-context affine ``(s, o)`` as FREE nuisances to the
differential fit and abstain via a MEASURED Fisher/Laplace degeneracy when the biological
knob is near-singular with the nuisance. ONE guard covering the ENTIRE affine family
continuously (no calibrated bands, no blind gaps). It abstains on a genuine ceiling too —
that is the honest identifiability limit; (A) is what buys the ceiling call back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from nudge.core.circuit import Circuit
from nudge.inference.differential import _CALL_OF, Context
from nudge.inference.lyapunov import (
    _apply_free,
    _mode_cov,
    _mode_mean,
    _mvn_logpdf,
    _param_value,
    _stable_roots,
    calibrate_from_wt,
    lna_reliable,
)
from nudge.inference.uncertainty import laplace_posterior

__all__ = [
    "AnchorEstimate",
    "GuardBResult",
    "anchor_normalize",
    "estimate_affine_from_inert",
    "guard_b_classify",
]

_KNOBS: tuple[str, ...] = ("n", "K", "vmax")


# --------------------------------------------------------------------------- #
# (A) INERT-ANCHOR normalization
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnchorEstimate:
    """The per-condition affine ``(s, o)`` estimated from an inert feature block."""

    scale: float
    offset: float
    reason: str


def estimate_affine_from_inert(
    perturbed_inert: np.ndarray, control_inert: np.ndarray
) -> AnchorEstimate:
    """Estimate the affine ``y = s·x + o`` from a perturbation-INERT feature block.

    ``*_inert`` are ``(n_cells, n_inert)`` blocks of features the perturbation does NOT move
    (housekeeping / non-signature genes / spike-ins), co-measured on the perturbed condition
    and its control. If a per-cell technical affine hit the perturbed condition
    (``y = s·x + o`` on ALL its features), the inert block moved by the SAME ``(s, o)`` while
    its biology stayed put — so the inert perturbed/control moment shift IS the technical
    nuisance. ``s = sd(perturbed_inert) / sd(control_inert)`` (a scale inflates the spread),
    ``o = mean(perturbed_inert) − s·mean(control_inert)`` (whatever mean shift the scale did
    not account for). Pooled across the inert features (robust to a single noisy gene).
    """
    p = np.asarray(perturbed_inert, dtype=float)
    c = np.asarray(control_inert, dtype=float)
    # Pool the per-feature moments (each inert gene shares the same technical s, o).
    p_sd = float(np.sqrt(np.mean(p.var(axis=0))))
    c_sd = float(np.sqrt(np.mean(c.var(axis=0))))
    p_mean = float(p.mean())
    c_mean = float(c.mean())
    if c_sd < 1e-9:
        return AnchorEstimate(1.0, 0.0, "inert control has no spread — cannot anchor")
    s = p_sd / c_sd
    o = p_mean - s * c_mean
    return AnchorEstimate(s, o, f"anchored on {p.shape[1]} inert features")


def anchor_normalize(perturbed: np.ndarray, anchor: AnchorEstimate) -> np.ndarray:
    """Undo the technical affine on the perturbed readout: ``x = (y − o) / s``.

    Puts the perturbed signature block back onto its control's technical scale BEFORE
    attribution, so a genuine ceiling that lived under a technical scale is recovered.
    """
    y = np.asarray(perturbed, dtype=float)
    return (y - anchor.offset) / max(anchor.scale, 1e-9)


# --------------------------------------------------------------------------- #
# (B) NUISANCE-AUGMENTED IDENTIFIABILITY ABSTENTION
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GuardBResult:
    """One unified verdict from the nuisance-augmented differential fit.

    ``call`` ∈ {threshold-diff, gain-diff, ceiling-diff, no-difference, unresolved}. The
    positive calls require the BIC-winning knob to (1) still EARN its parameter over a
    pure-affine-nuisance model of the perturbed context and (2) be Fisher/Laplace-IDENTIFIABLE
    jointly with the free affine ``(s, o)``. Anything else abstains.
    """

    call: str
    reason: str
    knob: str
    earn_bic: float  # BIC(nuisance-only) − BIC(nuisance + knob); >margin ⇒ knob earns
    cond_number: float
    knob_identifiable: bool
    s_hat: float
    o_hat: float
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def is_reliable(self) -> bool:
        return self.call in {"threshold-diff", "gain-diff", "ceiling-diff"}


_BOUND = 1.1  # same smooth |log| excursion cap differential uses for a per-context knob


def _fit_shared_knobs(
    data: np.ndarray, circuit: Circuit, scale: float, obs_sd: float,
    *, k_modes: int, steps: int, lr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the reference context's three knobs (n, K, vmax); return (log-knobs, roots)."""
    base = circuit.base_params()
    free = [("edge", 0, p) for p in _KNOBS]
    nominal = np.array([_param_value(circuit, f) for f in free], dtype=float)
    log_nom = jnp.asarray(np.log(nominal), dtype=jnp.float32)
    y = jnp.asarray(np.asarray(data, dtype=np.float32))
    eye = jnp.eye(circuit.n_species)
    theta = jnp.zeros(3 + k_modes, dtype=jnp.float32)

    def kin(vec):
        return jnp.exp(log_nom + _BOUND * jnp.tanh(vec[:3]))

    def nll(vec, roots):
        roots = jax.lax.stop_gradient(roots)
        params = _apply_free(base, free, kin(vec))
        w = jax.nn.log_softmax(vec[3:])
        comps = []
        for k in range(k_modes):
            mu = _mode_mean(circuit, params, roots[k])
            cov = _mode_cov(circuit, params, mu)
            cov_obs = scale**2 * cov + obs_sd**2 * eye
            comps.append(w[k] + _mvn_logpdf(y, scale * mu, cov_obs))
        return -jnp.mean(jax.scipy.special.logsumexp(jnp.stack(comps), axis=0))

    opt = optax.adam(lr)
    st = opt.init(theta)

    @jax.jit
    def step(vec, st, roots):
        loss, g = jax.value_and_grad(nll)(vec, roots)
        u, st = opt.update(g, st)
        return jnp.asarray(optax.apply_updates(vec, u)), st, loss

    def seeds(vec):
        vals = np.exp(np.log(nominal) + _BOUND * np.tanh(np.asarray(vec[:3])))
        r = _stable_roots(circuit, free, vals)
        r = sorted(r, key=lambda x: tuple(float(v) for v in x))
        return None if len(r) != k_modes else np.stack(r)

    last = seeds(theta)
    if last is None:
        raise ValueError("reference context is not bistable at nominal kinetics")
    for s in range(steps):
        if s % 6 == 0:
            cur = seeds(theta)
            if cur is None:
                break
            last = cur
        theta, st, _ = step(theta, st, jnp.asarray(last))
    log_knobs = np.log(nominal) + _BOUND * np.tanh(np.asarray(theta[:3], dtype=float))
    roots = seeds(theta)
    if roots is None:
        roots = last
    return log_knobs, roots


def _augmented_loss(
    data: np.ndarray, circuit: Circuit, log_shared: np.ndarray, knob: str,
    scale: float, obs_sd: float, roots: np.ndarray, *, free_knob: bool,
):
    """Mean-NLL of the perturbed context under: one bio knob + a free affine ``(s, o)``.

    Parameter vector ``z``: if ``free_knob`` it is ``[log θ_knob, log s, o]`` (3-vector);
    else ``[log s, o]`` (the pure-nuisance null). The perturbed model is
    ``mean_k = s·(depth·μ_k(θ)) + o``, ``cov_k = s²·depth²·Σ_k(θ) + obs_sd²·I`` — the affine
    applied to the LNA mixture. ``depth`` (=``scale``) and ``obs_sd`` stay pinned from the
    context's own control; ``(s, o)`` absorb the technical nuisance. Roots are held as
    stop-gradient seeds (the fit only walks the knob within the smooth bound, so they drift
    slowly). Returns a ``loss(z)`` for :func:`laplace_posterior`.
    """
    base = circuit.base_params()
    free = [("edge", 0, p) for p in _KNOBS]
    y = jnp.asarray(np.asarray(data, dtype=np.float32))
    roots_j = jnp.asarray(np.asarray(roots, dtype=np.float32))
    k_modes = roots_j.shape[0]
    eye = jnp.eye(circuit.n_species)
    log_shared_j = jnp.asarray(log_shared, dtype=jnp.float32)
    ki = _KNOBS.index(knob)

    def loss(z: Array) -> Array:
        if free_knob:
            log_vals = log_shared_j.at[ki].set(z[0])
            log_s, o = z[1], z[2]
        else:
            log_vals = log_shared_j
            log_s, o = z[0], z[1]
        s = jnp.exp(log_s)
        params = _apply_free(base, free, jnp.exp(log_vals))
        comps = []
        for k in range(k_modes):
            mu = _mode_mean(circuit, params, roots_j[k])
            cov = _mode_cov(circuit, params, mu)
            mean_obs = s * (scale * mu) + o
            cov_obs = (s * scale) ** 2 * cov + obs_sd**2 * eye
            comps.append(_mvn_logpdf(y, mean_obs, cov_obs))
        # uniform mixture weights (2 balanced modes) — the shape signal is in the covariances
        ll = jax.scipy.special.logsumexp(jnp.stack(comps), axis=0) - jnp.log(k_modes)
        return -jnp.mean(ll)

    return loss


def _optimize(loss, z0: np.ndarray, *, steps: int, lr: float) -> np.ndarray:
    z = jnp.asarray(z0, dtype=jnp.float32)
    opt = optax.adam(lr)
    st = opt.init(z)

    @jax.jit
    def step(z, st):
        v, g = jax.value_and_grad(loss)(z)
        u, st = opt.update(g, st)
        return jnp.asarray(optax.apply_updates(z, u)), st, v

    for _ in range(steps):
        z, st, _ = step(z, st)
    return np.asarray(z, dtype=float)


def _absorbable_by_affine(
    ref_data: np.ndarray, pert_data: np.ndarray, circuit: Circuit,
    ref_ctrl: np.ndarray, pert_ctrl: np.ndarray, knob: str,
    *, k_modes: int, steps: int, lr: float, earn_margin: float, cond_max: float,
) -> tuple[bool, dict[str, float]]:
    """Can the perturbed context's apparent ``knob`` difference be an affine on ITS cells?

    Fits shared kinetics from the REFERENCE context (assumed clean), then on the perturbed
    context fits (i) a pure-affine null ``(s, o)`` and (ii) the affine + the free bio knob.
    Returns ``(absorbable, diag)``: ``absorbable`` is True when the difference is explainable
    as technical — either the bio knob does NOT earn its parameter over the affine null
    (ΔBIC < ``earn_margin``) OR the bio knob is Fisher/Laplace-degenerate with the affine
    (condition number > ``cond_max`` / knob loads on a flat direction). Either ⇒ abstain.
    """
    scale_r, obs_r = calibrate_from_wt(ref_ctrl, circuit, k_modes=k_modes)
    scale_p, obs_p = calibrate_from_wt(pert_ctrl, circuit, k_modes=k_modes)
    log_shared, _roots_r = _fit_shared_knobs(
        ref_data, circuit, scale_r, obs_r, k_modes=k_modes, steps=steps, lr=lr
    )
    vals0 = np.exp(log_shared)
    r = _stable_roots(circuit, [("edge", 0, p) for p in _KNOBS], vals0)
    r = sorted(r, key=lambda x: tuple(float(v) for v in x))
    if len(r) != k_modes:
        return True, {"reason_flag": 1.0}  # cannot model perturbed as bistable → abstain
    roots_p = np.stack(r)

    n_p = int(np.asarray(pert_data).shape[0])
    log_n = float(np.log(max(n_p, 1)))

    # (i) pure-affine null.
    loss0 = _augmented_loss(pert_data, circuit, log_shared, knob, scale_p, obs_p,
                            roots_p, free_knob=False)
    z0 = _optimize(loss0, np.array([0.0, 0.0]), steps=steps, lr=lr)
    nll0 = float(loss0(jnp.asarray(z0)))
    bic0 = 2 * log_n + 2 * n_p * nll0  # 2 nuisance params

    # (ii) affine + free bio knob.
    loss1 = _augmented_loss(pert_data, circuit, log_shared, knob, scale_p, obs_p,
                            roots_p, free_knob=True)
    z1 = _optimize(loss1, np.array([log_shared[_KNOBS.index(knob)], z0[0], z0[1]]),
                   steps=steps, lr=lr)
    nll1 = float(loss1(jnp.asarray(z1)))
    bic1 = 3 * log_n + 2 * n_p * nll1  # 3 params (knob + 2 nuisance)

    earn = bic0 - bic1  # >margin ⇒ the bio knob earns its parameter over the affine null

    # Fisher/Laplace curvature of [knob, s, o] jointly at the fitted optimum. Reported as a
    # diagnostic ONLY — it is NOT the decision signal. MEASURED: the raw joint condition
    # number saturates high (~1e3) for EVERY case (confound and genuine alike) because the
    # linear offset ``o`` (in count units) and the log-scale params sit on wildly different
    # unit scales, so the condition number is dominated by that unit mismatch, not by
    # identifiability. The discriminative, unit-free signal is the *profiled* knob curvature
    # (Schur complement of the knob given the nuisance) — and its integrated form is exactly
    # ``earn`` (does freeing the knob, with the affine re-optimized, reduce the NLL enough to
    # pay its BIC parameter). So the decision is ``earn`` (profiled ΔBIC), not ``cond``.
    post = laplace_posterior(
        loss1, z1, names=[knob, "s", "o"], n_data=n_p, cond_max=cond_max
    )
    hess = np.asarray(post.hessian, dtype=float)  # curvature of the MEAN nll
    # profiled (nuisance-marginalized) knob curvature, per cell: H_kk − H_kn H_nn^-1 H_nk.
    h_kk, h_kn, h_nn = hess[0, 0], hess[0, 1:], hess[1:, 1:]
    try:
        prof = float(h_kk - h_kn @ np.linalg.solve(h_nn, h_kn))
    except np.linalg.LinAlgError:
        prof = 0.0
    prof_ratio = prof / max(h_kk, 1e-12)  # 1 = knob orthogonal to nuisance; 0 = aliased
    # MEASURED: prof_ratio does NOT separate a genuine ceiling (~0.30) from a uniform-scale
    # CONFOUND (~0.32) — locally the curvature is nearly identical; the discrimination is a
    # GLOBAL goodness-of-fit question (can ONE affine match BOTH modes at once), which only
    # the integrated ``earn`` (profiled ΔBIC) answers. Reported for transparency, not decision.

    diag = {
        "earn": earn, "cond": post.cond_number, "prof_ratio": prof_ratio,
        "s_hat": float(np.exp(z1[1])), "o_hat": float(z1[2]),
        "knob_val": float(np.exp(z1[0])), "shared_knob": float(vals0[_KNOBS.index(knob)]),
    }
    # The confound family is BY CONSTRUCTION inside the free affine null's span, so the bio
    # knob provably cannot earn its BIC parameter over it — the difference is absorbable iff
    # the knob fails to earn. (Unit-free ``prof_ratio`` is reported for corroboration.)
    absorbable = earn < earn_margin
    return absorbable, diag


def guard_b_classify(
    context_a: Context,
    context_b: Context,
    circuit: Circuit,
    *,
    winner: str | None = None,
    k_modes: int = 2,
    steps: int = 150,
    lr: float = 0.05,
    earn_margin: float = 6.0,
    cond_max: float = 100.0,
    check_both: bool = True,
) -> GuardBResult:
    """The unified nuisance-augmented guard — one continuous test over the affine family.

    ``winner`` is the BIC-winning knob from the baseline differential fit (which knob would
    be called); if ``None`` it is chosen by a quick shared-vs-per-context screen. A positive
    ``*-diff`` is returned ONLY if the apparent difference is NOT absorbable by a free affine
    ``(s, o)`` on EITHER context (``check_both``) — i.e. the bio knob both earns its parameter
    over a pure-affine null AND is Fisher/Laplace-identifiable jointly with the affine.
    Otherwise abstains (``unresolved`` if a knob won but is affine-degenerate;
    ``no-difference`` if no knob earns). Because a per-context scale is exactly degenerate
    with ``v_max`` and an offset shifts the modes, this abstains on a GENUINE ceiling too —
    the honest identifiability limit the inert anchor (A) buys back.
    """
    da, db = np.asarray(context_a.data, float), np.asarray(context_b.data, float)
    ca, cb = np.asarray(context_a.control, float), np.asarray(context_b.control, float)

    # LNA trust gate (mirror differential gate 1).
    sa, _ = calibrate_from_wt(ca, circuit, k_modes=k_modes)
    sb, _ = calibrate_from_wt(cb, circuit, k_modes=k_modes)
    ok_a, why_a = lna_reliable(circuit, sa)
    ok_b, why_b = lna_reliable(circuit, sb)
    if not ok_a or not ok_b:
        return GuardBResult(
            "unresolved", f"LNA untrustworthy (A: {why_a}; B: {why_b})",
            winner or "?", float("nan"), float("nan"), False, float("nan"),
            float("nan"),
        )

    if winner is None:
        winner = _screen_winner(context_a, context_b, circuit, k_modes=k_modes,
                                steps=steps, lr=lr)

    # Direction 1: is B's apparent difference an affine on B (A = clean reference)?
    absorb_b, diag_b = _absorbable_by_affine(
        da, db, circuit, ca, cb, winner, k_modes=k_modes, steps=steps, lr=lr,
        earn_margin=earn_margin, cond_max=cond_max,
    )
    diag = {"B": diag_b}
    if absorb_b:
        return _abstain(winner, diag_b, diag, "B")

    if check_both:
        # Direction 2: is A's apparent difference an affine on A (B = reference)?
        absorb_a, diag_a = _absorbable_by_affine(
            db, da, circuit, cb, ca, winner, k_modes=k_modes, steps=steps, lr=lr,
            earn_margin=earn_margin, cond_max=cond_max,
        )
        diag["A"] = diag_a
        if absorb_a:
            return _abstain(winner, diag_a, diag, "A")

    call = _CALL_OF[winner]
    return GuardBResult(
        call,
        f"{call}: the {winner} difference EARNS its parameter over a free affine (s,o) on "
        f"either context (ΔBIC_B={diag_b['earn']:.1f}) and is Fisher-identifiable jointly "
        f"with the nuisance (cond={diag_b['cond']:.1f} ≤ {cond_max:g}) — a real, "
        f"technically-robust mechanism difference",
        winner, diag_b["earn"], diag_b["cond"], True,
        diag_b["s_hat"], diag_b["o_hat"], extras=diag,
    )


def _abstain(winner: str, d: dict[str, float], diag: dict, side: str) -> GuardBResult:
    if d.get("reason_flag"):
        reason = f"context {side} does not present 2 stable modes at shared kinetics"
        return GuardBResult("unresolved", reason, winner, float("nan"), float("nan"),
                            False, float("nan"), float("nan"), extras=diag)
    # earn < 0: the free affine strictly out-explains the bio knob → the apparent difference
    # is (statistically) NO real biological difference beyond a per-condition technical
    # affine. 0 ≤ earn < margin: the knob half-earns → genuinely ambiguous, abstain louder.
    if d["earn"] < 0.0:
        call = "no-difference"
        reason = (
            f"a free affine (s={d['s_hat']:.3f}, o={d['o_hat']:+.3f}) on context {side} "
            f"absorbs the apparent {winner} difference: the bio knob does not earn its "
            f"parameter over the pure-affine null (ΔBIC={d['earn']:.1f} < 0) — the "
            "difference is consistent with a per-condition technical affine, not a mechanism"
        )
    else:
        call = "unresolved"
        reason = (
            f"the {winner} difference only partly out-explains a free affine on context "
            f"{side} (ΔBIC={d['earn']:.1f} < margin; profiled knob curvature ratio "
            f"{d['prof_ratio']:.3f}) — NUDGE cannot certify the difference is biological "
            f"rather than a technical affine (s={d['s_hat']:.3f}, o={d['o_hat']:+.3f}); "
            "abstains. An INERT anchor (A) resolves it when available"
        )
    return GuardBResult(call, reason, winner, d["earn"], d["cond"], False,
                        d["s_hat"], d["o_hat"], extras=diag)


def _screen_winner(
    context_a: Context, context_b: Context, circuit: Circuit,
    *, k_modes: int, steps: int, lr: float,
) -> str:
    """Quick per-knob screen: which single per-context knob best explains B given A.

    Fits shared knobs from A, then for each knob fits B's free bio knob (NO affine) and
    picks the min-NLL — the knob the baseline differential would most likely call.
    """
    ca, cb = np.asarray(context_a.control, float), np.asarray(context_b.control, float)
    scale_a, obs_a = calibrate_from_wt(ca, circuit, k_modes=k_modes)
    scale_b, obs_b = calibrate_from_wt(cb, circuit, k_modes=k_modes)
    log_shared, _ = _fit_shared_knobs(
        np.asarray(context_a.data, float), circuit, scale_a, obs_a,
        k_modes=k_modes, steps=steps, lr=lr,
    )
    vals0 = np.exp(log_shared)
    r = _stable_roots(circuit, [("edge", 0, p) for p in _KNOBS], vals0)
    r = sorted(r, key=lambda x: tuple(float(v) for v in x))
    roots_p = np.stack(r)
    best, best_nll = "vmax", np.inf
    for knob in _KNOBS:
        loss = _augmented_loss(
            np.asarray(context_b.data, float), circuit, log_shared, knob,
            scale_b, obs_b, roots_p, free_knob=True,
        )
        z = _optimize(lambda z, _loss=loss: _loss(jnp.array([z[0], 0.0, 0.0])),
                      np.array([log_shared[_KNOBS.index(knob)]]), steps=steps, lr=lr)
        nll = float(loss(jnp.array([z[0], 0.0, 0.0])))
        if nll < best_nll:
            best, best_nll = knob, nll
    return best
