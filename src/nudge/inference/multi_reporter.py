"""Multi-reporter joint attribution — break the K⇄v_max degeneracy with a panel.

NUDGE's measured, load-bearing weakness is the **K⇄v_max / gain⇄threshold
degeneracy** (``scripts/vv/FINDINGS.md`` §2): a *single* reporter of one latent
switch under-determines the mechanism, so NUDGE abstains. The reason a single
reporter cannot separate a **threshold** shift (where the latent switches) from a
**ceiling** change (the latent's max) is that one affine readout ``y = base +
gain·activity`` cannot tell a change in the *latent's* ceiling apart from a change in
its *own* gain — the two are the same free parameter.

The fix FINDINGS keeps naming: **fit several downstream reporters of ONE latent
switch jointly.** Each reporter ``y_j = base_j + gain_j · A · f(dose; K, n)`` observes
the *same* latent switch ``f`` with its own gain/offset. The mechanisms then project
DIFFERENTLY onto a panel of heterogeneous gains:

- a **threshold** shift moves the inflection in dose **identically** across reporters
  (a shape shift), so only a shared ``K`` change explains all curves at once;
- a **gain** change alters the shared steepness ``n`` identically across reporters;
- a **ceiling** change scales the latent's max ``A`` — hitting **every** reporter's ON
  amplitude by the *same fraction* (because the reporter gains are pinned from the
  control) — which a single reporter cannot distinguish from its own gain drifting, but
  a panel can, because the drop is consistent across heterogeneous gains.

Pinning each reporter's affine from the control and sharing one latent across the panel
makes the perturbed fit **over-determined** (M curves constrain 3 shared knobs) — higher
Fisher information, the degeneracy breaks. This is the multi-reporter analogue of the
second-operating-point ×16 result (``FINDINGS`` "Covariance attribution" M3): more
*reporters* of one latent, instead of more *conditions*.

**Fail-safe — this capability STRENGTHENS the guarantee.** A spurious mechanism must
now be consistent across ALL reporters, which is much harder to fake. And the honest new
gate is the **consistency guard**: if the reporters cannot be explained by one shared
latent (a reporter secretly reads a *different* latent — a hidden node / wrong panel),
the joint residual is large and NUDGE **abstains** (``off-model``) — "the panel is not
consistent with a single latent switch" — rather than silently average an inconsistent
panel into a confident call (``NUDGE-LIM-014``). With a single reporter (``M == 1``) the
consistency of the pinned gain cannot be checked, so the affine is *not* pinned and the
ceiling↔gain degeneracy remains — NUDGE returns ``unresolved``, exactly the abstention
the panel is built to break.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "MultiReporterFit",
    "MultiReporterResult",
    "ReporterFit",
    "ReporterObservation",
    "attribute_multi_reporter",
    "classify_multi_reporter",
    "fit_multi_reporter",
    "simulate_reporter_panel",
]

Direction = str  # "activate" (rises with dose) | "repress" (falls with dose)

#: The mechanism a perturbation localizes to on the shared latent, or an abstention.
CALLS = ("threshold", "gain", "ceiling", "no-effect", "unresolved", "off-model")


@dataclass(frozen=True)
class ReporterObservation:
    """One reporter's dose-response of the shared latent, control vs perturbed.

    ``dose`` / ``control`` / ``perturbed`` are paired 1-D arrays (one value per dose
    level). ``control`` is the reporter's response in the WT / reference condition;
    ``perturbed`` is the same reporter in the perturbed condition. All reporters observe
    the *same* latent switch through their own (heterogeneous) affine gain/offset.
    """

    name: str
    dose: Any  # 1-D array-like of dose levels
    control: Any  # 1-D array-like of the WT / reference response
    perturbed: Any  # 1-D array-like of the perturbed response


@dataclass(frozen=True)
class ReporterFit:
    """One reporter's calibrated affine + how well it shares the panel's latent."""

    name: str
    floor: float
    gain: float  # the reporter's heterogeneous gain (scale) on the latent
    r2_shared: float  # control-curve R² under the shared-latent joint fit
    r2_independent: float  # control-curve R² under its OWN free Hill fit


@dataclass(frozen=True)
class MultiReporterFit:
    """A joint multi-reporter fit + everything :func:`classify_multi_reporter` needs.

    The panel shares one latent ``f(dose; K, n)`` (WT-calibrated ``k_wt`` / ``n_wt``);
    the perturbation is localized by which shared knob best explains the perturbed
    panel — the restricted losses ``loss_threshold`` (free K) / ``loss_gain`` (free n) /
    ``loss_ceiling`` (free A), vs ``loss_no_effect`` (WT latent) and ``loss_full`` (all
    free). ``knob_margin`` is the runner-up/winner loss ratio (>1 favours the winner).
    The ``ci_*`` are bootstrap CIs on the shared perturbation (log2 ratios; a CI that
    excludes 0 means the knob moved). ``panel_r2`` / ``worst_reporter_r2`` /
    ``consistency_ratio`` drive the one-shared-latent consistency guard.
    """

    direction: str
    n_reporters: int
    pinned_affine: bool
    k_wt: float
    n_wt: float
    loss_no_effect: float
    loss_threshold: float
    loss_gain: float
    loss_ceiling: float
    loss_full: float
    winner: str
    knob_margin: float  # runner-up loss / winner loss (>1 favours the winner)
    effect_margin: float  # no-effect loss / winner loss (>1 means a real effect)
    k_ratio: float  # K_perturbed / K_wt
    n_ratio: float  # n_perturbed / n_wt
    ceiling_ratio: float  # A_perturbed / A_wt
    ci_log2_k: tuple[float, float]
    ci_log2_n: tuple[float, float]
    ci_log2_ceiling: tuple[float, float]
    panel_r2: float
    worst_reporter_r2: float
    worst_reporter_independent_r2: float
    consistency_ratio: float  # shared-latent RSS / independent-fit RSS (>1 worse)
    n_points_total: int
    reporters: tuple[ReporterFit, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MultiReporterResult:
    """A joint fit plus its conservative verdict and the human-readable reason."""

    fit: MultiReporterFit
    call: str
    reason: str

    @property
    def is_reliable(self) -> bool:
        """Trustworthy enough to invert (``design()``'s integrity gate).

        A resolved mechanism (``threshold`` / ``gain`` / ``ceiling``) is reliable; the
        abstentions (``unresolved`` / ``no-effect`` / ``off-model``) are not. Satisfies
        the :class:`~nudge.design.invert.AttributionResult` protocol additively.
        """
        return self.call in {"threshold", "gain", "ceiling"}


# --------------------------------------------------------------------------- #
# the shared-latent forward model + least-squares fitters
# --------------------------------------------------------------------------- #
def _frac(dose: np.ndarray, k: float, n: float, direction: str) -> np.ndarray:
    """The shared latent's Hill fraction ``f(dose) ∈ [0, 1]`` (the switch shape).

    ``activate`` rises with dose (``d^n / (K^n + d^n)``); ``repress`` falls. This is the
    *same* Hill primitive the circuit vector field and the dose-response path use
    (:func:`nudge.mechanisms.regulatory.hill_activation`), normalized to a unit ceiling
    so the latent's max lives in the shared amplitude ``A``.
    """
    d = np.maximum(np.asarray(dose, dtype=float), 0.0)
    k = max(float(k), 1e-9)
    n = min(max(float(n), 1e-6), 60.0)
    # Stable form f = 1/(1+(K/d)^n): no overflow (a huge ratio → 0 cleanly), 0 at d=0.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratio = np.where(d > 0.0, (k / np.where(d > 0.0, d, 1.0)) ** n, np.inf)
        f = 1.0 / (1.0 + ratio)
    f = np.nan_to_num(f, nan=0.0, posinf=1.0, neginf=0.0)
    return f if direction == "activate" else 1.0 - f


def _least_squares(resid: Any, p0: list[float]) -> tuple[np.ndarray, float] | None:
    """Robust TRF wrapper: returns ``(params, rss)`` or ``None`` on failure."""
    from scipy.optimize import least_squares

    try:
        sol = least_squares(resid, p0, method="trf", max_nfev=4000)
    except Exception:
        return None
    if not sol.success and sol.status <= 0:
        return None
    r = np.asarray(sol.fun, dtype=float)
    return np.asarray(sol.x, dtype=float), float(np.sum(r * r))


def _scales(obs: list[ReporterObservation]) -> np.ndarray:
    """Per-reporter residual scale so heterogeneous gains contribute equally."""
    out = []
    for o in obs:
        both = np.concatenate(
            [np.asarray(o.control, dtype=float), np.asarray(o.perturbed, dtype=float)]
        )
        out.append(max(float(np.std(both)), 1e-6))
    return np.asarray(out)


def _calibrate(
    obs: list[ReporterObservation], direction: str
) -> tuple[float, float, list[tuple[float, float]], list[float], list[float]] | None:
    """Jointly fit the shared latent ``(K, n)`` + per-reporter ``(floor, gain)`` to WT.

    Returns ``(k_wt, n_wt, [(floor_j, gain_j)], [rss_j], [tss_j])`` fit to the CONTROL
    curves with the latent amplitude pinned to ``A = 1`` (absorbed into the gains). The
    per-reporter residual/total sums of squares feed the consistency guard.
    """
    m = len(obs)
    scales = _scales(obs)
    doses = [np.asarray(o.dose, dtype=float) for o in obs]
    ctrl = [np.asarray(o.control, dtype=float) for o in obs]
    dpos_parts = [d[d > 0] for d in doses if np.any(d > 0)]
    dpos = np.concatenate(dpos_parts) if dpos_parts else np.array([1.0])

    def unpack(p: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
        k = float(np.exp(np.clip(p[0], -20.0, 20.0)))
        n = float(np.exp(np.clip(p[1], -20.0, 20.0)))
        floors = p[2 : 2 + m]
        gains = np.exp(np.clip(p[2 + m : 2 + 2 * m], -20.0, 20.0))
        return k, n, floors, gains

    def resid(p: np.ndarray) -> np.ndarray:
        k, n, floors, gains = unpack(p)
        parts = []
        for j in range(m):
            pred = floors[j] + gains[j] * _frac(doses[j], k, n, direction)
            parts.append((pred - ctrl[j]) / scales[j])
        return np.concatenate(parts)

    best: tuple[np.ndarray, float] | None = None
    for k0 in np.quantile(dpos, [0.35, 0.6]):
        for n0 in (2.0, 4.0):
            floor0 = [float(c.min()) for c in ctrl]
            gain0 = [float(np.log(max(float(c.max() - c.min()), 1e-3))) for c in ctrl]
            init = [
                float(np.log(max(k0, 1e-3))),
                float(np.log(n0)),
                *floor0,
                *gain0,
            ]
            got = _least_squares(resid, init)
            if got is not None and (best is None or got[1] < best[1]):
                best = got
    if best is None:
        return None
    k, n, floors, gains = unpack(best[0])
    if not (np.isfinite(k) and np.isfinite(n)):
        return None
    affine = [(float(floors[j]), float(gains[j])) for j in range(m)]
    rss_j, tss_j = [], []
    for j in range(m):
        pred = floors[j] + gains[j] * _frac(doses[j], k, n, direction)
        rss_j.append(float(np.sum((pred - ctrl[j]) ** 2)))
        tss_j.append(max(float(np.sum((ctrl[j] - ctrl[j].mean()) ** 2)), 1e-12))
    return k, n, affine, rss_j, tss_j


def _fit_perturbed(
    obs: list[ReporterObservation],
    direction: str,
    k_wt: float,
    n_wt: float,
    affine: list[tuple[float, float]],
    *,
    free: str,
    pin_affine: bool,
) -> tuple[dict[str, float], float] | None:
    """Fit the perturbed panel with one shared knob free (``K``/``n``/``A``), or none.

    ``free`` ∈ ``{"none", "K", "n", "A", "full"}`` selects the shared perturbation. When
    ``pin_affine`` the per-reporter ``(floor, gain)`` are held at their WT-calibrated
    values (the panel case, validated by the consistency guard); otherwise they are
    re-fit per reporter (the ``M == 1`` case — the honest admission that a lone
    reporter's gain cannot be trusted stable, so ceiling ``A`` is degenerate with it).
    Returns ``({"K","n","A"}, rss)``.
    """
    m = len(obs)
    scales = _scales(obs)
    doses = [np.asarray(o.dose, dtype=float) for o in obs]
    pert = [np.asarray(o.perturbed, dtype=float) for o in obs]
    n_shared = {"none": 0, "K": 1, "n": 1, "A": 1, "full": 3}[free]

    def unpack(p: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        def ex(v: float) -> float:
            return float(np.exp(np.clip(v, -20.0, 20.0)))

        k, n, a = k_wt, n_wt, 1.0
        if free == "K":
            k = ex(p[0])
        elif free == "n":
            n = ex(p[0])
        elif free == "A":
            a = ex(p[0])
        elif free == "full":
            k, n, a = ex(p[0]), ex(p[1]), ex(p[2])
        if pin_affine:
            floors = np.array([affine[j][0] for j in range(m)])
            gains = np.array([affine[j][1] for j in range(m)])
        else:
            floors = p[n_shared : n_shared + m]
            gains = np.exp(p[n_shared + m : n_shared + 2 * m])
        return k, n, a, floors, gains

    def resid(p: np.ndarray) -> np.ndarray:
        k, n, a, floors, gains = unpack(p)
        parts = []
        for j in range(m):
            pred = floors[j] + gains[j] * a * _frac(doses[j], k, n, direction)
            parts.append((pred - pert[j]) / scales[j])
        return np.concatenate(parts)

    p0: list[float] = []
    if free == "K":
        p0 = [float(np.log(k_wt))]
    elif free == "n":
        p0 = [float(np.log(n_wt))]
    elif free == "A":
        p0 = [0.0]
    elif free == "full":
        p0 = [float(np.log(k_wt)), float(np.log(n_wt)), 0.0]
    if not pin_affine:
        p0 += [affine[j][0] for j in range(m)]
        p0 += [float(np.log(max(affine[j][1], 1e-6))) for j in range(m)]

    if not p0:  # free == "none" and pinned: nothing to fit, just score the WT null
        k, n, a, floors, gains = unpack(np.zeros(0))
        rss = 0.0
        for j in range(m):
            pred = floors[j] + gains[j] * a * _frac(doses[j], k, n, direction)
            rss += float(np.sum(((pred - pert[j]) / scales[j]) ** 2))
        return {"K": k_wt, "n": n_wt, "A": 1.0}, rss

    got = _least_squares(resid, p0)
    if got is None:
        return None
    k, n, a, _floors, _gains = unpack(got[0])
    return {"K": k, "n": n, "A": a}, got[1]


def _independent_r2(o: ReporterObservation, direction: str) -> float:
    """A single reporter's OWN best free-Hill R² on its control curve (consistency ref).

    A reporter that reads a sigmoidal latent fits its own free Hill well; if it *also*
    fits the shared latent poorly, it reads a DIFFERENT latent (the guard fires).
    """
    dose = np.asarray(o.dose, dtype=float)
    y = np.asarray(o.control, dtype=float)
    tss = max(float(np.sum((y - y.mean()) ** 2)), 1e-12)
    scale = max(float(np.std(y)), 1e-6)
    dpos = dose[dose > 0]
    kseed = float(np.median(dpos)) if dpos.size else 1.0

    def resid(p: np.ndarray) -> np.ndarray:
        cp = np.clip(p, -20.0, 20.0)
        floor, gain, k, n = p[0], np.exp(cp[1]), np.exp(cp[2]), np.exp(cp[3])
        pred = floor + gain * _frac(dose, k, n, direction)
        return (pred - y) / scale

    best: tuple[np.ndarray, float] | None = None
    for n0 in (2.0, 4.0):
        init = [
            float(y.min()),
            float(np.log(max(y.max() - y.min(), 1e-3))),
            float(np.log(max(kseed, 1e-3))),
            float(np.log(n0)),
        ]
        got = _least_squares(resid, init)
        if got is not None and (best is None or got[1] < best[1]):
            best = got
    if best is None:
        return 0.0
    floor = best[0][0]
    gain, k, n = (float(np.exp(np.clip(best[0][i], -20.0, 20.0))) for i in (1, 2, 3))
    pred = floor + gain * _frac(dose, k, n, direction)
    return float(1.0 - np.sum((pred - y) ** 2) / tss)


def fit_multi_reporter(
    reporters: Sequence[ReporterObservation],
    *,
    direction: str = "activate",
    n_boot: int = 200,
    seed: int = 0,
) -> MultiReporterFit:
    """Jointly fit a panel of reporters of ONE latent switch, control vs perturbed.

    Calibrates the shared latent ``(K, n)`` + each reporter's affine ``(floor, gain)``
    from the CONTROL curves, then localizes the perturbation to a single shared knob by
    three restricted fits of the perturbed panel — free ``K`` (**threshold**), free
    ``n`` (**gain**), free ``A`` (**ceiling**) — against the WT-latent null and a free
    reference. With ≥ 2 reporters the affines are pinned from WT (the panel over-
    determines the latent, breaking the K⇄v_max degeneracy); with a single reporter they
    are re-fit (so the ceiling stays degenerate with the reporter gain — an honest
    ``unresolved``). Bootstrap CIs on the shared log2 ratios come from resampling doses.
    Raises ``ValueError`` on an empty panel or a reporter with < 4 dose points.
    """
    obs = list(reporters)
    if not obs:
        raise ValueError("need at least one reporter")
    for o in obs:
        if len(o.dose) < 4:
            raise ValueError(
                f"reporter {o.name!r} has {len(o.dose)} dose points; need >= 4"
            )
        if not (len(o.dose) == len(o.control) == len(o.perturbed)):
            raise ValueError(
                f"reporter {o.name!r}: dose/control/perturbed length mismatch"
            )
    if direction not in ("activate", "repress"):
        raise ValueError(f"direction must be 'activate'/'repress', got {direction!r}")

    m = len(obs)
    pin_affine = m >= 2

    calib = _calibrate(obs, direction)
    if calib is None:
        raise RuntimeError("multi-reporter calibration failed to converge")
    k_wt, n_wt, affine, rss_j, tss_j = calib

    reporter_fits: list[ReporterFit] = []
    shared_r2s, indep_r2s = [], []
    indep_rss = 0.0
    for j, o in enumerate(obs):
        r2_shared = float(1.0 - rss_j[j] / tss_j[j])
        r2_indep = _independent_r2(o, direction)
        shared_r2s.append(r2_shared)
        indep_r2s.append(r2_indep)
        indep_rss += max(1.0 - r2_indep, 0.0) * tss_j[j]
        reporter_fits.append(
            ReporterFit(
                name=o.name,
                floor=affine[j][0],
                gain=affine[j][1],
                r2_shared=r2_shared,
                r2_independent=r2_indep,
            )
        )
    panel_r2 = float(np.mean(shared_r2s))
    worst_idx = int(np.argmin(shared_r2s))
    worst_r2 = float(shared_r2s[worst_idx])
    worst_indep = float(indep_r2s[worst_idx])
    shared_rss = float(np.sum(rss_j))
    consistency_ratio = shared_rss / max(indep_rss, 1e-9)

    def score(free: str) -> tuple[dict[str, float], float]:
        got = _fit_perturbed(
            obs, direction, k_wt, n_wt, affine, free=free, pin_affine=pin_affine
        )
        if got is None:
            return {"K": k_wt, "n": n_wt, "A": 1.0}, float("inf")
        return got

    _p_none, loss_none = score("none")
    p_k, loss_k = score("K")
    p_n, loss_n = score("n")
    p_a, loss_a = score("A")
    p_full, loss_full = score("full")

    knob_losses = {"threshold": loss_k, "gain": loss_n, "ceiling": loss_a}
    ordered = sorted(knob_losses.items(), key=lambda kv: kv[1])
    winner, best_loss = ordered[0]
    runner_loss = ordered[1][1]
    knob_margin = float(runner_loss / max(best_loss, 1e-12))
    effect_margin = float(loss_none / max(best_loss, 1e-12))

    # Bootstrap the shared perturbation (resample dose indices within each reporter).
    rng = np.random.default_rng(seed)
    boot_log2k: list[float] = []
    boot_log2n: list[float] = []
    boot_log2a: list[float] = []
    for _ in range(max(n_boot, 0)):
        resampled: list[ReporterObservation] = []
        for o in obs:
            nidx = len(o.dose)
            idx = rng.integers(0, nidx, nidx)
            resampled.append(
                ReporterObservation(
                    name=o.name,
                    dose=np.asarray(o.dose, dtype=float)[idx],
                    control=np.asarray(o.control, dtype=float)[idx],
                    perturbed=np.asarray(o.perturbed, dtype=float)[idx],
                )
            )
        got = _fit_perturbed(
            resampled,
            direction,
            k_wt,
            n_wt,
            affine,
            free="full",
            pin_affine=pin_affine,
        )
        if got is None:
            continue
        p, _ = got
        boot_log2k.append(float(np.log2(max(p["K"], 1e-9) / max(k_wt, 1e-9))))
        boot_log2n.append(float(np.log2(max(p["n"], 1e-9) / max(n_wt, 1e-9))))
        boot_log2a.append(float(np.log2(max(p["A"], 1e-9))))

    def ci(vals: list[float]) -> tuple[float, float]:
        if not vals:
            return (float("nan"), float("nan"))
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    n_points = int(sum(len(o.dose) for o in obs))
    return MultiReporterFit(
        direction=direction,
        n_reporters=m,
        pinned_affine=pin_affine,
        k_wt=float(k_wt),
        n_wt=float(n_wt),
        loss_no_effect=float(loss_none),
        loss_threshold=float(loss_k),
        loss_gain=float(loss_n),
        loss_ceiling=float(loss_a),
        loss_full=float(loss_full),
        winner=winner,
        knob_margin=knob_margin,
        effect_margin=effect_margin,
        k_ratio=float(p_k["K"] / max(k_wt, 1e-9)),
        n_ratio=float(p_n["n"] / max(n_wt, 1e-9)),
        ceiling_ratio=float(p_a["A"]),
        ci_log2_k=ci(boot_log2k),
        ci_log2_n=ci(boot_log2n),
        ci_log2_ceiling=ci(boot_log2a),
        panel_r2=panel_r2,
        worst_reporter_r2=worst_r2,
        worst_reporter_independent_r2=worst_indep,
        consistency_ratio=float(consistency_ratio),
        n_points_total=n_points,
        reporters=tuple(reporter_fits),
        extras={"p_full": p_full},
    )


def _ci_excludes_zero(ci: tuple[float, float]) -> bool:
    lo, hi = ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return False
    return lo > 0.0 or hi < 0.0


def classify_multi_reporter(
    fit: MultiReporterFit,
    *,
    knob_margin: float = 1.5,
    effect_margin: float = 1.4,
    consistency_ratio_max: float = 3.0,
    min_panel_r2: float = 0.5,
    inconsistent_reporter_r2: float = 0.6,
    clean_reporter_r2: float = 0.75,
) -> tuple[str, str]:
    """Turn a joint fit into a conservative verdict — the fail-safe classifier.

    Gates, most-conservative first:

    1. **off-model (the consistency guard, ``NUDGE-LIM-014``)** — the reporters
       cannot be explained by ONE shared latent: the shared-latent ``panel_r2`` is
       poor, or some reporter fits its OWN Hill cleanly (``r2_independent`` high) yet
       the shared latent explains it badly (``r2_shared`` low) — it reads a
       *different* latent. NUDGE abstains rather than average an inconsistent panel.
    2. **unresolved (identifiability)** — the affine could not be pinned (a single
       reporter — the ceiling stays degenerate with the reporter gain, the very
       abstention the panel breaks), the winning knob does not beat the runner-up by
       ``knob_margin``, or the bootstrap CI of the winner straddles 0.
    3. **no-effect** — the WT-latent null is nearly as good as the best knob
       (``effect_margin`` small): the perturbation did not move the latent.
    4. **threshold / gain / ceiling** — the winning knob beats the runner-up by
       ``knob_margin`` *and* the WT null by ``effect_margin`` *and* its bootstrap CI
       excludes 0. Returns ``(call, reason)``.
    """
    # 1. consistency guard — is the panel really ONE latent switch?
    inconsistent = (
        fit.worst_reporter_independent_r2 >= clean_reporter_r2
        and fit.worst_reporter_r2 <= inconsistent_reporter_r2
    )
    if fit.n_reporters >= 2 and (
        fit.panel_r2 < min_panel_r2
        or inconsistent
        or fit.consistency_ratio > consistency_ratio_max
    ):
        extra = (
            f" and a reporter fits its OWN Hill well (R²="
            f"{fit.worst_reporter_independent_r2:.2f}) but the shared latent explains "
            f"it badly (R²={fit.worst_reporter_r2:.2f}) — it reads a DIFFERENT latent"
            if inconsistent
            else ""
        )
        return "off-model", (
            f"the panel is NOT consistent with a single latent switch: shared-latent "
            f"panel R²={fit.panel_r2:.2f}{extra} (consistency ratio "
            f"{fit.consistency_ratio:.1f}). NUDGE abstains rather than average an "
            "inconsistent panel into a mechanism (NUDGE-LIM-014)"
        )

    # 2a. a single reporter cannot pin its affine → ceiling ⇄ gain degeneracy stands.
    if not fit.pinned_affine:
        return "unresolved", (
            f"a single reporter (M={fit.n_reporters}) cannot separate a latent ceiling "
            "change from its own gain drifting (the K⇄v_max degeneracy) — the affine "
            "cannot be pinned or checked for consistency, so the mechanism is "
            "unidentifiable. Add reporters of the SAME latent to resolve it "
            "(the multi-reporter degeneracy-break, NUDGE-LIM-014)"
        )

    # 2b. no real effect on the latent (the WT null already explains the panel).
    if fit.effect_margin < effect_margin:
        return "no-effect", (
            f"the WT-latent null is nearly as good as the best knob (effect ratio "
            f"{fit.effect_margin:.2f} < {effect_margin:g}) — the perturbation did not "
            "move the shared latent above noise; nothing to attribute"
        )

    # 3. the mechanisms are not separable on this panel.
    ci_by_knob = {
        "threshold": fit.ci_log2_k,
        "gain": fit.ci_log2_n,
        "ceiling": fit.ci_log2_ceiling,
    }
    if fit.knob_margin < knob_margin:
        return "unresolved", (
            f"the best knob ({fit.winner}) does not beat the runner-up by the required "
            f"margin (loss ratio {fit.knob_margin:.2f} < {knob_margin:g}) — the panel "
            "cannot separate threshold / gain / ceiling here; NUDGE abstains"
        )

    # 4. the winning knob must also clear its bootstrap CI.
    if not _ci_excludes_zero(ci_by_knob[fit.winner]):
        lo, hi = ci_by_knob[fit.winner]
        return "unresolved", (
            f"the winning knob ({fit.winner}) wins the loss margin "
            f"(×{fit.knob_margin:.2f}) but its bootstrap CI straddles 0 (log2 ratio "
            f"[{lo:+.2f}, {hi:+.2f}]) — the shift is not resolved from resampling "
            "noise; NUDGE abstains"
        )

    detail = {
        "threshold": (
            f"the shared threshold K shifts (K_perturbed/K_wt={fit.k_ratio:.2f}, log2 "
            f"CI [{fit.ci_log2_k[0]:+.2f}, {fit.ci_log2_k[1]:+.2f}]) — the inflection "
            "moves IDENTICALLY across all reporters, which only a shared-K change "
            "explains"
        ),
        "gain": (
            f"the shared gain n changes (n_perturbed/n_wt={fit.n_ratio:.2f}, log2 CI "
            f"[{fit.ci_log2_n[0]:+.2f}, {fit.ci_log2_n[1]:+.2f}]) — the switch "
            "steepness changes identically across reporters"
        ),
        "ceiling": (
            f"the shared latent ceiling scales "
            f"(A_perturbed/A_wt={fit.ceiling_ratio:.2f}, log2 CI "
            f"[{fit.ci_log2_ceiling[0]:+.2f}, {fit.ci_log2_ceiling[1]:+.2f}]) — every "
            "reporter's ON amplitude drops by the SAME fraction (pinned gains), the "
            "signature a single reporter cannot tell from its own gain"
        ),
    }[fit.winner]
    return fit.winner, (
        f"{detail}. It beats the runner-up by ×{fit.knob_margin:.2f} and the WT null "
        f"by ×{fit.effect_margin:.2f} across {fit.n_reporters} reporters "
        f"(panel R²={fit.panel_r2:.2f}) — the joint panel breaks the K⇄v_max "
        "degeneracy a single reporter abstains on"
    )


def attribute_multi_reporter(
    reporters: Sequence[ReporterObservation],
    *,
    direction: str = "activate",
    n_boot: int = 200,
    seed: int = 0,
    knob_margin: float = 1.5,
    effect_margin: float = 1.4,
    consistency_ratio_max: float = 3.0,
    min_panel_r2: float = 0.5,
) -> MultiReporterResult:
    """Fit + classify a reporter panel in one call — the CLI / MCP entry point."""
    fit = fit_multi_reporter(reporters, direction=direction, n_boot=n_boot, seed=seed)
    call, reason = classify_multi_reporter(
        fit,
        knob_margin=knob_margin,
        effect_margin=effect_margin,
        consistency_ratio_max=consistency_ratio_max,
        min_panel_r2=min_panel_r2,
    )
    return MultiReporterResult(fit=fit, call=call, reason=reason)


# --------------------------------------------------------------------------- #
# synthetic ground truth — the force-multiplier demonstration
# --------------------------------------------------------------------------- #
def simulate_reporter_panel(
    *,
    mechanism: str,
    n_reporters: int = 4,
    doses: Sequence[float] | None = None,
    k_wt: float = 20.0,
    n_wt: float = 4.0,
    factor: float = 3.0,
    direction: str = "activate",
    gain_range: tuple[float, float] = (0.4, 3.0),
    floor_range: tuple[float, float] = (0.02, 0.4),
    noise: float = 0.03,
    seed: int = 0,
    hidden_latent_reporter: int | None = None,
) -> list[ReporterObservation]:
    """Build a ground-truth reporter panel of ONE latent switch (control vs perturbed).

    ``mechanism`` ∈ ``{"threshold", "gain", "ceiling", "none"}`` sets the perturbation
    on the *shared latent*: ``threshold`` scales ``K`` by ``factor``, ``gain`` divides
    ``n`` by ``factor`` (a less-cooperative switch), ``ceiling`` scales the latent max
    ``A`` by ``1/factor``, ``none`` leaves it unchanged. Each reporter is an affine map
    ``y_j = base_j + gain_j · A · f(dose; K, n)`` — genuinely a
    :class:`~nudge.mechanisms.readout.Readout` of the 1-D latent activity — with a
    heterogeneous gain/offset drawn from ``gain_range`` / ``floor_range`` and mild
    multiplicative ``noise``. Pass ``hidden_latent_reporter=j`` to make reporter ``j``
    read a DIFFERENT latent (a shifted ``K``) — the inconsistent panel the consistency
    guard must catch (``off-model``). The reporter affines are the SAME in control and
    perturbed (a reporter is a fixed measurement device).
    """
    import jax.numpy as jnp

    from nudge.mechanisms.readout import Readout

    if mechanism not in ("threshold", "gain", "ceiling", "none"):
        raise ValueError(f"unknown mechanism {mechanism!r}")
    rng = np.random.default_rng(seed)
    if doses is None:
        dose = np.concatenate([[0.0], np.geomspace(1.0, 200.0, 11)])
    else:
        dose = np.asarray(doses, dtype=float)

    k_p, n_p, a_p = k_wt, n_wt, 1.0
    if mechanism == "threshold":
        k_p = k_wt * factor
    elif mechanism == "gain":
        n_p = n_wt / factor
    elif mechanism == "ceiling":
        a_p = 1.0 / factor

    obs: list[ReporterObservation] = []
    for j in range(n_reporters):
        gain = float(rng.uniform(*gain_range))
        floor = float(rng.uniform(*floor_range))
        readout = Readout(
            weight=jnp.array([[gain]], dtype=float),
            base=jnp.array([floor], dtype=float),
        )
        k_ctrl, k_pert = k_wt, k_p
        if hidden_latent_reporter is not None and j == hidden_latent_reporter:
            k_ctrl = k_pert = k_wt * 6.0  # secretly a DIFFERENT latent (shifted K)
        act_ctrl = jnp.asarray((1.0 * _frac(dose, k_ctrl, n_wt, direction))[:, None])
        act_pert = jnp.asarray((a_p * _frac(dose, k_pert, n_p, direction))[:, None])
        y_ctrl = np.asarray(readout.expression(act_ctrl)).ravel()
        y_pert = np.asarray(readout.expression(act_pert)).ravel()
        y_ctrl = y_ctrl * rng.lognormal(0.0, noise, size=y_ctrl.shape)
        y_pert = y_pert * rng.lognormal(0.0, noise, size=y_pert.shape)
        obs.append(
            ReporterObservation(
                name=f"R{j}", dose=dose, control=y_ctrl, perturbed=y_pert
            )
        )
    return obs
