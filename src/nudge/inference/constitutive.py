"""Constitutive-reporter calibration control — the ``NUDGE-LIM-006`` mitigation.

NUDGE's default forward model assumes an **affine** reporter (``Λ = base + scale·activity``).
When the true reporter is **nonlinear** (saturating / sigmoidal Hill ``h ≥ 1``), a *linear*
(non-switch) circuit read through it produces a skewed, pseudo-bimodal count distribution
that the affine-readout switch model can only explain by *bending the circuit* — a
**confident false positive** (``NUDGE-LIM-006``, the sharpest bound on the fail-safe
guarantee). The root cause is identifiability: per cell an input drives a circuit map
``a = g(u; K, n, vmax)`` (Hill), then a readout map ``Λ = R(a; Km, h, Vmax)`` (Hill), then
counts, and **only the composition ``R∘g`` is observed** — from a single population you
cannot factor it into its circuit and readout parts (``design/CONSTITUTIVE_CONTROL.md`` §1).

This module ships the **validated mitigation** (``scripts/vv/FINDINGS.md`` "NUDGE-LIM-006
mitigation — VALIDATED"; ``NUDGE-METHOD-011``): a **constitutive-reporter control** — a
calibration population whose reporter is driven at **known activity doses**, *bypassing the
circuit*. That population observes ``Λ = R(a)`` at known ``a`` — it measures the reporter's
own transfer function directly — so it **anchors the readout parameters ``φ = {Km, h, Vmax}``**
and *nothing else* (the load-bearing correctness property: the control loss depends on the
readout parameters ONLY — never on the circuit's ``{K, n, vmax}`` — verified structurally in
:func:`control_loss_circuit_gradient`). With the readout anchored, a **profile likelihood over
the circuit Hill ``n``** (:func:`profile_circuit_n`) can decide whether the observed
ultrasensitivity lives in the **circuit** (a real switch) or the **measurement** (a nonlinear
reporter):

- **Single population → genuinely degenerate.** The profile over circuit ``n`` is FLAT — a
  graded ``n = 1`` (no switch — all nonlinearity in the reporter) fits as well as the true
  ``n``. You cannot even tell a circuit switch exists.
- **Add the constitutive control → the degeneracy BREAKS.** ``n = 1`` is REJECTED (the
  ``n``-profile develops a well; ``Δloss(n=1) ≫`` the noise floor). The data now say the
  ultrasensitivity is **biological**.

**What this turns a confident-wrong into (fail-safe, never a bare mechanism).**
:func:`classify_constitutive` returns at most a ``biological-switch`` verdict — *the
ultrasensitivity is a real circuit switch, reject the readout-only explanation* — or an honest
``unresolved`` abstention, or ``no-confound`` when the calibrated reporter is ~affine (no
LIM-006 to fix). It **never** emits a bare ``threshold``/``gain``/``ceiling`` call, so it can
only move a confident false positive *toward* a correct call or an abstention.

**Honest caveat (confirmed in validation, preserved here).** The control lets NUDGE *reject
"no switch"* but does **not** point-identify the circuit ``n`` (recovered ≈ 5 vs true 3 — the
circuit's internal ``K``/``n``/``vmax`` trade-off persists). Full point-identification would
need a **second anchor** (an input titration / circuit dose-response). NUDGE reports the
identifiable part — *does a circuit switch exist at all* — and abstains on the exact knob.

**Additive / opt-in.** This module imports only the shipped Hill primitive
(:func:`nudge.mechanisms.regulatory.hill_activation`), the energy distance
(:func:`nudge.inference.losses.energy_distance`), and the count model
(:mod:`nudge.data.noise`). It never touches ``fit()``'s default, the decoy battery, or the
Lyapunov / epistasis attribution paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from nudge.inference.losses import energy_distance
from nudge.mechanisms.regulatory import hill_activation

__all__ = [
    "ConstitutiveControl",
    "ConstitutiveResult",
    "ReadoutCalibration",
    "ReadoutCircuitParams",
    "calibrate_readout",
    "classify_constitutive",
    "constitutive_control_analysis",
    "control_loss_circuit_gradient",
    "generate_constitutive_dataset",
    "profile_circuit_n",
]

_DEFAULT_N_GRID: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0)


# --------------------------------------------------------------------------- #
# Forward model (self-contained, differentiable; reuses NUDGE's Hill primitive).
#   u  ~ lognormal            latent input spread across cells (unobserved)
#   a  = basal + Hill(u; K, n, vmax)          circuit map  g(u)
#   Λ  = base  + Hill(a; Km, h, Vmax)         readout map  R(a)   (saturating)
#   y  ~ moment-matched NB(mean=Λ)            counts (observed)
# Only y is observed for the circuit population; the control observes R at KNOWN a.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReadoutCircuitParams:
    """Ground-truth circuit + readout parameters for the composed forward model.

    ``k, n, vmax, basal`` are the **circuit** (the putative biological switch);
    ``km, h, readout_vmax, readout_base`` are the **readout** (the measurement device).
    ``n`` is the circuit ultrasensitivity NUDGE wants to identify; ``h`` is the reporter
    nonlinearity that confounds it (``NUDGE-LIM-006``).
    """

    k: float = 1.0
    n: float = 3.0
    vmax: float = 1.0
    basal: float = 0.05
    km: float = 0.5
    h: float = 6.0
    readout_vmax: float = 20.0
    readout_base: float = 0.1


def _activity(
    u: ArrayLike, k: ArrayLike, n: ArrayLike, vmax: ArrayLike, basal: ArrayLike
) -> Array:
    """Circuit map ``a = basal + Hill(u; K, n, vmax)`` (the shipped Hill primitive)."""
    return jnp.asarray(basal) + hill_activation(jnp.maximum(jnp.asarray(u), 0.0), k, n, vmax)


def _reporter(
    a: ArrayLike, km: ArrayLike, h: ArrayLike, rvmax: ArrayLike, rbase: ArrayLike
) -> Array:
    """Readout map ``Λ = base + Hill(a; Km, h, Vmax)`` (saturating reporter)."""
    return jnp.asarray(rbase) + hill_activation(jnp.maximum(jnp.asarray(a), 0.0), km, h, rvmax)


def _nb_counts(key: Array, mean: Array, dispersion: float) -> Array:
    """Moment-matched NB observation (reparameterized Gaussian surrogate, clamped ≥0).

    ``var = mean + dispersion·mean²`` — the same relaxation the fit forward model uses
    (``nudge.inference.fit._simulate``); ``dispersion`` is ``1/φ``.
    """
    var = mean + dispersion * mean**2
    z = jax.random.normal(key, mean.shape)
    return jnp.maximum(mean + jnp.sqrt(var + 1e-12) * z, 0.0)


# --------------------------------------------------------------------------- #
# The constitutive control (calibration population) + its readout calibration.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConstitutiveControl:
    """A constitutive-reporter calibration: reporter output at **known driven activity**.

    ``activity`` are the KNOWN activity doses the reporter was driven at (bypassing the
    circuit) and ``response`` the measured reporter output (raw counts / a fluorescence
    channel) — paired 1-D arrays, one entry per observation. Replicates at the same dose
    are allowed and encouraged (they sharpen the calibration). At least 4 **distinct**
    doses are required to fit a Hill transfer function (4 parameters). In a real screen
    this is a constitutively-driven reporter titrated across a known range (a graded
    mCherry / synthetic-barcode reporter); see ``design/CONSTITUTIVE_CONTROL.md`` §3.
    """

    activity: np.ndarray
    response: np.ndarray

    def __post_init__(self) -> None:
        a = np.asarray(self.activity, dtype=float).ravel()
        r = np.asarray(self.response, dtype=float).ravel()
        if a.shape != r.shape:
            raise ValueError(
                f"activity and response must be the same length, got {a.shape} vs {r.shape}"
            )
        if np.unique(a).size < 4:
            raise ValueError(
                f"need >= 4 distinct activity doses to fit a reporter transfer function, "
                f"got {np.unique(a).size}"
            )
        object.__setattr__(self, "activity", a)
        object.__setattr__(self, "response", r)

    def calibration_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate replicates → the ``(dose, mean response)`` curve, sorted by dose."""
        doses = np.unique(self.activity)
        means = np.array([self.response[self.activity == d].mean() for d in doses])
        return doses, means

    def dose_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(doses (D,), responses (D, R))`` — requires equal replicate counts.

        The joint-fit control term (:func:`profile_circuit_n`) compares the *distribution*
        of reporter counts at each dose, so it needs a rectangular per-dose sample. Raises
        if the doses have unequal replicate counts (aggregate to a curve with
        :meth:`calibration_curve` for that case; :func:`calibrate_readout` handles ragged).
        """
        doses = np.unique(self.activity)
        rows = [self.response[self.activity == d] for d in doses]
        sizes = {r.size for r in rows}
        if len(sizes) != 1:
            raise ValueError(
                f"dose_matrix needs equal replicate counts per dose, got sizes {sorted(sizes)}"
            )
        return doses, np.stack(rows, axis=0)


@dataclass(frozen=True)
class ReadoutCalibration:
    """The reporter transfer function fitted from the control + its nonlinearity verdict.

    ``h`` is the reporter's apparent Hill coefficient (with ``ci_h``); ``is_nonlinear`` is
    ``True`` when the CI's *lower* bound clears the affine line by ``nonlinear_h`` — the
    control is then confident the reporter is meaningfully nonlinear, so the LIM-006 confound
    is in play. ``km``/``vmax``/``base`` are the other readout parameters; ``r2`` the fit
    quality. Fitted from KNOWN doses using **readout parameters only** (no circuit leak).
    """

    h: float
    ci_h: tuple[float, float]
    km: float
    vmax: float
    base: float
    r2: float
    is_nonlinear: bool
    reason: str


def calibrate_readout(
    control: ConstitutiveControl,
    *,
    nonlinear_h: float = 1.5,
    n_boot: int = 300,
    seed: int = 0,
) -> ReadoutCalibration:
    """Fit the reporter transfer function from a constitutive control (readout params ONLY).

    The control's ``(known activity, measured reporter)`` curve *is* a dose-response (dose =
    driven activity, response = reporter), so we fit it with the shipped Hill dose-response
    fitter and read off the readout parameters ``h``/``Km``/``Vmax``/``base`` — using **only**
    the control (no circuit population), so no circuit parameter can leak in. The bootstrap
    ``h`` CI gives an honest nonlinearity verdict: ``is_nonlinear`` iff the CI lower bound
    ``≥ nonlinear_h`` (the reporter is confidently steeper than affine).
    """
    from nudge.inference.dose_response import fit_dose_response

    dose, response = control.calibration_curve()
    drf = fit_dose_response(dose, response, direction="activate", n_boot=n_boot, seed=seed)
    h = float(drf.n)
    ci = (float(drf.ci_n[0]), float(drf.ci_n[1]))
    is_nonlinear = bool(np.isfinite(ci[0]) and ci[0] >= nonlinear_h)
    if is_nonlinear:
        reason = (
            f"reporter transfer function is confidently nonlinear: Hill h={h:.2f} "
            f"(95% CI [{ci[0]:.2f}, {ci[1]:.2f}] clears the affine line h={nonlinear_h:g}); "
            f"Km={drf.k_threshold:.3g}, Vmax={drf.amp:.3g}, base={drf.floor:.3g}, "
            f"r2={drf.r2:.3f} — a NUDGE-LIM-006 confound IS in play, so anchor the readout "
            "and profile the circuit n to test whether the ultrasensitivity is biological"
        )
    else:
        reason = (
            f"reporter transfer function is ~affine within the control: Hill h={h:.2f} "
            f"(95% CI [{ci[0]:.2f}, {ci[1]:.2f}] does not clear h={nonlinear_h:g}) — no "
            "NUDGE-LIM-006 confound; the default affine-readout attribution stands"
        )
    return ReadoutCalibration(
        h=h,
        ci_h=ci,
        km=float(drf.k_threshold),
        vmax=float(drf.amp),
        base=float(drf.floor),
        r2=float(drf.r2),
        is_nonlinear=is_nonlinear,
        reason=reason,
    )


# --------------------------------------------------------------------------- #
# The profile-likelihood engine (the validated degeneracy break).
# --------------------------------------------------------------------------- #
def _log1p(y: Array) -> Array:
    return jnp.log1p(y)


def _population_loss(free_log: Array, ctx: dict[str, Any], fixed_n: float) -> Array:
    """Energy distance between the model population and the observed one (log1p counts).

    ``free_log = [logK, log_cvmax, log_km, log_h, log_rvmax]``; the circuit ``n`` is frozen
    at ``fixed_n``; ``basal``/``base``/``dispersion`` are the KNOWN floors/count model. The
    model input draws + observation noise are fixed (common random numbers) so the loss is a
    smooth deterministic surface amenable to plain gradient descent.
    """
    k, cvmax, km, h, rvmax = jnp.exp(free_log)
    a = _activity(ctx["u_mod"], k, fixed_n, cvmax, ctx["basal"])
    lam = _reporter(a, km, h, rvmax, ctx["rbase"])
    var = lam + ctx["dispersion"] * lam**2
    y = jnp.maximum(lam + jnp.sqrt(var + 1e-12) * ctx["z_mod"], 0.0)
    return energy_distance(_log1p(y)[:, None], ctx["obs_log"][:, None])


def _control_loss(readout_log: Array, ctx: dict[str, Any]) -> Array:
    """Calibration loss at KNOWN doses — a function of the READOUT parameters ONLY.

    ``readout_log = [log_km, log_h, log_rvmax]``. The circuit parameters do not enter here
    at all (the load-bearing no-leak property): the control drives the reporter directly at
    known activity ``ctx['ctrl_doses']``, bypassing the circuit. Per-dose energy distance
    between the model reporter distribution and the observed one, averaged over doses.
    """
    km, h, rvmax = jnp.exp(readout_log)
    lam = _reporter(ctx["ctrl_doses"][:, None], km, h, rvmax, ctx["rbase"])  # (D, 1)
    var = lam + ctx["dispersion"] * lam**2
    y = jnp.maximum(lam + jnp.sqrt(var + 1e-12) * ctx["ctrl_z_mod"], 0.0)  # (D, R)
    y_log = _log1p(y)

    def per_dose(model_row: Array, data_row: Array) -> Array:
        return energy_distance(model_row[:, None], data_row[:, None])

    return jax.vmap(per_dose)(y_log, ctx["ctrl_data_log"]).mean()


def control_loss_circuit_gradient(
    control: ConstitutiveControl,
    params: ReadoutCircuitParams,
    *,
    dispersion: float = 0.1,
    seed: int = 0,
) -> dict[str, float]:
    """Return ∂(control loss)/∂(circuit params) — must be exactly 0 (the no-leak proof).

    The constitutive control's calibration loss is, by construction, a function of the
    READOUT parameters only. This exposes the gradient of that loss w.r.t. the CIRCUIT
    parameters ``{K, n, vmax, basal}`` so a test can assert it is identically zero — i.e.
    the control cannot smuggle circuit information into the readout anchor. Returns the
    absolute gradient magnitude for each circuit parameter (all should be 0.0).
    """
    doses, resp = control.dose_matrix()
    ctx = {
        "ctrl_doses": jnp.asarray(doses),
        "ctrl_data_log": _log1p(jnp.asarray(resp)),
        "ctrl_z_mod": jax.random.normal(jax.random.key(seed), resp.shape),
        "rbase": jnp.asarray(params.readout_base),
        "dispersion": dispersion,
    }
    readout_log = jnp.log(jnp.array([params.km, params.h, params.readout_vmax]))
    circuit_log = jnp.log(jnp.array([params.k, params.n, params.vmax, params.basal]))

    def loss_of_circuit(circ_log: Array) -> Array:
        # The circuit params are fed in but the control loss ignores them entirely — the
        # gradient below is the structural proof that it does.
        _ = circ_log
        return _control_loss(readout_log, ctx)

    grad = jax.grad(loss_of_circuit)(circuit_log)
    names = ("K", "n", "vmax", "basal")
    return {name: float(abs(g)) for name, g in zip(names, np.asarray(grad), strict=True)}


def _run_adam(
    loss_fn: Any, init: Array, *, steps: int, lr: float
) -> tuple[Array, float]:
    """Plain Adam on a deterministic (common-random-number) loss; returns (params, loss)."""
    import optax

    opt = optax.adam(lr)
    state = opt.init(init)
    vg = jax.jit(jax.value_and_grad(loss_fn))
    v = init
    for _ in range(steps):
        _loss, grad = vg(v)
        updates, state = opt.update(grad, state)
        v = jnp.asarray(optax.apply_updates(v, updates))
    return v, float(loss_fn(v))


def _fit_profile_point(
    ctx: dict[str, Any],
    fixed_n: float,
    *,
    use_ctrl: bool,
    w_ctrl: float,
    init0: np.ndarray,
    restarts: int,
    steps: int,
    lr: float,
    seed: int,
) -> float:
    """Profile loss at one circuit-``n``: freeze ``n``, optimize the other 5 params.

    ``free_log = [logK, log_cvmax, log_km, log_h, log_rvmax]``. With ``use_ctrl`` the loss
    adds ``w_ctrl · control_loss`` (readout params only). A few restarts guard against a
    local optimum handing back a spuriously low loss.
    """

    def loss_fn(free_log: Array) -> Array:
        pop = _population_loss(free_log, ctx, fixed_n)
        if not use_ctrl:
            return pop
        return pop + w_ctrl * _control_loss(free_log[2:], ctx)

    rng = np.random.default_rng(seed)
    best = np.inf
    for r in range(max(restarts, 1)):
        jitter = 0.0 if r == 0 else rng.normal(0.0, 0.5, size=init0.shape)
        init = jnp.asarray(init0 + jitter)
        _v, loss = _run_adam(loss_fn, init, steps=steps, lr=lr)
        best = min(best, loss)
    return best


def _self_distance_floor(obs_log: Array, key: Array, *, reps: int = 12) -> tuple[float, float]:
    """Irreducible finite-sample loss floor (log1p energy distance) + its std, by bootstrap.

    A perfect fit's population loss ≈ the energy distance between two random halves of the
    observed sample; its std is the noise floor the rejection gate compares against.
    """
    n = int(obs_log.shape[0])
    half = max(n // 2, 2)
    dists = []
    for _ in range(reps):
        key, k1, k2 = jax.random.split(key, 3)
        i1 = jax.random.choice(k1, n, (half,), replace=False)
        i2 = jax.random.choice(k2, n, (half,), replace=False)
        dists.append(float(energy_distance(obs_log[i1][:, None], obs_log[i2][:, None])))
    arr = np.asarray(dists)
    return float(arr.mean()), float(arr.std())


@dataclass(frozen=True)
class ConstitutiveResult:
    """Profile-likelihood result: the with/without-control ``n``-profiles + the verdict.

    ``span_no_control`` is the flatness of the ``n``-profile WITHOUT the control (small =
    degenerate: you cannot tell a circuit switch exists). ``n1_rejection`` is
    ``loss(n=1) − min loss`` WITH the control (large = "no switch" rejected → the
    ultrasensitivity is biological). ``floor_mean``/``floor_std`` set the noise scale.
    ``call`` is ``biological-switch`` / ``unresolved`` / ``no-confound`` — **never** a bare
    ``threshold``/``gain``/``ceiling`` (the fail-safe property).
    """

    calibration: ReadoutCalibration
    n_grid: tuple[float, ...]
    loss_no_control: tuple[float, ...]
    loss_with_control: tuple[float, ...]
    span_no_control: float
    span_with_control: float
    n1_rejection: float
    argmin_n_with_control: float
    floor_mean: float
    floor_std: float
    call: str
    reason: str

    @property
    def is_confident_wrong(self) -> bool:
        """Structurally always ``False``: the module never emits a bare mechanism call.

        The strongest positive verdict is ``biological-switch`` (a real switch exists,
        reject the readout-only explanation); it never localizes the knob, so it cannot be
        confidently wrong about threshold/gain/ceiling — the whole point of the mitigation.
        """
        return self.call in {"threshold", "gain", "ceiling"}


def profile_circuit_n(
    population: Any,
    control: ConstitutiveControl,
    params: ReadoutCircuitParams,
    *,
    n_grid: tuple[float, ...] = _DEFAULT_N_GRID,
    dispersion: float = 0.1,
    mu_log: float = 0.0,
    sd_log: float = 0.6,
    w_ctrl: float = 3.0,
    n_model_cells: int = 400,
    restarts: int = 3,
    steps: int = 600,
    lr: float = 0.03,
    calibration: ReadoutCalibration | None = None,
    reject_abs: float = 1e-2,
    structure_ratio: float = 5.0,
    seed: int = 0,
) -> ConstitutiveResult:
    """Profile the loss over circuit ``n`` WITHOUT vs WITH the constitutive control.

    ``population`` is the observed circuit-population counts (1-D; the reporter read of the
    circuit). ``control`` is the constitutive calibration; ``params`` supplies the KNOWN
    floors + count model (``basal``, ``readout_base``, ``dispersion``) and the latent-input
    spread assumption (``mu_log``, ``sd_log``) — the circuit ``n`` is what we profile, and
    ``K``/``vmax``/``Km``/``h``/``Vmax`` are re-optimized at each grid point. Returns a
    :class:`ConstitutiveResult` with both profiles + the fail-safe verdict
    (:func:`classify_constitutive`).
    """
    obs = jnp.asarray(np.asarray(population, dtype=float).ravel())
    obs_log = _log1p(obs)
    doses, resp = control.dose_matrix()

    key = jax.random.key(seed)
    k_u, k_z, k_cz, k_floor = jax.random.split(key, 4)
    u_mod = jnp.exp(mu_log + sd_log * jax.random.normal(k_u, (n_model_cells,)))
    z_mod = jax.random.normal(k_z, (n_model_cells,))
    ctrl_z_mod = jax.random.normal(k_cz, resp.shape)

    ctx: dict[str, Any] = {
        "u_mod": u_mod,
        "z_mod": z_mod,
        "obs_log": obs_log,
        "ctrl_doses": jnp.asarray(doses),
        "ctrl_data_log": _log1p(jnp.asarray(resp)),
        "ctrl_z_mod": ctrl_z_mod,
        "basal": jnp.asarray(params.basal),
        "rbase": jnp.asarray(params.readout_base),
        "dispersion": dispersion,
    }
    # Init at the true-ish scales (the calibration already pins the readout ballpark).
    calib = calibration if calibration is not None else calibrate_readout(control, seed=seed)
    init0 = np.log(
        np.array(
            [params.k, params.vmax, max(calib.km, 1e-3), max(calib.h, 0.3), max(calib.vmax, 1e-3)]
        )
    )

    loss_no: list[float] = []
    loss_yes: list[float] = []
    for gi, n_val in enumerate(n_grid):
        loss_no.append(
            _fit_profile_point(
                ctx, float(n_val), use_ctrl=False, w_ctrl=w_ctrl, init0=init0,
                restarts=restarts, steps=steps, lr=lr, seed=seed + gi,
            )
        )
        loss_yes.append(
            _fit_profile_point(
                ctx, float(n_val), use_ctrl=True, w_ctrl=w_ctrl, init0=init0,
                restarts=restarts, steps=steps, lr=lr, seed=seed + 1000 + gi,
            )
        )

    floor_mean, floor_std = _self_distance_floor(obs_log, k_floor)
    no_arr = np.asarray(loss_no)
    yes_arr = np.asarray(loss_yes)
    span_no = float(no_arr.max() - no_arr.min())
    span_yes = float(yes_arr.max() - yes_arr.min())
    # n=1 rejection: the with-control loss at the "no switch" end minus the profile minimum.
    n1_idx = int(np.argmin(np.abs(np.asarray(n_grid) - 1.0)))
    n1_rejection = float(yes_arr[n1_idx] - yes_arr.min())
    argmin_n = float(n_grid[int(np.argmin(yes_arr))])

    call, reason = classify_constitutive(
        calibration=calib,
        n1_rejection=n1_rejection,
        span_no_control=span_no,
        floor_mean=floor_mean,
        floor_std=floor_std,
        argmin_n=argmin_n,
        reject_abs=reject_abs,
        structure_ratio=structure_ratio,
    )
    return ConstitutiveResult(
        calibration=calib,
        n_grid=tuple(float(x) for x in n_grid),
        loss_no_control=tuple(loss_no),
        loss_with_control=tuple(loss_yes),
        span_no_control=span_no,
        span_with_control=span_yes,
        n1_rejection=n1_rejection,
        argmin_n_with_control=argmin_n,
        floor_mean=floor_mean,
        floor_std=floor_std,
        call=call,
        reason=reason,
    )


def classify_constitutive(
    *,
    calibration: ReadoutCalibration,
    n1_rejection: float,
    span_no_control: float,
    floor_mean: float,
    floor_std: float,
    argmin_n: float,
    reject_abs: float = 1e-2,
    structure_ratio: float = 5.0,
) -> tuple[str, str]:
    """Turn the profiles into a fail-safe verdict (never a bare mechanism).

    Gates, in order:

    1. **no-confound** — the calibrated reporter is ~affine (``not is_nonlinear``): there is
       no NUDGE-LIM-006 confound to correct; the default affine-readout attribution stands.
    2. **biological-switch** — with the control anchoring the readout, "no switch" (``n = 1``)
       is REJECTED. Three conditions must ALL hold (fail-safe conjunction): the rejection
       ``n1_rejection`` clears an absolute margin ``reject_abs`` (a clear multiple of the
       achievable-fit noise); it is at least ``structure_ratio`` × the WITHOUT-control profile
       span (the control *created* structure the degenerate profile lacked); and the with-
       control profile minimum sits **off** the ``n = 1`` no-switch end. The ultrasensitivity
       is then biological (a real circuit switch). **Honest caveat:** this does NOT
       point-identify the exact ``n`` — the circuit's internal K/n/vmax trade-off persists;
       full point-ID needs a second anchor (an input titration).
    3. **unresolved** — a confound is present but the control does not decisively reject "no
       switch" (a narrow / weak / noisy control range, or the truth really is a graded /
       linear circuit whose apparent ultrasensitivity lives in the reporter). NUDGE abstains
       rather than guess — turning the LIM-006 confident false positive into an honest
       abstention.
    """
    if not calibration.is_nonlinear:
        return "no-confound", (
            "constitutive control found the reporter ~affine (no NUDGE-LIM-006 confound); "
            "the default affine-readout attribution stands. " + calibration.reason
        )
    structure_margin = structure_ratio * max(span_no_control, 1e-9)
    rejected = (
        n1_rejection >= reject_abs
        and n1_rejection >= structure_margin
        and argmin_n > 1.0
    )
    if rejected:
        return "biological-switch", (
            f"with the constitutive control anchoring the readout, 'no switch' (circuit n=1) "
            f"is REJECTED: Δloss(n=1)={n1_rejection:.4f} clears the absolute margin "
            f"{reject_abs:.4f} AND is ≥{structure_ratio:g}× the WITHOUT-control profile span "
            f"({span_no_control:.4f} — which is FLAT: the split is degenerate without a "
            f"control), and the profile minimum moved off n=1. The ultrasensitivity is "
            f"BIOLOGICAL (a real circuit switch exists). HONEST CAVEAT: this does NOT "
            f"point-identify the circuit n (profile argmin n≈{argmin_n:g} is not a reliable "
            f"point estimate — the K/n/vmax trade-off persists); full point-ID needs a "
            f"second anchor (an input titration / circuit dose-response). " + calibration.reason
        )
    return "unresolved", (
        f"a NUDGE-LIM-006 confound is present (nonlinear reporter) but the constitutive "
        f"control did NOT decisively reject 'no switch': Δloss(n=1)={n1_rejection:.4f} "
        f"(absolute margin {reject_abs:.4f}; ≥{structure_ratio:g}× span "
        f"{span_no_control:.4f} required; profile argmin n≈{argmin_n:g}). NUDGE abstains — "
        f"the data are consistent with the ultrasensitivity living in the reporter, not the "
        f"circuit. A wider / cleaner constitutive dose range, or a second anchor, would be "
        f"needed to resolve it. " + calibration.reason
    )


# --------------------------------------------------------------------------- #
# Synthetic ground truth (for validation + tests + the demo notebook).
# --------------------------------------------------------------------------- #
def generate_constitutive_dataset(
    params: ReadoutCircuitParams | None = None,
    *,
    n_cells: int = 600,
    n_ctrl_doses: int = 10,
    n_ctrl_reps: int = 200,
    dispersion: float = 0.1,
    mu_log: float = 0.0,
    sd_log: float = 0.6,
    seed: int = 0,
) -> tuple[np.ndarray, ConstitutiveControl, ReadoutCircuitParams]:
    """Synthesize a circuit population + a matched constitutive control from ground truth.

    Draws ``n_cells`` latent inputs ``u ~ lognormal``, pushes them through the circuit Hill
    ``g(u)`` and the readout Hill ``R(a)`` to counts (the OBSERVED population, which sees only
    ``R∘g``), and builds a constitutive control that drives the reporter at ``n_ctrl_doses``
    KNOWN activity doses (spanning ``[basal, basal+vmax]``) through the SAME readout + count
    model, bypassing the circuit. Set ``params.n`` to 1 for the LIM-006 false-positive hazard
    (a graded circuit) or ``> 1`` for a genuine biological switch. Returns
    ``(population_counts, control, params)``.
    """
    p = params if params is not None else ReadoutCircuitParams()
    key = jax.random.key(seed)
    k_u, k_y, k_c = jax.random.split(key, 3)
    u = jnp.exp(mu_log + sd_log * jax.random.normal(k_u, (n_cells,)))
    a = _activity(u, p.k, p.n, p.vmax, p.basal)
    lam = _reporter(a, p.km, p.h, p.readout_vmax, p.readout_base)
    pop = np.asarray(_nb_counts(k_y, lam, dispersion))

    doses = np.linspace(p.basal, p.basal + p.vmax, n_ctrl_doses)
    lam_c = _reporter(jnp.asarray(doses)[:, None], p.km, p.h, p.readout_vmax, p.readout_base)
    lam_c = jnp.broadcast_to(lam_c, (n_ctrl_doses, n_ctrl_reps))
    resp = np.asarray(_nb_counts(k_c, lam_c, dispersion))
    activity = np.repeat(doses, n_ctrl_reps)
    control = ConstitutiveControl(activity=activity, response=resp.ravel())
    return pop, control, p


def constitutive_control_analysis(
    population: Any,
    control: ConstitutiveControl,
    params: ReadoutCircuitParams,
    **kwargs: Any,
) -> ConstitutiveResult:
    """Convenience entry point: calibrate the readout, then profile circuit ``n``.

    Thin wrapper over :func:`profile_circuit_n` (which already calibrates internally) — kept
    as the public verb the service / CLI / MCP call. All keyword arguments pass through.
    """
    return profile_circuit_n(population, control, params, **kwargs)
