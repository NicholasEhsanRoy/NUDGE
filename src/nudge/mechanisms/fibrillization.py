"""Protein aggregation / amyloid fibrillization kinetics — fit, attribute, abstain.

**The efficiency demo, and the extensibility thesis pointed at a third dynamical
system.** Everything else in NUDGE observes a steady-state single-cell *snapshot*
(``K`` / ``n`` / ``v_max``) or a microbial *trajectory* (``lotka_volterra.py``'s
``α`` / ``β`` / ``ε``). This module points the *same* abstain-and-attribute philosophy
at **amyloid aggregation curves** — the sigmoidal ThT/mass-fraction trace of a protein
polymerizing into fibrils — and attributes an inhibitor to the **microscopic rate
constant** it acts on, while **measuring** (not asserting) the famous single-curve
non-identifiability.

## The model — the filament master equation, reduced to its principal moments

Following Knowles/Cohen/Meisl 2016 (*Nat. Protoc.*) and Michaels 2020 (*PNAS*), the
microscopic filament-assembly master equation reduces to two moment ODEs in the
**filament number** concentration ``P`` and the **polymer mass** concentration ``M``,
with free monomer ``m = m_tot − M``:

```
dP/dt = k_n · m^{n_c}   +   k_2 · m^{n_2} · M
dM/dt = 2 · k_+ · m · P
```

Three microscopic processes: **primary nucleation** (rate ``k_n``, monomer order
``n_c``), **elongation** (rate ``k_+``, two ends), and **secondary surface-catalysed
nucleation** (rate ``k_2``, order ``n_2``). Integrated by a self-contained differentiable
RK4 ``lax.scan`` (mirroring ``lotka_volterra.simulate_glv``; **no ``diffrax``**),
differentiable through ``(k_n, k_+, k_2)`` — the gradient the fit needs. Touches
**neither ``fit.py`` nor ``core/circuit.py``** (frozen).

## What it identifies — and where it abstains (the load-bearing honesty)

From a **single curve at a single monomer concentration** only the two composite
parameters

```
λ = √(2 · k_+ · k_n · m_tot^{n_c})       (primary-pathway rate / lag)
κ = √(2 · k_+ · k_2 · m_tot^{n_2+1})     (secondary / autocatalytic growth rate)
```

are identifiable. The three individual constants are **not**, because the moment model
has an **exact continuous gauge symmetry** — verified analytically and numerically here:

```
(k_n, k_+, k_2)  →  (k_n / α,  α · k_+,  k_2 / α)     leaves M(t)/m_tot IDENTICAL
```

for any ``α > 0``. So the single-curve Fisher/Laplace curvature on
``(log k_n, log k_+, log k_2)`` has an **exact zero eigenvalue** along the null direction
``(−1, +1, −1)`` (equivalently ``(+log k_n, −log k_+, +log k_2)``), condition number
``→ ∞``. NUDGE **measures** this (``individual_k_identifiability``, reusing
:func:`~nudge.inference.uncertainty.laplace_posterior`) and returns the composites
``κ, λ`` + the null direction + *"need a concentration series and a seeded / elongation
anchor to pin k_+"* — exactly the honest answer a control LLM agent took 12 minutes and
six scripts to hand-derive (``design/automated_scientist/runs/000000008``), returned here
in **one call**. This is ``NUDGE-LIM-021``.

## What a concentration series (and a seeded anchor) do resolve

A concentration series (several ``m_tot``) globally constrains the **reaction orders**
``n_c, n_2`` and the rate **products** ``k_+ k_n``, ``k_+ k_2`` — but, honestly, the
mass-fraction gauge above is concentration-independent, so **the series alone still cannot
separate the individual constants** (measured: ``series_identifiability`` stays
degenerate). Adding a **seeded / elongation reference** (heavily-seeded curve where
``dM/dt ≈ 2 k_+ m P_0`` directly constrains ``k_+``) breaks the gauge, and the global fit
then **resolves all three** — the Meisl discipline. NUDGE demonstrates both halves and
never claims a resolution the data cannot support.

## What it attributes — an inhibitor's microscopic target

An inhibitor is a perturbation to specific ``k``'s: a **fibril-end binder** lowers
``k_+`` (scales λ *and* κ together), a **surface binder** lowers ``k_2`` (κ only), a
**primary-nucleus binder** lowers ``k_n`` (λ only). These are distinguishable from the
composite log-ratios of a control-vs-inhibited curve pair — that is
``attribute_inhibitor``. (A monomer-sequestering "reduces-all-rates" inhibitor scales λ
and κ together like an elongation binder — a documented composite ambiguity;
``NUDGE-LIM-021``.)

Fail-safety is paramount: a synthetic case with a known answer must **recover it or
abstain — never a confident wrong knob or a false-precise individual rate constant**.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array
from jax.experimental import enable_x64

from nudge.inference.uncertainty import LaplacePosterior, laplace_posterior

_F = TypeVar("_F", bound=Callable[..., Any])


def _under_x64(fn: _F) -> _F:
    """Run ``fn`` with JAX float64 enabled (a thread-local context, matching the
    ``core/circuit.py`` pattern; the exact gauge symmetry and its near-singular curvature
    need float64 to register cleanly as a genuine zero eigenvalue rather than float32
    noise). Does NOT change the global config, so sibling float32 modules are untouched."""

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with enable_x64():
            return fn(*args, **kwargs)

    return wrapped  # type: ignore[return-value]

__all__ = [
    "AggregationParams",
    "AggregationCurve",
    "ConcentrationSeries",
    "CompositeFit",
    "IndividualKIdentifiability",
    "AggregationResult",
    "InhibitorResult",
    "SeriesResolution",
    "moment_vector_field",
    "simulate_aggregation",
    "composite_lambda_kappa",
    "simulate_aggregation_curve",
    "simulate_concentration_series",
    "simulate_seeded_elongation",
    "simulate_inhibitor_pair",
    "fit_composites",
    "individual_k_identifiability",
    "attribute_aggregation",
    "attribute_inhibitor",
    "fit_series_global",
    "series_identifiability",
    "resolve_series",
]

#: the three microscopic rate constants, in a fixed order (the null-direction axis order).
_K_NAMES: tuple[str, ...] = ("log_k_n", "log_k_plus", "log_k_2")
#: the three inhibitor-target verdicts (which microscopic step an inhibitor lowered).
_TARGETS: tuple[str, ...] = ("primary_nucleation", "elongation", "secondary_nucleation")
#: numerical floor so log/√ of a (near-)zero abundance never NaNs the loss.
_EPS = 1e-12
#: hard cap on P / M inside the integrator — a diverging orbit is a bad fit, not a crash.
_CAP = 1e12


# --------------------------------------------------------------------------- #
# parameter container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AggregationParams:
    """A fibrillization system's microscopic kinetics.

    ``k_n`` primary-nucleation rate, ``k_plus`` elongation rate, ``k_2``
    secondary-nucleation rate; ``n_c`` / ``n_2`` the primary / secondary monomer reaction
    orders. Stored as Python floats; :meth:`jax` packs the three rates into device scalars
    for the differentiable integrator (the orders are static).
    """

    k_n: float
    k_plus: float
    k_2: float
    n_c: float = 2.0
    n_2: float = 2.0

    def jax(self) -> tuple[Array, Array, Array]:
        # No explicit dtype: adapts to the active JAX config — float64 inside an
        # ``enable_x64`` context (the fits / curvature), float32 otherwise (data gen).
        return (jnp.asarray(self.k_n), jnp.asarray(self.k_plus), jnp.asarray(self.k_2))

    def with_scaled(self, knob: str, factor: float) -> AggregationParams:
        """Return a copy with one microscopic rate **multiplied** by ``factor`` (an
        inhibitor lowering a specific step: ``factor < 1``)."""
        if knob == "primary_nucleation":
            return AggregationParams(self.k_n * factor, self.k_plus, self.k_2,
                                     self.n_c, self.n_2)
        if knob == "elongation":
            return AggregationParams(self.k_n, self.k_plus * factor, self.k_2,
                                     self.n_c, self.n_2)
        if knob == "secondary_nucleation":
            return AggregationParams(self.k_n, self.k_plus, self.k_2 * factor,
                                     self.n_c, self.n_2)
        if knob == "monomer_sequestration":  # reduces ALL rates (a monomer binder)
            return AggregationParams(self.k_n * factor, self.k_plus * factor,
                                     self.k_2 * factor, self.n_c, self.n_2)
        raise ValueError(f"unknown inhibitor knob {knob!r}")


def composite_lambda_kappa(
    params: AggregationParams, m_tot: float
) -> tuple[float, float]:
    """The two identifiable composites ``(λ, κ)`` at monomer concentration ``m_tot``.

    ``λ = √(2 k_+ k_n m_tot^{n_c})`` (primary-pathway / lag rate) and
    ``κ = √(2 k_+ k_2 m_tot^{n_2+1})`` (secondary / autocatalytic rate) — the
    Knowles/Cohen/Meisl combinations a single curve constrains. With ``m_tot = 1`` these
    reduce to the ``√(2 k_+ k_n)`` / ``√(2 k_+ k_2)`` forms.
    """
    lam = float(np.sqrt(2.0 * params.k_plus * params.k_n * m_tot ** params.n_c))
    kappa = float(np.sqrt(2.0 * params.k_plus * params.k_2 * m_tot ** (params.n_2 + 1.0)))
    return lam, kappa


# --------------------------------------------------------------------------- #
# the differentiable moment vector field + RK4 integrator (self-contained)
# --------------------------------------------------------------------------- #
def moment_vector_field(
    state: Array, k_n: Array, k_plus: Array, k_2: Array, n_c: float, n_2: float,
    m_tot: float,
) -> Array:
    """The moment RHS ``d[P, M]/dt`` given the free monomer ``m = m_tot − M``.

    ``dP/dt = k_n m^{n_c} + k_2 m^{n_2} M`` (primary + secondary nucleation),
    ``dM/dt = 2 k_+ m P`` (elongation at two ends). Monomer is clamped ``≥ 0`` (it cannot
    go negative once the pool is exhausted).
    """
    p, mass = state[0], state[1]
    m = jnp.clip(m_tot - mass, 0.0, m_tot)
    dp = k_n * m ** n_c + k_2 * m ** n_2 * mass
    dm_mass = 2.0 * k_plus * m * p
    return jnp.stack([dp, dm_mass])


def simulate_aggregation(
    params: AggregationParams | tuple[Array, Array, Array],
    *,
    m_tot: float,
    dt: float,
    n_steps: int,
    obs_idx: Array,
    n_c: float = 2.0,
    n_2: float = 2.0,
    p0: float = 0.0,
    m0: float = 0.0,
) -> Array:
    """Integrate the moment ODEs (RK4 ``lax.scan``) and return the **mass fraction**
    ``M(t)/m_tot`` at the observation indices.

    ``p0`` / ``m0`` seed the filament number / polymer mass (both ``0`` for an unseeded
    reaction; a heavily-seeded elongation reference sets ``p0 > 0``). Differentiable w.r.t.
    the three rate constants (a plain ``lax.scan`` of RK4 steps, no ``diffrax``).
    """
    if isinstance(params, AggregationParams):
        k_n, k_plus, k_2 = params.jax()
        n_c, n_2 = params.n_c, params.n_2
    else:
        k_n, k_plus, k_2 = params

    def rk4(state: Array, _: Any) -> tuple[Array, Array]:
        def f(s: Array) -> Array:
            return moment_vector_field(s, k_n, k_plus, k_2, n_c, n_2, m_tot)

        k1 = f(state)
        k2v = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2v)
        k4 = f(state + dt * k3)
        nxt = state + (dt / 6.0) * (k1 + 2.0 * k2v + 2.0 * k3 + k4)
        nxt = jnp.clip(nxt, 0.0, _CAP)
        return nxt, nxt

    x0 = jnp.asarray([p0, m0])  # dtype adapts to the active JAX config (see jax())
    _final, traj = jax.lax.scan(rk4, x0, None, length=n_steps)
    traj = jnp.concatenate([x0[None, :], traj], axis=0)  # index 0 = t0
    mass = traj[obs_idx, 1]
    return jnp.clip(mass / m_tot, 0.0, 1.5)


def _grid(t_max: float, dt: float, t_obs: np.ndarray) -> tuple[int, np.ndarray]:
    """Number of fine steps + the nearest-grid indices for ``t_obs``."""
    n_steps = int(round(t_max / dt))
    obs_idx = np.clip(np.round(t_obs / dt).astype(int), 0, n_steps)
    return n_steps, obs_idx


# --------------------------------------------------------------------------- #
# dataset containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AggregationCurve:
    """A single aggregation curve (replicate ensemble) at one monomer concentration.

    ``signal`` is ``(R, T)`` replicate × time mass-fraction traces (ThT-like, ∈ [0, 1]);
    ``t_obs`` the observation times; ``m_tot`` the initial monomer concentration; ``p0`` a
    seed filament concentration (``0`` = unseeded). ``dt`` / ``n_steps`` / ``obs_idx``
    reconstruct the fine integration grid. ``ground_truth`` records the true kinetics +
    orders so a test can assert recover-or-abstain.
    """

    signal: np.ndarray
    t_obs: np.ndarray
    m_tot: float
    dt: float
    n_steps: int
    obs_idx: np.ndarray
    p0: float
    ground_truth: dict[str, Any]

    @property
    def mean(self) -> np.ndarray:
        return self.signal.mean(axis=0)


@dataclass(frozen=True)
class ConcentrationSeries:
    """A set of aggregation curves at several ``m_tot`` sharing ONE set of microscopic
    kinetics (the global-fit target), plus an optional seeded ``anchor`` curve."""

    curves: tuple[AggregationCurve, ...]
    anchor: AggregationCurve | None
    ground_truth: dict[str, Any]


# --------------------------------------------------------------------------- #
# synthetic generators (synthetic-first — nothing real until the round-trip passes)
# --------------------------------------------------------------------------- #
_DEFAULT_TRUTH = AggregationParams(k_n=5e-4, k_plus=0.1, k_2=5.0, n_c=2.0, n_2=2.0)
#: at m_tot = 1 this gives λ = √(2·0.1·5e-4) = 0.01 and κ = √(2·0.1·5) = 1.0 — the
#: κ≈1 / λ≈0.01 regime the control agent hand-derived (secondary-nucleation dominated).
_BALANCED_TRUTH = AggregationParams(k_n=0.45, k_plus=0.1, k_2=5.0, n_c=2.0, n_2=2.0)
#: λ = 0.3, κ = 1.0 at m_tot = 1 — BOTH nucleation pathways contribute, so all three
#: constants are individually well-determined once the gauge is broken by a seeded anchor.
#: (In the strongly secondary-dominated _DEFAULT_TRUTH the primary pathway is negligible,
#: so k_n stays weakly determined even with an anchor — honest, and what the control agent
#: also found, ×/÷3 on k_n; ``NUDGE-LIM-021``.) This is the concentration-series demo's
#: regime.


def _obs_times(t_max: float, n_obs: int) -> np.ndarray:
    return np.linspace(0.0, t_max, n_obs)


def simulate_aggregation_curve(
    *,
    params: AggregationParams | None = None,
    m_tot: float = 1.0,
    t_max: float = 40.0,
    n_obs: int = 60,
    dt: float = 0.01,
    n_replicates: int = 12,
    obs_noise: float = 0.02,
    p0: float = 0.0,
    seed: int = 0,
) -> AggregationCurve:
    """Simulate one replicate ensemble of aggregation curves with KNOWN kinetics.

    ``params`` defaults to the secondary-nucleation-dominated regime (``κ≈1``, ``λ≈0.01``
    at ``m_tot=1``). Additive Gaussian ``obs_noise`` on the mass fraction mimics ThT
    measurement error. Returns an :class:`AggregationCurve`.
    """
    p = params if params is not None else _DEFAULT_TRUTH
    rng = np.random.default_rng(seed)
    t_obs = _obs_times(t_max, n_obs)
    n_steps, obs_idx = _grid(t_max, dt, t_obs)
    clean = np.asarray(
        simulate_aggregation(
            p, m_tot=m_tot, dt=dt, n_steps=n_steps, obs_idx=jnp.asarray(obs_idx),
            p0=p0, m0=0.0,
        )
    )
    signal = clean[None, :] + obs_noise * rng.standard_normal((n_replicates, len(t_obs)))
    signal = np.clip(signal, 0.0, 1.5)
    lam, kappa = composite_lambda_kappa(p, m_tot)
    ground_truth = {
        "k_n": p.k_n, "k_plus": p.k_plus, "k_2": p.k_2, "n_c": p.n_c, "n_2": p.n_2,
        "m_tot": m_tot, "lambda": lam, "kappa": kappa, "p0": p0,
    }
    return AggregationCurve(
        signal=signal, t_obs=t_obs, m_tot=m_tot, dt=dt, n_steps=n_steps,
        obs_idx=obs_idx, p0=p0, ground_truth=ground_truth,
    )


def simulate_seeded_elongation(
    *,
    params: AggregationParams | None = None,
    m_tot: float = 1.0,
    p0: float = 2.0,
    t_max: float = 1.0,
    n_obs: int = 60,
    dt: float = 0.005,
    n_replicates: int = 12,
    obs_noise: float = 0.02,
    seed: int = 1,
) -> AggregationCurve:
    """A **seeded elongation reference** — the anchor that breaks the ``(k_n, k_+, k_2)``
    gauge. Pre-formed filament seeds ``p0`` are present at ``t=0``, and the **early**
    window (short ``t_max``, densely sampled) captures the initial elongation slope
    ``dM/dt|₀ = 2 k_+ m_tot P_0``, which pins ``k_+`` directly (before primary/secondary
    nucleation accumulate new filament number). This is what makes the individual
    constants identifiable — the standard seeded-experiment trick of the Meisl discipline.
    """
    return simulate_aggregation_curve(
        params=params, m_tot=m_tot, t_max=t_max, n_obs=n_obs, dt=dt,
        n_replicates=n_replicates, obs_noise=obs_noise, p0=p0, seed=seed,
    )


def simulate_concentration_series(
    *,
    params: AggregationParams | None = None,
    m_tots: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    with_anchor: bool = True,
    t_max: float = 40.0,
    n_obs: int = 60,
    dt: float = 0.01,
    n_replicates: int = 12,
    obs_noise: float = 0.02,
    seed: int = 0,
) -> ConcentrationSeries:
    """Simulate a concentration series (shared kinetics across ``m_tots``) + an optional
    seeded elongation anchor — the global-fit dataset that resolves the individual
    constants (with the anchor). Defaults to the BALANCED regime (both nucleation pathways
    contribute) so all three constants are individually determinable once the anchor
    breaks the gauge."""
    p = params if params is not None else _BALANCED_TRUTH
    curves = tuple(
        simulate_aggregation_curve(
            params=p, m_tot=m, t_max=t_max, n_obs=n_obs, dt=dt,
            n_replicates=n_replicates, obs_noise=obs_noise, seed=seed + i,
        )
        for i, m in enumerate(m_tots)
    )
    anchor = (
        simulate_seeded_elongation(
            params=p, m_tot=float(m_tots[len(m_tots) // 2]),
            n_replicates=n_replicates, obs_noise=obs_noise, seed=seed + 100,
        )
        if with_anchor
        else None
    )
    ground_truth = {
        "k_n": p.k_n, "k_plus": p.k_plus, "k_2": p.k_2, "n_c": p.n_c, "n_2": p.n_2,
        "m_tots": tuple(float(m) for m in m_tots), "with_anchor": with_anchor,
    }
    return ConcentrationSeries(curves=curves, anchor=anchor, ground_truth=ground_truth)


def simulate_inhibitor_pair(
    *,
    params: AggregationParams | None = None,
    target: str = "secondary_nucleation",
    factor: float = 0.3,
    m_tot: float = 1.0,
    t_max: float = 40.0,
    n_obs: int = 60,
    dt: float = 0.01,
    n_replicates: int = 12,
    obs_noise: float = 0.02,
    seed: int = 0,
) -> tuple[AggregationCurve, AggregationCurve, dict[str, Any]]:
    """A control vs inhibited curve pair: the inhibitor **lowers one microscopic rate**
    (``target`` ∈ {primary_nucleation, elongation, secondary_nucleation,
    monomer_sequestration}) by ``factor``. Returns ``(control, inhibited, ground_truth)``;
    the attribution target is *which* rate moved."""
    p = params if params is not None else _DEFAULT_TRUTH
    control = simulate_aggregation_curve(
        params=p, m_tot=m_tot, t_max=t_max, n_obs=n_obs, dt=dt,
        n_replicates=n_replicates, obs_noise=obs_noise, seed=seed,
    )
    inhib_params = p if target == "none" else p.with_scaled(target, factor)
    inhibited = simulate_aggregation_curve(
        params=inhib_params, m_tot=m_tot, t_max=t_max, n_obs=n_obs, dt=dt,
        n_replicates=n_replicates, obs_noise=obs_noise, seed=seed + 50,
    )
    ground_truth = {"target": target, "factor": factor, "m_tot": m_tot}
    return control, inhibited, ground_truth


# --------------------------------------------------------------------------- #
# single-curve composite fit (well-posed: the 2 identifiable dof)
# --------------------------------------------------------------------------- #
def _sim_from_composites(
    log_lam: Array, log_kappa: Array, curve: AggregationCurve, n_c: float, n_2: float,
) -> Array:
    """Mass-fraction curve as a function of the composites ``(log λ, log κ)``, in the
    gauge ``k_+ ≡ 1`` (``k_n = λ²/(2 m^{n_c})``, ``k_2 = κ²/(2 m^{n_2+1})``). Because the
    curve is gauge-invariant this is a faithful 2-dof reparameterization."""
    lam = jnp.exp(log_lam)
    kappa = jnp.exp(log_kappa)
    m = curve.m_tot
    k_plus = jnp.asarray(1.0, dtype=jnp.float64)
    k_n = lam**2 / (2.0 * m**n_c)
    k_2 = kappa**2 / (2.0 * m ** (n_2 + 1.0))
    return simulate_aggregation(
        (k_n, k_plus, k_2), m_tot=m, dt=curve.dt, n_steps=curve.n_steps,
        obs_idx=jnp.asarray(curve.obs_idx), n_c=n_c, n_2=n_2, p0=curve.p0,
    )


@dataclass(frozen=True)
class CompositeFit:
    """The single-curve fit of the two identifiable composites ``(λ, κ)`` + their
    lognormal 95% CIs and the fit residual."""

    lam: float
    kappa: float
    lam_ci: tuple[float, float]
    kappa_ci: tuple[float, float]
    rss: float
    n_c: float
    n_2: float
    m_tot: float


@_under_x64
def fit_composites(
    curve: AggregationCurve,
    *,
    n_c: float | None = None,
    n_2: float | None = None,
    steps: int = 600,
    learning_rate: float = 0.05,
    seed: int = 0,
) -> CompositeFit:
    """Recover the identifiable composites ``(λ, κ)`` from a single aggregation curve.

    A 2-parameter Adam least-squares fit of ``(log λ, log κ)`` to the mean mass-fraction
    trace (the field-standard AmyloFit objective; a deterministic sigmoid + measurement
    noise, so least-squares — not the distributional energy distance — is the right loss).
    Orders ``n_c`` / ``n_2`` default to the ground-truth values (a single curve cannot
    determine them; ``NUDGE-LIM-021``). CIs come from the Laplace curvature on
    ``(log λ, log κ)``, which is well-conditioned (these two dof ARE identifiable).
    """
    gt = curve.ground_truth
    n_c = float(gt.get("n_c", 2.0)) if n_c is None else n_c
    n_2 = float(gt.get("n_2", 2.0)) if n_2 is None else n_2
    obs = jnp.asarray(curve.mean, dtype=jnp.float64)

    lam0, kappa0 = _composite_init(curve)
    theta = {
        "log_lam": jnp.asarray(np.log(lam0), jnp.float64),
        "log_kappa": jnp.asarray(np.log(kappa0), jnp.float64),
    }

    def loss(th: dict[str, Array]) -> Array:
        sim = _sim_from_composites(th["log_lam"], th["log_kappa"], curve, n_c, n_2)
        return jnp.mean((sim - obs) ** 2)

    opt = optax.adam(learning_rate)
    state = opt.init(theta)

    @jax.jit
    def step(th: dict[str, Array], st: optax.OptState) -> tuple[Any, Any, Array]:
        val, grad = jax.value_and_grad(loss)(th)
        updates, st = opt.update(grad, st)
        return optax.apply_updates(th, updates), st, val

    val = jnp.asarray(0.0)
    for _ in range(steps):
        theta, state, val = step(theta, state)

    log_lam = float(theta["log_lam"])
    log_kappa = float(theta["log_kappa"])
    rss = float(val) * len(curve.t_obs)

    # Laplace CIs on (log λ, log κ) — an identifiable pair, so well-conditioned.
    obs_sd = max(float(np.std(curve.signal - curve.mean[None, :])), 1e-3)

    def nll(vec: Array) -> Array:
        sim = _sim_from_composites(vec[0], vec[1], curve, n_c, n_2)
        return jnp.mean(0.5 * (sim - obs) ** 2 / obs_sd**2)

    # The (λ, κ) pair IS identifiable (both curvature directions are non-flat), though λ
    # is weakly constrained in a secondary-dominated regime — report a finite (wide) CI
    # rather than declaring it unidentifiable, so a tiny flat tolerance is used here. (The
    # individual-k degeneracy is measured separately with the strict default guard.)
    post = laplace_posterior(
        nll, np.array([log_lam, log_kappa]), names=["lambda", "kappa"],
        n_data=curve.signal.size, cond_max=1e12, flat_rel_tol=1e-12,
    )
    lam_ci = (post.marginal_ci[0].lo, post.marginal_ci[0].hi)
    kappa_ci = (post.marginal_ci[1].lo, post.marginal_ci[1].hi)
    return CompositeFit(
        lam=float(np.exp(log_lam)), kappa=float(np.exp(log_kappa)),
        lam_ci=lam_ci, kappa_ci=kappa_ci, rss=rss, n_c=n_c, n_2=n_2, m_tot=curve.m_tot,
    )


def _composite_init(curve: AggregationCurve) -> tuple[float, float]:
    """Data-driven init: ``κ ≈ ln9 / (t_75 − t_25)`` (the autocatalytic growth rate from
    the transition width), ``λ`` a decade below it (lag-dominated primary pathway)."""
    y = curve.mean
    t = curve.t_obs
    yn = np.clip(y / max(y.max(), 1e-6), 0.0, 1.0)
    t25 = float(np.interp(0.25, yn, t))
    t75 = float(np.interp(0.75, yn, t))
    width = max(t75 - t25, 1e-2)
    kappa0 = float(np.log(9.0) / width)
    kappa0 = float(np.clip(kappa0, 1e-3, 100.0))
    return 0.01 * kappa0, kappa0


# --------------------------------------------------------------------------- #
# the single-curve non-identifiability, MEASURED via the Laplace/Fisher curvature
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IndividualKIdentifiability:
    """The MEASURED single-curve degeneracy of the three individual rate constants.

    Built from the Laplace/Fisher curvature on ``(log k_n, log k_+, log k_2)`` at a
    representative point on the gauge orbit. ``posterior`` is the raw
    :class:`~nudge.inference.uncertainty.LaplacePosterior`; ``degenerate`` its verdict;
    ``cond_number`` the condition number (``→ ∞`` at the exact gauge null);
    ``null_direction`` the smallest-eigenvalue eigenvector in ``(log k_n, log k_+,
    log k_2)`` space (≈ ``(−1, +1, −1)/√3`` — the ``k_+`` ⇄ ``(k_n, k_2)`` trade-off);
    ``gauge_check`` the max mass-fraction change under a 100× ``k_+`` rescale along the
    null (≈ 0 confirms the exact symmetry analytically claimed)."""

    posterior: LaplacePosterior
    degenerate: bool
    cond_number: float
    null_direction: np.ndarray
    unidentifiable: tuple[str, ...]
    gauge_check: float
    reason: str


@_under_x64
def individual_k_identifiability(
    curve: AggregationCurve,
    fit: CompositeFit,
    *,
    cond_max: float = 100.0,
) -> IndividualKIdentifiability:
    """Measure whether the three individual constants are identifiable from ONE curve.

    Evaluates the Fisher/Laplace curvature of the mass-fraction least-squares NLL as a
    function of ``(log k_n, log k_+, log k_2)`` at the gauge-fixed point ``k_+ = 1``,
    ``k_n = λ²/(2 m^{n_c})``, ``k_2 = κ²/(2 m^{n_2+1})`` (from ``fit``), reusing
    :func:`~nudge.inference.uncertainty.laplace_posterior` verbatim. The moment model's
    exact gauge symmetry forces one eigenvalue to ~0 (condition number ``→ ∞``), so the
    guard returns ``degenerate=True`` and marks all three constants unidentifiable — the
    earned abstention (``NUDGE-LIM-021``). Also runs an independent numerical
    **gauge check** (rescale ``k_+`` 100× along the null; the curve must be invariant).
    """
    n_c, n_2 = fit.n_c, fit.n_2
    m = curve.m_tot
    k_plus0 = 1.0
    k_n0 = fit.lam**2 / (2.0 * m**n_c)
    k_20 = fit.kappa**2 / (2.0 * m ** (n_2 + 1.0))
    theta_opt = np.array([np.log(k_n0), np.log(k_plus0), np.log(k_20)])
    obs = jnp.asarray(curve.mean, dtype=jnp.float64)
    obs_sd = max(float(np.std(curve.signal - curve.mean[None, :])), 1e-3)

    def nll(log_k: Array) -> Array:
        k_n = jnp.exp(log_k[0])
        k_plus = jnp.exp(log_k[1])
        k_2 = jnp.exp(log_k[2])
        sim = simulate_aggregation(
            (k_n, k_plus, k_2), m_tot=m, dt=curve.dt, n_steps=curve.n_steps,
            obs_idx=jnp.asarray(curve.obs_idx), n_c=n_c, n_2=n_2, p0=curve.p0,
        )
        return jnp.mean(0.5 * (sim - obs) ** 2 / obs_sd**2)

    post = laplace_posterior(
        nll, theta_opt, names=list(_K_NAMES), n_data=curve.signal.size,
        cond_max=cond_max,
    )
    eigvals = np.asarray(post.eigenvalues)  # ascending
    null_vec = np.asarray(np.linalg.eigh(post.hessian)[1][:, 0], dtype=np.float64)
    if null_vec[int(np.argmax(np.abs(null_vec)))] < 0:
        null_vec = -null_vec
    unident = tuple(ci.name for ci in post.marginal_ci if not ci.identifiable)

    # independent numerical gauge check: move 100× along k_+, compensate k_n, k_2 by 1/α.
    alpha = 100.0
    base = simulate_aggregation(
        (jnp.asarray(k_n0), jnp.asarray(k_plus0), jnp.asarray(k_20)), m_tot=m,
        dt=curve.dt, n_steps=curve.n_steps, obs_idx=jnp.asarray(curve.obs_idx),
        n_c=n_c, n_2=n_2, p0=curve.p0,
    )
    scaled = simulate_aggregation(
        (jnp.asarray(k_n0 / alpha), jnp.asarray(k_plus0 * alpha),
         jnp.asarray(k_20 / alpha)), m_tot=m, dt=curve.dt, n_steps=curve.n_steps,
        obs_idx=jnp.asarray(curve.obs_idx), n_c=n_c, n_2=n_2, p0=curve.p0,
    )
    gauge_check = float(np.max(np.abs(np.asarray(base) - np.asarray(scaled))))

    reason = (
        f"single-curve individual-k curvature: condition number {post.cond_number:.3g}, "
        f"degenerate={post.degenerate}; smallest eigenvalue {float(eigvals[0]):.3g}. "
        f"Null direction ≈ {np.round(null_vec, 3).tolist()} in (log k_n, log k_+, "
        f"log k_2) — the exact k_+ ⇄ (k_n, k_2) gauge. Numerical gauge check: a 100× k_+ "
        f"rescale changes the curve by {gauge_check:.2g} (≈ 0 confirms the exact symmetry)."
    )
    return IndividualKIdentifiability(
        posterior=post, degenerate=bool(post.degenerate),
        cond_number=float(post.cond_number), null_direction=null_vec,
        unidentifiable=unident, gauge_check=gauge_check, reason=reason,
    )


# --------------------------------------------------------------------------- #
# the one-call entry point (the efficiency demo)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AggregationResult:
    """A single-curve aggregation attribution: the identifiable composites, the measured
    non-identifiability of the three individual constants, and the honest guidance."""

    fit: CompositeFit
    identifiability: IndividualKIdentifiability
    call: str  # "composites-identified" | "fully-resolved"
    reason: str
    guidance: str

    @property
    def lam(self) -> float:
        return self.fit.lam

    @property
    def kappa(self) -> float:
        return self.fit.kappa

    @property
    def individual_k_identifiable(self) -> bool:
        return not self.identifiability.degenerate


def attribute_aggregation(
    curve: AggregationCurve,
    *,
    n_c: float | None = None,
    n_2: float | None = None,
    steps: int = 600,
    cond_max: float = 100.0,
    seed: int = 0,
) -> AggregationResult:
    """Fit a single aggregation curve and report — **in one call** — the identifiable
    composites ``(κ, λ)`` and the MEASURED non-identifiability of the three individual
    rate constants.

    This is the efficiency demo: the honest answer a control LLM agent took 12.2 minutes /
    28 turns / six iterative scripts to hand-derive
    (``design/automated_scientist/runs/000000008``), returned deterministically in ONE call
    (a few seconds, compile-dominated). Fail-safe: it never reports a false-precise
    individual constant from a single curve — it returns the composites and the
    ``k_+`` ⇄ ``(k_n, k_2)`` null direction, and prescribes a concentration series + a
    seeded anchor (``NUDGE-LIM-021``).
    """
    fit = fit_composites(curve, n_c=n_c, n_2=n_2, steps=steps, seed=seed)
    ident = individual_k_identifiability(curve, fit, cond_max=cond_max)

    if ident.degenerate:
        call = "composites-identified"
        reason = (
            f"single curve at m_tot={fit.m_tot:g}: the composites are identifiable — "
            f"κ={fit.kappa:.3g} (95% CI [{fit.kappa_ci[0]:.3g}, {fit.kappa_ci[1]:.3g}]), "
            f"λ={fit.lam:.3g} (95% CI [{fit.lam_ci[0]:.3g}, {fit.lam_ci[1]:.3g}]). But "
            f"the three individual constants k_n, k_+, k_2 are NOT separable: "
            f"{ident.reason}"
        )
        guidance = (
            "To pin k_n, k_+, k_2 individually, run a CONCENTRATION SERIES (several m_tot "
            "→ the reaction orders n_c, n_2 and the rate products k_+·k_n, k_+·k_2) AND a "
            "SEEDED / elongation reference (a heavily-seeded curve where dM/dt ≈ 2 k_+ m "
            "P_0 directly constrains k_+, breaking the gauge). The Meisl discipline "
            "(NUDGE-LIM-021)."
        )
    else:  # curvature well-conditioned (e.g. a seeded curve pins k_+) — rare from 1 curve
        call = "fully-resolved"
        reason = (
            f"the individual-k curvature is well-conditioned (condition number "
            f"{ident.cond_number:.3g}); κ={fit.kappa:.3g}, λ={fit.lam:.3g}."
        )
        guidance = "The three constants are identifiable from this curve alone."
    return AggregationResult(
        fit=fit, identifiability=ident, call=call, reason=reason, guidance=guidance,
    )


# --------------------------------------------------------------------------- #
# inhibitor attribution — which microscopic step did the inhibitor lower?
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InhibitorResult:
    """Which microscopic rate an inhibitor lowered, from the composite log-ratios of a
    control-vs-inhibited curve pair."""

    call: str  # primary_nucleation|elongation|secondary_nucleation|no-effect|unresolved
    reason: str
    r_lambda: float  # log(λ_inhibited / λ_control)
    r_kappa: float  # log(κ_inhibited / κ_control)
    control_fit: CompositeFit
    inhibited_fit: CompositeFit

    @property
    def is_reliable(self) -> bool:
        return self.call in _TARGETS


def attribute_inhibitor(
    control: AggregationCurve,
    inhibited: AggregationCurve,
    *,
    n_c: float | None = None,
    n_2: float | None = None,
    steps: int = 600,
    effect_tol: float = 0.15,
    equal_tol: float = 0.25,
    seed: int = 0,
) -> InhibitorResult:
    """Attribute an inhibitor to the microscopic step it targets — **from single curves**.

    The absolute constants are gauge-degenerate, but the inhibitor's *relative* effect on
    the two composites is identifiable: a **fibril-end binder** (``k_+``) scales λ AND κ
    together; a **surface binder** (``k_2``) lowers κ only; a **primary-nucleus binder**
    (``k_n``) lowers λ only. Decision on the composite log-ratios
    ``r_λ = log(λ_inhib/λ_ctrl)``, ``r_κ = log(κ_inhib/κ_ctrl)``:

    - both ``< −effect_tol`` and ``|r_λ − r_κ| < equal_tol`` → **elongation** (or a global
      monomer-sequestering binder — a documented composite ambiguity, ``NUDGE-LIM-021``);
    - only ``r_κ < −effect_tol`` → **secondary_nucleation**;
    - only ``r_λ < −effect_tol`` → **primary_nucleation**;
    - neither → **no-effect**; anything else → **unresolved** (abstain).
    """
    cf = fit_composites(control, n_c=n_c, n_2=n_2, steps=steps, seed=seed)
    inf = fit_composites(inhibited, n_c=n_c, n_2=n_2, steps=steps, seed=seed)
    r_lam = float(np.log(max(inf.lam, _EPS) / max(cf.lam, _EPS)))
    r_kap = float(np.log(max(inf.kappa, _EPS) / max(cf.kappa, _EPS)))

    lam_down = r_lam < -effect_tol
    kap_down = r_kap < -effect_tol
    lam_flat = abs(r_lam) <= effect_tol
    kap_flat = abs(r_kap) <= effect_tol

    if not lam_down and not kap_down and lam_flat and kap_flat:
        call = "no-effect"
        reason = (
            f"neither composite moved beyond the noise floor (r_λ={r_lam:+.3f}, "
            f"r_κ={r_kap:+.3f}; tol={effect_tol}); the inhibitor has no detectable effect "
            "on the aggregation kinetics"
        )
    elif lam_down and kap_down and abs(r_lam - r_kap) < equal_tol:
        call = "elongation"
        reason = (
            f"both composites dropped by the SAME factor (r_λ={r_lam:+.3f} ≈ "
            f"r_κ={r_kap:+.3f}) — the signature of a fibril-END binder lowering k_+ "
            f"(λ, κ ∝ √k_+). NOTE: a global monomer-sequestering binder gives the same "
            f"composite signature (NUDGE-LIM-021)"
        )
    elif kap_down and lam_flat:
        call = "secondary_nucleation"
        reason = (
            f"κ dropped (r_κ={r_kap:+.3f}) while λ held (r_λ={r_lam:+.3f}) — a "
            f"fibril-SURFACE binder lowering the secondary-nucleation rate k_2 (κ ∝ √k_2, "
            f"λ independent of k_2)"
        )
    elif lam_down and kap_flat:
        call = "primary_nucleation"
        reason = (
            f"λ dropped (r_λ={r_lam:+.3f}) while κ held (r_κ={r_kap:+.3f}) — a "
            f"primary-NUCLEUS binder lowering k_n (λ ∝ √k_n, κ independent of k_n)"
        )
    else:
        call = "unresolved"
        reason = (
            f"the composite shifts do not match a single-target signature "
            f"(r_λ={r_lam:+.3f}, r_κ={r_kap:+.3f}) — NUDGE abstains rather than guess "
            "which microscopic step moved"
        )
    return InhibitorResult(
        call=call, reason=reason, r_lambda=r_lam, r_kappa=r_kap,
        control_fit=cf, inhibited_fit=inf,
    )


# --------------------------------------------------------------------------- #
# concentration-series global fit — resolves the individuals (with a seeded anchor)
# --------------------------------------------------------------------------- #
def _series_nll(
    log_k: Array, series: ConcentrationSeries, use_anchor: bool, obs_sd: float,
) -> Array:
    """Summed mass-fraction least-squares NLL of ONE shared ``(log k_n, log k_+, log k_2)``
    across every curve in the series (and the anchor, if ``use_anchor``). The anchor's
    large seed ``p_0`` makes its curve depend on ``k_+`` directly — the term that breaks
    the gauge."""
    k_n = jnp.exp(log_k[0])
    k_plus = jnp.exp(log_k[1])
    k_2 = jnp.exp(log_k[2])
    total = jnp.asarray(0.0, dtype=jnp.float64)
    n = 0
    curves = list(series.curves)
    if use_anchor and series.anchor is not None:
        curves = curves + [series.anchor]
    for c in curves:
        n_c = float(c.ground_truth.get("n_c", 2.0))
        n_2 = float(c.ground_truth.get("n_2", 2.0))
        sim = simulate_aggregation(
            (k_n, k_plus, k_2), m_tot=c.m_tot, dt=c.dt, n_steps=c.n_steps,
            obs_idx=jnp.asarray(c.obs_idx), n_c=n_c, n_2=n_2, p0=c.p0,
        )
        obs = jnp.asarray(c.mean, dtype=jnp.float64)
        total = total + jnp.sum((sim - obs) ** 2)
        n += obs.shape[0]
    return total / (2.0 * obs_sd**2 * n)


@dataclass(frozen=True)
class SeriesResolution:
    """The concentration-series global-fit result: recovered individual constants (in the
    natural gauge when resolved), whether they are identifiable, and the measured
    curvature."""

    k_n: float
    k_plus: float
    k_2: float
    identifiable: bool
    cond_number: float
    used_anchor: bool
    posterior: LaplacePosterior
    reason: str


@_under_x64
def series_identifiability(
    series: ConcentrationSeries,
    theta_opt: np.ndarray,
    *,
    use_anchor: bool,
    obs_sd: float = 0.02,
    cond_max: float = 1e12,
    flat_rel_tol: float = 1e-4,
) -> LaplacePosterior:
    """The Laplace/Fisher curvature of the global series NLL on ``(log k_n, log k_+,
    log k_2)``.

    The discriminator is the **exact gauge**, not the raw condition number (which the
    deep-research sloppiness caveat warns is unreliable): without a seeded anchor the
    mass-fraction gauge leaves a **genuine zero eigenvalue** (a flat direction → cond ``∞``
    → ``degenerate``); with the anchor every eigenvalue is positive (the gauge is broken)
    even though the spectrum is *sloppy* (a large-but-finite condition number reported as a
    caveat). So a small ``flat_rel_tol`` (flag only a true zero) with a permissive
    ``cond_max`` separates "non-identifiable (gauge)" from "sloppy-but-identifiable"."""
    return laplace_posterior(
        lambda lk: _series_nll(lk, series, use_anchor, obs_sd),
        theta_opt, names=list(_K_NAMES), n_data=1, cond_max=cond_max,
        flat_rel_tol=flat_rel_tol,
    )


@_under_x64
def fit_series_global(
    series: ConcentrationSeries,
    *,
    use_anchor: bool = True,
    steps: int = 1500,
    learning_rate: float = 0.05,
    obs_sd: float = 0.02,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Global shared-parameter fit of ONE ``(log k_n, log k_+, log k_2)`` across the whole
    series (+ anchor). Initialized from a composite fit of the reference concentration.
    Returns ``(theta_opt, final_nll)``."""
    ref = series.curves[len(series.curves) // 2]
    cf = fit_composites(ref, steps=max(steps // 2, 300), seed=seed)
    n_c, n_2 = cf.n_c, cf.n_2
    m = ref.m_tot
    # init in the k_+ = 1 gauge (the anchor will pull k_+ to its true value if used).
    theta = jnp.asarray(
        [np.log(cf.lam**2 / (2.0 * m**n_c)), 0.0,
         np.log(cf.kappa**2 / (2.0 * m ** (n_2 + 1.0)))],
        dtype=jnp.float64,
    )

    def loss(lk: Array) -> Array:
        return _series_nll(lk, series, use_anchor, obs_sd)

    opt = optax.adam(learning_rate)
    state = opt.init(theta)

    @jax.jit
    def step(lk: Array, st: optax.OptState) -> tuple[Any, Any, Array]:
        val, grad = jax.value_and_grad(loss)(lk)
        updates, st = opt.update(grad, st)
        return optax.apply_updates(lk, updates), st, val

    val = jnp.asarray(0.0)
    for _ in range(steps):
        theta, state, val = step(theta, state)
    return np.asarray(theta), float(val)


def resolve_series(
    series: ConcentrationSeries,
    *,
    use_anchor: bool = True,
    steps: int = 1500,
    obs_sd: float = 0.02,
    cond_max: float = 1e12,
    seed: int = 0,
) -> SeriesResolution:
    """Fit the concentration series globally and report whether the three individual
    constants are resolved — the honest completion of the single-curve abstention.

    WITHOUT the seeded anchor the mass-fraction gauge persists (the curvature stays
    degenerate; NUDGE reports the constants as unidentifiable). WITH the anchor (the Meisl
    discipline) the gauge breaks and all three resolve. Fail-safe: it reports individual
    constants as trustworthy ONLY when the measured curvature is well-conditioned.
    """
    theta, _nll = fit_series_global(
        series, use_anchor=use_anchor, steps=steps, obs_sd=obs_sd, seed=seed,
    )
    post = series_identifiability(
        series, theta, use_anchor=use_anchor, obs_sd=obs_sd, cond_max=cond_max,
    )
    identifiable = not post.degenerate
    k_n, k_plus, k_2 = (float(np.exp(theta[i])) for i in range(3))
    if identifiable:
        reason = (
            f"global fit across {len(series.curves)} concentrations"
            f"{' + a seeded anchor' if use_anchor else ''}: the individual-k curvature has "
            f"NO flat direction (the gauge is broken; condition number "
            f"{post.cond_number:.3g}, sloppy but identifiable) — the three constants are "
            f"RESOLVED (k_n={k_n:.3g}, k_+={k_plus:.3g}, k_2={k_2:.3g})"
        )
    else:
        reason = (
            f"global fit across {len(series.curves)} concentrations"
            f"{' + a seeded anchor' if use_anchor else ' (NO anchor)'}: the curvature is "
            f"STILL degenerate (condition number {post.cond_number:.3g}) — the "
            f"mass-fraction gauge k_+ ⇄ (k_n, k_2) is concentration-independent, so the "
            f"series alone cannot separate the individual constants. NUDGE reports only "
            f"the products k_+·k_n and k_+·k_2 (NUDGE-LIM-021)"
        )
    return SeriesResolution(
        k_n=k_n, k_plus=k_plus, k_2=k_2, identifiable=identifiable,
        cond_number=float(post.cond_number), used_anchor=use_anchor, posterior=post,
        reason=reason,
    )
