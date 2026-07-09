"""Stage-2 design — invert a fit to **propose an untested intervention**.

This is NUDGE's headline inversion: once a mechanism has been *diagnosed* (which knob
a switch turns), ``design()`` runs the same differentiable machinery **backwards** to
*prescribe* — what change to a kinetic parameter (or what dose) would move the system to
a target. It is the forward model run in reverse, behind two honesty gates:

- an **integrity gate** — it refuses to design off an *unreliable* attribution (a
  ``DoseResponseResult`` that abstained, a low-confidence fit); and
- a **bifurcation safety gate** — it flags an intervention that pushes a bistable switch
  toward a tipping point, reusing the shipped Cap-5 :func:`bifurcation_proximity` dial
  (:mod:`nudge.inference.bifurcation`).

Two modes, dispatched by what the attribution *carries* (structural, not a base class):

- **Circuit-level** (flagship / deep-tech): gradient inversion over a fitted
  differentiable :class:`~nudge.core.circuit.Circuit`. It optimizes an additive
  **log-delta Δ** over addressable kinetic knobs to minimise
  ``L(Δ) = ‖PredictedState(Δ) − target_state‖² + l1·‖Δ‖₁`` with Adam + autodiff (the
  ``fit_parameters`` loop, run backwards), then runs the safety gate on the intervened
  circuit. A **reachability abstention** fires when no Δ hits the target within ``tol``.
- **Curve-level** (real-data surface): closed-form inversion of a
  :class:`~nudge.inference.dose_response.DoseResponseFit` to the dose achieving a target
  response ``y`` (invert the Hill). No circuit ⇒ **no safety gate** (stated honestly),
  and a reachability abstention when ``y`` is outside ``(floor, floor+amp)``.

**Honesty (load-bearing).** Never design off an unreliable fit; abstain on unreachable
targets; the safety score inherits Cap 5's **one-sided lower bound** near the fold; and
every proposal is valid only within the fit's identifiable region — extrapolation is a
documented risk (``NUDGE-LIM-013``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, cast, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from nudge.core.circuit import Circuit, Params
from nudge.core.vocabulary import MechanismClass
from nudge.inference.bifurcation import BifurcationScore, bifurcation_proximity
from nudge.inference.fit import FreeParam
from nudge.mechanisms.readout import Readout

__all__ = [
    "AbstentionResult",
    "AttributionResult",
    "CircuitFit",
    "InterventionPlan",
    "SafetyReport",
    "design",
    "flip_target",
]


# --------------------------------------------------------------------------- #
# Input contract — a strictly-minimal structural Protocol (Interface Segregation):
# design()'s integrity gate needs ONLY these two members. Any current or future
# attribution result satisfies it by having them (DoseResponseResult / EpistasisResult
# gain an is_reliable property additively; CircuitFit carries the fields).
# --------------------------------------------------------------------------- #
@runtime_checkable
class AttributionResult(Protocol):
    """The minimal contract ``design()``'s integrity gate requires — nothing more."""

    @property
    def is_reliable(self) -> bool:
        """Whether the attribution is trustworthy enough to invert."""
        ...

    @property
    def reason(self) -> str:
        """The human-readable rationale carried by every NUDGE verdict."""
        ...


@dataclass(frozen=True)
class CircuitFit:
    """A fitted differentiable circuit + the invertible substrate for circuit design.

    ``params`` is the base parameter pytree (single-leaf arrays — the deterministic
    circuit kinetics); it defaults to ``circuit.base_params()``. ``free`` names the
    addressable knobs ``design()`` may move (edge ``K/n/vmax`` or species ``basal`` =
    dose); ``None`` means "the full addressable set" (:func:`_default_free`).
    ``is_reliable`` is the integrity-gate flag a real fit sets from its own diagnostics.
    """

    circuit: Circuit
    params: Params | None = None
    free: list[FreeParam] | None = None
    is_reliable: bool = True
    reason: str = "circuit fit supplied directly"

    def base(self) -> Params:
        """The base parameter pytree (explicit ``params`` or the circuit's declared)."""
        return self.params if self.params is not None else self.circuit.base_params()

    def knobs(self) -> list[FreeParam]:
        """The addressable knobs (explicit ``free`` or the full addressable set)."""
        return self.free if self.free is not None else _default_free(self.circuit)


@dataclass(frozen=True)
class SafetyReport:
    """The safety gate's verdict on an intervention (the Cap-5 dial, before→after).

    Fail-safe reading of :func:`bifurcation_proximity` on the base vs the intervened
    circuit. ``None`` proximity means *not bistable* (base) or *fold crossed* (after).
    ``crosses_fold`` is the sharpest instability signal — the intervention destroys the
    switch's bistability. ``one_sided`` inherits Cap 5's honesty: near the fold the risk
    is a **lower bound** ("at least this close") — the linear-noise Gaussian breaks
    down precisely there (``NUDGE-LIM-012`` / ``NUDGE-LIM-013``).
    """

    proximity_before: float | None
    proximity_after: float | None
    delta: float | None
    one_sided: bool
    high_risk_of_instability: bool
    crosses_fold: bool
    channels_before: Mapping[str, Any] = field(default_factory=dict)
    channels_after: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InterventionPlan:
    """A proposed intervention — the design() output when a target is reachable.

    ``mode`` is ``"circuit"`` (kinetic ``deltas``) or ``"dose"`` (a scalar ``dose``).
    ``deltas`` is the ranked, sparse set of ``(knob, log_delta, factor)`` the optimizer
    kept (``factor = exp(log_delta)`` — a multiplicative change to the base parameter).
    ``predicted_state`` is the reached readout-space state (comparable to the circuit
    ``target_state``); ``predicted_response`` the reached scalar ``y`` (curve mode).
    ``predicted_trajectory`` is the activity-space relaxation to the new steady state
    (the flip-ON plot). ``achieved_loss`` is the relative residual gap that remained.
    ``safety`` is the Cap-5 report (``None`` in curve mode: no circuit, stated).
    """

    mode: str
    deltas: tuple[tuple[FreeParam, float, float], ...] = ()
    dose: float | None = None
    predicted_state: tuple[float, ...] | None = None
    predicted_response: float | None = None
    predicted_trajectory: Any = field(default=None, compare=False, repr=False)
    achieved_loss: float = 0.0
    safety: SafetyReport | None = None
    reason: str = ""


@dataclass(frozen=True)
class AbstentionResult:
    """design() declined — an abstention verdict + why (the ``(call, reason)`` idiom).

    ``verdict`` is a :class:`~nudge.core.vocabulary.MechanismClass` abstention; the
    two gates that produce it are the **integrity gate** (won't invert an unreliable
    attribution) and the **reachability gate** (no intervention reaches the target).
    """

    verdict: MechanismClass
    reason: str


# --------------------------------------------------------------------------- #
# Addressable knobs + functional Δ application (mirror inference.lyapunov._apply_free —
# autodiff-clean single-leaf params; NOT the per-cell fit._override).
# --------------------------------------------------------------------------- #
def _default_free(circuit: Circuit) -> list[FreeParam]:
    """The full knob set: each edge ``K/n/vmax`` + each species ``basal`` (= dose)."""
    free: list[FreeParam] = []
    for i in range(circuit.n_edges):
        free.extend([("edge", i, "K"), ("edge", i, "n"), ("edge", i, "vmax")])
    for j in range(circuit.n_species):
        free.append(("species", j, "basal"))
    return free


def _base_value(base: Params, f: FreeParam) -> float:
    scope, index, name = f
    coll = "edges" if scope == "edge" else "species"
    return float(np.asarray(base[coll][name])[index])


def _apply_delta(base: Params, free: list[FreeParam], vals: Array) -> Params:
    """Functionally set ``free`` to ``vals`` in a copy of ``base`` (autodiff-clean)."""
    params: Params = {"species": dict(base["species"]), "edges": dict(base["edges"])}
    for (scope, index, name), v in zip(free, vals, strict=True):
        coll = "edges" if scope == "edge" else "species"
        params[coll][name] = params[coll][name].at[index].set(v)
    return params


def _rebuild(circuit: Circuit, free: list[FreeParam], vals: np.ndarray) -> Circuit:
    """Rebuild ``circuit`` with the ``free`` kinetics overridden (the safety gate)."""
    species = list(circuit.species)
    edges = list(circuit.edges)
    for (scope, index, name), v in zip(free, vals, strict=True):
        if scope == "edge":
            edges[index] = replace(edges[index], **{name: float(v)})
        else:
            species[index] = replace(species[index], **{name: float(v)})
    return Circuit(species, edges)


# --------------------------------------------------------------------------- #
# Fixed-point seeding + the "flip" target helper.
# --------------------------------------------------------------------------- #
def _stable_states(circuit: Circuit) -> list[np.ndarray]:
    """The stable fixed points, ascending by norm (``[low, …, high]``)."""
    fps = circuit.fixed_points()
    if not fps:
        return []
    states = [np.asarray(s, dtype=float) for s, lab in fps if lab == "stable"]
    return sorted(states, key=lambda s: float(np.linalg.norm(s)))


def flip_target(
    circuit: Circuit, *, to: str = "high", readout: Readout | None = None
) -> np.ndarray:
    """The readout-space target for flipping the switch ``to`` its high/low basin.

    A convenience for circuit design: returns ``readout.expression(state)`` for the
    high (``to="high"``) or low (``to="low"``) stable fixed point — pass it as
    ``design()``'s ``target_state``. Raises if the circuit has no stable fixed point.
    """
    readout = readout or Readout.identity(circuit.n_species)
    states = _stable_states(circuit)
    if not states:
        raise ValueError("circuit has no stable fixed point to target")
    state = states[-1] if to == "high" else states[0]
    return np.asarray(readout.expression(jnp.asarray(state, jnp.float32)), dtype=float)


def _resolve_x0(circuit: Circuit, start: Any) -> Array:
    """Seed the solve at the circuit's **current stable state** to intervene *from*.

    ``start`` is ``"low"`` (the resting basin — the natural "flip ON" default),
    ``"high"`` (the activated basin — to adjust the ON level from), or an explicit
    activity vector. Falls back to the zero state when the circuit has no located stable
    fixed point (a monostable solve is initial-condition-independent anyway).
    """
    if not isinstance(start, str):
        return jnp.asarray(np.asarray(start, dtype=float), dtype=jnp.float32)
    states = _stable_states(circuit)
    if not states:
        return jnp.zeros(circuit.n_species)
    state = states[-1] if start == "high" else states[0]
    return jnp.asarray(state, dtype=jnp.float32)


def _trajectory(
    circuit: Circuit, params: Params, x0: Array, *, dt: float = 0.1, n_steps: int = 500
) -> np.ndarray:
    """Activity relaxation ``x0 → steady state`` (the semi-implicit scan, stacked)."""
    decay = params["species"]["decay"]

    def step(x: Array, _: None) -> tuple[Array, Array]:
        x_new = jnp.maximum(
            (x + dt * circuit.production(x, params)) / (1.0 + dt * decay), 0.0
        )
        return x_new, x_new

    _, traj = jax.lax.scan(step, x0, None, length=n_steps)
    return np.asarray(traj, dtype=float)


# --------------------------------------------------------------------------- #
# Circuit-level inversion (the flagship) — mirror fit_parameters, run backwards.
# --------------------------------------------------------------------------- #
def _invert_circuit(
    fit: CircuitFit,
    target_state: Any,
    *,
    free: list[FreeParam] | None,
    steps: int,
    seed: int,
    l1: float,
    tol: float,
    learning_rate: float,
    readout: Readout | None,
    margin: float,
    start: Any,
) -> InterventionPlan | AbstentionResult:
    circuit = fit.circuit
    base = fit.base()
    knobs = free if free is not None else fit.knobs()
    readout = readout or Readout.identity(circuit.n_species)
    target = jnp.asarray(np.asarray(target_state, dtype=float).ravel())
    x0 = _resolve_x0(circuit, start)
    log_base = jnp.log(
        jnp.asarray([_base_value(base, f) for f in knobs], dtype=jnp.float32)
    )

    def predicted(delta: Array) -> Array:
        vals = jnp.exp(log_base + delta)
        params = _apply_delta(base, knobs, vals)
        activity = circuit.steady_state(params, x0)
        return readout.expression(activity)

    def loss_fn(delta: Array) -> Array:
        resid = predicted(delta) - target
        return jnp.sum(resid**2) + l1 * jnp.sum(jnp.abs(delta))

    delta = jnp.zeros(len(knobs))
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(delta)

    @jax.jit
    def step(
        d: Array, state: optax.OptState
    ) -> tuple[Array, optax.OptState, Array]:
        loss, grad = jax.value_and_grad(loss_fn)(d)
        updates, state = optimizer.update(grad, state)
        return jnp.asarray(optax.apply_updates(d, updates)), state, loss

    for _ in range(steps):
        delta, opt_state, _loss = step(delta, opt_state)

    # Reachability uses the DATA term only (the l1 term is a design preference,
    # not a fidelity), relative to the Δ=0 gap so it is scale-free across readouts.
    base_gap = float(jnp.sum((predicted(jnp.zeros(len(knobs))) - target) ** 2))
    final_pred = predicted(delta)
    data_loss = float(jnp.sum((final_pred - target) ** 2))
    rel = data_loss / (base_gap + 1e-12)

    if rel > tol:
        return AbstentionResult(
            MechanismClass.UNRESOLVED,
            f"reachability abstention: no intervention over the fitted knobs reached "
            f"target within tolerance (closed {100 * (1 - rel):.0f}% of the gap, need "
            f"{100 * (1 - tol):.0f}%). The target is unreachable within the fit's "
            f"region — NUDGE will not extrapolate a proposal (NUDGE-LIM-013).",
        )

    delta_np = np.asarray(delta, dtype=float)
    vals_np = np.asarray(np.exp(np.asarray(log_base) + delta_np), dtype=float)

    # Rank the surviving knobs by |Δ| and drop negligible moves (a sparse prescription).
    ranked = sorted(
        (
            (knobs[i], float(delta_np[i]), float(np.exp(delta_np[i])))
            for i in range(len(knobs))
            if abs(float(delta_np[i])) > 1e-3
        ),
        key=lambda t: abs(t[1]),
        reverse=True,
    )

    # The safety gate (Cap 5) — base vs intervened circuit.
    circuit_after = _rebuild(circuit, knobs, vals_np)
    safety = _safety_report(
        bifurcation_proximity(circuit), bifurcation_proximity(circuit_after), margin
    )

    params_after = _apply_delta(base, knobs, jnp.asarray(vals_np, jnp.float32))
    traj = _trajectory(circuit, params_after, x0)

    reason = _circuit_reason(ranked, rel, safety)
    return InterventionPlan(
        mode="circuit",
        deltas=tuple(ranked),
        predicted_state=tuple(float(v) for v in np.asarray(final_pred, dtype=float)),
        predicted_trajectory=traj,
        achieved_loss=rel,
        safety=safety,
        reason=reason,
    )


def _safety_report(
    s0: BifurcationScore | None, s1: BifurcationScore | None, margin: float
) -> SafetyReport:
    """Fail-safe Cap-5 verdict on base (``s0``) vs intervened (``s1``) proximity."""
    if s0 is None:
        # Base is not bistable — there is no switch to destabilize.
        return SafetyReport(
            proximity_before=None,
            proximity_after=None if s1 is None else s1.proximity,
            delta=None,
            one_sided=bool(s1 is not None and s1.one_sided),
            high_risk_of_instability=False,
            crosses_fold=False,
            channels_before={},
            channels_after={} if s1 is None else dict(s1.channels),
        )
    if s1 is None:
        # The intervention crossed the fold — the switch LOSES bistability (sharpest).
        return SafetyReport(
            proximity_before=s0.proximity,
            proximity_after=None,
            delta=None,
            one_sided=s0.one_sided,
            high_risk_of_instability=True,
            crosses_fold=True,
            channels_before=dict(s0.channels),
            channels_after={},
        )
    delta = s1.proximity - s0.proximity
    return SafetyReport(
        proximity_before=s0.proximity,
        proximity_after=s1.proximity,
        delta=delta,
        one_sided=s1.one_sided,  # a LOWER BOUND — "at least this close" — near the fold
        high_risk_of_instability=bool(delta > margin),
        crosses_fold=False,
        channels_before=dict(s0.channels),
        channels_after=dict(s1.channels),
    )


def _circuit_reason(
    ranked: list[tuple[FreeParam, float, float]], rel: float, safety: SafetyReport
) -> str:
    if ranked:
        (scope, index, name), _ld, factor = ranked[0]
        knob = f"{scope}[{index}].{name}"
        head = (
            f"proposed intervention: scale {knob} by x{factor:.2f} "
            f"({len(ranked)} knob(s) moved); closes {100 * (1 - rel):.0f}% of the "
            f"target gap"
        )
    else:
        head = f"target already reached ({100 * (1 - rel):.0f}% of the gap closed)"
    if safety.crosses_fold:
        tail = (
            " — HIGH RISK OF INSTABILITY: the intervention CROSSES THE FOLD; switch "
            "loses bistability (NUDGE-LIM-013)."
        )
    elif safety.high_risk_of_instability:
        bound = " (a one-sided LOWER bound)" if safety.one_sided else ""
        before = cast(float, safety.proximity_before)
        after = cast(float, safety.proximity_after)
        tail = (
            f" — HIGH RISK OF INSTABILITY: pushes the switch toward its fold "
            f"(proximity {before:.2f}->{after:.2f}{bound}; NUDGE-LIM-013)."
        )
    elif safety.proximity_before is None:
        tail = " — safety: base circuit is not bistable (no switch to destabilize)."
    else:
        after = cast(float, safety.proximity_after)
        tail = (
            f" — safety: OK, stays away from the fold "
            f"(proximity {safety.proximity_before:.2f}->{after:.2f})."
        )
    return head + tail


# --------------------------------------------------------------------------- #
# Curve-level inversion (real-data surface) — closed-form Hill inverse, NO safety gate.
# --------------------------------------------------------------------------- #
def _invert_curve(res: Any, target_state: Any) -> InterventionPlan | AbstentionResult:
    fit = res.fit
    floor, amp, k, n = fit.floor, fit.amp, fit.k_threshold, fit.n
    direction = fit.direction
    y = float(np.asarray(target_state, dtype=float).ravel()[0])

    lo, hi = floor, floor + amp
    if not (min(lo, hi) < y < max(lo, hi)):
        return AbstentionResult(
            MechanismClass.UNRESOLVED,
            f"reachability abstention: target y={y:.3g} is outside the curve's "
            f"achievable range ({min(lo, hi):.3g}, {max(lo, hi):.3g}) = "
            f"(floor, floor+amp). No dose achieves it — NUDGE will not extrapolate "
            f"beyond the fitted saturating asymptotes (NUDGE-LIM-013).",
        )

    span = y - floor
    if direction == "repress":
        dose = k * ((amp / span) - 1.0) ** (1.0 / n)
    else:  # activate
        dose = k * (span / (amp - span)) ** (1.0 / n)

    in_range = fit.dose_min <= dose <= fit.dose_max
    extrap = (
        ""
        if in_range
        else " (NOTE: the required dose is OUTSIDE the observed dose range "
        f"[{fit.dose_min:.3g}, {fit.dose_max:.3g}] — an extrapolation; NUDGE-LIM-013)"
    )
    reason = (
        f"invert the {direction} Hill (apparent gain n={n:.2f}, K={k:.3g}): to reach "
        f"response y={y:.3g}, apply dose={dose:.3g}{extrap}. Curve mode carries NO "
        f"bifurcation safety gate — no circuit/fold to score (stated honestly)."
    )
    return InterventionPlan(
        mode="dose",
        dose=float(dose),
        predicted_response=y,
        achieved_loss=0.0,
        safety=None,
        reason=reason,
    )


# --------------------------------------------------------------------------- #
# The public verb.
# --------------------------------------------------------------------------- #
def design(
    attribution: AttributionResult,
    target_state: Any,
    *,
    free: list[FreeParam] | None = None,
    steps: int = 300,
    seed: int = 0,
    l1: float = 1e-2,
    tol: float = 0.05,
    learning_rate: float = 0.05,
    readout: Readout | None = None,
    margin: float = 0.15,
    start: Any = "low",
) -> InterventionPlan | AbstentionResult:
    """Invert a *reliable* attribution to propose an intervention reaching a target.

    **Integrity gate first:** if ``attribution.is_reliable`` is false (an abstained
    dose-response, a low-confidence fit), returns an :class:`AbstentionResult` at once
    — NUDGE never designs off a fit it does not trust.

    Then dispatches by the invertible substrate the attribution carries:

    - a :class:`CircuitFit` (``.circuit`` / ``.free``) → **circuit-level** gradient
      inversion over log-delta Δ on the fitted kinetics, with the Cap-5 bifurcation
      **safety gate**. ``target_state`` is a readout vector (see :func:`flip_target`
      for "flip ON/OFF"). ``start`` seeds the solve at the current stable state to
      intervene *from* — ``"low"`` (the resting basin; the default, for "flip ON"),
      ``"high"`` (to adjust the ON level), or an explicit activity vector. Abstains
      (reachability) if no Δ closes ``tol`` of the gap.
    - a ``DoseResponseResult`` (``.fit``) → **curve-level** closed-form Hill inverse to
      the dose achieving scalar response ``target_state``. No circuit ⇒ ``safety=None``
      (stated). Abstains if ``target_state`` is outside ``(floor, floor+amp)``.

    Returns an :class:`InterventionPlan` (the proposal) or an :class:`AbstentionResult`.
    Every proposal is valid only within the fit's identifiable region — extrapolation
    is a documented risk (``NUDGE-LIM-013``).
    """
    if not getattr(attribution, "is_reliable", False):
        return AbstentionResult(
            MechanismClass.UNRESOLVED,
            "integrity gate: design() refuses to invert an unreliable attribution — "
            f"{getattr(attribution, 'reason', 'no reason given')}. Diagnose reliably "
            "before prescribing (never design off a fit you do not trust).",
        )

    circuit = getattr(attribution, "circuit", None)
    if circuit is not None and hasattr(attribution, "free"):
        return _invert_circuit(
            cast("CircuitFit", attribution),
            target_state,
            free=free,
            steps=steps,
            seed=seed,
            l1=l1,
            tol=tol,
            learning_rate=learning_rate,
            readout=readout,
            margin=margin,
            start=start,
        )
    if getattr(attribution, "fit", None) is not None:
        return _invert_curve(attribution, target_state)

    raise TypeError(
        "design(): the attribution carries no invertible substrate — expected a "
        "CircuitFit (.circuit/.free) or a DoseResponseResult (.fit)."
    )
