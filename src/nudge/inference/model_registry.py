"""A general model registry for the ``identifiability`` and ``oed`` tools.

**The generality proof.** NUDGE's matrix-free identifiability
(:mod:`nudge.inference.sloppiness`) and gradient optimal experimental design
(:mod:`nudge.inference.oed`) are *model-agnostic*: they work on **any** differentiable
forward model. This module makes that concrete by registering a handful of genuinely
different differentiable models — an ecological Lotka–Volterra community, a linear reaction
cascade, a published Alzheimer's amyloid-β QSP model, a single-species logistic growth
curve, and the canonical sloppy/structurally-redundant/well-conditioned toy models — behind
one name→builder table, so the two tools can be pointed at a model **by reference** (a
string), run the *real* analysis, and return whatever it measures. If they only worked on
one demo model this table would have one entry; it has several, across domains, on purpose.

Two problem shapes, one per tool:

- :class:`IdentifiabilityProblem` — a differentiable ``predict_fn(theta) -> observations``
  at a nominal ``theta0`` (RAW parameters, so the sloppiness diagnostic's ``θ·∂/∂θ``
  recovers the log-sensitivity), consumed by
  :func:`nudge.inference.sloppiness.sloppiness_diagnostic_matrixfree`.
- :class:`nudge.inference.oed.DesignProblem` — a differentiable ``observe(theta, phi)`` over
  a design knob ``phi`` (the measurement schedule), consumed by
  :func:`nudge.inference.oed.optimize_design`.

**Registering your own model** takes a few lines — build a small function that returns an
:class:`IdentifiabilityProblem` and/or a :class:`~nudge.inference.oed.DesignProblem`, then
call :func:`register_model`::

    from nudge.inference.model_registry import register_model, IdentifiabilityProblem

    def my_ident(**opts):
        return IdentifiabilityProblem(predict_fn=my_predict, theta0=theta0,
                                      param_names=("a", "b"), sigma=0.05)

    register_model("my_model", summary="my ODE", domain="chemistry",
                   identifiability_builder=my_ident)

Arbitrary user models that don't fit these two shapes remain a plain ``import nudge`` library
path (call :func:`nudge.inference.sloppiness.analyze_model_matrixfree` /
:func:`nudge.inference.oed.optimize_design` directly) — the registry is the *convenience*
surface the MCP tools drive by name, not the only way in (``NUDGE-LIM-027``).

Additive / opt-in and self-contained: it only *composes* existing builders
(:mod:`nudge.inference.adjoint`, :mod:`nudge.inference.oed`,
:mod:`nudge.inference.sloppiness`, :mod:`nudge.mechanisms.ad_qsp`); it touches neither
``fit.py`` nor ``core/``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "IdentifiabilityProblem",
    "RegisteredModel",
    "register_model",
    "list_models",
    "get_model",
    "build_identifiability_problem",
    "build_oed_problem",
    "restrict_identifiability_problem",
]


# --------------------------------------------------------------------------- #
# problem container (OED reuses nudge.inference.oed.DesignProblem verbatim)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IdentifiabilityProblem:
    """A differentiable ``predict_fn(theta) -> observations`` + its nominal point.

    ``predict_fn`` is autodiff-differentiable in ``theta`` (RAW, not log, parameters — so the
    sloppiness diagnostic's ``θ·∂/∂θ`` recovers the log-sensitivity, matching
    :mod:`nudge.inference.sloppiness`). ``theta0`` is the nominal parameter vector the FIM is
    evaluated at; ``param_names`` labels it; ``sigma`` is the iid Gaussian observation noise
    the FIM is scaled by. ``meta`` carries model provenance for the report / figure.
    """

    predict_fn: Callable[[Any], Any]
    theta0: np.ndarray
    param_names: tuple[str, ...]
    sigma: float
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_params(self) -> int:
        return int(np.asarray(self.theta0).shape[0])


@dataclass(frozen=True)
class RegisteredModel:
    """One registry entry: a named model + the builders the two tools drive it through."""

    name: str
    summary: str
    domain: str
    identifiability_builder: Callable[..., IdentifiabilityProblem] | None = None
    oed_builder: Callable[..., Any] | None = None
    default_oed_target: str | None = None

    @property
    def supports_identifiability(self) -> bool:
        return self.identifiability_builder is not None

    @property
    def supports_oed(self) -> bool:
        return self.oed_builder is not None


# --------------------------------------------------------------------------- #
# the registry + public API
# --------------------------------------------------------------------------- #
_MODELS: dict[str, RegisteredModel] = {}


def register_model(
    name: str,
    *,
    summary: str,
    domain: str,
    identifiability_builder: Callable[..., IdentifiabilityProblem] | None = None,
    oed_builder: Callable[..., Any] | None = None,
    default_oed_target: str | None = None,
) -> None:
    """Register a model under ``name`` (see the module docstring for the few-line recipe).

    At least one of ``identifiability_builder`` / ``oed_builder`` must be given. Re-registering
    a name replaces it (so a user can override a shipped model).
    """
    if identifiability_builder is None and oed_builder is None:
        raise ValueError(
            f"model {name!r} needs an identifiability_builder and/or an oed_builder"
        )
    _MODELS[name] = RegisteredModel(
        name=name,
        summary=summary,
        domain=domain,
        identifiability_builder=identifiability_builder,
        oed_builder=oed_builder,
        default_oed_target=default_oed_target,
    )


def get_model(name: str) -> RegisteredModel:
    """Return the :class:`RegisteredModel` for ``name`` (raises with the known names)."""
    if name not in _MODELS:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(_MODELS)}")
    return _MODELS[name]


def list_models() -> list[dict[str, Any]]:
    """List the registered models (name, summary, domain, which tools each supports)."""
    return [
        {
            "name": m.name,
            "summary": m.summary,
            "domain": m.domain,
            "supports_identifiability": m.supports_identifiability,
            "supports_oed": m.supports_oed,
            "default_oed_target": m.default_oed_target,
        }
        for m in sorted(_MODELS.values(), key=lambda x: x.name)
    ]


def restrict_identifiability_problem(
    problem: IdentifiabilityProblem, free_names: list[str]
) -> IdentifiabilityProblem:
    """Restrict a problem to a **subset** of its free parameters (the rest held at ``theta0``).

    The general "which of THESE parameters are identifiable" knob: builds a predict_fn over
    just the named subset, scattering them back into the full ``theta0`` before the forward
    solve, so the FIM is computed over exactly the chosen parameters. Names not present raise.
    """
    import jax.numpy as jnp

    names = list(problem.param_names)
    missing = [n for n in free_names if n not in names]
    if missing:
        raise ValueError(f"unknown free parameter(s) {missing}; available: {names}")
    keep = np.array([names.index(n) for n in free_names], dtype=int)
    theta_full0 = jnp.asarray(np.asarray(problem.theta0, dtype=np.float64))
    keep_j = jnp.asarray(keep)
    inner = problem.predict_fn

    def predict(theta_sub: Any) -> Any:
        full = theta_full0.at[keep_j].set(theta_sub.astype(theta_full0.dtype))
        return inner(full)

    return IdentifiabilityProblem(
        predict_fn=predict,
        theta0=np.asarray(problem.theta0, dtype=np.float64)[keep],
        param_names=tuple(free_names),
        sigma=problem.sigma,
        meta={**problem.meta, "restricted_to": list(free_names)},
    )


def build_identifiability_problem(
    name: str,
    *,
    free: list[str] | None = None,
    n_free: int = 0,
    sigma: float | None = None,
    seed: int = 0,
    scale: int = 0,
) -> IdentifiabilityProblem:
    """Build the :class:`IdentifiabilityProblem` for registered model ``name``.

    ``n_free`` / ``scale`` are the population-/dimension-scale knob passed through to models
    that support it (``glv`` / ``linear_pathway`` / ``ad_qsp`` — how many parameters are
    jointly estimated); ``free`` restricts to a named subset
    (:func:`restrict_identifiability_problem`); ``sigma`` overrides the observation noise.
    Raises if the model has no
    identifiability builder.
    """
    model = get_model(name)
    if model.identifiability_builder is None:
        raise ValueError(f"model {name!r} does not support the identifiability tool")
    problem = model.identifiability_builder(
        n_free=n_free or scale, sigma=sigma, seed=seed
    )
    if free:
        problem = restrict_identifiability_problem(problem, free)
    return problem


def build_oed_problem(
    name: str,
    *,
    target: str | None = None,
    sigma: float | None = None,
    seed: int = 0,
    **opts: Any,
) -> Any:
    """Build the :class:`~nudge.inference.oed.DesignProblem` for registered model ``name``.

    ``target`` selects the parameter to resolve (default: the model's ``default_oed_target``);
    ``sigma`` overrides the observation noise. Extra ``opts`` pass through to the builder.
    Raises if the model has no OED builder.
    """
    model = get_model(name)
    if model.oed_builder is None:
        raise ValueError(f"model {name!r} does not support the OED tool")
    return model.oed_builder(target=target, sigma=sigma, seed=seed, **opts)


# --------------------------------------------------------------------------- #
# shipped builders — identifiability
# --------------------------------------------------------------------------- #
def _glv_ident(*, n_free: int = 0, sigma: float | None = None, seed: int = 0, **_: Any):
    """gLV community trajectory identifiability (ecology). Free = the first ``n_free`` of the
    ``[α | vec(β) | ε]`` kinetics; α⇄βᵢᵢ is the canonical near-equilibrium sloppy pair."""
    import jax.numpy as jnp

    from nudge.inference.adjoint import make_glv_problem, ode_trajectory_predict_fn

    n_species = 6
    nf = int(n_free) if n_free and n_free > 0 else 18
    prob = make_glv_problem(n_species=n_species, n_free=nf, seed=seed, dtype=jnp.float64)
    predict = ode_trajectory_predict_fn(prob)
    names = _glv_param_names(n_species, prob.n_theta)
    return IdentifiabilityProblem(
        predict_fn=predict,
        theta0=np.asarray(prob.theta0, dtype=np.float64),
        param_names=names,
        sigma=1e-2 if sigma is None else float(sigma),
        meta={"model": "glv", "domain": "microbiome ecology", "n_species": n_species},
    )


def _glv_param_names(n_species: int, n_theta: int) -> tuple[str, ...]:
    """Names for the ``[α (S) | vec(β) (S²) | ε (S)]`` free-parameter vector's first entries."""
    # NOTE: names are comma-free (``beta[i][j]``, not ``beta[i,j]``) so the MCP tool's
    # comma-separated ``free=`` subset selector parses them unambiguously.
    labels: list[str] = [f"alpha[{i}]" for i in range(n_species)]
    labels += [f"beta[{i}][{j}]" for i in range(n_species) for j in range(n_species)]
    labels += [f"eps[{i}]" for i in range(n_species)]
    return tuple(labels[:n_theta])


def _linear_pathway_ident(
    *, n_free: int = 0, sigma: float | None = None, seed: int = 0, **_: Any
):
    """Linear reaction cascade ``x₀→x₁→…`` identifiability (chemistry). A well-posed,
    monotone network — the well-behaved contrast to the sloppy gLV."""
    import jax.numpy as jnp

    from nudge.inference.adjoint import make_linear_pathway_problem, ode_trajectory_predict_fn

    n_states = 8
    nf = int(n_free) if n_free and n_free > 0 else n_states
    prob = make_linear_pathway_problem(
        n_states=n_states, n_free=nf, seed=seed, dtype=jnp.float64
    )
    predict = ode_trajectory_predict_fn(prob)
    names = tuple(f"k[{i}]" for i in range(prob.n_theta))
    return IdentifiabilityProblem(
        predict_fn=predict,
        theta0=np.asarray(prob.theta0, dtype=np.float64),
        param_names=names,
        sigma=1e-2 if sigma is None else float(sigma),
        meta={"model": "linear_pathway", "domain": "reaction kinetics", "n_states": n_states},
    )


def _ad_qsp_ident(*, n_free: int = 0, sigma: float | None = None, seed: int = 0, **_: Any):
    """Alzheimer's amyloid-β QSP population identifiability (clinical pharmacology). Each
    subject carries its own kinetics; ``n_free`` grows the calibration's dimensionality
    (nonlinear mixed effects). With a sparse biomarker budget it is genuinely rank-deficient
    (NUDGE-LIM-023 fail-safe: certified ``unidentifiable`` by shape). Synthetic cohort,
    demo-scaled constants (NUDGE-LIM-026)."""
    from nudge.mechanisms.ad_qsp import AD_PARAM_VALUES, make_ad_cohort_predict_fn

    per = int(AD_PARAM_VALUES.shape[0])
    nf = int(n_free) if n_free and n_free > 0 else 24
    n_subjects = max(20, -(-nf // per))  # ceil(nf/per), ≥20
    cohort = make_ad_cohort_predict_fn(n_subjects=n_subjects, n_free=nf, seed=seed)
    return IdentifiabilityProblem(
        predict_fn=cohort.predict_fn,
        theta0=np.asarray(cohort.theta0, dtype=np.float64),
        param_names=tuple(cohort.param_names),
        sigma=0.05 if sigma is None else float(sigma),
        meta={"model": "ad_qsp", "domain": "clinical pharmacology (Alzheimer's Aβ)",
              "n_subjects": cohort.n_subjects, "n_obs": cohort.n_obs,
              "note": "synthetic cohort, demo-scaled (NUDGE-LIM-026)"},
    )


def _canonical_ident(kind: str):
    """A builder for a canonical sloppiness model (the honesty decoys + the sloppy showcase)."""

    def build(*, sigma: float | None = None, **_: Any) -> IdentifiabilityProblem:
        import numpy as _np

        from nudge.inference.sloppiness import (
            redundant_exponential_predict,
            sum_of_exponentials_predict,
            well_conditioned_predict,
        )

        t = _np.linspace(0.05, 6.0, 60)
        if kind == "sum_of_exponentials":
            predict = sum_of_exponentials_predict(
                rates=[0.5, 1.3, 2.5, 4.5], amps=[1.0, 1.0, 1.0, 1.0], t=t
            )
            domain, note = "canonical (sloppy)", "sloppy-but-predictive showcase"
        elif kind == "redundant_exponential":
            predict = redundant_exponential_predict(amp=1.0, k1=0.7, k2=0.9, t=t)
            domain, note = "canonical (structural null)", "DECOY: must abstain (unidentifiable)"
        elif kind == "well_conditioned":
            predict = well_conditioned_predict(slope=2.0, offset=1.0, t=t)
            domain, note = "canonical (well-posed)", "DECOY: must NOT be flagged sloppy"
        else:  # pragma: no cover - guarded by the registry
            raise ValueError(f"unknown canonical model {kind!r}")
        return IdentifiabilityProblem(
            predict_fn=predict,
            theta0=_np.asarray(predict.theta0, dtype=_np.float64),  # type: ignore[attr-defined]
            param_names=tuple(predict.names),  # type: ignore[attr-defined]
            sigma=0.01 if sigma is None else float(sigma),
            meta={"model": kind, "domain": domain, "note": note},
        )

    return build


def _logistic_ident(*, sigma: float | None = None, **_: Any):
    """Single-species logistic growth identifiability (population dynamics), derived from the
    OED design problem observed on a rich time grid (α⇄β is the growth/carrying-capacity
    degeneracy)."""
    return _ident_from_design("logistic")(sigma=sigma)


def _ident_from_design(name: str):
    """Derive an :class:`IdentifiabilityProblem` from a registered OED model observed on a rich
    default schedule — RAW-parameter wrapped so ``θ·∂/∂θ`` is the correct log-sensitivity."""

    def build(*, sigma: float | None = None, **_: Any) -> IdentifiabilityProblem:
        import jax.numpy as jnp
        import numpy as _np

        design = build_oed_problem(name)
        lo, hi = design.phi_bounds
        phi_grid = jnp.asarray(_np.linspace(lo, hi, 24), dtype=jnp.float64)
        names = tuple(n[len("log_"):] if n.startswith("log_") else n
                      for n in design.param_names)
        theta0_raw = _np.exp(_np.asarray(design.theta0, dtype=_np.float64))

        def predict(theta_raw: Any) -> Any:
            return design.observe(jnp.log(theta_raw), phi_grid)

        return IdentifiabilityProblem(
            predict_fn=predict,
            theta0=theta0_raw,
            param_names=names,
            sigma=float(design.sigma) if sigma is None else float(sigma),
            meta={"model": name, "domain": "population dynamics",
                  "note": "identifiability on a rich measurement grid"},
        )

    return build


# --------------------------------------------------------------------------- #
# shipped builders — OED (each returns a nudge.inference.oed.DesignProblem)
# --------------------------------------------------------------------------- #
def _logistic_oed(*, target: str | None = None, sigma: float | None = None,
                  seed: int = 0, **_: Any):
    from nudge.inference.oed import make_logistic_design_problem

    kw = {} if sigma is None else {"sigma": float(sigma)}
    return make_logistic_design_problem(**kw)


def _glv_oed(*, target: str | None = None, sigma: float | None = None,
             seed: int = 0, **_: Any):
    from nudge.inference.oed import make_glv_design_problem

    kw: dict[str, Any] = {"seed": seed}
    if sigma is not None:
        kw["sigma"] = float(sigma)
    return make_glv_design_problem(**kw)


def _ad_qsp_oed(*, target: str | None = None, sigma: float | None = None,
                seed: int = 0, pair: tuple[str, str] = ("k_on", "k_gl"), **_: Any):
    from nudge.mechanisms.ad_qsp import make_ad_oed_problem

    kw: dict[str, Any] = {"pair": pair, "target": pair[0]}
    if sigma is not None:
        kw["sigma"] = float(sigma)
    return make_ad_oed_problem(**kw)


# --------------------------------------------------------------------------- #
# populate the registry (the ≥3-models-per-tool generality proof)
# --------------------------------------------------------------------------- #
def _register_shipped() -> None:
    # identifiability + OED (one model, both tools)
    register_model(
        "glv", summary="generalized Lotka–Volterra community trajectory fit",
        domain="microbiome ecology",
        identifiability_builder=_glv_ident, oed_builder=_glv_oed,
        default_oed_target="log_alpha_t",
    )
    register_model(
        "ad_qsp", summary="Alzheimer's amyloid-β QSP (Proctor 2013; synthetic cohort)",
        domain="clinical pharmacology",
        identifiability_builder=_ad_qsp_ident, oed_builder=_ad_qsp_oed,
        default_oed_target="log_k_on",
    )
    register_model(
        "logistic", summary="single-species logistic growth (α⇄carrying-capacity)",
        domain="population dynamics",
        identifiability_builder=_logistic_ident, oed_builder=_logistic_oed,
        default_oed_target="log_alpha",
    )
    # identifiability only
    register_model(
        "linear_pathway", summary="linear reaction cascade x₀→x₁→… (well-posed)",
        domain="reaction kinetics",
        identifiability_builder=_linear_pathway_ident,
    )
    register_model(
        "sum_of_exponentials", summary="canonical sloppy model (loose params, tight fit)",
        domain="canonical", identifiability_builder=_canonical_ident("sum_of_exponentials"),
    )
    register_model(
        "redundant_exponential",
        summary="canonical structural null A·e^{-(k₁+k₂)t} (DECOY: must abstain)",
        domain="canonical", identifiability_builder=_canonical_ident("redundant_exponential"),
    )
    register_model(
        "well_conditioned",
        summary="canonical well-posed linear model (DECOY: must NOT be flagged sloppy)",
        domain="canonical", identifiability_builder=_canonical_ident("well_conditioned"),
    )


_register_shipped()
