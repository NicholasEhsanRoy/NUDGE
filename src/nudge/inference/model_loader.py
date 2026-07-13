"""Dynamic model ingestion for the ``identifiability`` / ``oed`` tools.

**What this adds.** The general :func:`nudge.service.identifiability_tool` /
:func:`nudge.service.oed_tool` normally take a model **by name** from the registry
(:mod:`nudge.inference.model_registry`). This loader lets them instead analyse a **user's own
differentiable model file** — supplied as an absolute ``path`` or as inline ``code`` — so the
same white-box machinery (matrix-free Fisher-information identifiability, gradient OED) runs on
a model NUDGE has never seen. It is the symmetric half of the "with-vs-without NUDGE" demo: the
raw agent reads the model file directly; NUDGE ingests the *same* file here.

**The loader interface (a convention, not an import).** A model file MUST NOT need to import
``nudge`` — it returns plain callables / arrays. It exposes one or both builders by name:

- ``nudge_identifiability(**opts) -> dict`` with keys
  ``{"predict_fn", "theta0", "param_names", "sigma"}`` — ``predict_fn(theta) -> observations`` is
  JAX-autodiff-differentiable in ``theta`` (RAW positive params, so ``θ·∂/∂θ`` is the
  log-sensitivity). Wrapped into an :class:`~nudge.inference.model_registry.IdentifiabilityProblem`.
  ``opts`` carries ``n_free`` / ``seed`` / ``sigma``.
- ``nudge_oed(**opts) -> dict`` with keys
  ``{"observe", "theta0", "param_names", "phi_bounds", "sigma"}`` — ``observe(theta, phi)`` is
  JAX-diff in both. Wrapped into a :class:`nudge.inference.oed.DesignProblem`. ``opts`` carries
  ``target`` / ``sigma`` / ``seed``.

**Security (``NUDGE-LIM-030``).** Loading a ``path`` / ``code`` **executes arbitrary user Python
in the server process** — exactly like running ``python your_model.py`` yourself. It is a LOCAL,
trusted-input convenience; it is NOT safe for untrusted / multi-tenant input. The registry-name
path is unaffected (no code execution beyond the shipped builders).

Additive / opt-in and self-contained: it only *composes* the existing problem types; it touches
neither ``fit.py`` nor ``core/`` (the frozen-core constraint).
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import uuid
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any, cast

import numpy as np

from nudge.inference.model_registry import IdentifiabilityProblem

__all__ = [
    "IDENTIFIABILITY_BUILDER",
    "OED_BUILDER",
    "load_module",
    "load_identifiability_problem",
    "load_oed_problem",
]

#: the by-convention builder names a user model file must expose.
IDENTIFIABILITY_BUILDER = "nudge_identifiability"
OED_BUILDER = "nudge_oed"

_IDENT_KEYS = ("predict_fn", "theta0", "param_names", "sigma")
_OED_KEYS = ("observe", "theta0", "param_names", "phi_bounds", "sigma")


def load_module(*, path: str | None = None, code: str | None = None) -> ModuleType:
    """Load a user model file into a FRESH, uniquely-named synthetic module.

    From ``path`` via ``importlib.util.spec_from_file_location`` + ``module_from_spec`` +
    ``exec_module``; from ``code`` via ``exec`` into a fresh module namespace. A unique synthetic
    module name is used and it is NOT left in :data:`sys.modules` (so repeated calls don't leak
    into one another). Exactly one of ``path`` / ``code`` must be given.

    **Executes arbitrary user Python** (``NUDGE-LIM-030``) — local, trusted-input only.
    """
    if (path is None) == (code is None):
        raise ValueError("load_module needs exactly one of path= or code=")
    mod_name = f"_nudge_dynmodel_{uuid.uuid4().hex}"
    if code is not None:
        module = ModuleType(mod_name)
        module.__dict__["__name__"] = mod_name
        try:
            exec(compile(code, f"<{mod_name}>", "exec"), module.__dict__)  # noqa: S102
        except Exception as exc:  # pragma: no cover - surfaced to the caller with context
            raise ValueError(f"failed to execute model_code: {exc}") from exc
        return module
    assert path is not None  # guaranteed by the exactly-one check above (type-checker narrowing)
    abs_path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location(mod_name, abs_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load a Python module from path {path!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # transient: some tooling needs the module resolvable while
    try:                            # its body executes; removed immediately after (no leak).
        # Put the model file's own directory on sys.path so sibling imports resolve — exactly as
        # they would if you ran `python your_model.py` (the NLME model imports its base model).
        with _sys_path_prepended(os.path.dirname(abs_path)):
            spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - surfaced to the caller with context
        raise ValueError(f"failed to import model from {path!r}: {exc}") from exc
    finally:
        sys.modules.pop(mod_name, None)
    return module


@contextlib.contextmanager
def _sys_path_prepended(directory: str) -> Iterator[None]:
    """Temporarily prepend ``directory`` to :data:`sys.path` (restored on exit)."""
    if directory and directory in sys.path:
        yield
        return
    sys.path.insert(0, directory)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(directory)


def _require_builder(module: ModuleType, builder_name: str) -> Any:
    fn = getattr(module, builder_name, None)
    if not callable(fn):
        public = sorted(n for n in vars(module) if not n.startswith("_"))
        raise ValueError(
            f"model file must define a callable `{builder_name}(**opts)`; "
            f"top-level names found: {public}"
        )
    return fn


def _validate_dict(value: Any, keys: tuple[str, ...], builder_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        got = type(value).__name__
        raise ValueError(
            f"`{builder_name}` must return a dict with keys {list(keys)}; got {got}"
        )
    missing = [k for k in keys if k not in value]
    if missing:
        raise ValueError(
            f"`{builder_name}` return is missing key(s) {missing}; it must supply {list(keys)}"
        )
    return value


def load_identifiability_problem(
    *,
    path: str | None = None,
    code: str | None = None,
    n_free: int = 0,
    seed: int = 0,
    sigma: float | None = None,
) -> IdentifiabilityProblem:
    """Load a user model file and wrap its ``nudge_identifiability`` builder into a problem.

    Calls ``module.nudge_identifiability(n_free=…, seed=…, sigma=…)``, validates the returned dict
    has ``{"predict_fn", "theta0", "param_names", "sigma"}`` (naming any missing key), and wraps it
    into an :class:`~nudge.inference.model_registry.IdentifiabilityProblem` consumed by the
    matrix-free sloppiness diagnostic. ``predict_fn`` must be JAX-autodiff-differentiable in
    ``theta`` (RAW positive params). Executes arbitrary user Python (``NUDGE-LIM-030``).
    """
    module = load_module(path=path, code=code)
    builder = _require_builder(module, IDENTIFIABILITY_BUILDER)
    spec = _validate_dict(
        builder(n_free=n_free, seed=seed, sigma=sigma), _IDENT_KEYS, IDENTIFIABILITY_BUILDER
    )
    predict_fn = spec["predict_fn"]
    if not callable(predict_fn):
        raise ValueError(
            f"`{IDENTIFIABILITY_BUILDER}['predict_fn']` must be callable; "
            f"got {type(predict_fn).__name__}"
        )
    theta0 = np.asarray(spec["theta0"], dtype=np.float64)
    if theta0.ndim != 1 or theta0.shape[0] < 1:
        raise ValueError(
            f"`{IDENTIFIABILITY_BUILDER}['theta0']` must be a 1-D array of ≥1 param; "
            f"got shape {theta0.shape}"
        )
    names = tuple(str(n) for n in spec["param_names"])
    if len(names) != theta0.shape[0]:
        raise ValueError(
            f"`{IDENTIFIABILITY_BUILDER}`: len(param_names)={len(names)} != len(theta0)="
            f"{theta0.shape[0]}"
        )
    meta = dict(spec.get("meta", {})) if isinstance(spec.get("meta"), dict) else {}
    meta.setdefault("source", "dynamic (model_path/model_code)")
    meta.setdefault("domain", str(spec.get("domain", "user model")))
    return IdentifiabilityProblem(
        predict_fn=predict_fn,
        theta0=theta0,
        param_names=names,
        sigma=float(spec["sigma"]),
        meta=meta,
    )


def load_oed_problem(
    *,
    path: str | None = None,
    code: str | None = None,
    target: str | None = None,
    sigma: float | None = None,
    seed: int = 0,
) -> Any:
    """Load a user model file and wrap its ``nudge_oed`` builder into a ``DesignProblem``.

    Calls ``module.nudge_oed(target=…, sigma=…, seed=…)``, validates the returned dict has
    ``{"observe", "theta0", "param_names", "phi_bounds", "sigma"}`` (naming any missing key), and
    wraps it into a :class:`nudge.inference.oed.DesignProblem` consumed by the gradient OED.
    ``observe(theta, phi)`` must be JAX-diff in both. Executes arbitrary user Python
    (``NUDGE-LIM-030``).
    """
    from nudge.inference.oed import DesignProblem

    module = load_module(path=path, code=code)
    builder = _require_builder(module, OED_BUILDER)
    spec = _validate_dict(
        builder(target=target, sigma=sigma, seed=seed), _OED_KEYS, OED_BUILDER
    )
    observe = spec["observe"]
    if not callable(observe):
        raise ValueError(
            f"`{OED_BUILDER}['observe']` must be callable; got {type(observe).__name__}"
        )
    observe = cast("Callable[[Any, Any], Any]", observe)
    theta0 = np.asarray(spec["theta0"], dtype=np.float64)
    if theta0.ndim != 1 or theta0.shape[0] < 1:
        raise ValueError(
            f"`{OED_BUILDER}['theta0']` must be a 1-D array of ≥1 param; got shape {theta0.shape}"
        )
    names = tuple(str(n) for n in spec["param_names"])
    if len(names) != theta0.shape[0]:
        raise ValueError(
            f"`{OED_BUILDER}`: len(param_names)={len(names)} != len(theta0)={theta0.shape[0]}"
        )
    bounds = tuple(float(b) for b in spec["phi_bounds"])
    if len(bounds) != 2 or not bounds[0] < bounds[1]:
        raise ValueError(
            f"`{OED_BUILDER}['phi_bounds']` must be (lo, hi) with lo<hi; got {spec['phi_bounds']!r}"
        )
    meta = dict(spec.get("meta", {})) if isinstance(spec.get("meta"), dict) else {}
    meta.setdefault("source", "dynamic (model_path/model_code)")
    meta.setdefault("domain", str(spec.get("domain", "user model")))
    return DesignProblem(
        observe=observe,
        theta0=theta0,
        param_names=names,
        sigma=float(spec["sigma"]),
        phi_bounds=(bounds[0], bounds[1]),
        meta=meta,
    )
