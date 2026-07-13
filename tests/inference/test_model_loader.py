"""Unit tests for dynamic model ingestion (:mod:`nudge.inference.model_loader`).

Fast: a tiny inline linear model exercises load-from-code, load-from-path, the wrapping into the
problem types, the sibling-import path, the no-``sys.modules``-leak guarantee, and the clear
errors on a missing builder / malformed return / bad source. The heavy numerical-parity checks
(loaded == registry to machine precision) live in ``tests/mcp/test_dynamic_model_ingestion.py``.
"""

from __future__ import annotations

import sys
import textwrap

import numpy as np
import pytest

from nudge.inference.model_loader import (
    load_identifiability_problem,
    load_module,
    load_oed_problem,
)

# a tiny, well-conditioned linear model file (no `nudge` import) exposing both builders.
_TINY = textwrap.dedent(
    """
    import jax.numpy as jnp
    import numpy as np

    def _predict(theta):
        t = jnp.linspace(0.0, 1.0, 5)
        return theta[0] * t + theta[1]

    def nudge_identifiability(n_free=0, seed=0, sigma=None):
        return {"predict_fn": _predict, "theta0": np.array([2.0, 1.0]),
                "param_names": ("slope", "offset"),
                "sigma": 0.05 if sigma is None else float(sigma)}

    def nudge_oed(target=None, sigma=None, seed=0):
        def observe(theta, phi):
            return theta[0] * phi + theta[1]
        return {"observe": observe, "theta0": np.array([2.0, 1.0]),
                "param_names": ("slope", "offset"), "phi_bounds": (0.0, 1.0),
                "sigma": 0.05 if sigma is None else float(sigma)}
    """
)


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return str(p)


def test_load_identifiability_from_code_and_path_agree(tmp_path):
    """The builder loads identically from inline code and from a file path."""
    from_code = load_identifiability_problem(code=_TINY)
    from_path = load_identifiability_problem(path=_write(tmp_path, "m.py", _TINY))
    assert from_code.param_names == from_path.param_names == ("slope", "offset")
    np.testing.assert_allclose(from_code.theta0, [2.0, 1.0])
    assert from_code.sigma == 0.05
    # predict_fn is a real JAX-differentiable map (θ·∂/∂θ makes sense on RAW params).
    import jax

    y = np.asarray(from_code.predict_fn(from_code.theta0))
    assert y.shape == (5,)
    jac = jax.jacobian(from_code.predict_fn)(from_code.theta0)
    assert np.asarray(jac).shape == (5, 2)


def test_load_oed_wraps_designproblem(tmp_path):
    """The oed builder wraps into a DesignProblem with the right observe / bounds / names."""
    prob = load_oed_problem(path=_write(tmp_path, "m.py", _TINY), target="slope")
    assert prob.param_names == ("slope", "offset")
    assert prob.phi_bounds == (0.0, 1.0)
    import jax.numpy as jnp

    obs = np.asarray(prob.observe(jnp.asarray(prob.theta0), jnp.asarray([0.0, 0.5, 1.0])))
    np.testing.assert_allclose(obs, [1.0, 2.0, 3.0], rtol=1e-5)


def test_sigma_override_passes_through(tmp_path):
    prob = load_identifiability_problem(code=_TINY, sigma=0.2)
    assert prob.sigma == 0.2


def test_missing_builder_is_a_clear_error():
    with pytest.raises(ValueError, match="nudge_identifiability"):
        load_identifiability_problem(code="x = 1\n")


def test_malformed_return_names_the_missing_key():
    bad = "def nudge_identifiability(**k):\n    return {'predict_fn': lambda t: t}\n"
    with pytest.raises(ValueError, match="missing key"):
        load_identifiability_problem(code=bad)


def test_non_dict_return_is_a_clear_error():
    bad = "def nudge_identifiability(**k):\n    return 42\n"
    with pytest.raises(ValueError, match="must return a dict"):
        load_identifiability_problem(code=bad)


def test_param_names_length_mismatch_raises():
    bad = (
        "import numpy as np\n"
        "def nudge_identifiability(**k):\n"
        "    return {'predict_fn': lambda t: t, 'theta0': np.array([1.0, 2.0]),\n"
        "            'param_names': ('only_one',), 'sigma': 0.1}\n"
    )
    with pytest.raises(ValueError, match="param_names"):
        load_identifiability_problem(code=bad)


def test_load_module_requires_exactly_one_source():
    with pytest.raises(ValueError, match="exactly one"):
        load_module()
    with pytest.raises(ValueError, match="exactly one"):
        load_module(path="x.py", code="y=1")


def test_load_module_does_not_leak_into_sys_modules(tmp_path):
    before = set(sys.modules)
    load_module(path=_write(tmp_path, "m.py", _TINY))
    load_module(code=_TINY)
    leaked = [n for n in set(sys.modules) - before if n.startswith("_nudge_dynmodel_")]
    assert leaked == []


def test_sibling_import_resolves_from_the_files_directory(tmp_path):
    """A model file that imports a sibling module resolves it (dir added to sys.path)."""
    _write(tmp_path, "base_model.py", "SLOPE = 3.0\n")
    child = (
        "import numpy as np\n"
        "from base_model import SLOPE\n"
        "def nudge_identifiability(**k):\n"
        "    return {'predict_fn': lambda t: SLOPE * t, 'theta0': np.array([1.0]),\n"
        "            'param_names': ('a',), 'sigma': 0.1}\n"
    )
    prob = load_identifiability_problem(path=_write(tmp_path, "child.py", child))
    assert prob.param_names == ("a",)
    # and the sibling dir was not left permanently on sys.path
    assert str(tmp_path) not in sys.path
