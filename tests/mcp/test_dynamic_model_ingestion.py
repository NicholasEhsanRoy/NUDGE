"""Dynamic model ingestion for the ``identifiability`` / ``oed`` tools (NUDGE-LIM-030).

The load-bearing checks: analysing a user's OWN differentiable model file (by ``model_path`` /
``model_code``) is NUMERICALLY IDENTICAL to the registered model that mirrors the same math — the
verdict + FIM eigenspectrum match the registry ``ad_qsp`` / ``ad_qsp_nlme`` to machine precision,
the OED ×259 CRLB lift + the NUDGE-LIM-029 rank-deficiency guard reproduce exactly, and the
registry-name path is unchanged. Plus the clear-error surface (missing builder / ambiguous
source). The heavy ad_qsp round-trips are ``slow``; the error + tiny-model checks are fast.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from nudge.service import identifiability_tool, oed_tool

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEMO = os.path.abspath(os.path.join(_HERE, "..", "..", "scripts", "demo_ab"))
_AD_MODEL = os.path.join(_DEMO, "ad_qsp_model.py")
_AD_NLME_MODEL = os.path.join(_DEMO, "ad_qsp_nlme_model.py")


def _close(a: float, b: float, rtol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= rtol * max(abs(float(a)), abs(float(b)), 1e-30)


# --------------------------------------------------------------------------- #
# fast: source resolution + a tiny inline model (no ad_qsp cohort)
# --------------------------------------------------------------------------- #
_TINY = (
    "import jax.numpy as jnp\n"
    "import numpy as np\n"
    "def nudge_identifiability(n_free=0, seed=0, sigma=None):\n"
    "    return {'predict_fn': lambda th: th[0]*jnp.linspace(0,1,5)+th[1],\n"
    "            'theta0': np.array([2.0, 1.0]), 'param_names': ('slope','offset'),\n"
    "            'sigma': 0.05 if sigma is None else float(sigma)}\n"
)


def test_no_source_is_an_error():
    out = identifiability_tool()
    assert "error" in out and "registered" in out


def test_ambiguous_source_is_an_error():
    out = identifiability_tool("ad_qsp", model_path=_AD_MODEL)
    assert "error" in out and "ambiguous" in out["error"]
    out2 = oed_tool("ad_qsp", model_code=_TINY)
    assert "error" in out2 and "ambiguous" in out2["error"]


def test_bad_model_file_is_a_graceful_error():
    out = identifiability_tool(model_code="x = 1\n", with_figure=False)
    assert "error" in out and "failed to load" in out["error"]


def test_tiny_inline_model_round_trips_fast():
    out = identifiability_tool(model_code=_TINY, with_figure=False)
    assert out["source"] == "dynamic" and out["dynamic_ingestion"] is True
    assert out["registry_scope"] == "NUDGE-LIM-030"
    assert out["verdict"] in {"well-constrained", "sloppy-but-predictive", "unidentifiable"}
    assert out["param_names"] == ["slope", "offset"]


def test_registry_path_unchanged_by_the_new_signature():
    """Regression: model= by name still returns the registry verdict + LIM-027 scope."""
    out = identifiability_tool("linear_pathway", with_figure=False)
    assert out["source"] == "registry" and out["dynamic_ingestion"] is False
    assert out["registry_scope"] == "NUDGE-LIM-027"
    assert out["verdict"] == "well-constrained"


# --------------------------------------------------------------------------- #
# slow: numerical parity with the registered ad_qsp models (machine precision)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_identifiability_path_matches_registry_ad_qsp():
    reg = identifiability_tool("ad_qsp", with_figure=False)
    dyn = identifiability_tool(model_path=_AD_MODEL, with_figure=False)
    assert dyn["verdict"] == reg["verdict"]  # label parity
    assert dyn["n_params"] == reg["n_params"]
    assert dyn["param_names"] == reg["param_names"]
    for key in ("smallest_eigenvalue", "largest_eigenvalue", "cond_number"):
        assert _close(dyn[key], reg[key]), (key, dyn[key], reg[key])  # eigenspectrum parity ~1e-9
    assert dyn["dynamic_ingestion"] is True and dyn["registry_scope"] == "NUDGE-LIM-030"


@pytest.mark.slow
def test_identifiability_model_code_matches_registry_ad_qsp():
    with open(_AD_MODEL, encoding="utf-8") as fh:
        src = fh.read()
    reg = identifiability_tool("ad_qsp", with_figure=False)
    dyn = identifiability_tool(model_code=src, with_figure=False)
    assert dyn["verdict"] == reg["verdict"]
    assert _close(dyn["smallest_eigenvalue"], reg["smallest_eigenvalue"])
    assert _close(dyn["cond_number"], reg["cond_number"])


@pytest.mark.slow
def test_oed_path_matches_registry_and_reproduces_259():
    reg = oed_tool("ad_qsp", with_figure=False)
    dyn = oed_tool(model_path=_AD_MODEL, with_figure=False)
    # ×259 default preserved and the two paths agree to machine precision.
    assert 250.0 < dyn["crlb_improvement"] < 270.0
    assert _close(dyn["crlb_improvement"], reg["crlb_improvement"])
    assert _close(dyn["min_eig_improvement"], reg["min_eig_improvement"])
    assert dyn["naive_rank_deficient"] is False and reg["naive_rank_deficient"] is False


@pytest.mark.slow
def test_oed_path_rank_deficiency_guard_fires():
    """NUDGE-LIM-029: a naive baseline+end schedule (0, 12) does not identify k_on → LOWER BOUND."""
    guarded = oed_tool(model_path=_AD_MODEL, naive=[0.0, 12.0], with_figure=False)
    assert guarded["naive_rank_deficient"] is True
    assert guarded["naive_target_identifiable"] is False
    assert guarded["crlb_improvement_is_lower_bound"] is True
    assert "rank" in guarded["rank_deficiency_note"].lower()


@pytest.mark.slow
def test_nlme_path_matches_registry_and_preserves_failsafe():
    reg = identifiability_tool("ad_qsp_nlme", with_figure=False)
    dyn = identifiability_tool(model_path=_AD_NLME_MODEL, with_figure=False)
    # The coupled/arrowhead cohort is rank-deficient by shape → the unidentifiable fail-safe.
    assert reg["verdict"] == "unidentifiable"
    assert dyn["verdict"] == reg["verdict"]
    assert dyn["n_params"] == reg["n_params"]
    assert dyn["param_names"] == reg["param_names"]
    np.testing.assert_allclose(dyn["smallest_eigenvalue"], reg["smallest_eigenvalue"], atol=1e-12)
