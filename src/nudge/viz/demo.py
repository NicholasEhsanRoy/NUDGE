"""Zero-setup demo figure-data per renderer kind — the source for ``nudge viz --demo``.

:func:`demo_result` returns a renderable result/figure-data for any registered ``kind``,
so the CLI ``nudge viz KIND --demo`` and the MCP ``render_figure`` tool can draw every
figure with no input file. Where a genuine synthetic-analysis demo already exists in
:mod:`nudge.service` (``constitutive`` / ``temporal`` / ``oed`` / ``aggregation`` /
``attribution`` / ``identifiability``) it is reused, so the numbers are MEASURED, not
fabricated; ``dose_response`` builds the flagship switch-beside-abstain pair from a real
(synthetic) Hill fit. The remaining figures are illustrative *layout* examples with generic
labels (clearly examples, not measured findings) — their abstaining variants still drive
the honesty overlay, so the demo cannot misrepresent an abstention as a call.

Kept out of :mod:`nudge.service` so the demo catalogue lives next to the renderers it feeds;
the heavy (JAX / pipeline) demos are imported lazily inside :func:`demo_result`.
"""

from __future__ import annotations

from typing import Any

_NAN = float("nan")

#: Kinds whose demo runs a genuine synthetic analysis (reused from nudge.service).
_MEASURED_KINDS = frozenset(
    {"dose_response", "attribution", "identifiability", "constitutive", "temporal",
     "oed", "aggregation"}
)


def demo_kinds() -> list[str]:
    """All kinds ``demo_result`` can build (the registered renderer set)."""
    from nudge.viz import _RENDERERS

    return sorted(_RENDERERS)


def is_measured_demo(kind: str) -> bool:
    """True when this kind's demo is a genuine synthetic analysis (not an illustrative layout)."""
    return kind in _MEASURED_KINDS


#: Kinds whose ANIMATION needs an enriched demo (a frame sequence the static dict lacks) —
#: a sweep / trajectory / gauge orbit computed in the analysis layer, then only DRAWN by viz.
_ANIMATION_ENRICHED = frozenset({"oed", "robustness", "aggregation", "temporal"})


def demo_result(kind: str, *, variant: str = "resolved", animate: bool = False) -> Any:
    """Build a renderable demo result/figure-data for ``kind``.

    ``variant`` ∈ {``"resolved"``, ``"abstain"``} selects a resolved call vs the honest
    abstention for kinds that offer both illustrative examples (the measured-analysis kinds
    ignore it and return their genuine demo output). ``animate=True`` returns the
    animation-enriched demo for the kinds whose GIF needs a frame sequence the static figure
    doesn't carry (the sweep / trajectory / gauge orbit is computed in the analysis layer;
    viz only DRAWS it).
    """
    from nudge.viz import _RENDERERS

    if kind not in _RENDERERS:
        raise ValueError(f"unknown figure kind {kind!r} (known: {sorted(_RENDERERS)})")

    if animate and kind in _ANIMATION_ENRICHED:
        return _animation_enriched_demo(kind)

    if kind == "dose_response":
        return _dose_response_demo()
    if kind == "attribution":
        from nudge.service import attribution_demo

        return attribution_demo()
    if kind == "identifiability":
        from nudge.service import identifiability_demo

        return identifiability_demo(case="sloppy")
    if kind == "constitutive":
        from nudge.service import constitutive_demo

        return constitutive_demo()
    if kind == "temporal":
        from nudge.service import lotka_demo

        return lotka_demo()
    if kind == "oed":
        from nudge.service import oed_demo

        return oed_demo()
    if kind == "aggregation":
        from nudge.service import fibrillization_demo

        return fibrillization_demo()

    builder = _ILLUSTRATIVE.get(kind)
    if builder is None:  # pragma: no cover - every registered kind is covered above/here
        raise ValueError(f"no demo for kind {kind!r}")
    return builder(variant == "abstain")


def _animation_enriched_demo(kind: str) -> Any:
    """The animation-enriched demo (a frame sequence) for a sweep/orbit-based animator."""
    if kind == "oed":
        from nudge.service import oed_animation_demo

        return oed_animation_demo()
    if kind == "robustness":
        from nudge.service import robustness_animation_demo

        return robustness_animation_demo()
    if kind == "aggregation":
        from nudge.service import fibrillization_animation_demo

        return fibrillization_animation_demo()
    if kind == "temporal":
        from nudge.service import temporal_animation_demo

        return temporal_animation_demo()
    raise ValueError(f"no animation-enriched demo for kind {kind!r}")  # pragma: no cover


def _dose_response_demo() -> list[tuple[str, Any, Any, Any]]:
    """The flagship dual panel: a resolved ultrasensitive switch beside an honest abstain."""
    import numpy as np

    from nudge.inference.dose_response import attribute_dose_response

    def hill(d: Any, floor: float, amp: float, k: float, n: float) -> Any:
        d = np.maximum(np.asarray(d, dtype=float), 1e-9)
        return floor + amp * k**n / (k**n + d**n)

    d1 = np.linspace(0.0, 1.0, 16)
    r1 = hill(d1, 0.1, 0.9, 0.5, 6.0) + np.random.default_rng(0).normal(0.0, 0.01, d1.size)
    sw = attribute_dose_response(d1, r1, direction="repress", n_boot=200, seed=0)

    d2 = np.linspace(0.0, 0.6, 14)
    r2 = 0.5 + np.random.default_rng(0).normal(0.0, 0.02, d2.size)
    ab = attribute_dose_response(d2, r2, direction="repress", n_boot=200, seed=0)
    return [("switch-like", sw, d1, r1), ("flat / abstain", ab, d2, r2)]


# --------------------------------------------------------------------------- #
# Illustrative layout examples (generic labels — examples, not measured findings).
# --------------------------------------------------------------------------- #
def _epistasis_demo(abstain: bool) -> dict[str, Any]:
    inter, ci, call = ((0.72, [0.55, 0.90], "synergistic") if not abstain
                       else (0.02, [-0.3, 0.34], "unresolved"))
    return {
        "kind": "epistasis", "call": call, "reason": "illustrative example", "label": "A × B",
        "effect_a": 0.5, "effect_b": 0.4, "additive_pred": 0.9, "effect_ab": 1.6,
        "ci_a": [0.4, 0.6], "ci_b": [0.3, 0.5], "ci_ab": [1.4, 1.8],
        "interaction": inter, "ci_interaction": ci, "off_axis_residual": _NAN,
        "neomorphic_ratio": _NAN, "effect_space": "log-fold-change",
    }


def _differential_demo(abstain: bool) -> dict[str, Any]:
    call = "unresolved" if abstain else "gain-diff"
    return {
        "kind": "differential", "call": call, "reason": "illustrative example",
        "label": "resistant vs sensitive", "selected": "shared" if abstain else "n",
        "best_diff": "n", "bic": {"shared": 100.0, "K": 105.0, "n": 80.0, "vmax": 92.0},
        "log2_ratio": -2.0, "ci_log2": [-2.3, -1.1],
    }


def _multi_reporter_demo(abstain: bool) -> dict[str, Any]:
    call = "off-model" if abstain else "ceiling"
    return {
        "kind": "multi_reporter", "call": call, "reason": "illustrative example",
        "label": "3-reporter panel", "winner": "" if abstain else "ceiling",
        "knob_margin": 3.0, "effect_margin": 6.0, "n_reporters": 3,
        "losses": {"no_effect": 60.0, "threshold": 11.0, "gain": 35.0, "ceiling": 1.0,
                   "full": 0.9},
        "reporters": [{"name": f"R{i}", "r2_shared": 0.98, "r2_independent": 0.99}
                      for i in range(3)],
    }


def _diagnose_demo(abstain: bool) -> dict[str, Any]:
    return {
        "kind": "diagnose", "label": "attribution",
        "is_adequate": not abstain,
        "verdict": "off-model" if abstain else "adequate",
        "call": "off-model" if abstain else "",
        "reason": "illustrative example",
        "causes": ([{"name": "hidden regulator", "qualitative_rank": "leading",
                     "limitation_ref": "NUDGE-LIM-015", "limitation_title": ""},
                    {"name": "off-target", "qualitative_rank": "plausible",
                     "limitation_ref": "NUDGE-LIM-002", "limitation_title": ""}]
                   if abstain else []),
    }


def _design_demo(abstain: bool) -> dict[str, Any]:
    if abstain:
        return {"kind": "design", "design_kind": "abstention", "call": "abstain",
                "label": "flip ON", "verdict": "unreachable",
                "reason": "requested state out of the reachable range", "mode": "",
                "deltas": [], "dose": _NAN, "predicted_response": _NAN, "safety": None}
    return {"kind": "design", "design_kind": "intervention", "call": "", "label": "flip ON",
            "verdict": "", "reason": "reachable", "mode": "circuit",
            "deltas": [{"name": "edge0.K", "factor": 0.6}, {"name": "edge1.n", "factor": 1.4}],
            "dose": _NAN, "predicted_response": _NAN,
            "safety": {"proximity_before": 0.3, "proximity_after": 0.55,
                       "crosses_fold": False, "high_risk": False, "one_sided": False}}


def _cross_modality_demo(abstain: bool) -> dict[str, Any]:
    call = "non-responsive" if abstain else "threshold"
    return {"variants": [
        {"variant": "WT", "call": "reference", "K_threshold": 1.0, "n_apparent_gain": 2.0,
         "amp": 1.0, "floor": 0.05, "r2": 0.99, "ci_n": [1.8, 2.2], "ci_K": [0.9, 1.1],
         "direction": "activate"},
        {"variant": "mutant", "call": call, "K_threshold": 3.0, "n_apparent_gain": 2.0,
         "amp": 0.02 if abstain else 1.0, "floor": 0.05, "r2": 0.98, "ci_n": [1.7, 2.3],
         "ci_K": [2.6, 3.4], "direction": "activate"},
    ]}


def _robustness_demo(abstain: bool) -> dict[str, Any]:
    prox = _NAN if abstain else 0.33
    call = "not-bistable" if abstain else "robust"
    return {
        "kind": "robustness", "call": call, "reason": "illustrative example",
        "label": "1-node switch", "proximity": prox, "one_sided": not abstain,
        "channels": {"critical_slowing": 0.5, "basin_collapse": 0.2, "lobe_overlap": 0.6},
    }


_ILLUSTRATIVE: dict[str, Any] = {
    "epistasis": _epistasis_demo,
    "differential": _differential_demo,
    "multi_reporter": _multi_reporter_demo,
    "diagnose": _diagnose_demo,
    "design": _design_demo,
    "cross_modality": _cross_modality_demo,
    "robustness": _robustness_demo,
}
