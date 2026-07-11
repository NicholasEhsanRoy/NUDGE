"""Per-kind renderer tests for the nudge.viz figure battery (Phase-1 slices).

Two guarantees are locked for every new renderer, off fast, JAX-free canonical dicts:

* **render works** — a resolved result renders to a PNG (+ provenance code/data) and does
  NOT abstain.
* **abstention overlay fires** — an abstaining result of that kind yields
  ``FigureResult.abstained is True`` AND the overlay is actually drawn on a panel
  (``_nudge_abstained``). This is the load-bearing honesty guarantee inherited from the
  render pipeline; a renderer structurally cannot forget it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("matplotlib")

import nudge.viz as viz  # noqa: E402

_NAN = float("nan")


def _epistasis(call: str) -> dict[str, Any]:
    inter, ci = (0.72, [0.55, 0.90]) if call != "unresolved" else (0.02, [-0.3, 0.34])
    return {
        "kind": "epistasis", "call": call, "reason": "test", "label": "A×B",
        "effect_a": 0.5, "effect_b": 0.4, "additive_pred": 0.9, "effect_ab": 1.6,
        "ci_a": [0.4, 0.6], "ci_b": [0.3, 0.5], "ci_ab": [1.4, 1.8],
        "interaction": inter, "ci_interaction": ci, "off_axis_residual": _NAN,
        "neomorphic_ratio": _NAN, "effect_space": "log-fold-change",
    }


def _differential(call: str) -> dict[str, Any]:
    return {
        "kind": "differential", "call": call, "reason": "test", "label": "A vs B",
        "selected": "n" if call == "gain-diff" else "shared",
        "best_diff": "n",
        "bic": {"shared": 100.0, "K": 105.0, "n": 80.0, "vmax": 92.0},
        "log2_ratio": -2.0, "ci_log2": [-2.3, -1.1],
    }


def _multi_reporter(call: str) -> dict[str, Any]:
    return {
        "kind": "multi_reporter", "call": call, "reason": "test", "label": "panel",
        "winner": "ceiling" if call == "ceiling" else "", "knob_margin": 3.0,
        "effect_margin": 6.0, "n_reporters": 3,
        "losses": {"no_effect": 60.0, "threshold": 11.0, "gain": 35.0, "ceiling": 1.0,
                   "full": 0.9},
        "reporters": [{"name": f"R{i}", "r2_shared": 0.98, "r2_independent": 0.99}
                      for i in range(3)],
    }


def _temporal(call: str) -> dict[str, Any]:
    return {
        "kind": "temporal", "call": call, "reason": "test", "label": "community",
        "selected_knob": "susceptibility" if call == "susceptibility" else "growth",
        "bic": {"null": 100.0, "growth": 96.0, "interaction": 95.0,
                "susceptibility": 20.0},
        "delta": {"susceptibility": -1.2},
        "cond_number": 12.0, "abs_corr": 0.99 if call == "unresolved" else 0.4,
        "degenerate": call == "unresolved",
        "degeneracy_direction": [0.7, -0.7, 0.0], "trajectories": None,
    }


def _aggregation(call: str) -> dict[str, Any]:
    return {
        "kind": "aggregation", "call": call, "reason": "test", "label": "amyloid",
        "kappa": 2.0, "lam": 0.5, "kappa_ci": [1.8, 2.2], "lambda_ci": [0.4, 0.6],
        "individual_k_identifiable": False, "cond_number": 1e6,
        "null_direction": [0.6, -0.7, 0.3], "curve": None,
    }


def _constitutive(call: str) -> dict[str, Any]:
    ng = [1.0, 2.0, 3.0, 4.0, 5.0]
    return {
        "kind": "constitutive", "call": call, "reason": "test", "label": "circuit",
        "asserts_biological_switch": call == "biological-switch",
        "n_grid": ng, "loss_no_control": [1.0, 1.0, 1.0, 1.0, 1.0],
        "loss_with_control": [5.0, 2.0, 1.0, 1.5, 2.0],
        "n1_rejection": 4.0, "argmin_n_with_control": 3.0,
        "calibration": {"h": 6.0, "km": 0.5, "vmax": 20.0, "base": 0.1, "r2": 0.99,
                        "is_nonlinear": True},
        "ground_truth": None,
    }


def _diagnose(inadequate: bool) -> dict[str, Any]:
    return {
        "kind": "diagnose", "label": "attribution",
        "is_adequate": not inadequate,
        "verdict": "off-model" if inadequate else "adequate",
        "call": "off-model" if inadequate else "",
        "reason": "test",
        "causes": ([{"name": "hidden regulator", "qualitative_rank": "leading",
                     "limitation_ref": "NUDGE-LIM-015", "limitation_title": ""},
                    {"name": "off-target", "qualitative_rank": "plausible",
                     "limitation_ref": "NUDGE-LIM-002", "limitation_title": ""}]
                   if inadequate else []),
    }


def _design(abstain: bool) -> dict[str, Any]:
    if abstain:
        return {"kind": "design", "design_kind": "abstention", "call": "abstain",
                "label": "flip", "verdict": "unreachable", "reason": "out of range",
                "mode": "", "deltas": [], "dose": _NAN, "predicted_response": _NAN,
                "safety": None}
    return {"kind": "design", "design_kind": "intervention", "call": "", "label": "flip",
            "verdict": "", "reason": "ok", "mode": "circuit",
            "deltas": [{"name": "edge0.K", "factor": 0.6}],
            "dose": _NAN, "predicted_response": _NAN,
            "safety": {"proximity_before": 0.3, "proximity_after": 1.05,
                       "crosses_fold": True, "high_risk": True, "one_sided": True}}


def _oed(call: str) -> dict[str, Any]:
    return {
        "kind": "oed", "call": call, "reason": "test", "label": "design",
        "model": "logistic", "objective": "crlb", "target_parameter": "log_alpha",
        "phi_init": [8.0, 8.5, 9.0], "phi_opt": [1.0, 3.0, 9.0],
        "target_crlb_init": 1.0, "target_crlb_opt": 0.1, "crlb_improvement": 10.0,
        "min_eig_init": 0.01, "min_eig_opt": 0.5, "min_eig_improvement": 50.0,
    }


def _robustness(call: str) -> dict[str, Any]:
    prox = {"robust": 0.33, "near-fold": 0.9, "unresolved": 0.02,
            "not-bistable": float("nan")}[call]
    return {
        "kind": "robustness", "call": call, "reason": "test", "label": "1node",
        "proximity": prox, "one_sided": call in ("robust", "near-fold"),
        "channels": {"critical_slowing": 0.5, "basin_collapse": 0.2,
                     "lobe_overlap": 0.6},
    }


def _attribution(resolved: bool) -> dict[str, Any]:
    """A ``report_to_dict``-shaped AttributionReport: resolved joint vs single-op abstain."""
    if resolved:
        return {
            "target": "SOS1",
            "n_cells": {"Stim8hr": 1500, "Stim48hr": 1500, "Rest": 40},
            "single": {
                "Stim8hr": {"call": "gain_or_threshold",
                            "nlls": {"n": 10.1, "K": 10.0, "vmax": 12.0}},
                "Stim48hr": {"call": "gain_or_threshold",
                             "nlls": {"n": 8.0, "K": 8.1, "vmax": 9.5}},
            },
            "multi": {"call": "threshold",
                      "nlls": {"gain": 20.0, "threshold": 15.0, "ceiling": 22.0}},
            "skipped": {"Rest": "only 40 target cells (< 200)"},
        }
    # A single usable operating point cannot separate gain from threshold → abstain.
    return {
        "target": "SOS1", "n_cells": {"Stim8hr": 1500},
        "single": {"Stim8hr": {"call": "gain_or_threshold",
                               "nlls": {"n": 10.0, "K": 10.02, "vmax": 12.0}}},
        "multi": None, "skipped": {},
    }


def _identifiability(resolved: bool) -> dict[str, Any]:
    """A SloppinessReport-shaped dict: sloppy-but-predictive (usable) vs unidentifiable."""
    base = {
        "model_label": "sum-of-exponentials",
        "param_names": ["A1", "k1", "A2", "k2"],
        "fim_eigenvalues": [1e-4, 1e-2, 1.0, 100.0], "cond_number": 1e6,
        "span_decades": 6.0, "smallest_eigenvalue": 1e-4, "largest_eigenvalue": 100.0,
        "n_sloppy_dims": 1, "is_sloppy": True, "pred_rel_tol": 0.05,
        "sloppy_decade_threshold": 3.0, "null_hint": "",
    }
    if resolved:
        return {**base, "call": "sloppy-but-predictive",
                "verdict": "sloppy-but-predictive", "reason": "sloppy but predictive",
                "n_null_dims": 0, "predictive": True, "relative_prediction_std": 0.01,
                "naive_verdict": "unidentifiable", "naive_is_wrong": True}
    return {**base, "model_label": "redundant-exp", "call": "unidentifiable",
            "verdict": "unidentifiable", "reason": "structural null (k1+k2)",
            "fim_eigenvalues": [1e-18, 1e-2, 1.0, 100.0], "n_null_dims": 1,
            "predictive": False, "relative_prediction_std": 0.6,
            "naive_verdict": "unidentifiable", "naive_is_wrong": False}


def _cross_modality(abstain: bool) -> dict[str, Any]:
    call = "non-responsive" if abstain else "threshold"
    return {"variants": [
        {"variant": "WT", "call": "reference", "K_threshold": 1.0,
         "n_apparent_gain": 2.0, "amp": 1.0, "floor": 0.05, "r2": 0.99,
         "ci_n": [1.8, 2.2], "ci_K": [0.9, 1.1], "direction": "activate"},
        {"variant": "mut", "call": call, "K_threshold": 3.0, "n_apparent_gain": 2.0,
         "amp": 1.0 if not abstain else 0.02, "floor": 0.05, "r2": 0.98,
         "ci_n": [1.7, 2.3], "ci_K": [2.6, 3.4], "direction": "activate"},
    ]}


# (kind, resolved-input, abstaining-input)
CASES = [
    ("attribution", _attribution(True), _attribution(False)),
    ("identifiability", _identifiability(True), _identifiability(False)),
    ("epistasis", _epistasis("synergistic"), _epistasis("unresolved")),
    ("differential", _differential("gain-diff"), _differential("unresolved")),
    ("multi_reporter", _multi_reporter("ceiling"), _multi_reporter("off-model")),
    ("temporal", _temporal("susceptibility"), _temporal("unresolved")),
    ("aggregation", _aggregation("composites-identified"), _aggregation("unresolved")),
    ("constitutive", _constitutive("biological-switch"), _constitutive("unresolved")),
    ("diagnose", _diagnose(False), _diagnose(True)),
    ("design", _design(False), _design(True)),
    ("oed", _oed(""), _oed("unresolved")),
    ("cross_modality", _cross_modality(False), _cross_modality(True)),
    ("robustness", _robustness("robust"), _robustness("not-bistable")),
]


@pytest.mark.parametrize("kind,resolved,_abstain", CASES,
                         ids=[c[0] for c in CASES])
def test_renderer_renders_resolved(kind: str, resolved: Any, _abstain: Any,
                                   tmp_path: Path) -> None:
    out = tmp_path / f"{kind}.png"
    fr = viz.render(resolved, str(out), kind=kind, emit_code=True)
    assert out.exists() and out.stat().st_size > 0
    assert fr.kind == kind
    assert fr.abstained is False, f"{kind} resolved case must not abstain"
    assert fr.code_path is not None and Path(fr.code_path).exists()
    assert fr.data_path is not None and Path(fr.data_path).exists()


@pytest.mark.parametrize("kind,_resolved,abstain", CASES,
                         ids=[c[0] for c in CASES])
def test_renderer_abstention_overlay_fires(kind: str, _resolved: Any, abstain: Any,
                                           tmp_path: Path) -> None:
    """LOAD-BEARING: an abstaining result of ANY kind draws the abstention overlay."""
    rf = viz.figure(abstain, kind=kind)
    assert rf.abstained is True, f"{kind} abstaining case must report abstained"
    marked = [p for p in rf.panels if getattr(p.ax, "_nudge_abstained", False)]
    assert marked, f"{kind}: no panel carries the abstention overlay"
    banners = [t.get_text() for p in marked for t in p.ax.texts]
    assert any("CAN'T TELL" in b for b in banners), banners


def test_replay_data_roundtrips_for_new_kind(tmp_path: Path) -> None:
    """The emitted fig.data.json replays through render() for a Phase-1 kind (epistasis)."""
    import json

    out = tmp_path / "epi.png"
    fr = viz.render(_epistasis("synergistic"), str(out), kind="epistasis")
    assert fr.data_path is not None
    data = json.loads(Path(fr.data_path).read_text())
    assert data["kind"] == "epistasis"
    replay = tmp_path / "epi_replay.png"
    fr2 = viz.render(data, str(replay), emit_code=False)
    assert replay.exists() and fr2.kind == "epistasis"
