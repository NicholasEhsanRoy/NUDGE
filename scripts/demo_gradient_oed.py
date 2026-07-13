#!/usr/bin/env python3
"""AD QSP gradient OED — resolve the antibody-binding ⇄ clearance confound + collapse the ellipse.

Action 3 of the flagship AD clinical demo. On the real, published Proctor 2013 amyloid-β QSP
model (:mod:`nudge.mechanisms.ad_qsp`; BioModels ``BIOMD0000000488``, CC0), two mechanistically
distinct parameters are **confounded** by a naive clinical schedule: the antibody–Aβ **binding
affinity** ``k_on`` and the microglial **clearance rate** ``k_gl``. Both lower amyloid burden,
so a realistic sparse amyloid-PET schedule (plaque measured at baseline + end of study) cannot
tell them apart — the Fisher information is near-singular. NUDGE's white-box gradient OED
(:mod:`nudge.inference.oed`, ``NUDGE-METHOD-014``) differentiates the D-optimality criterion of
``FIM(φ)`` w.r.t. the measurement schedule ``φ`` and gradient-ascends to the schedule that
resolves the pair — reporting the **measured** identifiability gain (local OED at θ₀,
``NUDGE-LIM-024``; nothing asserted).

It also renders the 95%-confidence-ellipse **collapse** animation (the ``oed`` animator,
:mod:`nudge.viz.oed`): the measurement times slide off the naive baseline/end cluster into the
antibody-dosing transient while the ``(k_on, k_gl)`` joint uncertainty ellipse shrinks.

Usage::

    uv run python scripts/demo_gradient_oed.py                 # measure + write GIF/JSON
    uv run python scripts/demo_gradient_oed.py --no-gif        # skip the GIF

Writes ``scripts/vv/ad_qsp_oed_RESULTS.json`` and (unless ``--no-gif``) the ellipse-collapse
GIF to ``tmp/ad_qsp_oed/`` (gitignored scratch).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from nudge.inference import oed  # noqa: E402
from nudge.mechanisms import ad_qsp as A  # noqa: E402

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "vv" / "ad_qsp_oed_RESULTS.json"


def _naive_schedule(t_max: float, n_pts: int = 6) -> np.ndarray:
    """A realistic sparse clinical schedule: cluster at baseline + end (miss the transient)."""
    h = n_pts // 2
    return np.concatenate([np.linspace(0.05, 0.5, h), np.linspace(t_max - 0.5, t_max, n_pts - h)])


def _confound(prob, phi: np.ndarray) -> dict:
    fim = oed.fisher_information(prob, phi)
    corr = float(fim[0, 1] / np.sqrt(fim[0, 0] * fim[1, 1]))
    crlb = oed.crlb(fim)
    return {"corr": corr, "cond": float(np.linalg.cond(fim)),
            "min_eig": oed.min_eigenvalue(fim),
            "crlb": [float(x) for x in crlb], "fim": fim.tolist()}


def _ellipse_frames(prob, res, steps: int, n_frames: int) -> list[dict]:
    chi2_95 = 5.991  # χ²(2 dof, 95%)
    idx = np.unique(np.linspace(0, steps - 1, num=min(n_frames, steps)).round().astype(int))
    frames = []
    for i in idx:
        phi = res.phi_history[int(i)]
        fim = oed.fisher_information(prob, phi)
        p = fim.shape[0]
        scale = max(float(np.trace(fim)) / p, 1e-30)
        cov = np.linalg.inv(fim + 1e-8 * scale * np.eye(p))
        evals, evecs = np.linalg.eigh(cov)
        evals = np.clip(evals, 0.0, None)
        frames.append({
            "step": int(i), "phi": [float(x) for x in np.sort(phi)],
            "ellipse": {"width": float(2.0 * np.sqrt(chi2_95 * evals[1])),
                        "height": float(2.0 * np.sqrt(chi2_95 * evals[0])),
                        "angle": float(np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1])))},
            "target_crlb": float(np.diag(cov)[res.target_index]),
        })
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--objective", default="d_opt", choices=["d_opt", "a_opt", "e_opt", "crlb"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--n-frames", type=int, default=24)
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    pair = ("k_on", "k_gl")
    prob = A.make_ad_oed_problem(pair=pair, biomarkers=(2,))
    t_max = float(prob.meta["t_max"])
    naive = _naive_schedule(t_max, n_pts=6)

    print("# AD QSP gradient OED — antibody-binding (k_on) ⇄ microglial-clearance (k_gl) confound")
    print(f"# model: Proctor 2013 Aβ QSP (BIOMD0000000488, CC0); target biomarker = plaque "
          f"(amyloid-PET); antibody dosed on {prob.meta['dose_window']}\n")

    naive_c = _confound(prob, naive)
    print(f"# NAIVE schedule (baseline+end, {len(naive)} plaque measurements): "
          f"corr(k_on,k_gl)={naive_c['corr']:+.3f}  FIM cond={naive_c['cond']:.2e}  "
          f"min-eig={naive_c['min_eig']:.3e}")

    res = oed.optimize_design(prob, naive, objective=args.objective, target=f"log_{pair[0]}",
                              steps=args.steps, learning_rate=0.15, capture_phi=True)
    opt_c = _confound(prob, res.phi_opt)
    print(f"# OPTIMISED schedule (gradient-ascended {args.objective}): "
          f"corr(k_on,k_gl)={opt_c['corr']:+.3f}  FIM cond={opt_c['cond']:.2e}  "
          f"min-eig={opt_c['min_eig']:.3e}")
    print("\n# MEASURED gains (local OED at θ₀; NUDGE-LIM-024):")
    print(f"#   CRLB(k_on):     {res.target_crlb_init:.3e} -> {res.target_crlb_opt:.3e}  "
          f"= x{res.crlb_improvement:.1f} better")
    print(f"#   FIM min-eig:    {res.min_eig_init:.3e} -> {res.min_eig_opt:.3e}  "
          f"= x{res.min_eig_improvement:.1f} lift")
    print(f"#   FIM cond:       {naive_c['cond']:.2e} -> {opt_c['cond']:.2e}")

    # animation backdrop: the plaque trajectory (with antibody dosing) the samples slide over.
    tr, tt = A.simulate_subject(dose=float(prob.meta["dose"]), t_max=t_max, dt=0.03)
    plaque = tr[:, 2]
    theta0 = [float(prob.theta0[0]), float(prob.theta0[1])]
    anim = {
        "kind": "oed", "label": "AD amyloid QSP — antibody OED", "model": "ad_qsp",
        "objective": args.objective, "target_parameter": "k_on (antibody binding)",
        "call": "", "reason": "",
        "crlb_improvement": float(res.crlb_improvement),
        "min_eig_improvement": float(res.min_eig_improvement),
        "animation": {
            "param_labels": ["log k_on (antibody binding)", "log k_gl (microglial clearance)"],
            "theta0": theta0, "t_bounds": [float(prob.phi_bounds[0]), float(prob.phi_bounds[1])],
            "traj_t": [float(x) for x in tt], "traj_x": [float(x) for x in plaque],
            "frames": _ellipse_frames(prob, res, args.steps, args.n_frames),
        },
        "caveat": "local OED: the optimal design + ellipse collapse are MEASURED at θ₀, not "
                  "extrapolated (NUDGE-LIM-024).",
    }

    gif_path = None
    if not args.no_gif:
        from nudge.viz.animate import render_animation

        out_dir = _HERE.parent / "tmp" / "ad_qsp_oed"
        out_dir.mkdir(parents=True, exist_ok=True)
        gif_path = str(out_dir / "ad_qsp_oed_ellipse_collapse.gif")
        fr = render_animation(anim, gif_path, kind="oed", frames=args.n_frames, fps=8,
                              emit_code=False)
        print(f"\n# ellipse-collapse GIF: {fr.path}")
        gif_path = fr.path

    _RESULTS.write_text(json.dumps({
        "model": "Proctor 2013 AD amyloid-β QSP (BIOMD0000000488, CC0) — Aβ subsystem",
        "pair": pair, "objective": args.objective, "biomarker": "plaque (amyloid-PET)",
        "naive_schedule": [float(x) for x in naive], "naive": naive_c,
        "optimised_schedule": [float(x) for x in np.sort(res.phi_opt)], "optimised": opt_c,
        "crlb_k_on_init": float(res.target_crlb_init), "crlb_k_on_opt": float(res.target_crlb_opt),
        "crlb_improvement": float(res.crlb_improvement),
        "min_eig_init": float(res.min_eig_init), "min_eig_opt": float(res.min_eig_opt),
        "min_eig_improvement": float(res.min_eig_improvement),
        "gif_path": gif_path,
        "caveat": "SYNTHETIC calibration cohort; local OED at θ₀ (NUDGE-LIM-024); demo-scaled "
                  "constants (NUDGE-LIM-026).",
    }, indent=2))
    print(f"# wrote {_RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
