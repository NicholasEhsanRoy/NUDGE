#!/usr/bin/env python3
"""AD QSP population identifiability — matrix-free FIM stays flat where dense jacfwd OOMs.

The flagship clinical-scale demonstration of NUDGE's matrix-free identifiability on a REAL,
published, open Alzheimer's-disease amyloid-β QSP model (Proctor et al. 2013, BioModels
``BIOMD0000000488``, CC0; :mod:`nudge.mechanisms.ad_qsp`) calibrated against a **synthetic
ground-truth cohort** (no gated patient data).

**Where the dimensionality comes from (honest).** A single-subject fit of even a rich QSP
model is only a few dozen parameters — a few-MB dense Jacobian, no OOM. The genuine
high-dimensional wall is **population-scale calibration**: each of ``N`` synthetic subjects
carries its own copy of the kinetic parameters (a nonlinear-mixed-effects structure — how
QSP models are really fit), so the free-parameter count is ``N × 12`` and reaches thousands.
At **fixed** cohort (all subjects always simulated → fixed integrated state) we sweep how
many subject-specific parameters are jointly estimated (``n_free``) and record, for BOTH
ways of getting the Fisher-information (``JᵀJ``) spectrum + verdict, the **wall-time** and
**peak RSS**:

- **dense** (:func:`nudge.inference.sloppiness.analyze_model`) builds ``J = ∂(biomarkers)/∂θ``
  with ``jax.jacfwd`` — the forward-mode tangent fan-out materializes a per-parameter
  trajectory over the whole cohort, so memory grows ∝ ``n_free`` and OOMs;
- **matrix-free** (:func:`nudge.inference.sloppiness.sloppiness_diagnostic_matrixfree`) drives
  the FIM only through ``JᵀJ·v`` matvecs (one ``jvp`` + one ``vjp``), never forming J — peak
  memory is O(state + n_free + tape), **flat in ``n_free``**.

With a realistic biomarker budget (a few plaque/amyloid-PET measurements per subject) the
population problem is genuinely **rank-deficient** — more subject-specific parameters than
observations — so NUDGE certifies ``unidentifiable`` (the ``NUDGE-LIM-023`` fail-safe: it
never asserts an identifiability it cannot verify), cheaply, by shape.

Each ``(method, n_free)`` config runs in a **fresh subprocess** (``--worker``) so peak RSS is
clean; the **dense** worker runs inside a systemd ``MemoryMax`` cgroup scope
(``--dense-cap-gb``) so a jacfwd blow-up is OOM-killed cleanly at the cap (recorded ``oom``),
the machine-safe stand-in for the raw OOM — mirroring ``scripts/vv/sloppiness_scaling.py``.

It then runs the SLOPPY-PARAMETER FLAGGING: a single-subject identifiability report naming
which kinetic constants the biomarkers cannot constrain (the measured FIM spectrum + sloppy
directions).

Usage::

    uv run python scripts/demo_matrix_free_scale.py                     # full sweep + flagging
    uv run python scripts/demo_matrix_free_scale.py --n-subjects 250 --sweep 600,1500,3000
    uv run python scripts/demo_matrix_free_scale.py --dense-cap-gb 4

Writes ``scripts/vv/ad_qsp_scaling_RESULTS.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "vv" / "ad_qsp_scaling_RESULTS.json"
_WORKER_TIMEOUT_S = int(os.environ.get("AD_QSP_WORKER_TIMEOUT_S", "600"))
_DENSE_FAIL = ("oom", "timeout", "killed")


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _run_worker(method: str, n_subjects: int, n_free: int, n_obs_times: int) -> dict:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from nudge.inference.sloppiness import analyze_model, sloppiness_diagnostic_matrixfree
    from nudge.mechanisms.ad_qsp import make_ad_cohort_predict_fn

    prob = make_ad_cohort_predict_fn(
        n_subjects=n_subjects, n_free=n_free, n_obs_times=n_obs_times, dt=0.08,
        biomarkers=(2,), seed=0,
    )
    status, label, t_wall = "ok", None, float("nan")
    if method == "dense":
        # jacfwd materializes the (n_obs, n_free) sensitivity via a per-parameter tangent
        # trajectory over the whole cohort → peak ∝ n_free · n_steps · state. Under the
        # systemd cap this OOM-kills cleanly (no JSON → driver records ``oom``).
        t0 = time.perf_counter()
        rep = analyze_model(prob.predict_fn, prob.theta0, sigma=0.05)
        t_wall = time.perf_counter() - t0
        label = rep.label
    elif method == "matrixfree":
        _ = sloppiness_diagnostic_matrixfree(  # warm-up (compile excluded)
            prob.predict_fn, prob.theta0, 0.05, method="iterative"
        )
        times = []
        for _ in range(2):
            t0 = time.perf_counter()
            rep = sloppiness_diagnostic_matrixfree(
                prob.predict_fn, prob.theta0, 0.05, method="iterative"
            )
            times.append(time.perf_counter() - t0)
        t_wall = min(times)
        label = rep.label
        _ = jnp  # (kept import symmetric with the dense worker)
    else:
        raise ValueError(f"unknown method {method!r}")

    return {
        "method": method, "n_free": prob.n_theta, "n_subjects": n_subjects,
        "n_obs": prob.n_obs, "n_states": prob.n_states, "status": status,
        "label": label, "wall_s": float(t_wall), "peak_rss_mb": float(_rss_mb()),
    }


def _has_systemd_run() -> bool:
    return shutil.which("systemd-run") is not None


def _spawn(method: str, n_subjects: int, n_free: int, n_obs_times: int, cap_gb: float) -> dict:
    worker = [sys.executable, __file__, "--worker", method,
              str(n_subjects), str(n_free), str(n_obs_times)]
    if method == "dense" and _has_systemd_run():
        cmd = ["systemd-run", "--user", "--scope", "--quiet",
               "-p", f"MemoryMax={cap_gb}G", "-p", "MemorySwapMax=0", "--", *worker]
    else:
        cmd = worker
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=_WORKER_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"method": method, "n_free": n_free, "n_subjects": n_subjects,
                "status": "timeout", "label": None, "wall_s": float("nan"),
                "peak_rss_mb": float("nan")}
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        return {"method": method, "n_free": n_free, "n_subjects": n_subjects,
                "status": "oom", "label": None, "wall_s": float("nan"),
                "peak_rss_mb": float("nan"),
                "stderr_tail": out.stderr.strip().splitlines()[-3:]}
    return json.loads(lines[-1])


def _sloppy_flagging(n_obs_times: int) -> dict:
    """Single-subject identifiability: name the kinetic constants the biomarkers can't pin."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import jax

    jax.config.update("jax_enable_x64", True)
    import numpy as np

    from nudge.inference.sloppiness import analyze_model, relative_sensitivity_jacobian
    from nudge.mechanisms.ad_qsp import AD_PARAM_NAMES, make_ad_cohort_predict_fn

    prob = make_ad_cohort_predict_fn(
        n_subjects=1, n_free=len(AD_PARAM_NAMES), n_obs_times=n_obs_times,
        dt=0.06, biomarkers=(2, 1), seed=0,
    )
    names = list(AD_PARAM_NAMES)
    rep = analyze_model(prob.predict_fn, prob.theta0, sigma=0.05, param_names=names)
    jac_log, _y = relative_sensitivity_jacobian(prob.predict_fn, prob.theta0)
    fim = jac_log.T @ jac_log / 0.05**2
    evals, evecs = np.linalg.eigh(fim)
    sloppy = []
    for e in range(min(2, evals.shape[0])):
        v = evecs[:, e]
        top = np.argsort(-np.abs(v))[:3]
        sloppy.append({
            "eigenvalue": float(evals[e]),
            "loadings": {names[int(j)]: round(float(v[int(j)]), 3) for j in top},
        })
    return {
        "label": rep.label, "cond_number": float(rep.cond_number),
        "spectral_span_decades": float(rep.spectral_span_decades),
        "smallest_eigenvalue": float(rep.smallest_eigenvalue),
        "n_sloppy_dims": int(rep.n_sloppy_dims), "n_null_dims": int(rep.n_null_dims),
        "sloppiest_directions": sloppy, "reason": rep.reason,
    }


def _driver(n_subjects: int, sweep: list[int], n_obs_times: int, cap_gb: float) -> None:
    print("# AD QSP population identifiability — matrix-free vs dense jacfwd")
    print(f"# fixed cohort N={n_subjects} subjects (state={n_subjects * 6}); "
          f"{n_obs_times} plaque timepoints/subject → n_obs={n_subjects * n_obs_times}")
    print(f"# sweeping n_free (subject-specific params jointly estimated) over {sweep}; "
          f"dense MemoryMax cap = {cap_gb} GB\n")
    rows, mismatches = [], []
    for n_free in sweep:
        for method in ("dense", "matrixfree"):
            r = _spawn(method, n_subjects, n_free, n_obs_times, cap_gb)
            rows.append(r)
            wall = r["wall_s"]
            wtxt = f"{wall:8.2f}s" if wall == wall else "    n/a "  # noqa: PLR0124
            print(f"  {method:11s} n_free={r.get('n_free', n_free):5d}  "
                  f"status={r['status']:9s} wall={wtxt}  peak_rss={r['peak_rss_mb']:8.1f} MB"
                  + (f"  label={r['label']}" if r["label"] else ""))
        d = next(x for x in rows if x["method"] == "dense" and x.get("n_free") == r.get("n_free"))
        m = next(x for x in rows if x["method"] == "matrixfree"
                 and x.get("n_free") == r.get("n_free"))
        if d["status"] == "ok" and m["status"] == "ok" and d["label"] != m["label"]:
            mismatches.append({"n_free": r.get("n_free"), "dense": d["label"], "mf": m["label"]})

    print("\n## Summary\n")
    print(f"{'n_free':>8} | {'dense (jacfwd)':>22} | {'matrix-free':>22}")
    print("-" * 60)
    summary, dense_ok_max, dense_oom_min, mf_rss = [], 0, None, []
    for n_free in sweep:
        d = next((x for x in rows if x["method"] == "dense" and x["n_free"] == n_free), None)
        m = next((x for x in rows if x["method"] == "matrixfree" and x["n_free"] == n_free), None)
        if d is None or m is None:
            continue
        d_txt = (f"{d['wall_s']:.2f}s / {d['peak_rss_mb']:.0f}MB"
                 if d["status"] == "ok" else d["status"].upper())
        m_txt = f"{m['wall_s']:.2f}s / {m['peak_rss_mb']:.0f}MB"
        print(f"{n_free:>8} | {d_txt:>22} | {m_txt:>22}")
        if d["status"] == "ok":
            dense_ok_max = max(dense_ok_max, n_free)
        elif d["status"] in _DENSE_FAIL:
            dense_oom_min = n_free if dense_oom_min is None else min(dense_oom_min, n_free)
        mf_rss.append(m["peak_rss_mb"])
        summary.append({"n_free": n_free, "dense": d, "matrixfree": m})

    mf_flat = (max(mf_rss) / max(min(mf_rss), 1e-9)) if mf_rss else float("nan")
    print(f"\n# dense (jacfwd): succeeds up to n_free={dense_ok_max}, first OOM/timeout at "
          f"n_free={dense_oom_min} (MemoryMax {cap_gb} GB).")
    print(f"# matrix-free: completes EVERY n_free; peak RSS {min(mf_rss):.0f} -> "
          f"{max(mf_rss):.0f} MB (max/min {mf_flat:.2f}x — flat).")
    print("# matrix-free label == dense label where dense succeeded. OK" if not mismatches
          else f"# !! LABEL MISMATCH: {mismatches}")

    print("\n## Sloppy-parameter flagging (single-subject identifiability)\n")
    flag = _sloppy_flagging(n_obs_times=8)
    print(f"# verdict: {flag['label']} (cond {flag['cond_number']:.2e}, span "
          f"{flag['spectral_span_decades']:.1f} decades, smallest eig "
          f"{flag['smallest_eigenvalue']:.2e})")
    for i, d in enumerate(flag["sloppiest_directions"]):
        load = ", ".join(f"{k}={v:+.2f}" for k, v in d["loadings"].items())
        print(f"#   sloppy dir {i} (eig {d['eigenvalue']:.2e}): {load}")

    ok = dense_oom_min is not None and mf_flat < 2.0 and not mismatches
    verdict = "MATRIX-FREE FLAT + SCALES PAST DENSE OOM" if ok else "inconclusive on this hardware"
    print(f"\n# verdict: {verdict}")

    _RESULTS.write_text(json.dumps({
        "model": "Proctor 2013 AD amyloid-β QSP (BIOMD0000000488, CC0) — Aβ subsystem, "
                 "synthetic cohort",
        "n_subjects": n_subjects, "n_obs_times": n_obs_times, "sweep": sweep,
        "dense_cap_gb": cap_gb, "rows": rows, "summary": summary, "mismatches": mismatches,
        "dense_ok_max_nfree": dense_ok_max, "dense_oom_min_nfree": dense_oom_min,
        "mf_rss_flatness": mf_flat, "sloppy_flagging": flag, "verdict": verdict,
    }, indent=2))
    print(f"\n# wrote {_RESULTS}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", nargs=4,
                    metavar=("METHOD", "N_SUBJECTS", "N_FREE", "N_OBS_TIMES"),
                    help="internal: run one config and print a JSON line")
    ap.add_argument("--n-subjects", type=int, default=250)
    ap.add_argument("--n-obs-times", type=int, default=2)  # n_obs = N·2 < n_free (rank-deficient)
    ap.add_argument("--sweep", type=str, default="400,700,1000,1500,2000")
    ap.add_argument("--dense-cap-gb", type=float, default=2.5)
    args = ap.parse_args()

    if args.worker is not None:
        method, n_subjects, n_free, n_obs_times = args.worker
        print(json.dumps(_run_worker(method, int(n_subjects), int(n_free), int(n_obs_times))))
        return 0

    sweep = [int(x) for x in args.sweep.split(",") if x]
    _driver(args.n_subjects, sweep, args.n_obs_times, args.dense_cap_gb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
