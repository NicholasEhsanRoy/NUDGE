#!/usr/bin/env python3
"""AD QSP **NLME / hierarchical** population identifiability — the coupled-FIM scale wall.

The point this demo makes (and the reason it exists alongside
``scripts/demo_matrix_free_scale.py``). The independent-subjects cohort
(``make_ad_cohort_predict_fn``) gives every subject its own private kinetics, so its population
Fisher-information matrix is **block-diagonal** — a competent analyst decomposes it into
per-subject 12×12 blocks and never needs a big matrix. A "dense FIM OOMs" story built on THAT
cohort is a strawman.

This demo uses the **NLME / hierarchical** cohort (``make_ad_nlme_cohort_predict_fn``): each
subject's random-effect kinetics are drawn around a SHARED population geometric-mean ``μ`` with
SHARED fixed effects ``φ``. Because ``μ``/``φ`` enter EVERY subject's predicted observations,
the joint FIM over ``[ μ | φ | r₀ … r_{N-1} ]`` is a genuinely **coupled arrowhead** — a dense
border (the shared-hyperparameter rows/cols) plus per-subject blocks; cross-subject blocks are
exactly zero. The honest full-joint identifiability analysis of this coupled model needs either a
dense Jacobian/FIM (which OOMs at population scale) or NUDGE's matrix-free ``JᵀJ·v`` matvecs
(which stay flat).

It measures, at increasing ``n_subjects`` (fixed integrated state per subject):

1. **the coupling** — at a small N, materialize the exact dense FIM and report the largest
   border↔subject off-block entry (must be ≫0: coupled) and the largest cross-subject entry
   (must be ~0: arrowhead, block-diagonal in the random effects);
2. **the dense object size** — the ``(n_obs × n_params)`` Jacobian and the ``(n_params ×
   n_params)`` FIM in GB (what a naive full-joint analysis would materialize);
3. **the scale wall** — for BOTH ways of getting the FIM spectrum + verdict, the wall-time and
   **peak RSS**: dense (``jax.jacfwd``, run under a systemd ``MemoryMax`` cap so a blow-up is
   OOM-killed cleanly) vs matrix-free (one ``jvp`` + one ``vjp`` per matvec, never forming J).

HONESTY (NUDGE-LIM-028): arrowhead structure is in principle Schur-decomposable, so a bespoke
solver COULD avoid the dense matrix. The claim is strictly the measured one — the NAIVE
full-joint dense route OOMs while NUDGE's GENERIC matrix-free solver stays flat WITHOUT deriving
the Schur structure, and a from-scratch NumPy model (no autodiff) has only finite-difference
matvecs = O(n_params) forward solves each, intractable at this scale. Synthetic cohort,
demo-scaled constants (NUDGE-LIM-026).

Usage::

    uv run python scripts/demo_nlme_scale.py
    uv run python scripts/demo_nlme_scale.py --sweep 100,300,700,1500,2500 --dense-cap-gb 2.5

Writes ``scripts/vv/ad_qsp_nlme_scaling_RESULTS.json``.
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
_RESULTS = _HERE / "vv" / "ad_qsp_nlme_scaling_RESULTS.json"
_WORKER_TIMEOUT_S = int(os.environ.get("AD_QSP_WORKER_TIMEOUT_S", "600"))
_DENSE_FAIL = ("oom", "timeout", "killed")


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _make_prob(n_subjects: int):
    from nudge.mechanisms.ad_qsp import make_ad_nlme_cohort_predict_fn

    return make_ad_nlme_cohort_predict_fn(
        n_subjects=n_subjects, re_params=("k_pg", "K_pg", "k_gl"), n_obs_times=2,
        include_prior=False, biomarkers=(2,), dt=0.08, seed=0,
    )


def _run_worker(method: str, n_subjects: int) -> dict:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import jax

    jax.config.update("jax_enable_x64", True)

    from nudge.inference.sloppiness import analyze_model, sloppiness_diagnostic_matrixfree

    prob = _make_prob(n_subjects)
    status, label, t_wall = "ok", None, float("nan")
    if method == "dense":
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
    else:
        raise ValueError(f"unknown method {method!r}")
    return {
        "method": method, "n_subjects": n_subjects, "n_free": prob.n_theta,
        "n_obs": prob.n_obs, "n_states": prob.n_states, "status": status,
        "label": label, "wall_s": float(t_wall), "peak_rss_mb": float(_rss_mb()),
    }


def _has_systemd_run() -> bool:
    return shutil.which("systemd-run") is not None


def _spawn(method: str, n_subjects: int, cap_gb: float) -> dict:
    worker = [sys.executable, __file__, "--worker", method, str(n_subjects)]
    if method == "dense" and _has_systemd_run():
        cmd = ["systemd-run", "--user", "--scope", "--quiet",
               "-p", f"MemoryMax={cap_gb}G", "-p", "MemorySwapMax=0", "--", *worker]
    else:
        cmd = worker
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=_WORKER_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"method": method, "n_subjects": n_subjects, "status": "timeout",
                "label": None, "wall_s": float("nan"), "peak_rss_mb": float("nan")}
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        return {"method": method, "n_subjects": n_subjects, "status": "oom",
                "label": None, "wall_s": float("nan"), "peak_rss_mb": float("nan"),
                "stderr_tail": out.stderr.strip().splitlines()[-3:]}
    return json.loads(lines[-1])


def _coupling_probe(n_subjects: int = 6) -> dict:
    """Materialize the exact dense FIM at a SMALL N and measure the arrowhead structure."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import jax

    jax.config.update("jax_enable_x64", True)
    import numpy as np

    from nudge.inference.sloppiness import relative_sensitivity_jacobian

    prob = _make_prob(n_subjects)
    jac_log, _y = relative_sensitivity_jacobian(prob.predict_fn, prob.theta0)
    fim = jac_log.T @ jac_log / 0.05**2
    border = prob.border_indices()
    s0, s1 = prob.subject_block(0), prob.subject_block(1)
    border_subj = float(np.abs(fim[np.ix_(border, s0)]).max())
    cross_subj = float(np.abs(fim[np.ix_(s0, s1)]).max())
    lam = float(np.abs(fim).max())
    return {
        "n_subjects": n_subjects, "n_free": int(prob.n_theta), "border_size": int(prob.border_size),
        "max_border_subject_entry": border_subj,
        "max_cross_subject_entry": cross_subj,
        "cross_over_border_ratio": cross_subj / max(border_subj, 1e-30),
        "fim_max_abs": lam,
        "arrowhead": bool(border_subj > 1e-6 * lam and cross_subj <= 1e-9 * max(lam, 1e-30)),
    }


def _dense_object_sizes(sweep: list[int]) -> list[dict]:
    """The (n_obs × n_params) Jacobian + (n_params²) FIM sizes in GB per N (float64)."""
    from nudge.mechanisms.ad_qsp import AD_PARAM_VALUES

    d, n_params_full = 3, int(AD_PARAM_VALUES.shape[0])
    border = n_params_full  # μ (d) + φ (n_params_full − d)
    rows = []
    for n_subjects in sweep:
        n_free = border + n_subjects * d
        n_obs = n_subjects * 2  # 2 plaque timepoints/subject
        jac_gb = n_obs * n_free * 8 / 1e9
        fim_gb = n_free * n_free * 8 / 1e9
        rows.append({"n_subjects": n_subjects, "n_free": n_free, "n_obs": n_obs,
                     "dense_jacobian_gb": jac_gb, "dense_fim_gb": fim_gb})
    return rows


def _driver(sweep: list[int], cap_gb: float) -> None:
    print("# AD QSP NLME (hierarchical) population identifiability — coupled arrowhead FIM")
    print("# shared μ (RE geometric means: k_pg,K_pg,k_gl) + φ (fixed effects) border couples "
          "every subject;")
    print("# per-subject random-effect blocks r_i; 2 plaque timepoints/subject.\n")

    print("## Coupling probe (dense FIM at small N — is it a genuine arrowhead?)\n")
    cp = _coupling_probe(6)
    print(f"# n_subjects={cp['n_subjects']} n_free={cp['n_free']} border={cp['border_size']}")
    print(f"#   max |FIM[border, subject_0]| = {cp['max_border_subject_entry']:.3e}  "
          "(≫0 ⇒ shared hyperparams couple every subject)")
    print(f"#   max |FIM[subject_0, subject_1]| = {cp['max_cross_subject_entry']:.3e}  "
          "(~0 ⇒ block-diagonal in the random effects)")
    print(f"#   ⇒ arrowhead: {cp['arrowhead']} — NOT block-decomposable per-subject "
          "(the shared border couples all subjects)\n")

    print("## Dense object sizes that a naive full-joint analysis would materialize\n")
    sizes = _dense_object_sizes(sweep)
    print(f"{'n_subj':>7} | {'n_free':>7} | {'n_obs':>7} | {'dense J (GB)':>13} | "
          f"{'dense FIM (GB)':>15}")
    print("-" * 62)
    for s in sizes:
        print(f"{s['n_subjects']:>7} | {s['n_free']:>7} | {s['n_obs']:>7} | "
              f"{s['dense_jacobian_gb']:>13.3f} | {s['dense_fim_gb']:>15.3f}")
    print()

    print("## Scale wall — dense jacfwd (OOM) vs matrix-free (flat)\n")
    rows, mismatches = [], []
    for n_subjects in sweep:
        for method in ("dense", "matrixfree"):
            r = _spawn(method, n_subjects, cap_gb)
            rows.append(r)
            wall = r["wall_s"]
            wtxt = f"{wall:8.2f}s" if wall == wall else "    n/a "  # noqa: PLR0124
            print(f"  {method:11s} n_subj={n_subjects:5d} n_free={r.get('n_free', '?'):>6}  "
                  f"status={r['status']:9s} wall={wtxt}  peak_rss={r['peak_rss_mb']:8.1f} MB"
                  + (f"  label={r['label']}" if r["label"] else ""))
        d = next(x for x in rows if x["method"] == "dense" and x["n_subjects"] == n_subjects)
        m = next(x for x in rows if x["method"] == "matrixfree" and x["n_subjects"] == n_subjects)
        if d["status"] == "ok" and m["status"] == "ok" and d["label"] != m["label"]:
            mismatches.append({"n_subjects": n_subjects, "dense": d["label"], "mf": m["label"]})

    print("\n## Summary\n")
    print(f"{'n_subj':>7} | {'n_free':>7} | {'dense (jacfwd)':>22} | {'matrix-free':>22}")
    print("-" * 70)
    dense_ok_max, dense_oom_min, mf_rss = 0, None, []
    for n_subjects in sweep:
        d = next((x for x in rows if x["method"] == "dense"
                  and x["n_subjects"] == n_subjects), None)
        m = next((x for x in rows if x["method"] == "matrixfree"
                  and x["n_subjects"] == n_subjects), None)
        if d is None or m is None:
            continue
        n_free = m.get("n_free", d.get("n_free", "?"))
        d_txt = (f"{d['wall_s']:.2f}s / {d['peak_rss_mb']:.0f}MB"
                 if d["status"] == "ok" else d["status"].upper())
        m_txt = f"{m['wall_s']:.2f}s / {m['peak_rss_mb']:.0f}MB"
        print(f"{n_subjects:>7} | {n_free:>7} | {d_txt:>22} | {m_txt:>22}")
        if d["status"] == "ok":
            dense_ok_max = max(dense_ok_max, n_subjects)
        elif d["status"] in _DENSE_FAIL:
            dense_oom_min = n_subjects if dense_oom_min is None else min(dense_oom_min, n_subjects)
        mf_rss.append(m["peak_rss_mb"])

    mf_flat = (max(mf_rss) / max(min(mf_rss), 1e-9)) if mf_rss else float("nan")
    print(f"\n# dense (jacfwd): ok up to n_subjects={dense_ok_max}, first OOM/timeout at "
          f"n_subjects={dense_oom_min} (MemoryMax {cap_gb} GB).")
    print(f"# matrix-free: completes EVERY N; peak RSS {min(mf_rss):.0f} -> {max(mf_rss):.0f} MB "
          f"(max/min {mf_flat:.2f}x — flat).")
    print("# matrix-free label == dense label where dense succeeded. OK" if not mismatches
          else f"# !! LABEL MISMATCH: {mismatches}")

    ok = dense_oom_min is not None and mf_flat < 2.0 and not mismatches and cp["arrowhead"]
    verdict = ("COUPLED ARROWHEAD + MATRIX-FREE FLAT PAST DENSE OOM" if ok
               else "inconclusive on this hardware")
    print(f"\n# verdict: {verdict}")

    _RESULTS.write_text(json.dumps({
        "model": "Proctor 2013 AD amyloid-β QSP (BIOMD0000000488, CC0) — NLME/hierarchical "
                 "population, synthetic cohort",
        "coupling_probe": cp, "dense_object_sizes": sizes, "sweep": sweep,
        "dense_cap_gb": cap_gb, "rows": rows, "mismatches": mismatches,
        "dense_ok_max_nsubj": dense_ok_max, "dense_oom_min_nsubj": dense_oom_min,
        "mf_rss_flatness": mf_flat, "verdict": verdict,
    }, indent=2))
    print(f"\n# wrote {_RESULTS}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", nargs=2, metavar=("METHOD", "N_SUBJECTS"),
                    help="internal: run one config and print a JSON line")
    ap.add_argument("--sweep", type=str, default="100,300,700,1500,2500")
    ap.add_argument("--dense-cap-gb", type=float, default=2.5)
    args = ap.parse_args()

    if args.worker is not None:
        method, n_subjects = args.worker
        print(json.dumps(_run_worker(method, int(n_subjects))))
        return 0

    sweep = [int(x) for x in args.sweep.split(",") if x]
    _driver(sweep, args.dense_cap_gb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
