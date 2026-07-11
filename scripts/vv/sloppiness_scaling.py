#!/usr/bin/env python3
"""Measure the matrix-free identifiability path's flat memory vs the dense J that OOMs.

The artifact behind the claim "NUDGE analyzes the identifiability of a *large* mechanistic
network where the dense Fisher/sensitivity-matrix approach OOMs" — **measured**, not
asserted. For a fixed ODE network (a gLV community) it sweeps the number of free parameters
``n_θ`` and records, for BOTH ways of getting the FIM (``JᵀJ``) eigenspectrum + the
sloppy-vs-unidentifiable verdict, the **wall-time** and **peak RSS**:

- **dense** (:func:`nudge.inference.sloppiness.sloppiness_diagnostic`): builds
  ``J = ∂(trajectory)/∂θ`` with ``jax.jacfwd`` — the forward-mode tangent fan-out materializes
  a per-parameter trajectory, so memory grows steeply with ``n_θ`` and OOMs;
- **matrix-free** (:func:`nudge.inference.adjoint.ode_identifiability`, i.e.
  :func:`~nudge.inference.sloppiness.sloppiness_diagnostic_matrixfree`): drives an iterative
  eigensolver with ``JᵀJ·v`` matvecs (one ``jvp`` + one ``vjp``), never forming J — memory is
  O(n_params + n_obs + reverse-mode tape), **flat in ``n_θ``**.

Each ``(method, n_θ)`` config runs in a **fresh subprocess** (``--worker``) so peak RSS is
clean (``resource.getrusage`` is a monotonic high-water mark). The **dense** worker runs inside
a systemd ``MemoryMax`` cgroup scope (``--dense-cap-gb``, default 2.5 GB, ``MemorySwapMax=0``)
so that when jacfwd's tangent fan-out exceeds the cap the worker is OOM-killed **cleanly at the
cap** (recorded as ``oom``) instead of a host-endangering ~60 GB SIGKILL — the machine-safe
stand-in for the raw OOM. Where dense still fits, the driver checks the dense and matrix-free
**labels agree**. (Without ``systemd-run`` the dense worker runs uncapped; raise the sweep to
observe the natural memory blow-up.)

Usage::

    uv run python scripts/vv/sloppiness_scaling.py                       # full sweep + table
    uv run python scripts/vv/sloppiness_scaling.py --n-species 77 --sweep 500,1000,2000,4000,6000
    uv run python scripts/vv/sloppiness_scaling.py --dense-cap-gb 4
    uv run python scripts/vv/sloppiness_scaling.py --worker dense 500 77 2.5 0 6   # one config

Writes ``scripts/vv/sloppiness_scaling_RESULTS.json``.
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
_RESULTS = _HERE / "sloppiness_scaling_RESULTS.json"

#: per-worker wall budget; a dense jacfwd that doesn't finish in this is "intractable here".
_WORKER_TIMEOUT_S = int(os.environ.get("SLOPPINESS_WORKER_TIMEOUT_S", "600"))
_DENSE_FAIL = ("oom", "timeout", "killed")


def _rss_mb() -> float:
    """Peak RSS of this process in MB (``ru_maxrss`` is KB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _run_worker(
    method: str, n_theta: int, n_species: int, cap_gb: float, seed: int, n_obs_times: int
) -> dict:
    """Time one identifiability route at one ``n_θ`` and report wall-time + peak RSS.

    The **dense** route caps its address space so a dense-J OOM fails cleanly (``oom``); the
    **matrix-free** route runs uncapped (it never gets near the cap). Where dense succeeds, it
    also records its label so the driver can check the two paths agree.
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import jax

    jax.config.update("jax_enable_x64", True)
    from nudge.inference.adjoint import (
        make_glv_problem,
        ode_identifiability,
        ode_trajectory_predict_fn,
    )
    from nudge.inference.sloppiness import analyze_model, sloppiness_diagnostic_matrixfree

    problem = make_glv_problem(
        n_species=n_species, n_free=n_theta, n_obs=n_obs_times,
        dtype=jax.numpy.float64, seed=seed,
    )
    n_theta_eff = problem.n_theta
    n_obs = int(problem.target.size)

    status = "ok"
    label = None
    t_wall = float("nan")
    if method == "dense":
        # The dense diagnostic materializes J via ``jax.jacfwd`` — the forward-mode tangent
        # fan-out builds a per-parameter trajectory (peak ∝ n_params · n_steps · n_states).
        # The driver runs THIS worker inside a systemd MemoryMax cgroup scope, so if the
        # tangent exceeds the cap the whole process is OOM-killed cleanly (no JSON emitted →
        # the driver records it as ``oom``); we do NOT catch it here.
        try:
            t0 = time.perf_counter()
            rep = analyze_model(ode_trajectory_predict_fn(problem), problem.theta0, sigma=1e-2)
            t_wall = time.perf_counter() - t0
            label = rep.label
        except MemoryError:
            status = "oom"
    elif method == "matrixfree":
        # warm-up (compile excluded), then time a couple of runs and take the min.
        _ = sloppiness_diagnostic_matrixfree(
            ode_trajectory_predict_fn(problem), problem.theta0, 1e-2, method="iterative"
        )
        times = []
        for _ in range(2):
            t0 = time.perf_counter()
            rep = ode_identifiability(problem, sigma=1e-2, method="iterative")
            times.append(time.perf_counter() - t0)
        t_wall = min(times)
        label = rep.label
    else:
        raise ValueError(f"unknown method {method!r}")

    return {
        "method": method,
        "n_theta": n_theta_eff,
        "n_species": n_species,
        "n_obs": n_obs,
        "status": status,
        "label": label,
        "wall_s": float(t_wall),
        "peak_rss_mb": float(_rss_mb()),
    }


def _has_systemd_run() -> bool:
    return shutil.which("systemd-run") is not None


def _spawn(
    method: str, n_theta: int, n_species: int, cap_gb: float, seed: int, n_obs_times: int
) -> dict:
    worker = [
        sys.executable, __file__, "--worker", method,
        str(n_theta), str(n_species), str(cap_gb), str(seed), str(n_obs_times),
    ]
    # Cap the DENSE worker's physical memory with a systemd cgroup scope so a jacfwd OOM is a
    # clean, deterministic kill at ``cap_gb`` (not a host-endangering 60 GB SIGKILL). The
    # matrix-free worker runs uncapped — it never gets near the cap.
    if method == "dense" and _has_systemd_run():
        cmd = [
            "systemd-run", "--user", "--scope", "--quiet",
            "-p", f"MemoryMax={cap_gb}G", "-p", "MemorySwapMax=0", "--",
            *worker,
        ]
    else:
        cmd = worker
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=_WORKER_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        # jacfwd's compile/materialization did not finish in the budget — dense is intractable
        # here (the same blow-up that OOMs at scale), distinct from a clean memory kill.
        return {
            "method": method, "n_theta": n_theta, "n_species": n_species,
            "status": "timeout", "label": None, "wall_s": float("nan"),
            "peak_rss_mb": float("nan"),
        }
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        # no JSON → the worker was OOM-killed by the cgroup cap (or died hard).
        return {
            "method": method, "n_theta": n_theta, "n_species": n_species,
            "status": "oom", "label": None, "wall_s": float("nan"),
            "peak_rss_mb": float("nan"),
            "stderr_tail": out.stderr.strip().splitlines()[-3:],
        }
    return json.loads(lines[-1])


def _driver(
    n_species: int, sweep: list[int], cap_gb: float, seed: int, n_obs_times: int
) -> None:
    n_obs = n_obs_times * n_species
    print(f"# matrix-free vs dense identifiability scaling — {n_species}-state gLV network")
    print(f"# n_obs = {n_obs_times} times x {n_species} states = {n_obs} observations")
    print(f"# sweeping n_theta over {sweep}; dense MemoryMax cap = {cap_gb} GB\n")
    rows: list[dict] = []
    mismatches: list[dict] = []
    for n_theta in sweep:
        for method in ("dense", "matrixfree"):
            r = _spawn(method, n_theta, n_species, cap_gb, seed, n_obs_times)
            rows.append(r)
            st = r["status"]
            wall = r["wall_s"]
            wtxt = f"{wall:8.2f}s" if wall == wall else "    n/a "  # noqa: PLR0124 (NaN check)
            print(
                f"  {method:11s} n_theta={r.get('n_theta', n_theta):5d}  "
                f"status={st:12s} wall={wtxt}  peak_rss={r['peak_rss_mb']:8.1f} MB"
                + (f"  label={r['label']}" if r["label"] else "")
            )
        # correctness cross-check where dense succeeded
        d = next(x for x in rows if x["method"] == "dense" and x.get("n_theta") == r.get("n_theta"))
        m = next(
            x for x in rows if x["method"] == "matrixfree" and x.get("n_theta") == r.get("n_theta")
        )
        if d["status"] == "ok" and m["status"] == "ok" and d["label"] != m["label"]:
            mismatches.append({"n_theta": r.get("n_theta"), "dense": d["label"], "mf": m["label"]})

    # summary
    print("\n## Summary\n")
    print(f"{'n_theta':>8} | {'dense':>22} | {'matrix-free':>22}")
    print("-" * 60)
    summary = []
    dense_ok_max = 0
    dense_oom_min = None
    mf_rss = []
    for n_theta in sweep:
        d = next((x for x in rows if x["method"] == "dense" and x["n_theta"] == n_theta), None)
        m = next((x for x in rows if x["method"] == "matrixfree" and x["n_theta"] == n_theta), None)
        if d is None or m is None:
            continue
        d_txt = (
            f"{d['wall_s']:.2f}s / {d['peak_rss_mb']:.0f}MB"
            if d["status"] == "ok" else d["status"].upper()
        )
        m_txt = f"{m['wall_s']:.2f}s / {m['peak_rss_mb']:.0f}MB"
        print(f"{n_theta:>8} | {d_txt:>22} | {m_txt:>22}")
        if d["status"] == "ok":
            dense_ok_max = max(dense_ok_max, n_theta)
        elif d["status"] in _DENSE_FAIL:
            dense_oom_min = n_theta if dense_oom_min is None else min(dense_oom_min, n_theta)
        mf_rss.append(m["peak_rss_mb"])
        summary.append({"n_theta": n_theta, "dense": d, "matrixfree": m})

    mf_flat = (max(mf_rss) / max(min(mf_rss), 1e-9)) if mf_rss else float("nan")
    print(
        f"\n# dense (jacfwd): succeeds up to n_theta={dense_ok_max}, first OOM/timeout at "
        f"n_theta={dense_oom_min} (MemoryMax {cap_gb} GB / {_WORKER_TIMEOUT_S}s budget).\n"
        f"# matrix-free: completes EVERY n_theta; peak RSS "
        f"{min(mf_rss):.0f} -> {max(mf_rss):.0f} MB (max/min {mf_flat:.2f}x — flat)."
    )
    if mismatches:
        print(f"# !! LABEL MISMATCH matrix-free vs dense: {mismatches}")
    else:
        print("# matrix-free label == dense label at every n_theta where dense succeeded. OK")

    ok = dense_oom_min is not None and mf_flat < 2.0 and not mismatches
    verdict = (
        "MATRIX-FREE FLAT + SCALES PAST DENSE OOM" if ok else "inconclusive on this hardware"
    )
    print(f"# verdict: {verdict}")

    _RESULTS.write_text(json.dumps(
        {
            "n_species": n_species, "sweep": sweep, "dense_cap_gb": cap_gb,
            "rows": rows, "summary": summary, "mismatches": mismatches,
            "dense_ok_max_ntheta": dense_ok_max, "dense_oom_min_ntheta": dense_oom_min,
            "mf_rss_flatness": mf_flat, "verdict": verdict,
        },
        indent=2,
    ))
    print(f"\n# wrote {_RESULTS}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--worker", nargs=6,
        metavar=("METHOD", "N_THETA", "N_SPECIES", "CAP_GB", "SEED", "N_OBS_TIMES"),
        help="internal: run one config and print a JSON line",
    )
    ap.add_argument("--n-species", type=int, default=77)  # p_full = 77²+2·77 = 6083
    # few observation times ⇒ n_params > n_obs across the large sweep (the realistic
    # "big network, limited data" regime: rank ≤ n_obs ⇒ certified unidentifiable, fast).
    ap.add_argument("--n-obs-times", type=int, default=6)
    ap.add_argument("--sweep", type=str, default="500,1000,2000,4000,6000")
    ap.add_argument("--dense-cap-gb", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.worker is not None:
        method, n_theta, n_species, cap_gb, seed, n_obs_times = args.worker
        print(json.dumps(
            _run_worker(
                method, int(n_theta), int(n_species), float(cap_gb), int(seed),
                int(n_obs_times),
            )
        ))
        return 0

    sweep = [int(x) for x in args.sweep.split(",") if x]
    _driver(args.n_species, sweep, args.dense_cap_gb, args.seed, args.n_obs_times)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
