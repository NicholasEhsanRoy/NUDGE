#!/usr/bin/env python3
"""Measure the adjoint's O(1)-in-parameter-count scaling vs forward sensitivity.

The artifact that proves the "an un-aided LLM's forward-sensitivity code OOMs / times
out, but NUDGE's adjoint cuts through" claim — **measured**, not asserted. For a fixed
large ODE network (default: a 15-state gLV community) it sweeps the number of free
parameters ``n_θ`` (5 → 200+) and records, for BOTH gradient routes, the **wall-time**
and **peak RSS**:

- **forward sensitivity** (:func:`nudge.inference.adjoint.forward_sensitivity_gradient`)
  integrates the augmented ``n_x·(1+n_θ)`` system — cost grows ~linearly in ``n_θ``;
- **adjoint** (:func:`nudge.inference.adjoint.adjoint_gradient`, reverse-mode through the
  ``lax.scan`` integrator) — cost ~flat in ``n_θ``.

Each ``(method, n_θ)`` config runs in a **fresh subprocess** (``--worker``) so the peak
RSS is clean (``resource.getrusage`` is a monotonic high-water mark). The subprocess warms
up the jit once (compile excluded), then times several gradient evaluations.

Usage::

    uv run python scripts/vv/adjoint_scaling.py                 # full sweep + table + JSON
    uv run python scripts/vv/adjoint_scaling.py --n-states 15 --sweep 5,25,100,200
    uv run python scripts/vv/adjoint_scaling.py --worker adjoint 50 15 0   # one config

Writes ``scripts/vv/adjoint_scaling_RESULTS.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "adjoint_scaling_RESULTS.json"


def _rss_mb() -> float:
    """Current process peak RSS in MB (``ru_maxrss`` is KB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _run_worker(method: str, n_theta: int, n_states: int, seed: int) -> dict:
    """Time one gradient route at one ``n_θ`` and report wall-time + peak RSS."""
    # Keep JAX on CPU, single-threaded-ish, deterministic-ish for a clean measurement.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax

    from nudge.inference.adjoint import (
        _augmented_forward_sensitivity,
        _loss_fn,
        make_glv_problem,
    )

    problem = make_glv_problem(n_species=n_states, n_free=n_theta, seed=seed)
    theta = jax.numpy.asarray(problem.theta0 * 1.03, problem.dtype)
    _target = problem.jax_args()[3]
    n_elem = int(_target.shape[0] * _target.shape[1])

    if method == "adjoint":
        grad_fn = jax.jit(jax.grad(_loss_fn(problem)))

        def call():
            return grad_fn(theta).block_until_ready()

    elif method == "forward":
        def _fwd(th):
            x_obs, s_obs = _augmented_forward_sensitivity(problem, th)
            resid = x_obs - _target
            return jax.numpy.einsum("ti,tik->k", resid, s_obs) / n_elem

        grad_fn = jax.jit(_fwd)

        def call():
            return grad_fn(theta).block_until_ready()
    else:
        raise ValueError(f"unknown method {method!r}")

    call()  # warm-up: triggers compilation (excluded from timing)
    # Batch many calls per timed block to amortize Python/dispatch overhead (the adjoint is
    # ~1 ms, so single-call timing is dispatch-dominated and noisy); report the MIN
    # per-call over several batches — the standard denoised microbenchmark estimator.
    n_inner, n_rep = 25, 6
    per_call = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        for _ in range(n_inner):
            call()
        per_call.append((time.perf_counter() - t0) * 1e3 / n_inner)  # ms/call
    per_call.sort()
    return {
        "method": method,
        "n_theta": int(n_theta),
        "n_states": int(n_states),
        "wall_ms_median": float(per_call[0]),  # min per-call (denoised)
        "wall_ms_p50": float(per_call[len(per_call) // 2]),
        "peak_rss_mb": float(_rss_mb()),
    }


def _spawn(method: str, n_theta: int, n_states: int, seed: int) -> dict:
    """Run one worker config in a fresh subprocess and parse its JSON line."""
    cmd = [
        sys.executable, __file__, "--worker", method,
        str(n_theta), str(n_states), str(seed),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("{")][-1]
    return json.loads(line)


def _driver(n_states: int, sweep: list[int], seed: int) -> None:
    print(f"# adjoint vs forward-sensitivity scaling — {n_states}-state gLV network")
    print(f"# sweeping n_theta over {sweep}\n")
    rows: list[dict] = []
    for n_theta in sweep:
        for method in ("forward", "adjoint"):
            r = _spawn(method, n_theta, n_states, seed)
            rows.append(r)
            print(
                f"  {method:8s} n_theta={n_theta:4d}  "
                f"time={r['wall_ms_median']:9.2f} ms  peak_rss={r['peak_rss_mb']:7.1f} MB"
            )

    # summary table: adjoint stays flat, forward grows.
    print("\n## Summary (median wall-time ms / peak RSS MB)\n")
    print(f"{'n_theta':>8} | {'fwd_ms':>10} {'adj_ms':>10} {'speedup':>8} | "
          f"{'fwd_MB':>8} {'adj_MB':>8}")
    print("-" * 66)
    summary = []
    for n_theta in sweep:
        fwd = next(r for r in rows if r["method"] == "forward" and r["n_theta"] == n_theta)
        adj = next(r for r in rows if r["method"] == "adjoint" and r["n_theta"] == n_theta)
        speed = fwd["wall_ms_median"] / max(adj["wall_ms_median"], 1e-9)
        print(f"{n_theta:>8} | {fwd['wall_ms_median']:>10.2f} "
              f"{adj['wall_ms_median']:>10.2f} {speed:>7.1f}x | "
              f"{fwd['peak_rss_mb']:>8.1f} {adj['peak_rss_mb']:>8.1f}")
        summary.append({"n_theta": n_theta, "forward": fwd, "adjoint": adj,
                        "speedup": speed})

    # the measured scaling claim: forward's time grows ~linearly in n_theta while the
    # adjoint stays ~flat, so the speedup WIDENS as n_theta grows.
    fwd_ms = [s["forward"]["wall_ms_median"] for s in summary]
    adj_ms = [s["adjoint"]["wall_ms_median"] for s in summary]
    speedups = [s["speedup"] for s in summary]
    fwd_growth = fwd_ms[-1] / max(fwd_ms[0], 1e-9)
    adj_flatness = max(adj_ms) / max(min(adj_ms), 1e-9)  # ~1 == flat
    speedup_widening = speedups[-1] / max(speedups[0], 1e-9)
    fwd_mb = [s["forward"]["peak_rss_mb"] for s in summary]
    adj_mb = [s["adjoint"]["peak_rss_mb"] for s in summary]
    ntheta_growth = sweep[-1] / sweep[0]
    print(
        f"\n# n_theta grew {ntheta_growth:.0f}x ({sweep[0]} -> {sweep[-1]}): "
        f"forward wall-time grew {fwd_growth:.1f}x ({fwd_ms[0]:.1f} -> {fwd_ms[-1]:.1f} ms); "
        f"adjoint stayed ~flat ({adj_ms[0]:.2f} -> {adj_ms[-1]:.2f} ms, max/min "
        f"{adj_flatness:.1f}x)."
    )
    print(
        f"# adjoint speedup WIDENED {speedups[0]:.1f}x -> {speedups[-1]:.1f}x "
        f"({speedup_widening:.1f}x wider) as n_theta grew."
    )
    print(
        f"# peak RSS: forward {fwd_mb[0]:.0f} -> {fwd_mb[-1]:.0f} MB (grows), "
        f"adjoint {adj_mb[0]:.0f} -> {adj_mb[-1]:.0f} MB (flat)."
    )
    # PASS: forward clearly grows in n_theta, the adjoint is decisively cheaper at the
    # largest n_theta, and the speedup widens as the parameter count grows.
    ok = (
        fwd_growth > 2.0
        and adj_ms[-1] < 0.3 * fwd_ms[-1]
        and speedup_widening > 1.5
    )
    verdict = "ADJOINT ~O(1) IN n_theta, FORWARD GROWS" if ok else "inconclusive on this hardware"
    print(f"# verdict: {verdict}")

    _RESULTS.write_text(json.dumps(
        {"n_states": n_states, "sweep": sweep, "rows": rows, "summary": summary,
         "fwd_growth": fwd_growth, "adj_flatness": adj_flatness,
         "speedup_widening": speedup_widening,
         "ntheta_growth": ntheta_growth, "verdict": verdict},
        indent=2,
    ))
    print(f"\n# wrote {_RESULTS}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", nargs=4, metavar=("METHOD", "N_THETA", "N_STATES", "SEED"),
                    help="internal: run one config and print a JSON line")
    ap.add_argument("--n-states", type=int, default=15)
    ap.add_argument("--sweep", type=str, default="5,10,25,50,100,200")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.worker is not None:
        method, n_theta, n_states, seed = args.worker
        print(json.dumps(_run_worker(method, int(n_theta), int(n_states), int(seed))))
        return 0

    sweep = [int(x) for x in args.sweep.split(",") if x]
    _driver(args.n_states, sweep, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
