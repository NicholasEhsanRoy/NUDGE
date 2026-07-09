"""Profile the dose-response fit: JAX jit warmup vs numerical work, and the bootstrap.

Answers three perf questions with measurements (no guessing):
  (2) how much of the FIRST call is one-time JAX jit compilation vs actual fitting;
  (3) is the ``n_boot`` bootstrap loop the dominant cost, and how do wall-time and the
      CI on ``n`` move as ``n_boot`` varies (→ a defensible default);
  (5) does a warmup call make subsequent (demo) calls fast.

Everything is tiny (a synthetic 16-point Hill curve) and runs in a few seconds.
Run: ``uv run python scripts/perf/bench_dose_response.py``.
"""

from __future__ import annotations

import time

import numpy as np

from nudge.inference.dose_response import _jax_model, fit_dose_response


def synth(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dose = np.linspace(0.0, 2.0, 16)
    k, n, amp, floor = 0.8, 5.0, 3.0, 0.5
    y = floor + amp * (1 - dose**n / (k**n + dose**n))  # repress
    y = y + rng.normal(0, 0.05, size=y.shape)
    return dose, y


def time_call(fn, *a, **k) -> tuple[float, object]:
    t = time.perf_counter()
    r = fn(*a, **k)
    return time.perf_counter() - t, r


def main() -> None:
    dose, y = synth()

    # (2)+(5) jit warmup vs warm. _jax_model is lru_cached; the first fit compiles
    # predict+jac for both free_n=True and free_n=False (two direction/free_n keys).
    _jax_model.cache_clear()
    t_cold, _ = time_call(fit_dose_response, dose, y, direction="repress", n_boot=0)
    t_warm, _ = time_call(fit_dose_response, dose, y, direction="repress", n_boot=0)
    print("=== jit warmup (n_boot=0: two MLE fits only) ===")
    print(f"    cold first call : {t_cold*1000:8.1f} ms  (incl. JAX jit compile)")
    print(f"    warm second call: {t_warm*1000:8.1f} ms")
    print(f"    one-time jit warmup cost ≈ {(t_cold - t_warm)*1000:.0f} ms")

    # explicit warmup helper: compile the two models on trivial data
    _jax_model.cache_clear()
    def warmup() -> None:
        d = np.array([0.0, 1.0, 2.0, 3.0])
        yy = np.array([0.0, 1.0, 2.0, 3.0])
        fit_dose_response(d, yy, direction="repress", n_boot=0)
    t_wu, _ = time_call(warmup)
    t_after, _ = time_call(fit_dose_response, dose, y, direction="repress", n_boot=0)
    print(f"    explicit warmup() call: {t_wu*1000:.0f} ms → next real fit "
          f"{t_after*1000:.1f} ms")

    # (3) bootstrap dominance + n_boot sweep (warm; CI stability)
    print("\n=== n_boot sweep (warm) — wall time and 95% CI on n ===")
    fit_dose_response(dose, y, n_boot=0)  # ensure warm
    base_mle = None
    for nb in (0, 50, 100, 200, 500, 1000):
        ts = []
        cis = []
        for s in range(3):
            t, fit = time_call(fit_dose_response, dose, y, n_boot=nb, seed=s)
            ts.append(t)
            cis.append(fit.ci_n)
        tmed = sorted(ts)[len(ts) // 2]
        if base_mle is None and nb == 0:
            base_mle = tmed
        lo = np.mean([c[0] for c in cis]) if nb else float("nan")
        hi = np.mean([c[1] for c in cis]) if nb else float("nan")
        width = hi - lo
        sd_lo = np.std([c[0] for c in cis]) if nb else 0.0
        sd_hi = np.std([c[1] for c in cis]) if nb else 0.0
        per_boot = (tmed - (base_mle or 0)) / nb * 1e3 if nb else 0.0
        print(f"    n_boot={nb:5d}  {tmed*1000:8.1f} ms  "
              f"CI≈[{lo:5.2f},{hi:5.2f}] w={width:5.2f} "
              f"(seed-sd lo/hi {sd_lo:.2f}/{sd_hi:.2f})  {per_boot:.2f} ms/boot")
    print(f"\n    (MLE baseline n_boot=0 ≈ {(base_mle or 0)*1000:.1f} ms; the rest is "
          "the bootstrap loop → it dominates at the default n_boot=500)")


if __name__ == "__main__":
    main()
