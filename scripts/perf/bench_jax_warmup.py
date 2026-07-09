"""Measure JAX jit warmup vs steady-state across NUDGE's core jitted paths.

Answers perf question (2): how much of each first call is one-time XLA compilation, and
— importantly — whether NUDGE **recompiles per call** (a jit cache miss) that a stable
static-arg design or a warmup would avoid. Two probes:

  A. ``Circuit`` fixed-point kernel (``_nd_kernel``, per-topology cached): first call
     (trace+compile) vs subsequent (execute), and whether a *new topology* recompiles.
  B. ``fit_lyapunov_parameters``: its optax ``step`` is a **fresh closure jitted inside
     each call**, so every attribution fit pays the compile again — measured here.

All circuits are tiny (1–2 species). Runs in a few seconds.
Run: ``uv run python scripts/perf/bench_jax_warmup.py``.
"""

from __future__ import annotations

import time

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef


def sw() -> Circuit:
    """A 1-species self-activation bistable switch."""
    return Circuit(
        [SpeciesDef("A", basal=0.2, decay=1.0)],
        [EdgeDef(0, 0, "hill_activation", K=1.0, n=6.0, vmax=3.0)],
    )


def toggle() -> Circuit:
    return Circuit(
        [SpeciesDef("A", basal=0.5, decay=1.0), SpeciesDef("B", basal=0.5, decay=1.0)],
        [EdgeDef(1, 0, "hill_repression", K=1.0, n=3.0, vmax=3.0),
         EdgeDef(0, 1, "hill_repression", K=1.0, n=3.0, vmax=3.0)],
    )


def t(fn, *a, reps=1):
    best = float("inf")
    for _ in range(reps):
        s = time.perf_counter()
        fn(*a)
        best = min(best, time.perf_counter() - s)
    return best


def main() -> None:
    from nudge.core.circuit import _ND_KERNEL_CACHE

    print("=== A. Circuit N-D fixed-point kernel (_nd_kernel, per-topology cache) ===")
    _ND_KERNEL_CACHE.clear()
    tg = toggle()
    t_cold = t(tg.fixed_points)               # trace + compile for the toggle topology
    t_warm = t(tg.fixed_points, reps=20)       # cached execute (kinetics are traced)
    print(f"    toggle 1st call (trace+compile): {t_cold*1000:8.1f} ms")
    print(f"    toggle warm call (execute)     : {t_warm*1000:8.2f} ms  "
          f"→ {t_cold/max(t_warm,1e-9):.0f}x")
    # a DIFFERENT topology → a fresh compile (expected: cache keyed on structure)
    t_sw = t(sw().fixed_points)
    print(f"    switch (new topology) 1st call : {t_sw*1000:8.1f} ms  (fresh compile)")
    print(f"    kernel cache now holds {len(_ND_KERNEL_CACHE)} compiled topologies")

    print("\n=== B. fit_lyapunov_parameters: is `step` recompiled per call? ===")
    import jax

    from nudge.inference.lyapunov import fit_lyapunov_parameters, sample_lna_mixture
    c = sw()
    data = sample_lna_mixture(c, 400, jax.random.PRNGKey(0), scale=30.0, obs_sd=0.1)
    # free-n restricted fit at a small step budget, timed cold then warm
    def fit(steps):
        return fit_lyapunov_parameters(
            data, c, [("edge", 0, "n")], k_modes=2, steps=steps,
            scale_init=30.0, fit_scale=False, fit_obs=False,
        )
    t_cold = t(lambda: fit(30))
    t_warm = t(lambda: fit(30))
    t_cold2 = t(lambda: fit(30))
    print(f"    30-step fit call #1: {t_cold*1000:8.1f} ms")
    print(f"    30-step fit call #2: {t_warm*1000:8.1f} ms")
    print(f"    30-step fit call #3: {t_cold2*1000:8.1f} ms")
    print("    (if #2/#3 ≈ #1, the jitted `step` closure recompiles every call — a "
          "per-attribution warmup cost, not amortized)")
    # per-step marginal cost from a longer run (compile amortized)
    t150 = t(lambda: fit(150))
    t30 = t_warm
    per_step = (t150 - t30) / (150 - 30) * 1000
    print(f"    marginal cost/optax step ≈ {per_step:.2f} ms  "
          f"(compile overhead ≈ {(t30 - 30*per_step/1000)*1000:.0f} ms/call)")


if __name__ == "__main__":
    main()
