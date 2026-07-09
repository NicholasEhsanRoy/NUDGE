"""Pre-compile the hot JAX paths so the FIRST real call in a demo isn't slow.

JAX compiles per-process on first use — the dose-response fit's cached ``_jax_model``
(~405 ms first call → ~55 ms after) and the N-D fixed-point kernel ``_nd_kernel``
(~512 ms → ~2 ms after, a 257× warm win; ``design/PERFORMANCE.md``). Both compile ONCE
per process and are then cached, so a demo running in a long-lived process — the MCP
server, a notebook kernel, an interactive session — pays that latency up front via
:func:`warmup`, and every subsequent attribution is snappy.

It runs tiny synthetic fits: no real data, no global config changes, and it's idempotent
(a process-level guard, so calling it repeatedly is free). A one-shot CLI command that
does a single fit does not benefit (you pay the compile either way), which is why warmup
is wired into the long-lived entry points (the MCP server; a notebook's first cell) and
exposed as ``nudge warmup`` / ``nudge.warmup()``, not run on every CLI invocation.
"""

from __future__ import annotations

import time

_WARMED = False


def warmup(*, quiet: bool = True) -> float:
    """Compile the cached hot JAX paths on tiny dummy data (idempotent per process).

    Returns the seconds spent compiling (``0.0`` if already warm). Triggers the two
    JITs that are cached across calls — the dose-response model/Jacobian and the circuit
    fixed-point kernel — so the first *real* attribution afterwards skips compilation.
    """
    global _WARMED
    if _WARMED:
        return 0.0

    t0 = time.perf_counter()
    import numpy as np

    # 1) dose-response: compiles the lru_cached, jitted `_jax_model` + its autodiff jac.
    from nudge.inference.dose_response import fit_dose_response

    dose = np.linspace(0.0, 1.0, 6)
    resp = 0.2 + 0.8 * (0.5**4) / (0.5**4 + np.maximum(dose, 1e-9) ** 4)
    fit_dose_response(dose, resp, direction="repress", n_boot=0)  # MLE only

    # 2) circuit: compiles the per-topology-cached `_nd_kernel` (fixed points) + the LNA
    #    covariance solve — the paths single-cell attribution reuses.
    from nudge.circuits import ras_switch_1node

    circuit = ras_switch_1node()
    circuit.fixed_points()
    circuit.mode_covariances()

    _WARMED = True
    dt = time.perf_counter() - t0
    if not quiet:
        print(f"nudge: warmed JAX compile caches in {dt:.2f}s")
    return dt
