"""RED-TEAM re-scan (post-P1-fix): the DEFLATING side of the perturbed-only multiplicative
confound — sharpens the documented ``NUDGE-LIM-016`` deflating bound.

Target: ``nudge.inference.differential`` (``NUDGE-METHOD-010`` / ``NUDGE-LIM-016``).

Companion to ``differential_multiplicative_confound.py`` (the INFLATING side, the NEW hole).
The P1 fix documents a one-sided residual: a *deflating* perturbed-only offset is unguarded
(gate 4b only catches OFF-baseline INFLATION). FINDINGS §P1 characterised that residual with
an ADDITIVE deflation ("in the partial sweep a deflating offset produced no-difference, not a
confident-wrong"). This probe shows the residual is SHARPER than that on the MULTIPLICATIVE
channel: a constant factor ``c < 1`` on context B's PERTURBED cells only (control clean)
produces a CONFIDENT (lowered) ``ceiling-diff`` — off_shift stays ~1 (a factor leaves the
near-zero OFF baseline near zero), so gate 4b is silent. A deflating ADDITIVE offset
(subtract a constant) still reads ``no-difference`` (matching FINDINGS).

This is NOT a new hole — it is the documented deflating bound, now shown to actually emit a
confident-wrong (worse than the "no-difference" the additive sweep observed). Reported for
honesty; the genuinely NEW, unanticipated hole is the INFLATING multiplicative side.

Ground truth: ``simulate_context_pair(mechanism="none")`` -> NO mechanistic difference.

Run: uv run python scripts/redteam/differential_multiplicative_deflating.py [nseeds]
Touches no src/ code and no fail-safe margins -- diagnostic only.
"""

from __future__ import annotations

import sys

import numpy as np

from nudge.circuits import ras_switch_1node
from nudge.inference.differential import (
    Context,
    attribute_differential,
    simulate_context_pair,
)

_DIFF = {"threshold-diff", "gain-diff", "ceiling-diff"}


def _run_one(seed: int, kind: str, amt: float) -> tuple[str, object]:
    circuit = ras_switch_1node()
    ctx_a, ctx_b = simulate_context_pair(
        circuit,
        mechanism="none",
        n_cells=3000,
        scale_a=20.0,
        scale_b=20.0,
        obs_sd=0.5,
        seed=seed,
    )
    b = np.asarray(ctx_b.data, dtype=float)
    b = b * amt if kind == "mult" else np.clip(b - amt, 0.0, None)
    res = attribute_differential(
        ctx_a,
        Context(name="B", data=b, control=ctx_b.control),
        circuit,
        target_edge=0,
        k_modes=2,
        steps=200,
        seed=seed,
        n_boot=0,
    )
    return res.call, res


def run(nseeds: int) -> int:
    print("=" * 84, flush=True)
    print("RE-SCAN: DEFLATING perturbed-only confound (documented LIM-016 bound, sharpened)")
    print("truth = no-difference; a confident *-diff confirms + sharpens the deflating bound")
    print("=" * 84, flush=True)
    n_mult = 0
    for seed in range(nseeds):
        for kind, amt in (("mult", 0.5), ("mult", 0.7), ("sub", 1.0)):
            call, res = _run_one(seed, kind, amt)
            fit = res.fit
            off = max(fit.off_shift_a, fit.off_shift_b)
            d_shared = fit.bic["shared"] - fit.bic[fit.best_diff]
            hole = call in _DIFF
            if kind == "mult":
                n_mult += int(hole)
            tag = "  <== DEFLATE CONFIDENT-WRONG" if hole else ""
            print(
                f"seed={seed} {kind} amt={amt} call={call!r}{tag}  off_shift={off:.3f}"
                f"  best_diff={fit.best_diff!r}  dBIC_shared={d_shared:.1f}",
                flush=True,
            )
    print("\n" + "=" * 84, flush=True)
    print(f"deflating MULTIPLICATIVE confident-wrong (sharpens the bound): {n_mult}")
    return 0


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    raise SystemExit(run(ns))
