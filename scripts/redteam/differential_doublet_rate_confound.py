"""RED-TEAM (P4 re-scan, unimpeachable ground truth): a higher DOUBLET rate in ONE
context's PERTURBED cells than in its control fakes a confident ``ceiling-diff`` while
keeping every guard statistic in-band.

Target: ``nudge.inference.differential`` (``NUDGE-METHOD-010`` / ``NUDGE-LIM-016``).

Why this is the honest corroboration of the ON-mode-subset hole
(``differential_subset_scale_confound.py``): a doublet is two cells captured in one
droplet, so its total counts are ~2x — a purely TECHNICAL artifact with NO biological
difference. Doublet rate routinely differs between conditions / batches (loading density,
viability). This is a *state-dependent* (a doublet doubles whatever state the two cells
were in) multiplicative artifact on a SUBSET of perturbed cells only. Gate 4c (P4) keys on
the OFF-cluster SPREAD assuming a UNIFORM per-cell scale; a subset-of-cells doubling does
not uniformly dilate the OFF cluster, so ``off_scale`` stays in the measured band
[0.80, 1.30] and gate 4c passes — yielding a confident ``ceiling-diff`` where the truth is
no biological difference (the honest answer is abstention).

Ground truth: ``simulate_context_pair(mechanism="none")``. The attack: replace a random
fraction ``f`` of context B's PERTURBED cells with doublets (each = the sum of two random
perturbed cells). B's control is left clean. Honest answer: ``no-difference`` /
``unresolved``. A confident ``*-diff`` is the HOLE.

Run: uv run python scripts/redteam/differential_doublet_rate_confound.py [nseeds]
Touches no src/ code and no fail-safe margins — diagnostic only.
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

SCALE = 20.0
N_CELLS = 3000
# doublet fractions injected into context B's perturbed cells only.
FRACTIONS = [0.10, 0.18]


def _inject_doublets(data: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Replace a random ``frac`` of rows with doublets (row = sum of two random rows)."""
    out = np.asarray(data, dtype=float).copy()
    n = out.shape[0]
    k = int(round(frac * n))
    if k == 0:
        return out
    idx = rng.choice(n, size=k, replace=False)
    partners = rng.integers(0, n, size=k)
    out[idx] = out[idx] + np.asarray(data, dtype=float)[partners]
    return out


def _run_one(seed: int, frac: float) -> tuple[str, object]:
    circuit = ras_switch_1node()
    ctx_a, ctx_b = simulate_context_pair(
        circuit,
        mechanism="none",
        n_cells=N_CELLS,
        scale_a=SCALE,
        scale_b=SCALE,
        obs_sd=0.5,
        seed=seed,
    )
    rng = np.random.default_rng(1000 + seed)
    b_data = _inject_doublets(ctx_b.data, frac, rng)
    ctx_b_attacked = Context(name="B", data=b_data, control=ctx_b.control)
    res = attribute_differential(
        ctx_a,
        ctx_b_attacked,
        circuit,
        target_edge=0,
        k_modes=2,
        steps=200,
        seed=seed,
        n_boot=0,
    )
    return res.call, res


def run(nseeds: int) -> int:
    print("=" * 80, flush=True)
    print("RED-TEAM P4-rescan: higher DOUBLET rate in ONE context's PERTURBED cells")
    print("truth = no biological difference; a confident *-diff is the HOLE")
    print("=" * 80, flush=True)
    holes = 0
    for seed in range(nseeds):
        for frac in FRACTIONS:
            call, res = _run_one(seed, frac)
            fit = res.fit
            d_shared = fit.bic["shared"] - fit.bic[fit.best_diff]
            others = sorted(m for m in ("n", "K", "vmax") if m != fit.best_diff)
            runner = min(others, key=lambda m: fit.bic[m])
            d_runner = fit.bic[runner] - fit.bic[fit.best_diff]
            off_infl = max(
                v for v in (fit.off_shift_a, fit.off_shift_b) if np.isfinite(v)
            )
            off_scale = max(
                (v for v in (fit.off_scale_a, fit.off_scale_b) if np.isfinite(v)),
                key=lambda v: abs(np.log(v)) if v > 0 else 0.0,
            )
            hole = call in {"threshold-diff", "gain-diff", "ceiling-diff"}
            holes += int(hole)
            tag = "  <== HOLE" if hole else ""
            va = fit.est_a[fit.best_diff][fit.best_diff]
            vb = fit.est_b[fit.best_diff][fit.best_diff]
            print(
                f"\nseed={seed}  [doublet frac={frac}]  call={call!r}{tag}\n"
                f"    off_shift={off_infl:.3f} (4b abstains >2.5)  "
                f"off_scale={off_scale:.3f} (4c band [0.80,1.30])\n"
                f"    depth_ratio={fit.depth_ratio:.3f}  best_diff={fit.best_diff!r} "
                f"({va:.3g} -> {vb:.3g})\n"
                f"    dBIC vs shared={d_shared:.1f}  dBIC vs runner "
                f"({runner})={d_runner:.1f}",
                flush=True,
            )
            if hole:
                print(f"    reason: {res.reason}", flush=True)
    print("\n" + "=" * 80, flush=True)
    print(f"confident-wrong *-diff calls (HOLES): {holes}", flush=True)
    return 0 if holes == 0 else 2


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    raise SystemExit(run(n))
