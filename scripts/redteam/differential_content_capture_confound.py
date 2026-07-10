"""RED-TEAM (P4 re-scan, realism check): a SMOOTH content-dependent capture bias on ONE
context's PERTURBED cells fakes a confident ``ceiling-diff`` while keeping every guard
statistic in-band. This is the realistic form of the ON-mode-subset hole.

Target: ``nudge.inference.differential`` (``NUDGE-METHOD-010`` / ``NUDGE-LIM-016``).

The artifact: droplet capture efficiency that rises smoothly with cell mRNA content is a
documented scRNA-seq effect (larger / higher-content cells are recovered more
efficiently). Modelled as a per-cell multiplicative gain
``g_i = 1 + b * sigmoid((s_i - median_s) / spread)`` where ``s_i`` is cell i's total
activity — ≈ 1 for OFF cells, ≈ 1 + b for ON cells, smooth in between. Applied to ONE
context's PERTURBED cells only (its control processed cleanly), it is a purely TECHNICAL
artifact: NO biological difference between the contexts.

Gate 4c (P4) keys on the OFF-cluster SPREAD, assuming a UNIFORM per-cell scale ``c`` that
would dilate the OFF cluster (``off_scale`` ≈ c). A content-dependent gain leaves the OFF
cluster ≈ unchanged (sigmoid ≈ 0 there) so ``off_scale`` ≈ 1 and gate 4c passes; it moves
only the ON mode — exactly a genuine ceiling fingerprint — so NUDGE emits a confident
``ceiling-diff`` where the honest answer is abstention (the artifact is indistinguishable
from a real ceiling change: the same fundamental degeneracy P4 abstains on for a deflating
scale, but here on the inflation side the guard does NOT abstain).

Ground truth: ``simulate_context_pair(mechanism="none")``. Honest answer: ``no-difference``
/ ``unresolved``. A confident ``*-diff`` is the HOLE.

Run: uv run python scripts/redteam/differential_content_capture_confound.py [nseeds]
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
# smooth content-dependent gain amplitudes b (ON cells scaled up to ~1+b).
AMPLITUDES = [1.2, 2.0]


def _content_capture(data: np.ndarray, b: float) -> np.ndarray:
    """Per-cell gain rising smoothly with cell content (≈1 at OFF, ≈1+b at ON)."""
    x = np.asarray(data, dtype=float)
    s = x.sum(axis=1)
    med = np.median(s)
    spread = np.median(np.abs(s - med)) + 1e-9
    gain = 1.0 + b / (1.0 + np.exp(-(s - med) / spread))
    return x * gain[:, None]


def _run_one(seed: int, b: float) -> tuple[str, object]:
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
    b_data = _content_capture(ctx_b.data, b)
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
    print("RED-TEAM P4-rescan: SMOOTH content-dependent capture on PERTURBED cells")
    print("truth = no biological difference; a confident *-diff is the HOLE")
    print("=" * 80, flush=True)
    holes = 0
    for seed in range(nseeds):
        for b in AMPLITUDES:
            call, res = _run_one(seed, b)
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
                f"\nseed={seed}  [content b={b}]  call={call!r}{tag}\n"
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
