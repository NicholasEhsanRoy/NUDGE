"""RED-TEAM (P1 re-scan): a MULTIPLICATIVE scale on the PERTURBED cells of ONE context
(but NOT its control) fakes a confident ``ceiling-diff`` past BOTH the depth-ratio guard
(gate 2) AND the additive-offset OFF-baseline guard (gate 4b, the P1 fix).

Target: ``nudge.inference.differential`` (``NUDGE-METHOD-010`` / ``NUDGE-LIM-016``).
The P1 fix (gate 4b) measures the perturbed OFF baseline vs the control and abstains when it
is *translated up* (``off_shift`` ≫ 1) — the fingerprint of an ADDITIVE offset. A
MULTIPLICATIVE factor ``c`` multiplies the near-zero OFF baseline to a still-near-zero value,
so the crude 0.3-quantile ``off_shift`` stays ≈ 1 and gate 4b is SILENT. Meanwhile the factor
scales the ON mode 1:1 with a ceiling (``v_max``) difference — the depth is pinned from the
CLEAN control, so the joint fit must explain the scaled ON mode via kinetics → a confident
spurious ``ceiling-diff`` where the truth is **no-difference**.

Ground truth: ``simulate_context_pair(mechanism="none")`` — the SAME perturbation, no
mechanistic difference. Then context B's ``data`` (perturbed cells) only is MULTIPLIED by a
constant ``c``; both controls are untouched. Honest answer: ``no-difference`` / ``unresolved``.
A confident ``*-diff`` (especially ``ceiling-diff``) is the hole.

Run: uv run python scripts/redteam/differential_multiplicative_confound.py [nseeds]
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

# Multiplicative factors on context B's PERTURBED activity (all cells). Inflating (>1) and
# deflating (<1) — the red-team found BOTH fake a ceiling-diff.
FACTORS = [1.5, 2.0, 2.4, 0.7, 0.5]
SCALE = 20.0
N_CELLS = 3000


def _run_one(seed: int, factor: float) -> tuple[str, object]:
    circuit = ras_switch_1node()
    ctx_a, ctx_b = simulate_context_pair(
        circuit,
        mechanism="none",  # ground truth: NO mechanistic difference between contexts
        n_cells=N_CELLS,
        scale_a=SCALE,
        scale_b=SCALE,
        obs_sd=0.5,
        seed=seed,
    )
    # The attack: a MULTIPLICATIVE factor on context B's PERTURBED cells only; control clean.
    b_data = np.asarray(ctx_b.data, dtype=float) * factor
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
    print("RED-TEAM P1-rescan: MULTIPLICATIVE scale on ONE context's PERTURBED cells")
    print("truth = no-difference; a confident *-diff is the HOLE (LIM-016 gate 4b bypass)")
    print("=" * 80, flush=True)
    holes = 0
    for seed in range(nseeds):
        # baseline: factor 1.0 -> must be no-difference / unresolved (positive control).
        call0, res0 = _run_one(seed, 1.0)
        print(
            f"\nseed={seed}  [factor=1.0 control]  call={call0!r}  "
            f"depth_ratio={res0.fit.depth_ratio:.3f}  selected={res0.fit.selected!r}",
            flush=True,
        )
        for c in FACTORS:
            call, res = _run_one(seed, c)
            fit = res.fit
            d_shared = fit.bic["shared"] - fit.bic[fit.best_diff]
            others = sorted(m for m in ("n", "K", "vmax") if m != fit.best_diff)
            runner = min(others, key=lambda m: fit.bic[m])
            d_runner = fit.bic[runner] - fit.bic[fit.best_diff]
            off_infl = max(
                v for v in (fit.off_shift_a, fit.off_shift_b) if np.isfinite(v)
            )
            hole = call in {"threshold-diff", "gain-diff", "ceiling-diff"}
            holes += int(hole)
            tag = "  <== HOLE" if hole else ""
            va = fit.est_a[fit.best_diff][fit.best_diff]
            vb = fit.est_b[fit.best_diff][fit.best_diff]
            print(
                f"seed={seed}  [factor={c:>4.1f}]  call={call!r}{tag}\n"
                f"    depth_ratio={fit.depth_ratio:.3f}  off_infl={off_infl:.3f} "
                f"(gate 4b abstains > 2.5)  best_diff={fit.best_diff!r} "
                f"({va:.3g} -> {vb:.3g})\n"
                f"    dBIC vs shared={d_shared:.1f}  dBIC vs runner "
                f"({runner})={d_runner:.1f}  (margins 6.0/6.0)",
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
