"""RED-TEAM re-scan (post-P1-fix): a MULTIPLICATIVE perturbed-only scale on ONE context
slips past BOTH the control-keyed depth guard (gate 2) AND the additive off_shift guard
(gate 4b) and fakes a confident ``ceiling-diff``.

Target: ``nudge.inference.differential`` (``NUDGE-METHOD-010`` / ``NUDGE-LIM-016``).

The P1 fix added gate 4b: abstain when ``max(off_shift_a, off_shift_b) > 2.5``. It was
MEASURED against an ADDITIVE offset, which TRANSLATES the perturbed OFF baseline up and so
trips off_shift only once it is large enough to fake a diff (off_shift >= 2.99 for a
confident additive call). This probe attacks a DIFFERENT channel the fix never measured: a
constant MULTIPLICATIVE factor ``c`` on context B's PERTURBED cells only (its control left
clean) — a per-condition batch / sequencing-depth / capture-efficiency difference between
the measurement of B's perturbed cells and B's control (very common; different plate/day).

Why it evades every gate:
  * Gate 2 (depth_ratio) keys on the two CONTROLS (both clean) -> depth_ratio ~ 1.
  * Gate 4b (off_shift) sees only ``off_shift ~ c`` because a multiplicative factor scales
    the OFF-mode quantile by exactly ``c`` (not the ``translation`` an additive offset does)
    -- so a MODEST ``c`` (<= ~2.4) keeps off_shift under the 2.5 cut.
  * A global perturbed-only scale is aliased by the ceiling knob (v_max scales the ON mode),
    and depth is PINNED from the clean control so the fit cannot absorb it as depth.

Ground truth: ``simulate_context_pair(mechanism="none")`` -> NO mechanistic difference.
Honest answer: ``no-difference`` / ``unresolved``. A confident ``*-diff`` is the HOLE.

Run: uv run python scripts/redteam/differential_multiplicative_confound.py [nseeds]
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

FACTORS = [1.5, 2.0, 2.4]
SCALE = 20.0
N_CELLS = 3000
OBS_SD = 0.5
_DIFF = {"threshold-diff", "gain-diff", "ceiling-diff"}


def _run_one(seed: int, factor: float) -> tuple[str, object]:
    circuit = ras_switch_1node()
    ctx_a, ctx_b = simulate_context_pair(
        circuit,
        mechanism="none",  # ground truth: NO mechanistic difference
        n_cells=N_CELLS,
        scale_a=SCALE,
        scale_b=SCALE,
        obs_sd=OBS_SD,
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
    print("=" * 84, flush=True)
    print(
        f"RE-SCAN: MULTIPLICATIVE perturbed-only scale  (N={N_CELLS}, obs_sd={OBS_SD})",
        flush=True,
    )
    print("truth = no-difference; a confident *-diff with off_shift<=2.5 is a NEW HOLE")
    print("=" * 84, flush=True)
    holes = 0
    for seed in range(nseeds):
        call0, res0 = _run_one(seed, 1.0)
        print(
            f"\nseed={seed} [factor=1.0 control] call={call0!r} "
            f"off_infl={max(res0.fit.off_shift_a, res0.fit.off_shift_b):.3f} "
            f"depth_ratio={res0.fit.depth_ratio:.3f}",
            flush=True,
        )
        for fac in FACTORS:
            call, res = _run_one(seed, fac)
            fit = res.fit
            d_shared = fit.bic["shared"] - fit.bic[fit.best_diff]
            others = sorted(m for m in ("n", "K", "vmax") if m != fit.best_diff)
            runner = min(others, key=lambda m: fit.bic[m])
            d_runner = fit.bic[runner] - fit.bic[fit.best_diff]
            off_infl = max(fit.off_shift_a, fit.off_shift_b)
            is_diff = call in _DIFF
            hole = is_diff and off_infl <= 2.5  # slipped past gate 4b
            holes += int(hole)
            tag = "  <== NEW HOLE" if hole else (
                "  [gate4b HELD]" if off_infl > 2.5 else ""
            )
            va = fit.est_a[fit.best_diff][fit.best_diff]
            vb = fit.est_b[fit.best_diff][fit.best_diff]
            print(
                f"seed={seed} [factor={fac:>4.1f}] call={call!r}{tag}\n"
                f"    off_infl={off_infl:.3f} (cut 2.5)  depth_ratio={fit.depth_ratio:.3f}"
                f"  best_diff={fit.best_diff!r} ({va:.3g}->{vb:.3g})\n"
                f"    dBIC_shared={d_shared:.1f}  dBIC_runner({runner})={d_runner:.1f}"
                f"  (margins 6.0/6.0)",
                flush=True,
            )
            if hole:
                print(f"    reason: {res.reason}", flush=True)
    print("\n" + "=" * 84, flush=True)
    print(f"NEW confident-wrong holes (sub-2.5 multiplicative): {holes}", flush=True)
    return 0 if holes == 0 else 2


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    raise SystemExit(run(ns))
