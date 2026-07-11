"""RED-TEAM re-scan (post-P1-fix): is there a confident-wrong band JUST UNDER gate 4b's
``off_shift`` cut of 2.5?

Target: ``nudge.inference.differential`` (``NUDGE-METHOD-010`` / ``NUDGE-LIM-016``), the
P1 fix (gate 4b: abstain when ``max(off_shift_a, off_shift_b) > _OFF_SHIFT_INFLATION_MAX
= 2.5``). The fix's separator was measured in ONE regime (N=3000, obs_sd=0.5): confident
additive-offset calls had ``off_shift >= 2.99``, genuine differences ``<= 1.96``, and 2.5
sits in that gap. But ``off_shift`` (a quantile RATIO) is essentially N-independent while
the BIC confidence scales with N. So a HIGHER-POWER regime (more cells / tighter modes)
should make a confident spurious ``*-diff`` appear at a SMALLER additive offset — hence a
smaller ``off_shift`` — potentially UNDER 2.5, where gate 4b stays silent.

Ground truth: ``simulate_context_pair(mechanism="none")`` — NO mechanistic difference. An
additive offset is added to context B's PERTURBED cells only (control clean). Honest
answer: ``no-difference`` / ``unresolved``. A confident ``*-diff`` output from the FIXED
code is by construction a case where ``off_shift <= 2.5`` (else gate 4b would fire) — a
NEW sub-2.5 hole the P1 fix does not cover.

Run: uv run python scripts/redteam/differential_sub25_band_probe.py [nseeds] [ncells] [obs_sd]
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

OFFSETS = [1.5, 2.0, 2.5, 3.0]
SCALE = 20.0
_DIFF = {"threshold-diff", "gain-diff", "ceiling-diff"}


def _run_one(
    seed: int, offset: float, n_cells: int, obs_sd: float
) -> tuple[str, object]:
    circuit = ras_switch_1node()
    ctx_a, ctx_b = simulate_context_pair(
        circuit,
        mechanism="none",
        n_cells=n_cells,
        scale_a=SCALE,
        scale_b=SCALE,
        obs_sd=obs_sd,
        seed=seed,
    )
    b_data = np.asarray(ctx_b.data, dtype=float) + offset
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


def run(nseeds: int, n_cells: int, obs_sd: float) -> int:
    print("=" * 84, flush=True)
    print(
        f"RE-SCAN: sub-2.5 off_shift band  (N={n_cells}, obs_sd={obs_sd}, scale={SCALE})",
        flush=True,
    )
    print("truth = no-difference; a confident *-diff with off_shift<=2.5 is a NEW HOLE")
    print("=" * 84, flush=True)
    holes = 0
    for seed in range(nseeds):
        call0, res0 = _run_one(seed, 0.0, n_cells, obs_sd)
        print(
            f"\nseed={seed} [offset=0.0 control] call={call0!r} "
            f"off_infl={max(res0.fit.off_shift_a, res0.fit.off_shift_b):.3f}",
            flush=True,
        )
        for off in OFFSETS:
            call, res = _run_one(seed, off, n_cells, obs_sd)
            fit = res.fit
            d_shared = fit.bic["shared"] - fit.bic[fit.best_diff]
            others = sorted(m for m in ("n", "K", "vmax") if m != fit.best_diff)
            runner = min(others, key=lambda m: fit.bic[m])
            d_runner = fit.bic[runner] - fit.bic[fit.best_diff]
            off_infl = max(fit.off_shift_a, fit.off_shift_b)
            is_diff = call in _DIFF
            # a NEW hole: a confident *-diff that slipped past gate 4b (off_shift <= 2.5)
            hole = is_diff and off_infl <= 2.5
            holes += int(hole)
            tag = "  <== NEW HOLE (sub-2.5)" if hole else (
                "  [gate4b HELD]" if off_infl > 2.5 else ""
            )
            print(
                f"seed={seed} [offset={off:>4.1f}] call={call!r}{tag}\n"
                f"    off_infl={off_infl:.3f} (gate4b cut 2.5)  best_diff={fit.best_diff!r}"
                f"  dBIC_shared={d_shared:.1f}  dBIC_runner({runner})={d_runner:.1f}",
                flush=True,
            )
    print("\n" + "=" * 84, flush=True)
    print(f"NEW sub-2.5 confident-wrong holes: {holes}", flush=True)
    return 0 if holes == 0 else 2


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    nc = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    sd = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    raise SystemExit(run(ns, nc, sd))
