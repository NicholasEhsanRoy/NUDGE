"""RED-TEAM (round 3): an ADDITIVE batch offset on the PERTURBED cells of ONE context
(but NOT its control) fakes a confident ``*-diff`` past the per-context depth-ratio guard.

Target: ``nudge.inference.differential`` (``NUDGE-METHOD-010`` / ``NUDGE-LIM-016``).
Flagged but never tested in round 1 (``design/FAILSAFE_REDTEAM.md`` "Not adversarially
exercised"): the differential's confound guard pins depth PER CONTEXT from each context's
own **control** (``calibrate_from_wt``) and abstains when the two controls' depths differ
beyond ``depth_ratio_max``. But that guard keys on the CONTROLS. An additive count/ambient
offset on context B's **perturbed** cells only — its control left clean — leaves the
control-derived ``depth_ratio`` ≈ 1, so gate 2 (the LIM-016 depth guard) NEVER ENGAGES.
The offset shifts B's perturbed mode means, which the joint BIC then explains with a
per-context knob → a confident ``ceiling-diff`` / ``gain-diff`` / ``threshold-diff`` where
the ground truth is **no-difference**.

Ground truth: ``simulate_context_pair(mechanism="none")`` — the SAME perturbation, no
mechanistic difference. Then an additive offset is added to context B's ``data`` (perturbed
cells) only; both controls are untouched. Honest answer: ``no-difference`` / ``unresolved``.
A confident ``*-diff`` is the hole.

Run: uv run python scripts/redteam/differential_additive_confound.py [nseeds]
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

# Additive offset magnitudes to try (added to context B's PERTURBED activity, all cells).
OFFSETS = [1.0, 2.0, 3.0, 5.0]
SCALE = 20.0
N_CELLS = 3000


def _run_one(seed: int, offset: float) -> tuple[str, object]:
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
    # The attack: an ADDITIVE offset on context B's PERTURBED cells only; control clean.
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


def run(nseeds: int) -> int:
    print("=" * 80, flush=True)
    print("RED-TEAM r3: additive offset on ONE context's PERTURBED cells (control clean)")
    print("truth = no-difference; a confident *-diff is the HOLE (LIM-016 guard bypass)")
    print("=" * 80, flush=True)
    holes = 0
    for seed in range(nseeds):
        # baseline: no offset -> must be no-difference / unresolved (positive control).
        call0, res0 = _run_one(seed, 0.0)
        print(
            f"\nseed={seed}  [offset=0.0 control]  call={call0!r}  "
            f"depth_ratio={res0.fit.depth_ratio:.3f}  selected={res0.fit.selected!r}",
            flush=True,
        )
        for off in OFFSETS:
            call, res = _run_one(seed, off)
            fit = res.fit
            d_shared = fit.bic["shared"] - fit.bic[fit.best_diff]
            others = sorted(m for m in ("n", "K", "vmax") if m != fit.best_diff)
            runner = min(others, key=lambda m: fit.bic[m])
            d_runner = fit.bic[runner] - fit.bic[fit.best_diff]
            hole = call in {"threshold-diff", "gain-diff", "ceiling-diff"}
            holes += int(hole)
            tag = "  <== HOLE" if hole else ""
            print(
                f"seed={seed}  [offset={off:>4.1f}]  call={call!r}{tag}\n"
                f"    depth_ratio={fit.depth_ratio:.3f} (guard keys on controls, "
                f"gate skips if <1.5)  best_diff={fit.best_diff!r}\n"
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
