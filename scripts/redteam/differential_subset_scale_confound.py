"""RED-TEAM (P4 re-scan): a MULTIPLICATIVE scale applied to a SUBSET of ONE context's
PERTURBED cells (a knife-edge scale on strictly the ON-mode / above-median cells) evades
gate 4c's OFF-cluster fingerprint and yields a confident ``*-diff``.

**HONEST VERDICT (see the sibling probes): this is NOT a valid confident-wrong hole.**
The evading construction is (i) observationally IDENTICAL to a genuine ceiling change — it
raises the ON mode and leaves the OFF-cluster spread ≈ 1, which is *by gate 4c's own
definition* the fingerprint of a real ceiling difference (genuine ceiling ×1.4–×4 →
``off_scale`` ≤ 1.18; this attack → ``off_scale`` ≈ 1.0–1.1) — so ``ceiling-diff`` is not
distinguishably wrong given the data; and (ii) not producible by any plausible physical
capture process (a step exactly at the population median). Every REALISTIC sibling confound
is CAUGHT: a smooth content-dependent capture bias trips gate 4c (its gain bleeds into the
upper OFF cluster → ``off_scale`` 1.45–1.81 > 1.30 → abstain,
``differential_content_capture_confound.py``); a doublet-rate difference abstains via the
gate-4 tie (``differential_doublet_rate_confound.py``); a uniform scale is the closed P4
case (``differential_multiplicative_confound.py`` → HOLES 0). So this probe documents the
PRECISE SCOPE of P4's "INFLATION is CLOSED": closed against uniform + smooth
content-dependent inflating scales; the sole evasion is a measure-zero knife-edge that is
degenerate with real biology (no method could separate it without an external anchor).

Target: ``nudge.inference.differential`` (``NUDGE-METHOD-010`` / ``NUDGE-LIM-016``).
Gate 4c (P4) keys on the OFF-cluster SPREAD of the LOW-activity (below-median) cells and
only runs when the BIC winner is ``vmax``. Two constructions probed:

  * **on_up** — scale only the above-median cells UP by c. ON mode moves (→ ``vmax`` BIC
    winner), OFF cluster untouched → ``off_scale`` ≈ 1, gate 4c passes.
  * **on_down** — scale the above-median cells DOWN. A uniform ON-mode reduction is a clean
    ``v_max`` reduction, so this too reads as ``ceiling-diff``, not gain.

Ground truth: ``simulate_context_pair(mechanism="none")`` — asserted NO mechanistic
difference (an unobservable technical origin; see verdict above).

Run: uv run python scripts/redteam/differential_subset_scale_confound.py [nseeds]
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
# (label, factor) applied to the ON-mode (above population-median row-sum) cells only.
ATTACKS = [
    ("on_up", 2.2),
    ("on_up", 3.0),
    ("on_down", 0.45),
    ("on_down", 0.6),
]


def _scale_on_mode(data: np.ndarray, factor: float) -> np.ndarray:
    """Multiply only the ON-mode cells (row-sum above the population median) by ``factor``.

    Leaves the OFF cluster (below-median cells) untouched, so the OFF-cluster spread that
    gate 4c keys on is unperturbed (``off_scale`` ≈ 1).
    """
    out = np.asarray(data, dtype=float).copy()
    s = out.sum(axis=1)
    on = s > np.median(s)
    out[on] *= factor
    return out


def _run_one(seed: int, label: str, factor: float) -> tuple[str, object]:
    circuit = ras_switch_1node()
    ctx_a, ctx_b = simulate_context_pair(
        circuit,
        mechanism="none",  # asserted: NO mechanistic difference (unobservable technical origin)
        n_cells=N_CELLS,
        scale_a=SCALE,
        scale_b=SCALE,
        obs_sd=0.5,
        seed=seed,
    )
    b_data = _scale_on_mode(ctx_b.data, factor)
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
    print("RED-TEAM P4-rescan: SUBSET (ON-mode knife-edge) multiplicative scale")
    print("NOTE: positives here are DEGENERATE with a genuine ceiling change (see module")
    print("docstring) — NOT a valid confident-wrong. Realistic siblings are all caught.")
    print("=" * 80, flush=True)
    positives = 0
    for seed in range(nseeds):
        for label, factor in ATTACKS:
            call, res = _run_one(seed, label, factor)
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
            positive = call in {"threshold-diff", "gain-diff", "ceiling-diff"}
            positives += int(positive)
            tag = "  <== POSITIVE (degenerate w/ genuine ceiling)" if positive else ""
            va = fit.est_a[fit.best_diff][fit.best_diff]
            vb = fit.est_b[fit.best_diff][fit.best_diff]
            print(
                f"\nseed={seed}  [{label} x{factor}]  call={call!r}{tag}\n"
                f"    off_shift={off_infl:.3f} (4b abstains >2.5)  "
                f"off_scale={off_scale:.3f} (4c band [0.80,1.30])\n"
                f"    depth_ratio={fit.depth_ratio:.3f}  best_diff={fit.best_diff!r} "
                f"({va:.3g} -> {vb:.3g})\n"
                f"    dBIC vs shared={d_shared:.1f}  dBIC vs runner "
                f"({runner})={d_runner:.1f}  (margins 6.0/6.0)",
                flush=True,
            )
    print("\n" + "=" * 80, flush=True)
    print(
        f"in-band positives (NOT valid confident-wrong; degenerate with real "
        f"ceiling): {positives}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    raise SystemExit(run(n))
