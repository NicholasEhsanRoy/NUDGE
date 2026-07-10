"""RED-TEAM (round 3): a per-CONDITION batch/depth scale on the perturbed panel fakes a
confident ``ceiling`` past the multi-reporter consistency guard.

Target: ``nudge.inference.multi_reporter`` (``NUDGE-METHOD-008`` / ``NUDGE-LIM-014``).
The round-1/round-2 analogue flagged and never tested. The consistency guard
(``classify_multi_reporter`` gate 1) checks whether the reporters share ONE latent —
but it computes ``panel_r2`` / ``consistency_ratio`` / the per-reporter R² from the
**CONTROL** curves only. Any confound applied to the **perturbed** condition is
structurally invisible to it. And multi_reporter pins each reporter's affine
``(floor, gain)`` from the control and then fits the perturbed panel WITHOUT any
per-condition depth/batch normalization (unlike ``differential``, which pins depth per
context from each context's control).

The attack: a single multiplicative factor ``c`` on EVERY perturbed reporter (a batch /
sequencing-depth / instrument-gain difference between the control-condition measurement and
the perturbed-condition measurement — consistent across the panel). With small reporter
floors, ``c·(floor + gain·f) ≈ floor + gain·(c·f)`` — i.e. it is indistinguishable from a
shared latent-ceiling change ``A = c``. Every reporter's ON amplitude scales by the SAME
fraction, which is the *exact* signature multi_reporter attributes to ``ceiling``.

Ground truth: ``simulate_reporter_panel(mechanism="none")`` — the perturbation did NOT move
the latent (``no-effect``). Honest answer: ``no-effect`` / ``unresolved``. A confident
``ceiling`` (with a CI that excludes 0) is the hole.

Run: uv run python scripts/redteam/multi_reporter_batch_confound.py [nseeds]
Touches no src/ code and no fail-safe margins — diagnostic only.
"""

from __future__ import annotations

import sys

import numpy as np

from nudge.inference.multi_reporter import (
    ReporterObservation,
    attribute_multi_reporter,
    simulate_reporter_panel,
)

# Multiplicative per-condition batch factors on the perturbed panel (c<1 = depth drop).
FACTORS = [0.5, 0.6, 0.75]


def _attack_panel(seed: int, factor: float) -> list[ReporterObservation]:
    # Truth: no-effect. Tiny floors so a multiplicative batch on the perturbed condition
    # is cleanly aliased to a latent-ceiling change (A = factor).
    panel = simulate_reporter_panel(
        mechanism="none",
        n_reporters=5,
        k_wt=20.0,
        n_wt=4.0,
        gain_range=(0.5, 3.0),
        floor_range=(0.0, 0.02),
        noise=0.02,
        seed=seed,
    )
    out: list[ReporterObservation] = []
    for o in panel:
        pert = np.asarray(o.perturbed, dtype=float) * factor  # per-condition batch scale
        out.append(
            ReporterObservation(
                name=o.name, dose=o.dose, control=o.control, perturbed=pert
            )
        )
    return out


def run(nseeds: int) -> int:
    print("=" * 80, flush=True)
    print("RED-TEAM r3: per-condition batch scale on the PERTURBED panel (control clean)")
    print("truth = no-effect; a confident 'ceiling' is the HOLE (LIM-014 guard bypass)")
    print("=" * 80, flush=True)
    holes = 0
    for seed in range(nseeds):
        # positive control: no batch -> must be no-effect / unresolved.
        clean = simulate_reporter_panel(
            mechanism="none", n_reporters=5, k_wt=20.0, n_wt=4.0,
            gain_range=(0.5, 3.0), floor_range=(0.0, 0.02), noise=0.02, seed=seed,
        )
        res0 = attribute_multi_reporter(clean, n_boot=200, seed=seed)
        print(
            f"\nseed={seed}  [factor=1.00 control]  call={res0.call!r}  "
            f"panel_r2={res0.fit.panel_r2:.3f}  consistency={res0.fit.consistency_ratio:.2f}",
            flush=True,
        )
        for c in FACTORS:
            panel = _attack_panel(seed, c)
            res = attribute_multi_reporter(panel, n_boot=200, seed=seed)
            fit = res.fit
            hole = res.call in {"threshold", "gain", "ceiling"}
            holes += int(hole)
            tag = f"  <== HOLE ({res.call})" if hole else ""
            print(
                f"seed={seed}  [factor={c:.2f}]  call={res.call!r}{tag}\n"
                f"    winner={fit.winner!r}  knob_margin={fit.knob_margin:.2f} (>1.5)  "
                f"effect_margin={fit.effect_margin:.2f} (>1.4)\n"
                f"    ceiling_ratio={fit.ceiling_ratio:.3f}  "
                f"ci_log2_ceiling=({fit.ci_log2_ceiling[0]:+.2f}, "
                f"{fit.ci_log2_ceiling[1]:+.2f})  "
                f"panel_r2(control)={fit.panel_r2:.3f}  "
                f"consistency={fit.consistency_ratio:.2f}",
                flush=True,
            )
            if hole:
                print(f"    reason: {res.reason}", flush=True)
    print("\n" + "=" * 80, flush=True)
    print(f"confident-wrong mechanism calls (HOLES): {holes}", flush=True)
    return 0 if holes == 0 else 2


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    raise SystemExit(run(n))
