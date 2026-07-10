"""RED-TEAM (fast, numpy/scipy only): epistasis + dose-response fail-safe probes.

ATTACK E1 — batch shift on A+B faking synergy at the classifier entry point.
  attribute_synergy(control, a, b, ab) takes per-cell SCALAR scores. Its depth/batch
  defense (size-factor normalization) lives UPSTREAM in combo_effect_scores (the bridge);
  the classifier itself has none. Truth = ADDITIVE (interaction 0), but the A+B condition
  carries a constant additive batch offset that survives any per-cell centering. Does the
  classifier call ``synergistic``? (Formalizes the "perfectly-aligned confound" soft spot
  of NUDGE-LIM-009 at the public attribute_synergy entry.)

ATTACK E2 — the same, but the offset tracks a real (large) single-arm effect so the
  additive scale is big and the CI is tight: the hardest case for the CI-width gate.

ATTACK D1 — a graded (n≈1) dose-response with an adversarial noise draw / a steep-looking
  outlier: can it earn a confident ``switch`` (both the BIC margin AND the whole n-CI > 2)?

Run: uv run python scripts/redteam/epistasis_dose_probe.py
Touches no src/ code and no fail-safe margins.
"""

from __future__ import annotations

import numpy as np

from nudge.inference.dose_response import attribute_dose_response
from nudge.inference.epistasis import attribute_synergy


def epistasis_batch_confound() -> int:
    print("=" * 78)
    print("ATTACK E1/E2: an additive batch offset on A+B, truth = ADDITIVE")
    print("=" * 78)
    holes = 0
    rng = np.random.default_rng(0)
    for label, base_shift, batch in (
        ("E1 weak arms", 0.3, 0.6),
        ("E2 strong arms", 1.5, 0.9),
    ):
        for seed in range(4):
            r = np.random.default_rng(seed)
            n = 800
            control = r.normal(0.0, 1.0, n)
            # Truth ADDITIVE: a = ctrl + s, b = ctrl + s, ab = ctrl + 2s (Bliss-additive).
            a = r.normal(base_shift, 1.0, n)
            b = r.normal(base_shift, 1.0, n)
            ab = r.normal(2 * base_shift, 1.0, n)
            # A batch/technical offset that hits ONLY the A+B condition (perfectly aligned).
            ab = ab + batch
            res = attribute_synergy(control, a, b, ab, n_boot=400, seed=seed)
            wrong = res.call in ("synergistic", "buffering")
            holes += int(wrong)
            print(
                f"  {label} seed={seed}: call={res.call!r} "
                f"interaction={res.fit.interaction:+.3f} "
                f"CI={res.fit.ci_interaction}" + ("  <== CONFIDENT-WRONG" if wrong else "")
            )
    _ = rng
    return holes


def dose_graded_faking_switch() -> int:
    print("=" * 78)
    print("ATTACK D1: a graded (n≈1) curve — can noise earn a confident SWITCH?")
    print("=" * 78)
    doses = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0], dtype=float)
    holes = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        # A genuinely GRADED response (n=1 Michaelis-Menten), K spanned.
        clean = 0.05 + 1.0 * doses / (1.0 + doses)  # n=1
        y = clean + rng.normal(0.0, 0.04, size=doses.shape)
        # Adversarial: nudge a low-dose point down + a mid point up to fake a knee.
        y[1] -= 0.06
        y[3] += 0.06
        res = attribute_dose_response(doses, y, direction="activate", n_boot=400,
                                      seed=seed)
        wrong = res.call == "switch"
        holes += int(wrong)
        print(
            f"  seed={seed}: TRUTH=graded -> call={res.call!r} n={res.fit.n:.2f} "
            f"ci_n={res.fit.ci_n}" + ("  <== CONFIDENT-WRONG SWITCH" if wrong else "")
        )
    return holes


if __name__ == "__main__":
    he = epistasis_batch_confound()
    hd = dose_graded_faking_switch()
    print("\n" + "=" * 78)
    print(f">>> epistasis batch-confound confident-wrong: {he}")
    print(f">>> dose-response graded->switch confident-wrong: {hd}")
