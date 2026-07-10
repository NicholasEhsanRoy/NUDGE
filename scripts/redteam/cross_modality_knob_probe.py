"""RED-TEAM: can cross_modality.classify_knob_shift emit a SPECIFIC WRONG knob?

The knob classifier localizes a variant's dose-response to threshold / gain / ceiling vs
a control, or abstains (inconclusive / non-responsive). Two attacks on NEW ground (the
existing NUDGE-LIM-008 is about the modality BOUNCER, not the knob localizer):

  ATTACK 1 — gate ordering / span-shrink. classify_knob_shift checks, IN ORDER:
    non-responsive → threshold(right) → CEILING(floor rise OR span shrunk) → threshold
    (left) → gain. The CEILING gate has NO disjoint-CI requirement — it fires on
    ``span_shrunk = amp < 0.6·control.amp`` alone. A TRUE GAIN-DOWN mutant (shallower Hill,
    n↓) that saturates more slowly can have its response span UNDER-estimated by the fit
    over a fixed dose grid → span_shrunk → a CONFIDENT ``ceiling`` call on a gain truth.

  ATTACK 2 — the knob localizer does NOT gate on the dose-response reliability verdict.
    attribute_variant_panel runs classify_knob_shift on the fit REGARDLESS of whether the
    dose_response call itself was ``unresolved`` (e.g. doses don't span the inflection —
    NUDGE-LIM-007). So a curve the shipped dose-response path correctly ABSTAINS on can
    still be handed a confident knob.

For each attack: generate synthetic activation dose-response curves (the affine, in-model
regime — NO readout nonlinearity, so NUDGE-LIM-006 is NOT the cause), fit with the shipped
fit_dose_response, run classify_knob_shift, and report the knob vs the generative TRUTH.

Run: uv run python scripts/redteam/cross_modality_knob_probe.py
Touches no src/ code and no fail-safe margins.
"""

from __future__ import annotations

import numpy as np

from nudge.inference.cross_modality import classify_knob_shift
from nudge.inference.dose_response import (
    attribute_dose_response,
    classify_dose_response,
    fit_dose_response,
)


def hill(dose: np.ndarray, floor: float, amp: float, k: float, n: float) -> np.ndarray:
    d = np.maximum(dose, 0.0)
    return floor + amp * d**n / (k**n + d**n)


def make_curve(
    doses: np.ndarray,
    *,
    floor: float,
    amp: float,
    k: float,
    n: float,
    noise: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    clean = hill(doses, floor, amp, k, n)
    return clean + rng.normal(0.0, noise, size=doses.shape)


def _fit(doses: np.ndarray, resp: np.ndarray, seed: int) -> object:
    return fit_dose_response(doses, resp, direction="activate", n_boot=300, seed=seed)


def attack_1_gain_mislabeled_ceiling() -> int:
    print("=" * 78)
    print("ATTACK 1: a TRUE GAIN-DOWN mutant mis-called via the span-shrink ceiling gate")
    print("=" * 78)
    # A dose grid that spans the control inflection but truncates a shallow (n↓) curve's
    # slow approach to plateau, so the shallow variant's fitted amp is UNDER-estimated.
    doses = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0], dtype=float)
    holes = 0
    for seed in range(6):
        ctrl_y = make_curve(
            doses, floor=0.05, amp=1.0, k=1.0, n=4.0, noise=0.03, seed=seed
        )
        # TRUTH = gain: only n changes (4 → 1.2). floor, amp, K held EXACTLY equal.
        var_y = make_curve(
            doses, floor=0.05, amp=1.0, k=1.0, n=1.2, noise=0.03, seed=seed + 100
        )
        ctrl = _fit(doses, ctrl_y, seed)
        var = _fit(doses, var_y, seed)
        knob, reason = classify_knob_shift(var, ctrl)
        wrong = knob in ("threshold", "ceiling")  # truth is gain
        holes += int(wrong)
        print(
            f"  seed={seed}: TRUTH=gain -> knob={knob!r}  "
            f"(ctrl amp={ctrl.amp:.3f} n={ctrl.n:.2f}; var amp={var.amp:.3f} "
            f"n={var.n:.2f})" + ("   <== WRONG KNOB" if wrong else "")
        )
        if wrong:
            print(f"      reason: {reason}")
    return holes


def attack_2_unresolved_curve_gets_knob() -> int:
    print("=" * 78)
    print("ATTACK 2: a dose-response that ABSTAINS (unresolved) still gets a confident knob")
    print("=" * 78)
    # Doses that do NOT span the variant inflection (K pushed past the last dose) — the
    # shipped dose-response path returns `unresolved` (NUDGE-LIM-007). The knob localizer
    # is run anyway.
    doses = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
    holes = 0
    for seed in range(6):
        ctrl_y = make_curve(
            doses, floor=0.05, amp=1.0, k=0.2, n=4.0, noise=0.02, seed=seed
        )
        # A variant whose K sits well past the last dose (0.5): only ONE arm is seen.
        var_y = make_curve(
            doses, floor=0.05, amp=1.0, k=2.0, n=4.0, noise=0.02, seed=seed + 100
        )
        ctrl_res = attribute_dose_response(doses, ctrl_y, direction="activate",
                                           n_boot=300, seed=seed)
        var_fit = _fit(doses, var_y, seed)
        var_call, _r = classify_dose_response(var_fit)
        knob, reason = classify_knob_shift(var_fit, ctrl_res.fit)
        # A confident knob emitted on a curve the dose-response path abstains on.
        leaked = var_call == "unresolved" and knob in ("threshold", "gain", "ceiling")
        holes += int(leaked)
        print(
            f"  seed={seed}: dose_response call={var_call!r} (spans_inflection="
            f"{var_fit.spans_inflection}) -> knob={knob!r}"
            + ("   <== KNOB LEAKED PAST ABSTENTION" if leaked else "")
        )
        if leaked:
            print(f"      knob reason: {reason}")
    return holes


if __name__ == "__main__":
    h1 = attack_1_gain_mislabeled_ceiling()
    h2 = attack_2_unresolved_curve_gets_knob()
    print("\n" + "=" * 78)
    print(f">>> ATTACK 1 wrong-knob count: {h1}")
    print(f">>> ATTACK 2 knob-leak count:  {h2}")
