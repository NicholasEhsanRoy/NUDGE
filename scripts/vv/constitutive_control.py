"""NUDGE-LIM-006 mitigation — the constitutive-control degeneracy break (shipped feature).

Reproduces the VALIDATED headline (FINDINGS "NUDGE-LIM-006 mitigation — VALIDATED") as a
shipped-module run: profile the loss over the circuit Hill ``n`` WITHOUT vs WITH a
constitutive-reporter control, on synthetic ground truth.

Two ground truths, one nonlinear (h=6) reporter:

- **Case A — a TRUE biological switch (circuit n=3).** WITHOUT the control the n-profile is
  FLAT (the split is degenerate — you cannot tell a switch exists). WITH the control, "no
  switch" (n=1) is REJECTED (Δloss ≫ the flat span) → NUDGE calls it BIOLOGICAL.
- **Case B — a LINEAR circuit (n=1), the LIM-006 false-positive hazard.** A linear circuit
  read through the nonlinear reporter is what fools the affine-readout fit into a confident
  switch. WITH the control the profile does NOT reject n=1 → NUDGE ABSTAINS (unresolved).
  The confident false positive becomes an honest abstention.

Fail-safe assertion: **0 confident-wrong** — the module never emits a bare threshold / gain /
ceiling. It reports a BIOLOGICAL-switch verdict (reject the readout-only explanation) or
abstains; it never point-identifies the exact knob (the honest caveat).

Run: ``uv run python scripts/vv/constitutive_control.py``
"""

from __future__ import annotations

import time

from nudge.inference.constitutive import (
    ReadoutCircuitParams,
    generate_constitutive_dataset,
    profile_circuit_n,
)

N_GRID = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)
SEEDS = (0, 1, 2)


def run_case(tag: str, n_true: float, seed: int) -> dict[str, object]:
    params = ReadoutCircuitParams(
        k=1.0, n=n_true, vmax=1.0, basal=0.05, km=0.5, h=6.0, readout_vmax=20.0, readout_base=0.1
    )
    pop, control, _ = generate_constitutive_dataset(
        params, n_cells=600, n_ctrl_doses=10, n_ctrl_reps=200, seed=seed
    )
    res = profile_circuit_n(
        pop, control, params, n_grid=N_GRID, restarts=3, steps=600, n_model_cells=400, seed=seed
    )
    print(f"\n[{tag}] seed={seed}  (true circuit n={n_true:g}, reporter h=6)")
    print(f"  calibrated reporter h = {res.calibration.h:.2f} "
          f"(95% CI {res.calibration.ci_h[0]:.2f}-{res.calibration.ci_h[1]:.2f}, "
          f"nonlinear={res.calibration.is_nonlinear})")
    print(f"  WITHOUT control: n-profile span = {res.span_no_control:.5f}   (flat => degenerate)")
    print(f"  WITH    control: n=1 rejection  = {res.n1_rejection:.5f}   "
          f"(argmin n≈{res.argmin_n_with_control:g})")
    print(f"  n-grid                : {list(res.n_grid)}")
    print(f"  loss (no control)     : {[round(x, 4) for x in res.loss_no_control]}")
    print(f"  loss (with control)   : {[round(x, 4) for x in res.loss_with_control]}")
    print(f"  VERDICT: {res.call}   (confident-wrong={res.is_confident_wrong})")
    print(f"  reason: {res.reason[:160]}...")
    return {
        "tag": tag,
        "seed": seed,
        "span_no_control": res.span_no_control,
        "n1_rejection": res.n1_rejection,
        "call": res.call,
        "confident_wrong": res.is_confident_wrong,
    }


def main() -> int:
    t0 = time.time()
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        rows.append(run_case("A true-switch n=3", 3.0, seed))
        rows.append(run_case("B linear n=1 (LIM-006 hazard)", 1.0, seed))

    print("\n" + "=" * 78)
    print("SUMMARY")
    n_cw = sum(1 for r in rows if r["confident_wrong"])
    a_bio = sum(1 for r in rows if r["tag"].startswith("A") and r["call"] == "biological-switch")
    b_abst = sum(1 for r in rows if r["tag"].startswith("B") and r["call"] == "unresolved")
    print(f"  confident-wrong calls               : {n_cw}   (MUST be 0)")
    print(f"  case A → 'biological-switch'         : {a_bio}/{len(SEEDS)}")
    print(f"  case B → 'unresolved' (honest abstain): {b_abst}/{len(SEEDS)}")
    print(f"  elapsed: {time.time() - t0:.1f}s")

    assert n_cw == 0, "FAIL-SAFE VIOLATED: a confident-wrong call was emitted"
    print("\nPASS: the constitutive control breaks the LIM-006 degeneracy, 0 confident-wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
