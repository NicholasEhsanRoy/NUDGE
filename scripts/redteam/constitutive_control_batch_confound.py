"""RED-TEAM (round 2) repro: a control-vs-population CAPTURE-EFFICIENCY confound makes
the constitutive control assert ``biological-switch`` on a TRULY LINEAR circuit.

Capability under attack: ``nudge.inference.constitutive`` (``NUDGE-METHOD-011``), the
shipped ``NUDGE-LIM-006`` mitigation. Its verdict is claimed structurally fail-safe:
``ConstitutiveResult.is_confident_wrong`` is ``True`` ONLY for a bare
threshold/gain/ceiling, so a ``biological-switch`` verdict is treated as never-wrong.

The blind spot: ``biological-switch`` ("the ultrasensitivity is a real circuit switch")
IS a positive, falsifiable claim, and ``is_confident_wrong`` does NOT count it. If the
control mis-anchors the readout, NUDGE can assert ``biological-switch`` on a circuit whose
true Hill ``n = 1`` (a purely readout-driven, LIM-006 artifact) — a confident-wrong the
module's own guard cannot see.

The attack (realistic, un-gated). The constitutive control is a SEPARATE population
(constitutively-driven reporter cells). Single-cell samples routinely differ in capture /
sequencing efficiency, so the control's raw reporter counts sit on a different multiplicative
scale than the circuit population's counts. The module compares them directly (same reporter,
same scale assumed) with NO control-to-population depth normalization. Reading the control at
~0.5x the population's efficiency mis-anchors the reporter ``Vmax`` low; the profile then
develops a spurious well away from ``n = 1`` and REJECTS "no switch" — a confident
``biological-switch`` on a linear circuit.

Truth in every LINEAR case: circuit ``n = 1`` -> honest verdict is ``unresolved`` (the
shipped validation result with a CLEAN control; ``tests/inference/test_constitutive.py``
``test_linear_circuit_lim006_hazard_abstains_not_confident_wrong``). A ``biological-switch``
here is a confident-wrong.

Run: uv run python scripts/redteam/constitutive_control_batch_confound.py [nseeds]
Touches no src/ code and no fail-safe margins — diagnostic only. Shipped default budget
(restarts=3, steps=600, n_model_cells=400); ~a few min per case.
"""

from __future__ import annotations

import sys

from nudge.inference.constitutive import (
    ConstitutiveControl,
    ReadoutCircuitParams,
    generate_constitutive_dataset,
    profile_circuit_n,
)

# The module's default n-grid + a linear (n=1) ground truth (the LIM-006 hazard).
GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0)
LINEAR = ReadoutCircuitParams(
    k=1.0, n=1.0, vmax=1.0, basal=0.05, km=0.5, h=6.0, readout_vmax=20.0, readout_base=0.1
)


def _analyze(ctrl_scale: float, seed: int) -> tuple[str, float, float, float]:
    """Generate a LINEAR-circuit population + control, rescale the control's capture
    efficiency by ``ctrl_scale``, and run the SHIPPED profile at the default budget."""
    pop, control, _ = generate_constitutive_dataset(
        LINEAR, n_cells=600, n_ctrl_doses=10, n_ctrl_reps=200, dispersion=0.1, seed=seed
    )
    if ctrl_scale != 1.0:
        control = ConstitutiveControl(
            activity=control.activity, response=control.response * ctrl_scale
        )
    res = profile_circuit_n(
        pop, control, LINEAR, n_grid=GRID, restarts=3, steps=600, n_model_cells=400, seed=seed
    )
    return res.call, res.n1_rejection, res.span_no_control, res.argmin_n_with_control


def run(nseeds: int) -> int:
    print("=" * 80, flush=True)
    print("RED-TEAM r2: constitutive control capture-efficiency confound", flush=True)
    print("TRUTH = LINEAR circuit (n=1); honest verdict = 'unresolved'. "
          "'biological-switch' = confident-wrong (is_confident_wrong is BLIND to it).",
          flush=True)
    print("=" * 80, flush=True)
    holes = 0
    for seed in range(nseeds):
        # Positive control: a CLEAN control (scale=1.0) must abstain (unresolved).
        call0, rej0, span0, amin0 = _analyze(1.0, seed)
        print(f"\nseed={seed}  [clean control scale=1.0]  call={call0!r}  "
              f"n1_rej={rej0:.4f}  5xspan={5 * span0:.4f}  argmin_n={amin0}", flush=True)
        # Attack: control under-read at 0.5x efficiency.
        call1, rej1, span1, amin1 = _analyze(0.5, seed)
        is_hole = call1 == "biological-switch"
        holes += int(is_hole)
        flag = "  <== CONFIDENT-WRONG (biological-switch on a LINEAR circuit)" if is_hole else ""
        print(f"seed={seed}  [confounded control scale=0.5] call={call1!r}  "
              f"n1_rej={rej1:.4f}  5xspan={5 * span1:.4f}  argmin_n={amin1}{flag}", flush=True)
    print(f"\n>>> confident-wrong 'biological-switch' holes: {holes}/{nseeds}", flush=True)
    return holes


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run(ns)
