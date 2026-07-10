"""RED-TEAM (round 3): a SECOND route to a confident ``biological-switch`` on a LINEAR
circuit — a misspecified count DISPERSION (not the round-2 capture-efficiency mismatch).

Target: ``nudge.inference.constitutive`` (``NUDGE-METHOD-011`` / ``NUDGE-LIM-019``). Round 2
locked ONE route (control read at a different capture efficiency). This probes a DIFFERENT,
also-realistic, also-un-gated route: the module takes the count-model ``dispersion`` as a
KNOWN input (``profile_circuit_n(dispersion=0.1)`` default). Overdispersion is rarely known
exactly. If the true population is more overdispersed than the assumed value, the extra
count spread on a LINEAR (n=1) circuit could be read as latent structure the profile
attributes to a circuit switch → ``biological-switch``. Because ``is_confident_wrong`` is
structurally BLIND to ``biological-switch`` (round-2 finding), such an error is invisible to
the module's own guard.

Truth: LINEAR circuit (n=1) → honest verdict ``unresolved``. A ``biological-switch`` is a
confident-wrong.

Run: uv run python scripts/redteam/constitutive_dispersion_route.py [nseeds]
Touches no src/ code and no fail-safe margins — diagnostic only. Full shipped budget.
"""

from __future__ import annotations

import sys

from nudge.inference.constitutive import (
    ReadoutCircuitParams,
    generate_constitutive_dataset,
    profile_circuit_n,
)

GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0)
LINEAR = ReadoutCircuitParams(
    k=1.0, n=1.0, vmax=1.0, basal=0.05, km=0.5, h=6.0, readout_vmax=20.0, readout_base=0.1
)
ASSUMED_DISP = 0.1  # what the analyst tells profile_circuit_n
TRUE_DISPS = [0.1, 0.4, 0.8]  # 0.1 = matched (positive control); >0.1 = misspecified


def _analyze(true_disp: float, seed: int) -> tuple[str, float, float, float]:
    pop, control, _ = generate_constitutive_dataset(
        LINEAR, n_cells=600, n_ctrl_doses=10, n_ctrl_reps=200,
        dispersion=true_disp, seed=seed,
    )
    res = profile_circuit_n(
        pop, control, LINEAR, n_grid=GRID, dispersion=ASSUMED_DISP,
        restarts=3, steps=600, n_model_cells=400, seed=seed,
    )
    return res.call, res.n1_rejection, res.span_no_control, res.argmin_n_with_control


def run(nseeds: int) -> int:
    print("=" * 80, flush=True)
    print("RED-TEAM r3: constitutive alt-route — misspecified DISPERSION (assumed=0.1)")
    print("TRUTH = LINEAR n=1; honest = 'unresolved'. 'biological-switch' = confident-wrong")
    print("=" * 80, flush=True)
    holes = 0
    for seed in range(nseeds):
        for td in TRUE_DISPS:
            call, rej, span, amin = _analyze(td, seed)
            hole = call == "biological-switch"
            holes += int(hole)
            tag = "  <== CONFIDENT-WRONG (biological-switch on LINEAR)" if hole else ""
            role = " [matched/positive-control]" if td == ASSUMED_DISP else ""
            print(
                f"seed={seed}  true_disp={td:.1f}{role}  call={call!r}{tag}\n"
                f"    n1_rej={rej:.4f}  5xspan={5 * span:.4f}  argmin_n={amin}",
                flush=True,
            )
    print(f"\n>>> confident-wrong 'biological-switch' holes: {holes}", flush=True)
    return holes


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    raise SystemExit(0 if run(ns) == 0 else 2)
