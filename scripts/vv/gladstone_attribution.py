#!/usr/bin/env python3
"""Run NUDGE covariance attribution on the Gladstone Ras-switch perturbations.

Each ``.h5ad`` passed is one **operating point** (a donor × stimulation condition). For
each Ras-switch target the pipeline: backed-loads only the target + control cells and the
Ras panel, maps the IEG readout to activation-space, selects the circuit topology by
parsimony, and attributes the mechanism — single-condition per operating point (expected to
**abstain** between gain and threshold) and, with ≥2 operating points, the joint **breaker**.
It abstains loudly on low-count / near-bifurcation states and prints that honestly.

Usage (once a donor file has been downloaded):
    uv run python scripts/vv/gladstone_attribution.py \\
        "/media/nick/Seagate Hub/gladstone/D1_Stim8hr.assigned_guide.h5ad"
    # the breaker needs ≥2 stim conditions:
    uv run python scripts/vv/gladstone_attribution.py D1_Rest....h5ad D1_Stim8hr....h5ad
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from nudge.circuits import ras_switch_1node
from nudge.data.loaders.tier2 import IEG_READOUT, load_gladstone
from nudge.inference.bridge import counts_to_activity
from nudge.inference.model_select import Candidate, select_topology
from nudge.inference.pipeline import attribute_across_operating_points

# expected biology: SOS1 KD → loss of feedback gain; RASGRP1 → input/threshold;
# RASA2 (a GAP) → lower OFF pressure → threshold/ceiling.
TARGETS = ("SOS1", "RASGRP1", "RASA2")
EXPECTED = {"SOS1": "gain", "RASGRP1": "threshold", "RASA2": "ceiling"}
MARKERS = {"Activation": IEG_READOUT}
_KIN1 = [("edge", 0, "n"), ("edge", 0, "K"), ("edge", 0, "vmax")]


def _op_label(path: str) -> str:
    return os.path.basename(path).split(".")[0]


def run(files: list[str], *, targets: tuple[str, ...] = TARGETS, steps: int = 250) -> None:
    ops: dict[str, Any] = {}
    for path in files:
        label = _op_label(path)
        print(f"[load] {label} ← {path}", flush=True)
        ops[label] = load_gladstone(path, target_genes=(*targets,))

    # --- topology: is it a switch at all, and which one? (1-D IEG readout → 1-node) ---
    first = next(iter(ops.values()))
    wt_act = counts_to_activity(
        first[first.obs["condition"] == "WT"], ras_switch_1node(), MARKERS
    )
    topo = select_topology(
        wt_act, [Candidate("1-node", ras_switch_1node(), _KIN1)], steps=steps
    )
    print(f"\n=== topology (from WT activation) ===\n  selected: {topo.selected}"
          f"  |  BIC: {{" + ", ".join(f'{k}={v:.0f}' for k, v in topo.bic.items()) + "}")
    if not topo.is_switch:
        print("  → no switch detected in the activation readout; attribution abstains.")
        return

    # --- attribute each target across the operating points ---
    for target in targets:
        rep = attribute_across_operating_points(
            ops, ras_switch_1node(), MARKERS, target, steps=steps
        )
        print(f"\n=== {target}  (expected: {EXPECTED.get(target, '?')}) ===")
        for label in ops:
            if label in rep.single:
                call, nlls = rep.single[label]
                prof = " ".join(f"{k}={v:.3f}" for k, v in nlls.items())
                print(f"  {label:10s} n={rep.n_cells[label]:6d}  single → {call}   [{prof}]")
            else:
                print(f"  {label:10s} n={rep.n_cells[label]:6d}  SKIPPED: "
                      f"{rep.skipped.get(label, '?')}")
        if rep.multi is not None:
            call, nlls = rep.multi
            prof = " ".join(f"{k}={v:.3f}" for k, v in nlls.items())
            print(f"  → BREAKER (joint over {len(rep.single)} ops): {call}   [{prof}]")
        else:
            print("  → breaker needs ≥2 usable operating points (add stim-condition files).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", help="one .h5ad per operating point")
    ap.add_argument("--steps", type=int, default=250)
    args = ap.parse_args()
    run(args.files, steps=args.steps)


if __name__ == "__main__":
    main()
