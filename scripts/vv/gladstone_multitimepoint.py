#!/usr/bin/env python3
"""Multi-timepoint capstone: do the T-cell stimulation timepoints, used as multiple
operating points, resolve the gain/threshold degeneracy a single snapshot abstained on?

At 8 h the Gladstone activation readout is a single (graded, heavy-tailed) mode and NUDGE
correctly returned ``no-switch`` (FINDINGS "Phase 4"). This runner asks the two follow-up
questions with later timepoints (Rest / Stim8hr / Stim48hr) as **operating points**:

  Q1. Does a later timepoint push the activation readout into a genuine bimodal switch
      that **survives the BIC parsimony gate**? (topology selected *per timepoint*.)
  Q2. Using the timepoints as multiple ``OperatingPoint`` s, does the joint **breaker**
      resolve gain vs threshold where each single point abstains? (the measured
      degeneracy-breaker — a second operating point.)

Honest by construction: topology is reported per timepoint (never short-circuited), and the
breaker only fires on operating points with enough target cells + a reliable LNA — the rest
are reported as SKIPPED, not hidden. A genome-*wide* screen is cell-count-limited per guide
(~24-233 cells vs the ~1000/condition the FIM analysis showed is needed), so underpowered
skips and honest abstention are expected, legitimate outcomes.

Usage (once the files are downloaded):
    uv run python scripts/vv/gladstone_multitimepoint.py \\
        "/media/nick/Seagate Hub/gladstone/D1_Stim48hr.assigned_guide.h5ad"          # Q1 only
    uv run python scripts/vv/gladstone_multitimepoint.py \\
        ".../D1_Stim8hr...h5ad" ".../D1_Stim48hr...h5ad"                             # Q1 + Q2
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
from scipy import stats

from nudge.circuits import ras_switch_1node
from nudge.data.loaders.tier2 import IEG_READOUT, load_gladstone
from nudge.inference.bridge import counts_to_activity
from nudge.inference.model_select import Candidate, select_topology
from nudge.inference.pipeline import attribute_across_operating_points

TARGETS = ("SOS1", "RASGRP1", "RASA2")
EXPECTED = {"SOS1": "gain", "RASGRP1": "threshold", "RASA2": "ceiling"}
MARKERS = {"Activation": IEG_READOUT}
_KIN1 = [("edge", 0, "n"), ("edge", 0, "K"), ("edge", 0, "vmax")]


def _label(path: str) -> str:
    return os.path.basename(path).split(".")[0]


def _distribution_note(col: np.ndarray) -> str:
    """A one-line honest summary of the WT activation shape (mode structure)."""
    lo = float(col.min())
    span = float(np.median(col) - lo) or 1.0
    frac_low = float((col <= lo + 0.25 * span).mean())
    return (
        f"skew={stats.skew(col):.1f} kurt={stats.kurtosis(col):.1f} "
        f"frac_near_low={frac_low:.2f}"
    )


def run(
    files: list[str], *, targets: tuple[str, ...] = TARGETS,
    steps: int = 250, min_cells: int = 200,
) -> None:
    ops: dict[str, Any] = {}
    for path in files:
        lab = _label(path)
        print(f"[load] {lab} ← {path}", flush=True)
        ops[lab] = load_gladstone(path, target_genes=(*targets,))

    # --- Q1: topology PER timepoint — does the switch emerge later? ---
    print("\n=== Q1. topology per timepoint (BIC: lower is better) ===", flush=True)
    for lab, adata in ops.items():
        wt = counts_to_activity(
            adata[adata.obs["condition"] == "WT"], ras_switch_1node(), MARKERS
        )
        topo = select_topology(
            wt, [Candidate("1-node", ras_switch_1node(), _KIN1)], steps=steps
        )
        bic = ", ".join(f"{k}={v:.0f}" for k, v in topo.bic.items())
        verdict = "SWITCH ✓" if topo.is_switch else "no-switch (abstain)"
        print(f"  {lab:14s} {verdict:20s} BIC[{bic}]  "
              f"WT n={len(wt):5d}  {_distribution_note(wt[:, 0])}", flush=True)

    # --- Q2: the joint breaker across timepoints, per target ---
    print(f"\n=== Q2. multi-timepoint breaker  (min_cells={min_cells}) ===", flush=True)
    for target in targets:
        rep = attribute_across_operating_points(
            ops, ras_switch_1node(), MARKERS, target, steps=steps, min_cells=min_cells
        )
        print(f"\n {target}  (expected mechanism: {EXPECTED.get(target, '?')})")
        for lab in ops:
            if lab in rep.single:
                call, nlls = rep.single[lab]
                prof = " ".join(f"{k}={v:.3f}" for k, v in nlls.items())
                print(f"   {lab:14s} n={rep.n_cells[lab]:6d}  single → {call}   [{prof}]")
            else:
                print(f"   {lab:14s} n={rep.n_cells[lab]:6d}  SKIPPED: "
                      f"{rep.skipped.get(lab, '?')}")
        if rep.multi is not None:
            call, nlls = rep.multi
            prof = " ".join(f"{k}={v:.3f}" for k, v in nlls.items())
            print(f"   → BREAKER (joint over {len(rep.single)} usable ops): "
                  f"{call}   [{prof}]")
        else:
            print(f"   → breaker needs ≥2 usable operating points; had "
                  f"{len(rep.single)} (see SKIPPED above).")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("files", nargs="+", help="one .h5ad per timepoint (operating point)")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument(
        "--min-cells", type=int, default=200,
        help="min target cells per op to attempt a call (lower = attempt underpowered)",
    )
    args = ap.parse_args()
    run(args.files, steps=args.steps, min_cells=args.min_cells)


if __name__ == "__main__":
    main()
