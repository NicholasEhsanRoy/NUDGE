#!/usr/bin/env python3
"""Generate the synthetic AD-QSP cohort dataset for the A/B comparison (NO ``nudge`` dependency).

Writes:
  - ``cohort.npz``  — the full arrays (true per-subject params, log-biomarker observations,
    obs times, dose schedule) for a programmatic fit / OED attempt.
  - ``cohort.csv``  — a tidy long table (subject, time, biomarker, value) for eyeballing.

The dataset is a SYNTHETIC ground-truth population from ``ad_qsp_forward.generate_cohort`` —
never real patients; demo-scaled constants (``NUDGE-LIM-026``). Run from this directory::

    python make_dataset.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from ad_qsp_forward import BIOMARKERS, SPECIES, generate_cohort

_HERE = Path(__file__).resolve().parent


def main() -> None:
    data = generate_cohort(n_subjects=40, n_obs_times=8, seed=0)
    np.savez(_HERE / "cohort.npz", **data)

    biom_names = [SPECIES[i] for i in BIOMARKERS]
    obs = data["observations"]  # (n_subjects, n_obs, n_biom) log-biomarkers
    obs_times = data["obs_times"]
    with (_HERE / "cohort.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "time", "biomarker", "log_value"])
        for s in range(obs.shape[0]):
            for ti, t in enumerate(obs_times):
                for bi, bname in enumerate(biom_names):
                    w.writerow([s, f"{t:.4f}", bname, f"{obs[s, ti, bi]:.6f}"])

    print(f"wrote cohort.npz  ({obs.shape[0]} subjects × {obs.shape[1]} times × "
          f"{obs.shape[2]} biomarkers)")
    print(f"wrote cohort.csv  ({obs.size} rows)")
    print("true_params shape:", data["true_params"].shape, "(the ground truth a fit must recover)")


if __name__ == "__main__":
    main()
