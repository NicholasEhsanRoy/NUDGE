"""READ-ONLY probe: WHY does toggle covariance attribution abstain on GAIN?

Diagnostic, not a feature. Imports NUDGE, runs synthetic experiments, prints
numbers. Touches no src/ code and no fail-safe margins.

Two questions:

  (A) MECHANISTIC (fast, deterministic). How much does each mechanism
      perturbation (gain n×0.6 / threshold K×1.6 / ceiling vmax×0.6 on ONE toggle
      edge) move the LNA observables the covariance loss actually reads --- the
      per-mode MEANS and COVARIANCES at the stable fixed points? If gain barely
      moves them, the covariance channel is structurally blind to gain and the
      abstention is FUNDAMENTAL in this channel (not a confound, not a tuning gap).

  (B) DOES A 3RD OPERATING POINT HELP? Re-run the shared-parameter multi fit on
      independent SSA data with 1, 2, and 3 basal operating points for a true GAIN
      perturbation, and measure the resolved-channel NLL gap (best vs 2nd-best).
      Compare to threshold/ceiling. resolve_margin = 0.03.

Run: uv run python scripts/analysis/toggle_gain_abstention_probe.py [A|B|both] [nseeds]
"""

from __future__ import annotations

import sys

import numpy as np

from nudge.circuits import toggle
from nudge.core.circuit import Circuit

FACTORS = {"n": 0.6, "K": 1.6, "vmax": 0.6}
MECH = {"n": "gain", "K": "threshold", "vmax": "ceiling"}


def _perturb(circ: Circuit, param: str, factor: float, edge: int = 0) -> Circuit:
    """Return a copy of the toggle with one edge's kinetic scaled by `factor`."""
    from dataclasses import replace

    edges = list(circ.edges)
    cur = getattr(edges[edge], param)
    edges[edge] = replace(edges[edge], **{param: cur * factor})
    return Circuit(list(circ.species), edges)


def _moment_report(basal: float) -> None:
    wt = toggle(basal=basal)
    wt_modes = wt.mode_covariances()
    assert wt_modes is not None
    wt_means = np.stack([m for m, _ in wt_modes])
    wt_covs = np.stack([c for _, c in wt_modes])

    print(f"\n--- basal={basal}: WT modes ---", flush=True)
    for i, (m, c) in enumerate(zip(wt_means, wt_covs, strict=True)):
        print(f"  mode {i}: mean={m}  cov diag={np.diag(c)}  "
              f"corr={c[0,1]/np.sqrt(c[0,0]*c[1,1]):+.3f}", flush=True)

    print(f"\n  {'mech':>10}  {'|dmean|/|mean|':>14}  {'|dcov|_F/|cov|_F':>16}  "
          f"{'dcorr(lobe0)':>12}", flush=True)
    for param in ("n", "K", "vmax"):
        p = _perturb(wt, param, FACTORS[param])
        modes = p.mode_covariances()
        if modes is None or len(modes) != len(wt_modes):
            print(f"  {MECH[param]:>10}  -> lost bistability / mode count changed",
                  flush=True)
            continue
        pm = np.stack([m for m, _ in modes])
        pc = np.stack([c for _, c in modes])
        dmean = np.linalg.norm(pm - wt_means) / (np.linalg.norm(wt_means) + 1e-12)
        dcov = np.linalg.norm(pc - wt_covs) / (np.linalg.norm(wt_covs) + 1e-12)
        c0, w0 = pc[0], wt_covs[0]
        dcorr = (c0[0, 1] / np.sqrt(c0[0, 0] * c0[1, 1])
                 - w0[0, 1] / np.sqrt(w0[0, 0] * w0[1, 1]))
        print(f"  {MECH[param]:>10}  {dmean:>14.4f}  {dcov:>16.4f}  {dcorr:>+12.4f}",
              flush=True)


def part_A() -> None:
    print("=" * 74, flush=True)
    print("(A) MECHANISTIC: how far each perturbation moves the LNA observables",
          flush=True)
    print("    (the mode MEANS + COVARIANCES the covariance loss reads).", flush=True)
    print("=" * 74, flush=True)
    for basal in (0.05, 0.30, 0.60):
        _moment_report(basal)


def part_B(nseeds: int) -> None:
    print("\n" + "=" * 74, flush=True)
    print("(B) DOES A 3RD OPERATING POINT HELP? shared-parameter multi fit on SSA",
          flush=True)
    print("    resolved-channel NLL gap = NLL(2nd best) - NLL(best); margin=0.03",
          flush=True)
    print("=" * 74, flush=True)

    from nudge.data.stochastic import generate_toggle_perturbseq
    from nudge.data.synthetic import PerturbationSpec
    from nudge.inference.bridge import counts_to_activity
    from nudge.inference.lyapunov import (
        OperatingPoint,
        attribute_lyapunov_multi,
        calibrate_from_wt,
    )
    from nudge.mechanisms.readout import Readout

    MARKERS = {"A": ["A"], "B": ["B"]}
    DEEP = Readout.identity(2, scale=15.0)
    BASALS = [0.05, 0.30, 0.60]

    def activity(adata, condition, circ):
        mask = np.asarray(adata.obs["condition"] == condition)
        return counts_to_activity(adata[mask], circ, MARKERS)

    for param in ("n", "K", "vmax"):
        for n_pts in (1, 2, 3):
            for seed in range(nseeds):
                points = []
                for basal in BASALS[:n_pts]:
                    circ = toggle(basal=basal)
                    adata = generate_toggle_perturbseq(
                        circ,
                        [PerturbationSpec("cond", "edge", 0, param, FACTORS[param])],
                        readout=DEEP,
                        n_cells_per_condition=3000,
                        seed=seed,
                    )
                    wt = activity(adata, "WT", circ)
                    cond = activity(adata, "cond", circ)
                    scale, obs = calibrate_from_wt(wt, circ)
                    points.append(
                        OperatingPoint(data=cond, circuit=circ, scale=scale, obs_sd=obs)
                    )
                label, nlls = attribute_lyapunov_multi(
                    points, target_edge=0, steps=200, seed=seed
                )
                if nlls:
                    ordered = sorted(nlls.values())
                    gap = ordered[1] - ordered[0]
                    detail = "  ".join(f"{k}={nlls[k]:.3f}" for k in
                                       ("gain", "threshold", "ceiling"))
                else:
                    gap = float("nan")
                    detail = "(abstained before fit)"
                print(f"  {MECH[param]:>10}  pts={n_pts}  seed={seed}  "
                      f"{label:>18}  gap={gap:.4f}   {detail}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    nseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    if which in ("A", "both"):
        part_A()
    if which in ("B", "both"):
        part_B(nseeds)
