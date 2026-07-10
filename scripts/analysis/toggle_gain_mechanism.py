"""READ-ONLY mechanistic diagnosis of the toggle-gain covariance blindness.

Complements toggle_gain_abstention_probe.py. Three deterministic experiments on
the LNA observables (mode means + covariances) --- no fitting, no SSA:

  1. SYMMETRIC-perturbation moment table. Perturb BOTH toggle edges (bistability
     preserved), matched so each knob's mean-field "effect size" is comparable, and
     measure how far each moves the mode means vs covariances. Isolates whether gain
     reshapes the covariance channel as strongly as threshold/ceiling.

  2. GAIN factor sweep vs distance-to-fold + lna_reliable. Does gain's covariance
     signature only become appreciable as the switch approaches the saddle-node ---
     i.e. exactly where lna_reliable ABSTAINS? If so the visible-gain regime and the
     trustworthy-LNA regime are disjoint (a fundamental tension).

  3. Is the gain covariance-change MIMICKABLE by K / vmax? For the gain-perturbed
     mode moments, find the best single-knob (K-only, vmax-only) match and report the
     residual --- the deterministic analogue of the restricted-fit tie.

Run: uv run python scripts/analysis/toggle_gain_mechanism.py
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from nudge.circuits import toggle
from nudge.core.circuit import Circuit
from nudge.inference.lyapunov import lna_reliable

MECH = {"n": "gain", "K": "threshold", "vmax": "ceiling"}


def _perturb(circ: Circuit, param: str, factor: float, edges=(0, 1)) -> Circuit:
    e = list(circ.edges)
    for i in edges:
        e[i] = replace(e[i], **{param: getattr(e[i], param) * factor})
    return Circuit(list(circ.species), e)


def _moments(circ: Circuit):
    m = circ.mode_covariances()
    if m is None:
        return None
    means = np.stack([a for a, _ in m])
    covs = np.stack([c for _, c in m])
    return means, covs


def _dist(a, b):
    (m0, c0), (m1, c1) = a, b
    if m0.shape != m1.shape:
        return None, None
    dm = np.linalg.norm(m1 - m0) / (np.linalg.norm(m0) + 1e-12)
    dc = np.linalg.norm(c1 - c0) / (np.linalg.norm(c0) + 1e-12)
    return dm, dc


def exp1_symmetric() -> None:
    print("=" * 74)
    print("1. SYMMETRIC perturbation (both edges): moment shift per knob")
    print("   dmean/dcov are relative Frobenius change vs WT (bistability preserved)")
    print("=" * 74)
    for basal in (0.05, 0.30):
        wt = toggle(basal=basal)
        wtm = _moments(wt)
        print(f"\n basal={basal}  WT lobe0 mean={wtm[0][0]}  "
              f"cov diag={np.diag(wtm[1][0])}")
        print(f"  {'knob':>10} {'factor':>7}  {'dmean':>8}  {'dcov':>8}  bistable?")
        # matched moderate factors
        for param, factor in (("n", 0.6), ("n", 0.4), ("K", 1.4), ("K", 1.7),
                               ("vmax", 0.7), ("vmax", 0.5)):
            pm = _moments(_perturb(wt, param, factor))
            if pm is None:
                print(f"  {MECH[param]:>10} {factor:>7.2f}   ---monostable---")
                continue
            dm, dc = _dist(wtm, pm)
            if dm is None:
                print(f"  {MECH[param]:>10} {factor:>7.2f}   ---mode-count change---")
                continue
            print(f"  {MECH[param]:>10} {factor:>7.2f}  {dm:>8.4f}  {dc:>8.4f}  yes")


def exp2_gain_vs_fold() -> None:
    print("\n" + "=" * 74)
    print("2. GAIN factor sweep: covariance signature vs distance-to-fold + guard")
    print("   symmetric n-scaling; lna_reliable(circuit, scale) at deep scale=15")
    print("=" * 74)
    for basal in (0.05, 0.30):
        wt = toggle(basal=basal)
        wtm = _moments(wt)
        print(f"\n basal={basal}")
        print(f"  {'n_factor':>8} {'n_eff':>6}  {'dmean':>8}  {'dcov':>8}  "
              f"{'lna_reliable':>28}")
        for factor in (0.9, 0.75, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3):
            circ = _perturb(wt, "n", factor)
            pm = _moments(circ)
            n_eff = wt.edges[0].n * factor
            ok, reason = lna_reliable(circ, 15.0)
            if pm is None:
                print(f"  {factor:>8.2f} {n_eff:>6.2f}   ---monostable---   "
                      f"{('OK' if ok else reason):>28}")
                continue
            dm, dc = _dist(wtm, pm)
            if dm is None:
                print(f"  {factor:>8.2f} {n_eff:>6.2f}   ---mode-count change---")
                continue
            print(f"  {factor:>8.2f} {n_eff:>6.2f}  {dm:>8.4f}  {dc:>8.4f}  "
                  f"{('OK' if ok else reason):>28}")


def exp3_mimic() -> None:
    print("\n" + "=" * 74)
    print("3. Is the GAIN covariance change mimickable by a K-only / vmax-only change?")
    print("   residual = min over factor of relative Frobenius dist to gain moments")
    print("   (means+covs stacked). Small residual = degenerate (knob interchangeable)")
    print("=" * 74)
    for basal in (0.05, 0.30):
        wt = toggle(basal=basal)
        gain = _perturb(wt, "n", 0.6)  # the "truth"
        gm = _moments(gain)
        target = np.concatenate([gm[0].ravel(), gm[1].ravel()])
        tnorm = np.linalg.norm(target) + 1e-12
        print(f"\n basal={basal}  (target = symmetric gain n*0.6)")
        for param in ("n", "K", "vmax"):
            best = np.inf
            best_f = None
            for factor in np.linspace(0.2, 2.5, 240):
                pm = _moments(_perturb(wt, param, float(factor)))
                if pm is None or pm[0].shape != gm[0].shape:
                    continue
                vec = np.concatenate([pm[0].ravel(), pm[1].ravel()])
                r = np.linalg.norm(vec - target) / tnorm
                if r < best:
                    best, best_f = r, float(factor)
            tag = "  <- true knob" if param == "n" else ""
            print(f"  {MECH[param]:>10}  best factor={best_f:.3f}  "
                  f"residual={best:.5f}{tag}")


if __name__ == "__main__":
    exp1_symmetric()
    exp2_gain_vs_fold()
    exp3_mimic()
