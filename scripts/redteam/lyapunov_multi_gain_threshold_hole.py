"""RED-TEAM (P7) — the multi-operating-point breaker resolves a THRESHOLD-DOMINATED
large-gain perturbation to a CONFIDENT WRONG 'threshold' (NUDGE-LIM-025).

``attribute_lyapunov_multi`` (M3, the gain-vs-threshold breaker) resolves the mechanism
whose shared-parameter joint NLL wins by ``resolve_margin`` (0.03). It has near-fold
proximity down-weighting + best-buffered-pair corroboration (NUDGE-LIM-017) — but those
inspect the WT/CONTROL circuit at each operating point. The degeneracy that bites here is
on the PERTURBED side.

THE HOLE (measured through the shipped API, LNA-mixture ground truth on a mutual-repression
toggle, SCALE=20, OBS_SD=0.5, two operating points bB ∈ {0.05, 0.30}, 3000 cells each):

  * gain n=1.5  → the true knob is GAIN, but the perturbed condition is threshold-DOMINATED
    in the LNA moments (flattening the Hill curve shifts the effective EC50; at bB=0.30 the
    perturbed circuit even slides MONOSTABLE, at bB=0.05 it sits AT the fold, prox≈0.54).
    The second operating point therefore does NOT break the gain⇄threshold degeneracy, yet
    a shared-K fit wins the NLL gap by ≈1.7 ≫ 0.03 → resolves 'threshold' = CONFIDENT WRONG.

Positive controls (must NOT regress):
  * threshold K=2.0 → resolves 'threshold' (a genuine, resolvable threshold shift).
  * gain n=2.4  → 'unresolved' (gain is the argmin but the gap is < 0.03 → safe abstain).
  * a monostable operating point → ('unresolved', {}) gracefully (never a ValueError).

THE FIX (NUDGE-LIM-025, additive in inference/lyapunov.py): an IDENTIFIABILITY GATE. After
the breaker resolves a single mechanism X, fit the joint (X, runner-up Y) two-mechanism
model and read its Laplace posterior (uncertainty.laplace_posterior). If a runner-up Y is
IDENTIFIABLE and DISPLACED from its no-change value beyond the MEASURED cut
(``_CONTAM_MARGIN`` = 0.5 log-units; genuine ≤ 0.12 vs the n=1.5 hole ≈ 1.0), the data
demonstrably needs a second mechanism → ABSTAIN. Plus a graceful "bistability lost"
degradation (abstain, don't crash) when an operating point is monostable.

After the fix this script reports HOLES: 0.

Run: uv run python scripts/redteam/lyapunov_multi_gain_threshold_hole.py [nseeds]
Touches no src/ code — diagnostic only.
"""

from __future__ import annotations

import sys

import jax
import numpy as np

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef
from nudge.inference.bifurcation import bifurcation_proximity
from nudge.inference.lyapunov import (
    OperatingPoint,
    attribute_lyapunov_multi,
    calibrate_from_wt,
    sample_lna_mixture,
)

SCALE, OBS_SD = 20.0, 0.5
BASALS = (0.05, 0.30)


def _toggle(bB: float) -> Circuit:
    return Circuit(
        [SpeciesDef("A", basal=0.05, decay=1.0), SpeciesDef("B", basal=bB, decay=1)],
        [
            EdgeDef(1, 0, "hill_repression", K=1.0, n=4.0, vmax=2.0),
            EdgeDef(0, 1, "hill_repression", K=1.0, n=4.0, vmax=2.0),
        ],
    )


def _perturbed(bB: float, param: str, val: float) -> Circuit:
    from dataclasses import replace

    wt = _toggle(bB)
    edges = list(wt.edges)
    edges[0] = replace(edges[0], **{param: val})
    return Circuit(list(wt.species), edges)


def _op(bB: float, param: str, val: float, seed: int) -> OperatingPoint:
    wt = _toggle(bB)
    wt_data = sample_lna_mixture(
        wt, 3000, jax.random.PRNGKey(500 + seed), scale=SCALE, obs_sd=OBS_SD
    )
    scale, obs = calibrate_from_wt(wt_data, wt)
    cond = sample_lna_mixture(
        wt, 3000, jax.random.PRNGKey(seed),
        free=[("edge", 0, param)], vals=np.array([val]), scale=SCALE, obs_sd=OBS_SD,
    )
    return OperatingPoint(data=cond, circuit=wt, scale=scale, obs_sd=obs)


def _perturbed_stability(param: str, val: float) -> str:
    out = []
    for bB in BASALS:
        circ = _perturbed(bB, param, val)
        sc = bifurcation_proximity(circ)
        fps = circ.fixed_points()
        n_stable = 0 if fps is None else sum(1 for _s, lab in fps if lab == "stable")
        prox = "None(monostable)" if sc is None else f"{sc.proximity:.3f}"
        out.append(f"bB={bB}:n_stable={n_stable},prox={prox}")
    return "  ".join(out)


def run(nseeds: int) -> int:
    print("=" * 80, flush=True)
    print("RED-TEAM P7: multi-point breaker on a threshold-dominated large-gain "
          "perturbation (NUDGE-LIM-025)", flush=True)
    print("=" * 80, flush=True)

    holes = 0
    # (name, param, val, is_true_gain, expect_after_fix)
    cases = [
        ("gain n=1.5  (HOLE: true=gain, threshold-dominated)", "n", 1.5, "gain",
         "unresolved"),
        ("threshold K=2.0 (control: genuine threshold)", "K", 2.0, "threshold",
         "threshold"),
        ("gain n=2.4  (control: abstains, gap<0.03)", "n", 2.4, "gain", "unresolved"),
    ]
    for name, param, val, truth, _expect in cases:
        print(f"\n--- {name} ---", flush=True)
        print(f"    perturbed stability: {_perturbed_stability(param, val)}", flush=True)
        for seed in range(nseeds):
            pts = [_op(b, param, val, seed) for b in BASALS]
            label, nlls = attribute_lyapunov_multi(pts, target_edge=0, steps=200, seed=0)
            gap = (sorted(nlls.values())[1] - sorted(nlls.values())[0]) if nlls else float("nan")
            # a confident-WRONG = a bare mechanism that is NOT the truth
            is_hole = label in ("gain", "threshold", "ceiling") and label != truth
            holes += int(is_hole)
            flag = "  <== CONFIDENT-WRONG HOLE" if is_hole else ""
            print(f"    seed={seed}: label={label!r}  gap={gap:.4f}"
                  f"  nlls={ {k: round(v,3) for k,v in nlls.items()} }{flag}", flush=True)

    # graceful monostable degradation (must not raise)
    print("\n--- monostable operating point (graceful degradation) ---", flush=True)
    mono = _toggle(0.5)
    data = np.zeros((50, 2), dtype=np.float32)
    op = OperatingPoint(data=data, circuit=mono, scale=20.0, obs_sd=0.5)
    try:
        label, nlls = attribute_lyapunov_multi([op, op], target_edge=0, steps=5, seed=0)
        crashed = not (label == "unresolved" and nlls == {})
        print(f"    label={label!r} nlls={nlls}"
              f"{'  <== EXPECTED unresolved/{}' if crashed else '  (graceful)'}",
              flush=True)
        holes += int(crashed)
    except Exception as e:  # noqa: BLE001
        print(f"    RAISED {type(e).__name__}: {e}  <== UNGRACEFUL CRASH", flush=True)
        holes += 1

    print(f"\n>>> HOLES: {holes}", flush=True)
    return holes


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    sys.exit(0 if run(ns) == 0 else 1)
