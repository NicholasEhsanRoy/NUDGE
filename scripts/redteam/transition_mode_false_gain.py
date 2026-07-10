"""RED-TEAM (round 2) HELD probe: can a NON-gain perturbation drive the saddle
transition-mode GAIN gate to a confident wrong 'gain'?

Target: the saddle gain gate (classify.decide_with_transition) inside the SHIPPED
fit_multibasin(transition_mode=True). It fires 'gain' when the free-n restricted fit spends
transition weight w_trans > gain_wtrans_tau (0.5) on a 1-species saddle -- claimed a clean
gain signature (w_trans ~0.9 gain vs ~0.01 else; FINDINGS T0.5-5).

Attack: NON-gain perturbations (ceiling / threshold, incl. fold-crossing) on a 1-species
self-activation switch, on INDEPENDENT tau-leaping SSA data (off the inverse crime, the
emergent-bistable regime the path targets). A 'gain' call on any of these is confident-wrong.

Result: HELD (0 false-gain). Cases either abstain at the WT parsimony gate
(beats_lin=False) or, where the switch is detected, resolve the CORRECT non-gain mechanism
(K x0.5 -> threshold, conf 1.0). The deep-switch variant (a clearly-detected SSA switch +
strong ceiling collapse) likewise produced no false gain.

Run: uv run python scripts/redteam/transition_mode_false_gain.py [nseeds]
"""

from __future__ import annotations

import sys

from nudge.circuits import ras_switch_1node
from nudge.data.stochastic import generate_stochastic_perturbseq
from nudge.data.synthetic import PerturbationSpec
from nudge.inference.fit import fit_multibasin


def _run(circ, cases, n_cells, omega, nseeds, tag):
    holes = 0
    for name, param, factor, truth in cases:
        for seed in range(nseeds):
            adata = generate_stochastic_perturbseq(
                circ, [PerturbationSpec("cond", "edge", 0, param, factor)],
                n_cells_per_condition=n_cells, omega=omega, seed=seed,
            )
            mm = fit_multibasin(
                adata, circ, transition_mode=True, seed=seed, conditions=["cond"]
            )
            c = mm.calls[0]
            got = c.mechanism.value
            is_hole = got == "gain"  # truth is NEVER gain here
            holes += int(is_hole)
            print(f"[{tag}:{name} truth={truth}] seed={seed} call={got!r} "
                  f"conf={c.confidence:.2f} beats_lin={mm.beats_linear_baseline}"
                  + ("  <== CONFIDENT-WRONG GAIN" if is_hole else ""))
    return holes


def main(nseeds: int = 2) -> int:
    holes = 0
    # (A) standard switch, ceiling + threshold movers (incl. fold-crossing).
    holes += _run(
        ras_switch_1node(n=6.0, vmax=2.0, K=1.0, basal=0.05),
        [
            ("ceil_0.5", "vmax", 0.5, "ceiling"),
            ("ceil_0.35", "vmax", 0.35, "ceiling"),
            ("thr_2.0", "K", 2.0, "threshold"),
            ("thr_3.0", "K", 3.0, "threshold"),
            ("thr_0.5", "K", 0.5, "threshold"),
        ],
        n_cells=3000, omega=50.0, nseeds=nseeds, tag="std",
    )
    # (B) deep, clearly-detectable switch + strong ceiling collapse (the fairest shot at the
    #     gate: WT switch reliably detected, high mode collapsed toward the saddle).
    holes += _run(
        ras_switch_1node(n=8.0, vmax=2.5, K=1.0, basal=0.05),
        [("ceil_0.45", "vmax", 0.45, "ceiling"), ("ceil_0.6", "vmax", 0.6, "ceiling")],
        n_cells=4000, omega=90.0, nseeds=nseeds, tag="deep",
    )
    print(f">>> false-gain holes: {holes}")
    return holes


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
