"""RED-TEAM (round 2) HELD probe: can a NON-switch static cell-type mixture, at ratios the
shipped mixture decoy (NUDGE-DECOY-002) does not cover, drive the CORE parsimony gate
(fit()->classify) to a CONFIDENT positive mechanism at the DEFAULT budget?

The shipped decoy battery tests at n_cells=384/steps=400; the SHIPPED fit() DEFAULT is
n_cells=256/steps=300/margin_k=1.7 -- the weaker, more attackable surface a bare
nudge.fit(adata, circuit) call uses. We sweep wider/more-balanced separations + a fraction-
shifting perturbation (truth: off-model / no switch) at that default budget.

Result: HELD -- every call is off-model, beats_linear_baseline=False. The single-basin fit()
solves the mechanistic hypothesis from x0=0, so the switch model is itself ~unimodal and
cannot out-fit the linear baseline on a bimodal STATIC mixture. A structural robustness of
the parsimony gate, not merely a calibration.

Run: uv run python scripts/redteam/core_parsimony_mixture.py
"""

from __future__ import annotations

import anndata as ad
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef
from nudge.data.noise import sample_counts, sample_library_sizes
from nudge.inference.fit import fit

POS = {"threshold", "gain", "ceiling"}


def _switch_hypothesis() -> Circuit:
    return Circuit(
        [SpeciesDef("SW", basal=0.05, decay=1.0)],
        [EdgeDef(0, 0, "hill_activation", K=1.0, n=6.0, vmax=2.0)],
    )


def mixture_adata(low, high, p_wt, p_pert, n, seed, disp=0.1, lib=0.2):
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)
    blocks, cond = [], []
    for name, p in (("WT", p_wt), ("pert", p_pert)):
        is_high = rng.uniform(size=n) < p
        activity = np.where(is_high, high, low)
        expr = jnp.asarray(0.2 + 5.0 * activity)[:, None]
        key, k_lib, k_c = jax.random.split(key, 3)
        library = sample_library_sizes(k_lib, n, log_sd=lib)
        counts = sample_counts(k_c, expr, library, dispersion=disp)
        blocks.append(np.asarray(counts))
        cond.extend([name] * n)
    counts = np.concatenate(blocks, axis=0)
    obs = pd.DataFrame(
        {"condition": cond, "true_mechanism": ["off-model"] * len(cond)},
        index=pd.Index([f"c{i}" for i in range(counts.shape[0])]),
    )
    return ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=pd.Index(["SW"])))


def main() -> int:
    hyp = _switch_hypothesis()
    configs = [
        ("wide_bal", 0.02, 4.0, 0.5, 0.2),
        ("wide_hi", 0.02, 4.0, 0.7, 0.3),
        ("vwide", 0.01, 8.0, 0.5, 0.15),
        ("crisp_shift", 0.02, 3.0, 0.6, 0.25),
    ]
    holes = 0
    for label, low, high, p_wt, p_pert in configs:
        for seed in (0, 1):
            adata = mixture_adata(low, high, p_wt, p_pert, 2000, seed)
            mm = fit(adata, hyp, seed=seed)  # DEFAULT budget
            for c in mm.calls:
                is_hole = c.mechanism.value in POS
                holes += int(is_hole)
                print(f"[{label}] seed={seed} beats_lin={mm.beats_linear_baseline} "
                      f"call={c.mechanism.value!r} conf={c.confidence:.2f}"
                      + ("  <== CONFIDENT-WRONG" if is_hole else ""))
    print(f">>> confident-wrong holes: {holes}")
    return holes


if __name__ == "__main__":
    main()
