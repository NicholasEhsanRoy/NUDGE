"""RED-TEAM (end-to-end): a NON-DEPTH batch offset on A+B fakes synergy through the
FULL bridge pipeline (combo_effect_scores → attribute_synergy), not just the classifier.

NUDGE-LIM-009's defense against a depth/batch confound is combo_effect_scores'
**size-factor** normalization (multiplicative library-size correction). This attack uses a
confound that is INVISIBLE to size-factor normalization: an ADDITIVE count offset applied
to the signature/HVG genes in the A+B condition ONLY (e.g. ambient-RNA contamination or a
batch-specific expression shift concentrated in the A+B 10x lane). Truth is ADDITIVE
(the A+B condition is generated as the Bliss-additive sum of A and B), yet the offset
survives normalization and the pipeline should read super-additive.

Builds a synthetic AnnData with a real gene panel, runs the SHIPPED bridge extractor and
the SHIPPED attribute_synergy, and checks whether it emits a confident ``synergistic``.

Run: uv run python scripts/redteam/epistasis_pipeline_confound.py
Touches no src/ code and no fail-safe margins.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from nudge.inference.bridge import combo_effect_scores
from nudge.inference.epistasis import attribute_synergy

N_GENES = 200
N_SIG = 20  # the signature genes that carry the (additive) effect + the batch offset
N_CELLS = 700


def _counts(rng: np.random.Generator, n: int, sig_rate: float, base_rate: float,
            batch: float) -> np.ndarray:
    """Poisson counts: N_SIG signature genes at ``sig_rate`` (+``batch`` offset), the
    rest at ``base_rate``."""
    lam = np.full((n, N_GENES), base_rate, dtype=float)
    lam[:, :N_SIG] = sig_rate + batch
    return rng.poisson(lam).astype(np.float32)


def build_adata(seed: int) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    # Truth ADDITIVE: signature rate rises additively (ctrl<A=B<A+B = ctrl + 2·delta).
    base = 1.0
    d = 1.5
    blocks = {
        "control": _counts(rng, N_CELLS, base, base, 0.0),
        "A": _counts(rng, N_CELLS, base + d, base, 0.0),
        "B": _counts(rng, N_CELLS, base + d, base, 0.0),
        # A+B is the additive sum in rate space, PLUS a batch offset on the signature
        # genes ONLY (the confound, perfectly aligned with the A+B condition).
        "AB": _counts(rng, N_CELLS, base + 2 * d, base, 3.0),
    }
    X = np.vstack(list(blocks.values()))
    cond = np.concatenate([[k] * N_CELLS for k in blocks])
    obs = pd.DataFrame({"condition": cond})
    obs["total_counts"] = X.sum(axis=1)
    var = pd.DataFrame(index=[f"g{i}" for i in range(N_GENES)])
    return ad.AnnData(X=X, obs=obs, var=var)


def run() -> int:
    print("=" * 78)
    print("END-TO-END: additive (non-depth) batch offset on A+B; TRUTH = ADDITIVE")
    print("=" * 78)
    holes = 0
    for seed in range(4):
        adata = build_adata(seed)
        c, a, b, ab, geom = combo_effect_scores(
            adata,
            control_label="control",
            a_label="A",
            b_label="B",
            ab_label="AB",
            return_geometry=True,
        )
        res = attribute_synergy(c, a, b, ab, n_boot=400, seed=seed, geometry=geom)
        wrong = res.call in ("synergistic", "buffering")
        holes += int(wrong)
        nr = None if geom is None else round(geom.neomorphic_ratio, 2)
        print(
            f"  seed={seed}: call={res.call!r} interaction={res.fit.interaction:+.3f} "
            f"CI={tuple(round(x, 3) for x in res.fit.ci_interaction)} "
            f"neomorphic_ratio={nr}" + ("  <== CONFIDENT-WRONG" if wrong else "")
        )
        if wrong:
            print(f"      reason: {res.reason[:200]}")
    print(f"\n>>> end-to-end confident-wrong (synergistic/buffering on additive): {holes}")
    return holes


if __name__ == "__main__":
    run()
