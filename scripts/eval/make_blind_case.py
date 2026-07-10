#!/usr/bin/env python3
"""Generate a BLIND mechanism-attribution test case for the automated-scientist eval.

Uses NUDGE's own synthetic generator to create a fresh dataset whose ground-truth mechanism
is chosen here and seeded — so it is **mathematically impossible for any model to have
memorized the answer**; the agent must use NUDGE (the MCP tools) and reason from the data.

The dataset written for the agent is SCRUBBED of every ground-truth leak:
  - ``obs['true_mechanism']`` is dropped,
  - ``uns['ground_truth']`` is dropped,
  - condition labels are neutralized (``WT`` -> ``control``, the perturbation -> ``treated``).
The held-out answer key (mechanism / params / seed) is written to a SEPARATE keys directory
that must NOT be placed in the agent's working directory.

Additive / opt-in — reuses ``data.synthetic`` + ``circuits``; touches neither fit.py nor core/.

Usage:
  uv run python scripts/eval/make_blind_case.py \
      --mechanism threshold --factor 1.6 --n-cells 5000 --seed 0 \
      --agent-dir eval_cases/case01 --keys-dir eval_keys
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from nudge.circuits import ras_switch_1node
from nudge.data.synthetic import PerturbationSpec, generate_synthetic_perturbseq
from nudge.mechanisms.readout import Readout

# The mechanism the agent must recover -> the circuit parameter the generator perturbs.
_MECH_PARAM = {"threshold": "K", "gain": "n", "ceiling": "vmax"}
# Honest, calibrated "right answer" for a SINGLE-snapshot attribution (what a well-behaved
# scientist using NUDGE should conclude), given NUDGE's measured identifiability:
#   ceiling is identifiable from one snapshot; gain/threshold are confounded -> abstain.
_SINGLE_SNAPSHOT_EXPECTATION = {
    "threshold": "gain_or_threshold (abstain between gain and threshold from ONE snapshot; "
    "the honest answer is 'threshold or gain — needs a 2nd operating point')",
    "gain": "gain_or_threshold (same confound; abstain, do not claim gain confidently)",
    "ceiling": "ceiling (identifiable from a single snapshot — a confident correct call)",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mechanism", choices=sorted(_MECH_PARAM), default="threshold")
    ap.add_argument("--factor", type=float, default=1.6,
                    help="multiplier on the perturbed parameter (the ground-truth effect size)")
    ap.add_argument("--n-cells", type=int, default=5000, help="cells PER condition (x2 total)")
    ap.add_argument("--depth-scale", type=float, default=150.0,
                    help="readout depth (counts/activity); high enough to clear NUDGE's LNA "
                    "depth guard so the fit tests the MECHANISM, not the depth guard")
    ap.add_argument("--n-background", type=int, default=100,
                    help="background (non-informative) genes so size-factor normalization is "
                    "meaningful — a single-gene screen degenerates under library normalization")
    ap.add_argument("--bg-rate", type=float, default=2.0, help="Poisson rate per background gene")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agent-dir", default="eval_cases/case01",
                    help="dir the agent sees — gets ONLY the scrubbed blind_test.h5ad")
    ap.add_argument("--keys-dir", default="eval_keys",
                    help="held-out dir for answer_key.json — must NOT be given to the agent")
    args = ap.parse_args()

    param = _MECH_PARAM[args.mechanism]
    circuit = ras_switch_1node()
    target_gene = str(circuit.names[0])  # the readout/target gene (fair info; not the answer)

    pert = PerturbationSpec(
        name="treated", scope="edge", index=0, param=param, factor=args.factor
    )
    readout = Readout.identity(circuit.n_species, scale=args.depth_scale)
    adata = generate_synthetic_perturbseq(
        circuit, [pert], readout=readout, n_cells_per_condition=args.n_cells, seed=args.seed
    )

    # Add background (non-informative) genes so library-size normalization is meaningful — a
    # single-gene screen degenerates (every cell normalizes to 1.0). A real Perturb-seq screen
    # has thousands of genes; the target gene's signal survives normalization against them.
    if args.n_background > 0:
        rng = np.random.default_rng(args.seed + 20260710)
        bg = rng.poisson(args.bg_rate, size=(adata.n_obs, args.n_background)).astype(np.float32)
        x = np.asarray(adata.X, dtype=np.float32)
        combined = np.hstack([x, bg])
        var = pd.DataFrame(index=[*adata.var_names, *(f"BG{i}" for i in range(args.n_background))])
        adata = ad.AnnData(X=combined, obs=adata.obs.copy(), var=var, uns=dict(adata.uns))

    # --- SCRUB (integrity-critical): remove every ground-truth leak ---
    if "true_mechanism" in adata.obs:
        del adata.obs["true_mechanism"]
    adata.uns.pop("ground_truth", None)
    # The perturbed condition is labeled with the TARGET GENE (standard Perturb-seq convention:
    # cells carrying the guide against gene X are labeled X — this is how the attribution path
    # selects target cells; it does NOT reveal the mechanism). Control cells -> "control".
    control_label = "control"
    adata.obs["condition"] = (
        adata.obs["condition"].map({"WT": control_label, "treated": target_gene}).astype(str)
    )
    # Sanity: the scrubbed file must carry NO mechanism/param/seed hint.
    leaks = [k for k in adata.uns if "truth" in k.lower() or "mechan" in k.lower()]
    assert not leaks and "true_mechanism" not in adata.obs, f"scrub failed: {leaks}"

    agent_dir = Path(args.agent_dir)
    keys_dir = Path(args.keys_dir)
    agent_dir.mkdir(parents=True, exist_ok=True)
    keys_dir.mkdir(parents=True, exist_ok=True)

    blind_path = agent_dir / "blind_test.h5ad"
    adata.write_h5ad(blind_path)

    case_id = agent_dir.name
    answer_key = {
        "case_id": case_id,
        "ground_truth_mechanism": args.mechanism,
        "perturbed_param": param,
        "factor": args.factor,
        "seed": args.seed,
        "topology": "1node",
        "target_gene": target_gene,
        "conditions": {"control": "control", "perturbed": "treated"},
        "n_cells_per_condition": args.n_cells,
        "single_snapshot_calibrated_expectation": _SINGLE_SNAPSHOT_EXPECTATION[args.mechanism],
        "grading_note": (
            "Score CALIBRATION, not point-accuracy. A confident-WRONG mechanism is the only "
            "hard fail. For a single snapshot, a calibrated abstention (gain_or_threshold / "
            "'need a 2nd operating point') is a PASS for a threshold/gain truth; ceiling is a "
            "confident correct call. Also check the agent reasoned from NUDGE's output, not priors."
        ),
    }
    key_path = keys_dir / f"{case_id}_answer_key.json"
    key_path.write_text(json.dumps(answer_key, indent=2))

    # Report to stdout (the operator sees this; the agent never does).
    print(f"BLIND CASE {case_id}: mechanism={args.mechanism} ({param}x{args.factor}), "
          f"seed={args.seed}, target_gene={target_gene}")
    print(f"  agent file  (scrubbed): {blind_path}  "
          f"[{adata.n_obs} cells x {adata.n_vars} genes, conditions="
          f"{sorted(set(adata.obs['condition']))}]")
    print(f"  answer key (HELD OUT): {key_path}")
    print(f"  obs columns exposed to agent: {list(adata.obs.columns)}  (no mechanism leak)")
    print(f"  uns keys exposed to agent:    {list(adata.uns)}  (no ground_truth leak)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
