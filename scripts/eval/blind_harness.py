#!/usr/bin/env python3
"""Generalizable BLIND-CASE harness for the automated-scientist eval — one registry, every
NUDGE use-case.

A blind case is generated from NUDGE's own synthetic generators, so its ground truth CANNOT be
memorized — the test agent must use the MCP tools and reason from the data. Each **surface**
(capability) registers a builder that (a) writes the agent-facing, SCRUBBED artifact(s), and
(b) returns a held-out ANSWER KEY + a calibrated expected verdict. The harness handles the
shared integrity concerns: the agent dir gets ONLY the scrubbed data; the answer key is labeled
+ preserved but written to a SEPARATE held-out keys dir (gitignored — completely inaccessible to
a test agent that may have repo access).

Adding a use-case (gLV microbiome, fibrillization, a new physics model…) = writing one
`@register("<surface>")` builder. Shared everywhere: the scrub, the held-out key, and the
CALIBRATION rubric (correct call OR honest abstention; a confident-WRONG is the only hard fail).

Surfaces today: `attribute` (gene-circuit snapshot), `dose-response` (the resolving flagship),
`lotka` (gLV / microbiome trajectories). Pending: `fibrillization` (a nudge-jax-physics build).

Usage:
  uv run python scripts/eval/blind_harness.py --surface dose-response --case switch \
      --seed 0 --agent-dir eval_cases/dose_switch --keys-dir eval_keys
  uv run python scripts/eval/blind_harness.py --list
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

# surface name -> builder(args, agent_dir) -> (answer_key: dict, summary: str)
Builder = Callable[[argparse.Namespace, Path], "tuple[dict[str, Any], str]"]
SURFACES: dict[str, Builder] = {}


def register(name: str) -> Callable[[Builder], Builder]:
    def deco(fn: Builder) -> Builder:
        SURFACES[name] = fn
        return fn
    return deco


# --------------------------------------------------------------------------- #
# Surface 1 — gene-circuit single-snapshot attribution (h5ad). Honest ABSTAIN case.
# --------------------------------------------------------------------------- #
@register("attribute")
def _build_attribute(args: argparse.Namespace, agent_dir: Path) -> tuple[dict[str, Any], str]:
    from nudge.circuits import ras_switch_1node
    from nudge.data.synthetic import PerturbationSpec, generate_synthetic_perturbseq
    from nudge.mechanisms.readout import Readout

    mech_param = {"threshold": "K", "gain": "n", "ceiling": "vmax"}
    param = mech_param[args.mechanism]
    circuit = ras_switch_1node()
    target_gene = str(circuit.names[0])
    readout = Readout.identity(circuit.n_species, scale=args.depth_scale)
    adata = generate_synthetic_perturbseq(
        circuit, [PerturbationSpec("treated", "edge", 0, param, args.factor)],
        readout=readout, n_cells_per_condition=args.n_cells, seed=args.seed,
    )
    # Background genes so library-size normalization is meaningful (a 1-gene screen degenerates).
    rng = np.random.default_rng(args.seed + 20260710)
    bg = rng.poisson(args.bg_rate, size=(adata.n_obs, args.n_background)).astype(np.float32)
    x = np.hstack([np.asarray(adata.X, dtype=np.float32), bg])
    var = pd.DataFrame(index=[*adata.var_names, *(f"BG{i}" for i in range(args.n_background))])
    obs = pd.DataFrame(adata.obs).copy()
    # SCRUB: drop the leak columns/uns; neutralize condition labels.
    if "true_mechanism" in obs.columns:
        obs = obs.drop(columns=["true_mechanism"])
    obs["condition"] = obs["condition"].map({"WT": "control", "treated": target_gene}).astype(str)
    out = ad.AnnData(X=x, obs=obs, var=var)  # note: no uns -> no ground_truth leak
    assert "true_mechanism" not in out.obs and not out.uns, "scrub failed"
    out.write_h5ad(agent_dir / "blind_test.h5ad")

    key = {
        "surface": "attribute", "ground_truth_mechanism": args.mechanism, "perturbed_param": param,
        "factor": args.factor, "seed": args.seed, "topology": "1node", "target_gene": target_gene,
        "conditions": {"control": "control", "perturbed": target_gene},
        "expected": "unresolved / gain_or_threshold — a single snapshot cannot separate gain from "
        "threshold; the honest answer is abstain + 'need a 2nd operating point'. Ceiling may "
        "resolve. Confident-wrong is the only hard fail.",
    }
    return key, (f"attribute: {args.mechanism} ({param}x{args.factor}), target={target_gene}, "
                 f"{out.n_obs} cells x {out.n_vars} genes")


# --------------------------------------------------------------------------- #
# Surface 2 — dose-response (CSV). The RESOLVING flagship + a truncated abstain probe.
# --------------------------------------------------------------------------- #
@register("dose-response")
def _build_dose(args: argparse.Namespace, agent_dir: Path) -> tuple[dict[str, Any], str]:
    cases = {"switch": (6.0, True, "switch"), "graded": (1.0, True, "graded"),
             "truncated": (6.0, False, "unresolved")}
    n_true, spans, expected = cases[args.case]
    k, amp, floor = 1.0, 1.0, 0.05
    rng = np.random.default_rng(args.seed)
    doses = (np.geomspace(0.05 * k, 20.0 * k, args.n_doses) if spans
             else np.geomspace(0.02 * k, 0.5 * k, args.n_doses))
    d = np.repeat(doses, args.n_reps)
    resp = floor + amp * d**n_true / (k**n_true + d**n_true) + rng.normal(0.0, args.noise, d.shape)
    pd.DataFrame({"dose": d, "response": resp}).to_csv(agent_dir / "blind_dose.csv", index=False)
    key = {
        "surface": "dose-response", "case": args.case, "true_n": n_true, "K": k,
        "doses_span_inflection": spans, "direction": "activate", "seed": args.seed,
        "expected_nudge_verdict": expected,
        "expected": f"{args.case}: NUDGE should return '{expected}'. For 'truncated' a confident "
        "'switch' is the confident-WRONG failure (NUDGE-LIM-007); abstaining is the PASS.",
    }
    return key, (f"dose-response: case={args.case} (true_n={n_true}, spans={spans}) "
                 f"-> expect {expected}")


# --------------------------------------------------------------------------- #
# Surface 3 — gLV / microbiome temporal attribution (npz trajectories).
# --------------------------------------------------------------------------- #
@register("lotka")
def _build_lotka(args: argparse.Namespace, agent_dir: Path) -> tuple[dict[str, Any], str]:
    from nudge.inference.lotka_volterra import simulate_glv_perturbseq

    ds = simulate_glv_perturbseq(
        n_species=args.n_species, mechanism=args.glv_mechanism, seed=args.seed
    )
    # SCRUB: serialize ONLY the observable arrays; the baseline kinetics + ground_truth (the
    # answer) are NEVER written to the agent file.
    np.savez(
        agent_dir / "blind_glv.npz",
        reference=ds.reference, perturbed=ds.perturbed, t_obs=ds.t_obs,
        u_grid=ds.u_grid, obs_idx=ds.obs_idx, dt=np.asarray(ds.dt),
    )
    gt = ds.ground_truth
    key = {
        "surface": "lotka", "ground_truth": {k: (v if _jsonable(v) else str(v))
                                              for k, v in gt.items()},
        "seed": args.seed, "n_species": ds.n_species,
        "expected": "Recover WHICH parameter moved (growth alpha / interaction beta / "
        "susceptibility epsilon) or ABSTAIN on the alpha<->beta_ii degeneracy — the antibiotic "
        "epsilon axis is the identifiable one. Confident-wrong is the only hard fail.",
        "note": "gLV file-ingestion for the agent (a CLI/MCP verb over blind_glv.npz) is the "
        "wiring to confirm/add; the blind DATA + held-out key generalize the pattern already.",
    }
    return key, (f"lotka: gLV {ds.n_species}-species, moved={args.glv_mechanism}, "
                 f"truth={gt.get('mechanism')}")


# --------------------------------------------------------------------------- #
# Surface 4 — fibrillization (pending a nudge-jax-physics build).
# --------------------------------------------------------------------------- #
@register("fibrillization")
def _build_fibrillization(args: argparse.Namespace, agent_dir: Path) -> tuple[dict[str, Any], str]:
    raise NotImplementedError(
        "The fibrillization (Smoluchowski / Oosawa-Kasai) capability is not built yet — it is a "
        "nudge-jax-physics hardening-loop deliverable. When it lands, register a builder here: "
        "generate a known-mechanism aggregation curve, scrub, hold out the answer key. The harness "
        "contract is unchanged."
    )


def _jsonable(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool, type(None)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surface", choices=sorted(SURFACES))
    ap.add_argument("--list", action="store_true", help="list registered surfaces and exit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agent-dir", default="eval_cases/case")
    ap.add_argument("--keys-dir", default="eval_keys")
    # attribute
    ap.add_argument("--mechanism", choices=["threshold", "gain", "ceiling"], default="threshold")
    ap.add_argument("--factor", type=float, default=1.6)
    ap.add_argument("--n-cells", type=int, default=4000)
    ap.add_argument("--depth-scale", type=float, default=150.0)
    ap.add_argument("--n-background", type=int, default=100)
    ap.add_argument("--bg-rate", type=float, default=2.0)
    # dose-response
    ap.add_argument("--case", choices=["switch", "graded", "truncated"], default="switch")
    ap.add_argument("--n-doses", type=int, default=12)
    ap.add_argument("--n-reps", type=int, default=4)
    ap.add_argument("--noise", type=float, default=0.03)
    # lotka
    ap.add_argument("--n-species", type=int, default=3)
    ap.add_argument("--glv-mechanism", default="susceptibility",
                    choices=["growth", "interaction", "susceptibility", "none"],
                    help="which gLV knob to move (susceptibility=the identifiable antibiotic axis)")
    args = ap.parse_args()

    if args.list or not args.surface:
        print("registered surfaces:", ", ".join(sorted(SURFACES)))
        return 0

    agent_dir, keys_dir = Path(args.agent_dir), Path(args.keys_dir)
    agent_dir.mkdir(parents=True, exist_ok=True)
    keys_dir.mkdir(parents=True, exist_ok=True)
    case_id = agent_dir.name

    answer_key, summary = SURFACES[args.surface](args, agent_dir)
    answer_key = {"case_id": case_id, "surface": args.surface, **answer_key,
                  "grading_rubric": "Score CALIBRATION, not point-accuracy. A confident-WRONG "
                  "mechanism is the ONLY hard fail; a calibrated abstention is a PASS. Also verify "
                  "the agent reasoned from NUDGE's output, not priors."}
    (keys_dir / f"{case_id}_answer_key.json").write_text(json.dumps(answer_key, indent=2))

    print(f"BLIND CASE {case_id} [{summary}], seed={args.seed}")
    print(f"  agent dir (scrubbed):  {agent_dir}/  (files: "
          f"{sorted(p.name for p in agent_dir.iterdir())})")
    print(f"  answer key (HELD OUT): {keys_dir / (case_id + '_answer_key.json')}  "
          "(gitignored — inaccessible to the agent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
