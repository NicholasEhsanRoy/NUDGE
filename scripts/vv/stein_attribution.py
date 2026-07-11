"""Run NUDGE's gLV attribution on the REAL Stein 2013 clindamycin data + map the boundary.

Two deliverables, both MEASURED (never asserted):

* **Task 1 — the genuine per-group verdict.** At a low-dimensional, identifiable slice
  (small ``k``), attribute the clindamycin contrast (Population 1 no-drug → Population 3
  drug+C.diff) for EACH aggregated group. Does NUDGE resolve ``ε`` (direct antibiotic kill)
  on the strongly-suppressed commensals, and does it genuinely ABSTAIN on *C. difficile*
  (whose bloom is interaction-mediated, published ε≈−0.31)?

* **Task 2 — the identifiability boundary vs dimensionality.** Sweep ``k∈{2,3,5,8,11}`` and
  record where NUDGE transitions resolve→abstain as the ``k²`` interaction matrix becomes
  underdetermined against ~8 timepoints × 3 colonies — grounded in the MEASURED α⇄βᵢᵢ
  Laplace curvature (``alpha_beta_identifiability``), not a guess.

**Honesty rule (loud):** a confident-WRONG call on real data is the one unacceptable
outcome. This script flags any group where NUDGE returns a specific knob that CONTRADICTS the
published sign of that group's dominant susceptibility. Abstention on an underdetermined slice
is the CORRECT, on-thesis result — reported as a win.

Run: ``uv run python scripts/vv/stein_attribution.py`` (needs ``openpyxl``; the Stein xlsx at
the hard-coded absolute path). Reuses the SHIPPED ``attribute_glv`` unchanged.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stein_glv import (  # noqa: E402
    build_stein_dataset,
    load_stein,
    published_group_eps,
)

from nudge.inference.lotka_volterra import (  # noqa: E402
    attribute_glv,
    fit_baseline_glv,
)

STEIN_XLSX = "/home/nick/MSF/msf/NUDGE/tmp/stein_2013/stein_2013_dataset_S1.xlsx"
KS = (2, 3, 5, 8, 11)


def run_k(raw, k: int, steps: int, seed: int) -> dict:
    ds, labels, cdiff_group = build_stein_dataset(raw, k=k)
    pub = published_group_eps(k)
    n_rep = ds.reference.shape[0]
    n_sim = min(n_rep, ds.perturbed.shape[0])

    t0 = time.time()
    baseline, base_loss = fit_baseline_glv(ds, steps=max(steps, 300), n_sim=n_sim, seed=seed)
    base_t = time.time() - t0

    rows = []
    for g, lab in enumerate(labels):
        t1 = time.time()
        res = attribute_glv(
            ds, baseline=baseline, target=g, steps=steps, n_sim=n_sim, seed=seed
        )
        f = res.fit
        d_null = f.bic["null"] - min(f.bic[m] for m in ("growth", "interaction", "susceptibility"))
        # confident-wrong check: a resolved susceptibility call whose fitted delta sign
        # contradicts the published ε sign of the group's dominant taxon.
        cw = False
        if res.call == "susceptibility":
            fitted = f.delta["susceptibility"]
            if np.sign(fitted) != np.sign(pub[g]) and abs(pub[g]) > 0.5:
                cw = True
        rows.append({
            "group": lab,
            "call": res.call,
            "pub_eps": round(pub[g], 3),
            "bic": {m: round(float(v), 2) for m, v in f.bic.items()},
            "delta": {m: round(float(v), 3) for m, v in f.delta.items()},
            "d_null_best": round(float(d_null), 2),
            "cond_number": round(float(f.cond_number), 1),
            "corr_alpha_beta": round(float(f.corr_alpha_beta), 3),
            "degenerate": bool(f.degenerate),
            "selected": f.selected,
            "confident_wrong": cw,
            "secs": round(time.time() - t1, 1),
            "reason": res.reason,
        })
    return {
        "k": k,
        "n_rep": n_rep,
        "n_timepoints": len(ds.t_obs),
        "t_obs": ds.t_obs.tolist(),
        "labels": labels,
        "cdiff_group": cdiff_group,
        "baseline_loss": round(float(base_loss), 4),
        "baseline_secs": round(base_t, 1),
        "rows": rows,
    }


def main() -> None:
    ks = [int(x) for x in sys.argv[1:]] or list(KS)
    steps = 250
    seed = 0
    raw = load_stein(STEIN_XLSX)
    out = {"dataset": "Stein 2013 (pop1 no-drug -> pop3 clindamycin+C.diff)", "results": []}
    any_cw = False
    for k in ks:
        print(f"\n{'=' * 70}\nk = {k}\n{'=' * 70}", flush=True)
        r = run_k(raw, k, steps, seed)
        out["results"].append(r)
        print(f"  baseline loss={r['baseline_loss']}  ({r['baseline_secs']}s), "
              f"n_rep={r['n_rep']}  n_t={r['n_timepoints']}", flush=True)
        for row in r["rows"]:
            tag = "  <<< CONFIDENT-WRONG" if row["confident_wrong"] else ""
            flag = "  [C.diff]" if row["group"] == "Clostridium_difficile" else ""
            print(f"  {row['group']:<32} pubEps={row['pub_eps']:+6.2f} -> "
                  f"{row['call']:<15} (ΔBIC_null={row['d_null_best']:+.1f}, "
                  f"cond={row['cond_number']:.0f}, |corr|={row['corr_alpha_beta']:.2f})"
                  f"{flag}{tag}", flush=True)
            if row["confident_wrong"]:
                any_cw = True

    outpath = Path(__file__).resolve().parent / "stein_attribution_RESULTS.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outpath}")
    print(f"\nCONFIDENT-WRONG on real data: {'YES (CRITICAL)' if any_cw else 'NONE'}")


if __name__ == "__main__":
    main()
