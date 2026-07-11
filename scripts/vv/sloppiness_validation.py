#!/usr/bin/env python3
"""Validate the sloppiness diagnostic: sloppy-but-predictive vs structurally-unidentifiable.

Proves the headline of :mod:`nudge.inference.sloppiness` on models with a KNOWN answer:

1. **A canonical SLOPPY model** — a sum of exponentials ``Σ A_m e^{-k_m t}`` (distinct
   rates → structurally identifiable, but the Fisher spectrum spans many decades). The
   naive condition-number / eigenvalue-gap test calls it **unidentifiable** (WRONG); the
   diagnostic correctly labels it **sloppy-but-predictive** and refuses to over-abstain.
2. **A structurally UNIDENTIFIABLE model** — ``A e^{-(k₁+k₂) t}`` (only the sum enters, so
   ``(k₁,k₂)`` is an exact null). The diagnostic labels it **unidentifiable** AND names the
   unrecoverable ``k₁/k₂`` combination.
3. **A WELL-CONSTRAINED control** — a linear model, a narrow spectrum.

Prints a table + the naive-vs-measured contrast and writes
``scripts/vv/sloppiness_validation_RESULTS.json``. Reproduce::

    JAX_ENABLE_X64=1 uv run python scripts/vv/sloppiness_validation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from nudge.inference.sloppiness import (  # noqa: E402
    analyze_model,
    redundant_exponential_predict,
    sum_of_exponentials_predict,
    well_conditioned_predict,
)

_RESULTS = Path(__file__).resolve().parent / "sloppiness_validation_RESULTS.json"


def _row(name: str, report, expect: str) -> dict:
    ok = report.label == expect
    print(f"\n## {name}")
    print(f"  measured label : {report.label}   (expected {expect})  {'OK' if ok else 'FAIL'}")
    print(f"  naive verdict  : {report.naive_verdict}   naive_is_wrong={report.naive_is_wrong}")
    print(f"  Fisher spectrum: cond={report.cond_number:.3e}  "
          f"span={report.spectral_span_decades:.1f} decades  "
          f"n_null={report.n_null_dims}")
    print(f"  predictive     : {report.predictive}  "
          f"(max relative prediction std {report.relative_prediction_std:.3%})")
    for nd in report.null_directions:
        print("  null direction : {" +
              ", ".join(f'{k}:{v:+.2f}' for k, v in nd.param_loadings.items()) +
              f"}}  ‖J·v‖={nd.prediction_sensitivity:.2e}")
        print(f"    hint: {nd.hint}")
    if report.fim_greedy_warning:
        print(f"  FIM-greedy WARNING: {report.fim_greedy_warning}")
    print(f"  reason: {report.reason}")
    return {
        "name": name, "expected": expect, "label": report.label, "ok": ok,
        "naive_verdict": report.naive_verdict, "naive_is_wrong": report.naive_is_wrong,
        "cond_number": report.cond_number,
        "spectral_span_decades": report.spectral_span_decades,
        "n_null_dims": report.n_null_dims, "predictive": report.predictive,
        "relative_prediction_std": report.relative_prediction_std,
        "sloppy_prediction_variance_fraction": report.sloppy_prediction_variance_fraction,
        "null_loadings": [nd.param_loadings for nd in report.null_directions],
    }


def main() -> int:
    t = np.linspace(0.05, 6.0, 60)
    sigma = 0.01
    rows = []

    # 1. sloppy-but-predictive (naive test WRONG)
    sloppy = sum_of_exponentials_predict(
        rates=[0.5, 1.3, 2.5, 4.5], amps=[1.0, 1.0, 1.0, 1.0], t=t
    )
    rows.append(_row("SLOPPY sum-of-exponentials", analyze_model(sloppy, sigma=sigma),
                     "sloppy-but-predictive"))

    # 2. structurally unidentifiable (a true null)
    redundant = redundant_exponential_predict(amp=1.0, k1=0.7, k2=0.9, t=t)
    rows.append(_row("UNIDENTIFIABLE A·e^{-(k1+k2)t}", analyze_model(redundant, sigma=sigma),
                     "unidentifiable"))

    # 3. well-constrained control
    linear = well_conditioned_predict(slope=2.0, offset=1.0, t=t)
    rows.append(_row("WELL-CONSTRAINED linear", analyze_model(linear, sigma=sigma),
                     "well-constrained"))

    n_ok = sum(r["ok"] for r in rows)
    naive_wrong = [r["name"] for r in rows if r["naive_is_wrong"]]
    print(f"\n# {n_ok}/{len(rows)} correctly labelled.")
    print(f"# the naive eigenvalue-gap test is WRONG on: {naive_wrong or 'none'} "
          "(NUDGE labels it correctly).")

    _RESULTS.write_text(json.dumps(
        {"n_correct": n_ok, "n_total": len(rows), "naive_wrong": naive_wrong, "rows": rows},
        indent=2,
    ))
    print(f"# wrote {_RESULTS}")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())