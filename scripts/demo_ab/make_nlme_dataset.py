#!/usr/bin/env python3
"""Generate the SYNTHETIC coupled NLME AD-QSP cohort for the A/B comparison (NO ``nudge`` dep).

Writes ``cohort_nlme.npz`` — the joint parameter vector ``[μ | φ | r₀ … r_{N-1}]`` at population
truth, the per-subject random-effect multipliers, and the stacked log-biomarker observations of
the COUPLED (arrowhead) hierarchical model. Hand this to a raw agent and ask for the full joint
identifiability spectrum: because ``μ``/``φ`` couple every subject, block-summing does not apply,
so the natural route is the dense ``(n_params × n_params)`` FIM — which OOMs at scale.

Synthetic ground-truth population (never real patients); demo-scaled constants (``NUDGE-LIM-026``);
the measured dense-OOM-vs-matrix-free-flat contrast is the claim (``NUDGE-LIM-028``). Run from
this directory::

    python make_nlme_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ad_qsp_forward import PARAM_NAMES
from ad_qsp_nlme_forward import RE_PARAMS, _split_indices, joint_theta0, nlme_predict

_HERE = Path(__file__).resolve().parent


def main() -> None:
    n_subjects, omega, obs_noise, seed = 200, 0.2, 0.05, 0
    obs_times = np.linspace(0.0, 12.0, 2)  # sparse plaque-only → rank-deficient by shape
    theta0 = joint_theta0(n_subjects, omega=omega, seed=seed)
    clean = nlme_predict(theta0, n_subjects, obs_times, biomarkers=(2,))
    rng = np.random.default_rng(seed + 1)
    obs = clean + obs_noise * rng.standard_normal(clean.shape)

    re_idx, fixed_idx = _split_indices(RE_PARAMS)
    d, n_fixed = len(re_idx), len(fixed_idx)
    mu = theta0[:d]
    phi = theta0[d:d + n_fixed]
    r = theta0[d + n_fixed:].reshape(n_subjects, d)

    np.savez(
        _HERE / "cohort_nlme.npz",
        theta0=theta0,                       # joint [μ | φ | r_i] ground truth
        mu=mu, phi=phi,                      # shared population hyperparameters (the border)
        subject_multipliers=r,               # (N, d) r_i (per-subject random effects)
        observations=obs,                    # stacked log-biomarkers (flat)
        clean_observations=clean,
        obs_times=obs_times,
        re_params=np.array(RE_PARAMS),
        re_indices=re_idx, fixed_indices=fixed_idx,
        border_size=np.array([d + n_fixed]),
        n_subjects=np.array([n_subjects]),
        param_names=np.array(PARAM_NAMES),
        omega=np.array([omega]),
        note=np.array(["SYNTHETIC coupled NLME cohort — shared μ/φ border couples every subject "
                       "(arrowhead FIM). NOT real patients; demo-scaled (NUDGE-LIM-026/028)."]),
    )
    n_params = theta0.shape[0]
    print(f"wrote cohort_nlme.npz  (N={n_subjects} subjects, joint n_params={n_params}, "
          f"border={d + n_fixed}, n_obs={obs.shape[0]})")
    print(f"dense FIM would be {n_params}×{n_params} = {n_params * n_params * 8 / 1e6:.1f} MB "
          "at this small N; scales as (12 + 3·N)² → ~O(10 GB) at N≈2000+.")


if __name__ == "__main__":
    main()
