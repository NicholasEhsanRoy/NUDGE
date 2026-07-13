"""Self-contained Alzheimer's amyloid-β QSP forward model — NO ``nudge`` dependency.

This is the *same* differentiable Aβ-aggregation cascade the NUDGE ``identifiability`` and
``oed`` MCP tools analyse (``nudge.mechanisms.ad_qsp`` — the demo-scaled Proctor et al. 2013
model, BioModels ``BIOMD0000000488``, CC0), re-expressed here as **plain NumPy** so a raw
agent WITHOUT NUDGE can load the identical model + cohort and attempt the same questions. It
exists for a fair **A/B comparison**: point a bare LLM/coding agent at this file + the
synthetic cohort and ask it to (a) fit the per-subject kinetic constants, or (b) design the
best measurement schedule — then compare its answer to NUDGE's honest one.

**Honesty (the labels that must not be dropped):**

- **Synthetic cohort, never real patients.** The dataset generated from this model is a
  synthetic ground-truth population (subject-specific parameters × log-normal variability).
- **Demo-scaled, dimensionless constants.** The published stiff seconds-to-years
  parameterization cannot be integrated by an explicit RK4; this keeps the published reaction
  topology + rate-law forms with non-dimensionalized constants (``NUDGE-LIM-026``). The
  identifiability *structure* (which constants are sloppy; which pairs are confounded) is a
  property of the preserved rate-law forms.
- **The A/B point (``NUDGE-LIM-026``, and why NUDGE matters):** a single sparse biomarker
  schedule under-determines these kinetics — the population calibration is rank-deficient and
  the antibody-binding ⇄ microglial-clearance pair is confounded by a naive baseline+end
  schedule. A bare least-squares fit returns confident-but-unidentifiable numbers; NUDGE
  measures the Fisher geometry and abstains / prescribes the resolving experiment.

Run ``python make_dataset.py`` to (re)generate ``cohort.npz`` / ``cohort.csv`` from this model.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# the model definition (mirrors nudge.mechanisms.ad_qsp, verbatim rate-law forms)
# --------------------------------------------------------------------------- #
#: dynamic states of the Aβ-aggregation subsystem.
SPECIES: tuple[str, ...] = (
    "Abeta_monomer",     # M — soluble Aβ monomer
    "Abeta_oligomer",    # O — soluble oligomer / dimer
    "Abeta_plaque",      # P — aggregated plaque (≈ amyloid-PET)
    "antibody",          # A — therapeutic anti-Aβ mAb
    "antibody_complex",  # C — antibody-bound Aβ, cleared
    "microglia",         # G — activated microglia clearance capacity
)

#: the 12 kinetic parameters (order = the free-parameter vector).
PARAM_NAMES: tuple[str, ...] = (
    "s_M",    # monomer production
    "d_M",    # monomer degradation
    "k_agg",  # monomer→oligomer aggregation
    "k_dis",  # oligomer→monomer disaggregation
    "k_pf",   # oligomer→plaque formation
    "k_pg",   # plaque-growth GAIN (autocatalytic Hill numerator)
    "K_pg",   # plaque-growth THRESHOLD (Hill half-saturation)
    "k_dp",   # plaque disaggregation
    "k_on",   # antibody–Aβ binding affinity
    "d_A",    # antibody clearance
    "k_gl",   # microglial plaque clearance
    "k_ga",   # microglial activation by plaque
)

#: demo-scaled nominal (population-truth) values (dimensionless; NUDGE-LIM-026).
PARAM_VALUES: np.ndarray = np.array(
    [0.05, 0.05, 0.6, 0.12, 0.5, 1.3, 1.0, 0.06, 0.9, 0.5, 0.2, 0.35], dtype=np.float64
)

#: default seeded initial state (a physiological monomer pool; everything else 0).
X0: np.ndarray = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

#: antibody dosing window (infusion on ``[t_on, t_off)``) — the transient that separates the
#: antibody-binding ⇄ microglial-clearance confound.
DOSE_WINDOW: tuple[float, float] = (2.0, 8.0)

#: biomarker readouts (indices into SPECIES): plaque (≈ amyloid-PET), soluble oligomer (≈ CSF).
BIOMARKERS: tuple[int, ...] = (2, 1)

_X_CAP = 1e6


def ad_field(x: np.ndarray, p: np.ndarray, u: float) -> np.ndarray:
    """The Aβ-cascade vector field ``dx/dt = f(x, p, u)`` (RAW params ``p``, dose input ``u``).

    The plaque-growth term ``k_pg·O·P²/(K_pg²+P²)`` is the published Proctor
    ``AbetaPlaqueGrowth`` autocatalytic Hill switch (gain ``k_pg`` / threshold ``K_pg``); the
    rest is the mass-action aggregation cascade + antibody binding/clearance + microglial
    clearance. Identical algebra to ``nudge.mechanisms.ad_qsp.ad_field``.
    """
    M, Ol, Pl, A, C, G = x
    s_M, d_M, k_agg, k_dis, k_pf, k_pg, K_pg, k_dp, k_on, d_A, k_gl, k_ga = p
    r_agg = 0.5 * k_agg * M * M
    r_dis = k_dis * Ol
    r_pf = 0.5 * k_pf * Ol * Ol
    r_pg = k_pg * Ol * Pl * Pl / (K_pg * K_pg + Pl * Pl)
    r_dp = k_dp * Pl
    r_clr = k_gl * G * Pl
    dM = s_M - d_M * M - 2.0 * r_agg + 2.0 * r_dis - k_on * A * M
    dO = r_agg - r_dis - 2.0 * r_pf - k_on * A * Ol + r_dp
    dPl = r_pf + r_pg - r_dp - r_clr
    dA = u - d_A * A
    dC = k_on * A * (M + Ol) - 0.5 * d_A * C
    dG = k_ga * Pl * (1.0 - G) - 0.2 * G
    return np.array([dM, dO, dPl, dA, dC, dG], dtype=np.float64)


def _dose(t: float, dose: float, window: tuple[float, float]) -> float:
    return dose if (window[0] <= t < window[1]) else 0.0


def simulate_subject(
    p: np.ndarray | None = None,
    *,
    dose: float = 0.0,
    t_max: float = 12.0,
    dt: float = 0.04,
    x0: np.ndarray | None = None,
    dose_window: tuple[float, float] = DOSE_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-simulate one subject with fixed-step RK4 → ``(trajectory, times)`` (n+1 rows)."""
    p_arr = PARAM_VALUES if p is None else np.asarray(p, dtype=np.float64)
    x = (X0 if x0 is None else np.asarray(x0, dtype=np.float64)).copy()
    n = int(round(t_max / dt))
    traj = np.empty((n + 1, 6), dtype=np.float64)
    times = np.empty(n + 1, dtype=np.float64)
    traj[0] = x
    times[0] = 0.0
    for i in range(n):
        t = i * dt
        u = _dose(t, dose, dose_window)
        k1 = ad_field(x, p_arr, u)
        k2 = ad_field(x + 0.5 * dt * k1, p_arr, u)
        k3 = ad_field(x + 0.5 * dt * k2, p_arr, u)
        k4 = ad_field(x + dt * k3, p_arr, u)
        x = np.clip(x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), 0.0, _X_CAP)
        traj[i + 1] = x
        times[i + 1] = (i + 1) * dt
    return traj, times


def observe_biomarkers(
    traj: np.ndarray,
    times: np.ndarray,
    obs_times: np.ndarray,
    biomarkers: tuple[int, ...] = BIOMARKERS,
) -> np.ndarray:
    """Sample the biomarker states at ``obs_times`` (nearest grid index) → ``(n_obs, n_biom)``."""
    idx = np.clip(np.searchsorted(times, obs_times), 0, len(times) - 1)
    return traj[np.ix_(idx, list(biomarkers))]


def generate_cohort(
    *,
    n_subjects: int = 40,
    n_obs_times: int = 8,
    t_max: float = 12.0,
    dt: float = 0.06,
    dose: float = 0.6,
    isv: float = 0.15,
    obs_noise: float = 0.05,
    biomarkers: tuple[int, ...] = BIOMARKERS,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Generate a SYNTHETIC ground-truth cohort: subject-specific parameters + biomarker obs.

    Each subject carries its own kinetics (population truth × ``isv`` log-normal variability) —
    a nonlinear-mixed-effects population, exactly how QSP models are really fit. Returns the
    true per-subject parameters (the ground truth a fit must recover) and the log-biomarker
    observations (with Gaussian ``obs_noise``). NOT real patient data.
    """
    rng = np.random.default_rng(seed)
    base = PARAM_VALUES[None, :]
    subj_p = base * np.exp(isv * rng.standard_normal((n_subjects, PARAM_VALUES.shape[0])))
    obs_times = np.linspace(0.0, t_max, n_obs_times)
    clean = np.empty((n_subjects, n_obs_times, len(biomarkers)), dtype=np.float64)
    for s in range(n_subjects):
        traj, times = simulate_subject(subj_p[s], dose=dose, t_max=t_max, dt=dt)
        obs = observe_biomarkers(traj, times, obs_times, biomarkers)
        clean[s] = np.log(np.clip(obs, 0.0, _X_CAP) + 1e-3)
    noisy = clean + obs_noise * rng.standard_normal(clean.shape) if obs_noise > 0 else clean
    return {
        "true_params": subj_p,                 # (n_subjects, 12) — the ground truth
        "observations": noisy,                 # (n_subjects, n_obs_times, n_biom) log-biomarkers
        "clean_observations": clean,
        "obs_times": obs_times,
        "param_names": np.array(PARAM_NAMES),
        "biomarker_indices": np.array(biomarkers),
        "dose": np.array([dose]),
        "dose_window": np.array(DOSE_WINDOW),
        "isv": np.array([isv]),
        "obs_noise": np.array([obs_noise]),
    }


if __name__ == "__main__":  # a tiny sanity forward-sim
    traj, times = simulate_subject(dose=0.0)
    print("plaque(t=end), no antibody :", round(float(traj[-1, 2]), 4))
    traj_dosed, _ = simulate_subject(dose=0.6)
    print("plaque(t=end), with antibody:", round(float(traj_dosed[-1, 2]), 4),
          "(antibody lowers plaque)")
