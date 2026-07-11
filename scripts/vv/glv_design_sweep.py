"""Experimental-design sweep for gLV attribution — "what would it take to identify ε?"

The companion to the *directional abstention* (``NUDGE-LIM-020``): once NUDGE tells you
*which* combination the data cannot separate, the natural next question is **what sampling
would separate it**. This script answers it MEASURED, on synthetic ground truth — it does
NOT change NUDGE's math or add regularization to force a fit (the abstention is correct);
it changes the **experiment** and watches the identifiability move.

Design: a known antibiotic-**susceptibility** (ε) perturbation on a target taxon under a
fixed antibiotic pulse ``u(t)=1`` on ``pulse_window``, with the total time span and the
out-of-pulse "backbone" observations **held fixed**. The one thing swept is the number of
observations placed **inside the pulse window** — ``0, 1, 2, 4, 8, 16``. The ε signature is
a time-localized on/off contrast that lives *inside* the pulse; without in-pulse samples the
direct kill is unobserved and NUDGE must abstain, and as in-pulse density rises ε becomes
identifiable and the call flips to a confident, correct ``susceptibility``.

At each density we record: the call, the α⇄βᵢᵢ Laplace condition number, whether ε
resolved, and — the load-bearing honesty check — whether any call was **confident-wrong**
(a resolved knob that is NOT the true ε). The guarantee is *resolve-correctly-or-abstain*,
never a mis-attribution, at every density.

Reuses the SHIPPED :mod:`nudge.inference.lotka_volterra` read-only (the same
``attribute_glv``); only the observation grid is bespoke. ``scripts/vv/`` throwaway grain.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from nudge.inference.lotka_volterra import (
    GLVDataset,
    GLVParams,
    _default_baseline,
    _fine_grid,
    _pulse,
    attribute_glv,
    simulate_glv,
)

_POSITIVE = {"growth", "interaction", "susceptibility"}
DEFAULT_DENSITIES: tuple[int, ...] = (0, 1, 2, 4, 8, 16)


# --------------------------------------------------------------------------- #
# a density-controlled dataset builder (fixed span + backbone, swept in-pulse)
# --------------------------------------------------------------------------- #
def build_density_dataset(
    n_pulse: int,
    *,
    n_species: int = 3,
    n_replicates: int = 40,
    target: int = 0,
    delta: float = -1.4,
    t_max: float = 12.0,
    pulse_window: tuple[float, float] = (4.0, 6.0),
    dt: float = 0.02,
    param_noise: float = 0.04,
    obs_noise: float = 0.05,
    coupling: float = 0.0,
    baseline: GLVParams | None = None,
    seed: int = 0,
) -> GLVDataset:
    """A reference (ε=0) vs perturbed (ε=``delta`` on ``target``) pair whose observation
    grid has exactly ``n_pulse`` samples INSIDE ``pulse_window`` on a FIXED out-of-pulse
    backbone and fixed span ``[0, t_max]``.

    The default baseline is a **decoupled** logistic community (``coupling=0`` off-diagonal
    β): each taxon has its own carrying capacity, so after the pulse the killed target
    regrows to *exactly* its pre-pulse equilibrium and the out-of-pulse contrast vanishes to
    noise. That isolation is deliberate — it makes the ε signature live ONLY inside the
    pulse, so in-pulse sampling density is the single thing that flips ε from unidentifiable
    to identifiable (in a strongly-coupled community the post-pulse competitive-release
    reshuffle would leak ε into the out-of-pulse tail and blur the threshold). Set
    ``coupling>0`` to study that leakage. Mirrors
    :func:`nudge.inference.lotka_volterra.simulate_glv_perturbseq`'s ensemble noise model
    verbatim; only the observation-time construction is bespoke.
    """
    rng = np.random.default_rng(seed)
    if baseline is not None:
        base = baseline
    elif coupling == 0.0:
        alpha = rng.uniform(0.6, 1.0, size=n_species)
        beta = np.diag(-rng.uniform(0.8, 1.2, size=n_species))
        base = GLVParams(alpha=alpha, beta=beta, eps=np.zeros(n_species))
    else:
        base = _default_baseline(n_species, rng)
    t_on, t_off = pulse_window

    # FIXED out-of-pulse backbone: pre-pulse transient (captures the growth climb + α/β) and
    # a LATE tail sampled only AFTER the community has fully recovered to its carrying
    # capacity. The immediate post-pulse recovery window (6..9) is deliberately UNSAMPLED so
    # the ε kill's transient aftermath is not captured out-of-pulse — the ε signature then
    # lives ONLY inside the pulse, making in-pulse density the load-bearing variable. Held
    # identical across the sweep so in-pulse density is the single thing that changes.
    backbone = np.array(
        [0.0, 0.5, 1.0, 2.0, 3.0, 3.8,   # pre-pulse transient (α/β) + approach to eq
         9.0, 10.5, t_max],              # LATE tail — post-recovery equilibrium only
        dtype=float,
    )
    if n_pulse > 0:
        # interior points of (t_on, t_off), evenly spaced (never on the boundaries).
        frac = (np.arange(n_pulse) + 0.5) / n_pulse
        in_pulse = t_on + frac * (t_off - t_on)
    else:
        in_pulse = np.array([], dtype=float)
    t_obs = np.unique(np.concatenate([backbone, in_pulse]))

    grid_t, obs_idx = _fine_grid(t_max, dt, t_obs)
    u_grid = _pulse(grid_t[:-1], t_on, t_off)

    # susceptibility perturbation: reference insusceptible (ε=0), perturbed hit on target.
    pert = base.with_knob("susceptibility", target, delta)
    x0 = rng.uniform(0.3, 0.7, size=n_species).astype(np.float32)

    def ensemble(p: GLVParams) -> np.ndarray:
        trajs = []
        for _ in range(n_replicates):
            a = p.alpha * (1.0 + param_noise * rng.standard_normal(n_species))
            b = p.beta * (1.0 + param_noise * rng.standard_normal((n_species, n_species)))
            e = p.eps + param_noise * rng.standard_normal(n_species)
            x0r = x0 * (1.0 + param_noise * rng.standard_normal(n_species))
            traj = np.asarray(
                simulate_glv(
                    (jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32),
                     jnp.asarray(e, jnp.float32)),
                    jnp.asarray(np.clip(x0r, 1e-3, None), jnp.float32),
                    jnp.asarray(u_grid), dt, jnp.asarray(obs_idx),
                )
            )
            traj = traj * np.exp(obs_noise * rng.standard_normal(traj.shape))
            trajs.append(traj)
        return np.stack(trajs)

    reference = ensemble(base)
    perturbed = ensemble(pert)
    ground_truth = {
        "mechanism": "susceptibility",
        "target": target,
        "delta": float(delta),
        "n_pulse": int(n_pulse),
        "pulse_window": pulse_window,
    }
    return GLVDataset(
        reference=reference, perturbed=perturbed, t_obs=t_obs, u_grid=u_grid,
        obs_idx=obs_idx, dt=dt, baseline=base, ground_truth=ground_truth,
    )


# --------------------------------------------------------------------------- #
# the sweep
# --------------------------------------------------------------------------- #
@dataclass
class DensityPoint:
    """Aggregated outcome at one in-pulse sampling density."""

    n_pulse: int
    n_obs_total: int
    resolve_rate: float          # fraction of seeds that resolve the true ε
    calls: list[str]
    cond_numbers: list[float]
    median_cond: float
    confident_wrong: int         # count of mis-attributions (must be 0)


@dataclass
class SweepResult:
    densities: list[DensityPoint] = field(default_factory=list)
    threshold: int | None = None            # min in-pulse count that resolves ε for ALL seeds
    n_confident_wrong: int = 0
    seeds: list[int] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "n_confident_wrong": self.n_confident_wrong,
            "seeds": self.seeds,
            "meta": self.meta,
            "densities": [
                {
                    "n_pulse": d.n_pulse,
                    "n_obs_total": d.n_obs_total,
                    "resolve_rate": d.resolve_rate,
                    "calls": d.calls,
                    "cond_numbers": d.cond_numbers,
                    "median_cond": d.median_cond,
                    "confident_wrong": d.confident_wrong,
                }
                for d in self.densities
            ],
        }


def run_sweep(
    *,
    densities: tuple[int, ...] = DEFAULT_DENSITIES,
    seeds: tuple[int, ...] = (0, 1, 2),
    n_species: int = 3,
    n_replicates: int = 40,
    steps: int = 140,
    n_sim: int = 30,
    delta: float = -1.4,
    verbose: bool = True,
) -> SweepResult:
    """Run the in-pulse density sweep and locate the ε-identifiability threshold.

    For each density × seed: build the dataset, ``attribute_glv``, and record whether the
    true ``susceptibility`` was resolved, the α⇄βᵢᵢ condition number, and any confident-
    wrong call. The **threshold** is the smallest in-pulse count at which ε resolves for
    every seed. Returns a :class:`SweepResult`.
    """
    result = SweepResult(seeds=list(seeds), meta={
        "n_species": n_species, "n_replicates": n_replicates, "steps": steps,
        "n_sim": n_sim, "delta": delta, "truth": "susceptibility",
    })
    total_cw = 0
    for n_pulse in densities:
        calls: list[str] = []
        conds: list[float] = []
        cw = 0
        n_obs_total = 0
        for seed in seeds:
            ds = build_density_dataset(
                n_pulse, n_species=n_species, n_replicates=n_replicates,
                delta=delta, seed=seed,
            )
            n_obs_total = len(ds.t_obs)
            t0 = time.time()
            res = attribute_glv(ds, steps=steps, n_sim=n_sim, seed=seed)
            calls.append(res.call)
            conds.append(float(res.fit.cond_number))
            # confident-wrong: a resolved knob that is NOT the true ε.
            if res.call in _POSITIVE and res.call != "susceptibility":
                cw += 1
            if verbose:
                print(f"  n_pulse={n_pulse:>2}  seed={seed}  -> {res.call:<14} "
                      f"cond={res.fit.cond_number:>8.1f}  ({time.time()-t0:.1f}s)",
                      flush=True)
        resolve_rate = float(np.mean([c == "susceptibility" for c in calls]))
        finite = [c for c in conds if np.isfinite(c)]
        median_cond = float(np.median(finite)) if finite else float("inf")
        result.densities.append(DensityPoint(
            n_pulse=n_pulse, n_obs_total=n_obs_total, resolve_rate=resolve_rate,
            calls=calls, cond_numbers=conds, median_cond=median_cond,
            confident_wrong=cw,
        ))
        total_cw += cw
        if verbose:
            print(f"  == n_pulse={n_pulse}: resolve_rate={resolve_rate:.2f} "
                  f"median_cond={median_cond:.1f} confident_wrong={cw}\n", flush=True)

    result.n_confident_wrong = total_cw
    # threshold: smallest density that resolves ε for ALL seeds.
    for d in result.densities:
        if d.resolve_rate >= 1.0:
            result.threshold = d.n_pulse
            break
    return result


def main() -> None:
    print("gLV experimental-design sweep — in-pulse sampling density vs ε identifiability")
    res = run_sweep()
    print(f"\nThreshold (≥N in-pulse samples resolve ε for all seeds): {res.threshold}")
    print(f"Confident-wrong across the whole sweep: {res.n_confident_wrong}")
    out = Path(__file__).resolve().parent / "glv_design_sweep_RESULTS.json"
    out.write_text(json.dumps(res.to_dict(), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
