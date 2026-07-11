"""RED-TEAM (round 7, STOP-gate re-scan): OED — false-precision / silent-regression.

Target: ``nudge.inference.oed`` (``NUDGE-METHOD-014`` / ``NUDGE-LIM-024``) — the second
surface the moat sweep (``runs/000000018``) left UNREACHED, probed on both the SHIPPED wiring
(``nudge.service.oed_demo`` == the ``nudge oed`` CLI) and the library ``optimize_design``.

Three attacks, each with truth = "this design does NOT resolve the target" (abstention):

  1. SHIPPED-DEMO REGRESSION / silent worse-than-init. Sweep ``oed_demo`` over
     model x objective x n_obs x steps x seed. A CONFIDENT-WRONG headline would be a reported
     ``crlb_improvement < 1`` (the CLI prints "0.X× better" + the "breaks the α⇄βᵢᵢ tie"
     narrative) OR a masked target CRLB (``target_crlb_opt`` small while ``min_eig_opt`` ≈ 0).
     Expect: none — the logistic/glv showcase genuinely improves.
  2. LAST-ITERATE HONESTY (``optimize_design`` returns the LAST Adam iterate, not
     best-of-history). Drive it with an aggressive learning rate and check the reported
     ``crlb_improvement`` is ALWAYS the honest measured ratio at the returned design (never a
     false >1 while the design is actually worse).
  3. GUARDED-RIDGE MASKING on a STRUCTURALLY degenerate target. A target parameter that enters
     the model only via a sum (unidentifiable at EVERY design): the relative ridge in ``crlb``
     can inflate ``crlb_improvement``, but the honest channels — the raw ``min_eig_opt`` and the
     ABSOLUTE ``target_crlb_opt`` — must still expose the degeneracy (min_eig ≈ 0, CRLB huge).
     A confident-wrong would be ``min_eig_opt`` deceptively far from 0 on a degenerate target.

Exit 0 = every invariant HOLDS (OED never emits a confident, false resolution) = HELD.
Exit 2 = a confident-wrong. Deterministic (fixed seeds). ``uv run python …`` (~4 min).
"""

from __future__ import annotations

import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from nudge.inference.oed import (  # noqa: E402
    DesignProblem,
    crlb,
    fisher_information,
    make_logistic_design_problem,
    optimize_design,
)
from nudge.service import oed_demo  # noqa: E402

_A, _OFF = 1.0, 1e-2


def _attack1_shipped_demo() -> int:
    """Sweep the shipped demo; flag crlb_improvement<1 or a masked (small-crlb, ~0-min-eig)."""
    holes = 0
    n = 0
    for model in ("logistic", "glv"):
        for obj in ("d_opt", "a_opt", "e_opt", "crlb"):
            for n_obs in (4, 8, 16):
                for steps in (50, 400):
                    for seed in (0, 1, 2):
                        n += 1
                        out = oed_demo(model=model, objective=obj, n_obs=n_obs,
                                       steps=steps, seed=seed)
                        ci = out["crlb_improvement"]
                        masked = out["target_crlb_opt"] < 1.0 and out["min_eig_opt"] < 1e-6
                        if ci < 1.0 or masked:
                            holes += 1
                            print(f"  HOLE demo model={model} obj={obj} n_obs={n_obs} "
                                  f"steps={steps} seed={seed} crlb_impr={ci:.3f} "
                                  f"min_eig_opt={out['min_eig_opt']:.2e}")
    print(f"[1] shipped oed_demo: {n} configs, {holes} confident-wrong "
          f"(crlb_improvement<1 or masked crlb)")
    return holes


def _attack2_last_iterate() -> int:
    """optimize_design must report the HONEST measured improvement at the returned design."""
    prob = make_logistic_design_problem()
    _lo, hi = prob.phi_bounds
    tidx = prob.param_index("log_alpha")
    holes = 0
    for lr in (0.5, 2.0, 5.0, 20.0):
        for seed in (0, 1):
            naive = np.linspace(0.6 * hi, hi, 8)
            res = optimize_design(prob, naive, objective="crlb", target="log_alpha",
                                  steps=200, learning_rate=lr, seed=seed)
            true_ratio = (float(crlb(fisher_information(prob, res.phi_init))[tidx])
                          / float(crlb(fisher_information(prob, res.phi_opt))[tidx]))
            # a FALSE claim = reported an improvement (>1.05) while the true ratio is a
            # regression (<0.95). The reported ratio must track the measured one.
            if res.crlb_improvement > 1.05 and true_ratio < 0.95:
                holes += 1
                print(f"  HOLE last-iterate lr={lr} seed={seed} "
                      f"reported={res.crlb_improvement:.3f} true={true_ratio:.3f}")
    print(f"[2] optimize_design last-iterate honesty: {holes} false-improvement claims")
    return holes


def _sum_degenerate_problem(seed: int) -> DesignProblem:
    rng = np.random.default_rng(seed)
    k1, k2 = rng.uniform(0.3, 0.6), rng.uniform(0.3, 0.6)

    def observe(theta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
        ksum = jnp.exp(theta[0]) + jnp.exp(theta[1])  # k1,k2 enter ONLY via their sum
        return jnp.log(jnp.clip(_A * jnp.exp(-ksum * phi) + _OFF, 1e-6, None))

    return DesignProblem(observe=observe,
                         theta0=np.array([np.log(k1), np.log(k2)], dtype=np.float64),
                         param_names=("log_k1", "log_k2"), sigma=0.05,
                         phi_bounds=(0.02, 12.0))


def _attack3_ridge_masking() -> int:
    """On a structurally-degenerate target, min_eig_opt (raw) must still flag it (~0)."""
    holes = 0
    for obj in ("crlb", "e_opt", "a_opt", "d_opt"):
        for seed in (0, 1, 2):
            prob = _sum_degenerate_problem(seed)
            res = optimize_design(prob, np.linspace(7.2, 12.0, 8), objective=obj,
                                  target="log_k1", steps=400, learning_rate=0.2, seed=seed)
            # The honest channels: raw smallest eigenvalue ~0 AND absolute CRLB huge.
            min_eig_flags = res.min_eig_opt < 1e-6
            abs_crlb_huge = res.target_crlb_opt > 1e2
            if not (min_eig_flags and abs_crlb_huge):
                holes += 1
                print(f"  HOLE ridge-mask obj={obj} seed={seed} "
                      f"min_eig_opt={res.min_eig_opt:.2e} "
                      f"target_crlb_opt={res.target_crlb_opt:.3g} "
                      f"crlb_improvement={res.crlb_improvement:.2f}")
    print(f"[3] guarded-ridge masking (min_eig + absolute CRLB honest): {holes} deceptive")
    return holes


def main() -> int:
    total = _attack1_shipped_demo() + _attack2_last_iterate() + _attack3_ridge_masking()
    print(f"\nCONFIDENT-WRONG total across the three OED attacks: {total}")
    if total:
        print("RESULT: HOLE — OED emitted a confident, false resolution.")
        return 2
    print("RESULT: HELD — OED never faked a resolution; min_eig + absolute CRLB stay honest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
