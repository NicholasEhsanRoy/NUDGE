"""RED-TEAM (round 7, STOP-gate re-scan): ``adjoint.ode_identifiability`` END-TO-END.

Target: ``nudge.inference.adjoint.ode_identifiability`` (``NUDGE-METHOD`` scaling layer /
``NUDGE-LIM-023``) — the surface the moat sweep (``runs/000000018``) left explicitly
**UNREACHED**: the P6 hole (an iterative Krylov solver missing an isolated Fisher null and
certifying ``well-constrained``) was closed for the abstract ``predict_fn``; this drives the
SAME diagnostic through a REAL ODE trajectory to verify the fix does not reappear end-to-end,
including the ``ODEProblem.dtype`` **float32 default** interaction the mandate flags.

Three genuinely-unidentifiable ODE fits, whose honest verdict is ABSTENTION
(``unidentifiable``):

  A. an ISOLATED EXACT structural null — a linear decay cascade whose species-0 decay rate is
     ``theta[0] + theta[1]`` (two free params enter ONLY via their sum), embedded in an
     otherwise well-conditioned spectrum. n_theta small -> the DENSE reconstruct+``eigh``
     route (``method="auto"``). This is the ODE analogue of the P6 ``redundant_exponential``.
  B. the SAME isolated null at a larger n_theta forced through the GENUINE ITERATIVE route
     (``method="iterative"``), where the inverse-iteration null probe (the P6 fix) must catch
     the isolated null that ``eigsh(which='SA')`` misses.
  C. a diffuse, data-driven rank-deficient gLV community (``make_glv_problem``) — the
     realistic large-network case.

The HOLE this guards against: ``ode_identifiability`` returning ``well-constrained`` /
``sloppy-but-predictive`` (a confident "the model is usable; do not abstain") on any of these.
Run at BOTH ``ODEProblem.dtype=float32`` (the shipped default — the forward model silently
downcasts even when the caller passes a float64 theta) AND ``float64``, across 2 seeds.

Exit 0 = every case abstains (``unidentifiable``) = the P6 fix HOLDS end-to-end (HELD).
Exit 2 = a confident-wrong (a NEW hole). ``uv run python scripts/redteam/…`` (~4 min).
"""

from __future__ import annotations

import sys

import jax

# The careful-caller / shipped-test convention (``pytest.mark.x64``): even so, the forward
# model downcasts to ``ODEProblem.dtype`` (float32 by default), so the FIM still carries
# float32 roundoff — exactly the interaction under test.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from nudge.inference.adjoint import (  # noqa: E402
    ODEProblem,
    make_glv_problem,
    ode_identifiability,
    rk4_integrate,
)

_POSITIVE = {"well-constrained", "sloppy-but-predictive"}


def _sum_null_ode(n_states: int, dtype, seed: int) -> ODEProblem:
    """Linear decay cascade; species-0 decay rate = ``theta[0]+theta[1]`` (an exact null)."""
    rng = np.random.default_rng(seed)
    k = rng.uniform(0.4, 1.6, size=n_states).astype(np.float64)
    theta0 = np.concatenate([[k[0] * 0.5, k[0] * 0.5], k[1:]]).astype(np.float64)

    def field(x: jnp.ndarray, theta: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        k0 = theta[0] + theta[1]
        kk = jnp.concatenate([jnp.reshape(k0, (1,)), theta[2:]])
        inflow = jnp.concatenate([jnp.zeros(1, kk.dtype), kk[:-1] * x[:-1]])
        return inflow - kk * x

    t_max, dt, n_obs = 4.0, 0.05, 30
    n_steps = int(round(t_max / dt))
    obs_idx = np.clip(np.round(np.linspace(0.0, t_max, n_obs) / dt).astype(int), 0, n_steps)
    u_grid = np.zeros(n_steps, dtype=np.float32)
    x0 = np.zeros(n_states, dtype=np.float64)
    x0[0] = 1.0
    target = np.asarray(
        rk4_integrate(
            field, jnp.asarray(x0, dtype), jnp.asarray(theta0, dtype),
            jnp.asarray(u_grid, dtype), dt, jnp.asarray(obs_idx),
        )
    )
    return ODEProblem(
        field=field, x0=x0, u_grid=u_grid, dt=dt, obs_idx=obs_idx,
        target=target.astype(np.float64), theta0=theta0, n_states=n_states, dtype=dtype,
    )


def main() -> int:
    holes = 0
    checks = 0
    for seed in (0, 1):
        for dt_name, dtype in (("float32(DEFAULT)", jnp.float32), ("float64", jnp.float64)):
            # A. isolated exact null, DENSE (auto) route.
            ra = ode_identifiability(_sum_null_ode(10, dtype, seed), sigma=1e-2, method="auto")
            # B. isolated exact null, GENUINE ITERATIVE route (inverse-iteration probe).
            rb = ode_identifiability(
                _sum_null_ode(16, dtype, seed), sigma=1e-2, method="iterative"
            )
            # C. diffuse data-driven rank-deficient gLV.
            rc = ode_identifiability(
                make_glv_problem(n_species=12, n_free=120, n_obs=24, dtype=dtype, seed=seed),
                sigma=1e-2, method="auto",
            )
            for tag, r in (("A-isolated/auto", ra), ("B-isolated/iter", rb),
                           ("C-diffuse-gLV", rc)):
                checks += 1
                bad = r.label in _POSITIVE
                holes += int(bad)
                flag = "  *** HOLE (confident-wrong) ***" if bad else ""
                print(
                    f"seed={seed} {dt_name:>16} {tag:<16}: label={r.label!r} "
                    f"n_null={r.n_null_dims}{flag}"
                )
    print(
        f"\nGROUND TRUTH: unidentifiable (all three ODE fits have a genuine null). "
        f"CONFIDENT-WRONG (well-constrained / sloppy-but-predictive): {holes} / {checks}"
    )
    if holes:
        print("RESULT: HOLE — ode_identifiability certified a genuinely-unidentifiable ODE.")
        return 2
    print("RESULT: HELD — every unidentifiable ODE abstains; the P6 fix holds end-to-end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
