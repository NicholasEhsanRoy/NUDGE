#!/usr/bin/env python3
"""Fisher-information / sloppiness analysis of the toggle attribution degeneracy.

Turns the *asserted* gain/ceiling/threshold confound of a 2-node toggle switch
(``design/TOGGLE_ATTRIBUTION_RESEARCH.md``) into a *measured* one: is any pair of
mechanisms formally unidentifiable from the steady-state snapshot — do they align on a
near-zero ("sloppy") eigenvector of the Fisher Information Matrix — and which extra
observable orthogonalizes them?

Model: the toggle's **linear-noise Gaussian mixture**. Each cell's (A, B) is drawn from a
2-component mixture whose mode means are the deterministic stable fixed points (made
differentiable via an implicit-function-theorem one-Newton-step with a stop-gradient) and
whose mode covariances solve the Lyapunov equation ``A Σ + Σ Aᵀ + D = 0`` (A = the autodiff
drift Jacobian at the mode, D = diag(2 d x*) birth-death diffusion). Parameters are the
perturbed edge's ``(m, v, K)`` = (gain, ceiling, threshold), analysed in **log** space
(Gutenkunst sloppiness convention → dimensionless eigenvalues). The FIM is the
empirical/observed Fisher — mean outer product of per-cell scores via
``jax.vmap(jax.grad(loglik))`` on cells sampled from the model, exact in expectation at θ0,
averaged over seeds.

MEASURED RESULT (see FINDINGS.md; corrects the medium-confidence literature synthesis):
  * The sloppy direction is gain(m) <-> threshold(K), NOT gain<->ceiling
    (corr(log m, log K) = -0.99; ceiling is the MOST identifiable parameter).
  * Analytic root: the high-repressor Hill term is (K/B)^m, so the snapshot constrains
    only the combination m*ln(K/B).
  * A constitutive control (reads ceiling v) does NOT break it (smallest eigenvalue x1.0);
    a SECOND OPERATING POINT (dose/basal shift) does (x16).

Usage:  uv run python scripts/vv/fisher_sloppiness.py
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")  # x64 before jax imports (Lyapunov stability)

import jax
import jax.numpy as jnp
import numpy as np

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef

B0, D0 = 0.05, 1.0  # basal, decay (both species)
M_NOM, V_NOM, K_NOM = 4.0, 2.0, 1.0  # perturbed edge (B --| A): gain, ceiling, threshold
M1, V1, K1 = 4.0, 2.0, 1.0  # fixed edge (A --| B)
DECAY = jnp.array([D0, D0])
PARAMS = ("log m (gain)", "log v (ceil)", "log K (thr)")


def toggle_circuit(m: float, v: float, K: float, bB: float = B0) -> Circuit:
    """Toggle with edge-0 (B represses A) set to (m, v, K); edge-1 fixed; basal-B = bB."""
    return Circuit(
        [SpeciesDef("A", basal=B0, decay=D0), SpeciesDef("B", basal=bB, decay=D0)],
        [
            EdgeDef(1, 0, "hill_repression", K=K, n=m, vmax=v),
            EdgeDef(0, 1, "hill_repression", K=K1, n=M1, vmax=V1),
        ],
    )


def drift(x: jax.Array, theta: jax.Array, bB: float = B0) -> jax.Array:
    """dx/dt; theta = (m, v, K) of the perturbed edge; bB = operating-point knob."""
    m, v, K = theta
    a, b = x[0], x[1]
    prod_a = B0 + v * K**m / (K**m + b**m)
    prod_b = bB + V1 * K1**M1 / (K1**M1 + a**M1)
    return jnp.stack([prod_a, prod_b]) - DECAY * x


def stable_roots(theta: jax.Array, bB: float = B0) -> list[np.ndarray]:
    m, v, K = (float(t) for t in theta)
    fps = toggle_circuit(m, v, K, bB).fixed_points()
    return [np.asarray(s, float) for s, lab in (fps or []) if lab == "stable"]


def mode_mean(root: jax.Array, theta: jax.Array, bB: float = B0) -> jax.Array:
    """IFT-differentiable fixed point: value = root, grad = -A^{-1} ∂f/∂θ."""
    x0 = jax.lax.stop_gradient(root)
    f = drift(x0, theta, bB)
    jac = jax.jacobian(lambda x: drift(x, theta, bB))(x0)
    return x0 - jnp.linalg.solve(jac, f)


def mode_cov(mu: jax.Array, theta: jax.Array, bB: float = B0) -> jax.Array:
    """LNA covariance via the Lyapunov equation A Σ + Σ Aᵀ + D = 0 (Kronecker solve)."""
    jac = jax.jacobian(lambda x: drift(x, theta, bB))(mu)
    diff = jnp.diag(2.0 * DECAY * jnp.clip(mu, 1e-6))
    n = jac.shape[0]
    kron = jnp.kron(jnp.eye(n), jac) + jnp.kron(jac, jnp.eye(n))
    sig = jnp.linalg.solve(kron, -diff.reshape(-1)).reshape(n, n)
    return 0.5 * (sig + sig.T)


def _mvn_logpdf(x: jax.Array, mu: jax.Array, cov: jax.Array) -> jax.Array:
    d = x - mu
    _, logdet = jnp.linalg.slogdet(cov)
    quad = d @ jnp.linalg.solve(cov, d)
    return -0.5 * (quad + logdet + mu.shape[0] * jnp.log(2 * jnp.pi))


def make_loglik(roots, bB=B0, *, constitutive=False):
    """loglik(obs, log_theta): the 2-component LNA mixture (+ optional constitutive dim)."""
    roots_j = [jnp.asarray(r) for r in roots]
    logw = jnp.log(jnp.array([1.0 / len(roots_j)] * len(roots_j)))

    def loglik(obs: jax.Array, log_theta: jax.Array) -> jax.Array:
        theta = jnp.exp(log_theta)
        comps = []
        for r in roots_j:
            mu = mode_mean(r, theta, bB)
            comps.append(_mvn_logpdf(obs[:2], mu, mode_cov(mu, theta, bB)))
        ll = jax.scipy.special.logsumexp(logw + jnp.stack(comps))
        if constitutive:  # z ~ N((b0+v)/d, ...) reads the un-repressed ceiling only
            z_mean = (B0 + theta[1]) / D0
            z_var = 2.0 * D0 * z_mean
            ll = ll - 0.5 * ((obs[2] - z_mean) ** 2 / z_var + jnp.log(2 * jnp.pi * z_var))
        return ll

    return loglik


def sample_data(roots, theta, n, key, bB=B0, *, constitutive=False):
    """Sample n cells from the LNA mixture at theta (concrete)."""
    tj = jnp.asarray(theta)
    mus = [mode_mean(jnp.asarray(r), tj, bB) for r in roots]
    covs = [mode_cov(m, tj, bB) for m in mus]
    key, ka, kz = jax.random.split(key, 3)
    assign = jax.random.bernoulli(ka, 0.5, (n,)).astype(int)
    xs = []
    for i in range(n):
        key, ks = jax.random.split(key)
        k = int(assign[i])
        xs.append(jax.random.multivariate_normal(ks, mus[k], covs[k]))
    x = jnp.stack(xs)
    if constitutive:
        z_mean = (B0 + theta[1]) / D0
        z = z_mean + jnp.sqrt(2.0 * D0 * z_mean) * jax.random.normal(kz, (n,))
        x = jnp.concatenate([x, z[:, None]], axis=1)
    return x


def fisher(roots, theta, n, key, bB=B0, *, constitutive=False) -> np.ndarray:
    """Empirical FIM over (log m, log v, log K) at theta0."""
    data = sample_data(roots, theta, n, key, bB, constitutive=constitutive)
    loglik = make_loglik(roots, bB, constitutive=constitutive)
    lt0 = jnp.log(jnp.asarray(theta))
    score = jax.vmap(lambda o: jax.grad(loglik, argnums=1)(o, lt0))(data)
    return np.asarray(score.T @ score / n)


def fisher_ms(roots, theta, n, seeds, bB=B0, *, constitutive=False):
    """FIM averaged over seeds + the eigenvalue seed-std (a stability check)."""
    fims = [
        fisher(roots, theta, n, jax.random.PRNGKey(s), bB, constitutive=constitutive)
        for s in seeds
    ]
    return np.mean(fims, axis=0), np.std([np.linalg.eigvalsh(f) for f in fims], axis=0)


def _corr(fim: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(fim))
    return fim / np.outer(d, d)


def report(name: str, fim: np.ndarray):
    evals, evecs = np.linalg.eigh(fim)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    print(f"\n=== {name} ===")
    print("FIM eigenvalues (stiff -> sloppy): " + "  ".join(f"{e:.3e}" for e in evals))
    print(f"condition number: {evals[0] / evals[-1]:.2e}")
    for i, lab in enumerate(PARAMS):
        print(f"  {lab:14s} " + "  ".join(f"{evecs[i, j]:+.3f}" for j in range(len(evals))))
    sl = evecs[:, -1]
    order2 = np.argsort(-np.abs(sl))
    print("SLOPPIEST dir: " + ", ".join(f"{PARAMS[i]}={sl[i]:+.2f}" for i in order2))
    return evals, evecs


def main() -> None:
    theta0 = (M_NOM, V_NOM, K_NOM)
    roots = stable_roots(jnp.asarray(theta0))
    print(f"nominal toggle: {len(roots)} stable modes")
    for r in roots:
        mu = mode_mean(jnp.asarray(r), jnp.asarray(theta0))
        cov = np.asarray(mode_cov(mu, jnp.asarray(theta0)))
        print(f"  mu={np.asarray(mu).round(3)}  cov diag={np.diag(cov).round(3)}")
    if len(roots) != 2:
        raise SystemExit("expected a bistable toggle at nominal params")

    print("\n--- mode-mean sensitivities d mu / d log(param)  [dA, dB per mode] ---")
    lt = jnp.log(jnp.asarray(theta0))
    for j, name in enumerate(PARAMS):
        cells = []
        for k, r in enumerate(roots):
            g = jax.jacobian(lambda x, r=r: mode_mean(jnp.asarray(r), jnp.exp(x)))(lt)
            cells.append(f"mode{k}:({float(g[0, j]):+.3f},{float(g[1, j]):+.3f})")
        print(f"  {name:14s} " + "  ".join(cells))

    n, seeds = 20000, list(range(6))
    fim2d, sd = fisher_ms(roots, theta0, n, seeds, constitutive=False)
    print(f"\n(FIM avg over {len(seeds)} seeds, N={n}; sloppy-eig seed-std={np.sort(sd)[0]:.1e})")
    ev2, _ = report("SNAPSHOT ONLY (A,B)", fim2d)
    print("  FIM correlation (pairwise identifiability), order = m, v, K:")
    print(np.array2string(_corr(fim2d), precision=3, prefix="    "))

    fim3d, _ = fisher_ms(roots, theta0, n, seeds, constitutive=True)
    ev3, _ = report("+ CONSTITUTIVE CONTROL (reads ceiling v)", fim3d)

    bb2 = 0.3  # a 6x operating-point (basal-B) shift; stays bistable
    roots2 = stable_roots(jnp.asarray(theta0), bb2)
    fim_b, _ = fisher_ms(roots2, theta0, n, seeds, bb2, constitutive=False)
    ev4, _ = report(f"TWO OPERATING POINTS (bB={B0} + bB={bb2}; Fisher additive)", fim2d + fim_b)

    print("\n--- what stiffens the sloppy gain<->threshold direction? ---")
    print("  smallest FIM eigenvalue:")
    print(f"    snapshot only            : {ev2[-1]:.3e}")
    print(f"    + constitutive control   : {ev3[-1]:.3e}  (x{ev3[-1] / ev2[-1]:.2f})  reads v")
    print(f"    + second operating point : {ev4[-1]:.3e}  (x{ev4[-1] / ev2[-1]:.2f})  breaks m<->K")
    print(
        f"  corr(log m, log K): snapshot={_corr(fim2d)[0, 2]:+.3f}"
        f" -> two-point={_corr(fim2d + fim_b)[0, 2]:+.3f}"
    )


if __name__ == "__main__":
    main()
