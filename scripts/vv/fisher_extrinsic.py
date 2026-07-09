#!/usr/bin/env python3
"""Does EXTRINSIC log-normal parameter spread re-confound the toggle FIM?

Extends scripts/vv/fisher_sloppiness.py (intrinsic-only) with the generator's
EXTRINSIC cell-to-cell variability. Faithful to src/nudge/data/synthetic.py::
_per_cell_params: extrinsic noise multiplies species **basal** and **decay** by a
per-cell log-normal factor exp(sigma * N(0,1)) — a SINGLE shared factor across
species per param (factor shape (n_cells,1) broadcast over the species axis). It
does NOT touch the switch-shape params (K, n) or vmax (edge params). So the
extrinsic latent per cell is 2-dimensional: z_basal, z_decay ~ N(0,1), scaling
ALL species' basal by exp(sigma z_basal) and ALL decays by exp(sigma z_decay).

Model of extrinsic spread on the LNA mixture (first order): the mode MEAN of each
cell shifts by the propagated parameter perturbation. If a cell's log-basal shifts
by (sigma z_b) and log-decay by (sigma z_d), then to first order
  mu_cell ~ mu0 + J_b (sigma z_b) + J_d (sigma z_d),
with J_b = d mu / d log(basal-shared), J_d = d mu / d log(decay-shared) (IFT
sensitivities, same one-Newton-step map as the committed FIM). The marginal
per-mode covariance therefore gains
  Sigma_ext(theta) = sigma^2 (J_b J_b^T + J_d J_d^T),
added to the intrinsic Lyapunov covariance in BOTH data-sampling AND loglik (the
fit KNOWS sigma — modeled nuisance, not hidden). Crucially Sigma_ext depends on
theta=(m,v,K) through the mode-mean map, so the FIM (still taken over ONLY log m,
log v, log K) picks up d Sigma_ext / d log theta.

A Monte-Carlo cross-check (full Newton re-solve per cell) validates the first-order
Sigma_ext at sigma=0.3.

Usage:  uv run python fisher_extrinsic.py
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

from nudge.core.circuit import Circuit, EdgeDef, SpeciesDef

B0, D0 = 0.05, 1.0
M_NOM, V_NOM, K_NOM = 4.0, 2.0, 1.0
M1, V1, K1 = 4.0, 2.0, 1.0
PARAMS = ("log m (gain)", "log v (ceil)", "log K (thr)")


def toggle_circuit(m: float, v: float, K: float, bB: float = B0) -> Circuit:
    return Circuit(
        [SpeciesDef("A", basal=B0, decay=D0), SpeciesDef("B", basal=bB, decay=D0)],
        [
            EdgeDef(1, 0, "hill_repression", K=K, n=m, vmax=v),
            EdgeDef(0, 1, "hill_repression", K=K1, n=M1, vmax=V1),
        ],
    )


def drift(x, theta, eb=0.0, ed=0.0, bB=B0):
    """dx/dt with extrinsic log-shifts eb (shared basal), ed (shared decay).

    basal_A=B0 e^eb, basal_B=bB e^eb, decay=D0 e^ed (both species) — faithful to
    synthetic.py where one lognormal factor scales all species' basal, another all
    decays. theta=(m,v,K) of the perturbed edge (extrinsic does NOT touch these).
    """
    m, v, K = theta
    a, b = x[0], x[1]
    decay = D0 * jnp.exp(ed)
    prod_a = B0 * jnp.exp(eb) + v * K**m / (K**m + b**m)
    prod_b = bB * jnp.exp(eb) + V1 * K1**M1 / (K1**M1 + a**M1)
    return jnp.stack([prod_a, prod_b]) - decay * x


def stable_roots(theta, bB=B0):
    m, v, K = (float(t) for t in theta)
    fps = toggle_circuit(m, v, K, bB).fixed_points()
    return [np.asarray(s, float) for s, lab in (fps or []) if lab == "stable"]


def mode_mean(root, theta, eb=0.0, ed=0.0, bB=B0):
    """IFT-differentiable fixed point (one Newton step; value=root, grad via A^-1)."""
    x0 = jax.lax.stop_gradient(root)
    f = drift(x0, theta, eb, ed, bB)
    jac = jax.jacobian(lambda x: drift(x, theta, eb, ed, bB))(x0)
    return x0 - jnp.linalg.solve(jac, f)


def mode_cov_intrinsic(mu, theta, eb=0.0, ed=0.0, bB=B0):
    """LNA (Lyapunov) covariance A Sig + Sig A^T + D = 0, D=diag(2 decay x*)."""
    jac = jax.jacobian(lambda x: drift(x, theta, eb, ed, bB))(mu)
    decay = D0 * jnp.exp(ed)
    diff = jnp.diag(2.0 * decay * jnp.clip(mu, 1e-6))
    n = jac.shape[0]
    kron = jnp.kron(jnp.eye(n), jac) + jnp.kron(jac, jnp.eye(n))
    sig = jnp.linalg.solve(kron, -diff.reshape(-1)).reshape(n, n)
    return 0.5 * (sig + sig.T)


def extrinsic_jac(root, theta, bB=B0):
    """(J_b, J_d) = d mu / d(eb, ed) at eb=ed=0 — the propagated log-param sens."""
    g = jax.jacobian(lambda e: mode_mean(root, theta, e[0], e[1], bB))(jnp.zeros(2))
    return g[:, 0], g[:, 1]  # each (2,)


def mode_cov_total(root, theta, sigma, bB=B0):
    """Intrinsic Lyapunov cov + first-order extrinsic cov sigma^2 (Jb Jb^T + Jd Jd^T).

    All differentiable in theta (Sigma_ext depends on theta via the mode-mean map).
    """
    mu = mode_mean(root, theta, 0.0, 0.0, bB)
    cov = mode_cov_intrinsic(mu, theta, 0.0, 0.0, bB)
    if sigma > 0:
        jb, jd = extrinsic_jac(root, theta, bB)
        cov = cov + sigma**2 * (jnp.outer(jb, jb) + jnp.outer(jd, jd))
    return 0.5 * (cov + cov.T)


def _mvn_logpdf(x, mu, cov):
    d = x - mu
    _, logdet = jnp.linalg.slogdet(cov)
    quad = d @ jnp.linalg.solve(cov, d)
    return -0.5 * (quad + logdet + mu.shape[0] * jnp.log(2 * jnp.pi))


def make_loglik(roots, sigma, bB=B0):
    roots_j = [jnp.asarray(r) for r in roots]
    logw = jnp.log(jnp.array([1.0 / len(roots_j)] * len(roots_j)))

    def loglik(obs, log_theta):
        theta = jnp.exp(log_theta)
        comps = []
        for r in roots_j:
            mu = mode_mean(r, theta, 0.0, 0.0, bB)
            cov = mode_cov_total(r, theta, sigma, bB)
            comps.append(_mvn_logpdf(obs[:2], mu, cov))
        return jax.scipy.special.logsumexp(logw + jnp.stack(comps))

    return loglik


def sample_data(roots, theta, sigma, n, key, bB=B0):
    tj = jnp.asarray(theta)
    mus = [mode_mean(jnp.asarray(r), tj, 0.0, 0.0, bB) for r in roots]
    covs = [mode_cov_total(jnp.asarray(r), tj, sigma, bB) for r in roots]
    key, ka = jax.random.split(key)
    assign = jax.random.bernoulli(ka, 0.5, (n,)).astype(int)
    xs = []
    for i in range(n):
        key, ks = jax.random.split(key)
        k = int(assign[i])
        xs.append(jax.random.multivariate_normal(ks, mus[k], covs[k]))
    return jnp.stack(xs)


def fisher(roots, theta, sigma, n, key, bB=B0):
    data = sample_data(roots, theta, sigma, n, key, bB)
    loglik = make_loglik(roots, sigma, bB)
    lt0 = jnp.log(jnp.asarray(theta))
    score = jax.vmap(lambda o: jax.grad(loglik, argnums=1)(o, lt0))(data)
    return np.asarray(score.T @ score / n)


def fisher_ms(roots, theta, sigma, n, seeds, bB=B0):
    fims = [fisher(roots, theta, sigma, n, jax.random.PRNGKey(s), bB) for s in seeds]
    return np.mean(fims, axis=0), np.std([np.linalg.eigvalsh(f) for f in fims], axis=0)


def _corr(fim):
    d = np.sqrt(np.diag(fim))
    return fim / np.outer(d, d)


def eig_sorted(fim):
    ev, evec = np.linalg.eigh(fim)
    order = np.argsort(ev)[::-1]
    return ev[order], evec[:, order]


# ---------------------------------------------------------------------------
# Monte-Carlo cross-check: full Newton re-solve of the shifted mode mean.
# ---------------------------------------------------------------------------
def newton_root(theta, eb, ed, start, bB=B0, iters=40):
    x = start
    for _ in range(iters):
        f = drift(x, theta, eb, ed, bB)
        jac = jax.jacobian(lambda z: drift(z, theta, eb, ed, bB))(x)
        x = x - jnp.linalg.solve(jac, f)
    return x


def mc_extrinsic_cov(root, theta, sigma, n, key, bB=B0):
    """Empirical covariance of the TRUE shifted mode mean over log-normal z draws."""
    tj = jnp.asarray(theta)
    z = jax.random.normal(key, (n, 2))
    solve = jax.vmap(
        lambda zz: newton_root(tj, sigma * zz[0], sigma * zz[1], jnp.asarray(root), bB)
    )
    mus = solve(z)  # (n,2) true shifted means
    return np.cov(np.asarray(mus).T)


def report(name, fim):
    ev, evec = eig_sorted(fim)
    print(f"\n=== {name} ===")
    print("FIM eigenvalues (stiff->sloppy): " + "  ".join(f"{e:.3e}" for e in ev))
    print(f"condition number: {ev[0] / ev[-1]:.2e}   smallest eig: {ev[-1]:.3e}")
    for i, lab in enumerate(PARAMS):
        print(f"  {lab:14s} " + "  ".join(f"{evec[i, j]:+.3f}" for j in range(3)))
    print("  FIM correlation (order m, v, K):")
    print(np.array2string(_corr(fim), precision=3, prefix="    "))
    return ev, evec


def main():
    theta0 = (M_NOM, V_NOM, K_NOM)
    roots = stable_roots(jnp.asarray(theta0))
    print(f"nominal toggle: {len(roots)} stable modes")
    if len(roots) != 2:
        raise SystemExit("expected a bistable toggle")
    for r in roots:
        mu = mode_mean(jnp.asarray(r), jnp.asarray(theta0))
        jb, jd = extrinsic_jac(jnp.asarray(r), jnp.asarray(theta0))
        print(
            f"  mu={np.asarray(mu).round(3)}  Jb(dmu/dlog basal)={np.asarray(jb).round(3)}"
            f"  Jd(dmu/dlog decay)={np.asarray(jd).round(3)}"
        )

    n, seeds = 20000, list(range(6))
    sigmas = [0.0, 0.1, 0.2, 0.3, 0.5]
    print(f"\n(FIM avg over {len(seeds)} seeds, N={n} cells)\n")
    print(f"{'sigma':>6} | {'eig_stiff':>10} {'eig_mid':>10} {'eig_sloppy':>10} "
          f"| {'cond#':>9} {'sloppy_std':>10} | {'c(m,K)':>7} {'c(m,v)':>7} {'c(v,K)':>7}")
    print("-" * 100)
    results = {}
    for sig in sigmas:
        fim, sd = fisher_ms(roots, theta0, sig, n, seeds)
        ev, evec = eig_sorted(fim)
        c = _corr(fim)
        results[sig] = (fim, ev, evec, sd)
        # seed-std of the smallest eigenvalue (eigvalsh returns ascending -> index 0)
        sloppy_std = float(np.sort(sd)[0])
        print(f"{sig:>6.1f} | {ev[0]:>10.3e} {ev[1]:>10.3e} {ev[2]:>10.3e} "
              f"| {ev[0] / ev[2]:>9.2e} {sloppy_std:>10.2e} "
              f"| {c[0, 2]:>+7.3f} {c[0, 1]:>+7.3f} {c[1, 2]:>+7.3f}")

    # Detailed eigen-structure at sigma=0 and sigma=0.3.
    report("INTRINSIC ONLY (sigma=0)", results[0.0][0])
    report("EXTRINSIC sigma=0.3 (generator default)", results[0.3][0])

    # Which param dominates the stiff & sloppy eigenvectors at sigma=0.3?
    print("\n--- eigenvector composition at sigma=0.3 ---")
    _, ev, evec, _ = results[0.3]
    stiff, sloppy = evec[:, 0], evec[:, -1]
    print("  STIFFEST  dir: " + ", ".join(
        f"{PARAMS[i]}={stiff[i]:+.2f}" for i in np.argsort(-np.abs(stiff))))
    print("  SLOPPIEST dir: " + ", ".join(
        f"{PARAMS[i]}={sloppy[i]:+.2f}" for i in np.argsort(-np.abs(sloppy))))
    v_load_stiff = abs(evec[1, 0])  # loading of log v on stiff eigenvector
    print(f"  |v loading| on stiffest eigvec: {v_load_stiff:.3f}"
          f"  (v stays identifiable if ~1)")

    # ratio: how much does the smallest eigenvalue move vs intrinsic?
    print("\n--- smallest FIM eigenvalue vs intrinsic ---")
    base = results[0.0][1][-1]
    for sig in sigmas:
        e = results[sig][1][-1]
        print(f"    sigma={sig:>3.1f}: {e:.3e}  (x{e / base:.3f} vs intrinsic)")

    # -------- Monte-Carlo cross-check of first-order Sigma_ext at sigma=0.3 -----
    print("\n--- Monte-Carlo cross-check: first-order vs true Sigma_ext (sigma=0.3) ---")
    sig = 0.3
    tj = jnp.asarray(theta0)
    for k, r in enumerate(roots):
        jb, jd = extrinsic_jac(jnp.asarray(r), tj)
        fo = np.asarray(sig**2 * (jnp.outer(jb, jb) + jnp.outer(jd, jd)))
        mc = mc_extrinsic_cov(jnp.asarray(r), theta0, sig, 40000, jax.random.PRNGKey(100 + k))
        rel = np.linalg.norm(mc - fo) / np.linalg.norm(fo)
        print(f"  mode {k}:")
        print(f"    first-order Sigma_ext diag = {np.diag(fo).round(4)}  offdiag={fo[0,1]:+.4f}")
        print(f"    Monte-Carlo  Sigma_ext diag = {np.diag(mc).round(4)}  offdiag={mc[0,1]:+.4f}")
        print(f"    relative Frobenius error   = {rel:.3%}")


if __name__ == "__main__":
    main()
