"""Identifiability spike: can we separate circuit vs readout nonlinearity?

Exp1: single population, joint fit of (theta_circuit, phi_readout). Degenerate?
Exp2: add a constitutive control (reporter driven at known activity doses).
      Does it break the degeneracy and recover the true circuit nonlinearity?

Run: uv run python spike_ident_run.py   (from repo root)
"""
import functools
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import jax
import jax.numpy as jnp
import optax

import model as M

jax.config.update("jax_enable_x64", True)

# ---------------- ground truth ----------------
# circuit (real switch): K, n, vmax, basal
# readout (saturating) : Km, h, Vmax, b
TRUE = dict(K=1.0, n=3.0, vmax=1.0, basal=0.05,
            Km=0.5, h=2.0, Vmax=20.0, b=0.10)
PHI = 6.0
N_CELLS = 600
FIT_NAMES = ["K", "n", "vmax", "Km", "h", "Vmax"]   # fitted (log-space)
FLOORS = dict(basal=TRUE["basal"], b=TRUE["b"])      # known floors

def pack_theta(fit_vec):
    """fit_vec (log-space, len6) -> full theta tuple for forward model."""
    K, n, vmax, Km, h, Vmax = jnp.exp(fit_vec)
    return (K, n, vmax, FLOORS["basal"], Km, h, Vmax, FLOORS["b"])

def true_fit_vec():
    return jnp.log(jnp.array([TRUE[k] for k in FIT_NAMES]))

# ---------------- data ----------------
KEY = jax.random.PRNGKey(0)
k_u_data, k_y_data, k_u_mod, k_y_mod, k_ctrl = jax.random.split(KEY, 5)

U_DATA = M.gen_input(k_u_data, N_CELLS)
theta_true = tuple(jnp.array(v) for v in
                   (TRUE["K"], TRUE["n"], TRUE["vmax"], TRUE["basal"],
                    TRUE["Km"], TRUE["h"], TRUE["Vmax"], TRUE["b"]))
Y_DATA, A_DATA, LAM_DATA = M.forward_counts(k_y_data, U_DATA, theta_true, PHI)
YL_DATA = M.log1p(Y_DATA)

# Fixed model simulation draws (common random numbers -> smooth loss surface)
U_MOD = M.gen_input(k_u_mod, N_CELLS)
Z_MOD = jax.random.normal(k_y_mod, (N_CELLS,))

def model_counts(fit_vec):
    """Deterministic-given-fixed-draws model counts (reparam noise)."""
    theta = pack_theta(fit_vec)
    K, n, vmax, basal, Km, h, Vmax, b = theta
    a = M.circuit(U_MOD, K, n, vmax, basal)
    Lam = M.readout(a, Km, h, Vmax, b)
    var = Lam + Lam ** 2 / PHI
    y = jnp.clip(Lam + jnp.sqrt(var) * Z_MOD, 0.0)
    return y

# ---------------- constitutive control ----------------
# Reporter driven at known activity doses a_c (bypasses circuit), observe R(a_c).
N_DOSE = 10
N_REP = 200
A_DOSE = jnp.linspace(TRUE["basal"], TRUE["vmax"] + TRUE["basal"], N_DOSE)

def _ctrl_counts(fit_vec, z):
    _, _, _, _, Km, h, Vmax, b = pack_theta(fit_vec)
    Lam = M.readout(A_DOSE[:, None], Km, h, Vmax, b)          # (N_DOSE,1)
    var = Lam + Lam ** 2 / PHI
    return jnp.clip(Lam + jnp.sqrt(var) * z, 0.0)             # (N_DOSE,N_REP)

Z_CTRL_DATA = jax.random.normal(k_ctrl, (N_DOSE, N_REP))
Z_CTRL_MOD = jax.random.normal(jax.random.fold_in(k_ctrl, 1), (N_DOSE, N_REP))
CTRL_DATA = _ctrl_counts(true_fit_vec(), Z_CTRL_DATA)
CTRL_DATA_L = M.log1p(CTRL_DATA)

def ctrl_loss(fit_vec):
    mod = M.log1p(_ctrl_counts(fit_vec, Z_CTRL_MOD))
    # sum of per-dose energy distances (reporter calibration curve)
    def per(i):
        return M.energy_distance(mod[i], CTRL_DATA_L[i])
    return jnp.mean(jax.vmap(per)(jnp.arange(N_DOSE)))

# ---------------- losses ----------------
def circuit_pop_loss(fit_vec):
    return M.energy_distance(M.log1p(model_counts(fit_vec)), YL_DATA)

def joint_loss(fit_vec, use_ctrl, w_ctrl=3.0):
    L = circuit_pop_loss(fit_vec)
    return L + w_ctrl * ctrl_loss(fit_vec) if use_ctrl else L

# ---------------- optimizer ----------------
def fit(fit_vec0, use_ctrl, steps=2500, lr=0.02, fixed_mask=None, fixed_vals=None):
    """Adam fit. fixed_mask: bool array len6, True=frozen at fixed_vals."""
    lossfn = functools.partial(joint_loss, use_ctrl=use_ctrl)
    if fixed_mask is not None:
        fixed_mask = jnp.asarray(fixed_mask)
        def project(v):
            return jnp.where(fixed_mask, fixed_vals, v)
        def obj(free):
            return lossfn(project(free))
    else:
        project = lambda v: v
        obj = lossfn
    opt = optax.adam(lr)
    v = fit_vec0
    st = opt.init(v)
    vg = jax.jit(jax.value_and_grad(obj))
    for _ in range(steps):
        l, g = vg(v)
        upd, st = opt.update(g, st)
        v = optax.apply_updates(v, upd)
    vfull = project(v)
    return vfull, float(joint_loss(vfull, use_ctrl))

# ---------------- experiments ----------------
def multistart(use_ctrl, n_init=40, seed=1):
    rng = np.random.default_rng(seed)
    tv = np.array(true_fit_vec())
    rows = []
    for i in range(n_init):
        # init: truth perturbed by large log-space noise (wide basins)
        v0 = jnp.array(tv + rng.normal(0, 1.0, size=6))
        vf, lf = fit(v0, use_ctrl)
        rec = np.exp(np.array(vf))
        rows.append(dict(loss=lf, **{FIT_NAMES[j]: float(rec[j]) for j in range(6)}))
    return rows

def profile(param, grid, use_ctrl, seed=7):
    """Profile loss over one param: freeze it on grid, optimize the rest."""
    rng = np.random.default_rng(seed)
    idx = FIT_NAMES.index(param)
    tv = np.array(true_fit_vec())
    out = []
    for val in grid:
        best = None
        for _ in range(4):  # a few restarts per grid point
            v0 = jnp.array(tv + rng.normal(0, 0.7, size=6))
            v0 = v0.at[idx].set(np.log(val))
            mask = np.zeros(6, bool); mask[idx] = True
            fixed_vals = jnp.array(tv).at[idx].set(np.log(val))
            vf, lf = fit(v0, use_ctrl, steps=1500,
                         fixed_mask=mask, fixed_vals=fixed_vals)
            if best is None or lf < best[1]:
                best = (vf, lf)
        rec = np.exp(np.array(best[0]))
        out.append((float(val), float(best[1]),
                    {FIT_NAMES[j]: float(rec[j]) for j in range(6)}))
    return out

def summarize_multistart(rows, tag):
    arr = {k: np.array([r[k] for r in rows]) for k in FIT_NAMES}
    losses = np.array([r["loss"] for r in rows])
    lo = losses.min()
    near = losses <= lo + 0.02 * abs(lo) + 1e-6  # near-optimal band
    print(f"\n=== MULTISTART [{tag}] : {len(rows)} inits ===")
    print(f"loss min={lo:.4f} max={losses.max():.4f} "
          f"near-opt (<=+2%) fraction={near.mean():.2f}")
    print(f"{'param':>6} {'true':>8} {'median':>9} {'CV%(near)':>10} "
          f"{'min':>8} {'max':>8}")
    res = {}
    for k in FIT_NAMES:
        vals = arr[k][near]
        cv = 100 * vals.std() / (abs(vals.mean()) + 1e-9)
        print(f"{k:>6} {TRUE[k]:>8.3f} {np.median(arr[k]):>9.3f} "
              f"{cv:>10.1f} {vals.min():>8.3f} {vals.max():>8.3f}")
        res[k] = dict(true=TRUE[k], cv_near=float(cv),
                      min=float(vals.min()), max=float(vals.max()))
    # the ridge: correlation of circuit-n vs readout-h among near-optimal fits
    if near.sum() >= 3:
        c = np.corrcoef(arr["n"][near], arr["h"][near])[0, 1]
        print(f"RIDGE: corr(circuit n, readout h) among near-opt = {c:+.3f}")
        res["corr_n_h"] = float(c)
    res["_near_frac"] = float(near.mean())
    # composition fidelity: does the fitted u->Lambda map match truth?
    comp_errs = []
    ug = jnp.linspace(0.2, 4.0, 60)
    K, n, vmax, ba, Km, h, Vmax, b = theta_true
    lam_true = M.readout(M.circuit(ug, K, n, vmax, ba), Km, h, Vmax, b)
    for r in [rr for rr, m in zip(rows, near) if m]:
        vv = jnp.log(jnp.array([r[k] for k in FIT_NAMES]))
        th = pack_theta(vv)
        lam = M.readout(M.circuit(ug, th[0], th[1], th[2], th[3]),
                        th[4], th[5], th[6], th[7])
        comp_errs.append(float(jnp.sqrt(jnp.mean((lam - lam_true) ** 2)) /
                               jnp.mean(lam_true)))
    print(f"composition R(g(u)) rel-RMSE across near-opt: "
          f"median={np.median(comp_errs):.4f} max={np.max(comp_errs):.4f}")
    res["comp_relRMSE_median"] = float(np.median(comp_errs))
    return res

def summarize_profile(prof, tag, param):
    losses = np.array([p[1] for p in prof])
    lo = losses.min()
    span = losses.max() - lo
    print(f"\n--- PROFILE over {param} [{tag}] ---")
    print(f"{'val':>8} {'loss':>10} {'dLoss':>10}")
    for val, l, _ in prof:
        print(f"{val:>8.3f} {l:>10.4f} {l-lo:>10.4f}")
    print(f"loss span across {param}-grid = {span:.4f}  "
          f"(flat => unidentifiable; deep well => identifiable)")
    # argmin location vs truth
    amin = prof[int(np.argmin(losses))][0]
    print(f"argmin {param} = {amin:.3f}  (true={TRUE[param]:.3f})")
    return dict(span=float(span), argmin=float(amin), true=TRUE[param],
                grid=[float(p[0]) for p in prof],
                loss=[float(p[1]) for p in prof])

if __name__ == "__main__":
    print("TRUE:", TRUE, "phi", PHI, "N_CELLS", N_CELLS)
    print(f"data counts: mean={float(Y_DATA.mean()):.2f} "
          f"sd={float(Y_DATA.std()):.2f} "
          f"frac_low(a<0.3)={float((A_DATA<0.3).mean()):.2f}")

    results = {}
    n_grid = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
    h_grid = [1.0, 1.5, 2.0, 3.0, 5.0]

    # ---- Experiment 1: no control ----
    print("\n############ EXPERIMENT 1: JOINT FIT, NO CONTROL ############")
    ms1 = multistart(use_ctrl=False)
    results["exp1_multistart"] = summarize_multistart(ms1, "no ctrl")
    results["exp1_profile_n"] = summarize_profile(
        profile("n", n_grid, use_ctrl=False), "no ctrl", "n")
    results["exp1_profile_h"] = summarize_profile(
        profile("h", h_grid, use_ctrl=False), "no ctrl", "h")

    # ---- Experiment 2: with constitutive control ----
    print("\n############ EXPERIMENT 2: + CONSTITUTIVE CONTROL ############")
    print(f"control: {N_DOSE} activity doses x {N_REP} reps, "
          f"doses in [{float(A_DOSE.min()):.2f},{float(A_DOSE.max()):.2f}]")
    ms2 = multistart(use_ctrl=True)
    results["exp2_multistart"] = summarize_multistart(ms2, "with ctrl")
    results["exp2_profile_n"] = summarize_profile(
        profile("n", n_grid, use_ctrl=True), "with ctrl", "n")
    results["exp2_profile_h"] = summarize_profile(
        profile("h", h_grid, use_ctrl=True), "with ctrl", "h")

    outp = os.path.join(SCRATCH, "spike_ident_results.json")
    with open(outp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {outp}")
