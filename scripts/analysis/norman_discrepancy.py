"""Forensic dump of NUDGE's Norman-2019 epistasis calls (5 pairs).

Per pair: full EpistasisFit numbers, per-condition depth (total_counts) to rule out a
batch/depth artifact in the A+B arm, and projection-axis diagnostics (‖vA‖, ‖vB‖, angle
between the single-arm shift vectors, each arm's contribution to the additive axis, and
whether the axis is dominated by one arm). Read-only; writes nothing.
"""

from __future__ import annotations

import numpy as np
import anndata as ad

from nudge.inference.bridge import combo_effect_scores, _norm_counts, _condition_mask
from nudge.inference.epistasis import attribute_synergy

H5AD = "/media/nick/Seagate Hub/norman_2019/norman_2019.h5ad"

PAIRS = [
    ("CBL", "CNN1"),        # control: explicit synergy (Fig 3)
    ("DUSP9", "ETS2"),      # control: explicit buffering (Fig 5)
    ("CBL", "UBASH3B"),     # under scrutiny
    ("CNN1", "UBASH3B"),    # under scrutiny
    ("FOXA1", "FOXA3"),     # under scrutiny (paralog)
]


def axis_diag(adata, ga, gb, n_top=2000):
    """Replicate combo_effect_scores' projection to expose the axis geometry."""
    ab_label = f"{ga}+{gb}"
    obs = adata.obs
    masks_full = {
        "control": _condition_mask(obs, "condition", "control"),
        "a": _condition_mask(obs, "condition", ga),
        "b": _condition_mask(obs, "condition", gb),
        "ab": _condition_mask(obs, "condition", ab_label),
    }
    keep = np.zeros(len(obs), dtype=bool)
    for m in masks_full.values():
        keep |= m
    sub = adata[keep]
    lab = np.asarray(sub.obs["condition"].astype(str))
    name = {"control": "control", "a": ga, "b": gb, "ab": ab_label}
    masks = {k: lab == name[k] for k in name}
    norm, _ = _norm_counts(sub, "total_counts")
    lognorm = np.log1p(norm)
    var = lognorm.var(axis=0)
    n_keep = min(n_top, lognorm.shape[1])
    sel = np.argsort(var)[::-1][:n_keep]
    s = lognorm[:, sel]
    m_ctrl = s[masks["control"]].mean(axis=0)
    v_a = s[masks["a"]].mean(axis=0) - m_ctrl
    v_b = s[masks["b"]].mean(axis=0) - m_ctrl
    v_ab = s[masks["ab"]].mean(axis=0) - m_ctrl
    u = v_a + v_b
    nu = np.linalg.norm(u)
    uhat = u / nu
    na, nb = np.linalg.norm(v_a), np.linalg.norm(v_b)
    cos = float(v_a @ v_b / (na * nb + 1e-12))
    proj_a = float(v_a @ uhat)
    proj_b = float(v_b @ uhat)
    proj_ab = float(v_ab @ uhat)
    # off-axis emergent magnitude: how much of the combo shift is ORTHOGONAL to the
    # additive axis (the component the scalar-Bliss null structurally cannot see)
    v_ab_par = proj_ab * uhat
    v_ab_perp = v_ab - v_ab_par
    off_axis = float(np.linalg.norm(v_ab_perp))
    on_axis = float(np.linalg.norm(v_ab_par))
    # additive prediction vector vs observed combo vector, in full n_top space
    v_add = v_a + v_b
    resid = v_ab - v_add
    resid_norm = float(np.linalg.norm(resid))
    resid_par = float(resid @ uhat)                      # signed on-axis residual
    resid_perp = float(np.linalg.norm(resid - resid_par * uhat))
    return dict(norm_vA=float(na), norm_vB=float(nb), cos_AB=cos,
                proj_A_on_axis=proj_a, proj_B_on_axis=proj_b, proj_AB_on_axis=proj_ab,
                off_axis=off_axis, on_axis=on_axis,
                resid_norm=resid_norm, resid_on_axis=resid_par, resid_off_axis=resid_perp)


def depth_diag(adata, labels):
    tc = np.asarray(adata.obs["total_counts"], dtype=float)
    cond = np.asarray(adata.obs["condition"].astype(str))
    out = {}
    for lab in labels:
        m = cond == lab
        out[lab] = (int(m.sum()), float(np.median(tc[m])) if m.any() else float("nan"))
    return out


def main():
    print("loading", H5AD)
    adata = ad.read_h5ad(H5AD)
    adata.obs["condition"] = adata.obs["perturbation_name"].astype(str).values

    for ga, gb in PAIRS:
        ab_label = f"{ga}+{gb}"
        print("\n" + "=" * 78)
        print(f"PAIR  {ga} + {gb}")
        print("=" * 78)
        ctrl, a, b, ab = combo_effect_scores(
            adata, control_label="control", a_label=ga, b_label=gb,
            ab_label=ab_label, condition_col="condition",
        )
        res = attribute_synergy(ctrl, a, b, ab, n_boot=2000, seed=0)
        f = res.fit
        print(f"CALL: {res.call}")
        print(f"  reason: {res.reason}")
        print(f"  n: ctrl={f.n_control} A={f.n_a} B={f.n_b} AB={f.n_ab}")
        print(f"  effect_A  = {f.effect_a:+.4f}  CI [{f.ci_a[0]:+.4f}, {f.ci_a[1]:+.4f}]")
        print(f"  effect_B  = {f.effect_b:+.4f}  CI [{f.ci_b[0]:+.4f}, {f.ci_b[1]:+.4f}]")
        print(f"  effect_AB = {f.effect_ab:+.4f}  CI [{f.ci_ab[0]:+.4f}, {f.ci_ab[1]:+.4f}]")
        print(f"  additive_pred (eA+eB) = {f.additive_pred:+.4f}")
        print(f"  interaction = {f.interaction:+.4f}  CI [{f.ci_interaction[0]:+.4f}, {f.ci_interaction[1]:+.4f}]")
        print(f"  bic_additive={f.bic_additive:.2f} bic_free={f.bic_free:.2f} dBIC={f.bic_additive-f.bic_free:.2f}")
        dom = "A" if abs(f.effect_a) > abs(f.effect_b) else "B"
        print(f"  arm magnitudes: |eA|={abs(f.effect_a):.3f} |eB|={abs(f.effect_b):.3f}  dominant={dom}")
        print(f"  AB - A = {f.effect_ab-f.effect_a:+.3f}   AB - B = {f.effect_ab-f.effect_b:+.3f}")

        d = axis_diag(adata, ga, gb)
        print("  --- projection-axis geometry (2000 HVG log1p space) ---")
        print(f"  ‖vA‖={d['norm_vA']:.3f} ‖vB‖={d['norm_vB']:.3f} cos(vA,vB)={d['cos_AB']:+.3f}")
        print(f"  proj(A on axis)={d['proj_A_on_axis']:+.3f} proj(B on axis)={d['proj_B_on_axis']:+.3f} proj(AB on axis)={d['proj_AB_on_axis']:+.3f}")
        print(f"  combo shift decomposed: on-axis ‖={d['on_axis']:.3f}  OFF-axis ‖={d['off_axis']:.3f}  (off/on ratio={d['off_axis']/max(d['on_axis'],1e-9):.2f})")
        print(f"  interaction residual (v_AB - v_A - v_B): ‖={d['resid_norm']:.3f}  on-axis={d['resid_on_axis']:+.3f}  off-axis ‖={d['resid_off_axis']:.3f}")

        dd = depth_diag(adata, ["control", ga, gb, ab_label])
        print("  --- depth (n, median total_counts) ---")
        for k, (n, md) in dd.items():
            print(f"    {k:24s} n={n:6d}  median_depth={md:.0f}")


if __name__ == "__main__":
    main()
