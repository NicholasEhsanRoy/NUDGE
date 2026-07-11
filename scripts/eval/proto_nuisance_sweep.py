"""PROTOTYPE eval — the honest 'does it work' gate for the nuisance-augmented guard (B)
and the inert anchor (A) in ``nudge.inference._proto_nuisance``.

Measures three things against the SHIPPED ``attribute_differential`` baseline (this
worktree's pre-4b/4c differential — the baseline to beat):

1. **Affine-family coverage** — a continuum sweep of a per-condition affine (multiplicative
   ``s`` incl. P5's interior and the (1.18,1.30] gap; additive ``o``; mixed) on ONE context's
   PERTURBED cells (control clean, truth=no-difference). Counts confident-wrong (a ``*-diff``)
   for the baseline vs guard B. The bar: guard B → 0 confident-wrong across the WHOLE family.
2. **Positive controls** — genuine gain (0.55) / ceiling (1.4) / threshold (1.4) / none.
   Guard B must preserve gain (reshapes the distribution). Ceiling is EXPECTED to abstain
   under B (degenerate with scale) — the anchor (A) buys it back.
3. **Anchor recovery** — a genuine ceiling that lives UNDER a technical scale, with an inert
   feature block: does (A) recover ceiling-diff where raw (no anchor) cannot?

Run: uv run python scripts/eval/proto_nuisance_sweep.py
Experimental; touches no shipped code.
"""

from __future__ import annotations

import time

import jax
import numpy as np

from nudge.circuits import ras_switch_1node
from nudge.inference._proto_nuisance import (
    anchor_normalize,
    estimate_affine_from_inert,
    guard_b_classify,
)
from nudge.inference.differential import (
    Context,
    attribute_differential,
    simulate_context_pair,
)

CIRC = ras_switch_1node(n=6.0, vmax=2.5, K=1.0, basal=0.2)  # the test-suite config
SCALE, OBS, NC = 25.0, 0.5, 2000
CONF = {"threshold-diff", "gain-diff", "ceiling-diff"}
BASE_STEPS, GUARD_STEPS = 250, 180


def _pair(mech: str, factor: float, seed: int):
    return simulate_context_pair(
        CIRC, mechanism=mech, factor=factor, n_cells=NC,
        scale_a=SCALE, scale_b=SCALE, obs_sd=OBS, seed=seed,
    )


def _row(tag, base, g, dt, hole_flag=None):
    hole = "" if hole_flag is None else ("  <== HOLE" if hole_flag else "")
    print(
        f"{tag:26s} base={base.call:14s} guardB={g.call:14s} "
        f"knob={g.knob:4s} earn={g.earn_bic:8.1f} cond={g.cond_number:8.1f} "
        f"s={g.s_hat:.3f} o={g.o_hat:+.3f}{hole} ({dt:.0f}s)",
        flush=True,
    )


def sweep_affine(seeds=(0, 1)) -> tuple[int, int, int]:
    print("=" * 100)
    print("1. AFFINE-FAMILY SWEEP — truth=no-difference; a *-diff is confident-wrong")
    print("=" * 100, flush=True)
    # (label, transform) on context B's perturbed cells (control clean).
    mults = [1.05, 1.10, 1.15, 1.18, 1.20, 1.25, 1.30, 1.40, 1.50]
    adds = [1.0, 2.0, 3.0, 5.0]
    mixed = [(1.15, 2.0), (1.30, 3.0)]
    base_holes = guard_holes = n = 0
    for seed in seeds:
        a, b = _pair("none", 1.0, seed)
        bd = np.asarray(b.data, float)
        cases = (
            [(f"mult s={s:.2f}", s * bd) for s in mults]
            + [(f"add  o={o:.1f}", bd + o) for o in adds]
            + [(f"mix s={s:.2f},o={o:.1f}", s * bd + o) for s, o in mixed]
        )
        for label, newb in cases:
            n += 1
            b_att = Context("B", newb, b.control)
            t0 = time.time()
            base = attribute_differential(a, b_att, CIRC, steps=BASE_STEPS, seed=seed)
            g = guard_b_classify(a, b_att, CIRC, winner=base.fit.best_diff,
                                 steps=GUARD_STEPS)
            bh, gh = base.call in CONF, g.call in CONF
            base_holes += int(bh)
            guard_holes += int(gh)
            _row(f"seed={seed} {label}", base, g, time.time() - t0, gh)
    print(f"\n  affine sweep: {n} cases/2 mechanisms — "
          f"baseline confident-wrong={base_holes}, guardB confident-wrong={guard_holes}",
          flush=True)
    return base_holes, guard_holes, n


def positive_controls(seeds=(1, 2)) -> dict:
    print("\n" + "=" * 100)
    print("2. POSITIVE CONTROLS — genuine single-knob differences (no affine)")
    print("   gain(0.55) must RESOLVE under B; ceiling EXPECTED to abstain (anchor buys "
          "back); threshold may abstain")
    print("=" * 100, flush=True)
    cases = [("gain", 0.55), ("ceiling", 1.4), ("threshold", 1.4), ("none", 1.0)]
    res: dict = {}
    for mech, fac in cases:
        for seed in seeds:
            a, b = _pair(mech, fac, seed)
            t0 = time.time()
            base = attribute_differential(a, b, CIRC, steps=BASE_STEPS, seed=0)
            g = guard_b_classify(a, b, CIRC, winner=base.fit.best_diff,
                                 steps=GUARD_STEPS)
            _row(f"{mech} fac={fac:.2f} seed={seed}", base, g, time.time() - t0)
            res.setdefault(mech, []).append((base.call, g.call))
    return res


def anchor_recovery(seeds=(1, 2)) -> dict:
    print("\n" + "=" * 100)
    print("3. ANCHOR RECOVERY (A) — a GENUINE ceiling under a technical scale s_tech")
    print("   inert block estimates s_tech; normalize; does ceiling-diff come back?")
    print("=" * 100, flush=True)
    s_tech = 1.3
    n_inert = 40
    out: dict = {"raw": [], "anchored": []}
    for seed in seeds:
        # genuine ceiling difference in B, then a technical scale applied to B's perturbed.
        a, b = _pair("ceiling", 1.4, seed)
        rng = np.random.default_rng(seed)
        bd = np.asarray(b.data, float)
        bd_scaled = s_tech * bd  # ceiling now hidden under a technical scale

        # inert block (perturbation-inert genes), same distribution in ctrl & perturbed,
        # then the SAME technical scale hits the perturbed inert block.
        mu = rng.uniform(2.0, 8.0, n_inert)
        sd = rng.uniform(0.5, 2.0, n_inert)
        ctrl_inert = rng.normal(mu, sd, (NC, n_inert))
        pert_inert = s_tech * rng.normal(mu, sd, (NC, n_inert))

        anchor = estimate_affine_from_inert(pert_inert, ctrl_inert)
        bd_norm = anchor_normalize(bd_scaled, anchor)

        b_raw = Context("B", bd_scaled, b.control)
        b_anch = Context("B", bd_norm, b.control)
        raw = attribute_differential(a, b_raw, CIRC, steps=BASE_STEPS, seed=0)
        anch = attribute_differential(a, b_anch, CIRC, steps=BASE_STEPS, seed=0)
        out["raw"].append(raw.call)
        out["anchored"].append(anch.call)
        print(f"seed={seed} s_tech={s_tech}  anchor s_hat={anchor.scale:.3f} "
              f"o_hat={anchor.offset:+.3f} | raw(no anchor)={raw.call:14s} "
              f"anchored={anch.call:14s}", flush=True)
    return out


def main() -> int:
    jax.config.update("jax_platform_name", "cpu")
    t0 = time.time()
    b_h, g_h, n = sweep_affine()
    pc = positive_controls()
    an = anchor_recovery()

    print("\n" + "=" * 100)
    print("VERDICT SUMMARY")
    print("=" * 100)
    print(f"1. Affine family ({n} confounded cases): baseline confident-wrong={b_h}, "
          f"guardB confident-wrong={g_h}")
    for mech, pairs in pc.items():
        bc = [p[0] for p in pairs]
        gc = [p[1] for p in pairs]
        print(f"2. positive {mech:10s}: baseline={bc}  guardB={gc}")
    print(f"3. anchor recovery: raw={an['raw']}  anchored={an['anchored']}")
    print(f"\ntotal wall time {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
