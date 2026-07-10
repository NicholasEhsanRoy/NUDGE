"""RED-TEAM (round 4, re-scan): regression + gaming check on the merged P3 fix.

Target: ``nudge.design.invert.design`` safety gate (``NUDGE-METHOD-007`` /
``NUDGE-LIM-013``). The P3 fix made ``_safety_report`` fire
``high_risk_of_instability`` on ``(delta > margin)`` OR ``(proximity_after >=
NEAR_FOLD)`` (an absolute near-fold alarm reusing the shipped
``nudge.inference.bifurcation.NEAR_FOLD`` constant), added ``SafetyReport.near_fold``,
reworded the near-fold reason to AGREE with ``classify_robustness``, and carried the
one-sided-lower-bound caveat on the SAFE branch.

This script re-attacks the fix for a NEW confident-wrong or a gameable failure, driving
the SHIPPED ``design()`` path end-to-end and asserting the fail-safe invariants below.
It exits 0 when every invariant HOLDS (the fix is sound) and 2 if any is violated (a
regression / new hole). All checks are deterministic (a gradient inversion to a fixed
target — no seed dependence).

Invariants asserted (each is a way the fix could have broken):

  I1  ABSOLUTE-CHECK COMPLETENESS. For every reachable plan, ``proximity_after`` at/above
      ``NEAR_FOLD`` MUST imply ``high_risk_of_instability`` (the absolute alarm can never
      be silently below the reported near-fold landing). Covers both the bistable->bistable
      path and the base-monostable->creates-near-fold path.
  I2  WORDING AGREES WITH CLASSIFY. A "stays away from the fold" reason MUST NOT co-occur
      with a reported ``proximity_after >= NEAR_FOLD``.
  I3  NO OVER-ABSTENTION. A robust intervention that lands strictly below ``NEAR_FOLD``
      with a sub-margin delta MUST be cleared (not high-risk) — the absolute alarm must
      not fire spuriously on a genuinely-robust switch (an over-abstention that pushes a
      user to disable the gate is itself a fail-safe defect).
  I4  MARGIN CANNOT DISABLE THE ABSOLUTE CHECK. Re-running the original P3 construction
      with a huge ``margin`` (a user routing around the relative alarm) MUST still flag
      high-risk via the absolute near-fold landing.
  I5  SAFE-BRANCH ONE-SIDED CAVEAT. When the cleared circuit's proximity is a one-sided
      lower bound, the SAFE reason MUST carry the "one-sided LOWER bound" caveat.

Run: uv run python scripts/redteam/design_p3_regression_check.py
Touches no src/ code and no fail-safe margins — diagnostic only.
"""

from __future__ import annotations

import itertools

import numpy as np

from nudge.circuits import ras_switch_1node, ras_switch_2node, toggle
from nudge.design.invert import (
    AbstentionResult,
    CircuitFit,
    _base_value,
    _rebuild,
    design,
    flip_target,
)
from nudge.inference.bifurcation import (
    NEAR_FOLD,
    bifurcation_proximity,
    classify_robustness,
)


def _after_circuit(base: object, knobs: list, plan: object) -> object:
    """Rebuild the intervened circuit from the plan's deltas (independent recompute)."""
    base_p = base.base_params()  # type: ignore[attr-defined]
    dmap = {f: ld for f, ld, _fac in plan.deltas}  # type: ignore[attr-defined]
    vals = []
    for f in knobs:
        lb = np.log(_base_value(base_p, f))
        vals.append(float(np.exp(lb + dmap.get(f, 0.0))))
    return _rebuild(base, knobs, np.array(vals))


def _sweep(failures: list[str]) -> int:
    """I1/I2/I3 across a grid of base circuits, targets, and start basins."""
    ctors = {
        "1node": ras_switch_1node,
        "2node": ras_switch_2node,
        "toggle": toggle,
    }
    n_plans = 0
    for name, ctor in ctors.items():
        grid = itertools.product([1.6, 2.0, 3.0], [2.5, 3.2], [0.9, 1.2, 1.5])
        for n, vmax, K in grid:
            base = ctor(n=n, vmax=vmax, K=K, basal=0.05)
            for tgt_scale, start in [(0.8, "high"), (1.0, "high"), (1.0, "low")]:
                tc = ctor(n=n, vmax=vmax, K=K * tgt_scale, basal=0.05)
                try:
                    target = flip_target(tc, to=start)
                except ValueError:
                    continue
                knobs = [("edge", 0, "K"), ("edge", 0, "vmax")]
                fit = CircuitFit(circuit=base, free=knobs, is_reliable=True, reason="r")
                plan = design(
                    fit, target, free=knobs, start=start, steps=180,
                    l1=1e-3, tol=0.25, margin=0.15,
                )
                if isinstance(plan, AbstentionResult):
                    continue
                n_plans += 1
                s = plan.safety
                assert s is not None
                pa = s.proximity_after
                if pa is not None and pa >= NEAR_FOLD and not s.high_risk_of_instability:
                    failures.append(
                        f"I1 {name} n={n} vmax={vmax} K={K}: prox_after={pa:.3f} "
                        f">= NEAR_FOLD but high_risk=False"
                    )
                if (
                    "stays away from the fold" in plan.reason
                    and pa is not None
                    and pa >= NEAR_FOLD
                ):
                    failures.append(
                        f"I2 {name} n={n} vmax={vmax} K={K}: 'stays away' wording with "
                        f"prox_after={pa:.3f} >= NEAR_FOLD"
                    )
                if (
                    s.high_risk_of_instability
                    and pa is not None
                    and pa < NEAR_FOLD
                    and not s.crosses_fold
                    and s.delta is not None
                    and s.delta <= 0.15
                ):
                    failures.append(
                        f"I3 {name} n={n} vmax={vmax} K={K}: spurious high_risk at "
                        f"prox_after={pa:.3f}<NEAR_FOLD, delta={s.delta:.3f}<=margin"
                    )
    print(f"I1/I2/I3 sweep: {n_plans} reachable plans checked")
    return n_plans


def _created_near_fold(failures: list[str]) -> None:
    """I1 on the base-monostable -> creates-near-fold path (the P3-claimed branch)."""
    base = ras_switch_1node(n=2.0, vmax=3.0, K=1.9, basal=0.05)
    assert bifurcation_proximity(base) is None, "base must be monostable"
    knobs = [("edge", 0, "K")]
    for ktar in [0.8, 0.9, 1.0]:
        tc = ras_switch_1node(n=2.0, vmax=3.0, K=ktar, basal=0.05)
        target = flip_target(tc, to="high")
        fit = CircuitFit(circuit=base, free=knobs, is_reliable=True, reason="r")
        plan = design(
            fit, target, free=knobs, start=np.array([float(target[0])]),
            steps=600, l1=1e-4, tol=0.3, margin=0.15,
        )
        if isinstance(plan, AbstentionResult):
            continue
        s = plan.safety
        assert s is not None
        after = _after_circuit(base, knobs, plan)
        sa = bifurcation_proximity(after)
        indep = classify_robustness(sa)[0] if sa is not None else "monostable"
        pa = s.proximity_after
        if pa is not None and pa >= NEAR_FOLD:
            if not (s.near_fold and s.high_risk_of_instability):
                failures.append(
                    f"I1(created) K->{ktar}: prox_after={pa:.3f}>=NEAR_FOLD but "
                    f"near_fold={s.near_fold} high_risk={s.high_risk_of_instability}"
                )
            if indep != "near-fold":
                failures.append(
                    f"I1(created) K->{ktar}: classify disagrees ({indep})"
                )
    print("I1(created near-fold): checked base-monostable->creates-near-fold path")


def _margin_cannot_disable(failures: list[str]) -> None:
    """I4: a huge margin must not disable the absolute near-fold alarm."""
    base = ras_switch_1node(n=2.0, vmax=3.0, K=1.5, basal=0.05)
    near = ras_switch_1node(n=2.0, vmax=3.0, K=1.0, basal=0.05)
    target = flip_target(near, to="high")
    fit = CircuitFit(circuit=base, free=[("edge", 0, "K")], is_reliable=True, reason="r")
    plan = design(
        fit, target, free=[("edge", 0, "K")], start="high", steps=400,
        l1=1e-3, tol=0.3, margin=100.0,
    )
    if isinstance(plan, AbstentionResult):
        failures.append("I4: original construction unexpectedly abstained")
        return
    s = plan.safety
    assert s is not None and s.proximity_after is not None
    if s.proximity_after >= NEAR_FOLD and not s.high_risk_of_instability:
        failures.append(
            f"I4: margin=100 disabled the absolute check (prox_after="
            f"{s.proximity_after:.3f}, high_risk=False)"
        )
    if "stays away from the fold" in plan.reason:
        failures.append("I4: 'stays away' wording under margin=100 near-fold landing")
    print(
        f"I4(margin=100): prox_after={s.proximity_after:.3f} "
        f"high_risk={s.high_risk_of_instability} (absolute check independent of margin)"
    )


def _safe_branch_caveat(failures: list[str]) -> None:
    """I5: the SAFE branch carries the one-sided caveat when proximity is a lower bound."""
    base = ras_switch_1node(n=2.0, vmax=3.0, K=1.5, basal=0.05)
    score = bifurcation_proximity(base)
    assert score is not None and score.one_sided, "base must be one-sided for this probe"
    target = flip_target(base, to="high")
    fit = CircuitFit(circuit=base, free=[("edge", 0, "vmax")], is_reliable=True, reason="r")
    plan = design(
        fit, target * 1.02, free=[("edge", 0, "vmax")], start="high", steps=300,
        l1=1e-3, tol=0.3, margin=0.15,
    )
    if isinstance(plan, AbstentionResult):
        failures.append("I5: robust ON-level adjust unexpectedly abstained")
        return
    s = plan.safety
    assert s is not None
    cleared = not s.high_risk_of_instability and not s.crosses_fold
    if cleared and s.one_sided and "one-sided LOWER bound" not in plan.reason:
        failures.append("I5: SAFE branch dropped the one-sided lower-bound caveat")
    print(
        f"I5(safe-branch caveat): cleared={cleared} one_sided={s.one_sided} "
        f"caveat_present={'one-sided LOWER bound' in plan.reason}"
    )


def run() -> int:
    print("=" * 80, flush=True)
    print("RED-TEAM r4 re-scan: P3-fix regression + gaming check on design() safety gate")
    print(f"(NEAR_FOLD={NEAR_FOLD}; asserting I1..I5 through the shipped design() path)")
    print("=" * 80, flush=True)

    failures: list[str] = []
    _sweep(failures)
    _created_near_fold(failures)
    _margin_cannot_disable(failures)
    _safe_branch_caveat(failures)

    print("\n" + "=" * 80, flush=True)
    if failures:
        print(f"REGRESSION / HOLE: {len(failures)} invariant(s) violated:")
        for f in failures:
            print("  -", f)
        return 2
    print("ALL INVARIANTS HELD — P3 fix is sound; no new confident-wrong, no regression,")
    print("no over-abstention, not gameable via margin. HOLES_FOUND: 0 for this target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
