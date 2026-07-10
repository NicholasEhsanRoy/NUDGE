"""RED-TEAM: can attribute_bifurcation call ``near-fold`` on a switch that is NOT near a
fold, or violate the deep-basin one-sided framing?

The robustness dial fuses three channels; two (critical slowing, basin collapse) are
deterministic and depth-independent, and the LNA lobe channel can only RAISE the alarm
(fail-safe ``max``). Attacks:

  B1 — a deep, well-buffered switch (high cooperativity, well-separated basins): must read
       ``robust`` or ``unresolved`` (deep basin), NEVER ``near-fold``.
  B2 — low sequencing depth: does a shallow-depth circuit get a spurious ``near-fold`` from
       the lobe channel? (The lobe channel is depth-calibrated; lna_reason should caveat.)
  B3 — a monostable circuit: must read ``not-bistable`` (score None), never a proximity.

Run: uv run python scripts/redteam/bifurcation_probe.py
Touches no src/ code and no fail-safe margins.
"""

from __future__ import annotations

import numpy as np

from nudge.circuits import ras_switch_1node
from nudge.inference.bifurcation import attribute_bifurcation, classify_robustness
from nudge.inference.lyapunov import sample_lna_mixture


def _data(circ: object, scale: float, seed: int) -> np.ndarray:
    import jax
    try:
        return np.asarray(
            sample_lna_mixture(circ, 2500, jax.random.PRNGKey(seed), scale=scale,
                               obs_sd=0.5)
        )
    except Exception:
        # monostable → sample a single Gaussian blob so attribute still runs
        rng = np.random.default_rng(seed)
        return rng.normal(1.0, 0.2, size=(2500, circ.n_species))


def run() -> int:
    print("=" * 78)
    print("RED-TEAM bifurcation: near-fold false-positive hunt")
    print("=" * 78)
    holes = 0

    print("\nB1 — deep well-buffered switches (should be robust/unresolved, never near-fold):")
    for n in (4.0, 6.0, 8.0, 12.0):
        circ = ras_switch_1node(n=n, K=1.0, basal=0.05)
        res = attribute_bifurcation(_data(circ, 20.0, 0), circ)
        wrong = res.call == "near-fold"
        holes += int(wrong)
        p = None if res.score is None else round(res.score.proximity, 3)
        print(f"  n={n:<5} call={res.call!r} proximity={p} one_sided="
              f"{None if res.score is None else res.score.one_sided}"
              + ("  <== FALSE near-fold" if wrong else ""))

    print("\nB2 — low sequencing depth (lobe channel stress; deterministic channels stand):")
    circ = ras_switch_1node(n=6.0, K=1.0, basal=0.05)
    for scale in (1.0, 3.0, 8.0, 20.0):
        res = attribute_bifurcation(_data(circ, scale, 1), circ)
        wrong = res.call == "near-fold"
        holes += int(wrong)
        p = None if res.score is None else round(res.score.proximity, 3)
        print(f"  scale={scale:<5} call={res.call!r} proximity={p} "
              f"lna=({res.lna_reason})" + ("  <== FALSE near-fold" if wrong else ""))

    print("\nB3 — monostable circuits (n<2 here → 1 stable mode; must be not-bistable):")
    from nudge.inference.bifurcation import bifurcation_proximity
    for n in (1.0, 1.5):  # verified 1 stable mode at these params
        circ = ras_switch_1node(n=n, K=1.0, basal=0.05)
        score = bifurcation_proximity(circ)
        call, _r = classify_robustness(score)
        wrong = call != "not-bistable"  # a monostable must not score a proximity
        holes += int(wrong)
        print(f"  n={n:<5} call={call!r} score={'None' if score is None else 'set'}"
              + ("  <== scored a monostable circuit" if wrong else ""))
    # Positive control: n≥2 IS bistable AND near the bistability onset (fold), so
    # ``near-fold`` here is the CORRECT answer, not a hole.
    for n in (2.0, 2.5):
        circ = ras_switch_1node(n=n, K=1.0, basal=0.05)
        call, _r = classify_robustness(bifurcation_proximity(circ))
        print(f"  n={n:<5} call={call!r}  (bistable near onset — correct, not a hole)")

    print(f"\n>>> bifurcation confident-wrong (false near-fold / scored monostable): {holes}")
    return holes


if __name__ == "__main__":
    run()
