"""RED-TEAM (round 3): can ``hidden_node`` be pushed to a bare POSITIVE hidden-node claim,
or to rank the hidden-node cause as the LONE leading answer?

Target: ``nudge.inference.hidden_node.diagnose_inadequacy`` (``NUDGE-METHOD-009`` /
``NUDGE-LIM-015``). It ships abstention-only: the hidden-node cause is rank-capped at
``plausible`` (never ``leading``) and always phrased "consistent with — does not prove".
The attack drives a huge off-axis / neomorphic residual as the ONLY firing diagnostic
(``off_model=False``), trying to make hidden-node the single top-ranked cause.

Expected (HELD): hidden-node stays ``plausible`` and is ALWAYS tied with the always-present
"not-a-switch", "nonlinear readout" and "off-target" causes — never the lone leading answer,
never a bare positive. This documents the structural guard.

Run: uv run python scripts/redteam/hidden_node_lone_leading_probe.py
Touches no src/ code — diagnostic only.
"""

from __future__ import annotations

from nudge.inference.hidden_node import diagnose_inadequacy

# Each case tries to make hidden-node dominate: a large off-axis residual, other flags off.
CASES = [
    dict(off_model=False, neomorphic_ratio=99.0),
    dict(off_model=False, neomorphic_ratio=5.0, readout_flag=False),
    dict(off_model=False, neomorphic_ratio=10.0, topology_uncertain=False,
         depth_confounded=False),
    dict(off_model=True, neomorphic_ratio=50.0),
]
_BARE = [
    "there is a hidden node", "hidden node detected", "proves a hidden",
    "confirms a hidden", "is a hidden node",
]


def run() -> int:
    print("=" * 80, flush=True)
    print("RED-TEAM r3: try to make hidden-node the LONE leading / a bare positive")
    print("=" * 80, flush=True)
    holes = 0
    for kw in CASES:
        r = diagnose_inadequacy(**kw)
        ranked = r.ranked_causes()
        top = ranked[0]
        n_top = sum(1 for c in ranked if c.qualitative_rank == top.qualitative_rank)
        hn = next(c for c in ranked if c.name.startswith("hidden node"))
        lone_hidden = ranked[0].name.startswith("hidden node") and n_top == 1
        blob = (hn.evidence + " " + hn.distinguishing_experiment).lower()
        bare = any(p in blob for p in _BARE)
        hole = lone_hidden or bare or hn.qualitative_rank == "leading"
        holes += int(hole)
        print(
            f"\ninput={kw}\n"
            f"  top={top.name!r} rank={top.qualitative_rank} (#tied={n_top})\n"
            f"  hidden-node rank={hn.qualitative_rank}  lone-leading={lone_hidden}  "
            f"bare-positive={bare}  -> {'HOLE' if hole else 'HELD'}",
            flush=True,
        )
    print("\n" + "=" * 80, flush=True)
    print(f"hidden-node holes: {holes}  ({'HELD' if holes == 0 else 'HOLE'})", flush=True)
    return 0 if holes == 0 else 2


if __name__ == "__main__":
    raise SystemExit(run())
