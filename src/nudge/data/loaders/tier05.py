"""Tier 0.5 — the independent stochastic simulator (inverse-crime guard).

NUDGE's own tau-leaping SSA of a self-activating gene (``nudge.data.stochastic``)
is the Tier-0.5 data source: genuinely stochastic dynamics with *emergent*
bimodality, independent of the deterministic model the fitter uses. External
simulators (SERGIO / BoolODE) remain a later option and would land here too.
"""

from __future__ import annotations

from nudge.data.stochastic import generate_stochastic_perturbseq

__all__ = ["generate_stochastic_perturbseq"]
