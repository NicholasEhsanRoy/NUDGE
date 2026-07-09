"""End-to-end attribution across operating points — the real-data capstone pipeline.

Composes the tested pieces into the flow Phase 4 needs: per operating point (a condition
file — e.g. a donor × stimulation), map a target perturbation vs its control to activity
(``bridge``), pin depth/noise, guard with ``lna_reliable``, and run the covariance
attribution — single-condition per operating point (expected to **abstain** between gain
and threshold, the measured degeneracy) and the multi-operating-point **breaker** across
them (which can resolve it). It abstains loudly on low-count / near-bifurcation states
and reports that honestly rather than forcing a call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nudge.core.circuit import Circuit
from nudge.inference.bridge import SpeciesMarkers, adata_to_operating_point
from nudge.inference.lyapunov import (
    OperatingPoint,
    attribute_lyapunov_multi,
    attribute_lyapunov_single,
    lna_reliable,
)

__all__ = ["AttributionReport", "attribute_across_operating_points"]


@dataclass(frozen=True)
class AttributionReport:
    """What NUDGE concluded for one target across the operating points, honestly."""

    target: str
    #: op label → (single-condition call, restricted NLLs) — abstains where guarded.
    single: dict[str, tuple[str, dict[str, float]]] = field(default_factory=dict)
    #: the multi-op joint call (the breaker), or ``None`` if < 2 usable ops.
    multi: tuple[str, dict[str, float]] | None = None
    #: op label → why it was unusable (too few cells / LNA unreliable), if so.
    skipped: dict[str, str] = field(default_factory=dict)
    #: op label → number of target-condition cells.
    n_cells: dict[str, int] = field(default_factory=dict)


def attribute_across_operating_points(
    ops: dict[str, Any],
    circuit: Circuit,
    markers: SpeciesMarkers,
    target: str,
    *,
    target_edge: int = 0,
    wt_condition: str = "WT",
    min_cells: int = 200,
    steps: int = 200,
    seed: int = 0,
) -> AttributionReport:
    """Attribute ``target`` across operating points ``ops`` (label → AnnData).

    Each AnnData holds ``target`` + ``wt_condition`` cells at one operating point. For
    each usable one (≥ ``min_cells`` target cells + a trustworthy LNA), builds a
    WT-calibrated :class:`OperatingPoint` + a single-condition call; then the joint fit
    over the usable points. Unusable operating points are recorded in ``skipped``, not
    silently dropped.
    """
    single: dict[str, tuple[str, dict[str, float]]] = {}
    skipped: dict[str, str] = {}
    n_cells: dict[str, int] = {}
    usable: list[OperatingPoint] = []

    for label, adata in ops.items():
        n = int(np.asarray(adata.obs["condition"] == target).sum())
        n_cells[label] = n
        if n < min_cells:
            skipped[label] = f"only {n} target cells (< {min_cells})"
            continue
        point = adata_to_operating_point(
            adata, circuit, markers, target, wt_condition=wt_condition
        )
        ok, why = lna_reliable(point.circuit, point.scale)
        if not ok:
            skipped[label] = f"LNA unreliable: {why}"
            continue
        usable.append(point)
        single[label] = attribute_lyapunov_single(
            point.data, circuit, scale=point.scale, obs_sd=point.obs_sd,
            target_edge=target_edge, steps=steps, seed=seed,
        )

    multi = None
    if len(usable) >= 2:
        multi = attribute_lyapunov_multi(
            usable, target_edge=target_edge, steps=steps, seed=seed
        )

    return AttributionReport(
        target=target, single=single, multi=multi, skipped=skipped, n_cells=n_cells
    )
