"""Bridge real Perturb-seq counts → the Lyapunov path's activity space.

The covariance attribution (``inference.lyapunov``) fits in **activity space**: an
``(n_cells, n_species)`` linear array, one column per circuit species. Real data is raw
counts of a gene panel. This module maps one to the other:

- ``counts_to_activity`` — per-cell **depth-normalize** the panel (library-size factors,
  the analogue of the ``scale`` we pin from WT — see the ``scale ⇄ vmax`` degeneracy in
  ``lyapunov``), then reduce the marker genes of each species to one activity column.
- ``adata_to_operating_point`` — turn one condition of an AnnData into an
  :class:`~nudge.inference.lyapunov.OperatingPoint` (its activity + the WT-calibrated
  ``scale``/``obs_sd``), ready for ``attribute_lyapunov_single`` / ``_multi``.

**Honest bounds.** There is no direct Ras-GTP readout: a species' activity is an
composite of its marker transcripts (the IEG panel for the activation output). And the
Lyapunov covariance is homoscedastic — it ignores the NB ``μ + φμ²`` mean-variance
growth of the counts; ``lna_reliable`` abstains where that (or low depth) bites.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from nudge.core.circuit import Circuit
from nudge.inference.lyapunov import OperatingPoint, calibrate_from_wt

__all__ = ["adata_to_operating_point", "counts_to_activity"]

#: species name → marker gene symbols whose (normalized) mean is that species' activity.
SpeciesMarkers = Mapping[str, Sequence[str]]


def counts_to_activity(
    adata: Any,
    circuit: Circuit,
    species_markers: SpeciesMarkers,
    *,
    library_col: str | None = "total_counts",
) -> np.ndarray:
    """Reduce a count AnnData to ``(n_cells, n_species)`` depth-normalized activity.

    Columns are in ``circuit.names`` order. Each cell's counts are size-factor-scaled
    by its library size (``obs[library_col]`` if present — the whole-transcriptome UMI
    total — else the panel row-sum) rescaled to the median, so per-cell sequencing
    depth is divided out; then each species' markers are averaged. Raises ``KeyError``
    if a marker symbol is absent from ``var_names`` or a species has no markers.
    """
    x = np.asarray(adata.X, dtype=float)
    genes = list(map(str, adata.var_names))
    gene_ix = {g: i for i, g in enumerate(genes)}

    if library_col is not None and library_col in getattr(adata, "obs", {}):
        lib = np.asarray(adata.obs[library_col], dtype=float)
    else:
        lib = x.sum(axis=1)
    lib = np.where(lib > 0, lib, 1.0)
    norm = x / lib[:, None] * float(np.median(lib))  # size-factor normalize

    cols = []
    for name in circuit.names:
        markers = species_markers.get(name)
        if not markers:
            raise KeyError(f"no markers given for species {name!r}")
        missing = [g for g in markers if g not in gene_ix]
        if missing:
            raise KeyError(f"marker genes absent from var_names: {missing}")
        idx = [gene_ix[g] for g in markers]
        cols.append(norm[:, idx].mean(axis=1))
    return np.stack(cols, axis=1)


def adata_to_operating_point(
    adata: Any,
    circuit: Circuit,
    species_markers: SpeciesMarkers,
    condition: str,
    *,
    wt_condition: str = "WT",
    library_col: str | None = "total_counts",
    scale: float | None = None,
    obs_sd: float | None = None,
) -> OperatingPoint:
    """Build one :class:`OperatingPoint` from ``condition`` vs ``wt_condition``.

    The condition's cells become the operating point's activity ``data``; ``scale``/
    ``obs_sd`` are pinned from the WT (control) cells via ``calibrate_from_wt`` unless
    given (pass a shared WT-derived pair when several conditions share one control).
    """
    cond_mask = np.asarray(adata.obs["condition"] == condition)
    cond_act = counts_to_activity(
        adata[cond_mask], circuit, species_markers, library_col=library_col
    )
    if scale is None or obs_sd is None:
        wt_mask = np.asarray(adata.obs["condition"] == wt_condition)
        wt_act = counts_to_activity(
            adata[wt_mask], circuit, species_markers, library_col=library_col
        )
        scale, obs_sd = calibrate_from_wt(wt_act, circuit)
    return OperatingPoint(
        data=cond_act, circuit=circuit, scale=scale, obs_sd=obs_sd
    )
