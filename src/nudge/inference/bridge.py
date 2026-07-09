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

__all__ = [
    "adata_to_operating_point",
    "counts_to_activity",
    "knockdown_dose_response",
]

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


def _norm_counts(
    adata: Any, library_col: str | None
) -> tuple[np.ndarray, dict[str, int]]:
    """Depth-normalized counts (size-factor to median) + a symbol→column index map."""
    x = np.asarray(
        adata.X.todense() if hasattr(adata.X, "todense") else adata.X, dtype=float
    )
    if library_col is not None and library_col in getattr(adata, "obs", {}):
        lib = np.asarray(adata.obs[library_col], dtype=float)
    else:
        lib = x.sum(axis=1)
    lib = np.where(lib > 0, lib, 1.0)
    norm = x / lib[:, None] * float(np.median(lib))
    gene_ix = {str(g): i for i, g in enumerate(adata.var_names)}
    return norm, gene_ix


def knockdown_dose_response(
    adata: Any,
    *,
    target_gene: str,
    signature: Sequence[str],
    group_prefix: str,
    group_col: str = "guide",
    condition_col: str = "condition",
    control_label: str = "WT",
    library_col: str | None = "total_counts",
    min_cells_per_group: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-guide ``(knockdown dose, signature response)`` points for a KD screen.

    Each guide (a value of ``group_col`` starting with ``group_prefix``, with at least
    ``min_cells_per_group`` cells) becomes **one dose point** — different guides against
    the
    same target achieve different knockdown strengths, so the guide axis *is* a dose
    axis
    (the operating-point structure the degeneracy-breaker exploits). For each such
    guide:

    - ``dose`` = ``1 − mean(target_gene) / mean(target_gene | control)`` — the
    *fractional
      knockdown* of ``target_gene`` (0 = no knockdown, 1 = fully silenced);
    - ``response`` = ``mean(signature) / mean(signature | control)`` — the readout
    signature
      (mean over ``signature`` genes) relative to control.

    Depth-normalized (size-factor to the median library size), so per-cell depth is
    divided
    out — the same normalization as :func:`counts_to_activity`. Feed the result to
    :func:`~nudge.inference.dose_response.fit_dose_response` with
    ``direction="repress"``
    when the signature *falls* with knockdown. Raises ``KeyError`` on missing
    genes/columns.
    """
    obs = getattr(adata, "obs", None)
    for col in (group_col, condition_col):
        if obs is None or col not in obs:
            raise KeyError(f"obs column {col!r} not found")
    sig = list(signature)
    norm, gene_ix = _norm_counts(adata, library_col)
    missing = [g for g in [target_gene, *sig] if g not in gene_ix]
    if missing:
        raise KeyError(f"genes absent from var_names: {missing}")

    tgt_col = gene_ix[target_gene]
    sig_cols = [gene_ix[g] for g in sig]
    cond = np.asarray(obs[condition_col].astype(str))
    guide = np.asarray(obs[group_col].astype(str))

    ctrl = cond == control_label
    if not ctrl.any():
        raise KeyError(f"no control cells (condition == {control_label!r})")
    base_tgt = float(norm[ctrl, tgt_col].mean())
    base_sig = float(norm[ctrl][:, sig_cols].mean(axis=1).mean())
    base_tgt = base_tgt if base_tgt > 0 else 1.0
    base_sig = base_sig if base_sig > 0 else 1.0

    dose: list[float] = []
    response: list[float] = []
    for g in np.unique(guide):
        if not g.startswith(group_prefix):
            continue
        m = guide == g
        if int(m.sum()) < min_cells_per_group:
            continue
        dose.append(1.0 - float(norm[m, tgt_col].mean()) / base_tgt)
        response.append(float(norm[m][:, sig_cols].mean(axis=1).mean()) / base_sig)
    order = np.argsort(dose)
    return np.asarray(dose)[order], np.asarray(response)[order]
