"""Generic Perturb-seq ``.h5ad`` loader — dataset-agnostic, backed-mode, extensible.

Turns *any* standard single-cell Perturb-seq ``.h5ad`` (guide-assigned cells + raw
counts)
into the AnnData shape NUDGE's ``fit`` / attribution expect: ``.X`` raw integer counts,
``.obs["condition"]`` (a control label + one label per perturbation), ``.var_names`` =
gene symbols. All dataset specifics live in a :class:`PerturbLoaderConfig`, so a new
experiment is
a new config, not new code — the Gladstone screen (``loaders/tier2.py``) is the first
instantiation.

**Backed mode is mandatory.** These files are enormous (the Gladstone donor files are
~150 GB). :func:`load_perturbseq` opens the file **backed**
(``read_h5ad(path, backed="r")``),
computes the cell/gene masks from the small ``.obs``/``.var`` metadata, and materializes
**only** the ``(selected cells × selected genes)`` block — the ~150 GB count matrix is
never
loaded into memory. It runs on modest RAM (the count matrix stays on disk; only the tiny
subset is read). ``check_counts`` (the raw-count guardrail) runs on the result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nudge.data.ingest import check_counts

__all__ = ["PerturbLoaderConfig", "load_perturbseq"]


@dataclass(frozen=True)
class PerturbLoaderConfig:
    """How to map one Perturb-seq ``.h5ad``'s metadata onto NUDGE's condition schema.

    Only ``condition_col`` and the control identification are required; all else is
    an optional filter/rename with a safe default, so the config stays small for simple
    files and expressive for messy ones.
    """

    #: obs column whose value names the perturbed gene → the condition label.
    condition_col: str
    #: obs column that distinguishes control (non-targeting) from targeting guides.
    control_col: str
    #: values of ``control_col`` that mark a control cell (matched case-insensitively).
    control_values: Sequence[str] = ("non-targeting", "NTC", "control")
    #: emitted ``obs["condition"]`` label for control cells.
    control_label: str = "WT"
    #: var column holding gene symbols; if given, ``var_names`` is set from it (raw
    #: ``var_names`` are often Ensembl IDs). ``None`` → use existing ``var_names``.
    gene_symbol_col: str | None = None
    #: keep only these gene symbols (the readout panel); ``None`` → all genes.
    gene_subset: Sequence[str] | None = None
    #: keep only these perturbed genes as conditions (plus controls); ``None`` → all.
    target_genes: Sequence[str] | None = None
    #: boolean obs column of cells to DROP (e.g. a low-quality flag); None → keep all.
    quality_drop_col: str | None = None
    #: obs column of guide IDs, used with ``exclude_guides`` to drop ambiguous calls.
    guide_id_col: str | None = None
    #: guide-ID values to drop (e.g. "multi-guide").
    exclude_guides: Sequence[str] = ()
    #: per-cell QC: minimum total counts / minimum genes detected.
    min_counts: int = 0
    min_genes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def _obs_series(obs: Any, col: str) -> pd.Series:
    if col not in obs.columns:
        raise KeyError(f"obs column {col!r} not found; have {list(obs.columns)[:20]}…")
    return obs[col].astype(str)


def _control_mask(obs: Any, cfg: PerturbLoaderConfig) -> np.ndarray:
    vals = {v.lower() for v in cfg.control_values}
    return _obs_series(obs, cfg.control_col).str.lower().isin(vals).to_numpy()


def load_perturbseq(
    path: str,
    config: PerturbLoaderConfig,
    *,
    backed: bool = True,
) -> Any:
    """Load + standardize a Perturb-seq ``.h5ad`` into NUDGE's condition schema.

    Returns an in-memory AnnData of only the selected cells × genes: ``.X`` raw integer
    counts (dense), ``.obs["condition"]`` (``control_label`` for controls, else the
    perturbed gene), ``var_names`` = symbols. Runs ``check_counts`` before returning.

    ``backed=True`` (default, the point) keeps the full count matrix on disk and reads
    only the masked block. Set ``backed=False`` only for tiny files/tests.
    """
    import anndata as ad

    adata: Any = ad.read_h5ad(path, backed="r" if backed else None)

    # --- cell mask, entirely from the (small) obs metadata ---
    obs = adata.obs
    is_control = _control_mask(obs, config)
    keep = np.ones(adata.n_obs, dtype=bool)
    if config.target_genes is not None:
        wanted = set(config.target_genes)
        is_target = _obs_series(obs, config.condition_col).isin(wanted).to_numpy()
        keep &= is_control | is_target
    if config.quality_drop_col is not None:
        keep &= ~obs[config.quality_drop_col].to_numpy(dtype=bool)
    if config.guide_id_col is not None and config.exclude_guides:
        excl = set(config.exclude_guides)
        keep &= ~_obs_series(obs, config.guide_id_col).isin(excl).to_numpy()

    # --- gene mask from var metadata ---
    if config.gene_symbol_col is not None:
        symbols = adata.var[config.gene_symbol_col].astype(str)
    else:
        symbols = pd.Series(adata.var_names.astype(str), index=adata.var_names)
    if config.gene_subset is not None:
        gene_keep = symbols.isin(set(config.gene_subset)).to_numpy()
        missing = set(config.gene_subset) - set(symbols[gene_keep])
        if missing:
            raise KeyError(f"gene_subset symbols absent from file: {sorted(missing)}")
    else:
        gene_keep = np.ones(adata.n_vars, dtype=bool)

    # --- materialize ONLY the block (backed → reads just these rows/cols from disk) ---
    sub: Any = adata[keep, gene_keep]
    sub = sub.to_memory() if backed else sub.copy()

    x: Any = sub.X
    if hasattr(x, "toarray"):  # scipy sparse → dense (the subset is small)
        x = x.toarray()
    sub.X = np.asarray(x)

    # --- rename genes to symbols, build the condition column ---
    if config.gene_symbol_col is not None:
        sub.var_names = pd.Index(
            sub.var[config.gene_symbol_col].astype(str), name=sub.var_names.name
        )
    cond = _obs_series(sub.obs, config.condition_col).to_numpy().copy()
    cond[_control_mask(sub.obs, config)] = config.control_label
    sub.obs["condition"] = pd.Categorical(cond)

    # --- per-cell QC on the small materialized subset ---
    if config.min_counts > 0 or config.min_genes > 0:
        totals = np.asarray(sub.X).sum(axis=1)
        ngenes = (np.asarray(sub.X) > 0).sum(axis=1)
        cell_ok = (totals >= config.min_counts) & (ngenes >= config.min_genes)
        sub = sub[cell_ok].copy()

    check_counts(sub, readout_genes=config.gene_subset or ())
    return sub
