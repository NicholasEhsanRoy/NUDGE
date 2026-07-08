"""Ingestion guardrail — NUDGE owns the count model (the "bouncer").

NUDGE fits **raw integer counts** with its own negative-binomial observation
model, so the standard scanpy/Seurat pipeline (log1p, CPM, scaling, imputation,
batch correction) is *hostile* to it and fails *silently* — the fit runs, the
answer is just wrong. ``check_counts`` inspects an ``AnnData`` and fails **loudly**
on the input before any of that can happen (design/WORKING_BACKWARDS.md Part 5).

Hard violations raise ``IngestError``; softer signals of preprocessing emit a
``warnings.warn``. Detectable from values alone: negatives (scaled/centered),
non-integers (log/CPM/imputed), missing readout genes (HVG/panel subsetting).
Not detectable from values alone (documented limitation): **pseudobulk** — it is
integer counts, just aggregated — is a shape/semantics issue the caller must avoid.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np

__all__ = ["IngestError", "check_counts"]

# obsm keys that betray a normalization/integration pipeline has run.
_CORRECTED_OBSM = ("X_pca", "X_umap", "X_tsne", "X_scvi", "X_scanvi", "X_harmony")
# layer names that usually hold the *real* raw counts (so .X may not be raw).
_RAW_LAYER_NAMES = ("counts", "raw", "raw_counts", "umi", "spliced")


class IngestError(ValueError):
    """Raised when input data violates NUDGE's raw integer-count contract."""


def _values_and_min(x: Any) -> tuple[np.ndarray, float]:
    """Return the (flattened stored values, minimum) of a dense or sparse matrix."""
    if x is None:
        raise IngestError("adata.X is None — NUDGE needs a raw integer count matrix.")
    if hasattr(x, "toarray") and hasattr(x, "data"):  # scipy sparse
        data = np.asarray(x.data).ravel()
        return data, (float(x.min()) if x.nnz else 0.0)
    arr = np.asarray(x)
    return arr.ravel(), (float(arr.min()) if arr.size else 0.0)


def check_counts(adata: Any, *, readout_genes: Iterable[str] = ()) -> None:
    """Validate that ``adata.X`` is raw integer counts and readout genes are present.

    Raises ``IngestError`` on hard violations (negative, non-integer, or missing
    readout genes); warns on softer signals (corrected embeddings, a raw-count
    layer suggesting ``.X`` is normalized). "Fails safely and loudly" — on the
    *input*, not just the output.
    """
    values, minimum = _values_and_min(adata.X)

    if minimum < 0:
        raise IngestError(
            "adata.X has negative values — looks scaled/centered, not raw counts. "
            "NUDGE owns the count model and needs raw integer UMI counts."
        )
    if values.size and not np.allclose(values, np.rint(values), atol=1e-6):
        raise IngestError(
            "adata.X has non-integer values — looks normalized/log-transformed "
            "(log1p, CPM/TPM, scaled, or imputed). NUDGE needs raw integer counts; "
            "pass the raw-count layer (e.g. adata.layers['counts'])."
        )

    var_names = set(getattr(adata, "var_names", []))
    missing = [g for g in readout_genes if g not in var_names]
    if missing:
        raise IngestError(
            f"readout genes absent from adata.var_names: {missing}. "
            "HVG selection or gene-panel subsetting may have dropped them."
        )

    # --- Softer signals: the data may have been through a pipeline ---
    layers = getattr(adata, "layers", {}) or {}
    present_raw_layers = [name for name in _RAW_LAYER_NAMES if name in layers]
    if present_raw_layers:
        warnings.warn(
            f"adata has a raw-count layer {present_raw_layers}; confirm adata.X is "
            "the raw counts and not a normalized layer.",
            stacklevel=2,
        )
    obsm = getattr(adata, "obsm", {}) or {}
    corrected = [name for name in _CORRECTED_OBSM if name in obsm]
    if corrected:
        warnings.warn(
            f"adata has corrected embeddings {corrected}; NUDGE fits raw counts and "
            "models batch itself — do not pass batch-corrected expression.",
            stacklevel=2,
        )
