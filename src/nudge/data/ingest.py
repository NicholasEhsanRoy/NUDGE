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

__all__ = ["IngestError", "check_counts", "check_readout"]

#: The readout modalities NUDGE knows how to route. ``"counts"`` is the raw-integer-UMI
#: path (the existing negative-binomial count model); the continuous modalities feed the
#: dose-response path (:mod:`nudge.inference.cross_modality`). NUDGE never *guesses* a
#: modality — the caller must declare it, and the bouncer refuses ambiguous input.
_CONTINUOUS_MODALITIES = ("fluorescence", "activity", "foldchange")
READOUT_MODALITIES = ("counts", *_CONTINUOUS_MODALITIES)

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


def _readout_values(x: Any, readout_col: str | None) -> np.ndarray:
    """Extract the 1-D readout vector from an array, a Series, or a DataFrame column."""
    if readout_col is not None and hasattr(x, "columns"):  # pandas DataFrame
        if readout_col not in x.columns:
            raise IngestError(
                f"readout_col {readout_col!r} not in columns {list(x.columns)}"
            )
        return np.asarray(x[readout_col], dtype=float).ravel()
    if hasattr(x, "X"):  # an AnnData passed to a continuous path is a category error
        raise IngestError(
            "an AnnData was passed with a continuous modality; NUDGE reads it "
            "from a table (array or DataFrame column), not a count matrix. "
            "Use modality='counts' for an AnnData."
        )
    return np.asarray(x, dtype=float).ravel()


def _check_continuous_readout(values: np.ndarray, modality: str) -> None:
    """The continuous-readout bouncer: refuse ambiguous / mislabeled input (LIM-008).

    NUDGE never guesses a modality: the caller declares ``fluorescence`` / ``activity``
    / ``foldchange``, and this refuses anything that is not plausibly a *continuous*
    single-channel measurement. The sharp failure it prevents is silently accepting
    **log-normalized counts** (or raw counts) dressed up as fluorescence, which would
    feed a garbage curve into the dose-response fit. Two witnesses:

    - **all-integer values**: raw counts mislabeled as continuous (real fluorescence /
      fold-change is continuous, not integer-quantized);
    - **zero-inflation** (a large fraction of *exact* zeros beside a continuous tail):
      the fingerprint of ``log1p``-normalized single-cell counts (dropout zeros), which
      a bulk fluorescence / fold-change measurement does not produce.

    Also refuses non-finite values (NaN/Inf) and substantially-negative values (a
    continuous readout is non-negative; negatives betray scaled / centered / log-ratio
    data).
    """
    if values.size == 0:
        raise IngestError(f"empty {modality} readout — nothing to validate.")
    if not np.all(np.isfinite(values)):
        raise IngestError(
            f"{modality} readout has non-finite values (NaN/Inf); clean the data first."
        )
    # A continuous readout is non-negative. A *few* tiny negatives near zero are okay
    # (background-subtracted fold-change noise), but a large fraction of negatives, or a
    # deep negative excursion, betrays scaled / centered / log-ratio data (roughly
    # symmetric about zero), which NUDGE refuses rather than fit.
    peak = max(float(values.max()), 1e-12)
    neg_frac = float(np.mean(values < 0.0))
    if neg_frac > 0.25 or float(values.min()) < -0.15 * peak:
        raise IngestError(
            f"{modality} readout has substantial negative values ({neg_frac:.0%} of "
            f"points, min {float(values.min()):.3g}); this looks scaled / centered or "
            "a log-ratio, not a non-negative continuous signal. NUDGE refuses "
            "ambiguous input rather than fit it (NUDGE-LIM-008)."
        )
    if np.allclose(values, np.rint(values), atol=1e-9):
        raise IngestError(
            f"{modality} readout is all-integer; these look like raw COUNTS mislabeled "
            "as continuous. Pass raw counts with modality='counts' (NUDGE owns the "
            "count model); a continuous readout must be truly continuous (LIM-008)."
        )
    zero_frac = float(np.mean(values == 0.0))
    if zero_frac >= 0.30:
        raise IngestError(
            f"{modality} readout is {zero_frac:.0%} exact zeros beside a continuous "
            "tail; the fingerprint of LOG-NORMALIZED single-cell counts (dropout "
            "zeros), not a bulk fold-change / fluorescence measurement. NUDGE refuses "
            "this ambiguous input rather than silently fit it (NUDGE-LIM-008)."
        )


def check_readout(
    x: Any,
    *,
    modality: str = "counts",
    readout_col: str | None = None,
    readout_genes: Iterable[str] = (),
) -> None:
    """Modality-aware ingestion bouncer — route by an **explicit** modality declaration.

    ``modality="counts"`` delegates to :func:`check_counts` (the raw-integer-UMI guard,
    unchanged). The continuous modalities (``"fluorescence"``, ``"activity"``,
    ``"foldchange"``) route to the continuous-readout bouncer
    (:func:`_check_continuous_readout`), which validates a 1-D readout vector (from an
    array, a ``pandas.Series``, or ``x[readout_col]`` of a DataFrame) and **refuses**
    ambiguous / mislabeled input — most sharply, log-normalized or raw counts dressed up
    as fluorescence (NUDGE-LIM-008). NUDGE never guesses the modality; an unknown one
    raises. This is the cross-modality adapter's fail-safe (``NUDGE-METHOD-002``).
    """
    if modality == "counts":
        check_counts(x, readout_genes=readout_genes)
        return
    if modality not in _CONTINUOUS_MODALITIES:
        raise IngestError(
            f"unknown readout modality {modality!r}; declare one of "
            f"{READOUT_MODALITIES}. NUDGE never guesses a modality — it refuses "
            "ambiguous input."
        )
    _check_continuous_readout(_readout_values(x, readout_col), modality)
