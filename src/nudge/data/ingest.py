"""Ingestion guardrail — NUDGE owns the count model.

NUDGE fits raw integer counts with its own negative-binomial + dropout model, so
the standard scanpy/Seurat pipeline (log1p, pseudobulk, imputation, batch
correction) is *hostile* to it and fails silently. ``check_counts`` inspects an
``AnnData`` and fails **loudly** on non-raw input or missing readout genes —
"fails safely and loudly" applied to the input, not just the output. Phase-0 stub.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__ = ["check_counts"]


def check_counts(adata: Any, *, readout_genes: Iterable[str] = ()) -> None:
    """Raise if ``adata.X`` is not raw integer counts, or readout genes are absent.

    Phase-0 stub: the concrete checks (integer dtype, no corrected layers/obsm,
    readout-gene presence) land in Phase 1.
    """
    raise NotImplementedError("check_counts — Phase 1")
