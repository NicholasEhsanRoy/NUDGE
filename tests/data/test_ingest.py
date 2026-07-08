"""The ingestion guardrail rejects non-raw input loudly and passes real counts."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from nudge.data.ingest import IngestError, check_counts


def _raw_adata() -> ad.AnnData:
    counts = np.array([[0, 3, 1], [2, 0, 5], [1, 1, 0]], dtype=np.int32)
    return ad.AnnData(X=counts, var=pd.DataFrame(index=pd.Index(["g0", "g1", "g2"])))


def test_raw_counts_pass() -> None:
    check_counts(_raw_adata())  # no raise


def test_readout_genes_present_pass() -> None:
    check_counts(_raw_adata(), readout_genes=["g0", "g2"])


def test_log1p_rejected() -> None:
    adata = _raw_adata()
    adata.X = np.log1p(adata.X).astype(np.float32)  # normalized → non-integer
    with pytest.raises(IngestError, match="non-integer"):
        check_counts(adata)


def test_scaled_negative_rejected() -> None:
    adata = _raw_adata()
    adata.X = (adata.X - adata.X.mean()).astype(np.float32)  # centered → negatives
    with pytest.raises(IngestError, match="negative"):
        check_counts(adata)


def test_missing_readout_gene_rejected() -> None:
    with pytest.raises(IngestError, match="absent"):
        check_counts(_raw_adata(), readout_genes=["g0", "MISSING"])


def test_corrected_embedding_warns() -> None:
    adata = _raw_adata()
    adata.obsm["X_pca"] = np.zeros((adata.n_obs, 2), dtype=np.float32)
    with pytest.warns(UserWarning, match="corrected embeddings"):
        check_counts(adata)


def test_raw_layer_warns() -> None:
    adata = _raw_adata()
    adata.layers["counts"] = adata.X.copy()
    with pytest.warns(UserWarning, match="raw-count layer"):
        check_counts(adata)
