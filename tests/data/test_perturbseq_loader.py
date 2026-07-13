"""The generic backed-mode Perturb-seq loader (Phase 4 M1).

Two things matter: (1) it maps any guide-assigned ``.h5ad`` onto NUDGE's condition
schema (control → "WT", perturbed gene → condition; gene panel subset; QC), verified on
a fixture that mimics the Gladstone schema; and (2) **backed mode** materializes only
the selected block — the count matrix is never fully loaded — proven by a peak-memory
test on a deliberately larger file (the requirement: these files are ~150 GB).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from nudge.data.loaders.perturbseq import PerturbLoaderConfig, load_perturbseq

# gene symbols (the panel + extras); var_names are Ensembl-like, symbols live in var.
_SYMBOLS = ["SOS1", "RASGRP1", "IL2", "MT-CO1", "ACTB"]
_ENSEMBL = [f"ENSG{i:03d}" for i in range(len(_SYMBOLS))]

CONFIG = PerturbLoaderConfig(
    condition_col="perturbed_gene_name",
    control_col="guide_type",
    control_values=("non-targeting",),
    gene_symbol_col="gene_name",
    gene_subset=("SOS1", "IL2"),
    quality_drop_col="low_quality",
    guide_id_col="guide_id",
    exclude_guides=("multi-guide",),
)


def _write_fixture(path: str) -> None:
    """A small CSR-count ``.h5ad`` with the Gladstone obs/var schema."""
    rng = np.random.default_rng(0)
    blocks = [  # (guide_type, perturbed_gene_name, guide_id, low_quality, n)
        ("non-targeting", "NA", "ntc-1", False, 20),
        ("targeting", "SOS1", "SOS1-1", False, 20),
        ("targeting", "RASGRP1", "RASGRP1-1", False, 20),
        ("targeting", "SOS1", "SOS1-2", True, 5),  # low-quality → dropped
        ("targeting", "SOS1", "multi-guide", False, 5),  # ambiguous → dropped
    ]
    rows = []
    for gtype, gene, gid, lowq, n in blocks:
        for _ in range(n):
            rows.append((gtype, gene, gid, lowq))
    obs = pd.DataFrame(
        rows, columns=["guide_type", "perturbed_gene_name", "guide_id", "low_quality"]
    )
    obs.index = pd.Index([f"cell_{i}" for i in range(len(obs))])
    counts = rng.poisson(3.0, size=(len(obs), len(_SYMBOLS))).astype(np.int32)
    var = pd.DataFrame({"gene_name": _SYMBOLS, "gene_ids": _ENSEMBL})
    var.index = pd.Index(_ENSEMBL)
    ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var).write_h5ad(path)


def test_loader_maps_conditions_and_subsets(tmp_path) -> None:
    p = str(tmp_path / "fixture.h5ad")
    _write_fixture(p)
    adata = load_perturbseq(p, CONFIG)

    assert set(adata.var_names) == {"SOS1", "IL2"}  # panel subset, symbols not Ensembl
    conds = set(adata.obs["condition"])
    assert conds == {"WT", "SOS1", "RASGRP1"}  # NTC → WT; targets kept
    assert adata.n_obs == 60  # 20+20+20; low-quality (5) + multi-guide (5) dropped
    x = np.asarray(adata.X)
    assert x.dtype.kind in "iu" or np.allclose(x, np.rint(x))  # raw integer
    assert (x >= 0).all()


def test_target_genes_filters_conditions(tmp_path) -> None:
    p = str(tmp_path / "fixture.h5ad")
    _write_fixture(p)
    cfg = PerturbLoaderConfig(**{**CONFIG.__dict__, "target_genes": ("SOS1",)})
    adata = load_perturbseq(p, cfg)
    assert set(adata.obs["condition"]) == {"WT", "SOS1"}  # RASGRP1 excluded


def test_missing_panel_gene_raises(tmp_path) -> None:
    p = str(tmp_path / "fixture.h5ad")
    _write_fixture(p)
    cfg = PerturbLoaderConfig(**{**CONFIG.__dict__, "gene_subset": ("SOS1", "NOPE")})
    with pytest.raises(KeyError, match="NOPE"):
        load_perturbseq(p, cfg)


def test_min_counts_qc_filters_cells(tmp_path) -> None:
    p = str(tmp_path / "fixture.h5ad")
    _write_fixture(p)
    # panel is 2 genes × Poisson(3) → ~6 counts/cell; a high threshold drops most.
    cfg = PerturbLoaderConfig(**{**CONFIG.__dict__, "min_counts": 1000})
    adata = load_perturbseq(p, cfg)
    assert adata.n_obs == 0


# --- the load-bearing requirement: backed mode never loads the full matrix ------------
_MEM_PROBE = textwrap.dedent(
    """
    import sys, resource, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
    from nudge.data.loaders.perturbseq import PerturbLoaderConfig, load_perturbseq
    path, mode = sys.argv[1], sys.argv[2]
    if mode == "full":
        a = ad.read_h5ad(path)          # loads the whole matrix
        _ = np.asarray(a.X.todense() if sp.issparse(a.X) else a.X)
    else:
        cfg = PerturbLoaderConfig(
            condition_col="perturbed_gene_name", control_col="guide_type",
            control_values=("non-targeting",), gene_symbol_col="gene_name",
            gene_subset=("g0",), target_genes=("SOS1",),
        )
        _ = load_perturbseq(path, cfg)   # backed: reads only the selected rows/cols
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)  # peak RSS (KB)
    """
)


def _peak_rss_kb(path: str, mode: str) -> int:
    out = subprocess.run(
        [sys.executable, "-c", _MEM_PROBE, path, mode],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip().splitlines()[-1])


@pytest.mark.slow
def test_backed_mode_bounds_peak_memory(tmp_path) -> None:
    # FULL matrix is ~480 MB dense, but only ~200 of 100k cells are selected. Backed
    # mode reads only those rows → peak RSS far below a full load. (Sized above the
    # ~320 MB JAX import spike so the matrix, not imports, dominates the signal.)
    n_cells, n_genes, nnz = 100_000, 1200, 40
    rng = np.random.default_rng(0)
    indptr = np.arange(0, n_cells * nnz + 1, nnz)
    indices = rng.integers(0, n_genes, size=n_cells * nnz)
    data = rng.integers(1, 6, size=n_cells * nnz).astype(np.int32)
    x = sp.csr_matrix((data, indices, indptr), shape=(n_cells, n_genes))
    gtype = np.where(np.arange(n_cells) < 100, "non-targeting", "targeting")
    gene = np.where(np.arange(n_cells) < 200, "SOS1", "OTHER").astype(object)
    gene[:100] = "NA"
    obs = pd.DataFrame(
        {"guide_type": gtype, "perturbed_gene_name": gene},
        index=pd.Index([f"c{i}" for i in range(n_cells)]),
    )
    var = pd.DataFrame(
        {"gene_name": [f"g{i}" for i in range(n_genes)]},
        index=pd.Index([f"ENSG{i}" for i in range(n_genes)]),
    )
    p = str(tmp_path / "big.h5ad")
    ad.AnnData(X=x, obs=obs, var=var).write_h5ad(p)

    full = _peak_rss_kb(p, "full")
    backed = _peak_rss_kb(p, "backed")
    full_matrix_kb = n_cells * n_genes * 4 // 1024  # ~469 MB dense
    # Backed mode reads only the selected rows, so it must save a meaningful chunk of the
    # dense materialization. The bar is deliberately conservative (0.15, not 0.40): peak
    # RSS on shared CI runners carries a large, variable import/allocator baseline (JAX +
    # anndata) that partially hides the transient dense-load peak and compresses the
    # measured delta run-to-run. 0.15 still cleanly separates "backed works" (measured
    # ~0.20-0.45 of the matrix across runners) from "backed broken" (loads the full matrix
    # -> delta ~= 0), which is the property under test — it is not a tight memory budget.
    assert backed < full, (backed, full, full_matrix_kb)
    assert full - backed > full_matrix_kb * 0.15, (backed, full, full_matrix_kb)
