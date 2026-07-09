"""Micro-benchmark the CSR row-gather at the heart of ``_read_h5ad_rows``.

The loader's hot path (prior profiling: ~99% of loader time) is the h5py fancy-index
gather ``node["data"][flat]`` where ``flat`` is the concatenation of the selected rows'
``indptr`` ranges. On SCATTERED rows (per-condition subsampling) ``flat`` is a large,
mostly-non-contiguous int array — h5py's fancy-index path is slow.

This script builds a synthetic CSR ``.h5ad`` with realistically scattered selected rows
and times candidate I/O-proportional gathers against the current implementation, on both
an uncompressed and a gzip-compressed file (real screens are often compressed). Every
strategy is checked byte-identical to the current one before timing.

Run: ``uv run python scripts/perf/bench_loader_gather.py``  (seconds; small synthetic).
"""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Callable

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


def make_file(path: str, n_cells: int, n_genes: int, nnz_per: int,
              *, compression: str | None) -> None:
    rng = np.random.default_rng(0)
    indptr = np.arange(0, n_cells * nnz_per + 1, nnz_per)
    indices = rng.integers(0, n_genes, size=n_cells * nnz_per).astype(np.int32)
    data = rng.integers(1, 6, size=n_cells * nnz_per).astype(np.int32)
    X = sp.csr_matrix((data, indices, indptr), shape=(n_cells, n_genes))
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    a = ad.AnnData(X=X, obs=obs, var=var)
    if compression:
        a.write_h5ad(path, compression=compression)
    else:
        a.write_h5ad(path)


def scattered_rows(n_cells: int, n_sel: int, seed: int = 0) -> np.ndarray:
    """A realistically-scattered selection: many small clusters (per-guide cells)."""
    rng = np.random.default_rng(seed)
    centres = rng.integers(0, n_cells, size=n_sel // 3 + 1)
    rows = np.concatenate([c + rng.integers(0, 4, size=3) for c in centres])
    rows = np.unique(np.clip(rows, 0, n_cells - 1))[:n_sel]
    return np.sort(rows)


def _flat_from_rows(indptr: np.ndarray, rows: np.ndarray) -> np.ndarray:
    starts, ends = indptr[rows], indptr[rows + 1]
    return np.concatenate(
        [np.arange(s, e) for s, e in zip(starts, ends, strict=True)]
    )


def _runs(flat: np.ndarray) -> list[tuple[int, int]]:
    """Coalesce a SORTED index array into contiguous ``[lo, hi)`` runs."""
    if flat.size == 0:
        return []
    brk = np.nonzero(np.diff(flat) != 1)[0]
    starts = np.concatenate([[flat[0]], flat[brk + 1]])
    ends = np.concatenate([flat[brk], [flat[-1]]]) + 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


# --- gather strategies: each returns (data, inds) for the selected rows ---------------
def gather_current(path: str, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        node = f["X"]
        indptr = node["indptr"][:]
        flat = _flat_from_rows(indptr, rows)
        return node["data"][flat], node["indices"][flat]


def gather_coalesced(path: str, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        node = f["X"]
        indptr = node["indptr"][:]
        flat = _flat_from_rows(indptr, rows)
        runs = _runs(flat)
        dset_d, dset_i = node["data"], node["indices"]
        dparts = [dset_d[lo:hi] for lo, hi in runs]
        iparts = [dset_i[lo:hi] for lo, hi in runs]
        d = np.concatenate(dparts) if dparts else np.zeros(0, dset_d.dtype)
        i = np.concatenate(iparts) if iparts else np.zeros(0, dset_i.dtype)
        return d, i


def gather_coalesced_span(path: str, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read one contiguous span min(flat)->max(flat)+1 then index in RAM.

    Only I/O-proportional when the selection is dense within its span; included to show
    the failure mode the constraint warns about (reads most of the file for scattered).
    """
    with h5py.File(path, "r") as f:
        node = f["X"]
        indptr = node["indptr"][:]
        flat = _flat_from_rows(indptr, rows)
        lo, hi = int(flat[0]), int(flat[-1]) + 1
        d = node["data"][lo:hi][flat - lo]
        i = node["indices"][lo:hi][flat - lo]
        return d, i


def gather_rdcc(path: str, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fancy index, but with a large chunk cache (rdcc) so chunk re-hits are cheap."""
    cache = dict(rdcc_nbytes=256 * 1024 * 1024, rdcc_nslots=1_000_003)
    with h5py.File(path, "r", **cache) as f:
        node = f["X"]
        indptr = node["indptr"][:]
        flat = _flat_from_rows(indptr, rows)
        return node["data"][flat], node["indices"][flat]


def gather_coalesced_threaded(
    path: str, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Coalesced runs read in parallel threads (h5py releases the GIL in C reads)."""
    from concurrent.futures import ThreadPoolExecutor
    with h5py.File(path, "r") as f:
        node = f["X"]
        indptr = node["indptr"][:]
        flat = _flat_from_rows(indptr, rows)
        runs = _runs(flat)
        dset_d, dset_i = node["data"], node["indices"]

        def rd(args: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
            lo, hi = args
            return dset_d[lo:hi], dset_i[lo:hi]

        with ThreadPoolExecutor(max_workers=8) as ex:
            parts = list(ex.map(rd, runs)) if runs else []
        dp = [p[0] for p in parts]
        ip = [p[1] for p in parts]
        d = np.concatenate(dp) if dp else np.zeros(0, dset_d.dtype)
        i = np.concatenate(ip) if ip else np.zeros(0, dset_i.dtype)
        return d, i


STRATEGIES: dict[str, Callable[[str, np.ndarray], tuple[np.ndarray, np.ndarray]]] = {
    "current (fancy-index)": gather_current,
    "coalesced-runs (slice)": gather_coalesced,
    "coalesced-span (mask)": gather_coalesced_span,
    "rdcc-256MB (fancy)": gather_rdcc,
    "coalesced+threads": gather_coalesced_threaded,
}


def time_it(fn: Callable[..., object], *a: object, reps: int = 5) -> float:
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn(*a)
        best = min(best, time.perf_counter() - t)
    return best


def run(compression: str | None) -> None:
    n_cells, n_genes, nnz_per, n_sel = 40_000, 500, 60, 2000
    tag = compression or "none"
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, f"bench_{tag}.h5ad")
        make_file(path, n_cells, n_genes, nnz_per, compression=compression)
        rows = scattered_rows(n_cells, n_sel)
        size_mb = os.path.getsize(path) / 1e6
        with h5py.File(path) as f:
            total_nnz = f["X"]["data"].shape[0]
            flat = _flat_from_rows(f["X"]["indptr"][:], rows)
        runs = _runs(flat)
        sel_frac = flat.size / total_nnz
        print(f"\n=== compression={tag}  file={size_mb:.1f} MB  "
              f"cells={n_cells} sel_rows={len(rows)} ===")
        print(f"    selected nnz={flat.size} ({sel_frac:.2%} of matrix)  "
              f"coalesced into {len(runs)} runs")
        ref = gather_current(path, rows)
        base = None
        for name, fn in STRATEGIES.items():
            d, i = fn(path, rows)
            ok = np.array_equal(d, ref[0]) and np.array_equal(i, ref[1])
            t = time_it(fn, path, rows)
            if base is None:
                base = t
            speed = base / t
            print(f"    {name:26s} {t*1000:8.1f} ms  {speed:5.1f}x  "
                  f"{'OK' if ok else 'MISMATCH!!'}")


if __name__ == "__main__":
    for comp in (None, "gzip"):
        run(comp)
