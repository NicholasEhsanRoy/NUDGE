"""MDSINE2 "Gibson" gnotobiotic cohort → :class:`GLVDataset` adapter (Task 3: push the boundary).

Gerber lab MDSINE2_Paper, ``datasets/gibson/healthy/raw_tables/`` (small TSVs; the raw input,
NOT the 18.7 GB Zenodo model output). Fetched files (a few MB):

* ``counts.tsv``       — 1088 ASVs × 339 samples (16S amplicon counts).
* ``qpcr.tsv``         — triplicate total-load qPCR per sample → converts relative → ABSOLUTE.
* ``metadata.tsv``     — sampleID | subject | time (days). Subjects 2–5: ~75–77 dense,
                         twice-daily samples over 65 days. Subject 1: sparse (15).
* ``perturbations.tsv``— three TIMED pulses on subjects 2–5: High Fat Diet [21.5,28.5],
                         **Vancomycin** [35.5,42.5] (gram-positive antibiotic), Gentamicin
                         [50.5,57.5] (gram-negative).
* ``rdp_species.tsv``  — ASV → Kingdom..Genus..Species taxonomy (for aggregation).

**The hypothesis under test (the user's):** denser sampling (twice-daily) + a strong antibiotic
pulse → higher dimensions become identifiable than Stein's sparse 8-timepoint slice.

**The NUDGE contrast (a within-subject before/during design).** MDSINE2 has no untreated
parallel arm for subjects 2–5, so each mouse is its OWN control across time: reference = the
immediate PRE-vancomycin window (post-HFD-recovery, drug-free ⇒ ε=0), perturbed = the
vancomycin window + recovery, both re-timed to a common local axis with the vancomycin pulse as
``u(t)``. Gram-positive genera the drug directly kills should resolve on the ``ε`` axis; the rest
should abstain as the ``k²`` interaction matrix outgrows the data.

Additive real-data adapter under ``scripts/vv/``; imports the SHIPPED
:mod:`nudge.inference.lotka_volterra` read-only.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nudge.inference.lotka_volterra import GLVDataset, GLVParams

PERTURBED_SUBJECTS = ("2", "3", "4", "5")
#: (name, start_day, end_day) — from perturbations.tsv.
VANCOMYCIN = ("Vancomycin", 35.5, 42.5)
GENTAMICIN = ("Gentamicin", 50.5, 57.5)


@dataclass(frozen=True)
class MDSRaw:
    counts: np.ndarray  # (n_asv, n_sample)
    asv_ids: list[str]
    sample_ids: list[str]
    genus: list[str]  # per-ASV genus label (NA -> "Other")
    family: list[str]
    qpcr: dict[str, float]  # sampleID -> mean total load
    subject: dict[str, str]  # sampleID -> subject
    time: dict[str, float]  # sampleID -> day


def _read_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    return rows[0], rows[1:]


def load_mdsine2(raw_dir: str | Path) -> MDSRaw:
    """Parse the five MDSINE2 raw tables into aligned arrays."""
    raw_dir = Path(raw_dir)
    # counts
    header, body = _read_tsv(raw_dir / "counts.tsv")
    sample_ids = [h.strip('"') for h in header[1:]]
    asv_ids = [r[0].strip('"') for r in body]
    counts = np.array([[float(v) for v in r[1:]] for r in body])  # (n_asv, n_sample)
    # taxonomy
    thead, tbody = _read_tsv(raw_dir / "rdp_species.tsv")
    gi, fi = thead.index("Genus"), thead.index("Family")
    tax = {r[0].strip('"'): r for r in tbody}
    genus = [(tax[a][gi] if tax[a][gi] not in ("NA", "") else "Other") for a in asv_ids]
    family = [(tax[a][fi] if tax[a][fi] not in ("NA", "") else "Other") for a in asv_ids]
    # qpcr (triplicate mean)
    qhead, qbody = _read_tsv(raw_dir / "qpcr.tsv")
    qpcr = {}
    for r in qbody:
        vals = [float(x) for x in r[1:] if x not in ("", "NA")]
        if r[0] and vals:
            qpcr[r[0]] = float(np.mean(vals))
    # metadata
    mhead, mbody = _read_tsv(raw_dir / "metadata.tsv")
    subject = {r[0]: r[1] for r in mbody}
    time = {r[0]: float(r[2]) for r in mbody}
    return MDSRaw(counts, asv_ids, sample_ids, genus, family, qpcr, subject, time)


def absolute_abundance(raw: MDSRaw) -> np.ndarray:
    """Per-sample ABSOLUTE abundance = (ASV count / sample total) × qPCR total load. (n_asv, n_sample)."""
    tot = raw.counts.sum(axis=0)  # (n_sample,)
    tot = np.where(tot > 0, tot, 1.0)
    rel = raw.counts / tot
    q = np.array([raw.qpcr.get(s, np.nan) for s in raw.sample_ids])
    return rel * q[None, :]


def top_genus_panel(raw: MDSRaw, k: int, *, level: str = "genus") -> tuple[list[str], np.ndarray]:
    """Pick the ``k-1`` most-abundant taxa at ``level`` + an aggregated 'Other' → k groups.

    Returns (group labels, membership matrix ``(k, n_asv)`` of 0/1). The membership sums ASV
    absolute abundances within each taxon (densities are additive).
    """
    labels_per_asv = raw.genus if level == "genus" else raw.family
    abs_ab = absolute_abundance(raw)
    # rank taxa by total absolute abundance across the perturbed subjects.
    keep_cols = [i for i, s in enumerate(raw.sample_ids) if raw.subject.get(s) in PERTURBED_SUBJECTS]
    tot_by_taxon: dict[str, float] = {}
    for i, lab in enumerate(labels_per_asv):
        tot_by_taxon[lab] = tot_by_taxon.get(lab, 0.0) + float(abs_ab[i, keep_cols].sum())
    ranked = sorted((t for t in tot_by_taxon if t != "Other"), key=lambda t: -tot_by_taxon[t])
    top = ranked[: k - 1]
    labels = [*top, "Other"]
    memb = np.zeros((k, len(raw.asv_ids)))
    lab_to_row = {lab: r for r, lab in enumerate(top)}
    for i, lab in enumerate(labels_per_asv):
        memb[lab_to_row.get(lab, k - 1), i] = 1.0  # unranked/NA -> Other row
    return labels, memb


def _subject_series(
    raw: MDSRaw, abs_ab: np.ndarray, subject: str
) -> tuple[np.ndarray, np.ndarray]:
    """(times, (T, n_asv) absolute-abundance trajectory) for one subject, time-sorted."""
    cols = [(raw.time[s], i) for i, s in enumerate(raw.sample_ids) if raw.subject.get(s) == subject]
    cols.sort()
    ts = np.array([t for t, _ in cols])
    idx = [i for _, i in cols]
    return ts, abs_ab[:, idx].T  # (T, n_asv)


def _interp(times: np.ndarray, traj: np.ndarray, grid: np.ndarray) -> np.ndarray:
    out = np.empty((len(grid), traj.shape[1]))
    for s in range(traj.shape[1]):
        out[:, s] = np.interp(grid, times, traj[:, s])
    return np.clip(out, 0.0, None)


def build_mdsine2_dataset(
    raw: MDSRaw,
    *,
    k: int,
    level: str = "genus",
    perturbation: tuple[str, float, float] = VANCOMYCIN,
    ref_span: float = 6.0,
    post_span: float = 6.0,
    n_grid: int = 14,
    dt: float = 0.1,
    normalize: bool = True,
) -> tuple[GLVDataset, list[str], np.ndarray]:
    """Assemble a within-subject before-vs-during-vancomycin :class:`GLVDataset`.

    reference = each subject's ``[start-ref_span, start)`` drug-free window; perturbed =
    ``[start, end+post_span)`` (drug + recovery). Both re-timed to a common local grid
    ``[0, ref_span]`` / mapped to the pulse; ``u(t)=1`` while the antibiotic is on. Aggregates
    ASVs to ``k`` groups (top genera + Other). Returns (dataset, labels, published-mean-none).
    """
    name, start, end = perturbation
    abs_ab = absolute_abundance(raw)
    labels, memb = top_genus_panel(raw, k, level=level)

    ref_grid = np.linspace(0.0, ref_span, n_grid)
    dur = (end - start) + post_span
    pert_grid = np.linspace(0.0, dur, n_grid)
    # a COMMON local observation grid + a shared fine integration grid for the pulse.
    t_obs = np.linspace(0.0, max(ref_span, dur), n_grid)

    def ensemble(local_grid: np.ndarray, win0: float) -> np.ndarray:
        trajs = []
        for subj in PERTURBED_SUBJECTS:
            ts, traj = _subject_series(raw, abs_ab, subj)
            seg = _interp(ts, traj, win0 + local_grid)  # (n_grid, n_asv)
            grouped = (memb @ seg.T).T  # (n_grid, k)
            trajs.append(grouped)
        return np.stack(trajs).astype(np.float32)  # (R, T, k)

    reference = ensemble(ref_grid, start - ref_span)
    perturbed = ensemble(pert_grid, start)

    scale = np.ones(k, dtype=np.float32)
    if normalize:
        both = np.concatenate([reference, perturbed], axis=0)
        scale = np.maximum(np.percentile(both, 95, axis=(0, 1)), 0.1).astype(np.float32)
        reference = reference / scale
        perturbed = perturbed / scale

    t_max = float(t_obs.max())
    n_steps = int(round(t_max / dt))
    grid_t = np.arange(n_steps + 1) * dt
    obs_idx = np.clip(np.round(t_obs / dt).astype(int), 0, n_steps)
    # pulse: drug on for its duration (end-start) from local t=0.
    u_grid = ((grid_t[:-1] >= 0.0) & (grid_t[:-1] < (end - start))).astype(np.float32)

    baseline = GLVParams(alpha=np.full(k, 0.4), beta=-np.eye(k), eps=np.zeros(k))
    ground_truth = {
        "mechanism": "real-data (unknown)",
        "target": 0,
        "perturbation": name,
        "k": k,
        "labels": labels,
        "scale": scale.tolist(),
    }
    ds = GLVDataset(
        reference=reference,
        perturbed=perturbed,
        t_obs=t_obs,
        u_grid=u_grid,
        obs_idx=obs_idx.astype(np.int64),
        dt=dt,
        baseline=baseline,
        ground_truth=ground_truth,
    )
    return ds, labels, memb
