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
from typing import Any, Literal, overload

import numpy as np

from nudge.core.circuit import Circuit
from nudge.inference.epistasis import ComboGeometry, combo_geometry
from nudge.inference.lyapunov import OperatingPoint, calibrate_from_wt

__all__ = [
    "adata_to_operating_point",
    "combo_effect_scores",
    "counts_to_activity",
    "fluorescence_dose_response",
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


def fluorescence_dose_response(
    df: Any,
    *,
    dose_col: str,
    response_col: str,
    variant: str,
    variant_col: str,
    filters: Mapping[str, Any] | None = None,
    control_variant: str | None = None,
    autofluor: float = 0.0,
    agg: str = "mean",
    modality: str = "fluorescence",
) -> tuple[np.ndarray, np.ndarray]:
    """Per-dose ``(dose, continuous response)`` for one ``variant`` — the adapter core.

    The cross-modality analogue of :func:`knockdown_dose_response`: it turns a tidy
    **continuous-readout** table (flow fluorescence, an activity reporter, or a
    fold-change summary — *not* UMI counts) into the ``(dose, response)`` pair the
    shipped :func:`~nudge.inference.dose_response.attribute_dose_response` consumes. One
    row per (variant, dose[, extra keys]); ``filters`` pins the other axes (e.g.
    ``{"operator": "O2", "repressors": 260}``) so a single induction curve is selected.

    The **modality bouncer runs first** (:func:`nudge.data.ingest.check_readout`): the
    raw ``response_col`` must pass the continuous-readout guard, so log-normalized or
    raw counts mislabeled as fluorescence are **refused here**, before any fitting
    (NUDGE-LIM-008). Then, per dose level, the response is aggregated (``agg`` =
    ``mean`` or ``median``), an optional ``autofluor`` offset is subtracted, and — if
    ``control_variant`` is given — divided by that variant's per-dose response to form a
    fold-change (Chure-style summaries are already fold-change, so the default is a
    plain per-dose summary). Returns dose-sorted arrays. Raises ``KeyError`` on missing
    columns / variants and ``IngestError`` on a modality violation.
    """
    from nudge.data.ingest import check_readout

    for col in (dose_col, response_col, variant_col):
        if col not in getattr(df, "columns", []):
            raise KeyError(f"column {col!r} not in {list(getattr(df, 'columns', []))}")

    sub = df
    for key, val in (filters or {}).items():
        if key not in df.columns:
            raise KeyError(f"filter column {key!r} not in {list(df.columns)}")
        sub = sub[sub[key] == val]

    def _curve(name: str) -> dict[float, float]:
        rows = sub[sub[variant_col].astype(str) == str(name)]
        if rows.empty:
            raise KeyError(
                f"no rows for {variant_col}=={name!r} under "
                f"filters {dict(filters or {})}"
            )
        # Bouncer: refuse counts / log-normalized values masquerading as continuous.
        check_readout(rows, modality=modality, readout_col=response_col)
        grouped = rows.groupby(dose_col)[response_col]
        summary = grouped.median() if agg == "median" else grouped.mean()
        return {float(d): float(v) - autofluor for d, v in summary.items()}

    var_curve = _curve(variant)
    if control_variant is not None:
        ctrl_curve = _curve(control_variant)
        doses = sorted(d for d in var_curve if d in ctrl_curve)
        response = [
            var_curve[d] / ctrl_curve[d] if ctrl_curve[d] != 0 else float("nan")
            for d in doses
        ]
    else:
        doses = sorted(var_curve)
        response = [var_curve[d] for d in doses]

    return np.asarray(doses, dtype=float), np.asarray(response, dtype=float)


def _condition_mask(obs: Any, condition_col: str, label: str) -> np.ndarray:
    return np.asarray(obs[condition_col].astype(str) == str(label))


@overload
def combo_effect_scores(
    adata: Any,
    *,
    control_label: str,
    a_label: str,
    b_label: str,
    ab_label: str,
    condition_col: str = ...,
    library_col: str | None = ...,
    signature: Sequence[str] | None = ...,
    n_top_genes: int = ...,
    return_geometry: Literal[False] = ...,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


@overload
def combo_effect_scores(
    adata: Any,
    *,
    control_label: str,
    a_label: str,
    b_label: str,
    ab_label: str,
    condition_col: str = ...,
    library_col: str | None = ...,
    signature: Sequence[str] | None = ...,
    n_top_genes: int = ...,
    return_geometry: Literal[True],
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, ComboGeometry | None
]: ...


def combo_effect_scores(
    adata: Any,
    *,
    control_label: str,
    a_label: str,
    b_label: str,
    ab_label: str,
    condition_col: str = "condition",
    library_col: str | None = "total_counts",
    signature: Sequence[str] | None = None,
    n_top_genes: int = 2000,
    return_geometry: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ComboGeometry | None]
):
    """Per-cell scalar **effect scores** for {control, A, B, A+B} of a combination.

    Returns four 1-D arrays (one score per cell) ready for
    :func:`~nudge.inference.epistasis.fit_synergy`. Counts are depth-normalized
    (size-factor to the median library size, the same normalization as
    :func:`counts_to_activity`) and ``log1p``-transformed, so effects live in
    **log-fold-change space** and the additive null is Bliss independence.

    Two scoring modes:

    - **projection (default, ``signature=None``)** — the **direction-safe** data-driven
      score. Compute the two single-arm mean shift vectors ``vA = mean(A) − mean(ctrl)``
      and ``vB`` over the ``n_top_genes`` most variable genes, and project every cell
      onto the **additive axis fixed by the singles** ``u = (vA + vB) / ‖vA + vB‖``.
      Because ``u`` comes from the singles only (never the combo), a positive
      interaction is unambiguously super-additive *along the axis the singles push* — no
      circularity, no manual sign convention. The recommended Norman-style extractor.
    - **signature (``signature=[genes…]``)** — the per-cell score is the mean
      ``log1p``-normalized expression over the given ``signature`` genes (a
      pre-specified phenotype readout). Simpler, but the caller must ensure the
      signature is oriented so the two arms push the same way.

    Raises ``KeyError`` if a condition label matches no cells or a signature gene is
    absent from ``var_names``.

    With ``return_geometry=True`` a fifth element is returned: the
    :class:`~nudge.inference.epistasis.ComboGeometry` **possible-neomorphic off-axis
    diagnostic** — the magnitude of the combo's interaction residual orthogonal to the
    additive axis (the emergent component this scalar structurally cannot see). It is
    ``None`` in the fixed-``signature`` mode (there is no additive axis to project).
    Feed it to :func:`~nudge.inference.epistasis.attribute_synergy` to surface an honest
    *under-count* warning with the call (NUDGE-LIM-009).
    """
    obs = getattr(adata, "obs", None)
    if obs is None or condition_col not in obs:
        raise KeyError(f"obs column {condition_col!r} not found")

    # Subset to only the four conditions BEFORE densifying — a genome-wide screen is far
    # too large to densify whole (111k×19k), and only these cells are ever scored.
    full_masks = {}
    for role, label in (
        ("control", control_label),
        ("a", a_label),
        ("b", b_label),
        ("ab", ab_label),
    ):
        m = _condition_mask(obs, condition_col, label)
        if not m.any():
            raise KeyError(f"no cells for {role} condition {label!r}")
        full_masks[role] = m
    keep_rows = np.zeros(len(obs), dtype=bool)
    for m in full_masks.values():
        keep_rows |= m
    sub_adata = adata[keep_rows]
    sub_labels = np.asarray(sub_adata.obs[condition_col].astype(str))
    masks = {
        "control": sub_labels == str(control_label),
        "a": sub_labels == str(a_label),
        "b": sub_labels == str(b_label),
        "ab": sub_labels == str(ab_label),
    }

    norm, gene_ix = _norm_counts(sub_adata, library_col)
    lognorm = np.log1p(norm)

    geometry: ComboGeometry | None = None
    if signature is not None:
        sig = list(signature)
        missing = [g for g in sig if g not in gene_ix]
        if missing:
            raise KeyError(f"signature genes absent from var_names: {missing}")
        cols = [gene_ix[g] for g in sig]
        score = lognorm[:, cols].mean(axis=1)
    else:
        # Direction from the singles only (control + A + B); the combo never defines it.
        var = lognorm.var(axis=0)
        n_keep = min(int(n_top_genes), lognorm.shape[1])
        keep = np.argsort(var)[::-1][:n_keep]
        sub = lognorm[:, keep]
        m_ctrl = sub[masks["control"]].mean(axis=0)
        v_a = sub[masks["a"]].mean(axis=0) - m_ctrl
        v_b = sub[masks["b"]].mean(axis=0) - m_ctrl
        u = v_a + v_b
        nu = float(np.linalg.norm(u))
        if nu <= 1e-12:
            raise ValueError(
                "the two single arms produce no net expression shift "
                "(additive direction is undefined) — cannot score this combination"
            )
        u = u / nu
        score = sub @ u
        # Off-axis (possible-neomorphic) diagnostic: the combo's interaction residual
        # r = v_AB − v_A − v_B decomposed into on-axis (the scalar NUDGE reports) and
        # the orthogonal component the scalar cannot see (NUDGE-LIM-009).
        v_ab = sub[masks["ab"]].mean(axis=0) - m_ctrl
        geometry = combo_geometry(v_a, v_b, v_ab, n_top_genes=n_keep)

    scores = (
        np.asarray(score[masks["control"]], dtype=float),
        np.asarray(score[masks["a"]], dtype=float),
        np.asarray(score[masks["b"]], dtype=float),
        np.asarray(score[masks["ab"]], dtype=float),
    )
    if return_geometry:
        return (*scores, geometry)
    return scores
