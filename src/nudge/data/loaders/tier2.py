"""Tier 2 loader — the Gladstone T-cell Perturb-seq screen (first instantiation).

A thin **config** over the generic backed-mode loader
(:mod:`nudge.data.loaders.perturbseq`): the Gladstone screen's obs column names + the
Ras activation-switch gene panel. The generic loader does the work (backed subsetting on
disk,
condition mapping, ``check_counts``); this module just says *which columns and genes*. A
new experiment is a new config like this one, not new code.

Schema confirmed from the dataset's ``data_sharing_readme.md`` (cell-level h5ads):
controls
are ``guide_type == "non-targeting"`` (→ "WT"), the perturbed gene is
``perturbed_gene_name``, gene symbols live in ``var["gene_name"]`` (raw ``var_names`` =
Ensembl), ambiguous guides are ``guide_id == "multi-guide"``, and ``low_quality`` flags
cells to drop. Confirm the exact ``guide_type`` control token against a real file header
before a production run.

The Ras switch (Das 2009): RASGRP1 (graded input GEF) → SOS1 (digital GEF, allosteric
Ras-GTP feedback → bistable); RASA2 sets the OFF threshold. There is no direct Ras-GTP
readout, so the switch **output** is inferred from immediate-early / activation
transcripts (the IEG panel).
"""

from __future__ import annotations

from typing import Any

from nudge.data.loaders.perturbseq import PerturbLoaderConfig, load_perturbseq

__all__ = ["GLADSTONE", "IEG_READOUT", "RAS_SWITCH_PANEL", "load_gladstone"]

#: Immediate-early / activation genes — the inferred switch output (no Ras-GTP probe).
IEG_READOUT: tuple[str, ...] = (
    "IL2", "IL2RA", "CD69", "EGR1", "EGR2", "FOS", "JUN", "NR4A1",
)
#: Ras-switch core + modulators (perturbation targets) + the IEG readout — loaded panel.
RAS_SWITCH_PANEL: tuple[str, ...] = (
    "SOS1", "SOS2", "RASGRP1", "RASGRP2", "RASA1", "RASA2", "RASA3",
    "DGKA", "DGKZ", "LAT", "PLCG1",
    *IEG_READOUT,
)

#: The Gladstone instantiation of the generic loader config.
GLADSTONE = PerturbLoaderConfig(
    condition_col="perturbed_gene_name",
    control_col="guide_type",
    control_values=("non-targeting",),
    control_label="WT",
    gene_symbol_col="gene_name",
    gene_subset=RAS_SWITCH_PANEL,
    quality_drop_col="low_quality",
    guide_id_col="guide_id",
    exclude_guides=("multi-guide",),
)


def load_gladstone(
    path: str,
    *,
    target_genes: tuple[str, ...] | None = None,
    gene_subset: tuple[str, ...] | None = None,
    **overrides: Any,
) -> Any:
    """Backed-load a Gladstone donor×condition ``.h5ad`` into NUDGE's condition schema.

    ``target_genes`` restricts which perturbations become conditions (plus controls) —
    e.g. ``("SOS1", "RASGRP1", "RASA2")`` for the Ras-switch operating points.
    ``gene_subset`` overrides the observed panel. Any other :class:`PerturbLoaderConfig`
    field can be overridden via keyword.
    """
    updates: dict[str, Any] = dict(overrides)
    if target_genes is not None:
        updates["target_genes"] = target_genes
    if gene_subset is not None:
        updates["gene_subset"] = gene_subset
    cfg = (
        PerturbLoaderConfig(**{**GLADSTONE.__dict__, **updates})
        if updates
        else GLADSTONE
    )
    return load_perturbseq(path, cfg)
