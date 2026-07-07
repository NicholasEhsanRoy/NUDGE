# Data contract — raw counts only

**NUDGE owns the count model.** It fits *raw integer counts* with its own
negative-binomial + dropout observation model, because the mechanism signal
(threshold vs gain) lives in the *shape of the single-cell distribution* — the
bimodal off/on split and how sharp it is. The standard scanpy/Seurat clustering
pipeline is optimised to *suppress* that variation, so most of its steps are
**actively destructive** to NUDGE, and they fail *silently* (the fit runs, the
answer is just wrong). `nudge.data.ingest.check_counts` enforces this at the
`fit()` boundary — fails safely and loudly on the *input*, not just the output.

| Standard step | Effect on NUDGE | Verdict |
|---|---|---|
| Imputation / denoising (MAGIC, DCA, ALRA, scVI-denoised, kNN smoothing) | erases the bimodality that *is* the signal | **Never** |
| Pseudobulk aggregation | deletes the distribution NUDGE fits | **Never** |
| Batch integration to a corrected space (Harmony, scVI, Combat, Scanorama) | not counts; can erase or invent effects | **Reject** — model batch inside NUDGE |
| log1p / CPM / TPM / size-factor norm; Seurat `data`/`scale.data` | breaks the count likelihood | **Reject — raw counts only** |
| Nuisance regression (`regress_out` cell-cycle, %mito, total counts) | can remove real signal | **Avoid** — pass covariates, don't pre-remove |
| HVG / gene-panel subsetting | may drop the readout genes | **Ensure readout genes survive** |
| Ambient-RNA decontam (SoupX, CellBender) + doublet removal (Scrublet, DoubletFinder) | removes genuine artifacts that would fake an "off" mode | **Do this** — on counts, upstream |
| Unequal cells-per-condition / sequencing depth | mimics an effect | **Model or match** |
| Guide-assignment thresholds / MOI filtering | define the perturbation labels | **Document the calling rule** |

The only preprocessing NUDGE *wants* upstream is genuine-artifact removal
(ambient RNA, doublets, empty droplets), performed on counts. Everything else it
does itself.
