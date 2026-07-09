# NUDGE demo notebooks

Guided, real-data walkthroughs of NUDGE's mechanism attribution. Each is committed
**with its outputs embedded** (plots + verdicts), so you can read the story without
running anything.

| Notebook | Dataset | What it shows |
|---|---|---|
| [`OCT4_NANOG_Flagship.ipynb`](OCT4_NANOG_Flagship.ipynb) | GSE283614 (Yao et al. 2025) | Dose-response attribution: **OCT4 → switch** (n≈6.7, R²=0.99) and **NANOG → unresolved** (a rigorous abstention — the knockdown doesn't span its threshold; `NUDGE-LIM-007`). |
| [`Norman_Synergy.ipynb`](Norman_Synergy.ipynb) | GSE133344 (Norman et al. 2019) | Synergy / epistasis: calls each CRISPRa combination **additive / synergistic / buffering** — or abstains — and agrees with the paper on its two explicitly-labeled pairs (`NUDGE-METHOD-003`). |
| [`Chure_LacI_Benchmark.ipynb`](Chure_LacI_Benchmark.ipynb) | Chure et al. 2019 (CaltechDATA D1.1241) | **Cross-modality** adapter: the *same* threshold/gain/ceiling attribution run on **fluorescence fold-change** (not counts), behind a modality bouncer (`NUDGE-LIM-008`). Recovers the authors' answer key — **inducer-binding mutants → threshold, DNA-binding → ceiling/leakiness** — and **abstains** on the rest, 0 mis-calls (`NUDGE-METHOD-002`). |

## Running them yourself

```bash
uv pip install -e ".[dev]"           # jupyter + matplotlib + nbconvert live in [dev]
uv run jupyter lab notebooks/        # or open the .ipynb in your IDE
```

The flagship notebook reads a converted `ESC.h5ad` (raw counts + guide calls) from
GSE283614. Point it at your copy with an env var (defaults to the drive path used to
build it):

```bash
export OCT4_NANOG_H5AD=/path/to/ESC.h5ad
```

To re-execute headlessly and re-embed outputs:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/OCT4_NANOG_Flagship.ipynb
```
