# NUDGE demo notebooks

Guided, real-data walkthroughs of NUDGE's mechanism attribution. Each is committed
**with its outputs embedded** (plots + verdicts), so you can read the story without
running anything.

| Notebook | Dataset | What it shows |
|---|---|---|
| [`OCT4_NANOG_Flagship.ipynb`](OCT4_NANOG_Flagship.ipynb) | GSE283614 (Yao et al. 2025) | Dose-response attribution: **OCT4 → switch** (n≈6.7, R²=0.99) and **NANOG → unresolved** (a rigorous abstention — the knockdown doesn't span its threshold; `NUDGE-LIM-007`). |
| [`Norman_Synergy.ipynb`](Norman_Synergy.ipynb) | GSE133344 (Norman et al. 2019) | Synergy / epistasis: calls each CRISPRa combination **additive / synergistic / buffering** — or abstains — and agrees with the paper on its two explicitly-labeled pairs (`NUDGE-METHOD-003`). |
| [`Chure_LacI_Benchmark.ipynb`](Chure_LacI_Benchmark.ipynb) | Chure et al. 2019 (CaltechDATA D1.1241) | **Cross-modality** adapter: the *same* threshold/gain/ceiling attribution run on **fluorescence fold-change** (not counts), behind a modality bouncer (`NUDGE-LIM-008`). Recovers the authors' answer key — **inducer-binding mutants → threshold, DNA-binding → ceiling/leakiness** — and **abstains** on the rest, 0 mis-calls (`NUDGE-METHOD-002`). |
| [`Robustness_Dial.ipynb`](Robustness_Dial.ipynb) | Synthetic (self-activation switch, known fold) | **Robustness dial** (`NUDGE-METHOD-006`): how close is a bistable switch to *losing* bistability (a saddle-node fold)? Sweeps toward the switch's **known analytic fold** and shows all three channels move monotonically + the fused 0..1 dial. The honesty crux (`NUDGE-LIM-012`): a **one-sided lower bound** near the fold (the noise model is weakest exactly there) and an **abstention** on the deep-basin side. |
| [`Inverse_Design.ipynb`](Inverse_Design.ipynb) | Synthetic switch (Part A) + GSE283614 OCT4 (Part B) | **Inverse / intervention design** (`NUDGE-METHOD-007`) — from diagnosis to prescription. **Part A:** `design()` flips a synthetic bistable switch **ON**, contrasting a **SAFE** intervention with a **FOLD-CROSSING** one flagged **HIGH RISK OF INSTABILITY** by the Cap-5 safety gate; plus the integrity + reachability **abstentions**. **Part B (real data):** inverts the OCT4 self-renewal dose-response fit to the **knockdown dose** achieving a target — with an honest **reachability abstention** out of range (`NUDGE-LIM-013`). |
| [`Multi_Reporter.ipynb`](Multi_Reporter.ipynb) | Synthetic (one latent switch, 4 heterogeneous reporters) | **Multi-reporter joint attribution** (`NUDGE-METHOD-008`) — the identifiability force-multiplier. Fits several reporters of ONE latent switch **jointly** to break the **K⇄v_max degeneracy**: the JOINT panel resolves threshold / gain / ceiling (**100%**) where a SINGLE reporter **abstains** (`unresolved`, **0%**), with **0 confident-wrong calls**. The consistency guard abstains **off-model** when a reporter reads a *different* latent (`NUDGE-LIM-014`). |
| [`Hidden_Node_Abstention.ipynb`](Hidden_Node_Abstention.ipynb) | Synthetic (verdict + diagnostic evidence) | **Hidden-node abstention** (`NUDGE-METHOD-009`) — the **abstention half ONLY**. Turns a bare **`off-model`** verdict into a legible **six-cause differential** (not-a-switch / nonlinear readout / off-target / wrong topology / batch-depth confound / hidden node), each with its `NUDGE-LIM-*` and the distinguishing experiment. The honesty crux (`NUDGE-LIM-015`): even with a huge off-axis residual it **NEVER** asserts a hidden node — the strongest statement is *consistent with, does not prove* — and an **adequate** model yields **no differential**. |
| [`Differential.ipynb`](Differential.ipynb) | Synthetic (same perturbation, two contexts) | **Comparative / differential attribution** (`NUDGE-METHOD-010`) — the SAME perturbation in **two contexts**: BIC-selects **which single knob differs** — threshold / gain / **ceiling**. A **raised ceiling** → *more of the same drug*; a **rewired gain** → *a different class* — a call linear DE can't make. The honesty crux (`NUDGE-LIM-016`): a **depth/batch shift aligned with the context axis** mimics a ceiling difference, so depth is pinned **per context** and NUDGE **abstains** when the two contexts' depths differ beyond a ratio — **0 confident-wrong** on the depth confound across seeds. |
| [`Temporal_Ecology.ipynb`](Temporal_Ecology.ipynb) | Synthetic gLV + Stein et al. 2013 (`needs_data`) | **Temporal / Lotka–Volterra attribution** (`NUDGE-METHOD-012`) — *same engine, new field of biology* (microbiome ecology). Attributes a community perturbation to **growth (α) / interaction (β) / antibiotic-susceptibility (ε)** from **trajectories** (not a snapshot). The **ε** axis is the identifiable positive; **α⇄βᵢᵢ** (growth vs self-limitation, `Kᵢ=−αᵢ/βᵢᵢ`) is **degenerate near equilibrium** and NUDGE **abstains** with the degeneracy **MEASURED** by the Laplace curvature (`NUDGE-LIM-020`), never asserted. **0 confident-wrong** across the battery. Real coda: the clindamycin→*C. difficile* series, surfacing the honest abstention — *C. difficile*'s bloom is **interaction-mediated** (published ε≈−0.31), the very α/β confound. |
| [`Constitutive_Control.ipynb`](Constitutive_Control.ipynb) | Synthetic (a circuit switch behind a nonlinear reporter) | **Constitutive-reporter calibration control** (`NUDGE-METHOD-011`) — removes a known **confident-wrong** failure mode (`NUDGE-LIM-006`: a **nonlinear readout** misattributed as a **circuit switch**). Only $R\circ g$ is observed, so the circuit-$n$ profile is **FLAT** — you can't even tell a switch exists. A **constitutive control** (reporter driven at KNOWN doses, bypassing the circuit) **anchors the readout** (readout params ONLY, gradient-proven no leak); a profile over circuit $n$ then **rejects "no switch"** for a real switch → **biological**, or **abstains** for a linear circuit (the LIM-006 hazard). **0 confident-wrong**, and it does **not** point-identify $n$ (needs a second anchor; `NUDGE-LIM-018`). |

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
