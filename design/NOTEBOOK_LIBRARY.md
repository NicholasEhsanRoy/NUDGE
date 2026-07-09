# NUDGE — validation-notebook library (real public datasets)

A prioritized catalogue of **validation-notebook ideas**, each grounded in a **real,
verified-public dataset** plus its primary literature. These are candidate demos to build
later in the week as additional validation points beyond the synthetic V&V backbone and the
Gladstone T-cell run. The point of a good NUDGE demo is not "NUDGE was right" — it is
"NUDGE made a **mechanism call** (threshold K / gain n / ceiling v_max) — or **abstained
loudly** — where linear/DE analysis structurally *cannot*, and the call matches an
independent ground truth." An abstention demo (NUDGE correctly says *unresolved* /
*off-model* / *no-effect*) is worth as much as a positive one, and several notebooks below
are built around exactly that. **Honesty rule (CLAUDE.md) applies to this doc too: every
"partial fit" and "stretch" is flagged, and no dataset is cited that a research pass could
not confirm is public.**

What NUDGE needs from a dataset, in priority order: **single-cell (or single-cell-resolution
flow) RAW counts** + a **perturbation/condition label** + a **control** + ideally **≥2
operating points** (doses / A·B·A+B combos / stimulation conditions / time points /
co-perturbation backgrounds) — the second operating point is what breaks the measured
gain⇄threshold Fisher-degeneracy (×16 identifiability). Ceiling is separately identifiable;
depth/scale must be pinned from a control. Everything here was chosen against that contract.

---

## Ranking — ~15 candidate notebooks by (demo impact × data accessibility)

| # | Notebook | Dataset (accession) | Domain | NUDGE capability | Feasibility |
|---|---|---|---|---|---|
| 1 | **Pluripotency dosage: does NANOG shift the threshold and OCT4 the gain?** | GSE283614 | Cell fate | Attribution K vs n, **with published ground truth** | ready |
| 2 | **LacI mutant benchmark: DNA-binding = K, inducer-binding = n** | CaltechDATA D1.1241 (Chure 2019) | Synthetic bio | Attribution K vs n, **ground-truth-labelled** | needs-work (flow adapter) |
| 3 | **sci-Plex dose-response: mechanism + abstention battery** | GSE139944 | Drug / MAPK | Attribution over a 4-pt dose axis + mass abstention | ready |
| 4 | **T-cell activation switch (Ras/SOS): stim vs unstim operating points** | GSE190604 (Schmidt 2022) | Cell fate | Attribution + operating-point breaker; NUDGE's own thesis | ready |
| 5 | **LacI/IPTG induction: operator swap = threshold tuning** | CaltechDATA D1.743 (Razo-Mejia 2018) | Synthetic bio | Attribution K; dose × part-variant matrix | needs-work (flow adapter) |
| 6 | **Morphogen dose ladder: Shh–Gli K/n/vmax on real cells** | GSE233574 (Pașca) | Morphogen | Attribution over a clean 4-pt SAG dose ladder | ready |
| 7 | **Genetic-interaction combos: A / B / A+B = the degeneracy breaker** | GSE133344 (Norman 2019) | Drug / hidden-node | Combination operating points; hidden-node/epistasis | ready |
| 8 | **Gladstone genome-scale CD4+ screen (the headline validation)** | GSE314342 (D1_Stim8hr) | Cell fate | Attribution + 3 stim × polarization operating points | needs-work (heavy; CZI mirror) |
| 9 | **Hidden node caught red-handed: co-measured miRNA** | GSE114071 | Hidden-node | **Hidden-node detection** vs co-measured ground truth | needs-work |
| 10 | **Bistable toggle with hysteresis: a memory a Hill fit can't explain** | Zenodo 11817798 | Synthetic bio | Attribution + abstention on bistability/hysteresis | needs-work (flow adapter) |
| 11 | **Drug resistance: raised ceiling ("more dose") vs rewired gain** | figshare 10298696 (MIX-seq) | Drug | Ceiling vs gain across sensitive/resistant lines | needs-work |
| 12 | **Continuous TF dose atlas: 384 titration curves** | E-MTAB-13010 (scTF-seq) | Cell fate / oncogene | Attribution over per-cell continuous dose | needs-work |
| 13 | **Genome-scale abstention/no-effect battery** | Zenodo 10044268 (Replogle) | Drug / control | Specificity: ~2.5M cells, single operating point → abstain | ready (subset) |
| 14 | **Epigenetic switch onset: X-inactivation with Xist as hidden node** | GSE151009 (Pacini 2021) | Epigenetic | Attribution on a silencing sigmoid + hidden-node | needs-work |
| 15 | **Classic switches with no public data → honest synthetic decoys** | BioModels + synthetic | Multi | Abstention / hidden-node where **no real data exists** | ready (synthetic) |

Feasibility legend: **ready** = raw counts + conditions + a loader path exist today (often a
one-line `pertpy`/`.h5ad` fetch); **needs-work** = real & public but needs an adapter (flow
FCS→readout channel, on-disk subsetting of a huge file, or metadata wrangling to assign
conditions); **stretch** = structurally partial or a large lift. NUDGE already ships a
generic backed-mode Perturb-seq loader (`data/loaders/perturbseq.py`) and a counts→activity
bridge (`inference/bridge.py`); the flow-cytometry cases need a small "fluorescence as the
`Readout` channel" adapter, which is a readout swap, not a structural blocker.

---

## Domain A — Cell fate & developmental biology (Waddington attractors)

### A1. Pluripotency dosage — does NANOG move the threshold and OCT4 the gain? ⟶ flagship
- **The question.** The pluripotency network (OCT4/NANOG) is the textbook ESC bistable
  switch (Nanog bistability: Chambers et al. 2007). A recent Perturb-seq **titrates OCT4 and
  NANOG dosage** and reports that **NANOG reduction shifts the state *gradually*, whereas
  OCT4 reduction triggers an *abrupt* exit** from self-renewal. That is *verbatim* NUDGE's
  threshold(K)-vs-gain(n) distinction — with an author-provided answer to check against.
- **NUDGE capability.** Mechanism attribution on a bistable switch **with independent ground
  truth** — the single strongest attribution-validation case found. Graded dosage is the
  operating-point axis that breaks the K⇄n degeneracy; two culture conditions (ESC
  self-renewal + definitive endoderm) add a second axis.
- **Dataset.** OCT4/NANOG dosage Perturb-seq — **GEO GSE283614**. Human hESC, CRISPRi/enhancer
  titration of OCT4 & NANOG, profiled in self-renewal and definitive-endoderm states.
  Format: 10x MTX + TSV, raw counts; ESC ≈ 218 MB, DE ≈ 420 MB; non-targeting controls
  present (build the `.h5ad` locally). **Structural fit: excellent** — raw counts ✓,
  perturbation + dose axis ✓, control ✓, 2 conditions ✓.
- **Expected honest result.** NUDGE calls **NANOG → threshold(K)** (graded) and **OCT4 →
  gain(n)** / abrupt (or abstains on the sparser dose rungs). Because the paper already
  decomposed this, a matched call is a real, checkable win; a mismatch is a documented
  limitation. Either way it is honest.
- **Feasibility: ready.** **Key papers:** OCT4/NANOG dosage Perturb-seq, bioRxiv
  2025.08.07.669196. Nanog bistability — Chambers et al., *Nature* 450:1230 (2007),
  doi:10.1038/nature06403.

### A2. T-cell activation switch (Ras/SOS) — stim vs unstim operating points ⟶ NUDGE's own thesis
- **The question.** The Activated⇄Exhausted decision runs through a **Ras activation switch**
  (RASGRP1 graded input → SOS1 digital/allosteric feedback → bistable; RASA2 sets the OFF
  threshold — Das et al. 2009). Which CRISPRa/CRISPRi perturbations move that switch's
  threshold vs gain? This is the exact circuit NUDGE's `circuits.py` (`ras_switch_1node`/
  `ras_switch_2node`) and Ras Mechanism Cards were built around.
- **NUDGE capability.** Attribution + the **operating-point breaker** (stimulated vs
  unstimulated = 2 operating points) + BIC topology model-selection (1-node vs 2-node Ras).
- **Dataset.** Schmidt et al. 2022 CRISPRa Perturb-seq in primary human CD4+ T cells — **GEO
  GSE190604** (the `[CRISPRa Perturb-seq]` sub-series). **PUBLIC** on GEO (human primary
  cells but *not* dbGaP/EGA-gated). Format: 10x `.mtx.gz` (~950 MB) + barcodes + features +
  **per-cell guide calls** (`cellranger-guidecalls-aggregated-unfiltered.txt.gz`); ~1 GB
  total, 16 samples, 2 donors. **Structural fit: strong** — raw counts ✓, per-cell guide +
  non-targeting controls ✓, stim/unstim = 2 operating points ✓.
- **Expected honest result.** On IEG readouts (IL2/CD69/EGR/NR4A1) NUDGE either attributes a
  Ras-pathway CRISPRa to threshold vs gain, or — the more likely and still-valuable outcome
  given the FIM analysis — the **single-condition arm abstains between gain/threshold and the
  stim-axis resolves it** (a live demonstration of the second-operating-point breaker on real
  data). Parsimony may also pick no-switch/1-node — an honest topology result.
- **Feasibility: ready** (small, clean, well-documented — the best "just works" T-cell demo).
  **Key papers:** Schmidt et al., *Science* 375:eabj4008 (2022), doi:10.1126/science.abj4008.
  Das et al. (Ras/SOS bistability), *Cell* 136:337 (2009), doi:10.1016/j.cell.2009.04.062.

### A3. Gladstone genome-scale CD4+ Perturb-seq — the headline validation (D1_Stim8hr)
- **The question.** The genome-scale sibling of A2 and NUDGE's designated Tier-2 target:
  which of ~12,748 CRISPRi knockdowns move the T-cell activation switch's threshold vs gain,
  across **Rest / Stim8hr / Stim48hr** and Th1/Th2/nonpolarized backgrounds?
- **NUDGE capability.** Full attribution pipeline + **3+ stimulation operating points ×
  polarization backgrounds** — the richest real degeneracy-breaker available. This is what
  `data/loaders/tier2.py` + `scripts/vv/gladstone_attribution.py` were built for.
- **Dataset.** Genome-scale Perturb-seq in primary human CD4+ T cells — **GEO GSE314342**
  (SRA SRP643211); **PUBLIC**. The `D1_Stim8hr`-style donor×condition file naming = this
  deposit. GEO holds cellranger `.h5` (RAW.tar ≈ **159 GB**, 577 samples, ~22M cells, 4
  donors); a curated **`.h5ad`/`.h5mu` mirror is on the CZI Virtual Cells Platform**
  (much friendlier). **Structural fit: excellent** — raw counts ✓, guide + non-targeting
  controls ✓, 3 stim × polarization operating points ✓✓.
- **Expected honest result.** Single-condition run: parsimony picks no-switch or 1-node and
  attribution abstains between gain/threshold (the FIM prediction); the multi-stim run
  *resolves* it. The falsifiable SOS/RasGRP1 prediction is the payoff.
- **Feasibility: needs-work** — use the **CZI `.h5ad` mirror** or on-disk subset a few
  perturbations; the 159 GB GEO tar is a heavy download (flag). **Key papers:** genome-scale
  CD4+ Perturb-seq, bioRxiv 10.64898/2025.12.23.696273; Das et al. 2009 (as A2).

### A4. In-vivo CD8 exhaustion regulome — the Tpex→Tex bistable flip (narrative-rich)
- **The question.** The canonical exhaustion switch: TOX⁺TCF7⁺ stem-like Tpex → intermediate
  → terminal Tex. Which TF perturbations (IKAROS/ETS1/RBPJ) act as node regulators that lower
  the barrier (gain) vs shift where it sits (threshold)?
- **NUDGE capability.** Attribution on the literal exhaustion switch NUDGE's pitch invokes.
- **Dataset.** Zhou/Chi et al. 2023 in-vivo single-cell CRISPR screen — **GEO GSE216800**
  (SuperSeries; child **GSE216909** `[scCRISPR]`, 16 samples; validation GSE216796/216798/
  218372). **PUBLIC.** Mouse OT-I CD8 in B16-OVA, 180 TFs / 360 dual guides + non-targeting.
  **Structural fit: partial ⚠** — raw counts ✓, guide + control ✓, but **one in-vivo
  timepoint/context ⇒ weak on ≥2 operating points**; the degeneracy-breaker would have to
  come from co-perturbation backgrounds, so expect **more abstention** (honest, and on-thesis).
- **Feasibility: needs-work / stretch** (best as the narrative exhaustion demo; identifiability
  thinner). **Key papers:** Zhou et al., *Nature* 624:154 (2023), doi:10.1038/s41586-023-06733-x.

### A5. Reprogramming barrier crossing — which condition lowers the Waddington wall?
- **The question.** In fibroblast→iPSC reprogramming, most cells never convert. Which culture
  condition (2i vs serum) lowers the barrier (a **gain** change), and can NUDGE tell the
  productive minority's switch from the non-productive majority?
- **NUDGE capability.** **Abstention as the headline** — NUDGE should abstain on the
  non-converting majority (no switch to attribute) and only attribute where a barrier-crossing
  signature exists; plus a genuinely watchable "flip" visual for Demo (30%).
- **Dataset.** Schiebinger et al. 2019 Waddington-OT time course — **GEO GSE122662**
  (SuperSeries) / **GSE115943** (126 samples). **PUBLIC**; RAW.tar (~1.1 GB `.h5`) + `.mtx`
  (~2.4 GB). ~250–315k cells, OKSM, half-day intervals over 18 days, **2i vs serum** = the
  second operating point. **Structural fit: partial ⚠** — raw counts ✓, day-0 control ✓,
  operating points ✓, but the "perturbation" axis is **time/induction, not discrete
  knockdowns** — a stretch for per-perturbation attribution.
- **Feasibility: stretch.** **Key papers:** Schiebinger et al., *Cell* 176:928 (2019),
  PMID 30712874, doi:10.1016/j.cell.2019.01.006.

---

## Domain B — Drug discovery & pharmacology

### B1. sci-Plex dose-response — mechanism attribution + a built-in abstention battery ⟶ top drug pick
- **The question.** A competitive inhibitor right-shifts a dose-response (threshold/K); an
  allosteric one flattens its slope (gain/n); a resistance/ceiling change caps it (v_max).
  Across 188 compounds × 4 doses, which drugs move which knob — and does NUDGE correctly
  **abstain on the many inert compounds**?
- **NUDGE capability.** Attribution over a genuine **4-point dose axis** (points on a Hill
  curve) + a large **false-positive/abstention battery** (inert compounds = negatives).
- **Dataset.** sci-Plex — **GEO GSE139944** (`GSE139944_RAW.tar` ≈ 9.2 GB; one-line `.h5ad`
  via `pertpy.data.srivatsan_2020_sciplex3`, figshare 33979517; Zenodo 13350497). A549/MCF7/
  K562 × 188 compounds × **10 nM/100 nM/1 µM/10 µM + vehicle**, ~650k cells, 24 h. **Structural
  fit: excellent** — raw UMI counts ✓, drug + vehicle control ✓, 4-pt dose series ✓✓. (Low
  median UMI/cell ~1–2.4k — the intended sparse raw-count regime.)
- **Expected honest result.** For HDAC inhibitors and other epigenetic movers (the panel's
  strength), NUDGE calls threshold vs gain from the dose-curve shape; for inert compounds it
  returns **no-effect/off-model** — a specificity demo on real data.
- **Feasibility: ready.** **Key papers:** Srivatsan et al., *Science* 367:45 (2020),
  doi:10.1126/science.aax6234. (Also the MAPK/ERK sub-story via **trametinib** — see C2.)

### B2. Genetic-interaction combos — A / B / A+B is the degeneracy breaker
- **The question.** Combination therapy is the multi-operating-point breaker: does co-perturbing
  two regulators shift the switch **additively** (same mechanism) or **super-additively**
  (rewired gain), and where does A+B land vs A and B alone?
- **NUDGE capability.** **Combination operating points** (A, B, A+B) → the FIM breaker; also a
  **hidden-node / epistasis** read (a synergy a 1-D per-gene model can't explain flags feedback).
- **Dataset.** Norman et al. 2019 combinatorial Perturb-seq — **GEO GSE133344** (MTX ~1.3 GB;
  one-line via `pertpy.data.norman_2019`, figshare 34027562). ~91k K562 cells, CRISPRa, **105
  single + 131 two-gene (A/B/A+B) activations** + non-targeting controls. **Structural fit:
  strong on structure, stretch on modality ⚠** — raw counts ✓, control ✓, A/B/A+B combos ✓,
  but perturbations are **genetic (CRISPRa on/off), not drug doses** — illustrates combination
  *logic*, not a literal drug combo, and there's no graded dose.
- **Expected honest result.** For coactivated pairs, NUDGE attributes additive vs synergistic
  gain/ceiling; for non-interacting pairs it abstains. Small and fast to iterate — the best
  sandbox for the combination-breaker.
- **Feasibility: ready.** **Key papers:** Norman et al., *Science* 365:786 (2019),
  doi:10.1126/science.aax4438.

### B3. Drug resistance — raised ceiling ("more dose") vs rewired gain ("different drug class")
- **The question.** When a line resists a targeted drug, is the switch's **ceiling raised**
  (needs more dose, same mechanism) or its **gain/threshold rewired** (a different drug class
  is needed)? That distinction changes the clinical decision and linear DE can't make it.
- **NUDGE capability.** Ceiling vs gain attribution across sensitive/resistant genotype
  backgrounds (many cell lines = many backgrounds), with dose **and** timepoint operating points.
- **Dataset.** MIX-seq — **figshare 10298696** (UMI matrices + drug-sensitivity/genomic
  metadata). Pools of 24–100+ cancer lines demultiplexed by SNP; **trametinib (MEK) and
  idasanutlin (MDM2)** with **staggered dose × post-treatment timepoints**; DMSO control.
  **Structural fit: good but needs-work ⚠** — raw UMI counts ✓, drug + DMSO control ✓, dose
  and time operating points ✓, but you must parse the demultiplexing metadata to assign
  conditions (the figshare landing page is public; a scraper hit a 403 bot-block, not an access
  gate). **Feasibility: needs-work.** **Key papers:** McFarland et al., *Nat Commun* 11:4296
  (2020), doi:10.1038/s41467-020-17440-w.

### B4. sci-Plex-GxE — gene × drug mechanism attribution (aspirational scale-up)
- **The question.** Which kinase knockouts move a targeted therapy's threshold vs gain — the
  purest "mechanism attribution of a combination," with the gene×drug (GxE) axis built in.
- **Dataset.** sci-Plex-GxE — **GEO GSE225775** (code: cole-trapnell-lab/sci-Plex-GxE).
  ~1.05M cells, 14,121 gene×drug combos, 522 kinases × RTK-pathway drugs in glioblastoma.
  **Structural fit: conceptually ideal, heavy ⚠** — GxE = second operating point ✓, raw counts
  ✓, but >1M cells and complex wrangling. **Feasibility: stretch.** **Key papers:**
  McFaline-Figueroa et al., *Cell Genomics* 4:100487 (2024), doi:10.1016/j.xgen.2023.100487.

### B5. Genome-scale abstention / no-effect battery (specificity at scale)
- **The question.** Turned on ~2.5M cells of single knockdowns with no dose axis, does NUDGE
  keep its measured <2% false-positive discipline — i.e. abstain (no-effect/off-model) almost
  everywhere and only flag genuine switch-movers?
- **NUDGE capability.** **Specificity / abstention at scale** — the honest counterpart to the
  attribution demos, and a real-data echo of the 0%-misclassification synthetic V&V.
- **Dataset.** Replogle et al. 2022 genome-scale Perturb-seq — **SRA BioProject PRJNA831566**;
  processed `.h5ad` on **Zenodo 10044268** (`ReplogleWeissman2022_*`) / figshare 20029387.
  K562 (essential/gwps) + RPE1, CRISPRi, huge non-targeting control. **Structural fit: partial
  by design ⚠** — raw counts ✓, control ✓, but **single knockdown level, single operating
  point** ⇒ can't break the degeneracy alone; that's the point (it's the negative/abstention
  library, not an attribution set). **Feasibility: ready if subset** (gwps is tens of GB — take
  essential or RPE1). **Key papers:** Replogle et al., *Cell* 185:2559 (2022),
  doi:10.1016/j.cell.2022.05.013.

---

## Domain C — Hidden species / ghost feedback, MAPK/ERK, morphogens, epigenetics

### C1. Hidden node caught red-handed — fit 1-D on mRNA, reveal the co-measured miRNA
- **The question.** NUDGE's most differentiated claim: if a "simple" 1-D model is consistently
  rejected as off-model (or needs an N-D saddle / shows a covariance comet-tail an ODE can't
  explain), that is computational **proof of an unmeasured regulator**. Can NUDGE flag a hidden
  node whose identity we can then check?
- **NUDGE capability.** **Hidden-node detection** (parsimony/off-model tripwire + LNA covariance
  ghost-feedback signal) validated against a co-measured ground truth.
- **Dataset.** Wang et al. 2018 single-cell **miRNA-mRNA co-sequencing** — **GEO GSE114071**.
  K562, 42 samples; measures the **hidden node (miRNA) in the same single cells** as its mRNA
  targets — the ideal substrate to fit 1-D on mRNA, show rejection, then reveal the co-measured
  miRNA. **Structural fit: partial ⚠** — `.gct.gz` log2-normalized (raw reads in SRA), **no
  dose axis, no control, only ~19–40 cells** ⇒ hidden-node *validation*, not attribution.
- **Expected honest result.** NUDGE rejects/abstains on the naive 1-D mRNA model where a strong
  miRNA regulator is co-present — and the co-measured miRNA is the "answer key."
- **Feasibility: needs-work.** Pair with a **synthetic IFFL/ceRNA generator** (below) for the
  controlled version. **Key papers:** Wang et al., *Nat Commun* 9:5395 (2018),
  doi:10.1038/s41467-018-07981-6.
- **Honest gap.** The canonical hidden-node synthetic-biology systems — **Mukherji et al. 2011**
  (miRNA thresholding), **Bleris et al. 2011**, ceRNA sponge circuits (**Yuan et al. 2015**,
  SRP052983 — but that deposit is **bulk** RNA-seq) — do **not** have ingestible single-cell
  public data. That absence is itself a finding; use them as mechanism grounding + a synthetic
  generator, and say so.

### C2. MAPK/ERK ultrasensitivity (Huang–Ferrell) — MEK-inhibitor dose on KRAS-mutant cells
- **The question.** The Huang–Ferrell MAPK cascade is the archetypal ultrasensitive (high-gain)
  switch. Does a MEK inhibitor's dose-response on a KRAS-driven line move threshold vs gain,
  and does NUDGE **abstain on a line where the drug is a true negative**?
- **NUDGE capability.** Attribution over a drug dose axis + a built-in true-negative abstention.
- **Dataset.** Reuse **sci-Plex GSE139944**, subset to **trametinib (MEK)** and RTK/MEK
  compounds: A549 (KRAS-mutant) trametinib dose series + vehicle; **MCF7-trametinib is a
  built-in true negative**. Raw counts ✓, dose ✓, control ✓. **Feasibility: ready** (same
  loader as B1). **Key papers:** Huang & Ferrell, *PNAS* 93:10078 (1996),
  doi:10.1073/pnas.93.19.10078; Srivatsan et al. 2020 (dataset).
- **Adjacent (activity-readout, not counts) options — flagged partial.** OptoFGFR1 → ERK-KTR
  with **7 calibrated light doses** (BioStudies **S-BIAD2275**, Kramar et al., *PRX Life* 2025)
  and ERK-KTR across 429 kinase inhibitors (IDR **idr0064**, Goglia & Toettcher, *Cell Systems*
  2020) are the cleanest single-cell ERK dose-responses, but the readout is **ERK activity, not
  counts** — feed the switch/readout layer, not the count model (needs-work/stretch).
- **Honest gap.** No verified public **EGF→ppERK phospho-flow dose FCS** was found; the founding
  Huang–Ferrell / Ferrell–Machleder oocyte results are pooled immunoblots, not single-cell
  deposits — mechanism grounding only.

### C3. Morphogen dose ladder — Shh–Gli K/n/vmax on real cells ⟶ cleanest morphogen fit
- **The question.** Morphogen interpretation (Shh→Gli) is ultrasensitive/bistable. Across a
  clean SAG (Shh-agonist) dose ladder, does NUDGE recover the threshold/gain/ceiling of the
  Gli response, and correctly separate it from co-applied morphogen arms?
- **NUDGE capability.** K/n/vmax attribution over a **single-agonist 4-point dose ladder** — a
  near-purpose-built switch test bench.
- **Dataset.** Amin et al. (Pașca) multiplexed morphogen screen — **GEO GSE233574** (Zenodo
  10.5281/zenodo.13835782; UCSC Cell Browser). MTX raw counts + metadata, 57 samples, ~36k QC
  cells, 46 conditions; **SAG 50/250/1000/2000 nM** + RA/CHIR/BMP/FGF arms + LDN-only control.
  **Structural fit: excellent** — raw counts ✓, clean dose ladder ✓, control ✓, light download.
- **Expected honest result.** NUDGE attributes the Shh dose-response mechanism (or abstains on
  the coarse rungs); the extra morphogen arms are a natural cross-mechanism decoy set.
- **Feasibility: ready.** **Key papers:** Amin et al., *Cell Stem Cell* (2024) — multiplexed
  morphogen screen in organoids. Shh-Gli ultrasensitivity grounding: Dessaud et al., *Nature*
  450:717 (2007), doi:10.1038/nature06347 (bulk/imaging — grounding only).
- **Bigger sibling.** Treutlein/Camp six-morphogen screen — **ArrayExpress E-MTAB-15667**
  (repro E-MTAB-15622; Zenodo 17225179), **SHH/WNT/FGF8/RA/BMP4/BMP7 each at 5 concentrations**
  + no-morphogen controls, ~100–210k cells, raw counts. A literal K/n/vmax bench across six
  switches; **needs-work** (larger, more assembly). Sanchís-Calleja/Azbukina et al., *Nat
  Methods* (2026), doi:10.1038/s41592-025-02927-5.

### C4. Epigenetic switch onset — X-inactivation with Xist as an intrinsic hidden node
- **The question.** Random X-inactivation is a bistable silencing switch driven by **Xist**, an
  intrinsic hidden regulator. Along the silencing sigmoid, is a perturbation moving threshold
  vs gain, and does NUDGE flag Xist-driven feedback as a hidden node?
- **NUDGE capability.** Attribution on a silencing sigmoid + **hidden-node** (Xist) detection.
- **Dataset.** Pacini et al. 2021 — **GEO GSE151009** ("Dissecting the onset of random
  X-chromosome inactivation at high temporal and allelic resolution"). Mouse TX1072 F1, 105
  samples; **allele-specific raw UMI matrices** (spliced/unspliced/combined) + SNP file. Time
  axis (diff days 0–4) = operating points; day 0 = control; a **dXic-deletion perturbation
  arm**. **Structural fit: good** — raw counts ✓, control ✓, time operating points ✓, hidden
  node (Xist) built in. **Feasibility: needs-work** (allele-specific handling).
  **Key papers:** Pacini et al., *Nat Commun* 12:3638 (2021), doi:10.1038/s41467-021-23643-6.
- **Honest gap.** The modern engineered epigenetic-memory switches — **CRISPRoff** (Nuñez et al.
  2021, GSE168012) and its KRAB-duration×strength successor — are the ideal *design* but deposit
  **bulk** (or imaging/CUT&RUN), not ingestible single-cell flow/counts. Bintu et al. 2016
  (canonical bistable chromatin) has **no public per-cell deposit**. Synthetic-generator territory.

---

## Domain D — Synthetic biology & bioengineering (engineered circuits)

> **Field-level honest finding.** Synthetic biology is **data-poor at the per-cell level**:
> the canonical circuit papers (Gardner–Collins toggle 2000; Elowitz repressilator; Cello/
> Nielsen 2016; To & Maheshri 2010) publish **figures/summary stats, not downloadable per-cell
> data**, and "sort-seq/flow-seq" datasets are *sort-then-sequence* enrichment (not per-cell
> traces) — a structural non-fit. The one outstanding exception is the Phillips-lab LacI/IPTG
> corpus, which is essentially a purpose-built threshold-vs-gain tuning matrix with public
> per-cell flow + code. **Shared adaptation for every dataset here: single-fluorescence flow,
> not raw UMI counts — treat fluorescence as the `Readout` channel (a readout swap, not a
> structural blocker).**

### D1. LacI mutant benchmark — DNA-binding = threshold(K), inducer-binding = gain(n) ⟶ ground-truth attribution
- **The question.** The cleanest ground-truth K-vs-n test in existence: LacI point mutants split
  into **DNA-binding-domain** mutants (perturb *only* DNA affinity ⇒ pure **threshold/K** shift)
  and **inducer-binding-domain** mutants (perturb *only* allosteric sensitivity ⇒ a **gain**-like
  change). The authors decomposed this thermodynamically — so there are ground-truth labels for
  exactly NUDGE's distinction.
- **NUDGE capability.** Mechanism attribution K vs n **against author-provided ground truth** — a
  ready-made validation/benchmark of NUDGE's core claim.
- **Dataset.** Chure et al. 2019 "Energetics of Molecular Adaptation" — **CaltechDATA DOI
  10.22002/D1.1241** + `rpgroup.caltech.edu/mwc_mutants` (data + code). LacI DNA-binding and
  inducer-binding mutants across the full **IPTG dose series × repressor copy numbers ×
  operators**. **Structural fit: excellent structure, partial modality ⚠** — per-cell flow ✓,
  dose × variant operating points ✓, uninduced/Δrepressor controls ✓, but **single-fluorescence
  readout, not counts** (a subset of raw FCS was lost — use the preprocessed per-condition data).
- **Expected honest result.** NUDGE attributes DNA-binding mutants → threshold and inducer-binding
  mutants → gain, matching the published decomposition (or abstains where copy-number/operator
  degeneracy bites — a documented honest limit).
- **Feasibility: needs-work** (flow→readout adapter; small tidy files). **Key papers:** Chure et
  al., *PNAS* 116:18275 (2019), doi:10.1073/pnas.1907869116.

### D2. LacI/IPTG induction — operator swap = threshold tuning
- **The question.** Swapping the operator (O1→O2→O3, weaker DNA binding) shifts the induction
  midpoint (K) without changing the ceiling; changing repressor copy number tunes leakiness/
  dynamic range. Can NUDGE read "which knob moved" off the curves?
- **NUDGE capability.** Threshold(K) attribution across a **~12-dose × 3-operator × 6-copy-number**
  factorial — far more operating points than NUDGE needs.
- **Dataset.** Razo-Mejia et al. 2018 allosteric induction — **CaltechDATA DOI 10.22002/D1.743**
  + GitHub `RPGroup-PBoC/mwc_induction` (Zenodo-archived; processed single-cell CSVs incl.
  `flow_master.csv`). **~12 IPTG doses (0–5000 µM) × O1/O2/O3 × 6 repressor copies**, with
  autofluorescence + Δrepressor constitutive controls. **Structural fit: excellent structure,
  partial modality ⚠** — per-cell flow ✓, huge dose × variant matrix ✓, controls ✓, single YFP
  readout (tidy CSVs easier than the `.fcs`). **Feasibility: needs-work** (flow adapter).
  **Key papers:** Razo-Mejia et al., *Cell Systems* 6:456 (2018), doi:10.1016/j.cels.2018.02.004.

### D3. Bistable toggle with hysteresis — a memory a plain Hill fit can't explain
- **The question.** A LacI/TetR mutual-repression toggle is genuinely bistable and **hysteretic**
  (its state depends on history, not just current inducer). Does NUDGE attribute the switch
  mechanism, and — the interesting part — **abstain or flag off-model** on the hysteresis a 1-D
  memoryless Hill fit structurally cannot represent (a candidate hidden-feedback/off-model demo)?
- **NUDGE capability.** Attribution + **abstention/off-model on true bistability + hysteresis**
  (the 2-node toggle safety case from `test_toggle_nd_safety.py`, now on real data).
- **Dataset.** Chu/Zhu et al. 2025 synthetic toggle — **Zenodo 10.5281/zenodo.11817798** (214 MB,
  CC-BY): `response_curve.zip` = IPTG dose axis; `FACS_files_for_LO1…` = different initial states
  (hysteresis); ≥50k events/sample. **Structural fit: strong, one caveat ⚠** — single-cell flow
  ✓, inducer dose + initial-state operating points ✓, control ✓, genuine bistability+hysteresis;
  confirm the `.fcs` extension on download (one small unverified detail). **Feasibility:
  needs-work** (flow adapter). **Key papers:** "Colony pattern multistability emerges from a
  bistable switch," *PNAS* (2025), doi:10.1073/pnas.2424112122.

### D4. Continuous TF dose atlas — 384 titration curves in one experiment
- **The question.** With a **continuous per-cell TF dose** readout across 384 TFs, which TFs act
  as high-gain switches vs graded/threshold regulators — and where does NUDGE abstain?
- **NUDGE capability.** Attribution over the **richest possible continuous dose axis** (per-cell,
  not binned); spans oncogene dosage (MYCN characterized as a high-capacity dose-sensitive TF)
  and differentiation TFs (MYOD1/MYOG/PPARG/CEBPA/RUNX2/GATA2).
- **Dataset.** scTF-seq — **ArrayExpress E-MTAB-13010** (code: DeplanckeLab/TF-seq). Mouse
  C3H10T1/2, dox-inducible barcoded overexpression of 384 TFs, ~46k cells, **continuous per-cell
  dose**, mCherry controls; processed 10x `.h5` (raw counts) + `.rds`. **Structural fit: strong**
  — raw counts ✓, continuous dose ✓, control ✓; ~116 cells/TF and multi-GB assembly.
  **Feasibility: needs-work.** **Key papers:** Rai/Yang et al., *Nat Genet* (2025),
  doi:10.1038/s41588-025-02343-7.

### D5. Bimodality-without-bistability — the abstention decoy (synthetic; no public data)
- **The question.** To & Maheshri 2010 showed a **non-cooperative (n≈1) positive-feedback loop
  can be bimodal without any deterministic switch**. A naive reading of the histogram screams
  "cooperative switch"; NUDGE must **abstain (off-model / not-a-switch)** rather than call gain.
- **NUDGE capability.** The scientifically-validated **abstention decoy** — its deterministic
  Hill fit must not beat linear beyond the noise floor.
- **Dataset.** **Honest gap — no public per-cell data** (2010, pre-FlowRepository; Addgene has
  strains only). **Simulate** it (the Tier-0.5 stochastic route already in `data/stochastic.py`
  can carry an n≈1 noise-induced-bimodality decoy) and state the gap openly. **Feasibility: ready
  (synthetic).** **Key papers:** To & Maheshri, *Science* 327:1142 (2010),
  doi:10.1126/science.1188308.

---

## Domain E — Extra bistable-switch domains (oncogene dosage, microbial, differentiation)

### E1. Hematopoietic toggle — Gfi1–Irf8 (GATA1/PU.1-class) trapped intermediate
- **The question.** The myeloid Gfi1↔Irf8 mutual-antagonism toggle (GATA1/PU.1-class) produces a
  "trapped intermediate" in double-KO. Does NUDGE attribute the genotype effects, and correctly
  **abstain / flag hidden-node** on the trapped-intermediate state a 1-D model can't place?
- **Dataset.** Olsson/Grimes et al. 2016 — **GEO GSE70245** (WT + Gfi1⁻/⁻ + Irf8⁻/⁻ + double-KO
  GMPs). **Structural fit: partial ⚠** — the toggle *story* is textbook, but data are **plate
  SMART-seq quantified as TPM/RSEM, not raw UMI counts**, and perturbations are discrete
  genotypes (no dose). **Use as a qualitative toggle + abstention/hidden-node stress test, not a
  raw-count benchmark.** **Feasibility: stretch.** **Key papers:** Olsson et al., *Nature*
  537:698 (2016), doi:10.1038/nature19348.

### E2. Human TF-ORF atlas — lowest-friction breadth (ready `.h5ad`)
- **Dataset.** Joung et al. 2023 TF-ORF atlas — **GEO GSE216595** (90 TF ORFs, hESC; ships a
  **ready processed `.h5ad` ~1.4 GB** — the least-friction file in this whole library — + raw
  MTX; MYC lives in the broader GSE216481 superseries, 16.7 GB raw). **Structural fit: partial
  ⚠** — raw counts ✓, control ✓, but operating points are only **pseudo-dose** (per-cell ORF
  expression), a stretch for the degeneracy breaker. Good low-friction breadth / smoke-test.
  **Feasibility: ready (for I/O), stretch (for attribution).** **Key papers:** Joung et al.,
  *Cell* 186:209 (2023), doi:10.1016/j.cell.2022.11.026.

### E3. Endogenous morphogen gradient — hidden-gradient inference (Briscoe spinal cord)
- **Dataset.** Delile et al. 2019 mouse spinal cord atlas — **ArrayExpress E-MTAB-7320**, 53,549
  cells, raw counts + DV domain labels. **Structural fit: partial ⚠** — **no applied dose / no
  0-dose control**; the DV axis is the *endogenous* Shh gradient ⇒ good for **hidden-node /
  latent-gradient inference and abstention**, not applied-dose K/n/vmax. **Feasibility: stretch.**
  **Key papers:** Delile et al., *Development* 146:dev173807 (2019), doi:10.1242/dev.173807.

### E4. Classic biophysics switches with **no ingestible public data** → synthetic generators (honest gap)
- **The finding.** The founding single-cell switch systems were established by **imaging/immunoblot
  reporters, not deposited count/flow tables**, so they are **honest gaps** — best served by
  **synthetic generators from their published ODEs** (fitting NUDGE's verification-vs-validation
  split, and reinforcing the don't-overclaim thesis rather than papering over it):
  - **B. subtilis competence / ComK** (Süel et al., *Nature* 440:545 (2006), doi:10.1038/nature04588)
    — time-lapse movies, no tables. **λ phage lysis/lysogeny** (Zeng/Golding) — microscopy only.
    **Vibrio harveyi quorum sensing** (Long/Bassler 2009) — flow across autoinducer doses in
    concept, but **no public deposit located**.
  - **Rb-E2F restriction point** (Yao et al., *Nat Cell Biol* 10:476 (2008), doi:10.1038/ncb1711)
    → ODE at **BioModels BIOMD0000000318**. **TRAIL apoptosis** (Spencer/Sorger 2009) and the
    **Huang–Ferrell MAPK** cascade — reporter imaging + published ODEs, no count matrix.
  - **Oncogene copy-number dosage** (ecMYCN, Stöber et al., *Cell Reports* 2024) — the *ideal*
    paired single-cell DNA-CN + MYCN-RNA ultrasensitivity dataset, but **EGA controlled-access
    (EGAS50000000509)** — not a drop-in demo. Flag as controlled-access.
  - **Recommendation.** Ship these as **synthetic-ODE decoys/generators** with the gap stated
    plainly; they exercise abstention + hidden-node where real data honestly doesn't exist.

---

## Appendix 1 — Datasets shortlist (the 5 most-accessible, best-fit to start with)

1. **sci-Plex — GEO GSE139944** (drug dose + abstention). One-line `.h5ad` via `pertpy`; raw
   counts + 4-pt dose + vehicle control. The single most turnkey attribution+specificity demo.
2. **Norman 2019 — GEO GSE133344** (A/B/A+B combos). One-line `pertpy`; small (~91k cells); the
   cleanest combination-breaker sandbox. Raw counts + non-targeting control.
3. **OCT4/NANOG dosage Perturb-seq — GEO GSE283614** (K vs n with published ground truth). Small
   MTX (218/420 MB); the flagship *attribution-with-a-known-answer* case.
4. **Schmidt 2022 CD4+ T-cell CRISPRa — GEO GSE190604** (stim/unstim operating points). ~1 GB
   clean 10x + per-cell guide calls; ties directly to NUDGE's own Ras/SOS circuit + loader.
5. **Pașca morphogen screen — GEO GSE233574** (Shh–Gli dose ladder). MTX raw counts, light
   download, a near-purpose-built K/n/vmax bench with built-in cross-morphogen decoys.

(Runner-up / best ground-truth benchmark despite the flow adapter: **Chure 2019, CaltechDATA
D1.1241** — DNA-binding = K vs inducer-binding = n, author-labelled.)

## Appendix 2 — Flag list: controlled-access, heavy downloads, and unverified details

- **Heavy download (use a mirror / subset):** Gladstone genome-scale **GSE314342** (159 GB GEO
  RAW.tar — prefer the **CZI Virtual Cells `.h5ad` mirror** or on-disk subset); **sci-Plex-GxE
  GSE225775** (>1M cells); **Replogle gwps** (tens of GB — take essential/RPE1); Treutlein/Camp
  **E-MTAB-15667** (~100–210k cells + assembly); yeast logic-circuit flow (Zenodo 6562250, ~86 GB).
- **Controlled access:** **ecMYCN copy-number** dataset (EGA **EGAS50000000509**) — not a drop-in.
  Human primary-cell T-cell Perturb-seq (**GSE190604**, **GSE314342**) was verified **PUBLIC** on
  GEO (not dbGaP/EGA) — no gate.
- **Access friction / verify-on-download:** MIX-seq figshare 10298696 (scraper 403 = bot-block,
  landing page public); FlowRepository entries (e.g. **FR-FCM-Z3DB**) currently serve an expired
  TLS cert (browser-downloadable, not machine-verifiable); toggle-switch **Zenodo 11817798** —
  confirm the `.fcs` extension; scTF-seq panel — GATA1/SPI1 membership unconfirmed.
- **Honest gaps (no ingestible public single-cell data — use synthetic generators):** canonical
  hidden-node circuits (Mukherji 2011, Bleris 2011, ceRNA sponges); classic biophysics switches
  (ComK, λ phage, Vibrio QS, Rb-E2F, TRAIL apoptosis, Huang–Ferrell MAPK); To & Maheshri 2010
  bimodality-without-bistability; EGF→ppERK phospho-flow dose FCS. Each is mechanism grounding +
  a synthetic-ODE decoy, stated openly.

## Appendix 3 — Adapters these notebooks assume (small, reusable)

- **Flow-cytometry readout adapter** (D1–D3, parts of C2): map single-channel fluorescence to the
  `Readout` channel; NUDGE already accepts single-cell-resolution flow per its data contract, so
  this is a readout swap, not a new model. Reuses `inference/bridge.py` conceptually.
- **On-disk subsetting** for the giant Perturb-seq files (A3, B4, B5): already provided by
  `data/loaders/perturbseq.py` backed mode (the Gladstone config in `tier2.py` is the template —
  a new dataset is a new config, not new code).
- **Metadata→condition mapping** (B3 MIX-seq, C4 XCI allele-specific): parse the demultiplexing /
  SNP / allele tables into `obs['condition']`. The generic loader config covers column renames.
