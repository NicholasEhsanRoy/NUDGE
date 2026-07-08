# Synthetic Perturb-seq generator — design decisions (literature-grounded)

Distilled from a `/deep-research` pass (112 agents; 29 primary sources; 24/25
claims survived 3-0 adversarial verification). Each area gives **(i)** the state
of the art, **(ii)** the decision for NUDGE's generator, **(iii)** the failure
mode to guard against. Confidence is HIGH unless marked. Citations at the end.

The generator's job: produce raw-count `AnnData` with ground-truth mechanism
labels, realistic enough that recovering mechanism from it *means something*.

---

## 1. Count / observation model — **NB, not zero-inflated** (HIGH)

- **State of the art.** The zero-inflation debate is, for UMI/droplet data,
  effectively settled: droplet scRNA-seq is **not** zero-inflated. Svensson 2020
  shows negative-control zero rates match a plain NB to ~99.9%; Townes 2019 shows
  UMI counts follow multinomial sampling with no zero inflation; Sarkar & Stephens
  2021 find only 0.6–9% of genes favour a zero-inflated (point-Gamma) over a plain
  Gamma; Jiang 2022 select non-zero-inflated Poisson/NB for almost all genes in
  10x/Drop-seq. Zero-inflation is **protocol-dependent** — real only for non-UMI
  (Smart-seq2), where it's an amplification artifact (Ahlmann-Eltze & Huber 2023).
- **Decision.** Perturb-seq is overwhelmingly 10x (UMI) → **sample counts from a
  negative binomial (Poisson–Gamma). No Bernoulli dropout mask.** Expose a
  `protocol` flag that switches to a compound amplification model only for
  Smart-seq2. Parameterize NB **dispersion as a decreasing function of the mean**
  (Splatter's mechanism) so the empirical mean–variance and zero-fraction-vs-mean
  curves come out right.
- **Failure mode.** A ZINB `π` term double-counts zeros already produced by
  low-depth sampling, biases the mean–variance trend, and — worst for us —
  **injects spurious bimodality a switch-detector could misread as
  ultrasensitivity**. Constant (mean-independent) dispersion gives the wrong
  zero-fraction curve and low-expression noise that can mimic or mask a switch's
  OFF mode.

## 2. Two-layer architecture — **biology → measurement** (HIGH)

- **State of the art.** Every credible simulator separates a **biological
  expression model** `p(Λ)` from a **technical measurement model** `p(X | Λ)`:
  SERGIO solves real-valued expression via an SDE then Poisson-samples; Splatter
  draws Gamma means then Poisson-samples (→ NB); SymSim's high zero fraction is
  *emergent* from its capture/amplification/sequencing pipeline, not a forced
  dropout. Sarkar & Stephens frame this as the resolution to the zero-inflation
  confusion.
- **Decision.** **The NUDGE circuit ODE produces per-cell steady-state
  expression Λ (carrying the threshold/gain/ceiling mechanism); a separate
  technical layer then applies library-size scaling and NB/Poisson capture.**
  Keep the two layers distinct in code (`Readout` = biology→Λ; `noise.py` =
  Λ→counts).
- **Failure mode.** Collapsing biology and technical noise into one fitted
  distribution makes the mechanism parameters **non-identifiable** and blocks
  clean ground-truth labelling — it would defeat the entire benchmark.

## 3. Raw counts, not log-CPM (HIGH)

- **State of the art.** log-CPM + HVG selection injects false variability (zero
  fraction can drive PC1); GLM-PCA / Pearson residuals on raw counts is the fix
  (Townes 2019; Lause/Kobak/Berens 2021).
- **Decision.** The generator's **output contract and self-tests are defined on
  raw UMI counts** (library-size distribution, dropout-vs-mean curve, gene–gene
  correlation). Downstream normalization is the *method's* concern, never the
  generator's. (This matches NUDGE's raw-counts ingestion guardrail.)
- **Failure mode.** Fitting the generator to reproduce log-normalized statistics
  encodes a normalization artifact as ground truth.

## 4. Mechanistic layer — **add cooperativity explicitly** (HIGH)

- **State of the art.** SERGIO (chemical-Langevin + Hill) is the reference for
  GRN-driven dynamics, but it combines multiple regulators **additively** (a sum
  of independent per-regulator Hill terms + basal); the authors state cooperative
  regulation is *not* considered, and an additive sum **cannot** produce
  AND-gate / combinatorial ultrasensitivity. (Separately: a claim that SERGIO
  reproduces realistic gene–gene co-expression was **refuted 0-3** — do not
  assume any simulator's emergent correlation structure is validated.)
- **Decision.** NUDGE's circuit must represent **cooperativity** (cooperative /
  multiplicative Hill terms) so that **threshold (K), gain (Hill n), and ceiling
  (v_max) are independently tunable knobs** — exactly the kernels already built
  in `mechanisms/regulatory.py`. Use SERGIO/BoolODE as an *independent* Tier-0.5
  simulator, **not** the core model. This is a direct validation of our approach:
  the switch-like signal NUDGE attributes is *structurally impossible* to
  generate additively, so our multiplicative-Hill circuit is the right generator.
- **Failure mode.** An additive regulator combination makes the ultrasensitivity
  signal structurally impossible to generate — the benchmark would have no true
  positives.

## 5. Perturbation modeling — **partial, causal, cell-variable** (HIGH)

- **State of the art.** GRouNdGAN imposes a GRN causally (a target sees only its
  regulators' expression) and samples interventional distributions; a single-TF
  knockout significantly altered targets in **66.5%** of cases — i.e. a KO does
  **not** cleanly change *all* descendants (multiple regulators dilute the
  effect). Genome-scale Perturb-seq (Replogle 2022; Norman 2019) supplies the
  empirical knock-down magnitudes and variability.
- **Decision.** Model a perturbation as a **continuous, cell-variable reduction
  of the target's production** that **propagates only to causal descendants**
  through the circuit edges — *not* a binary, uniform, all-targets knockout.
  Include: partial + cell-to-cell-variable knockdown efficiency, MOI (multiple
  guides/cell), guide-assignment error, and non-targeting controls. This is the
  `Perturbation` latent.
- **Failure mode.** Assuming every guide yields a clean, complete, uniform
  knockdown of all downstream targets is unrealistic and would make the fit look
  artificially easy (a Tier-0 inverse-crime amplifier).

## 6. Realism self-tests — **countsimQC / SimBench battery** (HIGH)

- **State of the art.** `countsimQC` (Soneson 2018) evaluates synthetic vs real
  counts across: **mean–dispersion relationship, library-size distribution,
  expression distribution, fraction-of-zeros per gene and per sample, and how
  those zero fractions relate to expression level and to total reads.** SimBench
  (Cao 2021) grades 12 simulators over 35 datasets on 13 data properties via a
  KDE measure. These are ready-made self-test batteries.
- **Decision.** The generator ships a **realism self-test suite** asserting, on
  raw counts: mean–variance trend, zero-fraction-vs-mean curve, library-size
  distribution, and gene–gene correlation fall within tolerance of a real
  reference (a small Tier-1 dataset). These become `tests/data/` checks.
- **Failure mode.** Skipping realism tests lets the generator drift into
  unrealistic regimes where NUDGE's measured accuracy is meaningless.

## 7. Anti-inverse-crime tiering (HIGH for the principle)

- **State of the art.** ADEMP (Morris, White & Crowther 2019) is the canonical
  simulation-study framework. A benchmarking study found **simulators that better
  mimic reference data do NOT necessarily yield more similar method-comparison
  results** — realism does not guarantee benchmark reliability, and self-simulation
  (generator and method sharing assumptions) is a circularity risk.
- **Decision.** Keep the **tiered ladder**, and sharpen its rationale: **Tier 0**
  = NUDGE's own deterministic circuit (fast, exact ground truth, mechanism labels
  we control) — *inverse-crime-prone by construction*; **Tier 0.5** = an
  **independent, stochastic** simulator (SERGIO's Langevin, or a Gillespie
  toggle-switch) whose bimodality is *emergent*, not designed-in. Passing Tier 0.5
  is the real robustness claim.
- **Failure mode.** Reporting only Tier-0 accuracy would be a textbook inverse
  crime — NUDGE recovering exactly the deterministic structure it was given.

---

## THE CRUX, RESOLVED — deterministic-vmap near bistability (focused review, HIGH)

A second focused `/deep-research` pass (103 agents; primary stochastic-gene-
expression theory) settled this. **The elegant `vmap` design survives —
validated, with its limits named.**

**The settled physics.**

- **Validity boundary (Elf & Ehrenberg 2003; Thomas & Grima 2013).** The
  deterministic-steady-state + Gaussian/parameter-draw (LNA-regime) picture is
  valid only *locally around a single stable fixed point, at high copy number,
  far from the bifurcation*, with fast intrinsic relaxation vs extrinsic drift.
  It **fails near a saddle-node / bistable point**: it cannot represent bimodality
  or a continuum of states, and near criticality (near-singular Jacobian /
  critical slowing down) fluctuations are large and slow; at low copy number it
  can even get the *direction* of noise dependence wrong.
- **Occupancy is emergent, not imposed (Wang/Xu/Wang 2008; Zhou et al. 2012;
  Zheng et al. 2011).** In the true stochastic system the mode weights are set by
  the quasi-potential landscape (`U = −ln P_ss`, `P_ss ∝ exp(−U)`) — basin depths,
  noise-induced switching — and GRNs are *non-gradient* (a nonzero curl flux
  breaks detailed balance), so occupancy **cannot** be recovered by integrating
  the deterministic force. A deterministic solve locates the extrema of `U`, not
  their probability depths.
- **Kepler & Elston 2001.** Deterministic and stochastic models can disagree on
  the very *number* of stable states (stochastic bistability where the ODE is
  monostable and vice versa), and **promoter-switching rate is an independent
  bifurcation axis** a deterministic-plus-parameter generator structurally omits.

**Decision — the Tier-0 / Tier-0.5 split is confirmed and now rigorously grounded.**

- **Tier-0 = deterministic-`vmap`.** Ochab-Marcinek & Tabaka 2010 (PNAS) is
  *exactly* our mechanism: a deterministic nonlinear transfer function acting as a
  noise filter maps a **unimodal** extrinsic input into a **bimodal** output in a
  monostable, non-cooperative cascade — genuine bimodality with a **controllable,
  designed-in mode split**, ideal as exact ground truth. Named limitation:
  occupancy is *imposed by our input distribution*, not landscape-set, so Tier-0
  cannot produce noise-induced switching, barrier-height occupancy scaling, or
  promoter-rate bifurcations. Generation integrates to convergence (robust); the
  near-bifurcation fragility is a **fit-time** concern only.
- **Tier-0.5 = independent stochastic simulator** (SERGIO chemical-Langevin /
  Gillespie toggle / BoolODE-with-noise) with **emergent** bimodality — the honest
  test that NUDGE's deterministic fit generalizes to landscape-set occupancy.
  (Which simulator best reproduces correct occupancy scaling is itself open; the
  split is a physics-grounded inference, not a Perturb-seq-validated claim.)

**Two decisions this hands us beyond the split:**

1. **New, scientifically-validated decoys — "bimodality is NOT proof of a
   switch"** (To & Maheshri 2010, the key interpretive caveat). The battery gains
   three *real* bimodal-but-not-bistable negatives NUDGE must not misread as
   ultrasensitivity:
   - **Noise-induced feedback bimodality** (To & Maheshri 2010): non-cooperative
     (Hill `n ≈ 1`) positive feedback + short-lived TF + bursty promoter → bimodal
     with **no gain** → NUDGE must decline a gain/threshold switch call.
   - **Bursting bimodality** (Shahrezaei & Swain 2008): slow promoter switching,
     no feedback, first-order kinetics → bimodal (analytically NB) → `technical` /
     no switch.
   - **Steep-monostable transfer-function bimodality** (Ochab-Marcinek & Tabaka
     2010): a steep *monostable* dose-response on a unimodal input → bimodal
     without bistability. The hardest decoy — NUDGE must distinguish "steep
     monostable dose-response" from "bistable switch".
   - **(Reverse, open)** a deterministically bistable system can look *unimodal*
     in a snapshot (one basin dominant) — NUDGE must not over-claim its absence.
2. **A literature-grounded abstention trigger.** Near a saddle-node the Jacobian
   is near-singular → **threshold vs gain is genuinely non-identifiable on
   near-critical snapshot data**. This is the concrete, physics-based trigger for
   the `unresolved` verdict, dovetailing with the borrowed blindness diagnostic:
   detect critical slowing down (Jacobian condition number) and abstain. A
   Fisher-information / identifiability analysis to place that boundary is the
   recommended next derisking step.

**Net.** Tier-0 is a controllable deterministic ground-truth generator (occupancy
is a *knob* — a feature for benchmarking); Tier-0.5 supplies emergent-occupancy
realism; bimodality-without-bistability becomes a first-class decoy class; and
near-criticality becomes a principled abstention boundary. This shapes
`inference/population.py`, `inference/classify.py` (the abstention gate), and the
decoy battery.

---

## Citations (primary, DOI where available)

- Svensson 2020, *Nat Biotechnol* — droplet scRNA-seq is not zero-inflated. 10.1038/s41587-019-0379-5
- Sarkar & Stephens 2021, *Nat Genet* — separating measurement and expression. 10.1038/s41588-021-00873-4
- Townes et al. 2019, *Genome Biol* — GLM-PCA / multinomial / raw counts. 10.1186/s13059-019-1861-6
- Jiang et al. 2022, *Genome Biol* — zero-inflation is protocol-dependent. 10.1186/s13059-022-02601-5
- Ahlmann-Eltze & Huber 2023, *bioRxiv* — compound model for non-UMI amplification. 2023.08.02.551637
- Dibaeinia & Sinha 2020 (SERGIO), *Cell Systems* — Langevin+Hill, additive limitation. 10.1016/j.cels.2020.08.003
- Song et al. 2024 (scDesign3), *Nat Biotechnol* — in-silico controls, marginals+copula. 10.1038/s41587-023-01772-1
- GRouNdGAN 2024, *Nat Commun* — causal GRN, interventional perturbation, 66.5% KO efficacy. PMC11525796
- Zappia et al. 2017 (Splatter), *Genome Biol* — Gamma-Poisson, mean-dependent dispersion. 10.1186/s13059-017-1305-0
- Zhang et al. 2019 (SymSim), *Nat Commun* — emergent zeros from capture pipeline. 10.1038/s41467-019-10500-w
- Cao et al. 2021 (SimBench), *Nat Commun* — realism evaluation framework. 10.1038/s41467-021-27130-w
- Soneson & Robinson 2018 (countsimQC), *Bioinformatics* — count-data QC battery. PMC6612870
- Morris, White & Crowther 2019 (ADEMP), *Stat Med* — simulation-study design. arXiv:1712.03198
- Elowitz et al. 2002, *Science* — intrinsic vs extrinsic noise. 10.1126/science.1070919
- Swain, Elowitz & Siggia 2002, *PNAS* — noise-decomposition theory. 10.1073/pnas.162041399
- Replogle et al. 2022, *Cell* — genome-scale Perturb-seq empirics. S0092867422005979
- Norman et al. 2019, *Science* — Perturb-seq of genetic interactions. 10.1126/science.aax4438

**Bistability / noise (the resolved crux):**

- Kepler & Elston 2001, *Biophys J* — stochastic vs deterministic (non)bistability; promoter-rate bifurcations. 10.1016/S0006-3495(01)75949-8
- Wang, Xu & Wang 2008, *PNAS* — quasi-potential landscape U = −ln P_ss; non-gradient flux. 10.1073/pnas.0800579105
- Zhou, Aliyu, Aurell & Huang 2012, *J R Soc Interface* — GRNs non-gradient; attractor relative stability. 10.1098/rsif.2012.0434
- Zheng, Yang & Tao 2011, *PLoS ONE* — autoactivation ρ_ss ∝ exp(−U/D²); landscape-set occupancy. 10.1371/journal.pone.0017104
- To & Maheshri 2010, *Science* — bimodality without bistability (noise-induced). 10.1126/science.1178962
- Shahrezaei & Swain 2008, *PNAS* — three-stage bursting; bimodality from slow promoter switching; NB. 10.1073/pnas.0803850105
- Ochab-Marcinek & Tabaka 2010, *PNAS* — deterministic transfer function → bimodal from unimodal input (= Tier-0 mechanism). 10.1073/pnas.1008965107
- Elf & Ehrenberg 2003, *Genome Res* — linear noise approximation; validity + failure near critical points. 10.1101/gr.1196503
- Thomas, Matuschek & Grima 2013, *BMC Genomics* — LNA severe qualitative failure at low copy number. PMC3849541
