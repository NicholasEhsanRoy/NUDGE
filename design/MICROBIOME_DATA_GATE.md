# MICROBIOME DATA GATE — can real longitudinal-microbiome data unlock a temporal (Cap-4) gLV attribution demo?

**Status:** research / verification ONLY. No code was written, no existing file changed.
This is the **Gate-2 data-gate** that `design/EXTENSIBILITY_SPIKE.md` deferred: it recommended a
generalized Lotka–Volterra (gLV) / microbiome extension but explicitly **could not verify the real
data** ("Exact contents/schema of the MDSINE2 Zenodo record and the gerberlab data files were not
downloaded and inspected"). This doc closes that gap by fetching and inspecting the actual repos,
Zenodo records, and raw data files.

**The reframe under test.** NUDGE observes STEADY-STATE snapshots; its deferred **Capability 4
(temporal / time-resolved attribution)** was shelved because scRNA-seq is destructive (a fresh
distribution per timepoint, never a tracked trajectory). The insight: **16S longitudinal microbiome
data provides real per-community TRAJECTORIES with a designed perturbation contrast**, so it could
unlock a trajectory-fit attribution capability — fit a gLV model and attribute WHICH parameter a
perturbation changed (growth **α** vs interaction **β** vs antibiotic-susceptibility **ε**),
abstaining when unidentifiable. NUDGE's philosophy: never invest before confirming the data can
showcase the effect.

---

## VERDICT (one line)

**GO.** The data gate **passes**. Real, accessible, small, clean longitudinal-microbiome data with
**designed** perturbation contrasts exists and directly supports a temporal gLV attribution demo. The
strongest real dataset is the **MDSINE2 "Gibson" gnotobiotic cohort** (richest, densely sampled,
three timed perturbations), with **Stein et al. 2013** as the smallest/cleanest/canonical
"which-parameter-did-the-antibiotic-move" coda and **MTIST** as an external simulated round-trip
benchmark. No need to fall back to amyloid ThT. Build **synthetic-first**, with these as the real coda.

---

## Per-dataset checklist (what I actually fetched and verified)

### 1. MDSINE2 "Gibson" gnotobiotic cohort — STRONGEST real dataset ✅

- **Code repo:** https://github.com/gerberlab/MDSINE2 — **GPLv3** (`LICENSE.txt` verified: GNU GPL v3).
  Open-source Python package, `mdsine2.Study` objects, Colab tutorials.
- **Raw input data lives in the PAPER repo, not the big Zenodo:**
  https://github.com/gerberlab/MDSINE2_Paper → `datasets/gibson/healthy/raw_tables/`. These are the
  small CSV/TSV inputs (verified by fetching them directly):
  - **`counts.tsv`** — ASV × sample count table. **Verified dims: 1,088 ASVs (rows) × 339 samples
    (columns).** Header `"" "1-D0AM" "1-D0PM" …`; rows `"ASV_1" 0 9 9 53 …`. Integer 16S amplicon
    counts.
  - **`metadata.tsv`** — `sampleID | subject | time`. Verified: subjects 1–5; `time` in days with
    **twice-daily AM/PM early sampling** (`0.0, 0.5, 1.0, 1.5, 2.0, 2.5 …`) thinning to daily/less
    later (`…, 5.0, 6.0, 7.0, 10.0, 11.0, 29.0, 29.5`).
  - **`qpcr.tsv`** — `sampleID | measurement1 | measurement2 | measurement3` (triplicate total-load
    qPCR, e.g. `8.1e6, 8.5e6, 9.5e6`). This is what converts relative → **ABSOLUTE abundance**.
  - **`perturbations.tsv`** — **the load-bearing file. Verified full contents:** three explicitly
    timed perturbations applied to subjects 2,3,4,5 —
    | name | start (day) | end (day) | subjects |
    |---|---|---|---|
    | **High Fat Diet** | 21.5 | 28.5 | 2,3,4,5 |
    | **Vancomycin** (gram-positive antibiotic) | 35.5 | 42.5 | 2,3,4,5 |
    | **Gentamicin** (gram-negative antibiotic) | 50.5 | 57.5 | 2,3,4,5 |
  - Plus phylogeny (`sequences.fa`, RDP/SILVA taxonomy) and preprocessed `.pkl` Study objects.
- **Schema summary:** subjects × taxa × timepoints, **absolute abundance** (counts × qPCR), **~76–77
  serial samples/mouse over 65 days** (per the paper; consistent with the 339 samples / 5 subjects I
  counted), **twice-daily during the dense early window.** These are genuine **trajectories** of the
  **same** tracked community — exactly the thing scRNA-seq cannot give.
- **Perturbation contrast:** designed and time-windowed as **pulses** — a diet shift, then a
  gram-positive antibiotic, then a gram-negative antibiotic, each with a clean before/during/after
  window and each mouse acting as its own within-subject control. This maps directly onto NUDGE's
  attribution axis: the antibiotic windows are the **external-perturbation susceptibility ε** contrast;
  the recovery dynamics after each pulse probe **α** (regrowth) vs **β** (which taxa rebound via
  interaction). MDSINE2's own model includes an explicit per-taxon perturbation-response term.
- **Also available:** a parallel **dysbiotic/UC cohort** under `datasets/gibson/` (same schema) — a
  ready second context (healthy vs dysbiotic) that would additionally feed the *differential*
  attribution capability (`inference/differential.py`).
- **Download size / cleanliness:** the raw input tables are a **few MB of CSV/TSV** — trivially
  hackathon-fast. (The Zenodo record 5781848 — "Dynamical systems inference at scale reveals intrinsic
  instability in the dysbiotic microbiome," CC-BY-4.0, **18.7 GB across 5 `.tgz`** — is **MODEL OUTPUTS
  and MCMC traces, NOT the raw input**; you do NOT need it to fit. Raw 16S reads: NCBI **PRJNA784519**.
  Pre-computed inference outputs: Zenodo 10.5281/zenodo.8006853 and 10.5281/zenodo.8208503.)
- **License caveat (verified):** the MDSINE2 *code* is GPLv3. The `MDSINE2_Paper` repo has **no
  top-level LICENSE file** (fetch returned 404) — the data CSVs are redistributed there without an
  explicit per-file license; the canonical open licenses are the CC-BY-4.0 Zenodo outputs and the
  public NCBI reads. For a demo, cite the paper + repo; the tables are openly posted in a public repo,
  but the absence of an explicit data license is a small honesty footnote.

### 2. Stein et al. 2013 (PLoS Comput Biol) — canonical clindamycin → C. difficile, cleanest coda ✅

- **Paper:** https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003388 (CC-BY, PLoS).
- **Data:** **Dataset S1 — a single Microsoft Excel (.xlsx)** supplementary file, "processed taxa
  densities as well as antibiotic profiles, optimal regularization parameters and inferred model
  parameters." Downloadable directly from the article page.
- **Schema (verified from the methods):** **11 groups** (the 10 most-abundant genera **including the
  pathogen C. difficile** + an "Other" category, ≈90% of sequences); **9 time courses** (3 experimental
  conditions × 3 replicate colonies); **≈86 timepoints total**; **absolute abundance** (normalized
  metagenomic abundances × universal 16S rRNA copies/g from qPCR).
- **Perturbation:** **clindamycin modeled explicitly as a unit pulse of length 1 day** — the canonical
  "which parameter did the antibiotic move" contrast, and the exact dataset the gLV-with-external-
  perturbation term was designed on. This is the tightest, smallest showcase of the ε axis.
- **Cleanliness:** one small xlsx, 11 low-dimensional populations — the **easiest possible real fit**;
  low species count is exactly where gLV inference is *least* ill-posed (MTIST shows algorithms succeed
  at 3–10 species and fail at 100). Ideal for a crisp positive.

### 3. MTIST — 648 SIMULATED gLV series, external round-trip / identifiability benchmark ✅

- **Paper:** https://www.biorxiv.org/content/10.1101/2022.10.18.512783 ; PMC11100882.
- **Repo:** https://github.com/granthussey/mtist_platform — **MIT license** (verified). (~857 MB repo.)
- **Contents (verified):** **24,570 simulated time series packaged into 648 datasets**, generated from
  gLV with varying initial abundances, **species counts (3, 10, 100)**, hosts, sampling strategies,
  frequencies, and added noise; ships an "ecological sign (ES)" scoring system. **Simulated, not real.**
- **Use for NUDGE:** an **external, independent round-trip / identifiability benchmark** — validates
  the gLV fit + the abstention boundary against a standard someone else built (the "MNIST for microbiome
  inference"). Its headline finding (all algorithms recover 3–10-species interactions but **fail at 100
  species**) *is* the identifiability story NUDGE would measure and abstain on — free external
  corroboration of the α⇄β / under-determined-β degeneracy.

### 4. Other candidates (documented in the spike; NOT re-verified here)

- **David et al. 2014** (daily gut time-series, incl. a *Salmonella* interval on subject B) and
  **Caporaso "moving pictures" 2011** — dense and real, but perturbations are **incidental** (travel,
  infection), not designed, so the "which parameter changed" contrast is weaker than the timed
  MDSINE2/Stein pulses. I did not re-fetch these; the spike's assessment stands and they are strictly
  dominated by the two designed-perturbation datasets above for *attribution*.
- **Lynx–hare** — single trajectory, **no perturbation contrast**; forward-model validation only.

---

## The load-bearing judgment (answered plainly)

**Q1 — genuine trajectories (dense longitudinal sampling of the same community), not cross-sectional
snapshots?** **YES, verified.** MDSINE2/Gibson: ~76 serial samples/mouse over 65 days, twice-daily in
the dense window, same tracked community (subjects 2–5). Stein: ≈86 timepoints across replicate
colonies. This is exactly the property scRNA-seq lacks and the whole reason the reframe could unlock
Cap-4.

**Q2 — a perturbation contrast that maps to "which gLV parameter changed"?** **YES, verified.**
MDSINE2: three designed timed pulses (diet, vancomycin, gentamicin) with before/during/after windows;
Stein: a clindamycin unit pulse. Both map onto NUDGE's {α growth / β interaction / ε
antibiotic-susceptibility} vocabulary, with the antibiotic windows giving a clean ε contrast and the
recovery dynamics probing α-vs-β.

**Q3 — accessible, open, downloadable now, small, clean enough to fit fast?** **YES.** Stein = one small
xlsx (11 populations). MDSINE2 raw tables = a few MB of CSV/TSV, openly posted, GPLv3 code +
pip-installable + Colab tutorials. Both fit at hackathon speed. (Honesty footnote: MDSINE2_Paper data
CSVs carry no explicit per-file license; cite the paper. Stein is cleanly CC-BY.)

**Q4 — realistically, positive attribution or mostly abstain?** **Mixed, in the on-thesis way — and I
expect a demoable positive on the ε axis.** Full gLV β-matrix inference from short/noisy series is
genuinely ill-posed (α⇄βᵢᵢ degeneracy; under-determined β — MTIST confirms failure at high species
count). BUT the attribution NUDGE actually makes is **narrower** than reconstructing the whole β
matrix: it is "for *this named, time-localized* perturbation, which of {α, β, ε} moved." The
**antibiotic-susceptibility ε axis is the most identifiable** — the drug window is an on/off contrast,
so "which taxa the antibiotic *directly* suppresses (ε) vs which change only *via interaction* (β)" is
plausibly a **crisp positive** and a genuinely cool thing to watch. Separating α from β for the *diet*
shift is where NUDGE should **abstain** — which is exactly the honest, on-thesis behavior. So the
expected shape is **positive on ε / abstain on α-vs-β**, i.e. a demoable positive AND earned
abstentions in the same figure. Keeping the real fit to the **low-species** regime (Stein's 11 groups,
or an aggregated MDSINE2 genus panel) stays on the identifiable side of MTIST's cliff.

**Q5 — synthetic-first + this-dataset-as-coda the right shape?** **Yes.** Prove the round-trip on
synthetic gLV with a known single-parameter perturbation (recover the right knob for clear cases,
abstain never-mis-call on degenerate ones), *measure* the α⇄β degeneracy with the Fisher/Laplace
machinery so the abstention is earned, then land **Stein** (smallest/cleanest) as the first real coda
and **MDSINE2/Gibson** as the richer second, with **MTIST** as the external round-trip check.

---

## GO / NO-GO

**GO — build the temporal / gLV capability.** The data gate passes on all four load-bearing questions:
real trajectories ✅, designed perturbation contrast ✅, open + small + clean ✅, and a realistic path to
a **demoable positive** (the antibiotic ε axis) alongside **honest abstentions** (diet α-vs-β). Do **not**
fall back to amyloid ThT — that was the spike's contingency for exactly this gate failing, and it did
not fail. Recommended build order, gated as in the spike:

1. **Synthetic gLV round-trip first** (Gate 0/1): trajectory field + generator with a known
   single-parameter perturbation; restricted-fit + BIC attribution of {α/β/ε}; Fisher/Laplace to
   *measure* the α⇄β degeneracy; abstain-never-mis-call decoy.
2. **Real coda — Stein 2013 first** (11 populations, clindamycin pulse; the identifiable low-D regime),
   then **MDSINE2/Gibson** (the three timed pulses; healthy vs dysbiotic also feeds `differential`).
3. **MTIST** as an external simulated benchmark for the round-trip + the identifiability boundary.

**Isolation reminder (from the spike, unchanged):** implement as a new `inference/<gLV>.py` module
(own trajectory observable + fit loop, pattern-copied from `fit_parameters`, reusing `losses.py` +
the BIC/abstention gate) that touches **neither `fit.py` nor `core/circuit.py`**.

---

## What I verified vs. could NOT verify (honesty)

**Verified by direct fetch/inspection:**
- MDSINE2 code license = **GPLv3** (`LICENSE.txt`).
- MDSINE2/Gibson raw tables exist and their schemas: `counts.tsv` **1,088 ASVs × 339 samples**;
  `metadata.tsv` subjects 1–5 with twice-daily AM/PM day-stamped sampling; `qpcr.tsv` triplicate
  total-load; **`perturbations.tsv` full contents** (High Fat Diet 21.5–28.5, Vancomycin 35.5–42.5,
  Gentamicin 50.5–57.5, subjects 2–5).
- Zenodo **5781848** = 18.7 GB of **model outputs/MCMC**, CC-BY-4.0 — **not** the raw input.
- MDSINE2 raw reads on NCBI **PRJNA784519**; output Zenodos 8006853 / 8208503.
- Stein 2013 data = **Dataset S1 .xlsx**, 11 groups (incl. C. difficile), 9 time courses, ≈86
  timepoints, absolute abundance, clindamycin as a 1-day unit pulse (from the article methods).
- MTIST = **648 simulated datasets / 24,570 series**, gLV, 3/10/100 species; repo
  `granthussey/mtist_platform`, **MIT**.

**Did NOT verify (flagged, not blocking the GO):**
- I did not download and open Stein's `Dataset S1.xlsx` cell-by-cell — schema is from the paper's
  data-availability + methods text, not the file bytes. (It is a single small public xlsx; low risk.)
- I did not confirm an explicit **per-file data license** on the MDSINE2_Paper CSVs (no top-level
  LICENSE; 404). Cite the paper; treat the tables as openly-posted-but-unlicensed.
- I did not re-fetch David 2014 / Caporaso 2011; I relied on the spike's assessment that their
  perturbations are incidental (weaker for attribution).
- **Whether the real fit yields a non-abstaining positive is an expectation, not a measured result** —
  the honest prediction is *positive on the antibiotic ε axis, abstain on diet α-vs-β*, but that must
  be earned by the actual fit (Gate-2 is only fully closed once the synthetic round-trip + one real
  series are run). The data existence/accessibility/contrast questions ARE now closed.
