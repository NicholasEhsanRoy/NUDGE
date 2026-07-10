# EXTENSIBILITY SPIKE — can NUDGE's engine + attribution philosophy point at a new dynamical-systems domain?

**Status:** scoping / feasibility research only. **No code was written, no file was
changed.** This document answers four questions — (a) is a new domain possible without
breaking the architecture, (b) which candidate is easiest given math *and* public data,
(c) how hard is it, (d) should it run as a parallel spike — and ends with an explicit
recommendation and go/no-go gates.

**One-line verdict.** Yes — a new domain plugs in cleanly as a *sibling* `inference/`
module (new forward-model vector field + new attribution vocabulary) that reuses the
distributional loss, the BIC/restricted-fit/abstention *pattern*, and the honesty
scaffolding, and touches **neither `fit.py` nor `core/circuit.py`**. This is not
speculative: roughly half the capabilities NUDGE has already shipped are exactly this
shape. **Recommended domain: generalized Lotka–Volterra (gLV) for microbial
communities**, run as a small parallel spike, with amyloid-aggregation kinetics as the
documented strong runner-up. The single biggest risk is domain-specific and stated in §C.

---

## (a) Is it possible WITHOUT breaking the existing architecture?

**Yes, and the pattern is already proven inside this repo.** The load-bearing evidence:
`inference/dose_response.py`, `inference/lyapunov.py`, `inference/epistasis.py`,
`inference/differential.py`, `inference/cross_modality.py`, and `inference/bifurcation.py`
are each a self-contained attribution capability that **imports primitives and copies
patterns but modifies neither `fit.py` nor `core/circuit.py`**. `lyapunov.py` even ships
its *own* fit loop (`fit_lyapunov_parameters`) rather than calling `fit()`.
`dose_response.py` fits a *curve* with its own scipy loop and reuses only the Hill
primitive + the BIC gate. That is the template a new domain follows.

### The seam — what is mechanism-AGNOSTIC (reuse verbatim / as a pattern)

| Asset | File | Reusable how |
|---|---|---|
| **Distributional losses** (`energy_distance`, `energy_distance_weighted`, `rbf_mmd`) | `inference/losses.py` | **Verbatim.** Purely sample-based, zero domain assumptions. Any forward model that emits samples can be fit against data with these. |
| **BIC model-selection + restricted-fit + abstention pattern** | `inference/model_select.py`, `inference/dose_response.py` | **As a copied pattern** (each module reimplements `_bic` / the gate ladder). "Free-one-knob restricted fits, winner is the mechanism, abstain unless it beats the noise floor / BIC margin." Directly transplantable. |
| **Honesty scaffolding** — abstention vocabulary, `MechanismRegistry`/`MechanismMeta`, Mechanism Cards, the decoy battery, `known_limitations.yaml`, Laplace/Fisher identifiability method | `core/vocabulary.py`, `core/metadata.py`, `mechanisms/registry.py`, `docs/mechanism_cards/`, `inference/uncertainty.py` | **Verbatim infra + method.** The "abstain-and-attribute, decoy-guarded, card-documented" discipline is domain-neutral. A new domain gets new cards/decoys, same machinery. |
| **The differentiable-fit *idea*** (optax/JAX log-space Adam over free params, minibatch resample each step) | `inference/fit.py` `fit_parameters` | **As a pattern, not a drop-in** (see below). ~40 lines to re-instantiate per domain. |

### What is Hill/switch-SPECIFIC (needs a new analogue in the new module)

| Asset | Why it is domain-bound |
|---|---|
| **The K/n/vmax vocabulary + Hill primitives** (`hill_activation` etc.) | `regulatory.py`. Threshold/gain/ceiling *is* the switch ontology. A new domain brings its own vocabulary (§B). |
| **`Circuit.production`** | Hard-codes dispatch on effect strings (`hill_activation`/`hill_repression`/`linear`) with **additive** combination and the vector field `production(x) − decay·x`. Adding an LV term here would **touch core → forbidden**. The new field is written fresh in the new module instead. |
| **The steady-state-snapshot assumption** | `solve_population` integrates to convergence and the population is a distribution over **per-cell steady states**. This is a *deeper* assumption than Hill: the observable is a snapshot at equilibrium, not a trajectory. It shapes which domains map naturally (§B). |
| **`fixed_points` / `transition_state` / `mode_covariances`** | Bistability-centric. (The N-D Newton finder and the LNA covariance are generic given *any* `production`, so they are reusable infra *if* a domain needs equilibria — but they presume you care about fixed points and their stability.) |
| **`switch_detected`** parsimony gate | Framed as "switch vs linear baseline." A new domain needs its own null (e.g. "no-interaction" for LV, "elongation-only" for amyloid). |
| **The count observation model** (NB moment-matched Gaussian, `check_counts` raw-count bouncer, `_per_cell_params`) | `fit.py` + `data/`. Specific to sequencing counts — but **reusable** for any count/relative-abundance readout (microbiome abundances *are* counts). |

### The clean statement

A new domain plugs in as **`inference/<domain>.py` = {a JAX vector field} + {an attribution
vocabulary} + {reuse `losses.py`} + {copy the BIC/abstention gate} + {new cards/decoys}**,
importing `fit.py`/`core` read-only or not at all. It satisfies the hard constraint. The
**only** way a domain *forces* a change to `fit.py` or `core` is if you try to reuse the
`Circuit`/`fit()` path *directly* by adding a new effect type to `Circuit.production` —
**don't**; write the field standalone, exactly as `dose_response.py`/`lyapunov.py` did.
If a domain cannot be expressed without editing `production`, that is a red flag — none of
the four candidates require it.

### MADDENING's role — a non-constraint

Per `STATE.md §4`, NUDGE already does **not** route through MADDENING's `GraphManager`
(which bakes params as compile-time constants); `Circuit` is a *self-contained JAX vector
field* and reuses only MADDENING *primitives* (`ift_linear_solve`) + `maddening.compliance`
traceability. Consequently **MADDENING does not constrain the vector-field form** — any
JAX-differentiable field (gLV product form, an amyloid moment-ODE, a TMDD binding system)
is admissible. MADDENING is a non-blocker for extensibility.

---

## (b) Which is EASIEST given BOTH the math AND public data?

Ranked. The deciding axis is not "can the ODE be written" (all four can) but the
interaction of **(i) how naturally it fits the engine's observable**, **(ii) whether the
attribution/abstention story is as rich as threshold-vs-gain**, and **(iii) whether real
public data with a perturbation contrast actually showcases the effect.**

| Domain | Forward model fit | Attribution vocabulary | Identifiability / abstention story | Public data w/ perturbation contrast | Rank |
|---|---|---|---|---|---|
| **1. gLV (microbiome/ecology)** | Easy field (`dxᵢ/dt = xᵢ(αᵢ + Σβᵢⱼxⱼ)`, ~10 lines JAX). But its parameter info lives in **transients**, not a steady-state snapshot → fit the trajectory (a *new* observable) rather than reuse `solve_population` as-is. | growth rate **α**, interaction **β** (sign+strength), self-limitation/carrying-capacity **Kᵢ=−αᵢ/βᵢᵢ**, and (MDSINE) **external-perturbation susceptibility ε** | **Rich and real.** α vs βᵢᵢ are degenerate from equilibrium data (separable only via transients); the β matrix is famously under-determined from short/noisy series. A genuine threshold-vs-gain-grade degeneracy → strong abstention story. *Honest caveat:* on real data it is **so** ill-posed that NUDGE may abstain often (honest, but a less flashy positive). | **Strong.** MDSINE/MDSINE2 (Stein 2013 + Buffie C.diff/clindamycin; new densely-sampled perturbation datasets) explicitly fit gLV with an antibiotic-susceptibility term and publish the parameters. Lynx–hare exists but has **no** perturbation contrast (forward-model validation only). | **1** |
| **2. Amyloid aggregation (α-syn / Aβ)** | Moment-ODE / integrated Knowles rate law — a **curve** fit, like `dose_response.py` (proven additive). More math than gLV, but a closed form exists. | primary nucleation **kₙ**, elongation **k₊**, secondary nucleation **k₂**, fragmentation **k₋** | **Richest & best-characterized.** Only *products* (kₙ·k₊, k₂·k₊) are identifiable from a single curve; separating them needs concentration-dependence / seeding. Textbook "abstain when only the product is resolved." Cleaner than LV. | **Clean & small.** WT-vs-mutant-vs-inhibitor ThT curves, AmyloFit-curated, with published reference rate constants to check against. Tiny arrays. | **2 (strong runner-up; cleaner showcase)** |
| **3. PK/PD with TMDD** | Well-posed ODE (drug–target binding + elimination). Fits a trajectory. | which PK/binding constant a covariate/dose changed | Real identifiability literature (QSS/MM approximations exist precisely because the full model is over-parameterized) — a decent story. | **Weak/closed.** Rich TMDD concentration–time data is largely proprietary/clinical; open datasets are sparse and mostly simulated. **Data-gate fails.** | **3 (down-rank: no open showcasing data)** |
| **4. ODE→PDE reaction–diffusion / Turing** | **Biggest leap.** Adds a spatial dimension: the steady-state per-cell snapshot becomes a spatial PDE solve — shares almost nothing with the engine except `losses.py`. Highest new-build. | which reaction/diffusion coefficient a perturbation changed | Diffusion-driven-instability constraints give *some* structure, but attribution to a single coefficient from images is hard and bespoke. | **Thin.** Quantitative perturbation datasets exist (e.g. palate-rugae multi-morphogen inhibitor study) but are imaging-heavy, bespoke, not turnkey; hard to demonstrate a crisp single-parameter attribution at hackathon speed. | **4 (down-rank: architecture leap + weak turnkey data)** |

### Data-availability findings (with sources)

**gLV / microbiome — the recommended domain's data:**
- **MDSINE (Bucci et al. 2016, *Genome Biology*)** and **MDSINE2 (Gerber lab, 2021→2025)** —
  gLV inference engines with an explicit **external-perturbation (antibiotic) susceptibility**
  term. MDSINE2 ships "two new densely-sampled longitudinal datasets with intentionally
  introduced perturbations." Code + data: https://github.com/gerberlab/MDSINE2 and
  https://github.com/gerberlab/MDSINE2_Paper; MDSINE2 data on Zenodo:
  https://zenodo.org/records/5781848. MDSINE paper:
  https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-0980-6
- **Stein et al. 2013, *PLoS Comput Biol*** — gLV fit to the **Buffie clindamycin →
  C. difficile** mouse time-series (11 genus-level populations; antibiotic as a
  time-varying external perturbation). This is the canonical "which parameter did the
  antibiotic move" contrast. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003388
- **David et al. 2014** (daily gut time-series incl. a *Salmonella* infection interval on
  subject B) and **Caporaso "moving pictures" 2011** (dense longitudinal) — good for
  temporal richness, but perturbations are *incidental* (travel, infection), so the
  "which parameter" contrast is weaker than the designed MDSINE/Stein screens.
  https://pubmed.ncbi.nlm.nih.gov/21624126/
- **MTIST (2022)** — 648 *simulated* gLV time-series with a scoring system: a ready-made
  benchmark to sanity-check the round-trip against an external standard.
  https://www.biorxiv.org/content/10.1101/2022.10.18.512783
- **Lynx–hare (Hudson's Bay pelts, 1845–1935)** — the classic predator–prey series, but
  a **single trajectory with no perturbation contrast**: usable to validate the forward
  model, **not** to demonstrate attribution. CSV:
  https://github.com/stan-dev/example-models/blob/master/knitr/lotka-volterra/hudson-bay-lynx-hare.csv

**Amyloid — the runner-up's data:** AmyloFit (Meisl et al. 2016, *Nature Protocols*) is the
curated standard; α-synuclein ThT kinetics with microscopic-rate extraction appear in many
open papers (e.g. Buell/Galvagnion 2014 PNAS; isoform study 2024 PNAS). WT-vs-mutant-vs-
inhibitor curves are small and published with reference constants. Sources:
https://www.pnas.org/doi/10.1073/pnas.1315346111 , https://www.pnas.org/doi/10.1073/pnas.2313465121

**TMDD / Turing — down-ranked on data:** TMDD identifiability is well-studied
(https://link.springer.com/article/10.1007/s10928-012-9260-6) but open time-course data
is scarce (mostly proprietary/simulated). Turing perturbation data exists (palate rugae,
https://journals.biologists.com/dev/article/147/20/dev190553) but is imaging-heavy and
bespoke. **Neither offers turnkey public data that cleanly showcases single-parameter
attribution at hackathon speed.**

### The honest tension (stated plainly)

LV is the **easiest to write** and best matches "an obviously ecological perturbation,"
but its *real-data* attribution is precisely the ill-posed regime — NUDGE will likely
abstain a lot on real microbiome series (honest, on-thesis, but an undemoable positive).
Amyloid has a **slightly harder forward model but a cleaner, better-characterized
identifiability story and smaller, curated data that more reliably produces a crisp
positive**. The recommendation (§D) leads with LV for speed/field-fit/user-preference and
keeps amyloid as the fallback the data-gate can switch to if LV's real-data showcase is
too degenerate.

---

## (c) How hard is it? (calibrated to this team's demonstrated velocity)

Effort for the **gLV thin slice**, broken down. This team has repeatedly shipped
week-scoped capabilities in ~8h; each additive capability above (dose_response, lyapunov,
differential…) is a comparable unit of work. Estimate assumes the same discipline
(synthetic-first, decoy-guarded, one notebook).

| Component | Scope | Effort |
|---|---|---|
| **Forward-model vector field** | gLV field `dxᵢ/dt = xᵢ(αᵢ + Σβᵢⱼxⱼ + εᵢ·u(t))` in JAX + a `diffrax`/`lax.scan` trajectory integrator. ~30–60 lines. | **S** (~1–2h) |
| **Synthetic generator** (mandatory first) | Simulate N replicate communities with per-replicate parameter draws + a known single-parameter perturbation (Δα vs Δβ vs Δε), emit an AnnData-shaped object mirroring `generate_synthetic_perturbseq`. | **S–M** (~2h) |
| **Attribution vocabulary + identifiability** | Restricted fits {free α / free β / free ε}, BIC winner, abstain on the α⇄βᵢᵢ degeneracy and under-determined β. Reuse `losses.py` + copy the `model_select`/`dose_response` gate. Add a Fisher/Laplace check (reuse `uncertainty.py` method) to *measure* the degeneracy. | **M** (~2–3h) |
| **Abstention/decoy story** | ≥1 decoy: a community whose apparent β-change is really a growth-rate change routed through the coupling (must abstain, not mis-call); a "no-interaction" null. One Mechanism Card + `known_limitations.yaml` entry. | **M** (~2h) |
| **Wiring / demo** | One notebook: synthetic round-trip → attribute → abstain-when-unidentifiable → (stretch) one MDSINE real series. Optional `nudge lotka` CLI verb. | **M** (~2–3h) |

**Total ≈ one focused day for the synthetic round-trip + notebook**, at this team's pace;
+~half a day to touch the MDSINE real data. Amyloid is comparable, trading the trajectory
integrator for a moment-ODE/closed-form rate law and a curve fit (closer to
`dose_response.py`, which is already done — so arguably *less* new fit code).

### The single biggest technical risk

**LV: the observable/identifiability mismatch.** NUDGE's engine is shaped for
**steady-state snapshots**, but gLV's parameter information lives in **transients**, and
gLV inference is famously **ill-posed** (α vs βᵢᵢ degenerate at equilibrium; β
under-determined from short series). Two failure modes: (1) if the spike matches
steady-state compositions it will abstain on nearly everything (honest but undemoable);
(2) so the fit must match **trajectories** — a *new observable* the existing loop doesn't
provide, meaning the fit loop is **re-instantiated in the new module, not reused
verbatim** (still additive; still no `fit.py`/`core` edits). The thing that makes LV easy
to *write* (a clean autonomous ODE) is not the thing NUDGE's fit is *shaped for*. Mitigation:
prove the round-trip on **synthetic trajectory data with a known single-parameter
perturbation** first, and let the identifiability analysis *measure* the degeneracy so the
abstention is earned, not asserted.

(For amyloid the analogous risk is milder and better-understood: single-curve fits resolve
only rate *products*, so the spike must use concentration-series/seeding to separate the
microscopic steps — a known, documented requirement.)

---

## (d) Recommendation

**Yes — run it as a separate, parallel spike** while the main effort keeps solidifying the
Demo (the 30% criterion, currently weakest) and red-teaming the core engine. Reasoning:

- It is **fully isolated** (a new `inference/<domain>.py` + generator + card + decoy; zero
  edits to `fit.py`/`core`), so it **cannot regress** the demo or the fail-safe guarantees.
  It is the same additive shape shipped ~10 times already.
- It directly strengthens **Impact (25%)** — a demonstrated "same engine, new field of
  biology" is the extensibility thesis made concrete (microbiome ecology, neurodegeneration)
  — and **Claude Use / Depth (25%/20%)**: pointing the *attribution philosophy* at a new
  dynamical system is exactly the "surprising capability, wrestled-with engineering" the
  judges reward.
- It must **not** compete with the demo for the critical path: gate it so it only lands if
  the synthetic round-trip is clean and honest.

**Recommended domain:** generalized Lotka–Volterra (gLV) for microbial communities.
Rationale: easiest math, best field-fit for "an obviously ecological perturbation," the
user's front-runner, MDSINE gives a real attribution axis (growth vs interaction vs
antibiotic-susceptibility) + a Zenodo dataset, and a rich abstention story from the
α⇄βᵢᵢ degeneracy. **Documented fallback: amyloid aggregation kinetics** — switch to it if
the LV data-gate (below) shows the real-data attribution is too degenerate to showcase a
crisp positive; amyloid's forward model is a curve fit (near-identical to the already-shipped
`dose_response.py`) with a cleaner, published identifiability story.

**Proposed thin-slice scope (smallest thing that proves extensibility):**
1. `inference/lotka_volterra.py`: a JAX gLV trajectory field + a small optax fit loop
   (pattern-copied from `fit_parameters`, **not** importing it), reusing `losses.py`.
2. A synthetic gLV generator with a **known single-parameter perturbation** (Δα / Δβ / Δε).
3. Restricted-fit + BIC attribution of *which* parameter moved; **abstain** on the
   α⇄βᵢᵢ degeneracy and under-determined β, with a Fisher/Laplace measurement of it.
4. One decoy (growth-change masquerading as interaction-change → must abstain) + one
   Mechanism Card + a `known_limitations.yaml` entry.
5. **One notebook**: synthetic round-trip → attribute → abstain-when-unidentifiable, and
   *only if the gate passes*, one MDSINE/Stein real series as a coda.

**Go/no-go checkpoints (in order — do not skip the data gate before investing):**
- **Gate 0 (synthetic round-trip).** On synthetic gLV with a known single-parameter
  perturbation, the module recovers the right knob for clear cases and **abstains, never
  mis-calls**, on the degenerate ones. *If it mis-calls confidently → stop / redesign.*
- **Gate 1 (identifiability measured, not asserted).** The Fisher/Laplace analysis
  reproduces the α⇄βᵢᵢ degeneracy quantitatively, so the abstention is earned. *If the
  degeneracy can't be characterized → the abstention story is hollow; stop.*
- **Gate 2 — the data-gate (CONFIRM BEFORE touching real data).** Pull one MDSINE/Stein
  series and confirm it has a usable perturbation contrast at a signal-to-noise where
  attribution is *possible* (not guaranteed-abstain). *If real gLV inference is so
  ill-posed that NUDGE can only ever abstain → down-rank LV's real-data coda and either
  ship the spike as synthetic-only (still proves extensibility) or switch the real-data
  showcase to amyloid.*
- **Gate 3 (isolation audit).** Confirm the diff touches only new files + additive
  registrations; `fit.py` and `core/circuit.py` unchanged; decoy green in CI. *Any edit
  to `fit.py`/`core` → revert; the design is wrong.*

---

## What I could NOT verify (honesty)

- **Exact contents/schema of the MDSINE2 Zenodo record and the gerberlab data files** were
  not downloaded and inspected — I confirmed the repos/record *exist* and that MDSINE(2)
  models an explicit antibiotic-susceptibility parameter, but did not open the raw tables
  to confirm cell counts, sampling density, or license per file. The Gate-2 data-gate
  exists precisely to check this before investing.
- **Whether real MDSINE/Stein data will yield a *positive* (non-abstaining) single-parameter
  attribution** is genuinely unknown and, per the ill-posedness literature, plausibly *no*.
  That uncertainty is the reason the recommendation leads with a **synthetic** round-trip
  and treats the real-data coda as gated/optional.
- **Amyloid forward-model exact form** (which AmyloFit model class best fits a chosen α-syn
  dataset) was not pinned down; only that the closed-form/moment-ODE machinery and curated
  data exist.
