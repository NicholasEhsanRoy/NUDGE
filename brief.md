# NUDGE

**N**ode/edge **U**ltrasensitivity **D**iagnostic for **G**ene-regulatory **E**ffects

Built with Claude: Life Sciences — Build track, July 2026

---

## One-line pitch

Does MADDENING — a differentiable graph-based physics engine built for magnetically-actuated microrobots — generalize to a domain with no physical analog at all? NUDGE tests this by fitting a mechanistic, compositional circuit model to real Perturb-seq data, distinguishing perturbations that shift a switch's *threshold* from ones that change its *gain* — a distinction the field's default linear models cannot make — and then inverting the fit to propose untested interventions.

## Why this, why now

Standard Perturb-seq analysis — including the analysis published on the Gladstone T cell dataset itself — reduces every perturbation's effect to a linear coefficient. The dataset's own authors flag dynamic, nonlinear models of gene-regulatory behavior as work they haven't done. T cell activation is a good test case because it's a documented switch, not a dial: Ras/ERK activation downstream of TCR signaling is bistable and hysteretic, driven by SOS's positive feedback loop (Das et al., *Cell* 2009), not graded.

**Falsifiable prediction:** SOS knockdown should collapse the digital activation signature toward graded. RasGRP1 knockdown shouldn't (it drives the analog/graded arm, no feedback).

**Sanity check before building:** pull `polarization_prediction_condition_comparison_regulator_coefficients.csv` from the dataset's public GitHub repo and eyeball where SOS1, SOS2, RASGRP1, RASA1, and the DUSPs rank as activation regulators under the existing linear model, before writing any dynamical-systems code.

## Who this is for

A computational/systems biologist who already suspects a pathway they're screening is switch-like, but only has linear DE/regression tools to check that suspicion — not a general Perturb-seq hit-caller. The deliverable is a reusable tool this person could point at their own dataset and circuit hypothesis, not a one-off analysis of the T cell data.

## Related work — read and cite, don't get blindsided by

- **CellBox** (Yuan et al., *Cell Systems* 2021, doi:10.1016/j.cels.2020.11.013) — closest prior art. Differentiable ODE network fit via autodiff to real perturbation data (melanoma RPPA), later reimplemented with adjoint sensitivity. **Key difference to lean on:** CellBox uses *one global nonlinearity function* shared across the whole network; NUDGE composes *different mechanism types per edge and per node*. CellBox's deliverable is predictive accuracy (leave-one-drug-out validation); NUDGE's is mechanism attribution.
- **RACIPE / sRACIPE** — randomizes kinetic parameters over a *given* topology to explore the ensemble of possible dynamics. Forward/generative, not a fit to a specific real dataset. Different category, still worth a citation.
- **GEARS, CPA** — black-box nonlinear predictors of perturbation response. No mechanistic attribution; NUDGE answers a question they structurally can't.
- **Das et al., Cell 2009** — establishes the Ras/SOS/RasGRP1 bistable switch mechanism this whole project leans on.
- **Huang & Ferrell, PNAS 1996** — MAPK cascade ultrasensitivity via zero-order kinetics, *independent of any cooperative binding*. Justifies treating ERK's own integrator as a separate source of nonlinearity from SOS's feedback edge.

## Architecture

### Stage 1 — Fit
Build one differentiable object (not a classifier and a separate optimizer) that maps perturbation → outcome. Both the linear baseline and the mechanistic circuit are fit through the *same* graph machinery — the linear model is just the same topology with every edge swapped to `LinearEffect`.

### Stage 2 — Design
Invert the fitted, differentiable circuit via gradient descent: given a target outcome, optimize over perturbation-space to propose interventions — including gene combinations never in the original screen.

### Node/edge library (small and generic — specificity lives in configuration, not code)

| Component | Type | Options |
|---|---|---|
| `Species` | node | generic continuous activity state — one class for RasGRP1, SOS, Ras-GTP, ERK, etc. |
| `RegulatoryEffect` | edge | `LinearEffect` (baseline), `HillActivationEffect`, `HillRepressionEffect`. No separate "feedback" type — feedback is just an edge that closes a cycle. |
| `IntegratorModel` | node | `LinearIntegrator` (simple relaxation — SOS, RasGRP1), `SaturatingIntegrator` (MM/Hill on production), `ZeroOrderIntegrator` (Goldbeter–Koshland implicit steady-state — natural fit for ERK; solved via root-find, differentiated via IFT-through-converged-solve, same trick as the physics adjoints) |
| `Perturbation` | modifier | continuous strength on a node or edge, fit from CRISPRi dose — not a binary flag |
| `Readout` | node | smooth surrogate mapping internal state to the observed single-cell signature, calibrated against real discrete outcome |
| `GenePrior` | node (stretch) | maps gene identity → which parameter it plausibly moves, via PPI-network proximity (Krogan networks) — lets Stage 2 propose untested genes |

### Reference circuit (T cell activation test case)

```
RasGRP1 ──(Hill/linear, no feedback)──▶ Ras-GTP ──▶ ERK ──▶ Readout
                                          ▲  │
                              (feedback) │  │
SOS ◀─────────────────────────────────────┘  (cooperative Hill)
```
Perturbation taps: RasGRP1 knockdown, SOS knockdown (both reduce the node's own production rate).

## Numerical notes

- Write the circuit solve as pure JAX, differentiated via `diffrax` (adjoint ODE integration) or `optimistix` (IFT through root-finding for the zero-order integrator) — **not** naive backprop through an unrolled trajectory. Near a bifurcation this is exactly as unstable as backprop through a near-singular solve in the physics work — decide this on day one, don't retrofit later.
- Cheap uncertainty on Stage 2 proposals: Laplace approximation (Hessian of the fit loss at the optimum) — nearly free since everything is already differentiable.

## Scope for the week

**Core (must ship):**
- `Species`, `LinearEffect`, `HillActivationEffect`, `LinearIntegrator`, `SaturatingIntegrator`
- `Perturbation`, `Readout`, `LinearBaseline` comparison
- Fit pipeline on the T cell circuit; test the SOS/RasGRP1 falsifiable prediction
- Basic package interface: `fit(adata, circuit) → classification table`

**Stretch:**
- `ZeroOrderIntegrator` for ERK
- `GenePrior` + Stage 2 gradient-based design/inversion
- Laplace uncertainty on proposals
- Dashboard/demo layer on top of the library

## Assets

- MADDENING: https://github.com/Microrobotics-Simulation-Framework/MADDENING
- MIME docs: https://microrobotica.org/mime
- Gladstone dataset: https://virtualcellmodels.cziscience.com/dataset/genome-scale-tcell-perturb-seq
- Dataset paper (bioRxiv): https://www.biorxiv.org/content/10.64898/2025.12.23.696273   
