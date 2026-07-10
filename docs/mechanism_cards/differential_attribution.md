---
id: NUDGE-METHOD-010
name: differential_attribution
role: attribution-method
registry_name: DifferentialAttribution
vulnerable_to_decoys: []
documented_limitation: [NUDGE-LIM-006, NUDGE-LIM-016]
validated_in_regime: {min_cells_per_context: 300, requires: "a per-context control + an approximately-correct shared switch topology", notes: "Fits the SAME perturbation in TWO contexts (drug-resistant vs sensitive line; donor A vs B; disease vs healthy) JOINTLY with a shared-vs-per-context parameter structure and BIC-selects which SINGLE knob differs — threshold (K) / gain (n) / ceiling (v_max) — or abstains (no-difference / unresolved). Reuses the shipped LNA Gaussian-mixture forward model (mode means + Lyapunov covariances) and the BIC parsimony pattern; depth/noise is pinned PER CONTEXT from each context's OWN control (calibrate_from_wt). Confound guard (NUDGE-LIM-016): a depth/batch shift aligned with the context axis is degenerate with a ceiling difference (scale⇄vmax), so a ceiling call whose OFF baseline moved vs its control, or whose per-context depths differ, ABSTAINS. Validated on synthetic ground truth: a Δv_max / Δn pair recovers WHICH knob differs, a no-difference pair reads no-difference, a ΔK threshold pair recovers-or-abstains (threshold is the hardest from a bistable snapshot, FINDINGS §2), and a confounded (depth-aligned-with-context, no real mechanism) pair abstains unresolved — never a spurious mechanism-difference call, 0 confident-wrong across seeds. Reported Δ estimates are APPARENT population parameters, not molecular constants (NUDGE-LIM-006)."}
references: [Das2009, HuangFerrell1996, ElfEhrenberg2003]
---

# Mechanism Card — Comparative / differential attribution

> **ID:** `NUDGE-METHOD-010`  ·  **Role:** attribution-method
> **Stability:** experimental  ·  **Registry name:** `DifferentialAttribution`

## Summary

Given the **same perturbation** run in **two contexts** — a drug-resistant vs sensitive
cell line, donor A vs B, disease vs healthy — isolate whether the mechanistic difference
is in the switch's **threshold** (`K`), **gain** (`n`), or **ceiling** (`v_max`). A
resistant line that raised its *ceiling* needs **more dose of the same drug**; one that
rewired its *gain / threshold* needs a **different drug class**. Linear differential
expression structurally cannot make that call — it sees only "the level moved". NUDGE fits
the two contexts **jointly** with a **shared-vs-per-context** parameter structure and
**BIC-selects which single knob must differ**, or abstains.

## Why this exists (the distinction DE cannot make)

Two contexts can present the *same* fold-change in a signature yet differ in the underlying
switch in mechanistically opposite ways: a shifted **threshold** (the dose at which the
switch flips), a changed **gain** (how sharply it flips), or a raised/lowered **ceiling**
(the ON amplitude). These prescribe different interventions. NUDGE already speaks this
`K` / `n` / `v_max` vocabulary for a single condition; differential attribution asks the
comparative question — *which of these three differs between the contexts?* — as a
model-selection problem, the difference being the **target**, not a nuisance.

## Governing equation

Each context `x ∈ {A, B}` is a bistable switch read as a linear-noise (LNA) Gaussian
mixture: the stable fixed points `μ_k(θ_x)` of the shared circuit are the mode means and
the Lyapunov covariances `A Σ + Σ Aᵀ + D = 0` are the mode spreads (the same forward model
as `nudge.inference.lyapunov`). A candidate **model** shares all three target-edge knobs
between the contexts except (at most) one, which takes a **per-context** value:

```
shared      : K, n, v_max shared            (the no-difference null;  k = 3 kinetic)
ΔK-only     : K per-context; n, v_max shared (threshold-diff;         k = 4 kinetic)
Δn-only     : n per-context; K, v_max shared (gain-diff;              k = 4 kinetic)
Δv_max-only : v_max per-context; K, n shared (ceiling-diff;           k = 4 kinetic)
```

Each model is fit by maximizing the summed LNA mixture log-likelihood across both
contexts (`scale` / `obs_sd` pinned per context), and scored by

```
BIC = k · ln N − 2 · log L        (N = total cells; lower is more parsimonious)
```

The min-BIC model names the difference — a Δ model must **earn** its extra per-context
parameter over the shared null. Each per-context knob is bounded (a smooth `tanh` reparam
around nominal) so it cannot run off to the LNA variance-collapse likelihood spike.

## The classifier (fail-safe, in order)

1. **unresolved — underpowered / untrustworthy (confound guard, part 1).** Either context
   is below `min_cells`, or its LNA is unreliable (`lna_reliable` — low depth / near a
   bifurcation / monostable): depth cannot be pinned cleanly enough to separate it from a
   mechanism difference.
2. **no-difference.** No Δ model beats the shared null by `bic_margin` — the extra
   per-context parameter is not earned; one `K` / `n` / `v_max` explains both contexts.
3. **unresolved — the Δ models tie.** The winning Δ model does not beat the runner-up Δ
   model by `resolve_margin` (the measured gain⇄threshold confound): the difference is real
   but *which* knob moved is unidentifiable — abstain, don't guess.
4. **unresolved — the depth confound (`NUDGE-LIM-016`, the load-bearing guard).** The two
   contexts' per-context depths (pinned from their controls) differ beyond
   `depth_ratio_max` — a depth/batch difference *aligned with the context axis*. Depth
   (global scale) and `v_max` are degenerate, so NUDGE cannot certify that an apparent
   ceiling / no-clear difference is not a masked depth artifact and **abstains** — *unless*
   the winner is a cleanly-resolved **threshold** or **gain** difference, which reshapes
   the distribution (orthogonal to a global scale) and survives.
5. **threshold-diff / gain-diff / ceiling-diff.** The winning Δ model earns its parameter
   over the shared null and beats the other Δ models (and the ceiling channel is only
   called when the per-context depths match).

## Assumptions & simplifications

- **A per-context control is required, from the same library as its perturbed cells
  (`NUDGE-LIM-016`).** Depth/noise is pinned PER CONTEXT from each context's OWN control
  (`calibrate_from_wt`); a depth/batch difference aligned with the context axis is otherwise
  indistinguishable from a **ceiling** difference (the `scale ⇄ v_max` degeneracy). When
  the per-context depths differ beyond a ratio, NUDGE **abstains** rather than risk a
  spurious ceiling-diff. Threshold / gain differences *reshape* the distribution and are
  orthogonal to a global scale, so they survive a depth difference; the **ceiling** channel
  does not.
- **The two contexts share an approximately-correct switch topology.** The difference is
  localized to one edge's `K` / `n` / `v_max`; a wrong topology voids the guarantee (the
  Tier-0.5 T0.5-2 boundary), as for all NUDGE attribution.
- **The LNA Gaussian is local** — it degrades near a bifurcation and at low copy number,
  exactly where a large perturbation pushes the switch; `lna_reliable` abstains there.
- Reported Δ estimates (with bootstrap CIs) are **apparent population** parameters, not
  molecular constants, and hold under an approximately-affine readout (`NUDGE-LIM-006`).

## Known failure modes

| Failure mode | Guard / witness | Limitation |
|---|---|---|
| A depth/batch shift aligned with the context axis faking a ceiling difference | per-context depth pinning + the depth-ratio abstention → `unresolved` (`tests/inference/test_differential.py::test_confound_depth_aligned_with_context_abstains`) | `NUDGE-LIM-016` |
| Gain vs threshold not separable from a bistable snapshot | the Δ-model tie gate → `unresolved` (`test_fail_safe_never_confident_wrong`) | `NUDGE-LIM-016` |
| An underpowered / near-bifurcation context | the `lna_reliable` + `min_cells` gate → `unresolved` (`test_underpowered_context_abstains`) | `NUDGE-LIM-016` |
| A nonlinear readout faking ultrasensitivity | the affine-readout bound shared with all attribution | `NUDGE-LIM-006` |

There is **no dedicated differential decoy battery yet** (`vulnerable_to_decoys: []`) — the
confounded-abstention test *is* the batch-aligned-with-context decoy at the context level;
a broader synthetic decoy battery is future work.

## Identifiability regime

- **≥ ~300 cells per context** (the fail-safe floor; ~1000+ for a confident positive),
  each context bistable and `lna_reliable`, with **a per-context control** to pin depth.
- **Validated on synthetic ground truth (the load-bearing result, `FINDINGS` Phase 4j).**
  A KNOWN single-knob difference between two contexts on the 1-node self-activation switch:
  a **Δv_max** pair recovers `ceiling-diff` and a **Δn** pair recovers `gain-diff`; a
  **no-difference** pair reads `no-difference`; a **ΔK** pair recovers-or-abstains
  (threshold is the hardest from a bistable snapshot — its stable modes barely move with
  `K` — consistent with the measured hierarchy **gain > ceiling > threshold**, `FINDINGS`
  §2); and a **confounded** pair (a depth/batch shift aligned with the context axis but NO
  real mechanism difference) **abstains `unresolved`** — never a spurious
  mechanism-difference call, **0 confident-wrong across seeds**. A real-data touch (e.g.
  sci-Plex A549 vs MCF7, or Gladstone donors) is a deferred best-effort follow-up; the
  synthetic ground truth is the load-bearing validation.

## Implementation Mapping

| Step | Code |
|---|---|
| the two-context input (data + each context's own control) | `nudge.inference.differential.Context` |
| per-context depth/noise pinning (the confound guard) | `nudge.inference.lyapunov.calibrate_from_wt` |
| the shared LNA forward model reused verbatim | `nudge.inference.lyapunov.OperatingPoint` |
| fit the nested models + BIC (shared / ΔK / Δn / Δv_max) | `nudge.inference.differential.fit_differential` |
| the fail-safe classifier (+ the per-context depth-ratio confound guard) | `nudge.inference.differential.classify_differential` |
| attribute a two-context differential in one call | `nudge.inference.differential.attribute_differential` |
| synthetic ground-truth context-pair generator | `nudge.inference.differential.simulate_context_pair` |
| CLI / MCP orchestration | `nudge.service.differential_file` |

*(This table is machine-checked by `scripts/check_impl_mapping.py`: every `nudge.*`
reference must resolve to a real attribute.)*

## Verification evidence

- `tests/inference/test_differential.py::test_recovers_ceiling_difference` — a Δv_max pair
  recovers `ceiling-diff`.
- `tests/inference/test_differential.py::test_recovers_gain_difference` — a Δn pair
  recovers `gain-diff`.
- `tests/inference/test_differential.py::test_no_difference_reads_no_difference` — an
  identical pair reads `no-difference`.
- `tests/inference/test_differential.py::test_threshold_recovers_or_abstains` — a ΔK pair
  recovers `threshold-diff` or safely abstains (never gain/ceiling-diff).
- `tests/inference/test_differential.py::test_confound_depth_aligned_with_context_abstains`
  — the confounded case (batch aligned with the context axis, no real mechanism) abstains
  `unresolved` (`NUDGE-LIM-016`).
- `tests/inference/test_differential.py::test_fail_safe_never_confident_wrong` — 0
  confident-wrong mechanism-difference calls across a mechanism / seed sweep.
- `tests/test_service.py::test_differential_file_npz_wiring` — the `.npz` service
  round-trip the CLI / MCP share.

## References

- [@Das2009] — the bistable Ras activation switch whose threshold-vs-gain-vs-ceiling
  differences this resolves between contexts.
- [@HuangFerrell1996] — ultrasensitivity / the `K` / `n` / `v_max` response-magnitude
  vocabulary shared with the single-cell, dose-response, and cross-modality paths.
- [@ElfEhrenberg2003] — the linear-noise / Fisher-information framing behind the
  covariance-structured forward model the joint fit uses.
