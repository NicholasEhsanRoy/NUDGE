---
id: NUDGE-METHOD-006
name: bifurcation_proximity
role: attribution-method
registry_name: BifurcationProximity
vulnerable_to_decoys: []
documented_limitation: [NUDGE-LIM-012]
validated_in_regime: {min_stable_modes: 2, notes: "Scores how close a BISTABLE switch is to a saddle-node fold as a 0..1 robustness DIAL from three channels — critical slowing (min|Re lambda| of the drift Jacobian at each stable mode -> 0), basin collapse (stable-node -> index-1-saddle distance -> 0), and LNA lobe swell (sqrt(lambda_max(Sigma)) / inter-mode separation -> 1). The load-bearing honesty point (NUDGE-LIM-012): the linear-noise Gaussian breaks down PRECISELY at the fold (variance diverges), so the dial is a ONE-SIDED LOWER BOUND near the fold, never a point estimate, and NUDGE ABSTAINS (unresolved) on the deep-basin far side rather than emit a false-precise 'far' number. Validated on the self-activation switch's KNOWN analytic saddle-node in cooperativity n and in threshold K: sweeping toward the fold, all three channels move MONOTONICALLY and the fused dial ranks proximity correctly, with one_sided setting near the fold (FINDINGS 'Phase 4f'). A real-data dose-ladder lock-in is a deferred follow-up; the synthetic sweep is the load-bearing validation."}
references: [Scheffer2009, Strogatz2015, ElfEhrenberg2003, FerrellMachleder1998]
---

# Mechanism Card — Bifurcation / tipping-point proximity (the robustness dial)

> **ID:** `NUDGE-METHOD-006`  ·  **Role:** attribution-method
> **Stability:** experimental  ·  **Registry name:** `BifurcationProximity`

## Summary

NUDGE's other capabilities ask *which knob* a perturbation moves (threshold / gain /
ceiling). This one answers a different, high-value question: **how close is a bistable
switch to *losing* bistability** — a saddle-node / fold? The answer is a scalar
**robustness dial** — a hair-trigger cliff vs a well-buffered switch — from **three
complementary channels**, each with a known analytic limit at the fold, fused into a
normalized **0..1 proximity**. Audience: resilience / critical-transition biology (aging,
disease progression, cell-fate commitment) and engineered-circuit robustness QA. It is
also the hard dependency for the future `design()` capability's **safety gate**: an
intervention that pushes a switch toward a tipping point must be flagged, which needs a
real, honestly-bounded proximity score.

The proximity signal is *already computed but buried* inside the circuit engine — no
public accessor returned it. This capability lifts it out **properly** (all three
channels, rigorous one-sided UQ), reusing the existing fixed-point / saddle / LNA
machinery.

## Why this exists (a number where there was none)

`Circuit.fixed_points()` returns the stable nodes + the index-1 saddle with Jacobian
labels, but **discarded the eigenvalues**; `lna_reliable` computed a lobe-swell ratio
only as an abstention trigger. Neither exposed a proximity **number**. Yet the question
"is this switch near a tipping point?" is exactly what a `design()` safety gate must
answer, and it is the cheapest capability to build (it re-exposes existing internals). So
this module is a full, honestly-bounded capability — not a minimal inline re-expose — so
`design()` inherits a solid dependency.

## Governing equation

Three channels of one bistable circuit's steady-state geometry, each → its fold limit:

```
critical slowing :  p_slow  = 1 − min|Re λ(A_k)| / min(decay)        (min|Re λ| → 0)
basin collapse   :  p_basin = 1 − ‖node − saddle‖ / (½·‖μ_i − μ_j‖)  (node→saddle → 0)
lobe swell       :  p_lobe  = clip( √λ_max(Σ_k) / min‖μ_i − μ_j‖ − 1, 0, 1)   (ratio → 1)
```

Here `A_k = ∂/∂x[production(x) − decay·x]` is the drift Jacobian at stable mode `k`
(critical slowing down — the slowest relaxation mode stalls as the fold nears; Scheffer
2009); `Σ_k` is the linear-noise (Lyapunov) covariance `A Σ + Σ Aᵀ + D = 0` at that mode;
`μ_i` are the stable-mode means and `saddle` the index-1 saddle
(`Circuit.transition_state`). The fused dial is

```
proximity = max( ½·(p_slow + p_basin) , p_lobe )       ∈ [0, 1]
```

— the two **deterministic, depth-independent** channels averaged, then `max`'d with the
LNA overlap so the noise channel can only *raise* the alarm, never lower it (fail-safe).

## The classifier (fail-safe, in order)

`classify_robustness` returns one of four calls:

1. **not-bistable** — `score is None` (< 2 stable modes): no switch to be near a fold.
2. **near-fold** — `proximity ≥ 0.55`: close to the fold. Because the LNA Gaussian is
   breaking down here (`one_sided` is set once the lobes overlap), the number is a
   **ONE-SIDED LOWER BOUND** — "at least this close" — never a point estimate.
3. **unresolved** — `proximity < 0.05`: the deep-basin far side, where the slowest
   relaxation rate has saturated at the intrinsic decay rate and the noise lobes carry
   *no* fold information. NUDGE **abstains** rather than emit a false-precise "far"
   number.
4. **robust** — `0.05 ≤ proximity < 0.55`: a well-buffered switch, comfortably away from
   the fold, with the proximity signal still responsive (a trustworthy call).

## Assumptions & simplifications

- **The LNA Gaussian breaks down *precisely at the fold*** — a mode's variance diverges
  as its Jacobian eigenvalue → 0 — so the lobe channel is *least* trustworthy exactly
  where it matters most. This is why the dial is a **one-sided lower bound** near the fold
  and why NUDGE abstains on the deep-basin side rather than emit a precise "far" number
  (`NUDGE-LIM-012`). The two deterministic channels (critical slowing, basin collapse) are
  depth-independent and stand; only the lobe channel needs the reliability caveat.
- **A well-conditioned Jacobian near the fold needs float64.** The eigenvalue and Lyapunov
  computations run under a local x64 context (an eigenvalue → 0 cancels catastrophically at
  float32), matching `Circuit._lna_covariance`.
- **Depth affects only the lobe channel.** `attribute_bifurcation` calibrates the
  sequencing depth `scale` from data and reports `lna_reliable`'s verdict as a
  lobe-channel caveat — a low-depth Gaussian relaxation of the counts is untrustworthy,
  but the proximity is a property of the *circuit*, not the depth.
- **The score is a property of the fitted circuit.** From data, `attribute_bifurcation`
  can fit free kinetics (the shipped LNA mixture fit) before scoring; the honesty bound on
  a *fit* (gain⇄threshold degeneracy, `NUDGE-LIM-006`/`007`) is inherited from that fit,
  not added here.
- **Any bistable topology.** The dial is not restricted to 1-D — the deterministic
  channels and the LNA lobes generalise to N-species switches (toggle, 2-node) via the
  same finder — because it is meant to be a general `design()` safety gate. Attribution of
  *which knob* remains 1-D-guarded elsewhere; this capability only measures *proximity*.

## Known failure modes

| Failure mode | Guard / witness | Limitation |
|---|---|---|
| A point estimate emitted where the LNA is diverging (at the fold) | the one-sided lower bound + `one_sided` flag; `test_near_fold_is_a_one_sided_lower_bound` | `NUDGE-LIM-012` |
| A false-precise "far" number on the deep-basin side | the deep-basin abstention (`unresolved`); `test_deep_basin_abstains_with_no_precise_far_number` | `NUDGE-LIM-012` |
| A monostable circuit forced to report a proximity | the `< 2 stable modes → None → not-bistable` gate; `test_monostable_is_not_bistable` | `NUDGE-LIM-012` |
| The lobe channel trusted at low sequencing depth | `attribute_bifurcation` reports `lna_reliable`'s depth verdict as a caveat | `NUDGE-LIM-012` |

There is **no dedicated bifurcation decoy battery yet** (`vulnerable_to_decoys: []`) — the
failure modes above are guarded by the one-sided framing, the deep-basin abstention, the
not-bistable gate, and the monotonic parameter-sweep ground truth; a synthetic decoy
(e.g. noise-induced bimodality that is *not* near a deterministic fold) is future work.

## Identifiability regime

- **≥ 2 stable modes** (a genuine bistable switch); else the score is `None` and the call
  is `not-bistable`.
- **Verified on synthetic ground truth (the load-bearing validation).** The
  self-activation switch has a **known analytic saddle-node** in its cooperativity `n`
  (and in its threshold `K`). Sweeping toward the fold, all three channels move
  **monotonically** — `min|Re λ|` 0.92 → 0.30 → 0, node→saddle 0.93 → 0.62 → 0, lobe ratio
  0.75 → 1.66 → past 1 — and the fused 0..1 dial **ranks proximity correctly**
  (0.073 → 0.66), with `one_sided` setting near the fold and the switch going `not-bistable`
  past it (`n = 1`, `K ≥ 1.33`). The four verdicts partition the sweep: `n ≥ 8`
  (deep-basin) → **unresolved**, `n ∈ {4, 6}` → **robust**, `n ≤ 2` → **near-fold**,
  `n = 1` → **not-bistable** (FINDINGS "Phase 4f").
- **Real-data lock-in is a deferred follow-up.** A small **bistable dose ladder
  approaching a fold** (toggle + hysteresis, Zenodo 11817798; or a morphogen top rung,
  GSE233574) would show the dial rising along the ladder, grounded in critical-slowing-down
  theory (rising variance / autocorrelation; Scheffer 2009). The synthetic parameter sweep
  is the load-bearing validation; the real-data test is marked `needs_data` and skips if
  the processed dataset is absent.

## Implementation Mapping

| Step | Code |
|---|---|
| critical-slowing Jacobian eigenvalues at each stable mode | `nudge.inference.bifurcation._stable_eig_reals` |
| the buried LNA lobe covariances (reused) | `nudge.core.circuit.Circuit.mode_covariances` |
| the index-1 saddle for the basin-collapse channel (reused) | `nudge.core.circuit.Circuit.transition_state` |
| fuse the three channels into the 0..1 dial | `nudge.inference.bifurcation.bifurcation_proximity` |
| the honest four-way verdict (one-sided near-fold; deep-basin abstention) | `nudge.inference.bifurcation.classify_robustness` |
| score from data (fit optional; depth-calibrated LNA caveat) | `nudge.inference.bifurcation.attribute_bifurcation` |
| CLI / MCP orchestration | `nudge.service.robustness_circuit` |

*(This table is machine-checked by `scripts/check_impl_mapping.py`: every `nudge.*`
reference must resolve to a real attribute.)*

## Verification evidence

- `tests/inference/test_bifurcation.py::test_channels_move_monotonically_toward_the_known_fold`
  — the load-bearing ground truth: all three channels move monotonically toward the known
  saddle-node and the dial ranks proximity.
- `tests/inference/test_bifurcation.py::test_near_fold_is_a_one_sided_lower_bound` — near
  the fold the call is `near-fold`, `one_sided` is set, and the reason states the
  lower-bound caveat (`NUDGE-LIM-012`).
- `tests/inference/test_bifurcation.py::test_deep_basin_abstains_with_no_precise_far_number`
  — the deep-basin side abstains (`unresolved`) with no precise "far" number.
- `tests/inference/test_bifurcation.py::test_well_buffered_switch_is_robust` /
  `::test_monostable_is_not_bistable` — the `robust` and `not-bistable` verdicts.
- `tests/inference/test_bifurcation.py::test_metadata_channels_are_populated_and_json_serialisable`
  — the raw per-mode channels are populated and JSON-serialisable (for the demo).
- `tests/test_service.py::test_robustness_circuit_wiring` — the CLI/MCP service round-trip.

## References

- [@Scheffer2009] — early-warning signals for critical transitions: the critical-slowing-
  down theory (rising variance / autocorrelation / recovery time) that grounds the min|Re
  λ| and LNA-lobe channels as a switch nears a fold.
- [@Strogatz2015] — *Nonlinear Dynamics and Chaos*: the saddle-node / fold bifurcation and
  the Jacobian-eigenvalue-crossing that the min|Re λ| channel measures.
- [@ElfEhrenberg2003] — the linear-noise approximation `A Σ + Σ Aᵀ + D = 0` whose lobe
  covariance is the tertiary channel (and whose divergence at the fold is the honesty
  crux).
- [@FerrellMachleder1998] — the ultrasensitive bistable switch (Xenopus MAPK) as the
  archetype of the cell-fate switch whose robustness this dial measures.
