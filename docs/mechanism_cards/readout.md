---
id: NUDGE-MECH-020
name: readout
role: readout
registry_name: Readout
vulnerable_to_decoys: [NUDGE-DECOY-003]
documented_limitation: [NUDGE-LIM-003, NUDGE-LIM-006]
validated_in_regime: {min_cells_per_condition: 1000, notes: "The affine readout is the identifiability boundary: a strongly nonlinear reporter is indistinguishable from a circuit switch from one population (LIM-006 profile flat 0.0003); a constitutive-reporter control breaks it (n=1 rejected, 0.017). FINDINGS NUDGE-LIM-006 mitigation."}
references: [Svensson2020, HuangFerrell1996]
---

# Mechanism Card — Readout

> **ID:** `NUDGE-MECH-020`  ·  **Role:** readout
> **Stability:** stable  ·  **Registry name:** `Readout`

## Summary

The biology→measurement boundary: an **affine, non-negative** reporter map from
latent species activity to per-gene expression rate `Λ ≥ 0`. Keeping the readout an
explicit, separate layer is what keeps the circuit's mechanism parameters
identifiable — but its *affineness* is also the sharpest bound on the fail-safe
guarantee.

## Governing equation

```
Λ = max(base + activity · Wᵀ, 0)
```

- `activity` — latent species activity, shape `(n_cells, n_species)`.
- `W` (`weight`) — reporter loadings, shape `(n_genes, n_species)`.
- `base` — per-gene intercept, shape `(n_genes,)`.
- The `max(·, 0)` clamp keeps rates non-negative. `Λ` is a **rate**, not counts — the
  negative-binomial count model (`nudge.data.noise`) turns `Λ` into raw UMI counts.

## What it represents

The reporter/observation channel: how latent circuit activity is read out as
expression of measured genes. `Readout.identity` gives a 1-gene-per-species reporter
`Λ_g = base + scale·activity_g`. Separating this from the circuit is what lets NUDGE
own the count model and consume raw counts.

## Assumptions & simplifications

- Expression rate is **affine** in activity (clamped non-negative). A genuinely
  nonlinear (saturating/sigmoidal) reporter is *not* represented.
- No explicit capture/dropout process is modelled inside the readout — dropout is left
  to the NB count model (UMI counts are not zero-inflated; [@Svensson2020]).
- Reporter genes are assumed present and un-normalised (raw counts in).

## Known failure modes

| Failure mode | Decoy that exercises it | Limitation |
|---|---|---|
| A strongly nonlinear reporter's ultrasensitivity misattributed to a circuit switch | `nudge.data.decoy_generators.generate_readout_nonlinearity_decoy` (xfail witness) | `NUDGE-LIM-006` |
| Technical dropout / depth variation producing a switch-like zero-peak | `NUDGE-DECOY-003` | `NUDGE-LIM-003` |

`NUDGE-LIM-006` is the major bound: since only `readout ∘ circuit` is observed, an
affine readout cannot tell circuit ultrasensitivity from measurement
ultrasensitivity. It is **not** a passing decoy — under a strongly nonlinear reporter
NUDGE can be a *confident false positive* rather than abstain, so it is captured by an
**xfail witness** (`generate_readout_nonlinearity_decoy`), not a battery case NUDGE
passes. `NUDGE-LIM-003` is handled by the parsimony gate at the count level (the
affine readout emits rates; the NB model absorbs depth-driven zeros).

## Identifiability regime

FINDINGS "NUDGE-LIM-006 mitigation" (a standalone identifiability study, independently
reproduced): from a **single population** the circuit/readout split is **genuinely
degenerate** — the loss profile over circuit Hill `n` is **flat (span 0.0003 across
n ∈ [1,10])**; a graded `n = 1` circuit fits within 0.0001 of the true `n = 3`;
`corr(circuit n, readout h) = −0.905`. **Adding a constitutive-reporter control**
(known activity doses that bypass the circuit) breaks it: `n = 1` is **rejected**
(Δloss 0.017 ≫ floor), the near-optimal multistart fraction goes 0.07 → 1.00, and the
data conclude the ultrasensitivity is biological (though it does not point-identify
`n` exactly). Absent such a control, NUDGE should **abstain on the circuit-vs-readout
axis**.

## Implementation Mapping

| Equation term | Code |
|---|---|
| affine map `max(base + A·Wᵀ, 0)` | `nudge.mechanisms.readout.Readout.expression` |
| identity reporter constructor | `nudge.mechanisms.readout.Readout.identity` |

*(Machine-checked by `scripts/check_impl_mapping.py`.)*

## Verification evidence

- `tests/mechanisms/test_readout.py::test_affine_map` — the reporter map is affine.
- `tests/mechanisms/test_readout.py::test_non_negative_clamp` — rates are clamped non-negative.
- `tests/verification/test_readout_nonlinearity_limitation.py::test_nonlinear_readout_should_be_declined_but_is_not`
  — the LIM-006 xfail witness: asserts the *desired* abstention under a nonlinear
  reporter (currently xfails, so a future fix flips it and forces a docs update).

## References

- [@Svensson2020] — droplet scRNA-seq UMI counts are not zero-inflated (why dropout lives in the NB model, not the readout).
- [@HuangFerrell1996] — ultrasensitivity, here as a *measurement* nonlinearity confounder.
