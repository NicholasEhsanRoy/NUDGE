---
id: NUDGE-MOTIF-003
name: toggle
role: motif
registry_name: null
vulnerable_to_decoys: [NUDGE-DECOY-004]
documented_limitation: [NUDGE-LIM-004, NUDGE-LIM-006]
validated_in_regime: {min_cells_per_condition: null, notes: "N-D saddle finder reproduces the toggle saddle exactly (symmetric [1.017,1.017]). The 1-D gain gate does NOT extend (measured NO-GO: free-n w_trans 0.00/0.25/0.22 << tau=0.5), so NUDGE abstains. Single-snapshot FIM: gain<->threshold confound corr -0.986, cond# 210; a second operating point breaks it (cond# 22). LNA covariance attribution validated on synthetic only. FINDINGS N-D saddle + Lyapunov."}
references: [GardnerCollins2000, HuangFerrell1996, ElfEhrenberg2003]
---

# Mechanism Card — Toggle switch (motif)

> **ID:** `NUDGE-MOTIF-003`  ·  **Role:** motif
> **Stability:** experimental (attribution abstains on a single snapshot)  ·  **Registry name:** — (a motif)

## Summary

The canonical 2-node **mutual-repression** toggle (Gardner–Collins): two species each
`hill_repression`-inhibiting the other, giving **anti-correlated** bistable modes
(A-high/B-low ↔ A-low/B-high). NUDGE's testbed for extending fail-safe attribution
beyond the 1-species self-activation switch — where it found a principled boundary.

## Governing equation

Two species (`A`, `B`), each `basal = 0.05`, `decay = 1.0`, with two repression edges
(defaults `n = 4`, `vmax = 2`, `K = 1`):

```
dA/dt = basal + vmax · Kⁿ/(Kⁿ + Bⁿ) − decay · A     (B ⊣ A)
dB/dt = basal + vmax · Kⁿ/(Kⁿ + Aⁿ) − decay · B     (A ⊣ B)
```

Per edge: gain = `n`, threshold = `K`, ceiling = `vmax`.

## What it represents

The engineered genetic toggle — bistable memory from mutual repression. Its saddle
sits on the diagonal for symmetric parameters (`[1.017, 1.017]`) and moves
off-diagonal when the edges are asymmetric (`[0.933, 1.061]`).

## Assumptions & simplifications

- Two lumped species, symmetric edges by default.
- Quasi-steady-state edge responses.
- Attribution analyses assume the linear-noise (LNA) Gaussian-mixture approximation
  is locally valid (breaks at low depth / near a saddle-node / monostability).

## Known failure modes

| Failure mode | Decoy that exercises it | Limitation |
|---|---|---|
| A dead-guide (null) perturbation reported as a mechanism | `NUDGE-DECOY-004` | `NUDGE-LIM-004` |
| Readout ultrasensitivity misattributed to the circuit | `nudge.data.decoy_generators.generate_readout_nonlinearity_decoy` (xfail witness) | `NUDGE-LIM-006` |

The load-bearing honest limitation is the **single-snapshot gain⇄threshold
degeneracy** (below), not a decoy — it is an information-geometry property of the
motif. The readout row above is likewise **not** a passing decoy: it is
`NUDGE-LIM-006`, a documented boundary where NUDGE can be a *confident false positive*
rather than abstain, witnessed by an **xfail** (`generate_readout_nonlinearity_decoy`).

## Identifiability regime

Established, and deliberately conservative (FINDINGS "N-D saddle" + "Covariance
attribution"):

- **The 1-D gain gate does NOT extend — a measured NO-GO.** Reducing cooperativity on
  one repression edge does not collapse the toggle (the other edge keeps it bistable),
  so the saddle-centred `w_trans` signature is absent: free-`n` `w_trans` for the gain
  condition across seeds is **0.00 / 0.25 / 0.22** (vs the clean 1-D 0.87–0.94), far
  below `τ = 0.5`. The gate stays guarded to `n_species == 1` and NUDGE **abstains**
  (off-model) on a toggle rather than misclassify.
- **Single-snapshot Fisher information:** the sloppy direction is **gain (`n`) ⇄
  threshold (`K`)** — `corr(log m, log K) = −0.986`, condition number ≈ 210 — because
  the snapshot constrains only `m·ln(K/B)`. **Ceiling (`vmax`) is the *most*
  identifiable** parameter (it sets the high-mode plateau). A **second operating
  point** (dose/basal shift) breaks the degeneracy (smallest FIM eigenvalue ×16.5;
  condition number 210 → 22) — a constitutive control does **not** (×1.01; that is a
  different axis, LIM-006).
- **LNA covariance attribution** (`nudge.inference.lyapunov.attribute_lyapunov_single`)
  is validated on synthetic/LNA ground truth only: from one snapshot it **identifies ceiling and
  abstains (`gain_or_threshold`)**; across operating points the gain↔threshold gap
  widens ~20× and it resolves. Not yet validated on real data.

## Implementation Mapping

| Equation term | Code |
|---|---|
| the circuit builder | `nudge.circuits.toggle` |
| repression edge response | `nudge.mechanisms.regulatory.HillRepressionEffect.response` |
| stable basins + index-1 saddle | `nudge.core.circuit.Circuit.fixed_points` |
| N-D saddle (transition state) | `nudge.core.circuit.Circuit.transition_state` |
| per-mode LNA covariances (Lyapunov solve) | `nudge.core.circuit.Circuit.mode_covariances` |

*(Machine-checked by `scripts/check_impl_mapping.py`.)*

## Verification evidence

- `tests/core/test_fixed_points.py::test_symmetric_toggle_two_stable_one_saddle` —
  two stable modes and one saddle for the symmetric toggle.
- `tests/core/test_fixed_points.py::test_asymmetric_toggle_saddle_moves_off_diagonal`
  — the saddle moves off-diagonal for asymmetric edges.
- `tests/core/test_mode_covariance.py::test_toggle_covariances_match_lna_reference`
  — per-mode covariances match the LNA/FIM reference.
- `tests/verification/test_toggle_nd_safety.py::test_toggle_transition_never_misclassifies`
  — NUDGE never emits a wrong positive on toggle threshold/gain/ceiling data.

## References

- [@GardnerCollins2000] — the engineered genetic toggle switch (mutual repression).
- [@HuangFerrell1996] — cooperativity/ultrasensitivity of the repression edges.
- [@ElfEhrenberg2003] — the linear-noise approximation underlying the covariance attribution.
