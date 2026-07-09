---
name: mechanism-card
description: Use when writing or updating a Mechanism Card (docs/mechanism_cards/*.md) — the literature-grounded, machine-checkable doc that ships with every mechanism or motif. Enforces the required YAML front-matter, the fixed body sections, real cross-references (decoys / limitations / FINDINGS numbers / test IDs), and the three CI validators. Use it standalone (documenting an existing primitive or motif) or as step 2 of new-mechanism.
---

# mechanism-card

A Mechanism Card is the documentation half of a mechanism (the other half is the
in-code `MechanismMeta` in `src/nudge/core/metadata.py`). It is what makes NUDGE's
abstain-and-attribute vocabulary legible: when NUDGE returns `unresolved` or
`off-model`, the card is where the user learns why that was the honest answer.

## Write the card

Copy `docs/mechanism_cards/_template.md` to `docs/mechanism_cards/<name>.md`. Give it:

1. **YAML front-matter** (machine-readable, for a future ontology) with ALL of these keys:
   ```yaml
   ---
   id: NUDGE-MECH-002          # primitives: the real algorithm_id; motifs: NUDGE-MOTIF-00N
   name: hill_activation
   role: regulatory-edge       # species | integrator | regulatory-edge | readout | perturbation | motif
   registry_name: HillActivation   # the default_registry key for a primitive; null for a motif
   vulnerable_to_decoys: [NUDGE-DECOY-005]      # real DECOY ids only (src/nudge/data/decoys.py)
   documented_limitation: [NUDGE-LIM-005]       # real LIM ids only (docs/known_limitations.yaml)
   validated_in_regime: {min_cells_per_condition: 1000, notes: "..."}
   references: [HuangFerrell1996, Das2009]      # bare bib keys (docs/bibliography.bib)
   ---
   ```
   `registry_name` MUST equal the `default_registry` key for a primitive (this is what
   `scripts/check_mechanism_cards.py` matches); use `null` for motifs.

2. **Body sections** (keep the template's order): governing equation with each
   parameter's biological meaning; what it represents; assumptions & simplifications;
   a **Known failure modes** table mapping each mis-fit to the real `NUDGE-DECOY-*`
   that exercises it and the `NUDGE-LIM-*` it documents; identifiability regime;
   an **Implementation Mapping** table of real `nudge.module.Class.method` qualnames;
   verification evidence; references as `[@Key]` Pandoc citations.

## The honesty rule (hard)

Never claim a validation or identifiability result that is not in
`scripts/vv/FINDINGS.md`. Cite its real numbers (e.g. min ≈1000 cells/condition;
gain > ceiling ≈ threshold; `margin_k` 1.7; the K/vmax degeneracy). If a motif's
regime is uncharacterised, write **"not yet characterised"** — do not invent numbers,
test IDs, or citations. A card must never overclaim.

## Cross-reference real things only

- **Decoys**: `src/nudge/data/decoys.py` (`DECOY_BATTERY`) — plus the LIM-006 readout witness.
- **Limitations**: `docs/known_limitations.yaml` (`NUDGE-LIM-*`).
- **Verification evidence**: a `NUDGE-VER-*` id if one exists; otherwise reference the
  real test file/function that exercises it (e.g.
  `tests/inference/test_fit_end_to_end.py::test_fit_attributes_threshold_gain_ceiling`).
- **References**: add any new bib entry to `docs/bibliography.bib` in the existing
  BibTeX style; every `[@Key]` you cite must exist there.

## Register it

Add a row to `docs/mechanism_cards/README.md` (card | role | id | mechanism).

## Run the validators (all must pass)

```
uv run python scripts/check_citations.py       # every [@Key] resolves in the .bib
uv run python scripts/check_impl_mapping.py     # every `nudge.*` qualname resolves
uv run python scripts/check_mechanism_cards.py  # every registered mechanism has a matching card
uv run pytest tests/docs/ -q                    # the same check, in CI
uv run --extra dev ruff check docs scripts tests
```

`check_impl_mapping` only accepts backticked `nudge.*` refs that resolve to a real
attribute — a bare submodule (e.g. `` `nudge.inference.lyapunov` ``) fails unless it
is re-exported; reference a concrete function/method instead. Avoid writing a literal
`[@Key]`-shaped token in prose (README, examples) — `check_citations` will try to
resolve it.
