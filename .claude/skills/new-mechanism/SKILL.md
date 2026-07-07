---
name: new-mechanism
description: Use when adding a mechanism (Species, integrator, regulatory effect, Readout, or Perturbation) to the NUDGE library. Enforces the contract that keeps the abstain-and-attribute guarantees from rotting: metadata + registry entry + Mechanism Card + a synthetic-recovery test + at least one decoy the mechanism must correctly resist.
---

# new-mechanism

Every mechanism in NUDGE ships as a bundle, not just a class. To add one:

1. **Implement the class** under `src/nudge/mechanisms/` (a `SimulationNode`
   subclass for nodes, an edge for regulatory effects). Register it on the
   `MechanismRegistry` with a `type` name, and attach a `MechanismMeta`
   (`algorithm_id="NUDGE-MECH-NNN"`, role, assumptions, known failure modes,
   references).
2. **Write the Mechanism Card** in `docs/mechanism_cards/<name>.md` from
   `_template.md`: governing equation, assumptions, known failure modes
   cross-referencing the decoy(s), identifiability regime, an Implementation
   Mapping table (checked by `scripts/check_impl_mapping.py`), and references.
3. **Add a synthetic-recovery test** in `tests/verification/` with a
   `@verification_benchmark(benchmark_id="NUDGE-VER-NNN", ...)` — generate data
   from a circuit using this mechanism and assert NUDGE recovers it.
4. **Add at least one decoy it must resist** in `src/nudge/data/decoys.py`
   (`NUDGE-DECOY-NNN`) and, if it names a real failure mode, an entry in
   `docs/known_limitations.yaml` (`NUDGE-LIM-NNN`).
5. **Run the checks:** `ruff check src tests scripts`, `pyright src`, `pytest`,
   and the three `scripts/check_*.py` validators.

Creative-AI idea 2 automates steps 1–2 from a paper via
`scripts/ai/paper_to_mechanism.py`; this skill is the human mirror and the
review checklist for its output.
