# NUDGE Documentation Architecture

**Inheritance statement.** NUDGE inherits its traceability architecture from
[MADDENING's documentation architecture](https://github.com/Microrobotics-Simulation-Framework/MADDENING),
via the `maddening.compliance` package. This document records what NUDGE
*inherits by reference* versus what it *provides itself*, and — crucially — what
it deliberately **leaves out**. The guiding principle (see
`design/WORKING_BACKWARDS.md` Part 6) is **inherit the posture, not the
paperwork**: NUDGE is research software that proposes hypotheses for a wet lab,
not a component of a medical device, so it takes the *discipline* (explicit
scope, documented mechanisms, known failure modes, verification/validation
split, provenance) at a fraction of the weight.

## Inherited by reference (from `maddening.compliance`)

- **Metadata schema** — `NodeMeta`, `EdgeMeta`, `ValidatedRegime`, `Reference`.
- **The `@verification_benchmark` decorator** and its registry.
- **The `@stability` decorator** and stability levels.
- **The anomaly-registry validator** (`validate_anomaly_registry`) and its
  `check-anomalies` CLI, run over `docs/known_limitations.yaml` with the
  `NUDGE-LIM-` prefix.

NUDGE pins `maddening>=0.3.0,<0.4`. (Note: the published 0.3.0 wheel ships the
compliance surface but **not** `ift_linear_solve`; the zero-order integrator and
the re-expressed blindness diagnostic that want it are gated on a MADDENING
release that includes `core.solver_utils`, or a local build.)

## Provided by NUDGE

- **Mechanism Cards** — one per mechanism, from `docs/mechanism_cards/_template.md`;
  the in-code half is `MechanismMeta` (`src/nudge/core/metadata.py`).
- **The known-limitations registry** — `docs/known_limitations.yaml`
  (`NUDGE-LIM-*`), most entries mirroring a decoy in the battery.
- **The decoy battery** — `src/nudge/data/decoys.py` (`NUDGE-DECOY-*`).
- **Verification evidence** — synthetic-recovery + decoy tests
  (`tests/verification/`, `tests/decoys/`; `NUDGE-VER-*`).
- **Per-result provenance** — `src/nudge/provenance.py`, which composes with
  Claude Science's per-figure provenance.
- **CI validators** — `scripts/check_{anomalies,citations,impl_mapping}.py`.

## Verification vs validation (the boundary)

- **Verification** — "does the code do what we claim, and fail when it should?"
  The synthetic ground-truth suite + the decoy battery. Runs in CI.
- **Validation** — "does it answer a real biological question?" The T-cell
  SOS/RasGRP1 falsifiable prediction on real data. Marked `validation` +
  `needs_data`; never gates CI.
- **Boundary statement.** NUDGE attributes mechanism *given a circuit hypothesis
  and a context of use* (raw-count Perturb-seq, steady-state snapshot, a powered
  screen). It does not prove the hypothesis, certify the data quality, or own the
  biological validity of the conclusion — the biologist does.

## Deliberately left out

Named so nobody reflexively ports them from MADDENING/MIME: the full **IEC 62304
SOUP package**, **ISO 14971 risk file**, **EU-MDR / FDA boundary documents**,
**safety classification**, **Notified-Body / QMS apparatus**, and **health-check
watchdog nodes**. Those exist upstream because *something downstream might be a
device*. Nothing downstream of NUDGE is. The one trace of the SOUP idea worth
keeping is a lightweight dependency/version note (recorded in provenance),
because reproducibility needs it anyway.
