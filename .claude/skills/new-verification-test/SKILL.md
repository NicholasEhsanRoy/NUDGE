---
name: new-verification-test
description: Use when adding a synthetic ground-truth verification benchmark (NUDGE-VER-*) — mechanism recovery, parameter recovery, calibration coverage, false-positive guard, the SOS dry-run, or a blindness-diagnostic check. Verification runs in CI; validation (real data) does not.
---

# new-verification-test

Verification answers "does the code do what we claim, and fail when it should?"
using synthetic data with known ground truth.

1. **Add the test** in `tests/verification/`, marked `@pytest.mark.verification`
   (module-level `pytestmark`), and decorate with the inherited
   `maddening.compliance.verification_benchmark(benchmark_id="NUDGE-VER-NNN",
   ...)`.
2. **Generate ground truth** via `nudge.generate_synthetic_perturbseq(...)` with
   the true mechanism labels in `.uns`, then assert recovery / coverage / the
   guard.
3. **Keep it CI-affordable.** If a benchmark needs a full population fit, also
   mark it `slow` so it runs in the scheduled lane, not on every PR.
4. Distinguish from **validation** (`tests/validation/`, markers `validation` +
   `needs_data`): the T-cell real-data prediction is validation and never gates
   CI. See `docs/architecture/verification_vs_validation.md`.
