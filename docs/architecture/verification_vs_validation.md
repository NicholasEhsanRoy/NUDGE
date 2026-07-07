# Verification vs validation

NUDGE separates two questions, borrowing the distinction from MADDENING's V&V
architecture (right-sized for research software).

## Verification — "does it do what we claim, and fail when it should?"

The synthetic ground-truth suite. Because a NUDGE circuit is itself a generative
model, we can synthesise data with known-true mechanism labels and assert that:

- mechanism recovery beats a target accuracy (confusion matrix over
  threshold / gain / ceiling);
- parameters are recovered within tolerance;
- **uncertainty is calibrated** (Laplace 90% intervals cover truth ≈ 90% of the
  time — the check that makes "fails safely and loudly" a *tested* property);
- the **false-positive guard** holds (linear-generated data ties the linear
  baseline — no invented mechanism);
- the **decoy battery** passes (each adversarial negative gets the correct
  abstention verdict);
- the blindness diagnostic fires on a synthetic bifurcation case.

Verification runs in CI (markers `verification`, `decoy`).

## Validation — "does it answer a real biological question?"

The T-cell Ras/SOS falsifiable prediction on the real Gladstone dataset: SOS
knockdown should collapse the digital activation signature toward graded;
RasGRP1 knockdown should not. Marked `validation` + `needs_data`; run manually,
documented, **never gating CI**.

## The boundary (context of use)

NUDGE attributes mechanism *given a circuit hypothesis and a context of use* —
raw-count Perturb-seq, a steady-state snapshot, a powered screen. It does **not**
prove the hypothesis, certify data quality, or own the biological validity of the
conclusion. That responsibility is the user's.
