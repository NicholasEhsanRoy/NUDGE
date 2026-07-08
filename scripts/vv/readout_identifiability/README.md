# Readout–circuit identifiability (NUDGE-LIM-006 mitigation validation)

Standalone JAX study behind `design/CONSTITUTIVE_CONTROL.md`: a single-population joint
fit of circuit + readout nonlinearity is degenerate (you can't tell a circuit switch
exists), and a **constitutive-reporter control** breaks the degeneracy (rejects "no
switch"). Not part of the `nudge` library — reproducible evidence.

Run (from the repo root):

```bash
uv run python scripts/vv/readout_identifiability/run.py
```

- `model.py` — the minimal forward model (Hill circuit → Hill readout → NB counts).
- `run.py` — Experiment 1 (no control, degenerate) + Experiment 2 (constitutive control,
  degeneracy breaks); writes `results.json`.
- `results.json`, `profiles.png` — the recorded output and profile-likelihood plot.
