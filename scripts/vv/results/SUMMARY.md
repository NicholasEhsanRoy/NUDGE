# Overnight V&V results

**Recommended margin_k = 1.70**  (false-positive rate on linear data 1.7%, correct-attribution 65.0%, misclassification 0.0%, unresolved 24.2%)

- Calibrated against 300 synthetic linear datasets and 120 switch datasets.


## Confusion (rows = true mechanism, at recommended margin_k)

| true \ called | threshold | gain | ceiling | unresolved | off-model |
|---|---|---|---|---|---|
| threshold | 0.55 | 0.00 | 0.00 | 0.40 | 0.05 |
| gain | 0.00 | 0.79 | 0.00 | 0.08 | 0.13 |
| ceiling | 0.00 | 0.00 | 0.62 | 0.25 | 0.12 |


## Figures
- `calibration.png`
- `identifiability_cells_noise.png`
- `identifiability_effect_cells.png`
