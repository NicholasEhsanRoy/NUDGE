# Quickstart

> Pre-alpha: the surface below is defined; the engines are being filled in
> (Phases 1–4). This is the intended shape.

```python
import nudge

# 1. Describe a circuit — fluent builder (power users) ...
circuit = (
    nudge.CircuitBuilder()
    .add_species("RasGRP1")
    .add_species("RasGTP")
    .add_species("SOS")
    .regulate("RasGRP1", "RasGTP", effect="HillActivation")
    .feedback("RasGTP", "SOS", effect="HillActivation")   # feedback = an edge that closes a cycle
    .regulate("SOS", "RasGTP", effect="HillActivation")
    .build()
)
# ... or from a serializable CircuitSpec (YAML / visual builder).

# 2. Fit to raw-count Perturb-seq data → a MechanismMap.
result = nudge.fit(adata, circuit)   # adata.X must be raw integer counts
for call in result.calls:
    print(call.perturbation, call.mechanism, call.confidence)

# 3. (Stretch) invert the fit to propose interventions.
plan = nudge.design(target_outcome)
```

See the [data contract](data_contract.md) before passing your own `adata`.
