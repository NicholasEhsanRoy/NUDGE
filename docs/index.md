# NUDGE documentation

NUDGE fits a differentiable gene-regulatory circuit to Perturb-seq data and
attributes each perturbation to a **mechanism** (threshold / gain / ceiling) —
or abstains, loudly, when the data cannot say.

- **Start here:** [`../design/PITCH.md`](../design/PITCH.md) (plain language) and
  [`../design/WORKING_BACKWARDS.md`](../design/WORKING_BACKWARDS.md) (full reasoning).
- **User guide:** [installation](user_guide/installation.md) ·
  [quickstart](user_guide/quickstart.md) ·
  [data contract](user_guide/data_contract.md) (raw counts only — read this).
- **Mechanism Cards:** [`mechanism_cards/`](mechanism_cards/) — one per mechanism.
- **Architecture:** [verification vs validation](architecture/verification_vs_validation.md).
- **Known limitations:** [`known_limitations.yaml`](known_limitations.yaml).
