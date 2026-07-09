---
name: nudge-attribute
description: Use when the user wants to attribute a Perturb-seq perturbation's mechanism with NUDGE — does a knockdown move a switch's threshold (K), gain (n), or ceiling (vmax), or should NUDGE abstain. Drives the `nudge attribute` CLI / the `attribute` MCP tool over a raw-count .h5ad, then reads the honest output (including skips/abstentions) back to the user. Composes with nudge-explain for any non-positive verdict.
---

# nudge-attribute

NUDGE fits a compositional differentiable circuit to single-cell Perturb-seq
counts and attributes each perturbation to a **mechanism** — threshold / gain /
ceiling — and **abstains loudly** when the data cannot say. This skill runs an
attribution and interprets the result *honestly*: an abstention or a skip is a
first-class, correct outcome, never a failure to paper over.

## Preconditions (the data contract)

- Input is a **raw integer-count** `.h5ad` (NUDGE owns its NB count model —
  log1p / CPM / scaled / imputed data is rejected). Check first:
  ```
  nudge check-data <file.h5ad>
  ```
- The file holds a control condition (default label `WT`) and the target
  perturbation in `obs['condition']`. Inspect with `nudge load <file.h5ad>`.

## Run the attribution

```
nudge attribute <file.h5ad> --target <GENE> [--topology 1node|2node|toggle] \
    [--marker SPECIES=GENE1,GENE2 ...] [--control WT] [--steps 200]
```
- `--topology` is the circuit hypothesis (default `1node` self-activation switch).
- `--marker` maps a circuit species to its readout genes; it defaults to the
  species' own name, which is what synthetic data uses.
- Programmatic / MCP equivalent: the `attribute(h5ad_path, target, ...)` tool, or
  `nudge.service.attribute_file(...)`.

## Read the output honestly (the important part)

The report prints, per operating point, either a **call** or a **SKIPPED** reason:

- **A positive call** (`threshold` / `gain` / `ceiling`) — report it with the
  restricted-fit NLLs shown, and note it is a *single-operating-point* call.
- **`unresolved`** — expected at a single operating point: gain and threshold are
  Fisher-degenerate there. The breaker is a **second operating point**
  (pass more condition files), *not* more cells. Say so.
- **`SKIPPED: LNA unreliable ...` / too few cells** — NUDGE declined because the
  linear-noise approximation is untrustworthy (low sequencing depth, near a
  bifurcation, or monostable) or the condition is underpowered (<200 target
  cells). This is the fail-safe working. Report the reason verbatim.
- **`off-model` / `no-effect`** — the parsimony gate found no switch, or the
  perturbation ≈ WT (a dead guide).

**Never** upgrade a skip/abstention into a confident mechanism call. For any
non-positive verdict, hand the user the *why* via the `nudge-explain` skill:
`nudge explain <verdict>` pulls the matching decoy + documented limitation +
Mechanism Card.

## Guardrails

- If `check-data` rejects the input, stop and surface the exact reason — do not
  attempt to coerce normalized data into counts.
- Single-condition attribution abstaining between gain/threshold is the *designed*
  behaviour (see `scripts/vv/FINDINGS.md`), not a bug. Set expectations up front.
