---
name: nudge-explain
description: Use when NUDGE returned an abstention or negative verdict (off-model, unresolved, no-effect, technical-artifact) or a skip, and the user asks "why?". Also use to look up a decoy (NUDGE-DECOY-00N), a documented limitation (NUDGE-LIM-00N), or a mechanism's card. Drives the `nudge explain` CLI / the `explain_abstention` MCP tool, which pull the matching decoy + limitation + Mechanism Card so the abstention is legible, not opaque.
---

# nudge-explain

NUDGE's whole value is that it **fails safely and loudly** — but a loud "I can't
tell" is only useful if the user learns *why*. This skill turns any NUDGE verdict
into its documented reason: the failure mode, the decoy in the battery that pins
it, the `known_limitations.yaml` entry that bounds it, and the Mechanism Card that
explains the math.

## The one command

```
nudge explain <query>
```
`<query>` may be:

| Query | Returns |
|---|---|
| an abstention class — `off-model`, `unresolved`, `no-effect`, `technical-artifact` | its meaning + the decoys that pin it + documented limitations + which cards to read |
| a decoy id — `NUDGE-DECOY-001` | the adversarial case, its expected verdict, and the limitation it maps to |
| a limitation id — `NUDGE-LIM-006` | the documented failure mode + severity |
| a mechanism name — `hill_activation` (or `HillActivation`) | the full Mechanism Card |

MCP / programmatic equivalents: the `explain_abstention(context)` tool, or
`nudge.knowledge.explain(query)` (returns a structured dict).

## How to answer with it

1. Take the verdict NUDGE actually returned (e.g. `unresolved`, or a `SKIPPED:
   LNA unreliable` reason) and run `nudge explain <verdict>`.
2. Relay the **meaning** plainly, then the **evidence**: name the specific decoy
   (e.g. `NUDGE-DECOY-001` — noise-induced bimodality) and limitation
   (e.g. `NUDGE-LIM-001`) that make this the honest answer.
3. If a Mechanism Card is listed, offer it: `nudge explain hill_activation` (or
   the `get_mechanism_card` MCP tool) for the governing equation + identifiability
   regime.
4. If relevant, state the **fix**: e.g. `unresolved` gain⇄threshold needs a second
   operating point; `off-model` under a nonlinear reporter (`NUDGE-LIM-006`) needs
   a constitutive-reporter control.

## Guardrail

Only report what the tools return. The limitations registry is deliberately the
*honest counterweight* to the pitch — do not soften a documented failure mode
(e.g. `NUDGE-LIM-006`, where NUDGE can be confidently wrong under a strongly
nonlinear readout). Surfacing that bound is the point.
