# NUDGE Mechanism Ontology — design (SPARQL/RDF vision)

**Status: DESIGN + a costed prototype sketch.** Per the build brief, the core
(CLI + MCP + skills) ships first; this document records the ontology vision and
assesses whether a lightweight implementation is cheap enough to prototype. The
answer: **yes, cheap** (an `rdflib` graph built from card front-matter + a handful
of SPARQL queries, ~1 evening), but it is deliberately **not on the critical
path** — the Python knowledge layer (`nudge.knowledge`) already answers the
"why did NUDGE abstain?" question by following the same relations. The ontology is
the *formal, queryable* version of that, and the design below is what would guide
building it.

---

## 1. Why an ontology at all

NUDGE's Mechanism Cards already carry machine-readable relations in their YAML
front-matter (authored in Phase C):

```yaml
id: NUDGE-MECH-002
name: hill_activation
role: regulatory-edge
registry_name: HillActivation
vulnerable_to_decoys: [NUDGE-DECOY-005]
documented_limitation: [NUDGE-LIM-005, NUDGE-LIM-006]
validated_in_regime: {min_cells_per_condition: 1000, notes: "..."}
references: [HuangFerrell1996, Das2009, FerrellMachleder1998]
```

Today `nudge.knowledge.explain()` reads these plus `known_limitations.yaml` and
`DECOY_BATTERY` and *follows the edges in Python*. That works, but the relations
are implicit in code. An ontology makes them **first-class, queryable, and
composable**: instead of a bespoke function per question, any question that is a
path through the graph becomes one SPARQL query — and the graph can be *reasoned
over* (transitive closure, consistency checks, joining in external ontologies
like the Gene Ontology or a drug-target KG).

The concrete payoff for the demo: when NUDGE abstains on a dataset, the MCP server
answers *"why?"* by a graph traversal —
**abstention verdict → the decoy that produces it → the limitation that bounds it
→ the mechanism card that explains it** — and can say, e.g., *"this resembles the
noise-induced-bimodality failure (NUDGE-DECOY-001 → NUDGE-LIM-001); the honest
verdict is off-model — see `self_activation_switch.md`."*

---

## 2. The schema (classes + relations)

A tiny domain vocabulary under a `nudge:` namespace
(`https://nudge.bio/ontology#`):

**Classes**
- `nudge:Mechanism` (a primitive: Hill activation, the integrators, the readout)
- `nudge:Motif` (a circuit: ras_switch_1node, toggle, …)
- `nudge:Decoy` (a `NUDGE-DECOY-*` adversarial case)
- `nudge:Limitation` (a `NUDGE-LIM-*` documented failure mode)
- `nudge:Verdict` (the attribution vocabulary: threshold/gain/ceiling +
  off-model/unresolved/no-effect/technical-artifact)
- `nudge:Regime` (an identifiability regime: min cells, max dropout, …)
- `nudge:Reference` (a bibliography entry)

**Object properties (the edges)**
- `nudge:vulnerableToDecoy`  (Mechanism/Motif → Decoy)
- `nudge:documentedLimitation` (Mechanism/Motif → Limitation)
- `nudge:exercisedByDecoy`  (Limitation → Decoy)   *(inverse of decoy→limitation)*
- `nudge:expectsVerdict`   (Decoy → Verdict)      *(the correct negative answer)*
- `nudge:validatedInRegime` (Mechanism → Regime)
- `nudge:citesReference`   (Mechanism/Motif → Reference)
- `nudge:explainsVerdict`  (Limitation → Verdict) *(why an abstention was honest)*

Every node's IRI is its existing ID (`NUDGE-MECH-002`, `NUDGE-DECOY-001`,
`NUDGE-LIM-006`, …), so the graph is a *lift* of artifacts that already exist —
no new source of truth, no drift risk beyond what `check_mechanism_cards.py`
already guards.

---

## 3. The queries that matter (SPARQL)

**Q1 — "NUDGE abstained `off-model` on my data. Why, and what should I read?"**
```sparql
SELECT ?decoy ?limitation ?card WHERE {
  ?decoy      nudge:expectsVerdict     nudge:off-model ;
              a                        nudge:Decoy .
  ?mechanism  nudge:vulnerableToDecoy  ?decoy ;
              nudge:documentedLimitation ?limitation .
  ?limitation nudge:explainsVerdict    nudge:off-model .
  ?mechanism  nudge:hasCard            ?card .
}
```

**Q2 — "Which mechanisms are unidentifiable below 1000 cells/condition?"**
```sparql
SELECT ?mechanism ?minCells WHERE {
  ?mechanism nudge:validatedInRegime ?r .
  ?r         nudge:minCellsPerCondition ?minCells .
  FILTER (?minCells >= 1000)
}
```

**Q3 — "Every decoy that a Hill-activation edge must resist, with its limitation."**
```sparql
SELECT ?decoy ?limitation WHERE {
  nudge:NUDGE-MECH-002 nudge:vulnerableToDecoy ?decoy .
  ?decoy nudge:exercisedByLimitation ?limitation .
}
```

Each of these is a question `nudge.knowledge` answers today in Python; the
ontology's value is that *new* questions (paths nobody hard-coded) become free.

---

## 4. A costed prototype (rdflib) — cheap, opt-in, not-yet-built

The whole builder is ~60 lines because the front-matter already carries the
relations. Sketch (would live at `src/nudge/ontology.py`, guarded behind an
optional `ontology` extra pulling `rdflib`):

```python
from rdflib import Graph, Namespace, Literal, RDF
from nudge.knowledge import list_mechanisms, list_decoys, list_limitations
from nudge.knowledge import _front_matter, _cards_dir   # front-matter per card

NS = Namespace("https://nudge.bio/ontology#")

def build_graph() -> Graph:
    g = Graph(); g.bind("nudge", NS)
    for card in _cards_dir().glob("*.md"):
        fm = _front_matter(card)
        if not fm.get("id"):
            continue
        subj = NS[fm["id"]]
        g.add((subj, RDF.type, NS.Motif if fm.get("registry_name") is None else NS.Mechanism))
        g.add((subj, NS.hasCard, Literal(card.name)))
        for d in fm.get("vulnerable_to_decoys", []):
            g.add((subj, NS.vulnerableToDecoy, NS[d]))
        for lim in fm.get("documented_limitation", []):
            g.add((subj, NS.documentedLimitation, NS[lim]))
        for ref in fm.get("references", []):
            g.add((subj, NS.citesReference, NS[ref]))
    for d in list_decoys():
        g.add((NS[d["decoy_id"]], RDF.type, NS.Decoy))
        g.add((NS[d["decoy_id"]], NS.expectsVerdict, NS[d["expected_verdict"]]))
        if d["limitation_ref"]:
            g.add((NS[d["decoy_id"]], NS.exercisedByLimitation, NS[d["limitation_ref"]]))
    # … limitations → Verdict via a small static map (as in _ABSTENTION_GUIDANCE) …
    return g
```

**Effort:** ~1 evening including the SPARQL queries and a `test_ontology.py` that
asserts the graph round-trips the card relations. **Risk:** low — additive, opt-in,
reads existing artifacts. **Why it is not built now:** the CLI/MCP already answer
the demo's questions; the ontology is the formalization, best added when a second
consumer appears (e.g. joining NUDGE relations to an external target-KG). The MCP
server would gain one tool, `query_ontology(sparql)`, delegating to `build_graph()`.

---

## 5. Stretch: problem-specific ontologies updated by Bayesian reasoning

The forward-looking idea the user described: an ontology is not only a static map
of NUDGE's *method* knowledge — it can hold a **problem-specific** posterior over
mechanism hypotheses for a given dataset, updated as NUDGE results arrive.

- Seed a per-dataset graph with prior beliefs (`hypothesis: SOS1 moves gain`,
  `RASGRP1 moves threshold`) drawn from the literature (`references` edges into
  PubMed/GO).
- Each NUDGE run adds evidence nodes: `(SOS1, attributedVerdict, unresolved,
  operatingPoint: Stim8hr, nll_gap: 0.005)`. A second operating point adds
  another; the breaker's NLL gap is the likelihood term.
- A Bayesian update over the hypothesis nodes turns the accumulating evidence into
  a posterior the reviewer agent can *trace* — every belief edge cites the exact
  fit (provenance hash) that moved it. This composes directly with Claude
  Science's per-artifact provenance and with NUDGE's own `provenance.py` (a stub
  today).

This is a research direction, not a hackathon deliverable — recorded here so the
lightweight ontology above is built with the right target in mind: **the graph is
the substrate a reasoning agent updates, not just a lookup table.**

---

## 6. Recommendation

1. **Ship** CLI + MCP + skills + Mechanism Cards (done) — they answer the demo's
   questions with the relations expressed in code + front-matter.
2. **When a second consumer appears** (external KG join, a reviewer agent that
   wants SPARQL), add `src/nudge/ontology.py` + the `ontology` extra + a
   `query_ontology` MCP tool, per §4. Keep `check_mechanism_cards.py` as the
   anti-drift guard so the graph never outruns the evidence.
3. **Treat §5 as the north star** for the schema: model *hypotheses and evidence*,
   not just *mechanisms and failure modes*, so the same graph later carries a
   traceable, updatable posterior.
