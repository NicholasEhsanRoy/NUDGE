#!/usr/bin/env python3
"""Offline code generator: Proctor 2013 amyloid-β SBML → a differentiable JAX vector field.

Reads the **CC0** BioModels SBML ``BIOMD0000000488`` (Proctor, Boche, Gray & Nicoll,
*"Investigating interventions in Alzheimer's disease with computer simulation models"*,
PLoS ONE 2013; 8(9):e73631; PMID 24098635; DOI 10.1371/journal.pone.0073631) and emits
``src/nudge/mechanisms/_proctor2013.py`` — the species order, initial amounts, the 73
kinetic parameters, the constant boundary species, and a pure-JAX ``rhs(y, p)`` whose
right-hand side is ``dy/dt = S · v(y)`` transcribed **directly from the SBML MathML rate
laws** (mass-action plus the handful of saturating / Hill / dimerization forms the model
uses). Generating the field from the machine-readable SBML — rather than hand-typing 112
reactions — is what makes the reimplementation faithful and auditable.

The SBML rate laws use only ``times / plus / divide / power / ci / cn`` (surveyed), so the
MathML→infix conversion here is exact for this model. Boundary species (``ATP``, ``ADP``,
``AMP``, ``Source``, ``Sink``) are held constant (their ``dy/dt`` is dropped) and appear as
literal constants in the rate laws.

Usage::

    python scripts/vv/gen_proctor2013.py path/to/BIOMD0000000488_url.xml

This is a build tool, not shipped runtime code; the generated module is committed.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_HERE = Path(__file__).resolve().parents[2]
_OUT = _HERE / "src" / "nudge" / "mechanisms" / "_proctor2013.py"


def _tag(e: ET.Element) -> str:
    return e.tag.rsplit("}", 1)[-1]


def _children(e: ET.Element) -> list[ET.Element]:
    return [c for c in e if isinstance(c.tag, str)]


def mathml_to_expr(node: ET.Element) -> str:
    """Convert a MathML content node to a parenthesized Python/JAX infix string."""
    t = _tag(node)
    if t == "math":
        return mathml_to_expr(_children(node)[0])
    if t == "cn":
        return (node.text or "0").strip()
    if t == "ci":
        return (node.text or "").strip()
    if t == "apply":
        kids = _children(node)
        op = _tag(kids[0])
        args = [mathml_to_expr(k) for k in kids[1:]]
        if op == "times":
            return "(" + " * ".join(args) + ")"
        if op == "plus":
            return "(" + " + ".join(args) + ")"
        if op == "minus":
            return "(-" + args[0] + ")" if len(args) == 1 else f"({args[0]} - {args[1]})"
        if op == "divide":
            return f"({args[0]} / {args[1]})"
        if op == "power":
            return f"({args[0]} ** {args[1]})"
        raise ValueError(f"unsupported MathML operator {op!r}")
    raise ValueError(f"unsupported MathML node {t!r}")


def parse(sbml_path: Path) -> dict:
    root = ET.parse(sbml_path).getroot()
    model = next(e for e in root.iter() if _tag(e) == "model")

    species: list[tuple[str, float, bool]] = []
    params: list[tuple[str, float]] = []
    reactions: list[dict] = []

    # top-level (global) parameters live in the model's own listOfParameters
    for lop in _children(model):
        if _tag(lop) == "listOfParameters":
            for pe in _children(lop):
                if _tag(pe) == "parameter":
                    params.append((pe.get("id", ""), float(pe.get("value", "0") or 0.0)))

    for e in model.iter():
        tg = _tag(e)
        if tg == "species":
            sid = e.get("id", "")
            init = float(e.get("initialAmount", e.get("initialConcentration", "0")) or 0.0)
            boundary = e.get("boundaryCondition", "false") == "true"
            species.append((sid, init, boundary))
        elif tg == "reaction":
            rid = e.get("id", "")
            reac: list[tuple[str, float]] = []
            prod: list[tuple[str, float]] = []
            rate = "0"
            for sub in _children(e):
                st = _tag(sub)
                if st == "listOfReactants":
                    reac = [(sr.get("species", ""), float(sr.get("stoichiometry", "1") or 1))
                            for sr in _children(sub)]
                elif st == "listOfProducts":
                    prod = [(sr.get("species", ""), float(sr.get("stoichiometry", "1") or 1))
                            for sr in _children(sub)]
                elif st == "kineticLaw":
                    math = next(m for m in _children(sub) if _tag(m) == "math")
                    rate = mathml_to_expr(math)
            reactions.append({"id": rid, "reactants": reac, "products": prod, "rate": rate})

    return {"species": species, "params": params, "reactions": reactions}


def generate(spec: dict) -> str:
    species = spec["species"]
    params = spec["params"]
    reactions = spec["reactions"]

    dyn = [(s, init) for (s, init, b) in species if not b]
    boundary = [(s, init) for (s, init, b) in species if b]
    dyn_names = [s for s, _ in dyn]
    param_names = [p for p, _ in params]

    # accumulate dy/dt terms per dynamic species from stoichiometry
    accum: dict[str, list[str]] = {s: [] for s in dyn_names}
    for j, rx in enumerate(reactions):
        rj = f"r{j}"
        for sp, st in rx["reactants"]:
            if sp in accum:
                coef = "" if st == 1.0 else f"{st:g} * "
                accum[sp].append(f"- {coef}{rj}")
        for sp, st in rx["products"]:
            if sp in accum:
                coef = "" if st == 1.0 else f"{st:g} * "
                accum[sp].append(f"+ {coef}{rj}")

    L = []
    L.append('"""AUTO-GENERATED — do not edit by hand. Regenerate with')
    L.append("``scripts/vv/gen_proctor2013.py``.")
    L.append("")
    L.append("The Proctor et al. 2013 Alzheimer's-disease amyloid-β / tau / p53 / microglia ODE")
    L.append("model as a differentiable JAX vector field, transcribed directly from the CC0")
    L.append("BioModels SBML ``BIOMD0000000488``.  Mathematical facts (the reaction network +")
    L.append("published rate constants); see the citation in ``ad_qsp.py``.")
    L.append("")
    L.append(f"{len(dyn_names)} dynamic states, {len(boundary)} constant boundary species,")
    L.append(f"{len(param_names)} kinetic parameters, {len(reactions)} reactions.")
    L.append('"""')
    L.append("from __future__ import annotations")
    L.append("")
    L.append("import jax.numpy as jnp")
    L.append("import numpy as np")
    L.append("")
    L.append("SPECIES = (")
    for s in dyn_names:
        L.append(f"    {s!r},")
    L.append(")")
    L.append("")
    L.append("Y0 = np.array([")
    L.append("    " + ", ".join(f"{init:g}" for _, init in dyn) + ",")
    L.append("], dtype=np.float64)")
    L.append("")
    L.append("PARAM_NAMES = (")
    for p in param_names:
        L.append(f"    {p!r},")
    L.append(")")
    L.append("")
    L.append("PARAM_VALUES = np.array([")
    L.append("    " + ", ".join(f"{v:g}" for _, v in params) + ",")
    L.append("], dtype=np.float64)")
    L.append("")
    L.append("BOUNDARY = {")
    for s, init in boundary:
        L.append(f"    {s!r}: {init:g},")
    L.append("}")
    L.append("")
    L.append("N_STATES = len(SPECIES)")
    L.append("N_PARAMS = len(PARAM_NAMES)")
    L.append("")
    L.append("")
    L.append("def rhs(y, p):")
    L.append('    """dy/dt = S . v(y) for the Proctor 2013 network (y: (64,), p: (73,)).')
    L.append("")
    L.append("    ``y`` are the dynamic species (order = ``SPECIES``); ``p`` the kinetic")
    L.append("    parameters (order = ``PARAM_NAMES``). Boundary species are constants.")
    L.append('    """')
    # unpack species
    for i, s in enumerate(dyn_names):
        L.append(f"    {s} = y[{i}]")
    L.append("    # boundary species (held constant)")
    for s, init in boundary:
        L.append(f"    {s} = {init:g}")
    L.append("    # kinetic parameters")
    for i, p in enumerate(param_names):
        L.append(f"    {p} = p[{i}]")
    L.append("    # reaction rates (SBML kinetic laws)")
    for j, rx in enumerate(reactions):
        L.append(f"    r{j} = {rx['rate']}  # {rx['id']}")
    L.append("    # dy/dt accumulation")
    for s in dyn_names:
        terms = accum[s]
        expr = " ".join(terms) if terms else "0.0 * " + s
        # tidy leading '+ '
        expr = expr[2:] if expr.startswith("+ ") else expr
        L.append(f"    d_{s} = {expr}")
    L.append("    return jnp.stack([")
    for s in dyn_names:
        L.append(f"        d_{s},")
    L.append("    ])")
    L.append("")
    return "\n".join(L)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: gen_proctor2013.py path/to/BIOMD0000000488_url.xml", file=sys.stderr)
        return 2
    spec = parse(Path(sys.argv[1]))
    src = generate(spec)
    _OUT.write_text(src)
    print(f"wrote {_OUT} ({len(spec['reactions'])} reactions, "
          f"{sum(1 for _, _, b in spec['species'] if not b)} dynamic states, "
          f"{len(spec['params'])} params)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
