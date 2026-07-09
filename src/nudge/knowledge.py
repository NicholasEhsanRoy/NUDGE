"""Read-only knowledge base — mechanisms, decoys, limitations, Mechanism Cards.

The single tested source that the ``nudge`` CLI, the MCP server, and the Agent
Skills all read, so a human and Claude get the *same* honest answer — including,
when NUDGE abstains, *which* documented failure mode / decoy explains it.

Pure lookups over four artifacts already in the repo:

- the mechanism **registry** (``nudge.mechanisms.registry.default_registry``),
- the **decoy battery** (``nudge.data.decoys.DECOY_BATTERY``),
- the **limitations registry** (``docs/known_limitations.yaml``), and
- the **Mechanism Cards** (``docs/mechanism_cards/*.md``, with YAML front-matter).

No fitting, no JAX math of its own — this module only *reads*. It degrades
gracefully if the cards are absent (the card knowledge base is authored
separately) so the CLI/MCP still answer from the registry + decoys + limitations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from nudge.core.vocabulary import ABSTENTION_CLASSES, MechanismClass

__all__ = [
    "AbstentionExplanation",
    "explain",
    "get_mechanism_card",
    "list_decoys",
    "list_limitations",
    "list_mechanisms",
    "repo_root",
]


def repo_root() -> Path:
    """Locate the repository root (the dir holding ``docs/`` and ``pyproject.toml``).

    Walks up from this file; robust to editable installs and to running from a git
    worktree. Falls back to ``parents[2]`` (``src/nudge/knowledge.py`` → repo).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "docs").is_dir():
            return parent
    return here.parents[2]


def _cards_dir() -> Path:
    return repo_root() / "docs" / "mechanism_cards"


def _limitations_path() -> Path:
    return repo_root() / "docs" / "known_limitations.yaml"


# --------------------------------------------------------------------------- #
# Mechanisms (the registry + their in-code MechanismMeta cards)
# --------------------------------------------------------------------------- #
def list_mechanisms() -> list[dict[str, Any]]:
    """Return every registered mechanism with its metadata and card path (if any).

    Importing :mod:`nudge.mechanisms` populates the default registry as a side
    effect; we do it here so callers need not remember to.
    """
    import nudge.mechanisms  # noqa: F401  (registration side effect)
    from nudge.mechanisms.registry import default_registry

    out: list[dict[str, Any]] = []
    for name in default_registry.list():
        cls = default_registry.get(name)
        meta = getattr(cls, "meta", None)
        card = _card_for_registry_name(name)
        out.append(
            {
                "registry_name": name,
                "class": cls.__name__,
                "algorithm_id": getattr(meta, "algorithm_id", None),
                "role": getattr(getattr(meta, "role", None), "value", None),
                "summary": getattr(meta, "summary", ""),
                "references": list(getattr(meta, "references", ())),
                "card": card.name if card is not None else None,
            }
        )
    return out


@lru_cache(maxsize=1)
def _card_index() -> dict[str, Path]:
    """Map every lookup key → card path: stem, ``name:``, and ``registry_name:``."""
    index: dict[str, Path] = {}
    cards = _cards_dir()
    if not cards.is_dir():
        return index
    for md in sorted(cards.glob("*.md")):
        if md.name.startswith(("_", "README")):
            continue
        index[md.stem.lower()] = md
        fm = _front_matter(md)
        for key in ("name", "registry_name"):
            val = fm.get(key)
            if isinstance(val, str) and val:
                index[val.lower()] = md
    return index


def _card_for_registry_name(name: str) -> Path | None:
    return _card_index().get(name.lower())


def _front_matter(md: Path) -> dict[str, Any]:
    """Parse the leading ``---`` YAML front-matter block of a Markdown file."""
    try:
        import yaml
    except ImportError:  # pragma: no cover - pyyaml is a ci/dev dep
        return {}
    text = md.read_text()
    if not text.startswith("---"):
        return {}
    _, _, rest = text.partition("---")
    block, _, _ = rest.partition("\n---")
    try:
        data = yaml.safe_load(block)
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def get_mechanism_card(name: str) -> str | None:
    """Return the raw Markdown of a Mechanism Card, or ``None`` if not found.

    ``name`` may be the card file stem (``hill_activation``), the front-matter
    ``name:``, or the registry key (``HillActivation``).
    """
    card = _card_index().get(name.lower())
    return card.read_text() if card is not None else None


# --------------------------------------------------------------------------- #
# Decoys + limitations
# --------------------------------------------------------------------------- #
def list_decoys() -> list[dict[str, Any]]:
    """Return the decoy battery as plain dicts (id, summary, verdict, limitation)."""
    from nudge.data.decoys import DECOY_BATTERY

    return [
        {
            "decoy_id": d.decoy_id,
            "summary": d.summary,
            "expected_verdict": d.expected_verdict.value,
            "limitation_ref": d.limitation_ref,
            "authored_by": d.authored_by,
        }
        for d in DECOY_BATTERY
    ]


def list_limitations() -> list[dict[str, Any]]:
    """Return the ``known_limitations.yaml`` anomaly entries (empty if absent)."""
    path = _limitations_path()
    if not path.is_file():
        return []
    try:
        import yaml
    except ImportError:  # pragma: no cover
        return []
    data = yaml.safe_load(path.read_text()) or {}
    anomalies = data.get("anomalies", []) if isinstance(data, dict) else []
    return [a for a in anomalies if isinstance(a, dict)]


def _limitation(anomaly_id: str) -> dict[str, Any] | None:
    for entry in list_limitations():
        if entry.get("anomaly_id") == anomaly_id:
            return entry
    return None


# --------------------------------------------------------------------------- #
# The headline lookup: "why did NUDGE abstain?"
# --------------------------------------------------------------------------- #

#: For each abstention class, the human-facing meaning + the cards a user should read.
_ABSTENTION_GUIDANCE: dict[MechanismClass, tuple[str, tuple[str, ...]]] = {
    MechanismClass.OFF_MODEL: (
        "The mechanistic switch model did not beat a linear baseline beyond the loss "
        "noise floor (the circuit-level parsimony gate). Either the data holds no "
        "switch, or a nonlinear *readout* is manufacturing apparent ultrasensitivity "
        "the affine-reporter model cannot separate from a circuit switch.",
        ("hill_activation", "readout", "self_activation_switch"),
    ),
    MechanismClass.NO_EFFECT: (
        "The perturbed distribution is within the effect margin of WT (a multiple of "
        "the WT self-distance) — e.g. a dead guide that targets but does not knock "
        "down. NUDGE declines to call a mechanism merely because the WT is a switch.",
        ("hill_activation",),
    ),
    MechanismClass.UNRESOLVED: (
        "The restricted-fit posteriors overlap: at a single operating point a gain "
        "change and a threshold shift give near-identical distributions (the measured "
        "gain⇄threshold Fisher degeneracy). The breaker is a *second operating point*, "
        "not more cells at one.",
        ("hill_activation", "ras_switch_1node"),
    ),
    MechanismClass.TECHNICAL_ARTIFACT: (
        "The bimodality is population/technical structure (a cell-type or doublet "
        "mixture, or a depth-driven dropout zero-peak), not a dynamical switch.",
        ("readout",),
    ),
}


@dataclass(frozen=True)
class AbstentionExplanation:
    """Why NUDGE returned a given non-positive verdict, with the evidence to read."""

    verdict: str
    meaning: str
    decoys: list[dict[str, Any]]
    limitations: list[dict[str, Any]]
    cards: list[str]


def _explain_abstention(cls: MechanismClass) -> AbstentionExplanation:
    meaning, card_names = _ABSTENTION_GUIDANCE.get(
        cls, (f"NUDGE returned '{cls.value}'.", ())
    )
    decoys = [d for d in list_decoys() if d["expected_verdict"] == cls.value]
    lim_ids = {d["limitation_ref"] for d in decoys if d["limitation_ref"]}
    limitations = [entry for lid in sorted(lim_ids) if (entry := _limitation(lid))]
    cards = [c for c in card_names if get_mechanism_card(c) is not None]
    return AbstentionExplanation(
        verdict=cls.value,
        meaning=meaning,
        decoys=decoys,
        limitations=limitations,
        cards=cards,
    )


def explain(query: str) -> dict[str, Any]:
    """Resolve ``query`` to a structured explanation.

    ``query`` may be an abstention class (``off-model``, ``unresolved``,
    ``no-effect``, ``technical-artifact``), a decoy id (``NUDGE-DECOY-001``), a
    limitation id (``NUDGE-LIM-006``), or a mechanism/card name (``hill_activation``,
    ``HillActivation``). Returns ``{"kind": ..., ...}`` — always a dict, never raises
    on an unknown query (returns ``{"kind": "unknown", ...}`` with suggestions).
    """
    q = query.strip()
    ql = q.lower()
    qu = q.upper()

    # 1) an abstention (or positive) mechanism class
    for cls in MechanismClass:
        if ql == cls.value:
            if cls in ABSTENTION_CLASSES:
                exp = _explain_abstention(cls)
                return {"kind": "abstention", **exp.__dict__}
            return {
                "kind": "attribution",
                "verdict": cls.value,
                "meaning": _POSITIVE_MEANING.get(cls, ""),
            }

    # 2) a decoy id
    if qu.startswith("NUDGE-DECOY"):
        for d in list_decoys():
            if d["decoy_id"] == qu:
                lim = _limitation(d["limitation_ref"]) if d["limitation_ref"] else None
                return {"kind": "decoy", **d, "limitation": lim}
        ids = [d["decoy_id"] for d in list_decoys()]
        return {"kind": "unknown", "query": q, "suggestions": ids}

    # 3) a limitation id
    if qu.startswith("NUDGE-LIM"):
        lim = _limitation(qu)
        if lim is not None:
            return {"kind": "limitation", **lim}
        ids = [e["anomaly_id"] for e in list_limitations()]
        return {"kind": "unknown", "query": q, "suggestions": ids}

    # 4) a mechanism / card name
    card = get_mechanism_card(ql)
    if card is not None:
        return {"kind": "mechanism_card", "name": ql, "markdown": card}

    return {
        "kind": "unknown",
        "query": q,
        "suggestions": sorted(
            [c.value for c in ABSTENTION_CLASSES]
            + [m["registry_name"] for m in list_mechanisms()]
        ),
    }


_POSITIVE_MEANING: dict[MechanismClass, str] = {
    MechanismClass.THRESHOLD: "The perturbation moves K — *where* the switch trips.",
    MechanismClass.GAIN: "The perturbation moves n — *how sharply* the switch commits.",
    MechanismClass.CEILING: "The perturbation moves vmax — the maximal output.",
    MechanismClass.COMBO: "The perturbation moves more than one of K / n / vmax.",
}
