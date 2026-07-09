#!/usr/bin/env python3
"""Verify every registered mechanism has a matching Mechanism Card.

Two CI checks in one, keeping the card library from going stale:

1. **Coverage** — every mechanism in ``default_registry`` (populated by importing
   the mechanism modules) has a card in ``docs/mechanism_cards`` whose front-matter
   ``registry_name`` matches. A new registered mechanism without a card fails here.
2. **Well-formedness** — every card's YAML front-matter parses and carries the
   required machine-readable keys (the future-ontology relations).

Template files (``_*.md``) and the index (``README.md``) are skipped. Exit 0 if all
checks pass, 1 otherwise. Style matches the sibling ``scripts/check_*.py``
(argparse-free, ``raise SystemExit(main())``).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import yaml

# Importing these modules populates ``default_registry`` at import time (the classes
# register via decorator). ``nudge.mechanisms`` alone does not pull in every
# integrator, so the leaf modules are imported explicitly.
_MECHANISM_MODULES = (
    "nudge.mechanisms.regulatory",
    "nudge.mechanisms.integrators.linear",
    "nudge.mechanisms.integrators.saturating",
    "nudge.mechanisms.readout",
)

_REQUIRED_KEYS = (
    "id",
    "name",
    "role",
    "registry_name",
    "vulnerable_to_decoys",
    "documented_limitation",
    "validated_in_regime",
    "references",
)

_CARDS_DIR = Path("docs/mechanism_cards")


def _front_matter(text: str) -> dict[str, object] | None:
    """Parse a card's leading ``---`` YAML front-matter block, or ``None``."""
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    loaded = yaml.safe_load(parts[1])
    return loaded if isinstance(loaded, dict) else None


def _registered_names() -> list[str]:
    for module in _MECHANISM_MODULES:
        importlib.import_module(module)
    from nudge.mechanisms.registry import default_registry

    return default_registry.list()


def main() -> int:
    errors: list[str] = []

    registry_to_card: dict[str, Path] = {}
    for md in sorted(_CARDS_DIR.rglob("*.md")):
        if md.name.startswith("_") or md.name == "README.md":
            continue
        fm = _front_matter(md.read_text())
        if fm is None:
            errors.append(f"{md}: missing or unparseable YAML front-matter")
            continue
        missing = [k for k in _REQUIRED_KEYS if k not in fm]
        if missing:
            errors.append(f"{md}: front-matter missing keys {missing}")
        registry_name = fm.get("registry_name")
        if isinstance(registry_name, str) and registry_name:
            if registry_name in registry_to_card:
                errors.append(
                    f"{md}: registry_name {registry_name!r} also claimed by "
                    f"{registry_to_card[registry_name]}"
                )
            registry_to_card[registry_name] = md

    for name in _registered_names():
        if name not in registry_to_card:
            errors.append(
                f"registered mechanism {name!r} has no Mechanism Card "
                f"(no card front-matter with registry_name: {name})"
            )

    for msg in errors:
        print(f"check_mechanism_cards: {msg}", file=sys.stderr)
    if errors:
        return 1
    print("check_mechanism_cards: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
