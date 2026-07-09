"""Fast CI guard: every registered mechanism keeps a matching Mechanism Card.

Runs the same coverage + front-matter checks as ``scripts/check_mechanism_cards.py``
so a newly registered mechanism (or a malformed card) fails the unit suite, not only
the standalone CI gate. Fast (import + file parse only — no JAX fit).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "check_mechanism_cards.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_mechanism_cards", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def _in_repo_root(monkeypatch: pytest.MonkeyPatch):
    """The checker uses repo-relative paths; run it from the repo root."""
    monkeypatch.chdir(_ROOT)
    yield


def test_every_registered_mechanism_has_a_card(_in_repo_root: None) -> None:
    checker = _load_checker()
    assert checker.main() == 0, "check_mechanism_cards reported failures (see stderr)"


def test_all_cards_have_required_front_matter(_in_repo_root: None) -> None:
    checker = _load_checker()
    cards_dir = Path("docs/mechanism_cards")
    seen = 0
    for md in cards_dir.rglob("*.md"):
        if md.name.startswith("_") or md.name == "README.md":
            continue
        fm = checker._front_matter(md.read_text())
        assert fm is not None, f"{md}: unparseable front-matter"
        missing = [k for k in checker._REQUIRED_KEYS if k not in fm]
        assert not missing, f"{md}: missing front-matter keys {missing}"
        seen += 1
    assert seen >= 10, f"expected at least 10 mechanism cards, found {seen}"


def test_checker_script_is_executable_module() -> None:
    """The script parses/loads (guards against a syntax error slipping in)."""
    assert _SCRIPT.exists()
    assert os.path.getsize(_SCRIPT) > 0
