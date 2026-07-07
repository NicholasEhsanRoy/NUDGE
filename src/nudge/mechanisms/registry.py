"""The mechanism registry — the single source of truth mapping a config
``type`` string to a mechanism class.

Deliberately an *encapsulated object*, not a bare module-global dict: multiple
registries can coexist, there are no import-order side effects, and the surface
stays auditable and testable. A module-level ``default_registry`` instance is
provided for convenience, and ``register`` targets it by default.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class MechanismRegistry:
    """Maps mechanism ``type`` names to their implementing classes."""

    def __init__(self) -> None:
        self._entries: dict[str, type] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Return a decorator registering a mechanism class under ``name``."""

        def _decorate(cls: type[T]) -> type[T]:
            if name in self._entries:
                raise ValueError(f"mechanism {name!r} is already registered")
            self._entries[name] = cls
            return cls

        return _decorate

    def get(self, name: str) -> type:
        """Return the class registered under ``name`` (``KeyError`` if absent)."""
        try:
            return self._entries[name]
        except KeyError:
            known = ", ".join(self.list()) or "<none>"
            raise KeyError(
                f"unknown mechanism type {name!r}; registered: {known}"
            ) from None

    def list(self) -> list[str]:
        """Return the sorted list of registered mechanism names."""
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries


#: The default registry that mechanism modules register into at import time.
default_registry = MechanismRegistry()
