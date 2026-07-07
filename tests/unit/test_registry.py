"""The mechanism registry: register / get / list, with duplicate + unknown guards."""

from __future__ import annotations

import pytest

from nudge.mechanisms.registry import MechanismRegistry


def test_register_and_get() -> None:
    reg = MechanismRegistry()

    @reg.register("Foo")
    class Foo:
        pass

    assert "Foo" in reg
    assert reg.get("Foo") is Foo
    assert reg.list() == ["Foo"]


def test_duplicate_registration_raises() -> None:
    reg = MechanismRegistry()

    @reg.register("Foo")
    class Foo:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @reg.register("Foo")
        class Bar:
            pass


def test_unknown_type_raises_keyerror() -> None:
    reg = MechanismRegistry()
    with pytest.raises(KeyError):
        reg.get("Nope")
