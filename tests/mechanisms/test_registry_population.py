"""Importing the built-in mechanism modules registers them on the default registry."""

from __future__ import annotations

from nudge.mechanisms.registry import default_registry


def test_builtin_mechanisms_register() -> None:
    import nudge.mechanisms.integrators.linear  # noqa: F401
    import nudge.mechanisms.integrators.saturating  # noqa: F401
    import nudge.mechanisms.regulatory  # noqa: F401

    for name in (
        "LinearEffect",
        "HillActivation",
        "HillRepression",
        "LinearIntegrator",
        "SaturatingIntegrator",
    ):
        assert name in default_registry
