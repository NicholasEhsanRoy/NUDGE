"""The attribution vocabulary partitions into positive and abstention classes."""

from __future__ import annotations

from nudge.core.vocabulary import ABSTENTION_CLASSES, POSITIVE_CLASSES, MechanismClass


def test_positive_and_abstention_partition() -> None:
    assert frozenset(MechanismClass) == POSITIVE_CLASSES | ABSTENTION_CLASSES
    assert not (POSITIVE_CLASSES & ABSTENTION_CLASSES)


def test_fail_loud_vocabulary_present() -> None:
    # The whole point: NUDGE must be able to *say* it declined.
    for cls in (
        MechanismClass.NO_EFFECT,
        MechanismClass.UNRESOLVED,
        MechanismClass.TECHNICAL_ARTIFACT,
        MechanismClass.OFF_MODEL,
    ):
        assert cls in ABSTENTION_CLASSES
