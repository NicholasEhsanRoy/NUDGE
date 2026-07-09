"""The demo-latency warmup: it runs, is exported, and is idempotent per process."""

from __future__ import annotations

import nudge
from nudge.warmup import warmup


def test_warmup_is_public_api() -> None:
    assert nudge.warmup is warmup
    assert "warmup" in nudge.__all__


def test_warmup_runs_and_is_idempotent() -> None:
    first = warmup()
    assert isinstance(first, float) and first >= 0.0
    # a second call in the same process is a no-op (the compile caches are warm).
    assert warmup() == 0.0
