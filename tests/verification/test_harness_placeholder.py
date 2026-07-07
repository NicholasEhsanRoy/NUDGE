"""Placeholder verification benchmark — proves the verification lane runs in CI.

Also exercises the inherited ``maddening.compliance.verification_benchmark``
decorator (Appendix F: register at least one). Replaced by the real recovery
benchmarks (confusion matrix, parameter recovery, false-positive guard, SOS
dry-run, blindness) in Phase 3.
"""

from __future__ import annotations

import pytest
from maddening.compliance import BenchmarkType, verification_benchmark

pytestmark = pytest.mark.verification

_BENCHMARK_TYPE = next(iter(BenchmarkType))


@verification_benchmark(
    benchmark_id="NUDGE-VER-000",
    description="Harness placeholder proving the verification lane runs in CI.",
    node_type="none",
    benchmark_type=_BENCHMARK_TYPE,
    acceptance_criteria="Trivially passes; real recovery benchmarks land in Phase 3.",
)
def test_verification_harness_placeholder() -> None:
    assert True
