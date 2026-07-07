"""Test-suite conftest.

Sets JAX env vars *before* any test module imports JAX, scopes
``jax_enable_x64`` to tests that opt in via ``@pytest.mark.x64`` (so it does not
leak across the session under ``pytest-xdist``), and auto-skips ``@pytest.mark.gpu``
tests when no CUDA device is present. Ported from the MIME sibling repo; all
knobs are no-ops on CPU.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Env vars: must be set BEFORE any JAX import.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

# Persistent compile cache.
_cache = Path(
    os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        str(Path.home() / ".cache" / "jax_compilation_cache"),
    )
)
_cache.mkdir(parents=True, exist_ok=True)

import jax  # noqa: E402 — must follow the env setup above

jax.config.update("jax_compilation_cache_dir", str(_cache))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)


@pytest.fixture(autouse=True)
def _manage_jax_x64(request: pytest.FixtureRequest):
    """Scope ``jax_enable_x64`` to tests carrying ``@pytest.mark.x64``."""
    want = request.node.get_closest_marker("x64") is not None
    prev = jax.config.jax_enable_x64
    if prev != want:
        jax.config.update("jax_enable_x64", want)
    try:
        yield
    finally:
        if jax.config.jax_enable_x64 != prev:
            jax.config.update("jax_enable_x64", prev)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip ``@pytest.mark.gpu`` tests when no CUDA device is available."""
    try:
        has_gpu = any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        has_gpu = False
    if has_gpu:
        return
    skip_gpu = pytest.mark.skip(reason="@pytest.mark.gpu — no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
