"""Stage-2 design — invert a fitted circuit to propose interventions.

``design(target_outcome)`` gradient-descends over perturbation space on the same
differentiable circuit ``fit`` produced, returning ranked interventions
(including untested gene combinations) each with a Laplace error bar. Because the
circuit is one differentiable object, this is the forward machinery run backwards.
Phase-0 stub (stretch).
"""

from __future__ import annotations

from typing import Any

__all__ = ["design"]


def design(target_outcome: Any, **kwargs: Any) -> Any:
    """Propose ranked interventions achieving ``target_outcome``. Phase-0 stub."""
    raise NotImplementedError("design — stretch (Stage 2)")
