"""Small shared helpers for the per-mechanism renderers (dataclass⇄dict duality, colours).

Every renderer reads a frozen result dataclass OR its ``*_to_dict()`` / demo dict OR the
already-canonical figure-data dict (the replay path). :func:`get` hides that duality;
:func:`verdict_color` maps any verdict to the semantic palette (abstentions → the grey
abstain slot) so the picture and the honesty overlay always agree.
"""

from __future__ import annotations

from typing import Any

from nudge.viz.base import is_abstention


def get(obj: Any, *names: str, default: Any = None) -> Any:
    """Read an attribute or dict key by any of ``names`` (first hit wins)."""
    for name in names:
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
        elif hasattr(obj, name):
            return getattr(obj, name)
    return default


# Semantic verdict → palette-key. Positive mechanism calls get a distinct hue; every
# abstention resolves to the grey ``abstain`` slot (handled in :func:`verdict_color`).
_VERDICT_KEY = {
    "switch": "switch",
    "graded": "graded",
    "synergistic": "gain",
    "buffering": "threshold",
    "additive": "ceiling",
    "threshold": "threshold",
    "threshold-diff": "threshold",
    "gain": "gain",
    "gain-diff": "gain",
    "ceiling": "ceiling",
    "ceiling-diff": "ceiling",
    "biological-switch": "switch",
    "near-fold": "gain",
    "robust": "ceiling",
    "reachable": "ceiling",
    "intervention": "ceiling",
}


def verdict_color(call: str, pal: dict[str, str]) -> str:
    """Map a verdict to its semantic colour; any abstention → the grey abstain slot."""
    key = (call or "").strip().lower()
    if is_abstention(key):
        return pal["abstain"]
    return pal.get(_VERDICT_KEY.get(key, "threshold"), pal["threshold"])
