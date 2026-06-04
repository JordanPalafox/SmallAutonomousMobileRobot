"""bb_helpers — yasmin Blackboard utilities.

The C++-backed Blackboard only accepts a single-argument ``get(key)`` form
and raises ``RuntimeError`` when the key is missing. ``bb_get`` patches that
to behave like ``dict.get`` so the state implementations stay terse.
"""

from __future__ import annotations

from typing import Any


def bb_get(bb, key: str, default: Any = None) -> Any:
    try:
        return bb.get(key)
    except Exception:
        return default
