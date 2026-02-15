"""
package: dgcv - Differential Geometry with Complex Variables
module: backends/_display_engine.py

Description:
    Utilities for detecting whether dgcv is running in a rich-display
    environment (e.g. Jupyter/IPython kernel) as opposed to a plain
    terminal session.

Author (of this module): David Sykes (https://realandimaginary.com/dgcv/)

License:
    MIT License
"""

from __future__ import annotations

from .._config import get_dgcv_settings_registry
from ._notebooks import (
    in_notebook,
    invalidate_notebook_cache,
)

__all__ = ["is_rich_displaying_available", "invalidate_display_engine_cache"]

_rich_display_available: bool | None = None


def invalidate_display_engine_cache():
    global _rich_display_available
    _rich_display_available = None
    invalidate_notebook_cache()


def _force_rich_display_enabled() -> bool:
    try:
        settings = get_dgcv_settings_registry()
        return bool(settings.get("force_rich_display", False))
    except Exception:
        return False


def is_rich_displaying_available() -> bool:
    global _rich_display_available

    if _force_rich_display_enabled():
        return True

    if _rich_display_available is not None:
        return _rich_display_available

    _rich_display_available = bool(in_notebook())
    return _rich_display_available
