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

from ._notebooks import in_notebook, invalidate_notebook_cache

__all__ = ["is_rich_displaying_available", "invalidate_display_engine_cache"]

_rich_display_available: bool | None = None


def invalidate_display_engine_cache():
    global _rich_display_available
    _rich_display_available = None
    invalidate_notebook_cache()


def is_rich_displaying_available() -> bool:
    global _rich_display_available
    if _rich_display_available is not None:
        return _rich_display_available
    _rich_display_available = bool(in_notebook())
    return _rich_display_available
