# dgcv/backends/_notebooks.py

from __future__ import annotations

import importlib.util
from typing import Any, Callable

from .._config import get_dgcv_settings_registry

_ipython_available = None
_ipython_shell = None
_ipython_shell_checked = False


def invalidate_notebook_cache() -> None:
    global _ipython_available, _ipython_shell, _ipython_shell_checked
    _ipython_available = None
    _ipython_shell = None
    _ipython_shell_checked = False


def is_ipython_available() -> bool:
    global _ipython_available
    if _ipython_available is not None:
        return _ipython_available
    try:
        _ipython_available = importlib.util.find_spec("IPython") is not None
    except Exception:
        _ipython_available = False
    return _ipython_available


def _get_ipython_shell():
    global _ipython_shell, _ipython_shell_checked
    if _ipython_shell_checked:
        return _ipython_shell
    _ipython_shell_checked = True
    if not is_ipython_available():
        _ipython_shell = None
        return None
    try:
        from IPython import get_ipython

        _ipython_shell = get_ipython()
    except Exception:
        _ipython_shell = None
    return _ipython_shell


def in_ipython() -> bool:
    return _get_ipython_shell() is not None


def _format_displays_override() -> bool:
    try:
        dgcvSR = get_dgcv_settings_registry()
        return bool(dgcvSR.get("format_displays", False))
    except Exception:
        return False


def in_notebook() -> bool:
    sh = _get_ipython_shell()
    if sh is None:
        return False

    name = getattr(getattr(sh, "__class__", None), "__name__", "")
    return name == "ZMQInteractiveShell"


def can_rich_display_latex() -> bool:
    if not is_ipython_available():
        return False
    return in_notebook()


def register_latex_formatter_for_type(tp: type, fn: Callable[[Any], str]) -> bool:
    sh = _get_ipython_shell()
    if sh is None:
        return False
    try:
        fmt = sh.display_formatter.formatters.get("text/latex", None)
        if fmt is None:
            return False
        fmt.for_type(tp, fn)
        return True
    except Exception:
        return False


def display_latex(s: str) -> bool:
    if not is_ipython_available():
        return False
    try:
        from IPython.display import Latex, display

        display(Latex(s))
        return True
    except Exception:
        return False


def display_html(s: str) -> bool:
    if not is_ipython_available():
        return False
    try:
        from IPython.display import HTML, display

        display(HTML(s))
        return True
    except Exception:
        return False
