# backends/_numeric_engines.py

from __future__ import annotations

import importlib
import importlib.util
import warnings

from .._config import get_dgcv_settings_registry

_numeric_kind = None
_numpy_module = None


def is_numpy_available() -> bool:
    try:
        return importlib.util.find_spec("numpy") is not None
    except (ImportError, ModuleNotFoundError):
        return False


def _get_numpy_module():
    global _numpy_module
    if _numpy_module is not None:
        return _numpy_module
    if not is_numpy_available():
        raise RuntimeError("NumPy is not available in the current environment.")
    _numpy_module = importlib.import_module("numpy")
    return _numpy_module


def numpy_module_if_available():
    if not is_numpy_available():
        return None
    try:
        return _get_numpy_module()
    except Exception:
        return None


def available_numeric_kinds():
    out = []
    if is_numpy_available():
        out.append("numpy")
    return tuple(out)


def invalidate_numeric_cache():
    global _numeric_kind, _numpy_module
    _numeric_kind = None
    _numpy_module = None


def _resolve_numeric_kind():
    settings = get_dgcv_settings_registry()
    requested = str(settings.get("default_numeric_engine", "numpy")).lower()

    if requested in ("np",):
        requested = "numpy"

    if requested == "numpy":
        if is_numpy_available():
            return "numpy"
        warnings.warn(
            "dgcv: default numeric engine setting is 'numpy' but NumPy is not available; "
            "numeric methods will be disabled.",
            stacklevel=2,
        )
        return None

    if is_numpy_available():
        warnings.warn(
            f"dgcv: unrecognized numeric engine {requested!r}; falling back to 'numpy'.",
            stacklevel=2,
        )
        return "numpy"

    warnings.warn(
        f"dgcv: unrecognized numeric engine {requested!r}, "
        "and no supported numeric engine is available.",
        stacklevel=2,
    )
    return None


def numeric_kind():
    global _numeric_kind
    if _numeric_kind is None:
        _numeric_kind = _resolve_numeric_kind()
    return _numeric_kind


def numeric_module():
    kind = numeric_kind()
    if kind is None:
        return None
    if kind == "numpy":
        return _get_numpy_module()
    return None
