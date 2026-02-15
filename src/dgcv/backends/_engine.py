"""
package: dgcv - Differential Geometry with Complex Variables
module: backends/_engine.py

Description: manages dgcv's interfacing with available CAS libraries.

Author (of this module): David Sykes (https://realandimaginary.com/dgcv/)

License:
    MIT License
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
import importlib
import importlib.util
import warnings

from .._config import get_dgcv_settings_registry

__all__ = [
    "is_sage_available",
    "is_sympy_available",
    "sage_module_if_available",
    "sympy_module_if_available",
    "available_engine_kinds",
    "engine_module",
    "engine_kind",
    "invalidate_engine_cache",
]


# -----------------------------------------------------------------------------
# utilities
# -----------------------------------------------------------------------------
_engine_kind = None
_engine_module = None
_sympy_module = None
_sage_module = None
_sage_available = None


def is_sage_available():
    global _sage_available
    if _sage_available is not None:
        return _sage_available
    try:
        spec = importlib.util.find_spec("sage.all")
        _sage_available = spec is not None
    except (ImportError, ModuleNotFoundError):
        _sage_available = False
    return _sage_available


def _get_sage_module():
    global _sage_module
    if _sage_module is not None:
        return _sage_module
    if not is_sage_available():
        raise RuntimeError("Sage is not available in the current environment.")
    _sage_module = importlib.import_module("sage.all")
    return _sage_module


def sage_module_if_available():
    if not is_sage_available():
        return None
    try:
        return _get_sage_module()
    except Exception:
        return None


def is_sympy_available():
    try:
        return importlib.util.find_spec("sympy") is not None
    except (ImportError, ModuleNotFoundError):
        return False


def _get_sympy_module():
    global _sympy_module
    if _sympy_module is not None:
        return _sympy_module
    if not is_sympy_available():
        raise RuntimeError("SymPy is not available in the current environment.")
    _sympy_module = importlib.import_module("sympy")
    return _sympy_module


def sympy_module_if_available():
    if not is_sympy_available():
        return None
    try:
        return _get_sympy_module()
    except Exception:
        return None


def available_engine_kinds():
    out = []
    if is_sage_available():
        out.append("sage")
    if is_sympy_available():
        out.append("sympy")
    return tuple(out)


def invalidate_engine_cache():
    global \
        _engine_kind, \
        _engine_module, \
        _one_obj, \
        _zero_obj, \
        _fast_scalar_types, \
        _expr_types, \
        _expr_numeric_types, \
        _atomic_pred, \
        _I_obj, \
        _constant_scalar_types
    _engine_kind = None
    _engine_module = None
    _fast_scalar_types = None
    _expr_types = None
    _expr_numeric_types = None
    _atomic_pred = None
    _one_obj = None
    _zero_obj = None
    _I_obj = None
    _constant_scalar_types = None
    try:
        from ._cls_coercion import invalidate_cls_coercion_cache

        invalidate_cls_coercion_cache()
    except Exception:
        pass


def _resolve_engine_kind():
    settings = get_dgcv_settings_registry()
    requested = str(settings.get("default_symbolic_engine", "sympy")).lower()

    if requested in ("sagemath",):
        requested = "sage"

    if requested == "sage":
        if is_sage_available():
            return "sage"
        if is_sympy_available():
            warnings.warn(
                "dgcv: default symbolic engine setting is 'sage' but Sage is not available; "
                "falling back to 'sympy'.",
                stacklevel=2,
            )
            return "sympy"
        warnings.warn(
            "dgcv: no symbolic engine was found, and dgcv requires one (either Sage or Sympy)",
            stacklevel=2,
        )
        return None

    if requested == "sympy":
        if is_sympy_available():
            return "sympy"
        if is_sage_available():
            warnings.warn(
                "dgcv: default symbolic engine setting is 'sympy' but SymPy is not available; "
                "falling back to 'sage'.",
                stacklevel=2,
            )
            return "sage"
        warnings.warn(
            "dgcv: no symbolic engine was found, and dgcv requires one (either Sage or Sympy)",
            stacklevel=2,
        )
        return None

    if is_sympy_available():
        warnings.warn(
            f"dgcv: unrecognized symbolic engine {requested!r}; falling back to 'sympy'.",
            stacklevel=2,
        )
        return "sympy"

    if is_sage_available():
        warnings.warn(
            f"dgcv: unrecognized symbolic engine {requested!r}; falling back to 'sage'.",
            stacklevel=2,
        )
        return "sage"

    warnings.warn(
        f"dgcv: unrecognized symbolic engine {requested!r}, "
        "and no supported symbolic engine is available.",
        stacklevel=2,
    )
    return None


def engine_kind():
    global _engine_kind
    if _engine_kind is None:
        _engine_kind = _resolve_engine_kind()
    return _engine_kind


def engine_module():
    global _engine_module
    kind = engine_kind()
    if kind is None:
        return None
    if _engine_module is not None:
        return _engine_module
    if kind == "sage":
        _engine_module = _get_sage_module()
    else:
        _engine_module = _get_sympy_module()
    return _engine_module
