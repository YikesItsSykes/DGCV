# src/dgcv/backends/_cls_coersion.py

from __future__ import annotations

from typing import Any, Optional, Type

from ._engine import _get_sympy_module, is_sympy_available

_sympified_cls: Optional[Type[Any]] = None
_sympy_hook_marker = "_dgcv_sympy_hook_attached"


def invalidate_cls_coercion_cache() -> None:
    global _sympified_cls
    _sympified_cls = None


def sympify_dgcv_obj(obj: Any):
    cls = _get_sympified_cls()
    return cls(obj)


def _get_sympified_cls():
    global _sympified_cls
    if _sympified_cls is not None:
        return _sympified_cls

    if not is_sympy_available():
        raise ImportError("SymPy is not available.")

    sp = _get_sympy_module()

    class sympified_dgcv(sp.Basic):
        __slots__ = ("dgcv_obj",)

        def __new__(cls, dgcv_obj):
            o = sp.Basic.__new__(cls)
            o.dgcv_obj = dgcv_obj
            return o

        def doit(self, **hints):
            from ._symbolic_router import simplify as simplify_dgcv

            return simplify_dgcv(self.dgcv_obj)

        def _eval_simplify(self, **kwargs):
            return self.doit()

        def _latex(self, printer=None):
            f = getattr(self.dgcv_obj, "_latex", None)
            if callable(f):
                try:
                    s = f(raw=True)
                    if isinstance(s, str):
                        return s
                except Exception:
                    pass

            f = getattr(self.dgcv_obj, "_repr_latex_", None)
            if callable(f):
                try:
                    s = f(raw=False)
                    if isinstance(s, str):
                        return s.strip("$")
                except Exception:
                    pass

            return str(self.dgcv_obj)

        def _sympystr(self, printer):
            return str(self.dgcv_obj)

    _sympified_cls = sympified_dgcv
    return _sympified_cls


def _dgcv_sympy_hook(self) -> Any:
    from dgcv.backends._engine import engine_kind

    if engine_kind() != "sympy":
        raise AttributeError

    return sympify_dgcv_obj(self)


def attach_sympy_hook(cls: type) -> None:
    if getattr(cls, _sympy_hook_marker, False):
        return

    if "_sympy_" in cls.__dict__:
        return

    setattr(cls, "_sympy_", _dgcv_sympy_hook)
    setattr(cls, _sympy_hook_marker, True)


def detach_sympy_hook(cls: type) -> None:
    if not getattr(cls, _sympy_hook_marker, False):
        return

    try:
        delattr(cls, "_sympy_")
    except Exception:
        pass

    try:
        delattr(cls, _sympy_hook_marker)
    except Exception:
        pass
