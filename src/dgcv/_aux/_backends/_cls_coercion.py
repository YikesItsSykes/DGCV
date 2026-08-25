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


_legacy_sympy_classes: list = []
_legacy_converters_attached = False


def register_legacy_sympy_class(cls: type) -> None:
    if cls in _legacy_sympy_classes:
        return
    _legacy_sympy_classes.append(cls)
    if _legacy_converters_attached:
        _install_legacy_converter(cls)


_legacy_originals: dict = {}


def _is_legacy_obj(x) -> bool:
    if not _legacy_sympy_classes:
        return False
    return isinstance(x, tuple(_legacy_sympy_classes))


def _legacy_simplify(x, orig, rest, kwargs):
    f = getattr(x, "_eval_simplify", None)
    if not callable(f):
        from ._symbolic_router import simplify as routed_simplify

        return routed_simplify(x)
    try:
        import inspect

        bound = inspect.signature(orig).bind(x, *rest, **kwargs)
        bound.apply_defaults()
        opts = dict(bound.arguments)
        opts.pop(next(iter(inspect.signature(orig).parameters)), None)
        extra = opts.pop("kwargs", {}) or {}
        return f(**opts, **extra)
    except Exception:
        return f(**kwargs)


def _legacy_conjugate(x, orig, rest, kwargs):
    f = getattr(x, "_eval_conjugate", None)
    if callable(f):
        return f()
    from ._symbolic_router import conjugate as routed_conjugate

    return routed_conjugate(x)


_LEGACY_DISPATCH = {
    "simplify": _legacy_simplify,
    "conjugate": _legacy_conjugate,
}


def _clear_sympy_cache() -> None:
    try:
        from sympy.core.cache import clear_cache  # type: ignore

        clear_cache()
    except Exception:
        pass


def _make_legacy_wrapper(orig, routed):
    def wrapper(*args, **kwargs):
        if args and _is_legacy_obj(args[0]):
            return routed(args[0], orig, args[1:], kwargs)
        return orig(*args, **kwargs)

    wrapper.__name__ = getattr(orig, "__name__", "wrapper")
    wrapper.__doc__ = getattr(orig, "__doc__", None)
    wrapper._dgcv_legacy_wrapper = True
    return wrapper


def _install_legacy_converter(cls: type) -> bool:
    return False


def attach_legacy_sympy_converters() -> None:
    global _legacy_converters_attached
    if _legacy_converters_attached:
        return
    if not is_sympy_available():
        return
    try:
        sp = _get_sympy_module()
    except Exception:
        return
    for name, routed in _LEGACY_DISPATCH.items():
        orig = getattr(sp, name, None)
        if orig is None or getattr(orig, "_dgcv_legacy_wrapper", False):
            continue
        _legacy_originals[name] = orig
        setattr(sp, name, _make_legacy_wrapper(orig, routed))
    _legacy_converters_attached = True
    _clear_sympy_cache()


def detach_legacy_sympy_converters() -> None:
    global _legacy_converters_attached
    if not _legacy_converters_attached:
        return
    _legacy_converters_attached = False
    if not is_sympy_available():
        return
    try:
        sp = _get_sympy_module()
    except Exception:
        return
    for name, orig in list(_legacy_originals.items()):
        if getattr(getattr(sp, name, None), "_dgcv_legacy_wrapper", False):
            setattr(sp, name, orig)
        _legacy_originals.pop(name, None)
    _clear_sympy_cache()


def _dgcv_sympy_hook(self) -> Any:
    from dgcv._aux._backends._engine import engine_kind

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
