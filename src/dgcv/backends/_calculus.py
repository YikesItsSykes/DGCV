# backends/_calculus.py
from numbers import Integral

from ._engine import _get_sage_module, _get_sympy_module, engine_kind, engine_module
from ._types_and_constants import constant_scalar_types, zero


def diff(expr, *args, **kwargs):
    if not args:
        return expr

    if isinstance(expr, constant_scalar_types()) and not isinstance(expr, bool):
        return zero()

    f = getattr(expr, "diff", None)
    if f is not None:
        return f(*args, **kwargs)

    d = getattr(expr, "derivative", None)
    if d is not None:
        res = expr
        i = 0
        n = len(args)
        while i < n:
            v = args[i]
            if i + 1 < n and isinstance(args[i + 1], Integral):
                order = int(args[i + 1])
                res = res.derivative(v, order)
                i += 2
            else:
                res = res.derivative(v)
                i += 1
        return res

    kind = engine_kind()
    if kind == "sympy":
        return _get_sympy_module().diff(expr, *args, **kwargs)

    if kind == "sage":
        sage = _get_sage_module()
        fn = getattr(sage, "diff", None)
        if fn is not None:
            return fn(expr, *args, **kwargs)

    raise TypeError(
        f"Object of type {type(expr).__name__} does not support differentiation"
    )


def integrate(expr, *args, **kwargs):
    """
    Integrate expr using the active symbolic engine, intended as a backend hook for
    integrate_dgcv (and polynomial integration).
    """
    kind = engine_kind()

    f = getattr(expr, "integrate", None)
    if callable(f):
        try:
            return f(*args, **kwargs)
        except TypeError:
            return f(*args)

    if kind == "sympy":
        sp = engine_module()
        return sp.integrate(expr, *args, **kwargs)

    if kind == "sage":
        # Supposing Sage supports .integral() / .integrate() depending on type;
        g = getattr(expr, "integral", None)
        if callable(g):
            try:
                return g(*args, **kwargs)
            except TypeError:
                return g(*args)
        return expr

    return expr
