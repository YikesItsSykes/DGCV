# backends/_symbolic_router.py

import math
import numbers
from fractions import Fraction

from .._config import get_variable_registry
from ._engine import engine_kind, engine_module
from ._types_and_constants import constant_scalar_types, one, zero


def _scalar_is_zero(x) -> bool:
    if x is None:
        return False

    z = getattr(x, "is_zero", None)
    if z is True:
        return True
    if z is False:
        return False
    if callable(z):
        try:
            v = z()
            if isinstance(v, bool):
                return v
        except Exception:
            pass
    try:
        if isinstance(x, constant_scalar_types()) and not isinstance(x, bool):
            return x == 0
    except Exception:
        pass
    try:
        eq = x == 0
        return eq if isinstance(eq, bool) else False
    except Exception:
        return False


def get_free_symbols(expr):
    """
    Return the set of atomic elements in symbolic expr
    """
    if hasattr(expr, "free_symbols"):
        return expr.free_symbols

    if hasattr(expr, "variables"):
        return set(expr.variables())

    return set()


def simplify(expr, method=None, **kwargs):
    f = getattr(expr, "__dgcv_simplify__", None)
    if callable(f):
        return f(method=method, **kwargs)

    kind = engine_kind()

    if kind == "sympy":
        sp = engine_module()
        if method is None or method == "simplify":
            return sp.simplify(expr, **kwargs)
        fn = getattr(sp, method, None)
        if fn is None:
            raise ValueError(f"Unknown sympy simplify method {method!r}")
        return fn(expr, **kwargs)

    if kind == "sage":
        if method is None or method == "simplify":
            f = getattr(expr, "simplify_full", None)
            if f is not None:
                return f(**kwargs)
            f = getattr(expr, "simplify", None)
            if f is not None:
                return f(**kwargs)
            return expr
        f = getattr(expr, method, None)
        if f is None:
            raise ValueError(f"Unknown sage simplify method {method!r}")
        return f(**kwargs)

    return expr


def subs(expr, subs_data, **kwargs):
    if not subs_data:
        return expr
    f = getattr(expr, "subs", None)
    if f is None:
        return expr
    return f(subs_data, **kwargs)


def conjugate(expr):
    f = getattr(expr, "__dgcv_conjugate__", None)
    if callable(f):
        return f()

    kind = engine_kind()

    if kind == "sympy":
        f = getattr(expr, "conjugate", None)
        if f is not None:
            return f()
        return expr

    if kind == "sage":
        f = getattr(expr, "conjugate", None)
        if f is None:
            return expr

        c = f()

        registry = get_variable_registry()
        subs_map = registry.get("dgcv_enforced_real_atoms")
        if subs_map:
            g = getattr(c, "subs", None)
            if g is not None:
                return g(subs_map)
        return c

    f = getattr(expr, "conjugate", None)
    if f is not None:
        return f()
    return expr


def ratio(x, y=1):
    kind = engine_kind()

    if kind is None:
        if isinstance(x, (float, complex)) or isinstance(y, (float, complex)):
            return x / y
        if isinstance(x, Fraction) or isinstance(y, Fraction):
            return Fraction(x) / Fraction(y)
        if isinstance(x, numbers.Integral) and isinstance(y, numbers.Integral):
            return Fraction(int(x), int(y))
        return x / y

    eng = engine_module()

    if kind == "sympy":
        sp = eng

        if isinstance(x, sp.Float) or isinstance(y, sp.Float):
            return x / y
        if isinstance(x, (float, complex)) or isinstance(y, (float, complex)):
            return sp.sympify(x) / sp.sympify(y)

        if isinstance(x, numbers.Integral) and isinstance(y, numbers.Integral):
            return sp.Rational(int(x), int(y))

        if isinstance(x, (sp.Integer, sp.Rational)) and isinstance(
            y, (sp.Integer, sp.Rational)
        ):
            return x / y

        if isinstance(x, Fraction):
            x = sp.Rational(x.numerator, x.denominator)
        if isinstance(y, Fraction):
            y = sp.Rational(y.numerator, y.denominator)

        sx = x if isinstance(x, sp.Basic) else sp.sympify(x)
        sy = y if isinstance(y, sp.Basic) else sp.sympify(y)

        return sx / sy

    if kind == "sage":
        sage = eng
        if isinstance(x, (float, complex)) or isinstance(y, (float, complex)):
            return sage(x) / sage(y)
        return sage(x) / sage(y)

    return x / y


def re(expr):
    """
    Return the real part of expr in the active symbolic engine.
    """
    f = getattr(expr, "__dgcv_re__", None)
    if callable(f):
        return f()
    kind = engine_kind()

    if kind == "sympy":
        sp = engine_module()
        return sp.re(expr)

    if kind == "sage":
        f = getattr(expr, "real_part", None)
        if f is not None:
            return f()
        f = getattr(expr, "real", None)
        if f is not None and not callable(f):
            return f
        f = getattr(expr, "real", None)
        if callable(f):
            return f()
        try:
            return expr.real_part()
        except Exception:
            return expr

    z = getattr(expr, "real", None)
    if callable(z):
        return z()
    if z is not None:
        return z
    if isinstance(expr, numbers.Number):
        return expr.real
    return expr


def im(expr):
    """
    Return the imaginary part of expr in the active symbolic engine.
    """
    f = getattr(expr, "__dgcv_im__", None)
    if callable(f):
        return f()
    kind = engine_kind()

    if kind == "sympy":
        sp = engine_module()
        return sp.im(expr)

    if kind == "sage":
        f = getattr(expr, "imag_part", None)
        if f is not None:
            return f()
        f = getattr(expr, "imag", None)
        if f is not None and not callable(f):
            return f
        f = getattr(expr, "imag", None)
        if callable(f):
            return f()
        try:
            return expr.imag_part()
        except Exception:
            return zero()

    z = getattr(expr, "imag", None)
    if callable(z):
        return z()
    if z is not None:
        return z
    if isinstance(expr, numbers.Number):
        return expr.imag
    return zero()


def as_numer_denom(expr):
    """
    Return (numerator, denominator) for expr in the active symbolic engine.
    """
    kind = engine_kind()

    if kind == "sympy":
        f = getattr(expr, "as_numer_denom", None)
        if f is None:
            sp = engine_module()
            expr = sp.sympify(expr)
            return expr.as_numer_denom()
        return f()

    if kind == "sage":
        f = getattr(expr, "numerator", None)
        g = getattr(expr, "denominator", None)
        if callable(f) and callable(g):
            return f(), g()
        f = getattr(expr, "numerator", None)
        g = getattr(expr, "denominator", None)
        if callable(f) and callable(g):
            return f(), g()
        try:
            return expr.numerator(), expr.denominator()
        except Exception:
            return expr, one()

    f = getattr(expr, "as_numer_denom", None)
    if callable(f):
        return f()
    return expr, one()


def ilcm(*ints):
    """
    Integer least common multiple in the active symbolic engine.
    """
    ints = [int(x) for x in ints if x is not None]
    if not ints:
        return 1
    if len(ints) == 1:
        return ints[0]

    kind = engine_kind()

    if kind == "sympy":
        sp = engine_module()
        return sp.ilcm(*ints)

    if kind == "sage":
        sage = engine_module()
        fn = getattr(sage, "lcm", None)
        if fn is not None:
            return fn(ints)
        # fallback
        return math.lcm(*ints)

    return math.lcm(*ints)


def clear_denominators(seq, *, return_scale=False):
    """
    Multiply a sequence of scalars by the LCM of their denominators (when detectable),
    returning a new list.
    """
    if seq is None:
        return None

    denoms = []
    for x in seq:
        _, d = as_numer_denom(x)
        try:
            denoms.append(int(d))
        except Exception:
            pass

    L = ilcm(*denoms) if denoms else 1
    out = list(seq) if L == 1 else [L * x for x in seq]
    return (out, L) if return_scale else out


def expand(expr, **kwargs):
    """
    Expand expr using the active symbolic engine, intended as a backend hook for
    expand_dgcv (and polynomial expansion).
    """

    kind = engine_kind()
    f = getattr(expr, "__dgcv_expand__", getattr(expr, "__dgcv_apply__", None))
    if f:
        try:
            return f(expand, **kwargs)
        except TypeError:
            return f(expand)

    if kind == "sympy":
        sp = engine_module()
        return sp.expand(expr, **kwargs)

    if kind == "sage":
        # Sage has .expand() on symbolic expressions; if it wasn't callable above,
        # just return expr unchanged.
        return expr

    return expr


def factor(expr, **kwargs):
    """
    Factor expr using the active symbolic engine, intended as a backend hook for
    factor_dgcv (and polynomial factoring).
    """
    kind = engine_kind()

    f = getattr(expr, "__dgcv_apply__", None)
    if f:
        try:
            return f(factor, **kwargs)
        except TypeError:
            return f(factor)

    if kind == "sympy":
        sp = engine_module()
        return sp.factor(expr, **kwargs)

    if kind == "sage":
        # If Sage supports .factor() on the symbolic expressions/ring elements then
        # it was caught above already.
        return expr

    return expr


def cancel(expr, **kwargs):
    """
    Cancel common factors in a rational expression using the active symbolic engine,
    intended as a backend hook for cancel_dgcv (and rational simplification).
    """
    kind = engine_kind()

    f = getattr(expr, "__dgcv_apply__", None)
    if f:
        try:
            return f(cancel, **kwargs)
        except TypeError:
            return f(cancel)

    if kind == "sympy":
        sp = engine_module()
        return sp.cancel(expr, **kwargs)

    if kind == "sage":
        # no sage solution yet
        return expr

    return expr
