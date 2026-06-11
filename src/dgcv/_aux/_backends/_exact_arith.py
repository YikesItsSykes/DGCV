# backends/_exact_arith.py

import math
from numbers import Integral, Rational

from ._caches import _get_fast_scalar_types
from ._engine import _get_sage_module, _get_sympy_module, engine_kind
from ._polynomials import is_polynomial, poly_terms
from ._symbolic_router import _scalar_is_zero, ratio, subs
from ._types_and_constants import one, rational


def exact_reciprocal(x):
    if isinstance(x, _get_fast_scalar_types()):
        return one() / x

    if isinstance(x, Integral) and not isinstance(x, bool):
        return rational(1, x)

    if isinstance(x, Rational):
        try:
            return rational(x.numerator, x.denominator)
        except Exception:
            pass

    return one() / x


def factorial(n):
    f = getattr(n, "factorial", None)
    if f is not None:
        return f()

    if isinstance(n, Integral) and not isinstance(n, bool):
        return math.factorial(n)

    kind = engine_kind()

    if kind == "sympy":
        return _get_sympy_module().factorial(n)

    if kind == "sage":
        sage = _get_sage_module()
        return sage.factorial(n)

    return math.factorial(int(n))


def independent_equations(exprs, syms):
    seen = set()
    independent = set()
    import random

    pts = [{s: random.uniform(2, 1000) for s in syms} for _ in range(2)]

    for expr in exprs:
        key = _equation_key(expr, syms, pts)
        if key not in seen:
            seen.add(key)
            independent.add(expr)
    return independent


def _equation_key(expr, syms, pts):
    if is_polynomial(expr, syms):
        try:
            _, monoms, coeffs = poly_terms(expr, syms, assume_polynomial=True)
            ref = next((c for c in coeffs if not _scalar_is_zero(c)), None)
            if ref is not None:
                return tuple((m, ratio(c, ref)) for m, c in zip(monoms, coeffs))
        except Exception:
            pass

    vals = []
    for pt in pts:
        try:
            v = float(subs(expr, pt))
            vals.append(v)
        except Exception:
            return id(expr)
    ref = next((v for v in vals if abs(v) > 1e-10), None)
    if ref is None:
        return None
    return tuple(round(v / ref, 8) for v in vals)
