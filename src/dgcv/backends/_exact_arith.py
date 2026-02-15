# backends/_exact_arith.py

import math
from numbers import Integral, Rational

from ._caches import _get_fast_scalar_types
from ._engine import _get_sage_module, _get_sympy_module, engine_kind
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
