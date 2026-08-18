from ..._aux._backends._symbolic_router import simplify
from ..._aux._backends._types_and_constants import expr_numeric_types


def _is_zero(x):
    if isinstance(x, expr_numeric_types()) and not isinstance(x, bool):
        return x == 0
    iz = getattr(x, "is_zero", None)
    if iz is True:
        return True
    if callable(iz):
        try:
            return bool(iz())
        except Exception:
            pass
    try:
        return simplify(x) == 0
    except Exception:
        return x == 0


def _as_zero_expr(eq):
    lhs = getattr(eq, "lhs", None)
    rhs = getattr(eq, "rhs", None)
    if callable(lhs) and callable(rhs):
        try:
            return lhs() - rhs()
        except Exception:
            pass
    if lhs is not None and rhs is not None and not callable(lhs) and not callable(rhs):
        return lhs - rhs
    return eq
