from ..._aux._backends._symbolic_router import (
    _scalar_is_zero,
    is_zero_knowing_zero_is_expected,
)


def _is_zero(x):
    return _scalar_is_zero(x)


def _is_zero_after_simplify(x):
    return is_zero_knowing_zero_is_expected(x)


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
