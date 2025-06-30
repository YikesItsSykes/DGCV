import numbers

import sympy as sp

from ._sage_backend import get_sage_module, is_sage_available

_expr_types_cache = None
_expr_numeric_types = None
_atomic_predicate = None

def _get_expr_types():
    global _expr_types_cache
    if _expr_types_cache is None:
        types = [sp.Expr]
        if is_sage_available():
            get_sage_module()
            from sage.symbolic.expression import Expression as SageExpression
            types.append(SageExpression)
        _expr_types_cache = tuple(types)
    return _expr_types_cache

def _get_expr_num_types():
    global _expr_numeric_types
    if _expr_numeric_types is None:
        _expr_numeric_types = (numbers.Number,)+_get_expr_types()
    return _expr_numeric_types

def _is_atomic(expr):
    global _atomic_predicate
    if _atomic_predicate is None:
        def sympy_atomic(elem):
            return isinstance(elem, sp.Basic) and bool(elem.is_Atom)
        if is_sage_available():
            get_sage_module()
            from sage.symbolic.expression import Expression as _SageExpr
            def atomic(elem):
                if isinstance(elem, _SageExpr):
                    return bool(elem.is_symbol())
                return sympy_atomic(elem)
        else:
            atomic = sympy_atomic
        _atomic_predicate = atomic
    return _atomic_predicate(expr)
