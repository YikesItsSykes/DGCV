def get_free_symbols(expr):
    """
    Return the set of atomic elements in symbolic expr
    """
    # SymPy case
    if hasattr(expr, "free_symbols"):
        return expr.free_symbols

    # Sage case
    if hasattr(expr, "variables"):
        return set(expr.variables())

    # Fallback: coerce to SymPy
    try:
        sym = expr._sympy_()
        return sym.free_symbols
    except Exception:
        raise TypeError(f"Cannot extract symbols from expression {expr!r}")
