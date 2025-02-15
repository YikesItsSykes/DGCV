import sympy as sp

from ._safeguards import get_variable_registry


def _holToReal(expr, skipVar=None, simplify_everything=True):
    """
    Converts holomorphic variables from VMF in SymPy expression to real variables.

    Parameters:
    expr : sympy.Expr
        The expression to convert.
    skipVar : list of str, optional
        A list of complex variable system labels to skip during conversion.
        For any variable in skipVar, the associated real and imaginary variables
        will not be substituted.

    Returns:
    sympy.Expr
    """
    variable_registry = get_variable_registry()
    conversion_dict = (
        variable_registry.get("conversion_dictionaries", {}).get("holToReal", {}).copy()
    )
    def format(expr):
        return sp.sympify(expr).subs(conversion_dict)

    # Skip specified holomorphic variable systems if skipVar is provided
    if skipVar:
        for var in skipVar:
            complex_system = variable_registry.get("complex_variable_systems", {}).get(
                var, {}
            )
            family_values = complex_system.get("family_values", ((), (), (), ()))
            if complex_system.get("family_type", None) == "single":
                family_values = tuple([(j,) for j in family_values])

            # Remove the holomorphic variables (first tuple) from the conversion_dict
            holomorphic_vars = family_values[0]
            for hol_var in holomorphic_vars:
                if hol_var in conversion_dict:
                    del conversion_dict[hol_var]


    if hasattr(expr, "applyfunc"):
        return expr.applyfunc(format)
    else:
        return format(expr)


def _realToSym(expr, skipVar=None, simplify_everything=True):
    """
    Converts real variables from VMF in SymPy expression to symbolic conjugates.

    Parameters:
    expr : sympy.Expr
        The expression to convert.
    skipVar : list of str, optional
        A list of complex variable system labels to skip during conversion.
        For any variable in skipVar, the associated real and imaginary variables
        will not be substituted.

    Returns:
    sympy.Expr
    """
    variable_registry = get_variable_registry()
    conversion_dict = (
        variable_registry.get("conversion_dictionaries", {}).get("realToSym", {}).copy()
    )

    def format(expr):
        return sp.sympify(expr).subs(conversion_dict)

    # Skip specified real and imaginary variable systems if skipVar is provided
    if skipVar:
        for var in skipVar:
            complex_system = variable_registry.get("complex_variable_systems", {}).get(
                var, {}
            )
            family_values = complex_system.get("family_values", ((), (), (), ()))
            if complex_system.get("family_type", None) == "single":
                family_values = tuple([(j,) for j in family_values])

            # Remove both the real (third tuple) and imaginary (fourth tuple) variables from the conversion_dict
            real_vars = family_values[2] + family_values[3]
            for real_var in real_vars:
                if real_var in conversion_dict:
                    del conversion_dict[real_var]

    if hasattr(expr, "applyfunc"):
        return expr.applyfunc(format)
    else:
        return format(expr)


def _symToHol(expr, skipVar=None, simplify_everything=True):
    """
    Converts symbolic conjugated variables from VMF in SymPy expression to holomorphic variables.

    Parameters:
    expr : sympy.Expr
        The expression to convert.
    skipVar : list of str, optional
        A list of complex variable system labels to skip during conversion.
        For any variable in skipVar, the associated real and imaginary variables
        will not be substituted.

    Returns:
    sympy.Expr
    """
    variable_registry = get_variable_registry()
    conversion_dict = (
        variable_registry.get("conversion_dictionaries", {}).get("symToHol", {}).copy()
    )

    def format(expr):
        return sp.sympify(expr).subs(conversion_dict)

    # If skipVar is provided, modify the conversion_dict to exclude specified variables.
    if skipVar:
        for var in skipVar:
            # Access the complex variable system for the skipped variable
            complex_system = variable_registry.get("complex_variable_systems", {}).get(
                var, {}
            )
            family_values = complex_system.get("family_values", ((), (), (), ()))
            if complex_system.get("family_type", None) == "single":
                family_values = tuple([(j,) for j in family_values])

            # The second tuple contains the antiholomorphic variables
            antiholomorphic_vars = family_values[1]

            # Remove the antiholomorphic variables from the conversion_dict
            for anti_var in antiholomorphic_vars:
                if anti_var in conversion_dict:
                    del conversion_dict[anti_var]

    if hasattr(expr, "applyfunc"):
        return expr.applyfunc(format)
    else:
        return format(expr)


def _holToSym(expr, skipVar=None, simplify_everything=True):
    """
    Converts holomorphic variables from VMF in an expression to symbolic 
    conjugates. This is done by first converting holomorphic variables to real 
    variables, and then converting real variables to symbolic conjugates.

    Note: This process will also convert any present real variables from the 
    VMF (both real and imaginary parts) to their symbolic conjugate format.

    Parameters:
    expr : sympy.Expr
        The expression to convert.
    skipVar : list of str, optional
        A list of complex variable system labels to skip during conversion.
        For any variable in skipVar, the associated real and imaginary variables
        will not be substituted.

    Returns:
    sympy.Expr
    """
    # First apply holToReal()
    expr = _holToReal(expr, skipVar=skipVar, simplify_everything=simplify_everything)

    # Then apply realToSym()
    expr = _realToSym(expr, skipVar=skipVar, simplify_everything=simplify_everything)

    return expr


def _realToHol(expr, skipVar=None, simplify_everything=True):
    """
    Converts real variables from VMF in SymPy expression to holomorphic variables.

    Parameters:
    expr : sympy.Expr
        The expression to convert.
    skipVar : list of str, optional
        A list of complex variable system labels to skip during conversion.
        For any variable in skipVar, the associated real and imaginary variables
        will not be substituted.

    Returns:
    sympy.Expr
    """
    variable_registry = get_variable_registry()
    conversion_dict = (
        variable_registry.get("conversion_dictionaries", {}).get("realToHol", {}).copy()
    )

    def format(expr):
        return sp.sympify(expr).subs(conversion_dict)

    # Skip specified real and imaginary variable systems if skipVar is provided
    if skipVar:
        for var in skipVar:
            complex_system = variable_registry.get("complex_variable_systems", {}).get(
                var, {}
            )
            family_values = complex_system.get("family_values", ((), (), (), ()))
            if complex_system.get("family_type", None) == "single":
                family_values = tuple([(j,) for j in family_values])

            # Remove both the real (third tuple) and imaginary (fourth tuple) variables from the conversion_dict
            real_vars = family_values[2] + family_values[3]
            for real_var in real_vars:
                if real_var in conversion_dict:
                    del conversion_dict[real_var]

    if hasattr(expr, "applyfunc"):
        return expr.applyfunc(format)
    else:
        return format(expr)


def _symToReal(expr, skipVar=None, simplify_everything=True):
    """
    Converts symbolic conjugates from VMF in SymPy expression to real variables.

    Parameters:
    expr : sympy.Expr
        The expression to convert.
    skipVar : list of str, optional
        A list of complex variable system labels to skip during conversion.
        For any variable in skipVar, the associated real and imaginary variables
        will not be substituted.

    Returns:
    sympy.Expr
    """
    variable_registry = get_variable_registry()
    conversion_dict = (
        variable_registry.get("conversion_dictionaries", {}).get("symToReal", {}).copy()
    )

    def format(expr):
        return sp.sympify(expr).subs(conversion_dict)

    # Skip specified symbolic conjugate systems if skipVar is provided
    if skipVar:
        for var in skipVar:
            complex_system = variable_registry.get("complex_variable_systems", {}).get(
                var, {}
            )
            family_values = complex_system.get("family_values", ((), (), (), ()))
            if complex_system.get("family_type", None) == "single":
                family_values = tuple([(j,) for j in family_values])

            # Remove the symbolic conjugates (second tuple) from the conversion_dict
            antiholomorphic_vars = family_values[1]
            for anti_var in antiholomorphic_vars:
                if anti_var in conversion_dict:
                    del conversion_dict[anti_var]

    if hasattr(expr, "applyfunc"):
        return expr.applyfunc(format)
    else:
        return format(expr)


def _allToReal(expr, skipVar=None, simplify_everything=True):
    """
    Converts all variables in VMF (holomorphic, symbolic conjugates, and real)
    to real variables.

    Parameters:
    expr : sympy.Expr
        The expression to convert.
    skipVar : list of str, optional
        A list of variable system labels to skip during conversion.

    Returns:
    sympy.Expr
    """
    # First convert symbolic conjugates to real variables
    expr = _symToReal(expr, skipVar=skipVar, simplify_everything=simplify_everything)

    # Then convert holomorphic variables to real variables
    expr = _holToReal(expr, skipVar=skipVar, simplify_everything=simplify_everything)

    return expr


def _allToHol(expr, skipVar=None, simplify_everything=True):
    """
    Converts all variables in VMF (real, symbolic conjugates, and holomorphic)
    to holomorphic variables.

    Parameters:
    expr : sympy.Expr
        The expression to convert.
    skipVar : list of str, optional
        A list of variable system labels to skip during conversion.

    Returns:
    sympy.Expr
    """

    if isinstance(expr, list):
        return [_allToHol(j) for j in expr]
    if isinstance(expr, tuple):
        return [_allToHol(j) for j in expr]

    # First convert symbolic conjugates to real variables
    expr = _symToHol(expr, skipVar=skipVar, simplify_everything=simplify_everything)

    # Then convert real variables to holomorphic variables
    expr = _realToHol(expr, skipVar=skipVar, simplify_everything=simplify_everything)

    return expr


def _allToSym(expr, skipVar=None, simplify_everything=True):
    """
    Converts all variables from VMF (holomorphic, real, and symbolic conjugates) to
    symbolic conjugates.

    Parameters:
    expr : sympy.Expr
        The expression to convert.
    skipVar : list of str, optional
        A list of variable system labels to skip during conversion.

    Returns:
    sympy.Expr
    """
    return _holToSym(expr, skipVar=skipVar, simplify_everything=simplify_everything)

