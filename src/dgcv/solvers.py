from itertools import combinations

import sympy as sp

from ._config import _cached_caller_globals, get_dgcv_settings_registry
from ._safeguards import check_dgcv_category
from .backends._sage_backend import get_sage_module
from .backends._symbolic_api import get_free_symbols
from .eds.eds import (
    _equation_formatting,
    _sympy_to_abstract_ZF,
    abstract_ZF,
    zeroFormAtom,
)
from .eds.eds_representations import DF_representation


def simplify_dgcv(obj,aggressive=False):
    if check_dgcv_category(obj) and hasattr(obj, "_eval_simplify"):
        return obj._eval_simplify(aggressive=aggressive)
    elif isinstance(obj, sp.Basic):
        return sp.simplify(obj)
    else:
        return obj


def normalize_equations_and_vars(eqns, vars_to_solve):
    if isinstance(eqns, DF_representation):
        eqns = eqns.flatten()
    if not isinstance(eqns, (list, tuple)):
        eqns = [eqns]
    if vars_to_solve is None:
        vars_to_solve = set()
        for eqn in eqns:
            if hasattr(eqn, "free_symbols"):
                vars_to_solve |= eqn.free_symbols
    if isinstance(vars_to_solve, set):
        vars_to_solve = list(vars_to_solve)
    if not isinstance(vars_to_solve, (list, tuple)):
        vars_to_solve = [vars_to_solve]
    return eqns, vars_to_solve


def solve_carefully(eqns, vars_to_solve, dict=True):
    """
    Recursively applies sympy.solve() to handle underdetermined systems.
    If solve() fails due to "no valid subset found", it tries solving for smaller subsets of variables.

    Parameters:
    - eqns: list of sympy equations
    - vars_to_solve: list/tuple of variables to solve for
    - dict: whether to return solutions as a dictionary (default: True)

    Returns:
    - Solution from sympy.solve() if found
    - Otherwise, tries smaller variable subsets recursively
    - Raises NotImplementedError if no subset can be solved
    """

    try:
        # First, try to solve normally
        sol = sp.solve(eqns, vars_to_solve, dict=dict)
        if sol:  # Return only if non-empty
            return sol
    except NotImplementedError as e:
        if "no valid subset found" not in str(e):
            raise  # Re-raise other errors

    # If solve() fails, or returned an empty solution, try smaller subsets of variables
    num_vars = len(vars_to_solve)

    if num_vars == 1:
        raise NotImplementedError(
            "No valid subset found, even at minimal variable count."
        )

    # Try subsets with one fewer variable
    subset_list = list(combinations(vars_to_solve, num_vars - 1))
    for i, subset in enumerate(subset_list):
        try:
            sol = solve_carefully(eqns, subset, dict=dict)
            if (
                sol or i == len(subset_list) - 1
            ):  # Only return if non-empty or last subset
                return sol
        except NotImplementedError:
            continue  # Try the next subset

    # If no subset worked, raise the error
    raise NotImplementedError(f"No valid subset found for variables {vars_to_solve}")


def solve_dgcv(
    eqns, vars_to_solve=None, verbose=False, method="auto", simplify_result=True
):
    """
    Solve a dgcv-compatible system of equations.

    Parameters:
        eqns: One or more SymPy equations.
        vars_to_solve: Variables to solve for (inferred if None).
        verbose: If True, also return system_vars and extra_vars.
        method: One of 'auto', 'linsolve', or 'solve'.
        simplify_result: Whether to simplify the final results.

    Returns:
        List of dictionaries mapping variables to solutions.
        If verbose=True, returns (solutions, system_vars, extra_vars).
    """
    eqns, vars_to_solve = normalize_equations_and_vars(eqns, vars_to_solve)
    processed_eqns, system_vars, extra_vars, variables_dict = _equations_preprocessing(
        eqns, vars_to_solve
    )

    use_sage = (
        get_dgcv_settings_registry().get("default_symbolic_engine", "").lower()
        == "sage"
    )
    if use_sage:
        orig_expr_vars = set(system_vars)
        for eqn in processed_eqns:
            orig_expr_vars |= get_free_symbols(eqn)
        # Build Sage symbol map
        sage = get_sage_module()
        symbol_map = {}
        for sym in orig_expr_vars:
            if getattr(sym, "is_real", False):
                s = sage.var(str(sym), domain="real")
            else:
                s = sage.var(str(sym))
            symbol_map[sym] = s
        # Convert SymPy eqns and vars into Sage
        s_eqns = [sage.SR(str(e)) for e in processed_eqns]
        s_vars = [symbol_map[v] for v in system_vars]

    preformatted_solutions = []
    if use_sage:
        # Sage-first strategy: atomic conversion already done above
        if method in ("linsolve", "auto"):
            try:
                A, b = sage.linear_equations_matrix(s_eqns, s_vars)
                sol_vector = A.solve_right(b)
                sol_set = [dict(zip(s_vars, sol_vector))]
            except Exception:
                sol_set = []
            if method == "auto" and not sol_set:
                sol_set = sage.solve(s_eqns, s_vars, solution_dict=True)
        elif method == "solve":
            sol_set = sage.solve(s_eqns, s_vars, solution_dict=True)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Use 'auto', 'linsolve', or 'solve'."
            )
        # Convert Sage solutions back to SymPy
        inv_map = {v_sage: sym for sym, v_sage in symbol_map.items()}
        for sol in sol_set:
            sol_py = {}
            for v_sage, val_sage in sol.items():
                # key: original SymPy symbol
                sym = inv_map.get(v_sage, sp.sympify(str(v_sage), evaluate=False))
                # value: convert to SymPy and then replace atoms
                val = sp.sympify(str(val_sage), evaluate=False)
                # subs fresh symbols back to original ones
                subs_map = {sp.Symbol(str(orig)): orig for orig in symbol_map.keys()}
                val = val.subs(subs_map)
                sol_py[sym] = val
            preformatted_solutions.append(sol_py)
    else:
        # SymPy-first strategy (unchanged)
        if method == "linsolve":
            try:
                _cached_caller_globals["DEBUG"] = [processed_eqns, processed_eqns]
                sol_set = sp.linsolve(processed_eqns, *processed_eqns)
                _cached_caller_globals["DEBUG2"] = ("linsolve", sol_set)
                if sol_set:
                    try:
                        sol_tuple = next(iter(sol_set))
                        if hasattr(sol_tuple, "__len__") and len(sol_tuple) == len(
                            system_vars
                        ):
                            preformatted_solutions = [
                                dict(zip(system_vars, tuple(sol_tuple)))
                            ]
                        else:
                            preformatted_solutions = []
                    except Exception:
                        preformatted_solutions = []
                else:
                    preformatted_solutions = []
            except Exception:
                preformatted_solutions = []
        elif method == "solve":
            try:
                preformatted_solutions = sp.solve(
                    processed_eqns, system_vars, dict=True
                )
                _cached_caller_globals["DEBUG2"] = ("solve", preformatted_solutions)
            except Exception:
                preformatted_solutions = []
        elif method == "auto":
            try:
                sol_set = sp.linsolve(processed_eqns, *system_vars)
                if sol_set:
                    sol_tuple = next(iter(sol_set))
                    _cached_caller_globals["DEBUG2"] = ("auto-linsolve", sol_set)
                    _cached_caller_globals["DEBUG3"] = {
                        "sol_tuple": sol_tuple,
                        "system_vars": system_vars,
                        "tuple_length": len(sol_tuple),
                        "vars_length": len(system_vars),
                        "zipped": list(zip(system_vars, sol_tuple)),
                    }
                    if isinstance(sol_tuple, (tuple, list)) or hasattr(
                        sol_tuple, "__iter__"
                    ):
                        sol_tuple = tuple(sol_tuple)
                        if len(sol_tuple) == len(system_vars):
                            preformatted_solutions = [dict(zip(system_vars, sol_tuple))]
                        else:
                            preformatted_solutions = []
                    else:
                        preformatted_solutions = []
                else:
                    _cached_caller_globals["DEBUG3"] = "sol_set was empty"
                    preformatted_solutions = []
            except Exception as e:
                _cached_caller_globals["DEBUG3"] = f"exception: {e}"
                preformatted_solutions = []
            if not preformatted_solutions:
                try:
                    preformatted_solutions = sp.solve(
                        processed_eqns, system_vars, dict=True
                    )
                    if preformatted_solutions:
                        _cached_caller_globals["DEBUG2"] = (
                            "auto-solve",
                            preformatted_solutions,
                        )
                except Exception:
                    preformatted_solutions = []
        else:
            raise ValueError(
                f"Unknown method '{method}'. Use 'auto', 'linsolve', or 'solve'."
            )

    # Reformat solutions back into dgcv form
    solutions_formatted = []

    for solution in preformatted_solutions:

        def extract_reformatting(var):
            return variables_dict[str(var)][0] if str(var) in variables_dict else var

        def expr_reformatting(expr):
            if isinstance(expr, (int, float)) or not hasattr(expr, "subs"):
                return expr
            dgcv_var_dict = {v[1][0]: v[0] for _, v in variables_dict.items()}
            if not isinstance(expr, sp.Expr) or isinstance(expr, zeroFormAtom):
                return expr.subs(dgcv_var_dict)
            regular_var_dict = {
                k: v for k, v in dgcv_var_dict.items() if isinstance(k, sp.Symbol)
            }
            if not all(
                isinstance(v, (int, float, sp.Expr)) for v in regular_var_dict.values()
            ):
                return abstract_ZF(_sympy_to_abstract_ZF(expr, regular_var_dict))
            return expr.subs(regular_var_dict)

        solutions_formatted.append(
            {
                extract_reformatting(var): expr_reformatting(
                    expr if not simplify_result else sp.simplify(expr)
                )
                for var, expr in solution.items()
            }
        )

    if verbose:
        return solutions_formatted, system_vars, extra_vars
    else:
        return solutions_formatted


def _equations_preprocessing(eqns: tuple | list, vars: tuple | list):
    processed_eqns = []
    variables_dict = dict()
    for eqn in eqns:
        eqn_formatted, new_var_dict = _equation_formatting(eqn, variables_dict)
        processed_eqns += eqn_formatted
        variables_dict = variables_dict | new_var_dict
    subbedValues = {variables_dict[k][0]: variables_dict[k][1] for k in variables_dict}
    pre_system_vars = [
        subbedValues[var] if var in subbedValues else var for var in vars
    ]
    system_vars = []
    extra_vars = []
    for var in pre_system_vars:
        if isinstance(var, (list, tuple)) and len(var) == 1:
            var = var[0]
        if isinstance(var, sp.Symbol):
            system_vars += [var]
        else:
            extra_vars += [var]
    return processed_eqns, system_vars, extra_vars, variables_dict
