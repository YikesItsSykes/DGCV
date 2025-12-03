from itertools import combinations

import sympy as sp

from ._config import get_dgcv_settings_registry
from ._safeguards import check_dgcv_category
from .backends._caches import _get_expr_num_types
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
    eqns, vars_to_solve=None, verbose=False, method="auto",
    simplify_result=True, print_solve_stats=False
):
    import time
    t0 = time.perf_counter()

    def _log(*a):
        if print_solve_stats:
            print("[solve_dgcv]", *a)

    if isinstance(eqns,(list,tuple)) and len(eqns)==0:
        if isinstance(vars_to_solve, (list,tuple)):
            return [{var:var for var in vars_to_solve}]
        elif isinstance(vars_to_solve,_get_expr_num_types()):
            return [{vars_to_solve:vars_to_solve}]
        else:
            return [dict()]

    eqns, vars_to_solve = normalize_equations_and_vars(eqns, vars_to_solve)
    processed_eqns, system_vars, extra_vars, variables_dict = _equations_preprocessing(
        eqns, vars_to_solve
    )

    def _simplify(x):
        return x if not simplify_result else sp.simplify(x)

    def _expr_reformatting(expr):
        if isinstance(expr, (int, float)) or not hasattr(expr, "subs"):
            return expr
        dgcv_var_dict = {v[1][0]: v[0] for _, v in variables_dict.items()}
        if not isinstance(expr, sp.Expr) or isinstance(expr, zeroFormAtom):
            return expr.subs(dgcv_var_dict)
        regular_var_dict = {k: v for k, v in dgcv_var_dict.items() if isinstance(k, sp.Symbol)}
        if not all(isinstance(v, (int, float, sp.Expr)) for v in regular_var_dict.values()):
            return abstract_ZF(_sympy_to_abstract_ZF(expr, regular_var_dict))
        return expr.subs(regular_var_dict)

    def _extract_reformatting(var):
        return variables_dict[str(var)][0] if str(var) in variables_dict else var

    def _linsolve_to_dicts(solset, vars_):
        if not solset:
            return []
        out = []
        for tup in solset:
            tup = tuple(tup) if hasattr(tup, "__iter__") else ()
            if len(tup) == len(vars_):
                out.append(dict(zip(vars_, tup)))
        return out

    use_sage = (
        get_dgcv_settings_registry().get("default_symbolic_engine", "").lower()
        == "sage"
    )
    _log(f"engine={'sage' if use_sage else 'sympy'} method={method} #eqns={len(processed_eqns)} #vars={len(system_vars)}")

    preformatted_solutions = []

    if use_sage:
        orig_expr_vars = set(system_vars)
        for e in processed_eqns:
            orig_expr_vars |= get_free_symbols(e)
        sage = get_sage_module()
        symbol_map = {}
        for sym in orig_expr_vars:
            s = sage.var(str(sym), domain="real") if getattr(sym, "is_real", False) else sage.var(str(sym))
            symbol_map[sym] = s
        s_eqns = [sage.SR(str(e)) for e in processed_eqns]
        s_vars = [symbol_map[v] for v in system_vars]

        if method == "linsolve":
            try:
                A, b = sage.linear_equations_matrix(s_eqns, s_vars)
                _log(f"matrix_shape={A.nrows()}x{A.ncols()}")
                sol_vector = A.solve_right(b)
                sol_set = [dict(zip(s_vars, sol_vector))]
            except Exception:
                sol_set = []
        elif method == "solve":
            try:
                sol_set = sage.solve(s_eqns, s_vars, solution_dict=True)
            except Exception:
                sol_set = []
        elif method == "auto":
            try:
                A, b = sage.linear_equations_matrix(s_eqns, s_vars)
                _log(f"matrix_shape={A.nrows()}x{A.ncols()}")
                sol_vector = A.solve_right(b)
                sol_set = [dict(zip(s_vars, sol_vector))]
            except Exception:
                try:
                    sol_set = sage.solve(s_eqns, s_vars, solution_dict=True)
                except Exception:
                    sol_set = []
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'auto', 'linsolve', or 'solve'.")

        inv_map = {v_sage: sym for sym, v_sage in symbol_map.items()}
        for sol in sol_set:
            sol_py = {}
            for v_sage, val_sage in sol.items():
                sym = inv_map.get(v_sage, sp.sympify(str(v_sage), evaluate=False))
                val = sp.sympify(str(val_sage), evaluate=False)
                subs_map = {sp.Symbol(str(orig)): orig for orig in symbol_map.keys()}
                val = val.subs(subs_map)
                sol_py[sym] = val
            preformatted_solutions.append(sol_py)

    else:
        if method == "linsolve":
            try:
                sols = sp.linsolve(processed_eqns, tuple(system_vars))
                preformatted_solutions = _linsolve_to_dicts(sols, tuple(system_vars))
            except Exception:
                preformatted_solutions = []
        elif method == "solve":
            try:
                sols = sp.solve(processed_eqns, system_vars, dict=True)
                preformatted_solutions = sols if isinstance(sols, list) else ([sols] if isinstance(sols, dict) else [])
            except Exception:
                preformatted_solutions = []
        elif method == "auto":
            try:
                sols = sp.linsolve(processed_eqns, tuple(system_vars))
                preformatted_solutions = _linsolve_to_dicts(sols, tuple(system_vars))
            except Exception:
                preformatted_solutions = []
            if not preformatted_solutions:
                try:
                    sols = sp.solve(processed_eqns, system_vars, dict=True)
                    preformatted_solutions = sols if isinstance(sols, list) else ([sols] if isinstance(sols, dict) else [])
                except Exception:
                    preformatted_solutions = []
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'auto', 'linsolve', or 'solve'.")

    solutions_formatted = [
        {
            _extract_reformatting(var): _expr_reformatting(_simplify(expr))
            for var, expr in solution.items()
        }
        for solution in preformatted_solutions
    ]

    _log(f"solutions={len(solutions_formatted)} elapsed_s={time.perf_counter()-t0:.6f}")
    return (solutions_formatted, system_vars, extra_vars) if verbose else solutions_formatted

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
