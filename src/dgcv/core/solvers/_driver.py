from ..._aux._backends._engine import engine_kind, engine_module
from ..._aux._backends._polynomials import expr_union_primitives
from ..._aux._backends._symbolic_router import get_free_symbols, simplify
from ..._aux._backends._types_and_constants import (
    expr_numeric_types,
    expr_types,
    is_atomic,
)
from ..._aux._utilities._config import get_dgcv_settings_registry
from ..._aux._vmf.vmf import order_coordinates
from ...eds.eds import _sympy_to_abstract_ZF, abstract_ZF, zeroFormAtom
from ._linsolve import _dgcv_linsolve
from ._normalization import _equations_preprocessing, normalize_equations_and_vars
from ._solution_shapes import _engine_solve_to_dicts, _linsolve_to_dicts


def solve_dgcv(
    eqns,
    vars_to_solve=None,
    verbose=False,
    method="auto",
    simplify_result=True,
    print_solve_stats=False,
    return_divisors=False,
    pass_to_symbolic_engine=None,
    sample_if_overdetermined=False,
    allow_underdetermined_solution=True,
    simplify_pivots=False,
    surface_singularities=False,
):
    """
    Solve an equation system for the given variables.

    Parameters
    ----------
    eqns : object
        symbolic expression, list of expressions, or supported dgcv objects.
        Interpretted as equations by setting them equal to zero.
    vars_to_solve : object, optional
        Variable or list of variables w.r.t. which the system is solved.
        Defaults to all free symbols in `eqns`.
    method : {'auto', 'linsolve', 'solve'}, default 'auto'
        Solver dispatch mode. 'auto' resolves against version specific
        defaults in dgcv settings (remaining as a dynamically interpretted
        'auto' in most recent versions of dgcv)
    surface_singularities : bool, default False
        Assume a linear system and use dgcv's own solver, forcing
        `simplify_pivots` and `return_divisors` to True and normalizing the
        returned divisors.

    Returns
    -------
    list of dict
        Solutions, empty when none exists. Paired with a divisor list when
        `return_divisors` is True, and with `system_vars` and `extra_vars`
        when `verbose` is True.
    """
    if method == "auto":
        method = get_dgcv_settings_registry().get("_solve_default", "auto")
    if surface_singularities is True:
        simplify_pivots = True
        return_divisors = True
        pass_to_symbolic_engine = False
    if pass_to_symbolic_engine is None:
        pass_to_symbolic_engine = (
            get_dgcv_settings_registry().get(
                "pass_solve_requests_to_symbolic_engine", False
            )
            is True
        )

    import time

    t0 = time.perf_counter()

    def _log(*a):
        if print_solve_stats:
            print("[solve_dgcv]", *a)

    if isinstance(eqns, (list, tuple)) and len(eqns) == 0:
        if isinstance(vars_to_solve, (list, tuple)):
            out = [{var: var for var in vars_to_solve}]
        elif isinstance(vars_to_solve, expr_numeric_types()):
            out = [{vars_to_solve: vars_to_solve}]
        else:
            out = [dict()]
        return (out, []) if return_divisors else out

    eqns, vars_to_solve = normalize_equations_and_vars(eqns, vars_to_solve)
    processed_eqns, system_vars, extra_vars, variables_dict = _equations_preprocessing(
        eqns, vars_to_solve
    )

    def _simplify(x):
        if not simplify_result:
            return x
        try:
            return simplify(x)
        except Exception:
            return x

    def _expr_reformatting(expr):
        if not hasattr(expr, "subs"):
            return expr

        dgcv_var_dict = {v[1][0]: v[0] for _, v in variables_dict.items()}

        if not isinstance(expr, expr_types()) or isinstance(expr, zeroFormAtom):
            try:
                return expr.subs(dgcv_var_dict)
            except Exception:
                return abstract_ZF(_sympy_to_abstract_ZF(expr, dgcv_var_dict))

        regular_var_dict = {k: v for k, v in dgcv_var_dict.items() if is_atomic(k)}

        try:
            bad = not all(
                isinstance(v, expr_numeric_types()) or isinstance(v, expr_types())
                for v in regular_var_dict.values()
            )
        except Exception:
            bad = True

        if bad:
            return abstract_ZF(_sympy_to_abstract_ZF(expr, regular_var_dict))

        try:
            return expr.subs(regular_var_dict)
        except Exception:
            return abstract_ZF(_sympy_to_abstract_ZF(expr, regular_var_dict))

    def _extract_reformatting(var):
        s = str(var)
        return variables_dict[s][0] if s in variables_dict else var

    mod = engine_module()

    def _engine_linsolve(eqns_, vars_):
        if mod is None:
            return None
        fn = getattr(mod, "linsolve", None)
        if not callable(fn):
            return None
        try:
            sols = fn(eqns_, tuple(vars_))
        except Exception:
            return None
        return _linsolve_to_dicts(sols, tuple(vars_))

    def _engine_solve(eqns_, vars_):
        if mod is None:
            return []
        fn = getattr(mod, "solve", None)
        if not callable(fn):
            return []
        try:
            sols = fn(eqns_, vars_, dict=True)
        except TypeError:
            try:
                sols = fn(eqns_, vars_)
            except Exception:
                return []
        except Exception:
            return []
        new_sols = _engine_solve_to_dicts(sols, vars_)
        out = []
        for sol in new_sols:
            out.append({var: sol.get(var, var) for var in vars_})
        return out

    _log(
        f"engine={(engine_kind() or 'unknown')} method={method} #eqns={len(processed_eqns)} #vars={len(system_vars)}"
    )

    preformatted_solutions = []
    divisors = []
    custom_succeeded = False
    engine_linsolve_ran = False

    def _run_custom_linsolve():
        nonlocal preformatted_solutions, divisors, custom_succeeded
        try:
            if return_divisors:
                preformatted_solutions, d = _dgcv_linsolve(
                    processed_eqns,
                    system_vars,
                    return_divisors=True,
                    sample_if_overdetermined=sample_if_overdetermined,
                    allow_underdetermined_solution=allow_underdetermined_solution,
                    simplify_pivots=simplify_pivots,
                )
                divisors = d or []
            else:
                preformatted_solutions = _dgcv_linsolve(
                    processed_eqns,
                    system_vars,
                    return_divisors=False,
                    sample_if_overdetermined=sample_if_overdetermined,
                    allow_underdetermined_solution=allow_underdetermined_solution,
                    simplify_pivots=simplify_pivots,
                )
            custom_succeeded = True
        except Exception:
            preformatted_solutions = []
            custom_succeeded = False

    def _run_engine_linsolve():
        nonlocal preformatted_solutions, engine_linsolve_ran
        try:
            res = _engine_linsolve(processed_eqns, system_vars)
        except Exception:
            res = None
        engine_linsolve_ran = res is not None
        preformatted_solutions = res or []

    def _run_engine_solve():
        nonlocal preformatted_solutions
        try:
            if engine_kind() == "sage":
                preformatted_solutions = _engine_solve(eqns, vars_to_solve)
            else:
                preformatted_solutions = _engine_solve(processed_eqns, system_vars)
        except Exception:
            preformatted_solutions = []

    if method not in ("auto", "linsolve", "solve"):
        raise ValueError(
            f"Unknown method '{method}'. Use 'auto', 'linsolve', or 'solve'."
        )

    if method == "solve":
        _run_engine_solve()
    elif method == "linsolve":
        if not pass_to_symbolic_engine:
            _run_custom_linsolve()
        if not custom_succeeded and not preformatted_solutions:
            _run_engine_linsolve()
        if (
            not custom_succeeded
            and not engine_linsolve_ran
            and not preformatted_solutions
        ):
            _run_engine_solve()
    else:
        if not pass_to_symbolic_engine:
            _run_custom_linsolve()
        if not custom_succeeded and not preformatted_solutions:
            _run_engine_linsolve()
        if (
            not custom_succeeded
            and not engine_linsolve_ran
            and not preformatted_solutions
        ):
            _run_engine_solve()

    solutions_formatted = [
        {
            _extract_reformatting(var): _expr_reformatting(_simplify(expr))
            for var, expr in (solution or {}).items()
        }
        for solution in (preformatted_solutions or [])
        if isinstance(solution, dict)
    ]

    _log(
        f"solutions={len(solutions_formatted)} elapsed_s={time.perf_counter() - t0:.6f}"
    )

    if return_divisors:
        present_atoms = set()
        returnables = []
        for v in divisors:
            atoms = get_free_symbols(v)
            if atoms:
                returnables.append(v)
                present_atoms |= atoms
        if surface_singularities:
            divisors = expr_union_primitives(
                returnables,
                order_coordinates(present_atoms),
                process_rationals=True,
                fail_quietly=True,
            )
        if verbose:
            return solutions_formatted, system_vars, extra_vars, divisors
        return solutions_formatted, divisors

    return (
        (solutions_formatted, system_vars, extra_vars)
        if verbose
        else solutions_formatted
    )
