from ..._aux._backends._symbolic_router import get_free_symbols, subs
from ..._aux._backends._types_and_constants import symbol
from ..._aux._vmf._safeguards import create_key
from ._driver import solve_dgcv


def linear_filter(spanners, linear_constraint, light_simplify=True):
    try:
        dim = len(spanners)
        pref = create_key("var")
        vars = [symbol(f"{pref}_{idx}") for idx in range(dim)]
        combination = sum(
            [var * elem for var, elem in zip(vars[1:], spanners[1:])],
            vars[0] * spanners[0],
        )
        constraints = linear_constraint(combination)
        if not isinstance(constraints, (list, tuple)):
            constraints = [constraints]
        eqns = []
        for constraint in constraints:
            obst = getattr(constraint, "__dgcv_zero_obstr__")
            if obst is not None:
                constraint = list(obst[0])
            if isinstance(constraint, list):
                eqns += constraint
            else:
                eqns.append(constraint)
        solution = solve_dgcv(eqns, vars)
        if len(solution) == 0:
            return []
        filtered = subs(combination, solution[0])
        if light_simplify is True:
            f = getattr(filtered, "scale_to_polynomial_attempt", None)
            if callable(f):
                filtered = f()
        freeVars = set()
        available = set(vars)
        for expr in solution[0].values():
            survivors = {var for var in get_free_symbols(expr) if var in available}
            freeVars |= survivors
        if len(freeVars) == 0:
            if all(val == 0 for val in solution[0].values()):
                return []
            return [filtered]
        zeroing = {var: 0 for var in freeVars}

        # def scale(elem):
        #     f = getattr(elem, "scale_to_polynomial_attempt", None)
        #     if callable(f):
        #         return f()
        #     return elem

        # if light_simplify is True:
        #     return [scale(subs(filtered, zeroing | {var: 1})) for var in freeVars]
        return [subs(filtered, zeroing | {var: 1}) for var in freeVars]

    except Exception:
        raise (
            "`linear_filter` recieved unsupported parameters. The first argument should be a list of elements compatible with taking linear combinations over the symbolic ring of whichever engine is active. The second should be a callable object that can operate on such combinations, and that returns a scalar or list of scalars representing linear constraints when set equal to 0. The callable can also return types other than scalars that `dgcv` knows how to compare with zero (i.e., `matrix_dgcv`, `vector_field_class`, etc.)"
        )
