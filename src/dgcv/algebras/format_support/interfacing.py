from ..._aux._backends._symbolic_router import simplify
from ..._aux._utilities._misc import linear_combination
from ...core.arrays import array_dgcv, freeze_matrix, matrix_dgcv
from ...core.solvers import solve_dgcv


def _external_library_algebra_processing(objs, mul, zero_obst, assume_skew=False):
    dim = len(objs)
    sd = array_dgcv(
        dict(),
        shape=(dim, dim),
        null_return=freeze_matrix(matrix_dgcv.zeros(dim, 1)),
    )
    genel, variables = linear_combination(objs)
    for c1, obj1 in enumerate(objs):
        start = c1 + 1 if assume_skew else 0
        for c, obj2 in enumerate(objs[start:]):
            c2 = c + start
            bracket = mul(obj1, obj2)
            sol = solve_dgcv(
                zero_obst(bracket - genel),
                variables,
                method="linsolve",
                simplify_result=False,
            )
            if len(sol) == 0:
                bracket = simplify(bracket)
                sol = solve_dgcv(
                    zero_obst(bracket - genel),
                    variables,
                    method="linsolve",
                    pass_to_symbolic_engine=False,
                    simplify_pivots=True,
                    simplify_result=False,
                )
                if len(sol) == 0:
                    raise ValueError(
                        f"Unable to confirm closure under the given product rule between elements of orders {c1} and {c2}"
                    )
            sol = sol[0]
            s_el = matrix_dgcv(
                {
                    idx: sol.get(v, 0)
                    for idx, v in enumerate(variables)
                    if sol.get(v, 0) != 0
                },
                shape=(dim, 1),
            )
            sd[c1, c2] = s_el
            if assume_skew:
                sd[c2, c1] = -s_el
    return sd
