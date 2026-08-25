from ..._aux._backends._symbolic_router import (
    _scalar_is_zero,
    get_free_symbols,
    simplify,
)
from ..._aux._utilities._misc import linear_combination
from ...core.arrays import array_dgcv, freeze_matrix, matrix_dgcv
from ...core.solvers import solve_dgcv


def algebraDataFromTensorRep(tensor_list):
    """
    Create the structure data array from a list of tensor products closed under the `_contraction_product` operator (see dgcv.tensor_field_class documentation).
    """

    dim = len(tensor_list)
    if dim == 0:
        return (
            array_dgcv(
                dict(), shape=(0, 0), null_return=freeze_matrix(matrix_dgcv.zeros(0, 1))
            ),
            tensor_list,
            set(),
        )

    gen_elem, variables = linear_combination(tensor_list)

    params = set()

    def computeBracket(j, k, par):
        if k < j:
            return
        product = (tensor_list[j] * tensor_list[k]) - gen_elem
        solutions = solve_dgcv(
            product, variables, method="linsolve", simplify_result=False
        )
        if len(solutions) == 0:
            product = simplify(product)
            solutions = solve_dgcv(
                product,
                variables,
                method="linsolve",
                pass_to_symbolic_engine=False,
                simplify_pivots=True,
                simplify_result=False,
            )
        if len(solutions) > 0:
            sol_values = solutions[0]
            coeffs = matrix_dgcv.zeros(dim, 1)
            for idx, var in enumerate(variables):
                coeff = sol_values.get(var, var)
                if not _scalar_is_zero(coeff):
                    par |= get_free_symbols(coeff)
                    coeffs[idx] = coeff
            return coeffs
        else:
            raise Exception(
                f"Contraction product of tensors at positions {j} and {k} are not in the given tensor list."
            )

    structure_data = array_dgcv(
        dict(), shape=(dim, dim), null_return=freeze_matrix(matrix_dgcv.zeros(dim, 1))
    )

    for j in range(dim):
        for k in range(j):
            br = computeBracket(k, j, params)
            if len(br._data) > 0:
                structure_data[k, j] = br  # CHECK index order!!!
                structure_data[j, k] = -br

    return structure_data, tensor_list, params  # filter independants
