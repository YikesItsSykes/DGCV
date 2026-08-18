from ..._aux._backends._symbolic_router import get_free_symbols, simplify
from ..._aux._utilities._misc import linear_combination
from ...core.arrays import array_dgcv, freeze_matrix, matrix_dgcv
from ...core.solvers import solve_dgcv
from ...core.vector_fields_and_differential_forms import (
    _extract_basis_by_wedge_vectorized,
)


def algebraDataFromMatRep(mat_list, assume_basis=True):
    """
    Create the structure data array for a Lie algebra from a list of matrices in *mat_list*.
    """
    if not mat_list:
        return (
            array_dgcv(
                dict(),
                shape=(0, 0),
                null_return=freeze_matrix(matrix_dgcv.zeros(0, 1)),
            ),
            mat_list,
            set(),
        )

    shape = mat_list[0].shape
    if (
        not len(shape) == 2
        or not shape[0] == shape[1]
        or not all(m.shape == shape for m in mat_list)
    ):
        raise Exception(
            "algorithm for extracting algebra data from matrices expects a list of square matrices of the same size."
        )
    if not assume_basis:
        mat_list = _extract_basis_by_wedge_vectorized(
            mat_list, array_shape_checked=True
        )
    indexRangeCap = len(mat_list)
    combiMatLoc, variables = linear_combination(mat_list)
    params = set()

    def pairValue(j, k, par):
        mat = (mat_list[j] @ mat_list[k]) - (mat_list[k] @ mat_list[j]) - combiMatLoc

        bracketVals = list(mat._data.values())
        solLoc = solve_dgcv(
            bracketVals, variables, method="linsolve", simplify_result=False
        )

        if len(solLoc) == 0:
            bracketVals = [simplify(expr) for expr in bracketVals]
            solLoc = solve_dgcv(
                bracketVals,
                variables,
                method="linsolve",
                pass_to_symbolic_engine=False,
                simplify_pivots=True,
                simplify_result=False,
            )
        if len(solLoc) > 0:
            soll = solLoc[0]
            coeffs = matrix_dgcv.zeros(indexRangeCap, 1)
            for idx, var in enumerate(variables):
                coeff = soll.get(var, var)
                if coeff != 0:
                    par |= get_free_symbols(coeff)
                    coeffs[idx] = coeff
            return coeffs
        raise Exception(
            f"Unable to determine if matrices are closed under commutators. "
            f"Problem matrices are in positions {j} and {k}."
        )

    structure_data = array_dgcv(
        dict(),
        shape=(indexRangeCap, indexRangeCap),
        null_return=freeze_matrix(matrix_dgcv.zeros(indexRangeCap, 1)),
    )
    for j in range(indexRangeCap):
        for k in range(j + 1, indexRangeCap):
            br = pairValue(k, j, params)
            if len(br._data) > 0:
                structure_data[k, j] = br
                structure_data[j, k] = -br

    return structure_data, mat_list, params
