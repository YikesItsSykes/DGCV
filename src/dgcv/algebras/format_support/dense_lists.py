from ..._aux._backends._symbolic_router import _scalar_is_zero, get_free_symbols
from ...core.arrays import array_dgcv, freeze_matrix, matrix_dgcv


def aDataFromNestedLists(nested_lists):
    dim = len(nested_lists)
    sd = array_dgcv(
        dict(),
        shape=(dim, dim),
        null_return=freeze_matrix(matrix_dgcv.zeros(dim, 1)),
    )
    params = set()
    for idx1, outer in enumerate(nested_lists):
        if len(outer) != dim:
            raise TypeError()
        for idx2, middle in enumerate(outer):
            if len(middle) != dim:
                raise TypeError()
            inner_dict = dict()
            for c, v in enumerate(middle):
                if v is not None and not _scalar_is_zero(v):
                    params |= get_free_symbols(v)
                    inner_dict[c] = v
            if inner_dict:
                sd[idx1, idx2] = matrix_dgcv(inner_dict, shape=(dim, 1))
    return sd, params
