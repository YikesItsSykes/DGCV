from ..._aux._backends._symbolic_router import _scalar_is_zero, get_free_symbols
from ...core.arrays import matrix_dgcv
from ..linear_algebra import _structure_array
from .sparse_dicts import _complete_skew


def aDataFromNestedLists(nested_lists, assume_skew=False):
    dim = len(nested_lists)
    cells = dict()
    params = set()
    for idx1, outer in enumerate(nested_lists):
        if len(outer) != dim:
            raise TypeError(
                f"Nested list structure data must have shape ({dim}, {dim}, {dim}), but entry {idx1} has length {len(outer)}."
            )
        for idx2, middle in enumerate(outer):
            if len(middle) != dim:
                raise TypeError(
                    f"Nested list structure data must have shape ({dim}, {dim}, {dim}), but entry ({idx1}, {idx2}) has length {len(middle)}."
                )
            inner_dict = dict()
            for c, v in enumerate(middle):
                if v is not None and not _scalar_is_zero(v):
                    params |= get_free_symbols(v)
                    inner_dict[c] = v
            if inner_dict:
                cells[(idx1, idx2)] = matrix_dgcv(inner_dict, shape=(dim, 1))
    if assume_skew:
        _complete_skew(cells, dim)
    return _structure_array(cells, dim), params
