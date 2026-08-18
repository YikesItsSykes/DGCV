from numbers import Integral


def _as_matrix_dgcv(obj):
    if isinstance(obj, matrix_dgcv):
        return obj
    if isinstance(obj, (list, tuple)):
        return matrix_dgcv(obj)
    rows = getattr(obj, "rows", None)
    cols = getattr(obj, "cols", None)
    if isinstance(rows, Integral) and isinstance(cols, Integral):
        return matrix_dgcv(obj)
    return None
