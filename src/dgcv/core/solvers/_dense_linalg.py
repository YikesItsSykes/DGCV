from ..._aux._backends._symbolic_router import _fast_simplify
from ..._aux._backends._types_and_constants import one, zero
from ._predicates import _is_zero, _is_zero_after_simplify


def _rows_from_engine_matrix(A):
    nrows = getattr(A, "nrows", None)
    ncols = getattr(A, "ncols", None)
    if callable(nrows) and callable(ncols):
        r = int(nrows())
        c = int(ncols())
        return [[A[i, j] for j in range(c)] for i in range(r)]

    shape = getattr(A, "shape", None)
    if isinstance(shape, tuple) and len(shape) == 2:
        r, c = shape
        return [[A[i, j] for j in range(c)] for i in range(r)]

    rows = getattr(A, "tolist", None)
    if callable(rows):
        v = A.tolist()
        if isinstance(v, list) and v and isinstance(v[0], list):
            return v

    raise TypeError("Unsupported matrix type returned by engine")


def _col_from_engine_vector(b):
    nrows = getattr(b, "nrows", None)
    ncols = getattr(b, "ncols", None)
    if callable(nrows) and callable(ncols):
        r = int(nrows())
        c = int(ncols())
        if c != 1:
            raise ValueError("Expected column vector")
        return [b[i, 0] for i in range(r)]

    shape = getattr(b, "shape", None)
    if isinstance(shape, tuple) and len(shape) == 2:
        r, c = shape
        if c != 1:
            raise ValueError("Expected column vector")
        return [b[i, 0] for i in range(r)]

    if isinstance(b, (list, tuple)):
        return list(b)

    raise TypeError("Unsupported vector type returned by engine")


def _rref_solve_unique(A_rows, b_col, *, simplify_steps=False):
    m = len(A_rows)
    if m == 0:
        return []

    n = len(A_rows[0])
    if len(b_col) != m:
        raise ValueError("Row mismatch between A and b")

    aug = [list(A_rows[i]) + [b_col[i]] for i in range(m)]
    row = 0
    pivots = []

    for col in range(n):
        if row >= m:
            break

        pivot_row = None
        for r in range(row, m):
            if not _is_zero(aug[r][col]):
                pivot_row = r
                break

        if pivot_row is None:
            continue

        if pivot_row != row:
            aug[row], aug[pivot_row] = aug[pivot_row], aug[row]

        piv = aug[row][col]
        inv_piv = one / piv
        aug[row] = [inv_piv * v for v in aug[row]]
        if simplify_steps:
            aug[row] = [_fast_simplify(v) for v in aug[row]]

        for r in range(m):
            if r == row:
                continue
            factor = aug[r][col]
            if _is_zero(factor):
                continue
            aug[r] = [aug[r][c] - factor * aug[row][c] for c in range(n + 1)]
            if simplify_steps:
                aug[r] = [_fast_simplify(v) for v in aug[r]]

        pivots.append(col)
        row += 1

    for r in range(m):
        if all(_is_zero(aug[r][c]) for c in range(n)) and not _is_zero_after_simplify(
            aug[r][n]
        ):
            raise ValueError("Inconsistent linear system")

    if len(pivots) != n:
        raise ValueError("Singular or underdetermined system")

    x = [zero for _ in range(n)]
    for r, col in enumerate(pivots):
        x[col] = aug[r][n]
        if simplify_steps:
            x[col] = _fast_simplify(x[col])
    return x
