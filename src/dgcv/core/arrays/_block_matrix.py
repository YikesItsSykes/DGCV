from ..._aux._backends._types_and_constants import expr_numeric_types
from ..._aux._vmf._safeguards import query_dgcv_categories
from ._validation import _as_seq


def assemble_block_matrix(blocks):
    B = _as_seq(blocks, "blocks")
    if B is None:
        B = []

    rows = []
    max_cols = 0
    for r in B:
        rr = _as_seq(r, "block row")
        if rr is None:
            rr = []
        rr = list(rr)
        rows.append(rr)
        if len(rr) > max_cols:
            max_cols = len(rr)

    for rr in rows:
        if len(rr) < max_cols:
            rr.extend([0] * (max_cols - len(rr)))

    n_block_rows = len(rows)
    n_block_cols = max_cols

    def _is_matrix(x):
        return bool(query_dgcv_categories(x, {"matrix"}))

    def _is_scalar(x):
        return isinstance(x, expr_numeric_types())

    row_heights = [None] * n_block_rows
    col_widths = [None] * n_block_cols

    for i in range(n_block_rows):
        for j in range(n_block_cols):
            x = rows[i][j]

            if _is_matrix(x):
                m = int(getattr(x, "nrows"))
                n = int(getattr(x, "ncols"))

                rh = row_heights[i]
                if rh is None:
                    row_heights[i] = m
                elif rh != m:
                    raise ValueError(
                        f"Incompatible block-row heights in row {i}: got {rh} and {m}."
                    )

                cw = col_widths[j]
                if cw is None:
                    col_widths[j] = n
                elif cw != n:
                    raise ValueError(
                        f"Incompatible block-column widths in col {j}: got {cw} and {n}."
                    )

            elif _is_scalar(x):
                continue
            else:
                raise TypeError(
                    "Block entries must satisfy "
                    "isinstance(x, expr_numeric_types()) or query_dgcv_categories(x, {'matrix'})."
                )

    for i in range(n_block_rows):
        if row_heights[i] is None:
            row_heights[i] = 1
    for j in range(n_block_cols):
        if col_widths[j] is None:
            col_widths[j] = 1

    for i in range(n_block_rows):
        for j in range(n_block_cols):
            x = rows[i][j]
            if _is_matrix(x):
                continue
            m = row_heights[i]
            n = col_widths[j]
            rows[i][j] = x * matrix_dgcv.padded_identity(m, n)

    total_m = sum(row_heights)
    total_n = sum(col_widths)

    out_data = {}
    row_off = 0
    for i in range(n_block_rows):
        col_off = 0
        for j in range(n_block_cols):
            blk = rows[i][j]
            for (ii, jj), v in blk.iter_nonzero_items(
                include_zeros=False, include_none=False
            ):
                out_data[(row_off + ii, col_off + jj)] = v
            col_off += col_widths[j]
        row_off += row_heights[i]

    return matrix_dgcv(out_data, shape=(total_m, total_n))
