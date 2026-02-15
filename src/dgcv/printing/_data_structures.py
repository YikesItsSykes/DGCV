"""
package: dgcv - Differential Geometry with Complex Variables
module: printing/_data_structures

Author (of this module): David Sykes (https://realandimaginary.com/dgcv/)

License:
    MIT License
"""


# -----------------------------------------------------------------------------
# body
# -----------------------------------------------------------------------------
def format_unicode_table(
    rows,
    *,
    row_labels=None,
    column_labels=None,
    caption=None,
    col_number_limit: int = 15,
    row_number_limit: int = 15,
    cell_char_lim: int = 20,
    align: str = "left",
    header_align: str = "center",
    row_label_align: str = "left",
):
    def _as_str(x) -> str:
        s = str(x)
        if cell_char_lim is not None and cell_char_lim > 0 and len(s) > cell_char_lim:
            k = max(0, cell_char_lim - 4)
            s = f"{s[:k]} ..."
        return s

    def _pad(s: str, w: int, how: str) -> str:
        if how == "right":
            return s.rjust(w)
        if how == "center":
            return s.center(w)
        return s.ljust(w)

    rows = list(rows or [])
    nrows = len(rows)
    ncols = max((len(r) for r in rows), default=0)

    norm = []
    for r in rows:
        rr = list(r)
        if len(rr) < ncols:
            rr.extend([""] * (ncols - len(rr)))
        norm.append(rr)
    rows = norm

    if row_labels is not None:
        row_labels = list(row_labels)
        if len(row_labels) != nrows:
            print(
                "format_unicode_table: row_labels length does not match number of rows."
            )
            return
    if column_labels is not None:
        column_labels = list(column_labels)
        if len(column_labels) != ncols:
            print(
                "format_unicode_table: column_labels length does not match number of columns."
            )
            return

    show_row_label_col = row_labels is not None

    lim_cols = col_number_limit if col_number_limit is not None else ncols
    lim_rows = row_number_limit if row_number_limit is not None else nrows
    if lim_cols is None or lim_cols <= 0:
        lim_cols = 0
    if lim_rows is None or lim_rows <= 0:
        lim_rows = 0

    col_truncated = ncols > lim_cols
    row_truncated = nrows > lim_rows

    keep_cols = min(ncols, lim_cols)
    keep_rows = min(nrows, lim_rows)

    col_idx = list(range(keep_cols))
    row_idx = list(range(keep_rows))

    cont_col_symbol = "⋯"
    cont_row_symbol = "⋮"
    add_cont_col = bool(col_truncated)

    disp_rows = []

    if column_labels is not None:
        hdr = []
        if show_row_label_col:
            hdr.append("")
        for j in col_idx:
            hdr.append(_as_str(column_labels[j]))
        if add_cont_col:
            hdr.append(cont_col_symbol)
        disp_rows.append(("__HEADER__", hdr))

    for i in row_idx:
        rr = []
        if show_row_label_col:
            rr.append(_as_str(row_labels[i]))
        for j in col_idx:
            rr.append(_as_str(rows[i][j]))
        if add_cont_col:
            rr.append(cont_col_symbol)
        disp_rows.append(("__BODY__", rr))

    if not disp_rows:
        out = []
        if caption:
            out.append(str(caption))
        out.append("(empty)")
        return "\n".join(out)

    ndisp_cols = len(disp_rows[0][1])

    if row_truncated:
        disp_rows.append(("__TRUNC__", [cont_row_symbol] * ndisp_cols))

    widths = [0] * ndisp_cols
    for kind, rr in disp_rows:
        for k, s in enumerate(rr):
            widths[k] = max(widths[k], len(s))

    def _border(left: str, mid: str, right: str) -> str:
        parts = ["─" * (w + 2) for w in widths]
        return left + mid.join(parts) + right

    top = _border("┌", "┬", "┐")
    mid = _border("├", "┼", "┤")
    bot = _border("└", "┴", "┘")

    def _render_row(kind: str, rr: list[str]) -> str:
        out_cells = []
        for k, s in enumerate(rr):
            if kind == "__TRUNC__":
                how = "center"
            elif column_labels is not None and kind == "__HEADER__":
                how = header_align
            elif show_row_label_col and k == 0:
                how = row_label_align
            else:
                how = align
            out_cells.append(" " + _pad(s, widths[k], how) + " ")
        return "│" + "│".join(out_cells) + "│"

    lines = []
    if caption:
        lines.append(str(caption))

    lines.append(top)

    start = 0
    if column_labels is not None:
        lines.append(_render_row("__HEADER__", disp_rows[0][1]))
        lines.append(mid)
        start = 1

    for idx in range(start, len(disp_rows)):
        kind, rr = disp_rows[idx]
        lines.append(_render_row(kind, rr))

    lines.append(bot)

    return "\n".join(lines)
