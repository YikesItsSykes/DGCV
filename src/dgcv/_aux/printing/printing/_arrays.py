from __future__ import annotations

from ._string_processing import (
    _latex_escape_text,
)


def array_VS_printer(A, _repr=False):
    shape = getattr(A, "shape", None)
    if not shape:
        return "array_dgcv(?)"
    if len(shape) == 2:
        r, c = shape
        lines = []
        form = repr if _repr else str
        for i in range(r):
            row = [A[i, j] for j in range(c)]
            row_str = ", ".join(
                ("" if _repr else "∅") if x is None else form(x) for x in row
            )
            lines.append("[" + row_str + "]")
        head = f"{A.__class__.__name__}(["
        return head + ", ".join(lines) + "])"
    return f"{A.__class__.__name__}(shape={shape}, ndim={len(shape)})"


def _array_latex_2d(A, env="bmatrix"):
    from dgcv._aux.printing.printing._dgcv_display import LaTeX

    r, c = A.shape
    rows = []
    for i in range(r):
        entries = []
        for j in range(c):
            x = A[i, j]
            entries.append("" if x is None else LaTeX(x))
        rows.append(" & ".join(entries))
    body = r" \\ ".join(rows)
    return rf"\begin{{{env}}}{body}\end{{{env}}}"


def _array_latex_nd(A, max_total_entries=800):
    shape = A.shape
    total = 1
    for s in shape:
        total *= s
    if total > max_total_entries:
        sh = _latex_escape_text(str(shape))
        return rf"\text{{array}}(\text{{shape}}={sh})"

    def rec(prefix, axis):
        from dgcv._aux.printing.printing._dgcv_display import LaTeX

        if axis == len(shape) - 1:
            parts = []
            for i in range(shape[axis]):
                x = A[tuple(prefix + [i])]
                parts.append("" if x is None else LaTeX(x))
            inner = r",\, ".join(parts)
            return rf"\left[{inner}\right]"
        parts = []
        for i in range(shape[axis]):
            parts.append(rec(prefix + [i], axis + 1))
        inner = r",\, ".join(parts)
        return rf"\left[{inner}\right]"

    return rec([], 0)


def array_latex_helper(A, env="bmatrix"):
    if getattr(A, "shape", None) is None:
        return r"\text{array}(\text{shape}=? )"
    if len(A.shape) == 2:
        return _array_latex_2d(A, env=env)
    return _array_latex_nd(A)
