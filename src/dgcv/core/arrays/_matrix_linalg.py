from ..._aux._backends._exact_arith import exact_reciprocal, ratio
from ..._aux._backends._symbolic_router import _scalar_is_zero, simplify
from ..._aux._backends._types_and_constants import fast_scalar_types, symbol
from ..._aux._vmf._safeguards import create_key
from ._indexing import _spool


def _invert(*, record_divisors=False, allow_formal=True):
    if record_divisors:
        divisors = []

        def _inv(denominator):
            divisors.append(denominator)
            try:
                return exact_reciprocal(denominator)
            except Exception:  ###!!! optimize for the formal divisor path
                if allow_formal:
                    return 1 / denominator
                raise

        return _inv, divisors

    def _inv(denominator):
        try:
            return exact_reciprocal(denominator)
        except Exception:
            if allow_formal:  ###!!! optimize for the formal divisor path
                return 1 / denominator
            raise

    return _inv, None


def _eliminate(
    work_rows,
    *,
    rhs_cols=0,
    record_divisors=False,
    allow_formal_inverse=True,
    simplify_steps=False,
    fast_only=False,
    want_pivmap=False,
):
    row_count = len(work_rows)
    total_cols = len(work_rows[0]) if row_count else 0
    col_count = total_cols - rhs_cols

    if row_count == 0 or col_count == 0:
        if want_pivmap:
            return 0, {}, [], [] if record_divisors else None
        return 0, [] if record_divisors else None
    _inv, divisors = _invert(
        record_divisors=record_divisors, allow_formal=allow_formal_inverse
    )
    pivot_index = 0
    pivcol_to_row = {}
    pivot_cols = []

    for col_idx in range(col_count):
        pivot_row = None
        for scan_row in range(pivot_index, row_count):
            if not _scalar_is_zero(work_rows[scan_row][col_idx]):
                pivot_row = scan_row
                break
        if pivot_row is None:
            continue

        if pivot_row != pivot_index:
            work_rows[pivot_index], work_rows[pivot_row] = (
                work_rows[pivot_row],
                work_rows[pivot_index],
            )

        pivot_value = work_rows[pivot_index][col_idx]
        inv_pivot = _inv(pivot_value)

        for entry_idx in range(col_idx, total_cols):
            work_rows[pivot_index][entry_idx] = (
                work_rows[pivot_index][entry_idx] * inv_pivot
            )
        if simplify_steps and not fast_only:
            for entry_idx in range(col_idx, total_cols):
                work_rows[pivot_index][entry_idx] = simplify(
                    work_rows[pivot_index][entry_idx]
                )

        for row_idx in range(row_count):
            if row_idx == pivot_index:
                continue
            factor = work_rows[row_idx][col_idx]
            if _scalar_is_zero(factor):
                continue
            for entry_idx in range(col_idx, total_cols):
                work_rows[row_idx][entry_idx] = (
                    work_rows[row_idx][entry_idx]
                    - factor * work_rows[pivot_index][entry_idx]
                )
            if simplify_steps and not fast_only:
                for entry_idx in range(col_idx, total_cols):
                    work_rows[row_idx][entry_idx] = simplify(
                        work_rows[row_idx][entry_idx]
                    )

        pivcol_to_row[col_idx] = pivot_index
        pivot_cols.append(col_idx)
        pivot_index += 1
        if pivot_index == row_count:
            break

    if want_pivmap:
        return pivot_index, pivcol_to_row, pivot_cols, divisors
    return pivot_index, divisors


class _matrix_linalg:
    def det(self):
        if self.nrows != self.ncols:
            raise ValueError("det only defined for square matrices")

        size = self.nrows
        if size == 0:
            return 1
        if size == 1:
            return self[0, 0]

        work_rows = [
            [self[idx, col_idx] for col_idx in range(size)] for idx in range(size)
        ]
        prev_pivot = 1
        sign = 1

        for step in range(size - 1):
            pivot_row = None
            for scan_row in range(step, size):
                if not _scalar_is_zero(work_rows[scan_row][step]):
                    pivot_row = scan_row
                    break
            if pivot_row is None:
                return 0
            if pivot_row != step:
                work_rows[step], work_rows[pivot_row] = (
                    work_rows[pivot_row],
                    work_rows[step],
                )
                sign = -sign

            pivot = work_rows[step][step]
            for idx in range(step + 1, size):
                leading_entry = work_rows[idx][step]
                for col_idx in range(step + 1, size):
                    work_rows[idx][col_idx] = ratio(
                        (
                            work_rows[idx][col_idx] * pivot
                            - leading_entry * work_rows[step][col_idx]
                        ),
                        prev_pivot,
                    )
                work_rows[idx][step] = 0

            prev_pivot = pivot
            if _scalar_is_zero(prev_pivot):
                return 0

        return sign * work_rows[size - 1][size - 1]

    def charpoly(self, var):
        """
        Characteristic polynomial of the matrix

        Parameters
        ----------
        var : symbolic variable
            used as the polynomial's variable
        """
        if self.nrows != self.ncols:
            raise ValueError("charpoly only defined for square matrices")
        return (var * self.identity(self.nrows) - self).det()

    def inverse(self):
        if self.nrows != self.ncols:
            raise ValueError("inverse only defined for square matrices")

        size = self.nrows
        work_rows = [
            [self[idx, col_idx] for col_idx in range(size)] for idx in range(size)
        ]
        inverse_rows = [[0 for _ in range(size)] for _ in range(size)]
        for idx in range(size):
            inverse_rows[idx][idx] = 1

        for pivot_col in range(size):
            pivot_row = None
            for scan_row in range(pivot_col, size):
                if not _scalar_is_zero(work_rows[scan_row][pivot_col]):
                    pivot_row = scan_row
                    break
            if pivot_row is None:
                raise ZeroDivisionError("matrix is singular")

            if pivot_row != pivot_col:
                work_rows[pivot_col], work_rows[pivot_row] = (
                    work_rows[pivot_row],
                    work_rows[pivot_col],
                )
                inverse_rows[pivot_col], inverse_rows[pivot_row] = (
                    inverse_rows[pivot_row],
                    inverse_rows[pivot_col],
                )

            pivot = work_rows[pivot_col][pivot_col]
            inv_pivot = exact_reciprocal(pivot)

            for col_idx in range(size):
                work_rows[pivot_col][col_idx] = (
                    work_rows[pivot_col][col_idx] * inv_pivot
                )
                inverse_rows[pivot_col][col_idx] = (
                    inverse_rows[pivot_col][col_idx] * inv_pivot
                )

            for idx in range(size):
                if idx == pivot_col:
                    continue
                factor = work_rows[idx][pivot_col]
                if _scalar_is_zero(factor):
                    continue
                for col_idx in range(size):
                    work_rows[idx][col_idx] = (
                        work_rows[idx][col_idx] - factor * work_rows[pivot_col][col_idx]
                    )
                    inverse_rows[idx][col_idx] = (
                        inverse_rows[idx][col_idx]
                        - factor * inverse_rows[pivot_col][col_idx]
                    )

        return self.__class__(inverse_rows)

    def rank(
        self,
        assume_fast_data_types=False,
        allow_formal_inverse=False,
        simplify_steps=False,
        record_divisors=False,
    ):
        if self.nrows == 0 or self.ncols == 0:
            if record_divisors:
                return 0, []
            return 0

        fast_types = fast_scalar_types()
        fast_case = assume_fast_data_types or all(
            isinstance(entry, fast_types) for entry in self._data.values()
        )

        work_rows = self._dense_copy()
        rank_value, divisors = _eliminate(
            work_rows,
            rhs_cols=0,
            record_divisors=record_divisors,
            allow_formal_inverse=allow_formal_inverse,
            simplify_steps=simplify_steps,
            fast_only=fast_case,
            want_pivmap=False,
        )
        if record_divisors:
            return rank_value, divisors
        return rank_value

    def nullspace(self):
        row_count = self.nrows
        col_count = self.ncols

        if col_count == 0:
            return []
        if row_count == 0:
            out = []
            for idx in range(col_count):
                basis_vector = self.__class__.zeros(col_count, 1)
                basis_vector._data[_spool((idx, 0), basis_vector.shape)] = 1
                out.append(basis_vector)
            return out

        work_rows = self._dense_copy()
        _, pivcol_to_row, pivot_cols, _ = _eliminate(
            work_rows,
            rhs_cols=0,
            record_divisors=False,
            allow_formal_inverse=False,
            simplify_steps=False,
            fast_only=False,
            want_pivmap=True,
        )

        free_cols = [idx for idx in range(col_count) if idx not in pivcol_to_row]
        if not free_cols:
            return []

        out = []
        for free_col in free_cols:
            components = [0] * col_count
            components[free_col] = 1
            for pivot_col in pivot_cols:
                pivot_row = pivcol_to_row[pivot_col]
                components[pivot_col] = -work_rows[pivot_row][free_col]

            basis_vector = self.__class__.zeros(col_count, 1)
            for idx, value in enumerate(components):
                if not _scalar_is_zero(value):
                    basis_vector._data[_spool((idx, 0), basis_vector.shape)] = value
            out.append(basis_vector)

        return out

    def solve_right(
        self,
        b,
        *,
        return_divisors=False,
        simplify_steps=False,
        stamp_divisors=False,
        allow_formal_inverse=True,
        return_parametric=True,
        parametric_vars=None,
    ):
        if self.nrows == 0:
            sol = [] if (getattr(b, "nrows", 0) == 0) else None
            return (sol, []) if return_divisors else sol

        rhs_matrix = b if isinstance(b, matrix_dgcv) else matrix_dgcv(b)
        if rhs_matrix.ncols != 1 or rhs_matrix.nrows != self.nrows:
            raise ValueError("b must be a column vector with matching nrows")

        fast_types = fast_scalar_types()
        fast_case = all(
            isinstance(entry, fast_types) for entry in self._data.values()
        ) and all(isinstance(entry, fast_types) for entry in rhs_matrix._data.values())

        row_count = self.nrows
        col_count = self.ncols
        work_rows = [
            [self[idx, col_idx] for col_idx in range(col_count)] + [rhs_matrix[idx, 0]]
            for idx in range(row_count)
        ]

        _rank, pivcol_to_row, _, divisors = _eliminate(
            work_rows,
            rhs_cols=1,
            record_divisors=return_divisors,
            allow_formal_inverse=allow_formal_inverse,
            simplify_steps=simplify_steps,
            fast_only=fast_case,
            want_pivmap=True,
        )

        for row_idx in range(row_count):
            row_is_zero = True
            for col_idx in range(col_count):
                if not _scalar_is_zero(work_rows[row_idx][col_idx]):
                    row_is_zero = False
                    break
            if row_is_zero and not _scalar_is_zero(work_rows[row_idx][col_count]):
                out = None
                if return_divisors:
                    divisor_list = divisors or []
                    if stamp_divisors and divisor_list:
                        seen = set()
                        deduped = []
                        for divisor in divisor_list:
                            if divisor in seen:
                                continue
                            seen.add(divisor)
                            deduped.append(divisor)
                        return out, deduped
                    return out, divisor_list
                return out

        if parametric_vars is None:
            prefix = create_key("_x", True, 4)
            params = [symbol(f"{prefix}_{idx}") for idx in range(col_count)]
        else:
            params = list(parametric_vars)
            if len(params) != col_count:
                raise ValueError("parametric_vars must have length equal to ncols")

        free_cols = [idx for idx in range(col_count) if idx not in pivcol_to_row]
        solution = [0] * col_count
        for free_col in free_cols:
            solution[free_col] = params[free_col]

        for pivot_col in sorted(pivcol_to_row):
            pivot_row = pivcol_to_row[pivot_col]
            value = work_rows[pivot_row][col_count]
            for free_col in free_cols:
                value = value - work_rows[pivot_row][free_col] * solution[free_col]
            if simplify_steps and not fast_case:
                value = simplify(value)
            solution[pivot_col] = value

        out = solution

        if return_divisors:
            divisor_list = divisors or []
            if stamp_divisors and divisor_list:
                seen = set()
                deduped = []
                for divisor in divisor_list:
                    if divisor in seen:
                        continue
                    seen.add(divisor)
                    deduped.append(divisor)
                return out, deduped
            return out, divisor_list
        return out

    def try_solve_right(self, b):
        return self.solve_right(b, return_divisors=False)
