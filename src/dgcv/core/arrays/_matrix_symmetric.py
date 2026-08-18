from ..._aux._backends._exact_arith import exact_reciprocal, ratio
from ..._aux._backends._symbolic_router import (
    _scalar_is_zero,
    _scalar_sign,
    clear_denominators,
)


class _matrix_symmetric:
    def symmetric_inertia(self):
        """
        Sylvester inertia of a symmetric matrix.

        Returns
        -------
        tuple of int
            `(p, n, z)`, the counts of positive, negative, and zero
            eigenvalues.

        Notes
        -----
        Assumes `self` is symmetric without validating. Computed via
        fraction-free Bareiss elimination with symmetric pivoting; falls
        back to a true-division Schur complement once a zero-diagonal
        2x2 pivot block is used, since that step is not fraction-free.
        Raises `IndeterminateSignError` (via `_scalar_sign`) if a
        pivot's sign cannot be certified.
        """
        if self.nrows != self.ncols:
            raise ValueError("symmetric_inertia only defined for square matrices")

        n = self.nrows
        if n == 0:
            return (0, 0, 0)

        flat = clear_denominators([self[i, j] for i in range(n) for j in range(n)])
        A = [flat[i * n : (i + 1) * n] for i in range(n)]

        pos = neg = zero = 0
        prev_pivot = 1
        prev_sign = 1
        k = 0
        while k < n:
            pivot_row = None
            for r in range(k, n):
                if not _scalar_is_zero(A[r][r]):
                    pivot_row = r
                    break

            if pivot_row is not None:
                if pivot_row != k:
                    A[k], A[pivot_row] = A[pivot_row], A[k]
                    for row in A:
                        row[k], row[pivot_row] = row[pivot_row], row[k]

                pivot = A[k][k]
                sign = _scalar_sign(pivot)
                if sign == prev_sign:
                    pos += 1
                else:
                    neg += 1

                for i in range(k + 1, n):
                    aik = A[i][k]
                    for j in range(k + 1, n):
                        A[i][j] = ratio(A[i][j] * pivot - aik * A[k][j], prev_pivot)
                for i in range(k + 1, n):
                    A[i][k] = A[k][i] = 0

                prev_pivot = pivot
                prev_sign = sign
                k += 1
                continue

            block = None
            for i in range(k, n):
                for j in range(i + 1, n):
                    if not _scalar_is_zero(A[i][j]):
                        block = (i, j)
                        break
                if block is not None:
                    break

            if block is None:
                zero += n - k
                break

            if prev_pivot != 1:
                for p in range(k, n):
                    for q in range(k, n):
                        A[p][q] = ratio(A[p][q], prev_pivot)

            i, j = block
            if i != k:
                A[k], A[i] = A[i], A[k]
                for row in A:
                    row[k], row[i] = row[i], row[k]
                if j == k:
                    j = i
            L = k + 1
            if j != L:
                A[L], A[j] = A[j], A[L]
                for row in A:
                    row[L], row[j] = row[j], row[L]

            b = A[k][L]
            inv_b = exact_reciprocal(b)
            new_entries = {}
            for p in range(L + 1, n):
                xp, yp = A[k][p], A[L][p]
                for q in range(L + 1, n):
                    new_entries[(p, q)] = A[p][q] - inv_b * (
                        xp * A[L][q] + yp * A[k][q]
                    )
            for (p, q), v in new_entries.items():
                A[p][q] = v
            for p in range(L + 1, n):
                A[p][k] = A[k][p] = 0
                A[p][L] = A[L][p] = 0
            A[k][L] = A[L][k] = 0

            pos += 1
            neg += 1
            prev_pivot = 1
            prev_sign = 1
            k += 2

        return (pos, neg, zero)

    def symmetric_signature(self):
        """
        Signature `p - n` of a symmetric matrix.

        Returns
        -------
        int

        Notes
        -----
        Assumes `self` is symmetric without validating
        """
        p, n, _ = self.symmetric_inertia()
        return p - n

    def is_positive_definite_symmetric(self):
        """
        Tests if a symmetric matrix is positive definite.

        Returns
        -------
        bool

        Notes
        -----
        Assumes `self` is symmetric without validating
        """
        if self.nrows != self.ncols:
            raise ValueError(
                "is_positive_definite_symmetric only defined for square matrices"
            )

        n = self.nrows
        if n == 0:
            return True

        flat = clear_denominators([self[i, j] for i in range(n) for j in range(n)])
        A = [flat[i * n : (i + 1) * n] for i in range(n)]

        prev_pivot = 1
        for k in range(n):
            pivot = A[k][k]
            if _scalar_sign(pivot) <= 0:
                return False
            for i in range(k + 1, n):
                aik = A[i][k]
                for j in range(k + 1, n):
                    A[i][j] = ratio(A[i][j] * pivot - aik * A[k][j], prev_pivot)
            prev_pivot = pivot
        return True

    def is_negative_definite_symmetric(self):
        """
        Tests if a symmetric matrix is negative definite.

        Returns
        -------
        bool

        Notes
        -----
        Assumes `self` is symmetric without validating
        """
        if self.nrows != self.ncols:
            raise ValueError(
                "is_negative_definite_symmetric only defined for square matrices"
            )

        n = self.nrows
        if n == 0:
            return True

        flat = clear_denominators([self[i, j] for i in range(n) for j in range(n)])
        A = [flat[i * n : (i + 1) * n] for i in range(n)]

        prev_pivot = 1
        for k in range(n):
            pivot = A[k][k]
            want = -1 if k % 2 == 0 else 1
            if _scalar_sign(pivot) != want:
                return False
            for i in range(k + 1, n):
                aik = A[i][k]
                for j in range(k + 1, n):
                    A[i][j] = ratio(A[i][j] * pivot - aik * A[k][j], prev_pivot)
            prev_pivot = pivot
        return True
