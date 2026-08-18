from ..._aux._vmf._safeguards import retrieve_passkey
from ._indexing import _spool
from ._validation import (
    _as_seq,
    _new_empty_matrix,
    _validate_int,
    _validate_pos_int,
)


class _matrix_constructors:
    @classmethod
    def identity(cls, n, one=1, zero=0):
        out = cls.__new__(cls)
        out._dgcv_class_check = retrieve_passkey()
        out._dgcv_categories = {"matrix"}
        out.shape = (n, n)
        out.ndim = 2
        out._data = {}
        out.null_return = 0
        out._engine_representation = dict()
        out._data_unspooled_cache = None
        for i in range(n):
            out._data[_spool((i, i), out.shape)] = one
        return out

    @classmethod
    def padded_identity(cls, nrows, ncols, one=1, zero=0):
        _validate_pos_int(nrows, "nrows")
        _validate_pos_int(ncols, "ncols")

        out = cls.__new__(cls)
        out._dgcv_class_check = retrieve_passkey()
        out._dgcv_categories = {"matrix"}
        out.shape = (int(nrows), int(ncols))
        out.ndim = 2
        out._data = {}
        out.null_return = 0

        d = min(out.shape[0], out.shape[1])
        if one != zero:
            for i in range(d):
                out._data[_spool((i, i), out.shape)] = one

        out._engine_representation = dict()
        out._data_unspooled_cache = None
        return out

    @classmethod
    def zeros(cls, nrows, ncols=None):
        if ncols is None:
            ncols = nrows
        out = _new_empty_matrix(cls, nrows, ncols, null_return=0)
        return out

    @classmethod
    def ones(cls, nrows, ncols=None, one=1):
        if ncols is None:
            ncols = nrows
        out = _new_empty_matrix(cls, nrows, ncols, null_return=0)
        for i in range(nrows):
            for j in range(ncols):
                out._data[_spool((i, j), out.shape)] = one
        return out

    @classmethod
    def eye(cls, n, one=1, zero=0):
        return cls.identity(n, one=one, zero=zero)

    @classmethod
    def diag(cls, diag_entries, nrows=None, ncols=None, k=0, zero=0):
        d = _as_seq(diag_entries, "diag_entries")
        if d is None:
            d = []
        _validate_int(k, "k")

        if nrows is None and ncols is None:
            nrows = len(d) + max(0, -k)
            ncols = len(d) + max(0, k)
        elif nrows is None:
            _validate_pos_int(ncols, "ncols")
            nrows = max(0, ncols - k)
        elif ncols is None:
            _validate_pos_int(nrows, "nrows")
            ncols = max(0, nrows + k)

        out = _new_empty_matrix(cls, nrows, ncols, null_return=0)
        for t, v in enumerate(d):
            i = t
            j = t + k
            if 0 <= i < nrows and 0 <= j < ncols and v != zero:
                out._data[_spool((i, j), out.shape)] = v
        return out

    @classmethod
    def from_rows(cls, rows):
        rows = _as_seq(rows, "rows")
        if rows is None:
            rows = []
        if not rows:
            return cls.zeros(0, 0)
        first = _as_seq(rows[0], "rows[0]")
        ncols = len(first)
        for r in rows:
            rr = _as_seq(r, "row")
            if len(rr) != ncols:
                raise ValueError("All rows must have the same length")
        return cls(rows)

    @classmethod
    def from_cols(cls, cols):
        cols = _as_seq(cols, "cols")
        if cols is None:
            cols = []
        if not cols:
            return cls.zeros(0, 0)
        ncols = len(cols)
        first = _as_seq(cols[0], "cols[0]")
        nrows = len(first)
        for c in cols:
            cc = _as_seq(c, "col")
            if len(cc) != nrows:
                raise ValueError("All columns must have the same length")
        rows = [[cols[j][i] for j in range(ncols)] for i in range(nrows)]
        return cls(rows)

    @classmethod
    def col_vector(cls, data):
        v = _as_seq(data, "data")
        if v is None:
            v = []
        return cls([[x] for x in v])

    @classmethod
    def toeplitz(cls, col0, row0=None, zero=0):
        c = _as_seq(col0, "col0")
        if c is None:
            c = []
        r = _as_seq(row0, "row0") if row0 is not None else None
        if r is None:
            r = [c[0]] + [zero] * (len(c) - 1) if c else []

        if c and r and c[0] != r[0]:
            raise ValueError("toeplitz requires col0[0] == row0[0]")

        nrows = len(c)
        ncols = len(r)
        out = _new_empty_matrix(cls, nrows, ncols, null_return=0)

        for i in range(nrows):
            for j in range(ncols):
                v = c[i - j] if i >= j else r[j - i]
                if v != zero:
                    out._data[_spool((i, j), out.shape)] = v
        return out

    @classmethod
    def hankel(cls, col0, row_last=None, zero=0):
        c = _as_seq(col0, "col0")
        if c is None:
            c = []
        r = _as_seq(row_last, "row_last") if row_last is not None else None
        if r is None:
            r = [c[-1]] + [zero] * (len(c) - 1) if c else []

        if c and r and c[-1] != r[0]:
            raise ValueError("hankel requires col0[-1] == row_last[0]")

        nrows = len(c)
        ncols = len(r)
        out = _new_empty_matrix(cls, nrows, ncols, null_return=0)

        for i in range(nrows):
            for j in range(ncols):
                s = i + j
                if s < nrows:
                    v = c[s]
                else:
                    v = r[s - (nrows - 1)]
                if v != zero:
                    out._data[_spool((i, j), out.shape)] = v
        return out

    @classmethod
    def band(cls, nrows, ncols=None, lower=0, upper=0, fill=1, zero=0):
        if ncols is None:
            ncols = nrows
        _validate_pos_int(nrows, "nrows")
        _validate_pos_int(ncols, "ncols")
        _validate_int(lower, "lower")
        _validate_int(upper, "upper")
        if lower < 0 or upper < 0:
            raise ValueError("lower/upper must be >= 0")

        out = _new_empty_matrix(cls, nrows, ncols, null_return=0)
        for i in range(nrows):
            j0 = max(0, i - lower)
            j1 = min(ncols - 1, i + upper)
            for j in range(j0, j1 + 1):
                if fill != zero:
                    out._data[_spool((i, j), out.shape)] = fill
        return out

    @classmethod
    def triu(cls, n, k=0, one=1, zero=0):
        _validate_int(k, "k")
        out = _new_empty_matrix(cls, n, n, null_return=0)
        for i in range(n):
            for j in range(max(0, i + k), n):
                if one != zero:
                    out._data[_spool((i, j), out.shape)] = one
        return out

    @classmethod
    def tril(cls, n, k=0, one=1, zero=0):
        _validate_int(k, "k")
        out = _new_empty_matrix(cls, n, n, null_return=0)
        for i in range(n):
            for j in range(0, min(n, i + k + 1)):
                if one != zero:
                    out._data[_spool((i, j), out.shape)] = one
        return out

    @classmethod
    def shift(cls, n, k=1, one=1, zero=0):
        _validate_int(k, "k")
        out = _new_empty_matrix(cls, n, n, null_return=0)
        for i in range(n):
            j = i + k
            if 0 <= j < n and one != zero:
                out._data[_spool((i, j), out.shape)] = one
        return out

    @classmethod
    def companion(cls, coeffs, one=1, zero=0, monic=True):
        a = _as_seq(coeffs, "coeffs")
        if a is None:
            a = []
        if not a:
            return cls.zeros(0, 0)

        n = len(a)
        out = _new_empty_matrix(cls, n, n, null_return=0)

        for i in range(1, n):
            out._data[_spool((i, i - 1), out.shape)] = one

        if monic:
            last_row = [-ai for ai in a]
        else:
            last_row = [-ai for ai in a]

        for j, v in enumerate(last_row):
            if v != zero:
                out._data[_spool((0, j), out.shape)] = v

        return out
