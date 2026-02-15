"""
package: dgcv - Differential Geometry with Complex Variables
module: arrays

Author (of this module): David Sykes (https://realandimaginary.com/dgcv/)

License:
    MIT License
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
from numbers import Integral

from ._safeguards import check_dgcv_category, create_key, retrieve_passkey
from .backends._engine import engine_kind, engine_module
from .backends._exact_arith import exact_reciprocal
from .backends._symbolic_router import (
    _scalar_is_zero,
    conjugate,
    get_free_symbols,
    simplify,
)
from .backends._types_and_constants import fast_scalar_types, symbol
from .base import dgcv_class
from .printing import array_latex_helper, array_VS_printer

__all__ = ["array_dgcv", "matrix_dgcv"]


# -----------------------------------------------------------------------------
# arrays
# -----------------------------------------------------------------------------
def _spool(multi_index, shape):
    idx = 0
    stride = 1
    for i, s in zip(reversed(multi_index), reversed(shape)):
        idx += i * stride
        stride *= s
    return idx


def _unspool(index, shape):
    multi = []
    for s in reversed(shape):
        multi.append(index % s)
        index //= s
    return tuple(reversed(multi))


class array_dgcv(dgcv_class):
    _dgcv_category = "array"

    def __init__(self, array_data, *, shape=None):
        if shape is not None:
            if not isinstance(shape, tuple) or not all(
                isinstance(s, Integral) and s >= 0 for s in shape
            ):
                raise TypeError("shape must be a tuple of non-negative ints")
            self.shape = shape

        self._data, self.shape = self._normalize(array_data)
        self.ndim = len(self.shape)
        self._dgcv_class_check = retrieve_passkey()

    def __str__(self):
        return array_VS_printer(self)

    def _latex(self, printer=None, raw=True, **kwargs):
        s = array_latex_helper(self, **kwargs)
        return s if raw else f"$\\displaystyle {s}$"

    def _repr_latex_(self, raw=False, **kwargs):
        return self._latex(**kwargs)

    def _normalize(self, array_data):
        if isinstance(array_data, dict):
            shp = getattr(self, "shape", None)
            if not isinstance(shp, tuple):
                raise TypeError("dict input requires shape=... in array_dgcv(...)")
            return dict(array_data), tuple(shp)

        if isinstance(array_data, (list, tuple)):
            shape = self._infer_shape(array_data)
            flat = {}
            self._fill(flat, array_data, shape, shape_full=shape)
            return flat, shape

        if isinstance(array_data, array_dgcv):
            return dict(array_data._data), tuple(array_data.shape)

        rows = getattr(array_data, "rows", None)
        cols = getattr(array_data, "cols", None)
        if isinstance(rows, Integral) and isinstance(cols, Integral):
            shape = (rows, cols)
            flat = {}
            for i in range(rows):
                base = i * cols
                for j in range(cols):
                    flat[base + j] = array_data[i, j]
            return flat, shape
        nrows = getattr(array_data, "nrows", None)
        ncols = getattr(array_data, "ncols", None)
        if callable(nrows) and callable(ncols):
            r = int(nrows())
            c = int(ncols())
            shape = (r, c)
            flat = {}
            for i in range(r):
                base = i * c
                for j in range(c):
                    flat[base + j] = array_data[i, j]
            return flat, shape

        raise TypeError("Unsupported array_data type")

    def _infer_shape(self, data):
        shape = []
        cur = data
        while isinstance(cur, (list, tuple)):
            shape.append(len(cur))
            children = [x for x in cur if isinstance(x, (list, tuple))]
            if not children:
                break
            cur = max(children, key=len)
        return tuple(shape)

    def _fill(self, flat, data, shape, shape_full, prefix=()):
        if not shape:
            flat[_spool(prefix, shape_full)] = data
            return
        if not isinstance(data, (list, tuple)):
            flat[_spool(prefix, shape_full)] = data
            return
        for i, val in enumerate(data):
            self._fill(flat, val, shape[1:], shape_full, prefix + (i,))

    def __getitem__(self, key):
        idx = _spool(key, self.shape) if isinstance(key, tuple) else key
        return self._data.get(idx, 0)

    def __setitem__(self, key, value):
        idx = _spool(key, self.shape) if isinstance(key, tuple) else key
        self._data[idx] = value

    def __len__(self):
        total = 1
        for s in self.shape:
            total *= s
        return total

    def __iter__(self):
        shp = getattr(self, "shape", None)
        if not isinstance(shp, tuple):
            raise TypeError("array_dgcv is missing a valid shape")

        if not shp:
            yield self._data.get(0, 0)
            return

        n = 1
        for s in shp:
            n *= s

        d = self._data
        for k in range(n):
            v = d.get(k, 0)
            yield 0 if v is None else v

    def iter_nonzero_items(self, *, include_zeros=False, include_none=False):
        shp = getattr(self, "shape", None)
        if not isinstance(shp, tuple):
            raise TypeError("array_dgcv is missing a valid shape")

        for k, v in self._data.items():
            if v is None:
                if include_none:
                    yield (_unspool(k, shp), v)
                continue

            if (not include_zeros) and _scalar_is_zero(v):
                continue

            yield (_unspool(k, shp), v)

    @property
    def free_symbols(self):
        out = set()
        for v in self._data.values():
            if v is None:
                continue
            fs = get_free_symbols(v)
            if fs is not None:
                out |= set(fs)
        return out

    @property
    def __dgcv_zero_obstr__(self):
        eqns = [v for v in self._data.values() if v is not None]
        return eqns, list(self.free_symbols)

    def apply(self, func, *, in_place=False, skip_none=True, default=None):
        if in_place:
            target = self
        else:
            target = self.__class__.__new__(self.__class__)
            target._data = {}
            target.shape = tuple(self.shape)
            target.ndim = self.ndim

            if hasattr(self, "_dgcv_class_check"):
                target._dgcv_class_check = self._dgcv_class_check
            if hasattr(self, "_dgcv_category"):
                target._dgcv_category = self._dgcv_category
            if hasattr(self, "_dgcv_categories"):
                target._dgcv_categories = set(self._dgcv_categories)
        if skip_none:
            for k, v in self._data.items():
                target._data[k] = None if v is None else func(v)
        else:
            n = 1
            for s in self.shape:
                n *= s
            for k in range(n):
                v = self._data.get(k, default)
                target._data[k] = func(v)

        return target

    def subs(self, *args, **kwargs):
        def f(x):
            if x is None:
                return None
            m = getattr(x, "subs", None)
            return m(*args, **kwargs) if callable(m) else x

        return self.apply(f, in_place=False, skip_none=True)

    def __dgcv_simplify__(self, *args, **kwargs):
        return self.apply(simplify, in_place=False, skip_none=True)


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


class matrix_dgcv(array_dgcv):
    """
    A general 2-d array structure with convenient properties for dgcv. It can be used for storing and displaying data, or more linear algebra intensive applications with standard matrix arithmetic. Although built-in methods for the latter applications are limited as no assumptions are enforced about object types in the array's entries.

    Parameters
    ----------
    array_data : list of lists, tuple of tuples, various matrix/array classes
        Data defining a 2-dimensional array

    Notes
    -----
    Types for entries are not restricted, but several class methods (matrix multiplication, scalar multiplication, addition, etc.) are written with the assumption that entry types behave as elements in some algebra, i.e., they need to have methods __add__, __mul__, etc. enabeling scalar multiplication and multiplication and addition between them.
    """

    _dgcv_categories = {"matrix"}

    def __init__(self, array_data):
        super().__init__(array_data)
        if self.ndim == 1:
            n = self.shape[0]
            new_data = {}
            for k, v in self._data.items():
                if v is None:
                    continue
                i = k
                new_data[_spool((i, 0), (n, 1))] = v
            self._data = new_data
            self.shape = (n, 1)
            self.ndim = 2
        if self.ndim != 2:
            raise ValueError("matrix_dgcv requires 2-dimensional data")
        self._dgcv_categories = {"matrix"}
        self._engine_representation = dict()

    @property
    def nrows(self):
        return self.shape[0]

    @property
    def ncols(self):
        return self.shape[1]

    def row(self, i):
        return [self[i, j] for j in range(self.ncols)]

    def col(self, j):
        return [self[i, j] for i in range(self.nrows)]

    def __getitem__(self, key):
        v = super().__getitem__(key)
        return 0 if v is None else v

    def __setitem__(self, key, value):
        rep = getattr(self, "_engine_representation", None)
        if isinstance(rep, dict):
            rep.clear()
        else:
            self._engine_representation = {}
        idx = _spool(key, self.shape) if isinstance(key, tuple) else key
        self._data[idx] = value

    def __str__(self):
        rows = []
        for i in range(self.nrows):
            rows.append(str(self.row(i)))
        return "matrix_dgcv([\n  " + ",\n  ".join(rows) + "\n])"

    def copy(self):
        out = self.__class__.__new__(self.__class__)
        out._dgcv_class_check = self._dgcv_class_check
        out._dgcv_categories = set(self._dgcv_categories)
        out.shape = tuple(self.shape)
        out.ndim = 2
        out._data = dict(self._data)
        return out

    @classmethod
    def identity(cls, n, one=1, zero=0):
        out = cls.__new__(cls)
        out._dgcv_class_check = retrieve_passkey()
        out._dgcv_categories = {"matrix"}
        out.shape = (n, n)
        out.ndim = 2
        out._data = {}
        for i in range(n):
            out._data[_spool((i, i), out.shape)] = one
        return out

    def transpose(self):
        def _entry_transpose(x):
            if x is None:
                return x

            try:
                is_dgcv = check_dgcv_category(x) is not None
            except Exception:
                is_dgcv = False

            if is_dgcv:
                t = getattr(x, "transpose", None)
                if callable(t):
                    try:
                        return t()
                    except Exception:
                        return x

            t = getattr(x, "transpose", None)
            if callable(t):
                try:
                    return t()
                except Exception:
                    pass

            rt = getattr(x, "T", None)
            if rt is not None:
                try:
                    return rt
                except Exception:
                    pass

            return x

        out = self.__class__.__new__(self.__class__)
        out._dgcv_class_check = self._dgcv_class_check
        out._dgcv_categories = set(self._dgcv_categories)
        out.shape = (self.ncols, self.nrows)
        out.ndim = 2
        out._data = {}

        for k, v in self._data.items():
            i, j = _unspool(k, self.shape)
            out._data[_spool((j, i), out.shape)] = _entry_transpose(v)

        return out

    def conjugate(self):
        return self.apply(conjugate)

    def conjugate_transpose(self):
        return self.apply(conjugate).transpose()

    @property
    def T(self):
        return self.transpose()

    def __add__(self, other):
        other = _as_matrix_dgcv(other)
        if other is None:
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Matrix shapes do not match for addition")

        out = self.__class__.__new__(self.__class__)
        out._dgcv_class_check = self._dgcv_class_check
        out._dgcv_categories = set(self._dgcv_categories)
        out.shape = self.shape
        out.ndim = 2
        out._data = {}

        keys = set(self._data) | set(other._data)
        for k in keys:
            a = self._data.get(k, 0)
            b = other._data.get(k, 0)
            v = a + b
            if not _scalar_is_zero(v):
                out._data[k] = v
        return out

    def __neg__(self):
        return (-1) * self

    def __sub__(self, other):
        return self.__add__((-1) * other)

    def __rsub__(self, other):
        return ((-1) * self).__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other_m = _as_matrix_dgcv(other)
        if other_m is not None:
            return self.__matmul__(other_m)

        out = self.__class__.__new__(self.__class__)
        out._dgcv_class_check = self._dgcv_class_check
        out._dgcv_categories = set(self._dgcv_categories)
        out.shape = self.shape
        out.ndim = 2
        out._data = {}
        for k, v in self._data.items():
            out._data[k] = v * other
        return out

    def __rmul__(self, other):
        out = self.__class__.__new__(self.__class__)
        out._dgcv_class_check = self._dgcv_class_check
        out._dgcv_categories = set(self._dgcv_categories)
        out.shape = self.shape
        out.ndim = 2
        out._data = {}
        for k, v in self._data.items():
            if v is not None:
                out._data[k] = other * v
        return out

    def __truediv__(self, other):
        return self.__mul__(exact_reciprocal(other))

    def __matmul__(self, other):
        other_m = _as_matrix_dgcv(other)
        if other_m is None:
            return NotImplemented
        if self.ncols != other_m.nrows:
            raise ValueError("Matrix shapes do not align for multiplication")

        out = self.__class__.__new__(self.__class__)
        out._dgcv_class_check = self._dgcv_class_check
        out._dgcv_categories = set(self._dgcv_categories)
        out.shape = (self.nrows, other_m.ncols)
        out.ndim = 2
        out._data = {}

        for i in range(self.nrows):
            for k in range(self.ncols):
                a = self[i, k]
                if _scalar_is_zero(a):
                    continue
                for j in range(other_m.ncols):
                    b = other_m[k, j]
                    if _scalar_is_zero(b):
                        continue
                    idx = _spool((i, j), out.shape)
                    out._data[idx] = out._data.get(idx, 0) + a * b

        return out

    def __rmatmul__(self, other):
        left = _as_matrix_dgcv(other)
        if left is None:
            return NotImplemented
        return left.__matmul__(self)

    def __pow__(self, n):
        if not isinstance(n, Integral):
            raise TypeError("Matrix exponent must be an integer")
        if self.nrows != self.ncols:
            raise ValueError("Matrix power only defined for square matrices")

        if n == 0:
            return self.identity(self.nrows)

        if n < 0:
            inv = self.inverse()
            return inv ** (-n)

        result = self.identity(self.nrows)
        base = self
        while n > 0:
            if n & 1:
                result = result @ base
            base = base @ base
            n >>= 1
        return result

    def det(self):
        if self.nrows != self.ncols:
            raise ValueError("det only defined for square matrices")

        n = self.nrows
        if n == 0:
            return 1
        if n == 1:
            return self[0, 0]

        A = [[self[i, j] for j in range(n)] for i in range(n)]
        prev_pivot = 1
        sign = 1

        for k in range(n - 1):
            pivot_row = None
            for r in range(k, n):
                if not _scalar_is_zero(A[r][k]):
                    pivot_row = r
                    break
            if pivot_row is None:
                return 0
            if pivot_row != k:
                A[k], A[pivot_row] = A[pivot_row], A[k]
                sign = -sign

            pivot = A[k][k]
            for i in range(k + 1, n):
                aik = A[i][k]
                for j in range(k + 1, n):
                    A[i][j] = (A[i][j] * pivot - aik * A[k][j]) / prev_pivot
                A[i][k] = 0

            prev_pivot = pivot
            if _scalar_is_zero(prev_pivot):
                return 0

        return sign * A[n - 1][n - 1]

    def inverse(self):
        if self.nrows != self.ncols:
            raise ValueError("inverse only defined for square matrices")

        n = self.nrows
        A = [[self[i, j] for j in range(n)] for i in range(n)]
        shell = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            shell[i][i] = 1

        for col in range(n):
            pivot_row = None
            for r in range(col, n):
                if not _scalar_is_zero(A[r][col]):
                    pivot_row = r
                    break
            if pivot_row is None:
                raise ZeroDivisionError("matrix is singular")

            if pivot_row != col:
                A[col], A[pivot_row] = A[pivot_row], A[col]
                shell[col], shell[pivot_row] = shell[pivot_row], shell[col]

            pivot = A[col][col]
            inv_pivot = exact_reciprocal(pivot)

            for j in range(n):
                A[col][j] = A[col][j] * inv_pivot
                shell[col][j] = shell[col][j] * inv_pivot

            for r in range(n):
                if r == col:
                    continue
                factor = A[r][col]
                if _scalar_is_zero(factor):
                    continue
                for j in range(n):
                    A[r][j] = A[r][j] - factor * A[col][j]
                    shell[r][j] = shell[r][j] - factor * shell[col][j]

        return self.__class__(shell)

    @classmethod
    def zeros(cls, nrows, ncols=None, zero=0):
        if ncols is None:
            ncols = nrows
        out = _new_empty_matrix(cls, nrows, ncols)
        return out

    @classmethod
    def ones(cls, nrows, ncols=None, one=1):
        if ncols is None:
            ncols = nrows
        out = _new_empty_matrix(cls, nrows, ncols)
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

        out = _new_empty_matrix(cls, nrows, ncols)
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
        out = _new_empty_matrix(cls, nrows, ncols)

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
        out = _new_empty_matrix(cls, nrows, ncols)

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

        out = _new_empty_matrix(cls, nrows, ncols)
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
        out = _new_empty_matrix(cls, n, n)
        for i in range(n):
            for j in range(max(0, i + k), n):
                if one != zero:
                    out._data[_spool((i, j), out.shape)] = one
        return out

    @classmethod
    def tril(cls, n, k=0, one=1, zero=0):
        _validate_int(k, "k")
        out = _new_empty_matrix(cls, n, n)
        for i in range(n):
            for j in range(0, min(n, i + k + 1)):
                if one != zero:
                    out._data[_spool((i, j), out.shape)] = one
        return out

    @classmethod
    def shift(cls, n, k=1, one=1, zero=0):
        _validate_int(k, "k")
        out = _new_empty_matrix(cls, n, n)
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
        out = _new_empty_matrix(cls, n, n)

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

    def trace(self):
        if self.nrows != self.ncols:
            raise ValueError("trace is only defined for square matrices")
        s = 0
        for i in range(self.nrows):
            s += self[i, i]
        return s

    def tolist(self):
        return [[self[i, j] for j in range(self.ncols)] for i in range(self.nrows)]

    def augment_col(self, b):
        bM = b if isinstance(b, matrix_dgcv) else matrix_dgcv(b)
        if bM.ncols != 1 or bM.nrows != self.nrows:
            raise ValueError("b must be a column vector with matching nrows")
        rows = [
            [self[i, j] for j in range(self.ncols)] + [bM[i, 0]]
            for i in range(self.nrows)
        ]
        return matrix_dgcv(rows)

    def _dense_copy(self):
        return [[self[i, j] for j in range(self.ncols)] for i in range(self.nrows)]

    def _make_is_zero(self):
        def _scalar_is_zero(x):
            return _scalar_is_zero(x)

    def _make_inv(self, *, record_divisors=False, allow_formal=True):
        if record_divisors:
            divisors = []

            def _inv(d):
                divisors.append(d)
                try:
                    return exact_reciprocal(d)
                except Exception:
                    if allow_formal:
                        return 1 / d
                    raise

            return _inv, divisors

        def _inv(d):
            try:
                return exact_reciprocal(d)
            except Exception:
                if allow_formal:
                    return 1 / d
                raise

        return _inv, None

    def _elim_core(
        self,
        A,
        *,
        rhs_cols=0,
        record_divisors=False,
        allow_formal_inverse=True,
        simplify_steps=False,
        fast_only=False,
        want_pivmap=False,
    ):
        m = len(A)
        n_total = len(A[0]) if m else 0
        n = n_total - rhs_cols

        if m == 0 or n == 0:
            if want_pivmap:
                return 0, {}, [], [] if record_divisors else None
            return 0, [] if record_divisors else None

        try:
            _inv, divisors = self._make_inv(
                record_divisors=record_divisors, allow_formal=allow_formal_inverse
            )
        except Exception:
            raise

        r = 0
        pivcol_to_row = {}
        pivot_cols = []

        for c in range(n):
            pivot_row = None
            for rr in range(r, m):
                if not _scalar_is_zero(A[rr][c]):
                    pivot_row = rr
                    break
            if pivot_row is None:
                continue

            if pivot_row != r:
                A[r], A[pivot_row] = A[pivot_row], A[r]

            pv = A[r][c]
            try:
                inv_pv = _inv(pv)
            except Exception:
                if fast_only or fast_only is None:
                    raise
                raise RuntimeError(
                    "matrix_dgcv: pivot is not invertible in the current coefficient domain."
                ) from None

            for j in range(c, n_total):
                A[r][j] = A[r][j] * inv_pv
            if simplify_steps and not fast_only:
                for j in range(c, n_total):
                    A[r][j] = simplify(A[r][j])

            for rr in range(m):
                if rr == r:
                    continue
                f = A[rr][c]
                if _scalar_is_zero(f):
                    continue
                for j in range(c, n_total):
                    A[rr][j] = A[rr][j] - f * A[r][j]
                if simplify_steps and not fast_only:
                    for j in range(c, n_total):
                        A[rr][j] = simplify(A[rr][j])

            pivcol_to_row[c] = r
            pivot_cols.append(c)
            r += 1
            if r == m:
                break

        if want_pivmap:
            return r, pivcol_to_row, pivot_cols, divisors
        return r, divisors

    def rank(self):
        if self.nrows == 0 or self.ncols == 0:
            return 0

        fast_types = fast_scalar_types()
        fast_case = all(isinstance(a, fast_types) for a in self._data.values())

        A = self._dense_copy()
        r, _ = self._elim_core(
            A,
            rhs_cols=0,
            record_divisors=False,
            allow_formal_inverse=False,
            simplify_steps=False,
            fast_only=fast_case,
            want_pivmap=False,
        )
        return r

    def nullspace(self):
        m = self.nrows
        n = self.ncols

        if n == 0:
            return []
        if m == 0:
            out = []
            for j in range(n):
                v = self.__class__.zeros(n, 1)
                v._data[_spool((j, 0), v.shape)] = 1
                out.append(v)
            return out

        A = self._dense_copy()
        _, pivcol_to_row, pivot_cols, _ = self._elim_core(
            A,
            rhs_cols=0,
            record_divisors=False,
            allow_formal_inverse=False,
            simplify_steps=False,
            fast_only=False,
            want_pivmap=True,
        )

        free_cols = [j for j in range(n) if j not in pivcol_to_row]
        if not free_cols:
            return []

        out = []
        for fc in free_cols:
            vec = [0] * n
            vec[fc] = 1
            for pc in pivot_cols:
                pr = pivcol_to_row[pc]
                vec[pc] = -A[pr][fc]

            v = self.__class__.zeros(n, 1)
            for i, val in enumerate(vec):
                if not _scalar_is_zero(val):
                    v._data[_spool((i, 0), v.shape)] = val
            out.append(v)

        return out

    def solve_right(
        self,
        b,
        *,
        return_divisors=False,
        simplify_steps=False,
        dedupe_divisors=False,
        allow_formal_inverse=True,
        return_parametric=True,
        parametric_vars=None,
    ):
        if self.nrows == 0:
            sol = [] if (getattr(b, "nrows", 0) == 0) else None
            return (sol, []) if return_divisors else sol

        bM = b if isinstance(b, matrix_dgcv) else matrix_dgcv(b)
        if bM.ncols != 1 or bM.nrows != self.nrows:
            raise ValueError("b must be a column vector with matching nrows")

        fast_types = fast_scalar_types()
        fast_case = all(isinstance(a, fast_types) for a in self._data.values()) and all(
            isinstance(v, fast_types) for v in bM._data.values()
        )

        m = self.nrows
        n = self.ncols
        A = [[self[i, j] for j in range(n)] + [bM[i, 0]] for i in range(m)]

        r, pivcol_to_row, _, divisors = self._elim_core(
            A,
            rhs_cols=1,
            record_divisors=return_divisors,
            allow_formal_inverse=allow_formal_inverse,
            simplify_steps=simplify_steps,
            fast_only=fast_case,
            want_pivmap=True,
        )

        for rr in range(m):
            all0 = True
            for c in range(n):
                if not _scalar_is_zero(A[rr][c]):
                    all0 = False
                    break
            if all0 and not _scalar_is_zero(A[rr][n]):
                out = None
                if return_divisors:
                    divs = divisors or []
                    if dedupe_divisors and divs:
                        seen = set()
                        dd = []
                        for d in divs:
                            if d in seen:
                                continue
                            seen.add(d)
                            dd.append(d)
                        return out, dd
                    return out, divs
                return out

        if parametric_vars is None:
            prefix = create_key("_x", True, 4)
            params = [symbol(f"{prefix}_{i}") for i in range(n)]
        else:
            params = list(parametric_vars)
            if len(params) != n:
                raise ValueError("parametric_vars must have length equal to ncols")

        free_cols = [c for c in range(n) if c not in pivcol_to_row]
        x = [0] * n
        for fc in free_cols:
            x[fc] = params[fc]

        for pc in sorted(pivcol_to_row):
            pr = pivcol_to_row[pc]
            val = A[pr][n]
            for fc in free_cols:
                val = val - A[pr][fc] * x[fc]
            if simplify_steps and not fast_case:
                val = simplify(val)
            x[pc] = val

        out = x

        if return_divisors:
            divs = divisors or []
            if dedupe_divisors and divs:
                seen = set()
                dd = []
                for d in divs:
                    if d in seen:
                        continue
                    seen.add(d)
                    dd.append(d)
                return out, dd
            return out, divs
        return out

    def try_solve_right(self, b):
        return self.solve_right(b, return_divisors=False)

    def _to_engine_matrix(self, kind: str | None = None):
        if kind is None:
            kind = engine_kind()
        if kind not in ("sage", "sympy"):
            raise RuntimeError(f"Unsupported engine kind {kind!r}")

        rep = self._engine_representation.get(kind, None)
        if rep is not None:
            return rep

        mod = engine_module()
        if mod is None:
            raise RuntimeError("No symbolic engine is available.")

        rows = [[self[i, j] for j in range(self.ncols)] for i in range(self.nrows)]

        if kind == "sage":
            M = mod.matrix(rows)
        else:
            M = mod.Matrix(rows)

        self._engine_representation[kind] = M
        return M

    def _eigenvals_dict_by_engine(self, *, kind: str | None = None) -> dict:
        if kind is None:
            kind = engine_kind()
        M = self._to_engine_matrix(kind=kind)

        if kind == "sage":
            vals = M.eigenvalues()
            out = {}
            for v in vals:
                out[v] = out.get(v, 0) + 1
            return out

        ev = M.eigenvals()
        if isinstance(ev, dict):
            return dict(ev)
        out = {}
        try:
            for v in list(ev):
                out[v] = out.get(v, 0) + 1
        except Exception:
            pass
        return out

    def _eigenvects_by_engine(self, *, kind: str | None = None):
        if kind is None:
            kind = engine_kind()
        M = self._to_engine_matrix(kind=kind)

        if kind == "sage":
            data = M.eigenvectors_right()
            out = []
            for lam, vecs, mult in data:
                vv = []
                for v in vecs:
                    vv.append(matrix_dgcv.col_vector(list(v)))
                out.append((lam, int(mult), vv))
            return out

        data = M.eigenvects()
        out = []
        for lam, mult, vecs in data:
            vv = []
            for v in vecs:
                vv.append(matrix_dgcv(v))
            out.append((lam, int(mult), vv))
        return out


def _validate_pos_int(n, name="n"):
    if not isinstance(n, Integral) or n < 0:
        raise TypeError(f"{name} must be a nonnegative integer, got {n!r}")


def _validate_int(i, name="i"):
    if not isinstance(i, Integral):
        raise TypeError(f"{name} must be an integer, got {i!r}")


def _as_seq(x, name="data"):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    raise TypeError(f"{name} must be a list/tuple, got {type(x).__name__}")


def _new_empty_matrix(cls, nrows, ncols, passkey=None):
    _validate_pos_int(nrows, "nrows")
    _validate_pos_int(ncols, "ncols")
    out = cls.__new__(cls)
    out._dgcv_class_check = retrieve_passkey() if passkey is None else passkey
    out._dgcv_categories = {"matrix"}
    out.shape = (nrows, ncols)
    out.ndim = 2
    out._data = {}
    return out
