from numbers import Integral

from ..._aux._backends._exact_arith import exact_reciprocal
from ..._aux._backends._symbolic_router import _scalar_is_zero
from ._coercion import _as_matrix_dgcv
from ._indexing import _spool


class _matrix_arithmetic:
    def __add__(self, other):
        if _scalar_is_zero(other):
            return self
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
        out._engine_representation = dict()
        out.null_return = self.null_return
        out._data_unspooled_cache = None

        keys = set(self._data) | set(other._data)
        for k in keys:
            a = self._data.get(k, self.null_return)
            b = other._data.get(k, other.null_return)
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
        out.null_return = self.null_return
        out._engine_representation = dict()
        out._data_unspooled_cache = None
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
        out.null_return = self.null_return
        out._engine_representation = dict()
        out._data_unspooled_cache = None
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
        out.null_return = self.null_return
        out._engine_representation = dict()
        out._data_unspooled_cache = None

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
                    out._data[idx] = out._data.get(idx, out.null_return) + a * b

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
