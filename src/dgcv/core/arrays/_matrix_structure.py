from ..._aux._backends._symbolic_router import conjugate
from ._indexing import _spool, _unspool
from ._validation import _is_array_like


class _matrix_structure:
    def transpose(self):
        def _entry_transpose(x):
            if not _is_array_like(x):
                return x

            t = getattr(x, "transpose", None)
            if callable(t):
                return t()
            elif t is not None:
                return t
            rt = getattr(x, "T", None)
            if callable(rt):
                return rt()
            elif rt is not None:
                return rt
            return x

        out = self.__class__.__new__(self.__class__)
        out._dgcv_class_check = self._dgcv_class_check
        out._dgcv_categories = set(self._dgcv_categories)
        out.shape = (self.ncols, self.nrows)
        out.ndim = 2
        out._data = {}
        out.null_return = self.null_return
        out._engine_representation = dict()
        out._data_unspooled_cache = None

        for k, v in self._data.items():
            i, j = _unspool(k, self.shape)
            out._data[_spool((j, i), out.shape)] = _entry_transpose(v)

        return out

    def conjugate(self, symbolic=False):
        if symbolic is True:
            return self.apply(lambda entry: conjugate(entry, symbolic=True))
        return self.apply(conjugate)

    def conjugate_transpose(self, symbolic=False):
        return self.conjugate(symbolic=symbolic).transpose()

    @property
    def T(self):
        return self.transpose()
