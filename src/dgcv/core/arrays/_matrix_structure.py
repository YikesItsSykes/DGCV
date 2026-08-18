from ..._aux._backends._symbolic_router import conjugate, subs
from ..._aux._backends._types_and_constants import imag_unit
from ..._aux._vmf._safeguards import check_dgcv_category, get_variable_registry
from ._indexing import _spool, _unspool


class _matrix_structure:
    def transpose(self):
        def _entry_transpose(x):
            if x is None:
                return x

            is_dgcv = check_dgcv_category(x) is not None

            if is_dgcv:
                t = getattr(x, "transpose", None)
                if callable(t):
                    return t()
                elif t is not None:
                    return t
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
            cd = dict(
                get_variable_registry()["conversion_dictionaries"]["conjugation"]
            ) | {imag_unit(): -imag_unit()}
            return subs(self, cd, simultaneous=True)
        return self.apply(conjugate)

    def conjugate_transpose(self, symbolic=False):
        return self.conjugate(symbolic=symbolic).transpose()

    @property
    def T(self):
        return self.transpose()
