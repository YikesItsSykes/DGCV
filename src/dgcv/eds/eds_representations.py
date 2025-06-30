import numbers

import sympy as sp

from ..backends._caches import _get_expr_num_types
from .eds import abstDFAtom, abstDFMonom, abstract_DF, abstract_ZF, zeroFormAtom


class DF_representation(sp.Basic):
    def __new__(cls, row_count=None, col_count=None, array_data=None):
        if all(arg is None for arg in [col_count,array_data]):
            if row_count is None:
                array_data = tuple()
            else:
                array_data = row_count  # for optional syntax where only array data is given


        dgcv_classes = [zeroFormAtom, abstDFAtom, abstDFMonom, abstract_DF, abstract_ZF]
        supported_classes = tuple(dgcv_classes)+ _get_expr_num_types()

        if callable(array_data):
            if row_count is None or col_count is None:
                raise ValueError("When using a lambda function for array_data, row_count and col_count must be specified.")
            array_data = tuple(tuple(array_data(j, k) for k in range(col_count)) for j in range(row_count))

        if row_count is None or col_count is None:
            if isinstance(array_data, dict):
                if array_data:
                    row_count = max(key[0] for key in array_data.keys()) + 1
                    col_count = max(key[1] for key in array_data.keys()) + 1
                else:
                    row_count, col_count = 0, 0
            elif isinstance(array_data, (list, tuple)):
                if all(isinstance(entry, (list, tuple)) for entry in array_data):
                    row_count = len(array_data)
                    col_count = max(len(row) for row in array_data) if row_count > 0 else 0
                else:
                    row_count = 1
                    col_count = len(array_data)
                    array_data = (tuple(array_data),)

        def validate_keys(key, r_count, c_count):
            if not (isinstance(key, tuple) and len(key) == 2):
                raise TypeError('Keys must be length-2 integers within specified row and column bounds')
            if any(entry < 0 for entry in key) or c_count <= key[1] or r_count <= key[0]:
                raise TypeError('Keys must be within row and column bounds')
            return True

        sparse_data = tuple()
        if isinstance(array_data, dict) and all(validate_keys(key, row_count, col_count) for key in array_data.keys()):
            if not all(isinstance(value, supported_classes) for value in array_data.values()):
                raise ValueError(f'DF_representation only supports entries from: {", ".join(str(cls) for cls in supported_classes)}')
            sparse_data = tuple({k: v for k, v in array_data.items() if v != 0}.items())
            array_data = tuple(tuple(array_data.get((j, k), 0) for k in range(col_count)) for j in range(row_count))
        elif isinstance(array_data, (list, tuple)) and all(isinstance(entry, (list, tuple)) for entry in array_data):
            if len(array_data) != row_count:
                raise ValueError('Incompatible row count')
            for row in array_data:
                if len(row) != col_count:
                    raise ValueError('Incompatible column count')
                if not all(isinstance(entry, supported_classes) for entry in row):
                    raise ValueError(f'Only supports entries from: {", ".join(str(cls) for cls in supported_classes)}')

            array_data = tuple(tuple(row) for row in array_data)
            sparse_data = {array_data[j][k] for j in range(row_count) for k in range(col_count)}

        obj = sp.Basic.__new__(cls, array_data)
        obj.row_count = row_count
        obj.col_count = col_count
        obj.array = array_data
        obj.sparse_data = sparse_data
        return obj

    def __init__(self, row_count=0, col_count=0, array_data=tuple()):
        pass

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            return self.array[key]  # Allow row-wise access

        if isinstance(row_key, numbers.Integral) and isinstance(col_key, numbers.Integral):
            return self.array[row_key][col_key]
        else:
            new_data = [row[col_key] for row in self.array[row_key]]
            return DF_representation(len(new_data), len(new_data[0]) if new_data else 0, new_data)

    def __iter__(self):
        for row in self.array:
            for entry in row:
                yield entry

    def __eq__(self, other):
        if not isinstance(other, DF_representation):
            return False
        return self.array == other.array

    def __add__(self, other):
        if not isinstance(other, DF_representation):
            raise TypeError("Can only add DF_representation objects")
        if self.row_count != other.row_count or self.col_count != other.col_count:
            raise ValueError("Matrix dimensions must match for addition")
        new_data = [[self.array[i][j] + other.array[i][j] for j in range(self.col_count)] for i in range(self.row_count)]
        return DF_representation(self.row_count, self.col_count, new_data)

    def __sub__(self, other):
        if not isinstance(other, DF_representation):
            raise TypeError("Can only subtract DF_representation objects")
        if self.row_count != other.row_count or self.col_count != other.col_count:
            raise ValueError("Matrix dimensions must match for subtraction")
        new_data = [[self.array[i][j] - other.array[i][j] for j in range(self.col_count)] for i in range(self.row_count)]
        return DF_representation(self.row_count, self.col_count, new_data)

    def __mul__(self, scalar):
        if not isinstance(scalar, (abstract_ZF,zeroFormAtom)) or isinstance(scalar, _get_expr_num_types()):
            raise TypeError("Can only multiply by scalar values")
        new_data = [[scalar * self.array[i][j] for j in range(self.col_count)] for i in range(self.row_count)]
        return DF_representation(self.row_count, self.col_count, new_data)

    __rmul__ = __mul__

    def __matmul__(self, other):
        if not isinstance(other, DF_representation):
            raise TypeError("Can only multiply with another DF_representation")
        if self.col_count != other.row_count:
            raise ValueError("Inner matrix dimensions must match for multiplication")
        new_data = [[sum(self.array[i][k] * other.array[k][j] for k in range(self.col_count)) for j in range(other.col_count)] for i in range(self.row_count)]
        return DF_representation(self.row_count, other.col_count, new_data)

    def __repr__(self):
        return f"DF_representation(row_count={self.row_count}, col_count={self.col_count}, array={self.array})"

    def __str__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.array])

    def flatten(self):
        return [entry for entry in self]

    @property
    def free_symbols(self):
        fs_set = set()
        for entry in self:
            if hasattr(entry,'free_symbols'):
                fs_set = fs_set | entry.free_symbols
        return fs_set

    def _latex(self,printer=None):
        rows = []
        for row in self.array:
            rendered_row = []
            for entry in row:
                if hasattr(entry, "_latex"):
                    latex = sp.latex(entry)
                else:
                    latex = str(entry)
                rendered_row.append(latex)
            rows.append(" & ".join(rendered_row))
        return r"\begin{pmatrix}" + " \\\\ ".join(rows) + r"\end{pmatrix}"

    def _repr_latex_(self):
        rows = []
        for row in self.array:
            rendered_row = []
            for entry in row:
                if hasattr(entry, "_latex"):
                    latex = sp.latex(entry)
                else:
                    latex = str(entry)
                rendered_row.append(latex)
            rows.append(" & ".join(rendered_row))
        latex_str = r"\begin{pmatrix}" + " \\\\ ".join(rows) + r"\end{pmatrix}"
        return "$" + latex_str + "$"

    def applyfunc(self, func):
        new_data = [
            [func(self.array[i][j]) for j in range(self.col_count)]
            for i in range(self.row_count)
        ]
        return DF_representation(self.row_count, self.col_count, new_data)

    def _dgcv_eds_applyfunc(self, func):
        return self.applyfunc(func)

    def _eval_conjugate(self):
        def _custom_conj(expr):
            if isinstance(expr,(abstract_ZF,zeroFormAtom,abstDFAtom,abstDFMonom,abstract_DF)):
                return expr._eval_conjugate()
            else:
                return sp.conjugate(expr)
        return self.applyfunc(_custom_conj)

    def _eval_simplify(self, ratio=None, measure=None, inverse=True, doit=True, rational=True, expand=False, **kwargs):
        return self.applyfunc(lambda entry: sp.simplify(entry,ratio=ratio, measure=measure, inverse=inverse, doit=doit, rational=rational, expand=expand, **kwargs))

    def _eval_canonicalize(self,depth = 1000):
        def _canon(obj):
            if hasattr(obj,'_eval_canonicalize'):
                return obj._eval_canonicalize(depth=depth)
            return obj
        return self.applyfunc(_canon)


    def _subs_dgcv(self, data, with_diff_corollaries=False):
        # an alias for regular subs so that other functions can know the with_diff_corollaries keyword is available
        return self.subs(data, with_diff_corollaries = with_diff_corollaries)

    def subs(self,subs_data,with_diff_corollaries=False):
        def _custom_subs(expr):
            if hasattr(expr,'_subs_dgcv'):
                return expr._subs_dgcv(subs_data,with_diff_corollaries=with_diff_corollaries)
            if hasattr(expr,'subs'):
                return expr.subs(subs_data)
            else:
                return expr
        return self.applyfunc(_custom_subs)

