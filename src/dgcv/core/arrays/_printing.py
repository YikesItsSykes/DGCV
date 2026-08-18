from ..._aux._vmf._safeguards import get_variable_registry
from ..._aux.printing.printing import array_latex_helper, array_VS_printer


class _array_printing:
    def __str__(self):
        return array_VS_printer(self)

    def __repr__(self):
        nr = (
            f",\n null_return={repr(self.null_return)}" if self.null_return != 0 else ""
        )
        return f"array_dgcv(\n {repr(self._data)},\n shape = {self.shape}{nr}\n )"

    def _latex(self, printer=None, raw=True, **kwargs):
        s = array_latex_helper(self, **kwargs)
        return s if raw else f"$\\displaystyle {s}$"

    def _repr_latex_(self, raw=False, **kwargs):
        return self._latex(**kwargs)


class _matrix_printing:
    def __str__(self):
        rows = []
        for i in range(self.nrows):
            rows.append(str(self.row(i)))
        if get_variable_registry().get("print_style", None) == "readable":
            return "matrix_dgcv([\n  " + ",\n  ".join(rows) + "\n])"
        return "matrix_dgcv([ " + ", ".join(rows) + "])"

    def __repr__(self):
        nr = (
            f",\n null_return={repr(self.null_return)}" if self.null_return != 0 else ""
        )
        return f"matrix_dgcv(\n {repr(self._data)},\n shape = {self.shape}{nr}\n )"
