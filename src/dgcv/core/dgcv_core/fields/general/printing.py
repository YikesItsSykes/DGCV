from __future__ import annotations

from ....._aux.printing.printing import tensor_field_latex2, tensor_field_printer2


class _tensor_field_printing:
    def __str__(self):
        return tensor_field_printer2(self)

    def _repr_latex_(self, raw: bool = False, **kwargs):
        return tensor_field_latex2(self, raw=raw)

    def _latex(self, printer=None, raw: bool = True, **kwargs):
        return self._repr_latex_(raw=raw)

    def _latex_(self):
        return self._repr_latex_(raw=True)
