from ..._aux.printing.printing import (
    tensor_latex_alias,
    tensor_latex_helper,
    tensor_VS_printer,
)


class _tp_printing:
    def __str__(self):
        return tensor_VS_printer(self)

    def _latex(self, printer=None, raw=True, **kwargs):
        """
        Defines the LaTeX representation for SymPy's latex() function.
        """
        if kwargs.get("alias", False) and self._properties.get("_hom_id", False):
            return (
                tensor_latex_alias(self._properties["_hom_id"])
                if raw
                else f"$\\displaystyle {tensor_latex_alias(self._properties['_hom_id'])}$"
            )
        return (
            tensor_latex_helper(self)
            if raw
            else f"$\\displaystyle {tensor_latex_helper(self)}$"
        )

    def _repr_latex_(self, raw=False, **kwargs):
        return self._latex(raw=raw, **kwargs)

    def _sympystr(self, printer):
        return self.__repr__()
