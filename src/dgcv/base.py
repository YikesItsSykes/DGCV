"""
package: dgcv - Differential Geometry with Complex Variables
module: base

Author (of this module): David Sykes (https://realandimaginary.com/dgcv/)

License:
    MIT License
"""

# -----------------------------------------------------------------------------
# imports
# -----------------------------------------------------------------------------
from ._config import get_dgcv_settings_registry


# -----------------------------------------------------------------------------
# body
# -----------------------------------------------------------------------------
class dgcv_class:
    def __dgcv_simplify__(self, method=None, **kwargs):
        return self._eval_simplify(**kwargs)

    def _eval_simplify(self, **kwargs):
        return self

    def _latex(self, printer=None, raw: bool = True, **kwargs):
        s = str(self)
        return s if raw else f"$\\displaystyle {s}$"

    def _repr_latex_(self, raw: bool = False, **kwargs):
        s = self._latex(raw=True, **kwargs)
        return s if raw else f"$\\displaystyle {s}$"

    def __str__(self):
        return f"<{self.__class__.__name__}>"

    def __repr__(self):
        dgcvSR = get_dgcv_settings_registry()
        if dgcvSR.get("print_style", None) == "readable":
            return str(self)
        return object.__repr__(self)

    def _repr_mimebundle_(self, include=None, exclude=None):
        try:
            latex = self._repr_latex_(raw=False)
        except Exception:
            latex = None

        try:
            plain = self.__repr__()
        except Exception:
            try:
                plain = str(self)
            except Exception:
                plain = ""

        bundle = {"text/plain": plain if isinstance(plain, str) else str(plain)}
        if isinstance(latex, str) and latex.strip():
            bundle["text/latex"] = latex
        return bundle
