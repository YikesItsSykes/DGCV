"""
package: dgcv - Differential Geometry with Complex Variables
module: light_wrappers

Notes: needs update for symbolic router architecture.

Author (of this module): David Sykes (https://realandimaginary.com/dgcv/)

License:
    MIT License
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
import re

from ._config import get_dgcv_settings_registry
from ._dgcv_display import LaTeX
from ._safeguards import retrieve_passkey
from .backends._engine import sympy_module_if_available
from .printing._string_processing import _format_label_with_hi_low

__all__ = ["function_dgcv"]

sp = sympy_module_if_available()


# -----------------------------------------------------------------------------
# body
# -----------------------------------------------------------------------------
class _function_dgcv(sp.Function):
    @classmethod
    def eval(cls, *args):
        return None

    def _sympystr(self, printer):
        return self.func.__name__

    def _latex(self, printer, **kwargs):
        dgcvSR = get_dgcv_settings_registry()
        name = self.func.__name__
        tex = _process_label(name)
        exp = kwargs.get("exp")
        verbosity = dgcvSR.get("__", dict())
        tail = ""
        if dgcvSR.get("verbose_label_printing", False) or "verbose" in verbosity:
            args = self.args
            if args:
                tail = LaTeX(args)
        if exp:
            tex = f"{tex}^{{{exp}}}"
        return tex + tail


def function_dgcv(name: str):
    clsname = str(name)
    cls = type(clsname, (_function_dgcv,), {})
    cls._dgcv_class_check = retrieve_passkey()
    cls._dgcv_category = "function"
    return cls


def _process_label(lbl: str) -> str:
    m = re.search(r"(\d+)$", lbl)
    if m and "_" not in lbl:
        lbl = lbl[: m.start(1)] + "_" + m.group(1)
    return _format_label_with_hi_low(lbl)
