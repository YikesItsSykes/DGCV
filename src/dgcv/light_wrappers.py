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
from ._safeguards import retrieve_passkey
from .backends._engine import sympy_module_if_available

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
        name = self.func.__name__
        tex = name
        exp = kwargs.get("exp")
        if exp:
            tex = f"{tex}^{{{exp}}}"
        return tex


def function_dgcv(name: str):
    clsname = str(name)
    cls = type(clsname, (_function_dgcv,), {})
    cls._dgcv_class_check = retrieve_passkey()
    cls._dgcv_category = "function"
    return cls
