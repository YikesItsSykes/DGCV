import sympy as sp

from ._safeguards import retrieve_passkey

__all__ = ["function_dgcv"]



class _function_dgcv(sp.Function):
    @classmethod
    def eval(cls, *args):
        return None

    def _sympystr(self, printer):
        return self.func.__name__

    def _latex(self, printer, **kwargs):
        name = self.func.__name__
        tex = name
        exp = kwargs.get('exp')
        if exp:
            tex = f"{tex}^{{{exp}}}"
        return tex

def function_dgcv(name: str):
    clsname = str(name)
    cls = type(clsname, (_function_dgcv,), {})
    cls._dgcv_class_check = retrieve_passkey()
    cls._dgcv_category = "function"
    return cls
