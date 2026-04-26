"""
package: dgcv - Differential Geometry with Complex Variables
module: dgcv.light_wrappers

Notes: needs update for symbolic router architecture.

Author (of this module): David Gamble Sykes
Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes
Licensed under the Apache License, Version 2.0
SPDX-License-Identifier: Apache-2.0
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
import re

from .._aux._backends._engine import sympy_module_if_available
from .._aux._utilities._config import get_dgcv_settings_registry
from .._aux._vmf._safeguards import retrieve_passkey
from .._aux.printing.printing._dgcv_display import LaTeX
from .._aux.printing.printing._string_processing import _format_label_with_hi_low

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
