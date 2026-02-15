"""
package: dgcv - Differential Geometry with Complex Variables
module: _config

Description: This module provides utility functions for dgcv's Variable Management Framework, maintaining
a registry of instantiated mathematical object for interaction with dgcv functions.

Functions
---------
- get_variable_registry: Returns the current state of the `variable_registry`,
  which holds information about objects tracked in the VMF.
- clear_variable_registry: Resets the `variable_registry` to its initial state.
- get_dgcv_settings_registry: Returns the current state of the dictionary storing setting default affecting dgcv functions.

Author (of this module): David Sykes (https://realandimaginary.com/dgcv/)

License:
    MIT License
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
from __future__ import annotations

import collections.abc
import inspect
import re
import warnings
from typing import List, Literal, Optional

from dgcv import __version__

_cached_caller_globals = None

dgcv_categories = {
    "vector_field",
    "tensor_field",
    "differential_form",
    "algebra",
    "algebra_element",
    "algebra_subspace",
    "subalgebra",
    "subalgebra_element",
    "vectorSpace",
    "vector_space_element",
}

greek_letters = {
    "alpha": "\\alpha",
    "beta": "\\beta",
    "gamma": "\\gamma",
    "delta": "\\delta",
    "epsilon": "\\epsilon",
    "varepsilon": "\\varepsilon",
    "zeta": "\\zeta",
    "eta": "\\eta",
    "theta": "\\theta",
    "vartheta": "\\vartheta",
    "iota": "\\iota",
    "kappa": "\\kappa",
    "lambda": "\\lambda",
    "mu": "\\mu",
    "nu": "\\nu",
    "xi": "\\xi",
    "pi": "\\pi",
    "varpi": "\\varpi",
    "rho": "\\rho",
    "varrho": "\\varrho",
    "sigma": "\\sigma",
    "varsigma": "\\varsigma",
    "tau": "\\tau",
    "upsilon": "\\upsilon",
    "phi": "\\phi",
    "varphi": "\\varphi",
    "chi": "\\chi",
    "psi": "\\psi",
    "omega": "\\omega",
    "Gamma": "\\Gamma",
    "Delta": "\\Delta",
    "Theta": "\\Theta",
    "Lambda": "\\Lambda",
    "Xi": "\\Xi",
    "Pi": "\\Pi",
    "Sigma": "\\Sigma",
    "Upsilon": "\\Upsilon",
    "Phi": "\\Phi",
    "Psi": "\\Psi",
    "Omega": "\\Omega",
    "ell": "\\ell",
    "hbar": "\\hbar",
}

__all__ = [
    "get_variable_registry",
    "get_dgcv_settings_registry",
    "configure_convenient_labels",
]


# -----------------------------------------------------------------------------
# body
# -----------------------------------------------------------------------------
def get_caller_globals():
    """
    Retrieve and cache the caller's global namespace.

    If not already cached, searches through the call stack to locate the global namespace of
    the `__main__` module and caches it. Then returns the cached dict.

    Returns
    -------
    dict or None
        The global namespace of the `__main__` module, or None if not found.

    Raises
    ------
    RuntimeError
        If the `__main__` module is not found in the call stack.
    """
    global _cached_caller_globals
    if _cached_caller_globals is not None:
        return _cached_caller_globals

    for frame_info in inspect.stack():
        if frame_info.frame.f_globals["__name__"] == "__main__":
            _cached_caller_globals = frame_info.frame.f_globals
            return _cached_caller_globals

    raise RuntimeError("Could not find the '__main__' module in the call stack.")


def cache_globals():
    """
    Initialize the global namespace cache.

    This function is intended to be called at package import to initialize and cache the
    global namespace for use with the VMF.
    """
    if _cached_caller_globals is None:
        get_caller_globals()


def configure_warnings():
    warnings.simplefilter("once")  # Only show each warning once

    # Optionally customize the format
    def custom_format_warning(
        message, category, filename, lineno, file=None, line=None
    ):
        return f"{category.__name__}: {message}\n"

    warnings.formatwarning = custom_format_warning


class StringifiedSymbolsDict(collections.abc.MutableMapping):
    """
    A lightweight dictionary that stores keys as their string representations.
    When setting or getting an item with a key, it is converted to its string form.
    """

    def __init__(self, initial_data=None):
        self._data = {}
        if initial_data:
            self.update(initial_data)

    def _convert_key(self, key):
        return key if isinstance(key, str) else str(key)

    def __getitem__(self, key):
        return self._data[self._convert_key(key)]

    def __setitem__(self, key, value):
        self._data[self._convert_key(key)] = value

    def __delitem__(self, key):
        del self._data[self._convert_key(key)]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def copy(self):
        new_copy = StringifiedSymbolsDict()
        new_copy._data = self._data.copy()
        return new_copy

    def __repr__(self):
        return f"StringifiedSymbolsDict({self._data})"


variable_registry = {
    "standard_variable_systems": {},
    "complex_variable_systems": {},
    "finite_algebra_systems": {},
    "misc": {},
    "eds": {"atoms": {}, "coframes": {}},
    "protected_variables": set(),
    "temporary_variables": set(),
    "obscure_variables": set(),
    "conversion_dictionaries": {
        "holToReal": StringifiedSymbolsDict(),
        "realToSym": StringifiedSymbolsDict(),
        "symToHol": StringifiedSymbolsDict(),
        "symToReal": StringifiedSymbolsDict(),
        "realToHol": StringifiedSymbolsDict(),
        "conjugation": StringifiedSymbolsDict(),
        "find_parents": StringifiedSymbolsDict(),
        "real_part": StringifiedSymbolsDict(),
        "im_part": StringifiedSymbolsDict(),
    },
    "dgcv_enforced_real_atoms": dict(),  # only for symbolic engines not supporting complex variables logic
    "_labels": {},
    "paths": {},
}
vlp = re.compile(
    r"""
    ^(?:\\left\((?P<content>.*)\\right\))?   
    _\{\\operatorname\{v\.\}(?P<j>\d+)\}$    
    """,
    re.VERBOSE,
)
_vscodepatch = None


def environment_inference():
    global _vscodepatch
    if _vscodepatch is None:
        try:
            import os

            from IPython import get_ipython

            ip = get_ipython()
            in_jupyter = ip is not None and ip.__class__.__name__.startswith("ZMQ")

            in_vscode = (
                "VSCODE_PID" in os.environ
                or "VSCODE_IPC_HOOK" in os.environ
                or os.environ.get("TERM_PROGRAM") == "vscode"
            )

            _vscodepatch = bool(in_jupyter and in_vscode)
        except Exception:
            _vscodepatch = False

    return _vscodepatch


dgcv_settings_registry = {
    "use_latex": True,
    "theme": "graph_paper",
    "format_displays": True,
    "version_specific_defaults": f"v{__version__}",
    "ask_before_overwriting_objects_in_vmf": True,
    "forgo_warnings": False,
    "default_symbolic_engine": "sympy",
    "verbose_label_printing": False,
    "print_style": "readable",
    "VLP": vlp,
    "conjugation_prefix": "BAR",
    "preferred_variable_format": "complex",
    "pass_solve_requests_to_symbolic_engine": True,
    "apply_awkward_workarounds_to_fix_VSCode_display_issues": environment_inference(),
    "use_numeric_methods": False,
    "default_numeric_engine": "numpy",
    "numeric_error_thresholds": {
        "abs_tolerance": 1e-9,
        "rel_tolerance": 1e-9,
        "policy": "balanced",  # balanced, reckless,
    },
    "DEBUG": False,
    "force_rich_display": False,
}
vs_registry = []


def get_variable_registry():
    return variable_registry


def get_dgcv_settings_registry():
    return dgcv_settings_registry


def get_vs_registry():
    return vs_registry


def from_vsr(idx):
    return vs_registry[idx]


def _vsr_inh_idx(idx):
    vs = from_vsr(idx)
    return getattr(vs, "ambient", vs).dgcv_vs_id


def clear_variable_registry():
    global variable_registry
    variable_registry = {
        "standard_variable_systems": {},
        "complex_variable_systems": {},
        "finite_algebra_systems": {},
        "protected_variables": set(),
        "temporary_variables": set(),
        "obscure_variables": set(),
        "conversion_dictionaries": {
            "holToReal": {},
            "realToSym": {},
            "symToHol": {},
            "symToReal": {},
            "realToHol": {},
            "conjugation": {},
            "find_parents": {},
            "real_part": {},
            "im_part": {},
        },
        "dgcv_enforced_real_atoms": dict(),
        "_labels": {},
    }


def canonicalize(obj, with_simplify=False, depth=1000):
    if hasattr(obj, "_eval_canonicalize"):
        obj = obj._eval_canonicalize(depth=depth)
    if with_simplify is True:
        return obj._eval_simplify() if hasattr(obj, "_eval_simplify") else obj
    else:
        return obj


class dgcv_exception_note(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


def _try_wrap_html(s: str):
    """
    Return an IPython.display.HTML object if IPython is available; otherwise
    return the string unchanged.
    """
    try:
        from dgcv.backends._notebooks import is_ipython_available
    except Exception:
        is_ipython_available = None

    if callable(is_ipython_available):
        try:
            if not is_ipython_available():
                return s
        except Exception:
            return s

    try:
        from IPython.display import HTML  # local import by design

        return HTML(s)
    except Exception:
        return s


def latex_in_html(html_string, apply_VSCode_workarounds=False):
    if (
        dgcv_settings_registry["apply_awkward_workarounds_to_fix_VSCode_display_issues"]
        is True
    ):
        apply_VSCode_workarounds = True

    if apply_VSCode_workarounds is True:
        katexInjectString = """<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" integrity="sha384-nB0miv6/jRmo5UMMR1wu3Gz6NLsoTkbqJghGIsx//Rlm+ZU03BU6SQNC66uf4l5+" crossorigin="anonymous">
<script type="module">
    import renderMathInElement from "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.mjs";
    renderMathInElement(document.body, {
        delimiters: [
            {left: "$$", right: "$$", display: true},
            {left: "$", right: "$", display: false}
        ]
    });
</script>"""
        if hasattr(html_string, "to_html"):
            try:
                body = html_string.to_html(escape=False)
            except Exception:
                body = str(html_string)
        else:
            body = str(html_string)

        return _try_wrap_html(katexInjectString + body)
    return html_string


def configure_convenient_labels(
    libraries: Optional[
        List[Literal["all", "complex variables", "symbolic expressions"]]
    ] = ["all"],
    verbose: bool = False,
    safe_labeling: bool = False,
    custom_labels: Optional[dict] = None,
):
    include_all = "all" in libraries
    relabeling_map = (
        {
            "simplify": "simplify_dgcv",
            "expand": "expand_dgcv",
            "factor": "factor_dgcv",
            "cancel": "cancel_dgcv",
            "conjugate": "conjugate_dgcv",
            "im": "im_dgcv",
            "re": "re_dgcv",
        }
        if safe_labeling
        else dict()
    )
    if isinstance(custom_labels, dict) and all(
        isinstance(k, str) for k in custom_labels
    ):
        relabeling_map |= custom_labels
    configured_by_library: dict[str, list[str]] = {}

    if include_all or "complex variables" in libraries:
        from .backends import im, re
        from .backends._types_and_constants import imag_unit
        from .dgcv_core import conjugate_dgcv

        new_functions = {
            "I": imag_unit(),
            "conjugate": conjugate_dgcv,
            "im": im,
            "re": re,
        }
        new_functions = {relabeling_map.get(k, k): v for k, v in new_functions.items()}
        _cached_caller_globals.update(new_functions)
        configured_by_library["complex variables"] = sorted(
            new_functions, key=str.lower
        )

    if include_all or "symbolic expressions" in libraries:
        from dgcv.backends._types_and_constants import symbol

        from .backends import (
            cancel_dgcv,
            expand_dgcv,
            factor_dgcv,
            ratio,
            simplify_dgcv,
            subs_dgcv,
        )

        new_functions = {
            "subs": subs_dgcv,
            "take_quotient": ratio,
            "simplify": simplify_dgcv,
            "expand": expand_dgcv,
            "factor": factor_dgcv,
            "cancel": cancel_dgcv,
            "symbol": symbol,
        }
        new_functions = {relabeling_map.get(k, k): v for k, v in new_functions.items()}
        _cached_caller_globals.update(new_functions)
        configured_by_library["symbolic expressions"] = sorted(
            new_functions, key=str.lower
        )

    if not configured_by_library:
        return

    to_print = (
        True if not dgcv_settings_registry.get("forgo_warnings", False) else verbose
    )

    if not to_print:
        return

    engine = dgcv_settings_registry.get("default_symbolic_engine", "dgcv_custom")

    bullets = "\n".join(
        f"  • For {lib}: {', '.join(funcs)}"
        for lib, funcs in configured_by_library.items()
    )

    message = f"The following labels were configured in the namespace:\n\n{bullets}"
    if not dgcv_settings_registry.get("forgo_warnings", False):
        message += (
            "\n\n"
            f"User warning: The current dgcv settings have {engine} registered as the default symbolic engine.\n"
            "The above labels supersede equivalently-named objects from the default engine, "
            "extending functionality where applicable and falling back to the "
            f"{engine} implementations otherwise."
        )

    print(message)
