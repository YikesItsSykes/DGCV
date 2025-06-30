"""
_config.py

This module provides utility functions for dgcv's Variable Management Framework, maintaining
a registry of instantiated mathematical object for interaction with dgcv functions. 

Functions
---------
- get_variable_registry: Returns the current state of the `variable_registry`, 
  which holds information about objects tracked in the VMF.
- clear_variable_registry: Resets the `variable_registry` to its initial state.
- get_dgcv_settings_registry: Returns the current state of the dictionary storing setting default affecting dgcv functions.
"""

import collections.abc
import inspect
import warnings

from dgcv import __version__

_cached_caller_globals = None

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
    "hbar": "\\hbar"
}

def get_caller_globals():
    """
    Retrieve and cache the caller's global namespace.

    This function searches through the call stack to locate the global namespace of
    the `__main__` module and caches it. If the globals have already been cached,
    it returns the cached value.

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
    "misc":{},
    "eds":{'atoms':{},'coframes':{}},
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
    "_labels":{}
}
dgcv_settings_registry = {
    'use_latex':False,
    'theme':'blue',
    'format_displays':False,
    'version_specific_defaults':f'v{__version__}',
    'ask_before_overwriting_objects_in_vmf':True,
    'forgo_warnings':False,
    'default_symbolic_engine':'sympy'
}

def get_variable_registry():
    return variable_registry

def get_dgcv_settings_registry():
    return dgcv_settings_registry

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
    }

def canonicalize(obj,with_simplify = False, depth = 1000):
    if hasattr(obj,'_eval_canonicalize'):
        obj = obj._eval_canonicalize(depth=depth)
    if with_simplify is True:
        return obj._eval_simplify() if hasattr(obj,'_eval_simplify') else obj
    else:
        return obj

class dgcv_exception_note(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message
