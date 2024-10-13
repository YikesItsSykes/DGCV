"""
config.py

This module provides utility functions for managing the `variable_registry` 
and caching/retrieving the caller's global namespace in the DGCV package. 
It uses the `inspect` module to search through the call stack for the `__main__` 
module and caches its globals for later use. Additionally, this module manages 
the centralized `variable_registry` that tracks DGCV variables, ensuring that 
it remains protected from user interference.

Functions
---------
- get_variable_registry: Returns the current state of the `variable_registry`, 
  which holds information about standard and complex variable systems.
- clear_variable_registry: Resets the `variable_registry` to its initial state.
- get_caller_globals: Searches the call stack for the globals of the `__main__` module.
- cache_globals: Initializes the globals cache at package import.

"""

import inspect
import warnings

# Cached caller globals
_cached_caller_globals = None

greek_letters = {
        'alpha': '\\alpha', 'beta': '\\beta', 'gamma': '\\gamma', 'delta': '\\delta', 'epsilon': '\\epsilon',
        'zeta': '\\zeta', 'eta': '\\eta', 'theta': '\\theta', 'iota': '\\iota', 'kappa': '\\kappa',
        'lambda': '\\lambda', 'mu': '\\mu', 'nu': '\\nu', 'xi': '\\xi', 'omicron': '\\omicron', 'pi': '\\pi',
        'rho': '\\rho', 'sigma': '\\sigma', 'tau': '\\tau', 'upsilon': '\\upsilon', 'phi': '\\phi', 'chi': '\\chi',
        'psi': '\\psi', 'omega': '\\omega', 'Gamma': '\\Gamma',
        'Delta': '\\Delta', 'Theta': '\\Theta', 'Lambda': '\\Lambda', 'Xi': '\\Xi',
            'Pi': '\\Pi', 'Sigma': '\\Sigma', 'Upsilon': '\\Upsilon',
        'Phi': '\\Phi',  'Psi': '\\Psi', 'Omega': '\\Omega'
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

    # Perform the search to find the caller's globals
    for frame_info in inspect.stack():
        if frame_info.frame.f_globals['__name__'] == '__main__':
            _cached_caller_globals = frame_info.frame.f_globals
            return _cached_caller_globals

    # If no '__main__' is found, raise an error
    raise RuntimeError("Could not find the '__main__' module in the call stack.")

def cache_globals():
    """
    Initialize the global namespace cache.
    
    This function should be called at package import to initialize and cache the 
    global namespace for later use.
    """
    if _cached_caller_globals is None:
        get_caller_globals()

# warning format

def configure_warnings():
    warnings.simplefilter('once')  # Only show each warning once
    # Optionally customize the format
    def custom_format_warning(message, category, filename, lineno, file=None, line=None):
        return f"{category.__name__}: {message}\n"
    
    warnings.formatwarning = custom_format_warning



# Initialize variable_registry in the DGCV module's scope
variable_registry = {
    'standard_variable_systems': {},
    'complex_variable_systems': {},
    'finite_algebra_systems': {},
    'protected_variables': set(),
    'temporary_variables': set(),
    'obscure_variables': set(),
    'conversion_dictionaries': {
        'holToReal': {},
        'realToSym': {},
        'symToHol': {},
        'symToReal': {},
        'realToHol': {},
        'conjugation': {},
        'find_parents': {}
    }
}

# Getter and Setter functions for accessing variable_registry
def get_variable_registry():
    return variable_registry

def clear_variable_registry():
    # Optional: function to reset variable_registry if needed
    global variable_registry
    variable_registry = {
    'standard_variable_systems': {},
    'complex_variable_systems': {},
    'finite_algebra_systems': {},
    'protected_variables': set(),
    'temporary_variables': set(),
    'obscure_variables': set(),
    'conversion_dictionaries': {
        'holToReal': {},
        'realToSym': {},
        'symToHol': {},
        'symToReal': {},
        'realToHol': {},
        'conjugation': {},
        'find_parents': {}
    }
    }