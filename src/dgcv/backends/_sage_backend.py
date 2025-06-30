import importlib
import importlib.util

_sage_module = None
_sage_available = None

def is_sage_available():
    global _sage_available
    if _sage_available is not None:
        return _sage_available
    try:
        spec = importlib.util.find_spec("sage.all")
        _sage_available = spec is not None
    except (ImportError, ModuleNotFoundError):
        _sage_available = False
    return _sage_available

def get_sage_module():
    global _sage_module
    if _sage_module is not None:
        return _sage_module
    if not is_sage_available():
        raise RuntimeError("Sage is not available in the current environment.")
    _sage_module = importlib.import_module("sage.all")
    return _sage_module
