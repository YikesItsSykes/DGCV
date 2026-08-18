from numbers import Integral

from ..._aux._vmf._safeguards import retrieve_passkey


def _validate_pos_int(n, name="n"):
    if not isinstance(n, Integral) or n < 0:
        raise TypeError(f"{name} must be a nonnegative integer, got {n!r}")


def _validate_int(i, name="i"):
    if not isinstance(i, Integral):
        raise TypeError(f"{name} must be an integer, got {i!r}")


def _as_seq(x, name="data"):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    raise TypeError(f"{name} must be a list/tuple, got {type(x).__name__}")


def _new_empty_matrix(cls, nrows, ncols, passkey=None, null_return=None):
    _validate_pos_int(nrows, "nrows")
    _validate_pos_int(ncols, "ncols")
    out = cls.__new__(cls)
    out._dgcv_class_check = retrieve_passkey() if passkey is None else passkey
    out._dgcv_categories = {"matrix"}
    out.shape = (nrows, ncols)
    out.ndim = 2
    out._data = {}
    out.null_return = 0 if null_return is None else null_return
    out._engine_representation = dict()
    out._data_unspooled_cache = None
    return out
