from ..._aux._backends._symbolic_router import conjugate, simplify
from ..._aux._backends._types_and_constants import half, imag_unit
from ..._aux._utilities._config import (
    dgcv_warning,
    dgcvDeprecationWarning,
)
from ..conversions.conversions import allToReal, allToSym


def conjugate_DGCV(expr, symbolic=False):
    dgcv_warning(
        "`conjugate_DGCV` has been deprecated as part of the shift toward standardized naming conventions in the `dgcv` library. "
        "It will be removed in 2026. Please use `conjugate_dgcv` instead.",
        dgcvDeprecationWarning,
        stacklevel=2,
        old_kw="conjugate_DGCV",
        new_kw="conjugate_dgcv",
        sunset="2026",
    )
    return conjugate_dgcv(expr, symbolic=symbolic)


def conjugate_dgcv(expr, symbolic=False):
    return conjugate(expr, symbolic=symbolic)


def conj_with_real_coor(expr):
    return conjugate(allToReal(expr), symbolic=True)


def re_with_real_coor(expr):
    expr = allToReal(expr)
    s = simplify(half() * (expr + conj_with_real_coor(expr)))
    return s


def im_with_real_coor(expr):
    expr = allToReal(expr)
    s = simplify(-imag_unit() * half() * (expr - conj_with_real_coor(expr)))
    return s


def conj_with_hol_coor(expr):
    return conjugate(allToSym(expr), symbolic=True)


def re_with_hol_coor(expr):
    expr = allToSym(expr)
    s = simplify(half() * (expr + conj_with_hol_coor(expr)))
    return s


def im_with_hol_coor(expr):
    expr = allToSym(expr)
    s = simplify(-imag_unit() * half() * (expr - conj_with_hol_coor(expr)))
    return s
