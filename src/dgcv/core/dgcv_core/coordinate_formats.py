from ..._aux._backends._symbolic_router import conjugate, simplify, subs
from ..._aux._backends._types_and_constants import imag_unit, rational
from ..._aux._utilities._config import (
    dgcv_warning,
    dgcvDeprecationWarning,
    get_variable_registry,
)
from ..conversions.conversions import allToReal, allToSym

half = rational(1, 2)


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
    return allToReal(expr).subs({imag_unit(): -imag_unit()})


def re_with_real_coor(expr):
    expr = allToReal(expr)
    s = simplify(half * (expr + conj_with_real_coor(expr)))
    return s


def im_with_real_coor(expr):
    expr = allToReal(expr)
    s = simplify(-imag_unit() * half * (expr - conj_with_real_coor(expr)))
    return s


def conj_with_hol_coor(expr):
    vr = get_variable_registry()
    subsDictA = dict(vr["conversion_dictionaries"]["conjugation"])
    subsDict = subsDictA | {imag_unit(): -imag_unit()}
    return subs(allToSym(expr), subsDict, simultaneous=True)


def re_with_hol_coor(expr):
    expr = allToSym(expr)
    s = simplify(half * (expr + conj_with_hol_coor(expr)))
    return s


def im_with_hol_coor(expr):
    expr = allToSym(expr)
    s = simplify(-imag_unit() * half * (expr - conj_with_hol_coor(expr)))
    return s
