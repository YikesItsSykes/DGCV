import pytest  # type: ignore

from dgcv._aux._backends import conjugate_dgcv_sym_router, simplify_dgcv
from dgcv._aux._backends._types_and_constants import imag_unit
from dgcv._aux._utilities._config import get_globals
from dgcv.core.conversions.conversions import (
    allToHol,
    allToReal,
    allToSym,
    realToHol,
    realToSym,
    symToHol,
    symToReal,
)
from dgcv.core.dgcv_core.dgcv_core import createVariables

# --- public conversions ------------------------------------------------------


def test_realToHol_on_real_coordinates(complex_system_factory):
    s = complex_system_factory(n=2)
    imag = s["imag"]

    z1, z2 = s["z"]
    BARz1, BARz2 = s["BARz"]
    x1, x2 = s["x"]
    y1, y2 = s["y"]

    assert realToHol(x1) - (z1 / 2 + conjugate_dgcv_sym_router(z1) / 2) == 0
    assert (
        simplify_dgcv(
            realToHol(y1) - (-imag * z1 / 2 + imag * conjugate_dgcv_sym_router(z1) / 2)
        )
        == 0
    )


def test_realToSym_on_real_coordinates(complex_system_factory):
    s = complex_system_factory(n=2)
    imag = s["imag"]

    z1, z2 = s["z"]
    BARz1, BARz2 = s["BARz"]
    x1, x2 = s["x"]
    y1, y2 = s["y"]

    assert realToSym(x1) - (z1 / 2 + BARz1 / 2) == 0
    assert simplify_dgcv(realToSym(y1) - (-imag * z1 / 2 + imag * BARz1 / 2)) == 0


def test_symToReal_on_symbolic_complex_coordinates(complex_system_factory):
    s = complex_system_factory(n=2)
    imag = s["imag"]

    z1, z2 = s["z"]
    BARz1, BARz2 = s["BARz"]
    x1, x2 = s["x"]
    y1, y2 = s["y"]

    assert simplify_dgcv(symToReal(z1) - (x1 + imag * y1)) == 0
    assert simplify_dgcv(symToReal(BARz1) - (x1 - imag * y1)) == 0


def test_symToHol_on_symbolic_complex_coordinates(complex_system_factory):
    s = complex_system_factory(n=2)

    z1, z2 = s["z"]
    BARz1, BARz2 = s["BARz"]

    assert symToHol(z1) == z1
    assert symToHol(BARz1) == conjugate_dgcv_sym_router(z1)


def test_allToReal_on_symbolic_complex_expression(complex_system_factory):
    s = complex_system_factory(n=2)

    z1, z2 = s["z"]
    BARz1, BARz2 = s["BARz"]
    x1, x2 = s["x"]

    assert allToReal(z1 + BARz1) == 2 * x1


def test_allToHol_on_real_expression(complex_system_factory):
    s = complex_system_factory(n=2)

    z1, z2 = s["z"]
    x1, x2 = s["x"]

    assert allToHol(x1) == realToHol(x1)


def test_allToSym_on_real_expression(complex_system_factory):
    s = complex_system_factory(n=2)

    z1, z2 = s["z"]
    x1, x2 = s["x"]

    assert allToSym(x1) == realToSym(x1)


# --- skipVar / variables_scope -----------------------------------------------


def test_allToReal_respects_skipVar(complex_system_factory):
    s = complex_system_factory(n=2)

    z1, z2 = s["z"]
    BARz1, BARz2 = s["BARz"]

    expr = z1 + BARz2
    assert allToReal(expr, skipVar=s["labels"]["z"]) == expr


def test_allToReal_respects_variables_scope(complex_system_factory):
    s = complex_system_factory(n=2)
    imag = s["imag"]

    z1, z2 = s["z"]
    BARz1, BARz2 = s["BARz"]
    x1, x2 = s["x"]
    y1, y2 = s["y"]

    expr = z1 + BARz2
    assert allToReal(expr, variables_scope=[z2]) == x2 - imag * y2 + z1
