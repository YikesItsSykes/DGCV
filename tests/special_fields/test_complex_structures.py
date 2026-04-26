from dgcv._aux._backends._types_and_constants import imag_unit
from dgcv._aux._utilities._config import get_globals
from dgcv.core.dgcv_core.dgcv_core import createVariables
from dgcv.special_fields.complex_structures import Del, DelBar


def test_Del(clean_vmf, fresh_label):
    z_label = fresh_label("z")
    x_label = fresh_label("x")
    y_label = fresh_label("y")

    createVariables(z_label, x_label, y_label, 2)

    g = get_globals()

    y1, y2 = [g[f"{y_label}{idx}"] for idx in range(1, 3)]

    d_z1, d_z2 = [g[f"d_{z_label}{idx}"] for idx in range(1, 3)]
    d_BARz1, d_BARz2 = [g[f"d_BAR{z_label}{idx}"] for idx in range(1, 3)]
    rho = 2 * y1**2 + 2 * y2**2
    assert (Del(DelBar(rho)) - d_z1 * d_BARz1 - d_z2 * d_BARz2).is_zero
