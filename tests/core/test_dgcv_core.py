import pytest  # type: ignore

from dgcv._aux._backends._symbolic_router import conjugate, im, re, simplify
from dgcv._aux._backends._types_and_constants import imag_unit
from dgcv._aux._utilities._config import get_globals, get_variable_registry
from dgcv.core.conversions.conversions import allToReal
from dgcv.core.dgcv_core.dgcv_core import createVariables, polynomial_dgcv
from dgcv.special_fields.complex_structures import complex_struct_op

# --- createVariables creation ------------------------------------------------


def test_createVariables_creates_single_standard_variable(clean_vmf, fresh_label):
    label = fresh_label("std")

    rv = createVariables(label, withVF=False, return_created_object=True)

    assert isinstance(rv, list)
    assert len(rv) == 1
    assert len(rv[0]) == 1

    registry = get_variable_registry()
    system = registry["standard_variable_systems"][label]

    assert system["family_type"] == "single"
    assert system["family_names"] == (label,)
    assert system["differential_system"] is None

    global_ns = get_globals()
    assert label in global_ns
    assert global_ns[label] == system["family_values"][0]


def test_createVariables_creates_tuple_standard_system(clean_vmf, fresh_label):
    label = fresh_label("std")

    rv = createVariables(
        label,
        number_of_variables=3,
        withVF=False,
        return_created_object=True,
    )

    assert isinstance(rv, list)
    assert len(rv) == 1
    assert len(rv[0]) == 3

    registry = get_variable_registry()
    system = registry["standard_variable_systems"][label]

    assert system["family_type"] == "tuple"
    assert system["family_names"] == (
        f"{label}1",
        f"{label}2",
        f"{label}3",
    )
    assert system["initial_index"] == 1
    assert system["differential_system"] is False

    global_ns = get_globals()
    assert label in global_ns
    assert f"{label}1" in global_ns
    assert f"{label}2" in global_ns
    assert f"{label}3" in global_ns
    assert global_ns[label] == system["family_values"]


def test_createVariables_respects_initial_index_for_standard_tuple(
    clean_vmf, fresh_label
):
    label = fresh_label("std")

    createVariables(
        label,
        number_of_variables=3,
        initialIndex=0,
        withVF=False,
    )

    registry = get_variable_registry()
    system = registry["standard_variable_systems"][label]

    assert system["family_names"] == (
        f"{label}0",
        f"{label}1",
        f"{label}2",
    )
    assert system["initial_index"] == 0

    global_ns = get_globals()
    assert label in global_ns
    assert f"{label}0" in global_ns
    assert f"{label}1" in global_ns
    assert f"{label}2" in global_ns


def test_createVariables_withVF_injects_standard_tuple_and_differential_labels(
    clean_vmf, fresh_label
):
    label = fresh_label("std")

    rv = createVariables(
        label,
        number_of_variables=2,
        withVF=True,
        return_created_object=True,
    )

    assert isinstance(rv, list)
    assert len(rv) == 1
    assert len(rv[0]) == 2

    registry = get_variable_registry()
    system = registry["standard_variable_systems"][label]

    assert system["family_type"] == "tuple"
    assert system["family_names"] == (
        f"{label}1",
        f"{label}2",
    )
    assert system["differential_system"] is True

    global_ns = get_globals()
    assert label in global_ns
    assert f"{label}1" in global_ns
    assert f"{label}2" in global_ns
    assert f"D_{label}1" in global_ns
    assert f"D_{label}2" in global_ns
    assert f"d_{label}1" in global_ns
    assert f"d_{label}2" in global_ns


def test_createVariables_injects_complex_tuple_labels_into_globals(
    clean_vmf, fresh_label
):
    z_label = fresh_label("z")
    x_label = fresh_label("x")
    y_label = fresh_label("y")

    rv = createVariables(
        z_label,
        x_label,
        y_label,
        2,
        withVF=True,
        return_created_object=True,
    )

    assert rv is not None

    registry = get_variable_registry()
    assert z_label in registry["complex_variable_systems"]

    global_ns = get_globals()

    assert f"{z_label}1" in global_ns
    assert f"BAR{z_label}1" in global_ns
    assert f"{x_label}1" in global_ns
    assert f"{y_label}1" in global_ns

    assert f"D_{z_label}1" in global_ns
    assert f"d_{z_label}1" in global_ns


def test_createVariables_requires_string_variable_label(clean_vmf):
    with pytest.raises(TypeError, match="first argument to be a string"):
        createVariables(123)


# --- createVariables validation ----------------------------------------------


@pytest.mark.parametrize("bad_shape", [3, "23", [2, 0], [2, -1], [2, 1.5]])
def test_createVariables_rejects_invalid_multiindex_shape(
    clean_vmf, fresh_label, bad_shape
):
    with pytest.raises(TypeError, match="multiindex_shape"):
        createVariables(fresh_label("mi"), multiindex_shape=bad_shape)


def test_createVariables_rejects_number_and_multiindex_together(clean_vmf, fresh_label):
    with pytest.raises(ValueError, match="Provide at most one"):
        createVariables(
            fresh_label("conflict"),
            number_of_variables=2,
            multiindex_shape=(2, 2),
        )


# --- complex coordinates ------------------------------------------------------


def test_createVariables_sets_expected_complex_structure_on_vector_fields(
    clean_vmf, fresh_label
):
    imag = imag_unit()

    z_label = fresh_label("z")
    x_label = fresh_label("x")
    y_label = fresh_label("y")

    createVariables(z_label, x_label, y_label)

    g = get_globals()

    D_z = g[f"D_{z_label}"]
    D_BARz = g[f"D_BAR{z_label}"]
    D_x = g[f"D_{x_label}"]
    D_y = g[f"D_{y_label}"]

    assert complex_struct_op(D_z) == imag * D_z
    assert complex_struct_op(D_BARz) == -imag * D_BARz
    assert complex_struct_op(D_x) == D_y
    assert complex_struct_op(D_y) == -D_x


def test_createVariables_complex_coordinate_conversions_on_vector_fields_and_forms(
    clean_vmf, fresh_label
):
    imag = imag_unit()

    z_label = fresh_label("z")
    x_label = fresh_label("x")
    y_label = fresh_label("y")

    createVariables(z_label, x_label, y_label)

    g = get_globals()

    z = g[z_label]
    BARz = g[f"BAR{z_label}"]
    x = g[x_label]
    y = g[y_label]

    D_z = g[f"D_{z_label}"]
    D_BARz = g[f"D_BAR{z_label}"]
    D_x = g[f"D_{x_label}"]
    D_y = g[f"D_{y_label}"]

    d_z = g[f"d_{z_label}"]
    d_BARz = g[f"d_BAR{z_label}"]
    d_x = g[f"d_{x_label}"]
    d_y = g[f"d_{y_label}"]

    assert (re(D_z) - D_x / 2).is_zero
    assert (im(D_z) + D_y / 2).is_zero
    assert (conjugate(D_z) - D_BARz).is_zero

    assert (re(D_BARz) - D_x / 2).is_zero
    assert (im(D_BARz) - D_y / 2).is_zero
    assert (conjugate(D_BARz) - D_z).is_zero

    assert (re(d_z) - d_x).is_zero
    assert (im(d_z) - d_y).is_zero
    assert (conjugate(d_z) - d_BARz).is_zero

    assert (re(d_BARz) - d_x).is_zero
    assert (im(d_BARz) + d_y).is_zero
    assert (conjugate(d_BARz) - d_z).is_zero

    assert allToReal(re(z * D_z) - (x / 2) * D_x - (y / 2) * D_y).is_zero
    assert allToReal(im(z * D_z) - (y / 2) * D_x + (x / 2) * D_y).is_zero
    assert allToReal(conjugate(z * D_z) - BARz * D_BARz).is_zero

    assert conjugate(z, symbolic=True) == BARz
    assert conjugate(imag * x, symbolic=True) == -imag * x


# --- polynomial_dgcv ---------------------------------------------------------


def test_polynomial_dgcv_builds_from_real_coordinate_expression(
    complex_system_factory,
):
    s = complex_system_factory(n=2, withVF=False)

    x1, x2 = s["x"]
    y1, y2 = s["y"]

    p = polynomial_dgcv(x1**2 + 3 * x1 * y1 + 7, varSpace=(x1, y1))

    assert p.polyExpr == x1**2 + 3 * x1 * y1 + 7
    assert p.coordinates == (x1, y1)
    assert p.varSpace == (x1, y1)
    assert p.degree == 2


def test_polynomial_dgcv_constant_and_sign_predicates(complex_system_factory):
    s = complex_system_factory(n=1, withVF=False)
    (x1,) = s["x"]

    p0 = polynomial_dgcv(0, varSpace=(x1,))
    p1 = polynomial_dgcv(1, varSpace=(x1,))
    pm1 = polynomial_dgcv(-1, varSpace=(x1,))
    pc = polynomial_dgcv(7, varSpace=(x1,))

    assert p0.is_zero is True
    assert p0.is_constant is True
    assert p0.constant_term == 0

    assert p1.is_one is True
    assert p1.is_constant is True
    assert p1.constant_term == 1

    assert pm1.is_minus_one is True
    assert pm1.is_constant is True
    assert pm1.constant_term == -1

    assert pc.is_constant is True
    assert pc.constant_term == 7
    assert pc.degree == 0


def test_polynomial_dgcv_detects_monomial(complex_system_factory):
    s = complex_system_factory(n=1, withVF=False)
    (x1,) = s["x"]
    (y1,) = s["y"]

    mon = polynomial_dgcv(3 * x1**2, varSpace=(x1, y1))
    nonmon = polynomial_dgcv(x1 + y1, varSpace=(x1, y1))

    assert mon.is_monomial is True
    assert nonmon.is_monomial is False


def test_polynomial_dgcv_get_monomials_and_coeffs(complex_system_factory):
    s = complex_system_factory(n=1, withVF=False)
    (x1,) = s["x"]
    (y1,) = s["y"]

    p = polynomial_dgcv(2 * x1**2 + 3 * x1 * y1 + 5, varSpace=(x1, y1))

    monoms = p.get_monomials(formatting="unformatted")
    coeffs = p.get_coeffs(formatting="unformatted")
    monom_dict = p.get_monomials(formatting="unformatted", as_dict=True)

    assert set(monoms) == {2 * x1**2, 3 * x1 * y1, 5}
    assert set(coeffs) == {2, 3, 5}
    assert monom_dict == {x1**2: 2, x1 * y1: 3, 1: 5}


def test_polynomial_dgcv_degree_filtered_monomials(complex_system_factory):
    s = complex_system_factory(n=1, withVF=False)
    (x1,) = s["x"]
    (y1,) = s["y"]

    p = polynomial_dgcv(x1**3 + x1 * y1 + 4, varSpace=(x1, y1))

    assert set(p.get_monomials(min_degree=1, max_degree=2)) == {x1 * y1}
    assert p.get_monomials(min_degree=0, max_degree=0) == [4]


def test_polynomial_dgcv_polynomial_and_scalar_arithmetic(complex_system_factory):
    s = complex_system_factory(n=1, withVF=False)
    (x1,) = s["x"]
    (y1,) = s["y"]

    p = polynomial_dgcv(x1 + 1, varSpace=(x1, y1))
    q = polynomial_dgcv(y1 - 2, varSpace=(x1, y1))

    assert (p + q).polyExpr == x1 + y1 - 1
    assert (p - q).polyExpr == x1 - y1 + 3
    assert simplify((p * q).polyExpr - (x1 + 1) * (y1 - 2)) == 0

    assert (p + 3).polyExpr == x1 + 4
    assert (3 + p).polyExpr == x1 + 4
    assert (p - 3).polyExpr == x1 - 2
    assert (3 - p).polyExpr == 2 - x1
    assert (2 * p).polyExpr == 2 * x1 + 2
    assert (p * 2).polyExpr == 2 * x1 + 2


def test_polynomial_dgcv_subs_diff_and_evaluate(complex_system_factory):
    s = complex_system_factory(n=1, withVF=False)
    (x1,) = s["x"]
    (y1,) = s["y"]

    p = polynomial_dgcv(x1**2 * y1 + y1, varSpace=(x1, y1))

    q = p.subs({y1: 2})
    dpx = p.diff(x1)

    assert simplify(q.polyExpr - (2 * x1**2 + 2)) == 0
    assert simplify(dpx.polyExpr - (2 * x1 * y1)) == 0
    assert p.evaluate({x1: 3, y1: 5}) == 50


def test_polynomial_dgcv_homogeneous_and_leading_term(complex_system_factory):
    s = complex_system_factory(n=1, withVF=False)
    (x1,) = s["x"]
    (y1,) = s["y"]

    homog = polynomial_dgcv(x1**2 + x1 * y1 + y1**2, varSpace=(x1, y1))
    nonhomog = polynomial_dgcv(x1**2 + y1 + 1, varSpace=(x1, y1))

    assert homog.is_homogeneous() is True
    assert nonhomog.is_homogeneous() is False
    assert nonhomog.leading_term() == x1**2


def test_polynomial_dgcv_parts_methods_on_promoted_real_polynomial(
    clean_vmf, complex_system_factory
):

    imag = imag_unit()

    s = complex_system_factory(n=4, withVF=False)

    x1, x2, x3, x4 = s["x"]

    y1, y2, y3, y4 = s["y"]

    z1, z2, z3, z4 = s["z"]

    BARz1, BARz2, BARz3, BARz4 = s["BARz"]

    plain_poly = (
        x1**4 + imag * x2**3 * y2 - x3**2 * y3**2 - imag * x4 * y4**3 + y4**4 + 1
    )
    promoted_poly = polynomial_dgcv(plain_poly)

    assert (
        promoted_poly.holomorphic_part
        - (z1**4 / 16 + z2**4 / 16 + z3**4 / 16 + z4**4 / 8 + 1)
        == 0
    )

    assert (
        promoted_poly.antiholomorphic_part
        - (BARz1**4 / 16 - BARz2**4 / 16 + BARz3**4 / 16 + 1)
        == 0
    )

    assert (
        promoted_poly.mixed_terms
        - (
            BARz1**3 * z1 / 4
            + 3 * BARz1**2 * z1**2 / 8
            + BARz1 * z1**3 / 4
            - BARz2**3 * z2 / 8
            + BARz2 * z2**3 / 8
            - BARz3**2 * z3**2 / 8
            - BARz4**3 * z4 / 8
            + 3 * BARz4**2 * z4**2 / 8
            - 3 * BARz4 * z4**3 / 8
        )
        == 0
    )

    assert (
        promoted_poly.pluriharmonic_part
        - (
            BARz1**4 / 16
            - BARz2**4 / 16
            + BARz3**4 / 16
            + z1**4 / 16
            + z2**4 / 16
            + z3**4 / 16
            + z4**4 / 8
            + 1
        )
        == 0
    )
