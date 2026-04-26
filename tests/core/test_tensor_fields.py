import pytest  # type: ignore

from dgcv._aux._backends._types_and_constants import imag_unit, symbol
from dgcv._aux._utilities._config import get_globals
from dgcv.core.dgcv_core.dgcv_core import (
    createVariables,
    differential_form_class,
    tensor_field_class,
    vector_field_class,
)

# --- tensor_field_class construction -----------------------------------------


def test_tensor_field_scalar_coeff_dict_initializes_as_degree_zero():
    tf = tensor_field_class(coeff_dict={tuple(): 5})

    assert tf.data_shape == "all"
    assert tf.valence == tuple()
    assert tf.coeff_dict == {tuple(): 5}
    assert tf.total_degree == 0
    assert tf.max_degree == 0
    assert tf.min_degree == 0
    assert tf.is_zero is False


def test_tensor_field_empty_coeff_dict_becomes_zero_scalar():
    tf = tensor_field_class(coeff_dict={})

    assert tf.coeff_dict == {tuple(): 0}
    assert tf.data_shape == "all"
    assert tf.total_degree == 0
    assert tf.is_zero is True


def test_tensor_field_invalid_coeff_dict_type_raises():
    with pytest.raises(TypeError, match="dictionary"):
        tensor_field_class(coeff_dict=5)


# --- tensor_field_class metadata ---------------------------------------------


def test_tensor_field_infers_variable_spaces_from_two_systems(two_standard_systems):
    s = two_standard_systems

    tf = tensor_field_class(
        coeff_dict={
            (0, 0, s["x_label"]): 1,
            (1, 1, s["y_label"]): 2,
        }
    )

    assert s["x_label"] in tf._variable_spaces
    assert s["y_label"] in tf._variable_spaces


def test_tensor_field_degree_metadata_for_mixed_two_system_tensor(two_standard_systems):
    s = two_standard_systems

    tf = tensor_field_class(
        coeff_dict={
            (0, 1, s["x_label"]): 3,
            (1, 1, s["y_label"]): 4,
        }
    )

    assert tf.max_degree == 1
    assert tf.min_degree == 1
    assert tf.total_degree == 1
    assert tf.degree == 1


def test_tensor_field_apply_transforms_coefficients(two_standard_systems):
    s = two_standard_systems

    tf = tensor_field_class(
        coeff_dict={
            (0, 1, s["x_label"]): 2,
            (1, 1, s["y_label"]): 3,
        }
    )

    out = tf.apply(lambda c: 10 * c)

    assert out.coeff_dict == {
        (0, 1, s["x_label"]): 20,
        (1, 1, s["y_label"]): 30,
    }


def test_tensor_field_subs_replaces_symbolic_coefficients(two_standard_systems):
    s = two_standard_systems
    a = symbol("a")

    tf = tensor_field_class(
        coeff_dict={
            (0, 1, s["x_label"]): a + 1,
        }
    )

    out = tf.subs({a: 4})

    assert out.coeff_dict == {
        (0, 1, s["x_label"]): 5,
    }


def test_tensor_field_infer_varSpace_any_includes_variables_from_two_systems(
    two_standard_systems,
):
    s = two_standard_systems

    tf = tensor_field_class(
        coeff_dict={
            (0, 1, s["x_label"]): 1,
            (1, 0, s["y_label"]): 1,
        }
    )

    vs = tf.infer_varSpace(formatting="any")

    assert s["x1"] in vs
    assert s["x2"] in vs
    assert s["y1"] in vs
    assert s["y2"] in vs


def test_tensor_field_infer_minimal_varSpace_restricts_to_used_slots(
    two_standard_systems,
):
    s = two_standard_systems

    tf = tensor_field_class(
        coeff_dict={
            (0, 1, s["x_label"]): 1,
            (1, 0, s["y_label"]): 1,
        }
    )

    vs = tf.infer_minimal_varSpace()

    assert s["x1"] in vs
    assert s["y2"] in vs
    assert s["x2"] not in vs
    assert s["y1"] not in vs


# --- vector_field_class ------------------------------------------------------


def test_createVariables_builds_vector_field_instances(two_standard_systems):
    s = two_standard_systems

    assert isinstance(s["Dx1"], vector_field_class)
    assert isinstance(s["Dx2"], vector_field_class)
    assert isinstance(s["Dy1"], vector_field_class)
    assert isinstance(s["Dy2"], vector_field_class)


def test_vector_field_as_tensor_field_returns_tensor_field(two_standard_systems):
    s = two_standard_systems

    tf = s["Dx1"].as_tensor_field()

    assert isinstance(tf, tensor_field_class)
    assert tf.valence == (1,)
    assert tf.total_degree == 1


def test_vector_field_negation_preserves_class(two_standard_systems):
    s = two_standard_systems

    neg_vf = -s["Dx1"]

    assert isinstance(neg_vf, vector_field_class)


def test_tensor_field_handles_multiple_coordinate_systems(two_standard_systems):
    s = two_standard_systems

    tf = tensor_field_class(
        coeff_dict={
            (0, 1, s["x_label"]): 1,
            (0, 1, s["y_label"]): 2,
        }
    )

    vs = tf.infer_varSpace(formatting="any")

    assert s["x1"] in vs
    assert s["y1"] in vs


# --- differential_form_class -------------------------------------------------


def test_createVariables_builds_differential_form_instances(two_standard_systems):
    s = two_standard_systems

    assert isinstance(s["dx1"], differential_form_class)
    assert isinstance(s["dx2"], differential_form_class)
    assert isinstance(s["dy1"], differential_form_class)
    assert isinstance(s["dy2"], differential_form_class)


def test_differential_form_as_tensor_field_returns_tensor_field(two_standard_systems):
    s = two_standard_systems

    tf = s["dx1"].as_tensor_field()

    assert isinstance(tf, tensor_field_class)
    assert tf.valence == (0,)
    assert tf.total_degree == 1


# --- operations --------------------------------------------------------------


def test_vector_field_contracts_with_differential_form(two_standard_systems):
    s = two_standard_systems

    vf = s["Dx1"]
    df = s["dx1"]

    result = df(vf)

    assert result == 1


def test_complex_coordinate_vector_fields_and_forms_are_consistent(
    clean_vmf, fresh_label
):

    imag = imag_unit()

    z_label = fresh_label("z")
    x_label = fresh_label("x")
    y_label = fresh_label("y")

    createVariables(z_label, x_label, y_label)

    g = get_globals()

    z = g[z_label]
    bar_z = g[f"BAR{z_label}"]
    x = g[x_label]
    y = g[y_label]

    D_z = g[f"D_{z_label}"]
    D_bar_z = g[f"D_BAR{z_label}"]
    D_x = g[f"D_{x_label}"]
    D_y = g[f"D_{y_label}"]

    d_z = g[f"d_{z_label}"]
    d_bar_z = g[f"d_BAR{z_label}"]
    d_x = g[f"d_{x_label}"]
    d_y = g[f"d_{y_label}"]

    checks = [
        2 * D_z(x) - 1,
        2 * imag * D_z(y) - 1,
        D_z(z) - 1,
        D_z(bar_z),
        2 * D_z(d_x) - 1,
        2 * imag * D_z(d_y) - 1,
        D_z(d_z) - 1,
        D_z(d_bar_z),
        d_z(D_x) - 1,
        -imag * d_z(D_y) - 1,
        d_z(D_z) - 1,
        d_z(D_bar_z),
        d_x(D_x) - 1,
        d_x(D_y),
        2 * d_x(D_z) - 1,
        2 * d_x(D_bar_z) - 1,
        d_y(D_x),
        d_y(D_y) - 1,
        2 * imag * d_y(D_z) - 1,
        -2 * imag * d_y(D_bar_z) - 1,
    ]

    assert checks == [0] * len(checks)
