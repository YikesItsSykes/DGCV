import pytest  # type: ignore

from dgcv._aux._backends._types_and_constants import symbol
from dgcv.core.polynomials.polynomials import (
    createBigradPolynomial,
    createMultigradedPolynomial,
    createPolynomial,
    getWeightedTerms,
    monomialWeight,
)

# --- polynomials module ------------------------------------------------------


def test_createPolynomial_builds_inhomogeneous_polynomial(complex_system_factory):
    s = complex_system_factory(n=2, withVF=False)
    x1, x2 = s["x"]
    y1, y2 = s["y"]

    p = createPolynomial("a", 2, (x1, x2))

    assert p.degree == 2
    assert set(p.varSpace) == {x1, x2, y1, y2}

    monoms, coeffs = p.get_monomials(
        formatting="unformatted",
        separate_coeffs=True,
    )

    assert set(monoms) == {1, x1, x2, x1**2, x1 * x2, x2**2}
    assert len(coeffs) == 6


def test_createPolynomial_builds_homogeneous_polynomial(complex_system_factory):
    s = complex_system_factory(n=2, withVF=False)
    x1, x2 = s["x"]

    p, monomials, coeffs = createPolynomial(
        "a",
        2,
        (x1, x2),
        homogeneous=True,
        return_components=True,
    )

    assert p.degree == 2
    assert len(monomials) == 3
    assert len(coeffs) == 3

    extracted = p.get_monomials(formatting="unformatted", as_dict=True)
    assert set(extracted.keys()) == {x1**2, x1 * x2, x2**2}


def test_createPolynomial_builds_weighted_homogeneous_polynomial(
    complex_system_factory,
):
    s = complex_system_factory(n=2, withVF=False)
    x1, x2 = s["x"]

    p = createPolynomial(
        "a",
        2,
        (x1, x2),
        weightedHomogeneity=(1, 2),
    )

    monoms = p.get_monomials(formatting="unformatted", as_dict=True)
    assert set(monoms.keys()) == {x1**2, x2}


def test_createPolynomial_weighted_homogeneity_with_zero_weight_uses_degreeCap(
    complex_system_factory,
):
    s = complex_system_factory(n=2, withVF=False)
    x1, x2 = s["x"]

    p = createPolynomial(
        "a",
        2,
        (x1, x2),
        weightedHomogeneity=(1, 0),
        degreeCap=1,
    )

    monoms = p.get_monomials(formatting="unformatted", as_dict=True)
    assert set(monoms.keys()) == {x1**2, x1**2 * x2}


def test_createPolynomial_registers_coefficients_in_vmf(clean_vmf, fresh_label):
    from dgcv._aux._utilities._config import get_variable_registry

    coeff_label = fresh_label("a")
    x = symbol(fresh_label("x"))
    y = symbol(fresh_label("y"))

    p, monomials, coeffs = createPolynomial(
        coeff_label,
        2,
        (x, y),
        homogeneous=True,
        register_coeffs_in_vmf=True,
        return_components=True,
    )

    registry = get_variable_registry()
    assert coeff_label in registry["standard_variable_systems"]
    system = registry["standard_variable_systems"][coeff_label]
    assert system["family_type"] == "tuple"
    assert len(system["family_names"]) == 3
    assert tuple(coeffs) == system["family_values"]


def test_createBigradPolynomial_filters_by_two_weight_systems(
    complex_system_factory,
):
    s = complex_system_factory(n=2, withVF=False)
    x1, x2 = s["x"]

    p = createBigradPolynomial(
        "b",
        (1, 2),
        (x1, x2),
        (1, 0),
        (0, 1),
    )

    monoms = p.get_monomials(formatting="unformatted", as_dict=True)
    assert set(monoms.keys()) == {x1 * x2**2}


def test_createMultigradedPolynomial_filters_by_multiple_weight_systems(
    complex_system_factory,
):
    s = complex_system_factory(n=3, withVF=False)
    x1, x2, x3 = s["x"]

    p = createMultigradedPolynomial(
        "c",
        (1, 2),
        (x1, x2, x3),
        (
            (1, 0, 0),
            (0, 1, 1),
        ),
    )

    monoms = p.get_monomials(formatting="unformatted", as_dict=True)
    assert set(monoms.keys()) == {x1 * x2**2, x1 * x2 * x3, x1 * x3**2}


def test_monomialWeight_returns_expected_weight(complex_system_factory):
    s = complex_system_factory(n=3, withVF=False)
    x1, x2, x3 = s["x"]

    assert monomialWeight(x1**2 * x2**3, (x1, x2, x3), (1, 2, 5)) == 8


def test_monomialWeight_returns_zero_on_zero_monomial(complex_system_factory):
    s = complex_system_factory(n=2, withVF=False)
    x1, x2 = s["x"]

    assert monomialWeight(0, (x1, x2), (1, 2)) == 0


def test_getWeightedTerms_filters_existing_polynomial_by_weight(
    complex_system_factory,
):
    s = complex_system_factory(n=3, withVF=False)
    x1, x2, x3 = s["x"]
    y1, y2, y3 = s["y"]

    poly = createPolynomial(
        "a",
        3,
        (x1, x2, x3),
    )

    filtered = getWeightedTerms(
        poly,
        target_degrees=(2,),
        weight_systems=(
            {
                x1: 1,
                x2: 2,
                x3: 3,
                y1: 1,
                y2: 2,
                y3: 3,
            },
        ),
    )

    monoms = filtered.get_monomials(formatting="unformatted", as_dict=True)
    assert set(monoms.keys()) == {x1**2, x2}


def test_getWeightedTerms_respects_multiple_weight_systems(complex_system_factory):
    s = complex_system_factory(n=3, withVF=False)
    x1, x2, x3 = s["x"]
    y1, y2, y3 = s["y"]

    poly = createPolynomial(
        "a",
        3,
        (x1, x2, x3),
    )

    filtered = getWeightedTerms(
        poly,
        target_degrees=(1, 2),
        weight_systems=(
            {x1: 1, y1: 1},
            {x2: 1, y2: 1, x3: 1, y3: 1},
        ),
    )

    monoms = filtered.get_monomials(formatting="unformatted", as_dict=True)
    assert set(monoms.keys()) == {
        x1 * x2**2,
        x1 * x2 * x3,
        x1 * x3**2,
    }


def test_getWeightedTerms_ignores_unspecified_variables_by_default(
    complex_system_factory,
):
    s = complex_system_factory(n=2, withVF=False)
    x1, x2 = s["x"]
    y1, y2 = s["y"]

    poly = createPolynomial("a", 2, (x1, x2))

    filtered = getWeightedTerms(
        poly,
        target_degrees=(2,),
        weight_systems=({x1: 1, y1: 1},),
    )

    monoms = filtered.get_monomials(formatting="unformatted", as_dict=True)
    assert set(monoms.keys()) == {x1**2}


@pytest.mark.parametrize(
    ("degrees", "weights_1", "weights_2"),
    [
        ((1,), (1, 0), (0, 1)),
    ],
)
def test_createBigradPolynomial_rejects_bad_degree_length(
    complex_system_factory,
    degrees,
    weights_1,
    weights_2,
):
    s = complex_system_factory(n=2, withVF=False)
    x1, x2 = s["x"]

    with pytest.raises(ValueError, match="degrees must have length 2"):
        createBigradPolynomial("b", degrees, (x1, x2), weights_1, weights_2)


def test_createMultigradedPolynomial_rejects_mismatched_degree_and_weight_counts(
    complex_system_factory,
):
    s = complex_system_factory(n=2, withVF=False)
    x1, x2 = s["x"]

    with pytest.raises(KeyError, match="degrees and weight_systems"):
        createMultigradedPolynomial(
            "c",
            (1, 2),
            (x1, x2),
            ((1, 0),),
        )


def test_getWeightedTerms_rejects_mismatched_target_and_weight_counts(
    complex_system_factory,
):
    s = complex_system_factory(n=2, withVF=False)
    x1, x2 = s["x"]

    poly = createPolynomial("a", 2, (x1, x2))

    with pytest.raises(ValueError, match="target_degrees and weight_systems"):
        getWeightedTerms(
            poly,
            target_degrees=(1, 2),
            weight_systems=({x1: 1},),
        )
