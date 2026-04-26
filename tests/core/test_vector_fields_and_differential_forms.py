from dgcv._aux._backends import simplify_dgcv
from dgcv._aux._utilities._config import get_globals
from dgcv.core.dgcv_core.dgcv_core import createVariables, exteriorProduct, wedge
from dgcv.core.vector_fields_and_differential_forms.vector_fields_and_differential_forms import (
    LieDerivative,
    annihilator,
    decompose,
    exteriorDerivative,
    get_coframe,
    interiorProduct,
)


def test_Lie_derivative_and_exterior_product(clean_vmf, fresh_label):
    x_label = fresh_label("x")
    zeta_label = fresh_label("zeta")
    xi_label = fresh_label("xi")
    gamma_label = fresh_label("gamma")
    g = get_globals()
    createVariables(x_label, 2)
    createVariables(zeta_label, xi_label, gamma_label, 3)

    x1 = g[f"{x_label}1"]
    x2 = g[f"{x_label}2"]
    xi2 = g[f"{xi_label}2"]
    gamma1 = g[f"{gamma_label}1"]
    gamma2 = g[f"{gamma_label}2"]
    BARzeta1 = g[f"BAR{zeta_label}1"]
    BARzeta2 = g[f"BAR{zeta_label}2"]

    D_x1 = g[f"D_{x_label}1"]
    D_x2 = g[f"D_{x_label}2"]
    D_xi1 = g[f"D_{xi_label}1"]
    D_xi3 = g[f"D_{xi_label}3"]
    D_gamma1 = g[f"D_{gamma_label}1"]
    D_gamma3 = g[f"D_{gamma_label}3"]

    d_x1 = g[f"d_{x_label}1"]
    d_x2 = g[f"d_{x_label}2"]
    d_zeta1 = g[f"d_{zeta_label}1"]
    d_zeta2 = g[f"d_{zeta_label}2"]
    d_BARzeta2 = g[f"d_BAR{zeta_label}2"]

    vf1 = D_x1 + gamma2 * D_xi3 - x2 * xi2 * D_gamma1
    vf2 = D_x2 + gamma1 * D_gamma3 - x1 * gamma2 * D_xi1
    bracket = LieDerivative(vf1, vf2)
    assert (bracket - (-x2 * xi2 * D_gamma3 - gamma2 * D_xi1 + xi2 * D_gamma1)).is_zero

    df_first_summand = exteriorProduct(
        d_zeta1, BARzeta2 * d_BARzeta2 + BARzeta1 * d_zeta2
    )
    assert (
        df_first_summand
        - (BARzeta2 * d_zeta1 * d_BARzeta2 + BARzeta1 * d_zeta1 * d_zeta2)
    ).is_zero

    assert LieDerivative(D_x1, x1**2 + x2) == 2 * x1
    assert LieDerivative(D_x1, x1**2 * d_x1 * d_x2) == 2 * x1 * d_x1 * d_x2
    assert exteriorDerivative(x1 + x2) == d_x1 + d_x2
    assert exteriorDerivative(x1 * d_x2) == d_x1 * d_x2
    assert interiorProduct(D_x1, d_x1 * d_x2 + d_x1) == d_x2 + 1
    assert wedge(d_x1 - d_x2, d_x1 + d_x2, D_gamma1).coeff_dict == {
        (0, 1, 9, 0, 0, 1, x_label, x_label, zeta_label): 2
    }


def test_decompose(two_standard_systems):
    s = two_standard_systems
    x2 = s["x2"]
    D_x1 = s["Dx1"]
    d_x1 = s["dx1"]
    y2 = s["y2"]
    D_y1 = s["Dy1"]
    d_y1 = s["dy1"]

    objs = [x2 * d_y1 + y2 * d_x1, d_y1, D_x1, x2 * D_y1 + y2 * D_x1]
    assert decompose(d_x1, objs)[0] == [1 / y2, -x2 / y2, 0, 0]
    assert decompose(D_y1, objs)[0] == [0, 0, -y2 / x2, 1 / x2]


def test_annihilator(clean_vmf, fresh_label):
    u_label = fresh_label("u")
    g = get_globals()
    createVariables(u_label, 4, withVF=True, initialIndex=0)
    u = [g[f"{u_label}{idx}"] for idx in range(4)]
    u0, u1, u2, u3 = u
    d_u0, d_u1, d_u2, d_u3 = [g[f"d_{u_label}{idx}"] for idx in range(4)]
    quadric = u0 + sum([j**2 for j in u[1:]])
    contact_form = exteriorDerivative(quadric)
    expected_form = d_u0 + 2 * u1 * d_u1 + 2 * u2 * d_u2 + 2 * u3 * d_u3
    assert wedge(contact_form, expected_form).is_zero
    contact_dist = annihilator([contact_form], u)
    assert len(contact_dist) == 3
    assert all(contact_form(vf) == 0 for vf in contact_dist)


def test_get_coframe(clean_vmf, fresh_label):
    eta_label = fresh_label("eta")
    g = get_globals()
    createVariables(eta_label, 4, withVF=True)
    eta1, eta2 = [g[f"{eta_label}{idx}"] for idx in range(1, 3)]
    D_eta1, D_eta2 = [g[f"D_{eta_label}{idx}"] for idx in range(1, 3)]
    vf_list = [D_eta1 - eta1 * D_eta2, D_eta2 - eta1 * D_eta1]
    coframe = get_coframe(vf_list)
    assert simplify_dgcv(coframe[0](vf_list[0])) == 1
    assert simplify_dgcv(coframe[0](vf_list[1])) == 0
    assert simplify_dgcv(coframe[1](vf_list[1])) == 1
    assert simplify_dgcv(coframe[1](vf_list[0])) == 0
