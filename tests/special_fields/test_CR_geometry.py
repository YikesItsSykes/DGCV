from dgcv._aux._backends import conjugate_dgcv_sym_router, im, re
from dgcv.core import createVariables
from dgcv.special_fields.CR_geometry import CR_structure


def test_CR_structure_class(clean_vmf):
    z = createVariables("z", "x", "y", 4, initialIndex=0, return_created_object=True)[0]
    graph_function = re(
        sum(
            z[1] ** (idx) * conjugate_dgcv_sym_router(z[idx + 1]) for idx in range(1, 3)
        )
    )
    graph_variable = im(z[0])
    equation_system = {graph_variable: graph_function}
    weights = [3, 1] + [3 - j for j in range(1, 3)]
    M = CR_structure(z, defining_equations=equation_system, weights=weights)
    symmetries = M.compute_weighted_symmetries(
        range(-3, -1), report_progress=True, verbose=True
    )
    assert len(symmetries[-3]) == 1 and len(symmetries[-2]) == 2

    assert M.nondegeneracy_order == 2
