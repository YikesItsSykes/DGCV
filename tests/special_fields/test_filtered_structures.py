from dgcv._aux._utilities._config import get_globals
from dgcv.algebras import createAlgebra
from dgcv.core import createVariables
from dgcv.core.arrays import array_dgcv, frozen_matrix_dgcv
from dgcv.secondary_library import defloat
from dgcv.special_fields.filtered_structures import Tanaka_symbol, distribution


def test_Tanaka_symbol_prolong(clean_vmf):
    grading_vector = [-1, -1, -1, -2, -2, -2, -3]
    sd = defloat(
        array_dgcv(
            {
                1: array_dgcv({0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0}, shape=(7, 1)),
                7: array_dgcv(
                    {0: 0, 1: 0, 2: 0, 3: -1, 4: 0, 5: 0, 6: 0}, shape=(7, 1)
                ),
                9: array_dgcv({0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0}, shape=(7, 1)),
                15: array_dgcv(
                    {0: 0, 1: 0, 2: 0, 3: 0, 4: -1, 5: 0, 6: 0}, shape=(7, 1)
                ),
                14: array_dgcv(
                    {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0}, shape=(7, 1)
                ),
                2: array_dgcv(
                    {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: -1, 6: 0}, shape=(7, 1)
                ),
                17: array_dgcv(
                    {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: -2}, shape=(7, 1)
                ),
                23: array_dgcv(
                    {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 2}, shape=(7, 1)
                ),
                4: array_dgcv({0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1}, shape=(7, 1)),
                28: array_dgcv(
                    {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: -1}, shape=(7, 1)
                ),
                12: array_dgcv(
                    {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1}, shape=(7, 1)
                ),
                36: array_dgcv(
                    {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: -1}, shape=(7, 1)
                ),
            },
            shape=(7, 7),
            null_return=frozen_matrix_dgcv({}, shape=(7, 1)),
        )
    )
    g = createAlgebra(sd, "g", grading=grading_vector, return_created_object=True)
    symbol = Tanaka_symbol(g)
    prolongation = symbol.prolong(5)
    alg = createAlgebra(prolongation, "alg", return_created_object=True)
    assert alg.dimension == 15
    ss, nilpotent = alg.Levi_decomposition()
    assert ss.dimension == 10


def test_prolongation_with_distinguished_subspaces(clean_vmf):
    sd = array_dgcv(
        {
            3: array_dgcv({0: 0, 1: 0, 2: 0, 3: 0, 4: 1}, shape=(5, 1)),
            15: array_dgcv({0: 0, 1: 0, 2: 0, 3: 0, 4: -1}, shape=(5, 1)),
            7: array_dgcv({0: 0, 1: 0, 2: 0, 3: 0, 4: 1}, shape=(5, 1)),
            11: array_dgcv({0: 0, 1: 0, 2: 0, 3: 0, 4: -1}, shape=(5, 1)),
        },
        shape=(5, 5),
        null_return=frozen_matrix_dgcv({}, shape=(5, 1)),
    )
    grading_vector = [-1, -1, -1, -1, -2]
    m = createAlgebra(sd, "m", "e", grading=grading_vector, return_created_object=True)
    e1, e2, e3, e4, _ = m.basis
    symbol = Tanaka_symbol(m, distinguished_subspaces=([e1, e2], [e3, e4]))
    prolongation = symbol.prolong(5, report_progress=True)
    alg = createAlgebra(prolongation, "alg", return_created_object=True)
    assert alg.is_simple()


def test_distribution_class(clean_vmf, fresh_label):
    x_label = fresh_label("x")
    g = get_globals()
    createVariables(x_label, 5, withVF=True)
    x1, x2, x3, x4, x5 = [g[f"{x_label}{idx}"] for idx in range(1, 6)]
    D_x1, D_x2, D_x3, D_x4, D_x5 = [g[f"D_{x_label}{idx}"] for idx in range(1, 6)]
    createVariables("x", 5, withVF=True)
    X1 = D_x1
    X2 = -2 * D_x2 + x1 * D_x3 + x1**2 * D_x4 / 2 + x1 * x2 * D_x5
    D = distribution([X1, X2])
    assert [len(j) for j in D.weak_derived_flag()] == [2, 1, 2]
    nil_approximation = D.nilpotent_approximation(label="n", exclude_from_VMF=True)
    assert nil_approximation.dimension == 5
    assert nil_approximation.grading == [(-1, -1, -2, -3, -3)]
