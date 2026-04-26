import pytest  # type: ignore

from dgcv._aux._utilities._config import get_globals
from dgcv.algebras.algebras_secondary import createAlgebra, createSimpleLieAlgebra
from dgcv.core.arrays.arrays import matrix_dgcv
from dgcv.core.dgcv_core.dgcv_core import createVariables
from dgcv.secondary_library import defloat


def test_create_algebra_from_mat_vf_CSL(clean_vmf, fresh_label):

    sl3_structure = {
        16: {2: -1},
        2: {2: 1},
        24: {3: 1},
        3: {3: -1},
        32: {4: -1},
        4: {4: 1},
        40: {5: 1},
        5: {5: -1},
        33: {4: -1},
        12: {4: 1},
        41: {5: 1},
        13: {5: -1},
        49: {6: -1},
        14: {6: 1},
        57: {7: 1},
        15: {7: -1},
        26: {0: -2, 1: 1},
        19: {0: 2, 1: -1},
        42: {7: 1},
        21: {7: -1},
        50: {4: -1},
        22: {4: 1},
        35: {6: -1},
        28: {6: 1},
        59: {5: 1},
        31: {5: -1},
        44: {0: -1, 1: -1},
        37: {0: 1, 1: 1},
        60: {2: -1},
        39: {2: 1},
        53: {3: 1},
        46: {3: -1},
        62: {0: 1, 1: -2},
        55: {0: -1, 1: 2},
    }

    grading = [(0, 0, 1, -1, 1, -1, 0, 0), (0, 0, 0, 0, 1, -1, 1, -1)]

    matrices = defloat(
        [
            matrix_dgcv([[2 / 3, 0, 0], [0, -1 / 3, 0], [0, 0, -1 / 3]]),
            matrix_dgcv([[1 / 3, 0, 0], [0, 1 / 3, 0], [0, 0, -2 / 3]]),
            matrix_dgcv([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
            matrix_dgcv([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
            matrix_dgcv([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
            matrix_dgcv([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
            matrix_dgcv([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
            matrix_dgcv([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
        ]
    )
    alg1 = createAlgebra(
        matrices,
        label="alg1",
        grading=grading,
        process_matrix_rep=True,
        return_created_object=True,
    )

    x_label = fresh_label("x")

    createVariables(x_label, 3)

    g = get_globals()

    x1, x2, x3 = [g[f"{x_label}{idx}"] for idx in range(1, 4)]
    D_x1, D_x2, D_x3 = [g[f"D_{x_label}{idx}"] for idx in range(1, 4)]
    vector_fields = [
        2 * x1 / 3 * D_x1 - x2 / 3 * D_x2 - x3 / 3 * D_x3,
        x1 / 3 * D_x1 + x2 / 3 * D_x2 - 2 * x3 / 3 * D_x3,
        x1 * D_x2,
        x2 * D_x1,
        x1 * D_x3,
        x3 * D_x1,
        x2 * D_x3,
        x3 * D_x2,
    ]
    alg2 = createAlgebra(
        vector_fields, "alg2", grading=grading, return_created_object=True
    )

    alg3 = createSimpleLieAlgebra("A2", "alg3", "e", return_created_object=True)

    for idx1 in range(64):
        for idx2 in range(8):
            for alg in [alg1, alg2, alg3]:
                assert alg.structureData[idx1][idx2] == sl3_structure.get(idx1, {}).get(
                    idx2, 0
                )
