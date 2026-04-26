import pytest  # type: ignore

from dgcv._aux._backends._types_and_constants import symbol
from dgcv.core.arrays.arrays import (
    _spool,
    _unspool,
    array_dgcv,
    freeze_array,
    freeze_matrix,
    frozen_array_dgcv,
    frozen_matrix_dgcv,
    matrix_dgcv,
)

# --- spool helpers -----------------------------------------------------------


def test_spool_and_unspool_are_inverses_on_small_shape():
    shape = (2, 3, 4)

    for idx in [
        (0, 0, 0),
        (0, 1, 2),
        (1, 2, 3),
    ]:
        flat = _spool(idx, shape)
        assert _unspool(flat, shape) == idx


# --- array_dgcv construction -------------------------------------------------


def test_array_dgcv_from_nested_list_infers_shape_and_values():
    arr = array_dgcv([[1, 0], [0, 2]])

    assert arr.shape == (2, 2)
    assert arr.ndim == 2
    assert arr[0, 0] == 1
    assert arr[0, 1] == 0
    assert arr[1, 0] == 0
    assert arr[1, 1] == 2


def test_array_dgcv_from_dict_requires_shape():
    with pytest.raises(TypeError, match="dict input requires shape"):
        array_dgcv({(0, 0): 1})


def test_array_dgcv_from_dict_respects_null_return():
    arr = array_dgcv({(0, 0): 5}, shape=(2, 2), null_return="missing")

    assert arr[0, 0] == 5
    assert arr[1, 1] == "missing"


def test_array_dgcv_entry_rule_builds_expected_array():
    arr = array_dgcv(shape=(2, 3), entry_rule=lambda i, j: i + j)

    assert arr.shape == (2, 3)
    assert arr[0, 0] == 0
    assert arr[0, 2] == 2
    assert arr[1, 2] == 3


# --- array_dgcv basic behavior ----------------------------------------------


def test_array_dgcv_len_and_iteration_follow_shape():
    arr = array_dgcv([[1, 0], [0, 2]])

    assert len(arr) == 4
    assert list(arr) == [1, 0, 0, 2]


def test_array_dgcv_slice_returns_array():
    arr = array_dgcv([10, 20, 30, 40])

    sub = arr[1:3]

    assert isinstance(sub, array_dgcv)
    assert sub.shape == (2,)
    assert list(sub) == [20, 30]


def test_array_dgcv_tuple_slice_returns_subarray():
    arr = array_dgcv([[1, 2, 3], [4, 5, 6]])

    sub = arr[:, 1:]

    assert isinstance(sub, array_dgcv)
    assert sub.shape == (2, 2)
    assert sub[0, 0] == 2
    assert sub[0, 1] == 3
    assert sub[1, 0] == 5
    assert sub[1, 1] == 6


def test_array_dgcv_apply_returns_new_array():
    arr = array_dgcv([[1, 2], [3, 4]])

    out = arr.apply(lambda x: 10 * x)

    assert out is not arr
    assert out.shape == arr.shape
    assert list(out) == [10, 20, 30, 40]
    assert list(arr) == [1, 2, 3, 4]


def test_array_dgcv_subs_replaces_symbolic_entries():
    a = symbol("a")
    arr = array_dgcv([[a, 0], [0, a + 1]])

    out = arr.subs({a: 3})

    assert out[0, 0] == 3
    assert out[1, 1] == 4


# --- frozen_array_dgcv ------------------------------------------------------


def test_freeze_array_returns_frozen_array():
    arr = array_dgcv([[1, 2], [3, 4]])

    frozen = freeze_array(arr)

    assert isinstance(frozen, frozen_array_dgcv)
    assert frozen.shape == arr.shape
    assert list(frozen) == list(arr)


def test_frozen_array_is_immutable():
    arr = array_dgcv([[1, 2], [3, 4]])
    frozen = freeze_array(arr)

    with pytest.raises(TypeError, match="immutable"):
        frozen[0, 0] = 99


# --- matrix_dgcv basics -----------------------------------------------------


def test_matrix_dgcv_from_nested_list_has_expected_shape():
    M = matrix_dgcv([[1, 2], [3, 4]])

    assert M.shape == (2, 2)
    assert M.nrows == 2
    assert M.ncols == 2
    assert M[0, 0] == 1
    assert M[1, 1] == 4


def test_matrix_dgcv_identity():
    M = matrix_dgcv.identity(3)

    assert M.shape == (3, 3)
    assert M[0, 0] == 1
    assert M[1, 1] == 1
    assert M[2, 2] == 1
    assert M[0, 1] == 0


def test_matrix_dgcv_addition():
    A = matrix_dgcv([[1, 2], [3, 4]])
    B = matrix_dgcv([[10, 20], [30, 40]])

    C = A + B

    assert C.tolist() == [[11, 22], [33, 44]]


def test_matrix_dgcv_matrix_multiplication():
    A = matrix_dgcv([[1, 2], [3, 4]])
    B = matrix_dgcv([[2, 0], [1, 2]])

    C = A @ B

    assert C.tolist() == [[4, 4], [10, 8]]


def test_matrix_dgcv_trace_and_det():
    M = matrix_dgcv([[1, 2], [3, 4]])

    assert M.trace() == 5
    assert M.det() == -2


# --- frozen_matrix_dgcv -----------------------------------------------------


def test_freeze_matrix_returns_frozen_matrix():
    M = matrix_dgcv([[1, 2], [3, 4]])

    frozen = freeze_matrix(M)

    assert isinstance(frozen, frozen_matrix_dgcv)
    assert frozen.tolist() == [[1, 2], [3, 4]]


def test_frozen_matrix_is_immutable():
    M = freeze_matrix(matrix_dgcv([[1, 2], [3, 4]]))

    with pytest.raises(TypeError, match="immutable"):
        M[0, 0] = 99
