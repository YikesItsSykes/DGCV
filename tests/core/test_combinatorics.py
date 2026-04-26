from dgcv.core.combinatorics.combinatorics import (
    Baker_Campbell_Hausdorff,
    carProd,
    chooseOp,
    permSign,
    shufflings,
    weightedPermSign,
)

# --- carProd -----------------------------------------------------------------


def test_carProd_two_lists():
    result = list(carProd([1, 2], ["a", "b"]))
    assert result == [
        (1, "a"),
        (1, "b"),
        (2, "a"),
        (2, "b"),
    ]


def test_carProd_single_list_wraps_entries_as_1_tuples():
    result = list(carProd([1, 2, 3]))
    assert result == [(1,), (2,), (3,)]


# --- chooseOp ----------------------------------------------------------------


def test_chooseOp_basic_with_replacement_and_order():
    result = list(chooseOp([1, 2], 2))
    assert result == [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ]


def test_chooseOp_without_replacement_and_with_order():
    result = list(chooseOp([1, 2, 3], 2, withOrder=True, withoutReplacement=True))
    assert result == [
        (1, 2),
        (1, 3),
        (2, 3),
    ]


def test_chooseOp_with_restrict_homogeneity():
    result = list(chooseOp([1, 2, 3], 2, restrictHomogeneity=4))
    assert result == [
        (1, 3),
        (2, 2),
        (3, 1),
    ]


# --- permSign ----------------------------------------------------------------


def test_permSign_identity_and_transposition():
    assert permSign([1, 2, 3]) == 1
    assert permSign([2, 1, 3]) == -1


def test_permSign_return_sorted():
    sign, sorted_list = permSign([3, 1, 2], returnSorted=True)
    assert sign == 1
    assert sorted_list == [1, 2, 3]


# --- weightedPermSign --------------------------------------------------------


def test_weightedPermSign_unweighted_even_case():
    sign = weightedPermSign([3, 1, 2], [1, 1, 1])
    assert sign == 1


def test_weightedPermSign_weighted_odd_case():
    sign = weightedPermSign([2, 1], [1, 1])
    assert sign == -1


def test_weightedPermSign_return_sorted():
    sign, sorted_perm, sorted_weights = weightedPermSign(
        [3, 1, 2],
        [2, 5, 7],
        returnSorted=True,
    )
    assert sign == 1
    assert sorted_perm == [1, 2, 3]
    assert sorted_weights == [5, 7, 2]


# --- shufflings --------------------------------------------------------------


def test_shufflings_small_case():
    result = list(shufflings([1, 2], ["a"]))
    assert result == [
        [1, 2, "a"],
        [1, "a", 2],
        ["a", 1, 2],
    ]


def test_shufflings_empty_second_list():
    result = list(shufflings([1, 2], []))
    assert result == [[1, 2]]


# --- Baker_Campbell_Hausdorff ------------------------------------------------


def test_Baker_Campbell_Hausdorff_smoke_with_scalar_like_ad_syntax():
    def ad_rule(x):
        def ad_x(y):
            return x - y

        return ad_x

    out = Baker_Campbell_Hausdorff(3, 5, truncation_degree=1, ad_op_syntax=ad_rule)
    assert out is not None
