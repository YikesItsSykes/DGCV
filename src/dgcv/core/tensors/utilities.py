from ..._aux._vmf._safeguards import get_dgcv_category


def multi_tensor_product(*tp):
    product = 1
    tpTypes = {
        "tensorProduct",
        "algebra_element",
        "subalgebra_element",
    }
    for elem in tp:
        if get_dgcv_category(elem) not in tpTypes:
            raise TypeError(
                f"multi_tensor_product only excepts arguments that `dgcv` can process as factors in a tensor product. Recieved type {type(elem)}"
            )
        product = product @ elem
    return product
