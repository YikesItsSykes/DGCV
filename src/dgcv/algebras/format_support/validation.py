import numbers

from ..._aux._backends._symbolic_router import (
    _scalar_is_zero,
    get_free_symbols,
    simplify,
)
from ..._aux._backends._types_and_constants import is_atomic
from ..._aux._utilities._config import dgcv_warning
from ..._aux._vmf._safeguards import get_dgcv_category, query_dgcv_categories
from ...core.arrays import array_dgcv, freeze_matrix, matrix_dgcv
from .dense_lists import aDataFromNestedLists
from .matrices import algebraDataFromMatRep
from .tensors import algebraDataFromTensorRep
from .vector_fields import aDataFromVFWithAnsatz


def _validate_structure_data(
    data,
    process_matrix_rep=False,
    assume_skew=False,
    assume_Lie_alg=False,
    basis_order_for_supplied_str_eqns=None,
    process_tensor_rep=False,
    determinacy_order_ansatz=None,
    dimension=None,
    process_with_decompose=False,
):
    if process_tensor_rep:
        # try:
        return algebraDataFromTensorRep(data), "tensor"
        # except Exception as e:
        #     raise dgcv_exception_note(f"{e}") from None
    if process_matrix_rep:
        mats, warningmessage = [], False
        for obj in data:
            if isinstance(obj, matrix_dgcv):
                m = obj
            try:
                m = matrix_dgcv(obj)
            except Exception:
                m = None
            if m is None or m.nrows != m.ncols:
                mats = None
                if warningmessage is False:
                    warningmessage = "Some basis objects could not be interpreted as square matrices for a matrix algebra interpretation, and were discarded."
                break
            mats.append(m)
        if warningmessage is not False:
            dgcv_warning(warningmessage)
        if mats is not None:
            # try:
            return algebraDataFromMatRep(data), "matrix"
            # except Exception as e:
            #     raise dgcv_exception_note(f"{e}") from None
        elif all(get_dgcv_category(elem) == "tensorProduct" for elem in data):
            dgcv_warning(
                "`_validate_structure_data` was given a list of tensorProduct instance, but `process_matrix_rep` was also marked True. The latter was ignored."
            )
            return _validate_structure_data(
                data,
                process_matrix_rep=False,
                assume_skew=assume_skew,
                assume_Lie_alg=assume_Lie_alg,
                basis_order_for_supplied_str_eqns=basis_order_for_supplied_str_eqns,
                process_tensor_rep=True,
            )
        else:
            raise ValueError(
                "matrix representation processing requires a list of square matrices."
            )

    if isinstance(data, (list, tuple)):
        if len(data) > 0:
            if all(query_dgcv_categories(obj, {"vector_field"}) for obj in data):
                return aDataFromVFWithAnsatz(
                    data,
                    determinacy_order_ansatz=determinacy_order_ansatz,
                    process_with_decompose=process_with_decompose,
                )
            elif all(query_dgcv_categories(obj, {"matrix"}) for obj in data):
                return algebraDataFromMatRep(data), "matrix"
            else:
                try:
                    return aDataFromNestedLists(data)
                except Exception:
                    raise TypeError(
                        "The algebra_class initializer recieved data in an unsupported format."
                    )
        else:
            return array_dgcv(
                dict(),
                shape=(0, 0),
                null_return=freeze_matrix(matrix_dgcv.zeros(0, 1)),
            ), set()
    try:
        if isinstance(data, dict):
            if all(
                isinstance(key, numbers.Integral)
                and all(query_dgcv_categories(obj, {"vector_field"}) for obj in val)
                for key, val in data.items()
            ):
                try:
                    return aDataFromVFWithAnsatz(
                        data,
                        determinacy_order_ansatz=determinacy_order_ansatz,
                        process_with_decompose=process_with_decompose,
                    )
                except Exception:
                    raise TypeError(
                        "`createAlgebra` could not extract a Lie algebra structure from the given vector fields with the indicated grading. If indicating a grading was unintended, then provide the fields in a list instead; if that also fails, then they may not span a Lie algebra."
                    )
            if all(
                isinstance(key, tuple)
                and len(key) == 2
                and all(is_atomic(idx) for idx in key)
                for key in data
            ):
                tuple_vars = set()
                for key in data:
                    tuple_vars.add(key[0])
                    tuple_vars.add(key[1])
                if basis_order_for_supplied_str_eqns is None:
                    build_basis_order = True
                    basis_order_for_supplied_str_eqns = []
                else:
                    build_basis_order = False
                    if len(tuple_vars) < len(basis_order_for_supplied_str_eqns):
                        tuple_vars |= set(basis_order_for_supplied_str_eqns)
                if not isinstance(
                    basis_order_for_supplied_str_eqns, (list, tuple)
                ) or not all(
                    is_atomic(var) for var in basis_order_for_supplied_str_eqns
                ):
                    raise ValueError(
                        "If initializing an algebra from structure equations and supplying the `basis_order_for_supplied_str_eqns` parameter, this parameter should be a list of the atomic variables appearing in the supplied structure equations."
                    )
                bo_names = {str(var) for var in basis_order_for_supplied_str_eqns}
                for var in tuple_vars:
                    if str(var) not in bo_names:
                        if build_basis_order:
                            basis_order_for_supplied_str_eqns.append(var)
                            bo_names.add(str(var))
                        else:
                            raise ValueError(
                                "If initializing an algebra from structure equations and supplying the `basis_order_for_supplied_str_eqns` parameter, this parameter should be a list containing all atomic variables appearing in the supplied structure equations."
                            )
                ordered_BV = basis_order_for_supplied_str_eqns
                zeroing = {var: 0 for var in ordered_BV}
                dim = len(ordered_BV)
                structure_data = array_dgcv(
                    dict(),
                    shape=(dim, dim),
                    null_return=freeze_matrix(matrix_dgcv.zeros(dim, 1)),
                )
                params = set()
                ordered_BV_index = {str(v): i for i, v in enumerate(ordered_BV)}
                for idx_pair, val in data.items():
                    if not _scalar_is_zero(val):
                        params |= get_free_symbols(val)
                        v1, v2 = idx_pair
                        idx1 = ordered_BV_index[str(v1)]
                        idx2 = ordered_BV_index[str(v2)]

                        if hasattr(val, "subs") and _scalar_is_zero(val.subs(zeroing)):
                            coeffs = matrix_dgcv.zeros(dim, 1)
                            for idx, var in enumerate(ordered_BV):
                                subdict = zeroing | {var: 1}
                                coeffs[idx] = simplify(val.subs(subdict))
                            structure_data[idx1, idx2] = coeffs
                            if assume_skew or assume_Lie_alg:
                                invert_idx = structure_data._spool((idx2, idx1))
                                if invert_idx in structure_data._data:
                                    if not all(
                                        v == 0
                                        for v in (
                                            coeffs + structure_data._data[invert_idx]
                                        )._data.values()
                                    ):
                                        raise ValueError(
                                            "Either `assume_skew=True` or `assume_Lie_alg=True` was passed to the algebra contructor, but the accompanying structure data was not skew symmetric."
                                        )
                                else:
                                    structure_data[idx2, idx1] = -coeffs
                        else:
                            raise ValueError(
                                "If initializing an algebra from structure equations, supplied structure equations should be a dictionary whose keys are tuples of atomic variables and whose values are linear combinations of variables representing the product of the elements in the key tuple. If that is the case then you are likely getting this error because you did not supply the algebra creator with a valid value for the `basis_order_for_supplied_str_eqns` parameter. If that paremeter were omitted, it is not always possible to unambiguously infer its proper value from general structure equations data, which can lead to this error."
                            )
                return structure_data, {x for x in params if x not in tuple_vars}
            if get_dgcv_category(data) == "array":
                shp = data.shape
                if (
                    isinstance(shp, (tuple, list))
                    and len(shp) == 2
                    and shp[0] == shp[1]
                ):
                    dimension = shp[0]
                    data = {k: v for k, v in data._data_unspooled.items() if v}
            if all(
                isinstance(key, tuple)
                and len(key) == 2
                and all(isinstance(idx, numbers.Integral) and idx >= 0 for idx in key)
                for key in data
            ):
                provided_index_bound = (
                    max(max(key) for key in data.keys())
                    if dimension is None
                    else max(dimension, max(max(key) for key in data.keys()))
                )
            else:
                raise ValueError(
                    "Structure data must be in one of several formats. E.g.: It can be a list/tuple with 3D shape of size (x, x, x). Or it can be a sparse dictionairy of the (i,j) entries for the structure data. Set `process_matrix_rep=True` to initialize from a matrix representation, or provide a list of vector fields to initialize from a VF rep."
                )
            try:
                formatted_data = {}
                base_dim = None
                for key, value in data.items():
                    mat = matrix_dgcv(value)
                    if not mat:
                        continue
                    if isinstance(value, (list, tuple)):
                        ol = len(value)
                        formatted_data[key] = mat
                    elif get_dgcv_category(value) == "array":
                        shp = value.shape
                        if len(shp) != 2:
                            raise RuntimeError()
                        if shp[1] != 1 and shp[0] == 1:
                            ol = shp[0]
                            formatted_data[key] = mat.transpose()
                        else:
                            formatted_data[key] = mat
                    else:
                        raise RuntimeError()
                    if base_dim is None:
                        ol = max(provided_index_bound, ol)
                        base_dim = ol
                    if base_dim != ol:
                        raise ValueError(
                            "If initializing an algebra with structure data from a dictionairy, its keys should be (i,j) index tuples and its values should be list-like structures of coefficients from the product of i and j basis elements. All values lists must have the same length in particular. Indices in the keys must not exceed the length of value tuples - 1 (as indexing starts from 0!)"
                        )
                base_dim = (
                    base_dim
                    if base_dim is not None
                    else dimension
                    if dimension is not None
                    else 0
                )
                if assume_skew or assume_Lie_alg:
                    seen = set()
                    initial_keys = list(formatted_data.keys())
                    for idx in initial_keys:
                        if idx in seen:
                            pass
                        else:
                            invert_idx = (idx[1], idx[0])
                            if invert_idx in formatted_data.keys():
                                if any(
                                    j != 0
                                    for j in formatted_data[idx]
                                    + formatted_data[invert_idx]
                                ):
                                    raise ValueError(
                                        "Either `assume_skew=True` or `assume_Lie_alg=True` was passed to the algebra contructor, but the accompanying structure data was not skew symmetric."
                                    )
                            else:
                                formatted_data[invert_idx] = matrix_dgcv(
                                    [-j for j in formatted_data[idx]]
                                )
                            seen.add(idx)
                            seen.add(invert_idx)

                data = array_dgcv(
                    formatted_data,
                    shape=(base_dim, base_dim),
                    null_return=freeze_matrix(matrix_dgcv({}, shape=(base_dim, 1))),
                )
            except Exception:
                raise ValueError(
                    "If initializing an algebra algebra with structure data from a dictionairy, its keys should be (i,j) index tuples and its values should be tuples of coefficients from the product of i and j basis elements. All values tuples must have the same length in particular."
                )
        params = set()
        for j in data._data.values():
            params |= get_free_symbols(j)

        return data, params  # structure data array, parameters

    except Exception as e:
        raise ValueError(f"Invalid structure data format: {type(data)} - {e}")
