"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.algebras

module: algebras.algebras_aux


---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

SPDX-License-Identifier: Apache-2.0


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# -----------------------------------------------------------------------------
# imports
# -----------------------------------------------------------------------------
import numbers

from .._aux._backends._calculus import diff
from .._aux._backends._symbolic_router import (
    _scalar_is_zero,
    get_free_symbols,
    simplify,
)
from .._aux._backends._types_and_constants import is_atomic, symbol
from .._aux._utilities._config import dgcv_exception_note, dgcv_warning
from .._aux._vmf._safeguards import (
    create_key,
    get_dgcv_category,
    query_dgcv_categories,
    retrieve_public_key,
)
from .._aux._vmf.vmf import clearVar, listVar
from ..core.arrays.arrays import array_dgcv, freeze_matrix, matrix_dgcv
from ..core.dgcv_core.dgcv_core import VF_bracket
from ..core.solvers.solvers import solve_dgcv


# -----------------------------------------------------------------------------
# utilities
# -----------------------------------------------------------------------------
def _as_matrix(obj):
    if isinstance(obj, matrix_dgcv):
        return obj
    try:
        return matrix_dgcv(obj)
    except Exception:
        return None


def _validate_structure_data(
    data,
    process_matrix_rep=False,
    assume_skew=False,
    assume_Lie_alg=False,
    basis_order_for_supplied_str_eqns=None,
    process_tensor_rep=False,
    determinacy_order_ansatz=None,
    dimension=None,
):
    if process_tensor_rep:
        # try:
        return algebraDataFromTensorRep(data), "tensor"
        # except Exception as e:
        #     raise dgcv_exception_note(f"{e}") from None
    if process_matrix_rep:
        mats = []
        for obj in data:
            m = _as_matrix(obj)
            if m is None or m.nrows != m.ncols:
                mats = None
                break
            mats.append(m)
        if mats is not None:
            try:
                return algebraDataFromMatRep(data), "matrix"
            except Exception as e:
                raise dgcv_exception_note(f"{e}") from None
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
                    data, determinacy_order_ansatz=determinacy_order_ansatz
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
                        data, determinacy_order_ansatz=determinacy_order_ansatz
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
                if not isinstance(
                    basis_order_for_supplied_str_eqns, (list, tuple)
                ) or not all(
                    is_atomic(var) for var in basis_order_for_supplied_str_eqns
                ):
                    raise ValueError(
                        "If initializing an algebra from structure equations and supplying the `basis_order_for_supplied_str_eqns` parameter, this parameter should be a list of the atomic variables appearing in the supplied structure equations."
                    )
                for var in tuple_vars:
                    if var not in basis_order_for_supplied_str_eqns:
                        if build_basis_order:
                            basis_order_for_supplied_str_eqns.append(var)
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
                for idx_pair, val in data.items():
                    if not _scalar_is_zero(val):
                        params |= get_free_symbols(val)
                        v1, v2 = idx_pair
                        idx1 = ordered_BV.index(v1)
                        idx2 = ordered_BV.index(v2)

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


def aDataFromVFWithAnsatz(graded_components, determinacy_order_ansatz=None):
    if not isinstance(graded_components, dict):
        grading = None
        basis = graded_components  # assumed to be iterable
        graded_components = {0: graded_components}
    else:
        basis = []
        grading = []
        for weight, component in graded_components.items():
            grading += [weight] * len(component)
            basis += list(component)
    free_symbols = set()
    for vf in basis:
        free_symbols |= get_free_symbols(vf)
    vlabel = create_key("var")
    var_dict = {
        k: [symbol(f"{vlabel}{k}_{idx}") for idx in range(len(v))]
        for k, v in graded_components.items()
    }
    gen_elements = {
        k: sum(var * elem for var, elem in zip(var_dict[k], v))
        for k, v in graded_components.items()
    }
    dim = len(basis)

    structure_data = array_dgcv(
        dict(), shape=(dim, dim), null_return=freeze_matrix(matrix_dgcv.zeros(dim, 1))
    )
    params = set()

    if determinacy_order_ansatz is None:
        order_bound = len(basis)
    else:
        order_bound = determinacy_order_ansatz
    for c1, vf1 in enumerate(basis):
        for c, vf2 in enumerate(basis[c1 + 1 :]):
            c2 = c1 + 1 + c
            new_weight = 0 if grading is None else grading[c1] + grading[c2]
            if new_weight in gen_elements:
                genelement = gen_elements[new_weight]
                variables = var_dict[new_weight]
            else:
                genelement = 0
                variables = [symbol("_dgcv_var_")]
            liebracket = VF_bracket(vf1, vf2)
            eqns = [
                eqn
                for eqn in (liebracket - (genelement)).coeff_dict.values()
                if eqn != 0
            ]
            prev_eqns = eqns
            for _ in range(order_bound):
                p_e = prev_eqns
                prev_eqns = []
                for j in p_e:
                    for var in free_symbols:
                        neqn = diff(
                            j, var
                        )  ###!!! may be good to prune free_symbols here
                        if neqn != 0:
                            prev_eqns.append(neqn)
                if len(prev_eqns) == 0:
                    break
                eqns += prev_eqns
            sol = solve_dgcv(eqns, variables, method="linsolve")
            if not sol:
                raise RuntimeError(
                    f"Given vector field list is not closed under Lie brackets at indices ({c1}, {c2})."
                )

            sol = sol[0]
            counter, result = 0, (matrix_dgcv.zeros(dim, 1))
            for idx in range(dim):
                weight = 0 if grading is None else grading[idx]
                if weight == new_weight:
                    newcoeff = sol.get(variables[counter])
                    if newcoeff != 0:
                        params |= get_free_symbols(newcoeff)
                        result[idx] = newcoeff
                    counter += 1
            structure_data[c1, c2] = result
            structure_data[c2, c1] = -result
    return structure_data, params, grading


def aDataFromNestedLists(nested_lists):
    dim = len(nested_lists)
    sd = array_dgcv(
        dict(),
        shape=(dim, dim),
        null_return=freeze_matrix(matrix_dgcv.zeros(dim, 1)),
    )
    params = set()
    for idx1, outer in enumerate(nested_lists):
        if len(outer) != dim:
            raise TypeError()
        for idx2, middle in enumerate(outer):
            if len(middle) != dim:
                raise TypeError()
            inner_dict = dict()
            for c, v in enumerate(middle):
                if v != 0 and v is not None:
                    params |= get_free_symbols(v)
                    inner_dict[c] = v
            if inner_dict:
                sd[idx1, idx2] = matrix_dgcv(inner_dict, shape=(dim, 1))
    return sd, params


def algebraDataFromMatRep(mat_list):
    """
    Create the structure data array for a Lie algebra from a list of matrices in *mat_list*.
    """
    if not mat_list:
        return (
            array_dgcv(
                dict(),
                shape=(0, 0),
                null_return=freeze_matrix(matrix_dgcv.zeros(0, 1)),
            ),
            mat_list,
            set(),
        )

    shape = mat_list[0].shape
    indexRangeCap = len(mat_list)
    if not all(m.shape == shape for m in mat_list):
        raise Exception(
            "algorithm for extracting algebra data from matrices expects a list of square matrices of the same size."
        )

    tempVarLabel = "T" + retrieve_public_key()
    variables = [symbol(f"{tempVarLabel}{idx}") for idx in range(indexRangeCap)]
    combiMatLoc = sum(var * mat for var, mat in zip(variables, mat_list))
    params = set()

    def pairValue(j, k, par):
        mat = (mat_list[j] @ mat_list[k]) - (mat_list[k] @ mat_list[j]) - combiMatLoc

        bracketVals = list(set(mat._data.values()))
        if not bracketVals:
            return
        if len(bracketVals) == 1 and _scalar_is_zero(bracketVals[0]):
            return

        solLoc = list(solve_dgcv(bracketVals, variables))
        if len(solLoc) > 0:
            soll = solLoc[0]
            coeffs = matrix_dgcv.zeros(indexRangeCap, 1)
            for idx, var in enumerate(variables):
                coeff = soll.get(var, var)
                if coeff != 0:
                    par |= get_free_symbols(coeff)
                    coeffs[idx] = coeff
            return coeffs
        raise Exception(
            f"Unable to determine if matrices are closed under commutators. "
            f"Problem matrices are in positions {j} and {k}."
        )

    structure_data = array_dgcv(
        dict(),
        shape=(indexRangeCap, indexRangeCap),
        null_return=freeze_matrix(matrix_dgcv.zeros(indexRangeCap, 1)),
    )
    for j in range(indexRangeCap):
        for k in range(j + 1, indexRangeCap):
            br = pairValue(k, j, params)
            if len(br._data) > 0:
                structure_data[k, j] = br
                structure_data[j, k] = -br

    return structure_data, mat_list, params


def algebraDataFromTensorRep(tensor_list):
    """
    Create the structure data array from a list of tensor products closed under the `_contraction_product` operator (see dgcv.tensor_field_class documentation).
    """

    tempVarLabel = "T" + create_key()
    dim = len(tensor_list)
    if dim == 0:
        return (
            array_dgcv(
                dict(), shape=(0, 0), null_return=freeze_matrix(matrix_dgcv.zeros(0, 1))
            ),
            tensor_list,
            set(),
        )
    vars = [symbol(f"{tempVarLabel}{idx}") for idx in range(dim)]
    gen_elem = sum([vars[j] * tensor_list[j] for j in range(dim)])

    params = set()

    def computeBracket(j, k, par):
        if k < j:
            return
        product = (tensor_list[j] * tensor_list[k]) - gen_elem
        solutions = solve_dgcv(product, vars)
        if len(solutions) > 0:
            sol_values = solutions[0]
            coeffs = matrix_dgcv.zeros(dim, 1)
            for idx, var in enumerate(vars):
                coeff = sol_values.get(var, var)
                if coeff != 0:
                    par |= get_free_symbols(coeff)
                    coeffs[idx] = coeff
            return coeffs
        else:
            clearVar(*listVar(temporary_only=True), report=False)
            raise Exception(
                f"Contraction product of tensors at positions {j} and {k} are not in the given tensor list."
            )

    structure_data = array_dgcv(
        dict(), shape=(dim, dim), null_return=freeze_matrix(matrix_dgcv.zeros(dim, 1))
    )

    for j in range(dim):
        for k in range(j):
            br = computeBracket(k, j, params)
            if len(br._data) > 0:
                structure_data[k, j] = br  # CHECK index order!!!
                structure_data[j, k] = -br

    return structure_data, tensor_list, params  # filter independants
