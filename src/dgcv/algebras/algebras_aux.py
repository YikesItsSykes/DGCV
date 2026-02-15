"""
package: dgcv - Differential Geometry with Complex Variables
module: algebras/algebras_aux

Author (of this module): David Sykes (https://realandimaginary.com/dgcv/)

License:
    MIT License
"""

# -----------------------------------------------------------------------------
# imports
# -----------------------------------------------------------------------------
import numbers
import random
import warnings

from .._config import dgcv_exception_note
from .._safeguards import (
    create_key,
    get_dgcv_category,
    get_dgcv_settings_registry,
    query_dgcv_categories,
    retrieve_passkey,
    retrieve_public_key,
)
from ..arrays import matrix_dgcv
from ..backends._numeric_router import _extract_basis_over_number_field
from ..backends._symbolic_router import (
    _scalar_is_zero,
    get_free_symbols,
    simplify,
    subs,
)
from ..backends._types_and_constants import is_atomic, rational, symbol
from ..dgcv_core import VF_bracket, allToReal, variableProcedure
from ..solvers import solve_dgcv
from ..vmf import clearVar, listVar


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
):
    if process_tensor_rep:
        # # DEBUG
        # return algebraDataFromTensorRep(data), "tensor"
        try:
            return algebraDataFromTensorRep(data), "tensor"
        except Exception as e:
            raise dgcv_exception_note(f"{e}") from None
    if process_matrix_rep:
        mats = []
        for obj in data:
            m = _as_matrix(obj)
            if m is None or m.nrows != m.ncols:
                mats = None
                break
            mats.append(m)
        if mats is not None:
            # # DEBUG:
            # return algebraDataFromMatRep(data), "matrix"
            try:
                return algebraDataFromMatRep(data), "matrix"
            except Exception as e:
                raise dgcv_exception_note(f"{e}") from None
        elif all(get_dgcv_category(elem) == "tensorProduct" for elem in data):
            warnings.warn(
                "`_validate_structure_data` was given a list of tensorProduct instance, but `process_matrix_rep` was also marked True. The latter was ignored."
            )
            return _validate_structure_data(
                data,
                process_matrix_rep=False,
                assume_skew=assume_skew,
                assume_Lie_alg=assume_Lie_alg,
                basis_order_for_supplied_str_eqns=basis_order_for_supplied_str_eqns,
                process_tensor_rep=True,
            ), "tensor"
        else:
            raise ValueError(
                f"matrix representation processing requires a list of square matrices. Recieved: {data}"
            )

    if isinstance(data, (list, tuple)):
        if len(data) > 0:
            if all(query_dgcv_categories(obj, {"vector_field"}) for obj in data):
                return algebraDataFromVF(data)
        else:
            return tuple(), set()
    try:
        if isinstance(data, dict):
            if all(
                isinstance(key, tuple)
                and len(key) == 2
                and all(is_atomic(idx) for idx in key)
                for key in data
            ):
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
                for var in set(sum([list(key) for key in data.keys()], [])):
                    if var not in basis_order_for_supplied_str_eqns:
                        if build_basis_order:
                            basis_order_for_supplied_str_eqns.append(var)
                        else:
                            raise ValueError(
                                "If initializing an algebra from structure equations and supplying the `basis_order_for_supplied_str_eqns` parameter, this parameter should be a list containing all atomic variables appearing in the supplied structure equations."
                            )
                ordered_BV = basis_order_for_supplied_str_eqns
                zeroing = {var: 0 for var in ordered_BV}
                new_data = dict()
                for idx_pair, val in data.items():
                    if not _scalar_is_zero(val):
                        v1, v2 = idx_pair
                        idx1 = ordered_BV.index(v1)
                        idx2 = ordered_BV.index(v2)
                        if hasattr(val, "subs") and _scalar_is_zero(val.subs(zeroing)):
                            coeffs = []
                            for var in ordered_BV:
                                coeffs.append(
                                    simplify(val.subs({var: 1}).subs(zeroing))
                                )
                            new_data[(idx2, idx1)] = tuple(coeffs)
                        else:
                            raise ValueError(
                                "If initializing an algebra from structure equations, supplied structure equations should be a dictionary whose keys are tuples of atomic variables and whose values are linear combinations of variables representing the product of the elements in the key tuple. If that is the case then you are likely getting this error because you did not supply the algebra creator with a valid value for the `basis_order_for_supplied_str_eqns` parameter. If that paremeter were omited, it is not always possible to unambiguously infer its proper value from general structure equations data, and hence this error arises."
                            )
                data = new_data
            if all(
                isinstance(key, tuple)
                and len(key) == 2
                and all(isinstance(idx, numbers.Integral) and idx >= 0 for idx in key)
                for key in data
            ):
                provided_index_bound = max(sum([list(key) for key in data.keys()], []))
            else:
                raise ValueError(
                    "Structure data must be have one of several formats: It can be a list/tuple with 3D shape of size (x, x, x). Or it can be a dictionairy of the (i,j) entries for the structure data. Set `process_matrix_rep=True` to initialize from a matrix representation, or provide a list of vector fields to initialize from a VF rep."
                )
            if all(isinstance(val, (tuple, list)) for val in data.values()):
                base_dims = list(len(val) for val in data.values())
                if len(set(base_dims)) != 1 or base_dims[0] < provided_index_bound + 1:
                    raise ValueError(
                        "If initializing an algebra algebra with structure data from a dictionairy, its keys should be (i,j) index tuples and its values should be tuples of coefficients from the product of i and j basis elements. All values tuples must have the same length in particular. Indices in the keys must not exceed the length of value tuples - 1 (as indexing starts from 0!)"
                    )
                else:
                    base_dim = base_dims[0]
                if assume_skew or assume_Lie_alg:
                    seen = []
                    initial_keys = list(data.keys())
                    for idx in initial_keys:
                        if idx in seen:
                            pass
                        else:
                            invert_idx = (idx[1], idx[0])
                            if invert_idx in data.keys():
                                if any(
                                    j + k != 0
                                    for j, k in zip(data[idx], data[invert_idx])
                                ):
                                    raise ValueError(
                                        "Either `assume_skew=True` or `assume_Lie_alg=True` was passed to the algebra contructor, but the accompanying structure data was not skew symmetric."
                                    )
                            else:
                                data[invert_idx] = [-j for j in data[idx]]
                            seen += [idx, invert_idx]
                data = [
                    [list(data.get((j, k), [0] * base_dim)) for j in range(base_dim)]
                    for k in range(base_dim)
                ]
            else:
                raise ValueError(
                    "If initializing an algebra algebra with structure data from a dictionairy, its keys should be (i,j) index tuples and its values should be tuples of coefficients from the product of i and j basis elements. All values tuples must have the same length in particular."
                )
        params = set()

        def _tuple_scan(elems, par: set):
            for elem in elems:
                par |= getattr(elem, "free_symbols", set())
            return tuple(elems)

        # Check that the data is a 3D list-like structure
        if (
            isinstance(data, (list, tuple))
            and len(data) > 0
            and isinstance(data[0], (list, tuple))
        ):
            if len(data) == len(data[0]) == len(data[0][0]):
                sd = tuple(
                    tuple(_tuple_scan(inner, params) for inner in outer)
                    for outer in data
                )
                return sd, params
            else:
                raise ValueError(
                    "Structure data must be a list with 3D shape of size (x, x, x). Or it can a  dictionairy of the (i,j) entries for the structure data. Set `process_matrix_rep=True` to initialize from a matrix representation, or provide a list of vector fields to initialize from a VF rep."
                )
        else:
            raise ValueError(
                "Structure data must be a list with 3D shape of size (x, x, x). Or it can a  dictionairy of the (i,j) entries for the structure data. Set `process_matrix_rep=True` to initialize from a matrix representation, or provide a list of vector fields to initialize from a VF rep."
            )
    except Exception as e:
        raise ValueError(f"Invalid structure data format: {type(data)} - {e}")


def algebraDataFromVF(
    vector_fields,
    *,
    samples=None,
    randomize_large=True,
    assume_basis=True,
    assume_consistent_coordinate_formatting=False,
    disable_sampling=False,
):
    vector_fields = list(vector_fields)
    dim0 = len(vector_fields)

    if dim0 == 0:
        return [], set()

    if not all(query_dgcv_categories(vf, {"vector_field"}) for vf in vector_fields):
        raise TypeError("algebraDataFromVF expects a list of vector fields.")

    vf_basis = vector_fields
    if not assume_basis:
        reduced = _extract_basis_over_number_field(vf_basis)
        if len(reduced) != len(vf_basis) and not get_dgcv_settings_registry().get(
            "forgo_warnings", False
        ):
            warnings.warn(
                "algebraDataFromVF: vector field list was not linearly independent; "
                "reduced to an extracted basis. Use assume_basis=True to bypass."
            )
        vf_basis = reduced

    dim = len(vf_basis)
    if dim == 0:
        return [], set()

    if samples is None:
        samples = max(2, dim)

    atoms = set()
    for vf in vf_basis:
        vs = getattr(vf, "_variable_spaces", None)
        if isinstance(vs, dict):
            for tup in vs.values():
                atoms.update(tup)

    atoms = tuple(sorted(atoms, key=lambda x: str(x)))
    nvars = len(atoms)

    if randomize_large:

        def _rand_rat():
            return rational(random.randint(1, 1000), random.randint(1001, 2000))

    def _make_subs_dicts():
        if nvars == 0:
            return [None] * samples

        if randomize_large:
            return [
                {atoms[i]: _rand_rat() for i in range(nvars)} for _ in range(samples)
            ]

        return [
            {atoms[i]: rational((i + 1) ** (s + 1), 32) for i in range(nvars)}
            for s in range(samples)
        ]

    subs_dicts = _make_subs_dicts()
    if disable_sampling:
        subs_dicts = [None]

    def _scalar_eqns_from_objects(eqns):
        out = []
        for e in eqns:
            if get_dgcv_category(e) == "tensor_field":
                out.extend(e.coeff_dict.values())
            else:
                out.append(e)
        return out

    def _local_unknowns(prefix, n):
        label = create_key(prefix=prefix)
        return [symbol(f"{label}{i}") for i in range(n)]

    tvars = _local_unknowns("T", dim)

    combi = None
    for t, vf in zip(tvars, vf_basis):
        term = t * vf
        combi = term if combi is None else (combi + term)

    def _orig_index_pairs():
        if vf_basis is vector_fields:
            return list(range(dim)), list(range(dim))

        kept_indices = []
        start = 0
        for kept in vf_basis:
            found = False
            for i in range(start, len(vector_fields)):
                if vector_fields[i] is kept:
                    kept_indices.append(i)
                    start = i + 1
                    found = True
                    break
            if not found:
                kept_indices.append(None)
        return kept_indices, kept_indices

    kept_idx_map, _ = _orig_index_pairs()

    def compute_bracket(j, k):
        residual = VF_bracket(vf_basis[j], vf_basis[k]) - combi
        coeffs = list(residual.coeff_dict.values())

        eqs = []
        for subs_d in subs_dicts:
            for c in coeffs:
                v = c if subs_d is None else subs(c, subs_d)
                if not assume_consistent_coordinate_formatting:
                    v = allToReal(v)
                eqs.append(v)

        eqs = _scalar_eqns_from_objects(eqs)
        sols = solve_dgcv(eqs, tvars, method="linsolve")
        if not sols:
            jj = kept_idx_map[j] if j < len(kept_idx_map) else j
            kk = kept_idx_map[k] if k < len(kept_idx_map) else k
            raise RuntimeError(
                f"algebraDataFromVF: not closed under Lie bracket at indices ({jj}, {kk})."
            )

        sol = sols[0]
        out = []
        local_params = set()
        for t in tvars:
            coeff = subs(t, sol)
            fs = set(get_free_symbols(coeff))
            if atoms:
                fs -= set(atoms)
            local_params |= fs
            out.append(coeff)

        return out, local_params

    structure_data = [[[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
    params = set()

    for j in range(dim):
        for k in range(j + 1, dim):
            c, p = compute_bracket(j, k)
            params |= p
            structure_data[j][k] = c
            structure_data[k][j] = [-elem for elem in c]

    return structure_data, params


def algebraDataFromMatRep(mat_list):
    """
    Create the structure data array for a Lie algebra from a list of matrices in *mat_list*.

    This function computes the Lie algebra structure constants from a matrix representation of a Lie algebra.
    The returned structure data can be used to initialize an algebra instance.

    Parameters
    ----------
    mat_list : list
        A list of square matrices of the same size representing the Lie algebra.

    Returns
    -------
    list
        A 3D list of lists of lists representing the Lie algebra structure data.

    Raises
    ------
    Exception
        If the matrices do not span a Lie algebra, or if the matrices are not square and of the same size.
    """
    if not isinstance(mat_list, (list, tuple)):
        raise Exception(
            "algorithm for extracting algebra data from matrices expects a list of square matrices."
        )

    mListLoc = []
    for j in mat_list:
        m = _as_matrix(j)
        if m is None:
            raise Exception(
                "algorithm for extracting algebra data from matrices expects a list of square matrices."
            )
        mListLoc.append(m)

    if not mListLoc:
        return ([], mat_list, set())

    shapeLoc = mListLoc[0].nrows
    indexRangeCap = len(mListLoc)

    if not all(m.shape == (shapeLoc, shapeLoc) for m in mListLoc):
        raise Exception(
            "algorithm for extracting algebra data from matrices expects a list of square matrices of the same size."
        )

    tempVarLabel = "T" + retrieve_public_key()
    vars = variableProcedure(
        tempVarLabel,
        indexRangeCap,
        return_created_object=True,
        _tempVar=retrieve_passkey(),
    )[0]

    combiMatLoc = matrix_dgcv.zeros(shapeLoc, shapeLoc)
    for j in range(indexRangeCap):
        combiMatLoc = combiMatLoc + vars[j] * mListLoc[j]

    params = set()

    def pairValue(j, k, par):
        mat = (mListLoc[j] @ mListLoc[k]) - (mListLoc[k] @ mListLoc[j]) - combiMatLoc

        bracketVals = list(set(mat._data.values()))
        if not bracketVals:
            return [0] * indexRangeCap
        if len(bracketVals) == 1 and _scalar_is_zero(bracketVals[0]):
            return [0] * indexRangeCap

        solLoc = list(solve_dgcv(bracketVals, vars))
        if len(solLoc) == 1:
            coeffs = []
            for var in vars:
                coeff = var.subs(solLoc[0])
                par |= get_free_symbols(coeff)
                coeffs.append(coeff)
            return coeffs

        clearVar(*listVar(temporary_only=True), report=False)
        raise Exception(
            f"Unable to determine if matrices are closed under commutators. "
            f"Problem matrices are in positions {j} and {k}."
        )

    structure_data = [
        [
            [0] * indexRangeCap if k <= j else pairValue(k, j, params)
            for j in range(indexRangeCap)
        ]
        for k in range(indexRangeCap)
    ]
    for k in range(indexRangeCap):
        for j in range(k + 1, indexRangeCap):
            structure_data[k][j] = [-entry for entry in structure_data[j][k]]

    clearVar(*listVar(temporary_only=True), report=False)

    return (structure_data, mat_list, params)


def algebraDataFromTensorRep(tensor_list):
    """
    Create the structure data array from a list of tensor products closed under the `_contraction_product` operator (see dgcv.tensorProduct documentation).

    Parameters
    ----------
    tensorProduct : list
        A list of tensorProduct instances

    Returns
    -------
    list
        A 3D array-like list of lists of lists representing the Lie algebra structure data.
    """

    tempVarLabel = "T" + create_key()
    dim = len(tensor_list)
    if dim == 0:
        return [[[]]], tensor_list, set()
    vars = variableProcedure(
        tempVarLabel, dim, return_created_object=True, _tempVar=retrieve_passkey()
    )[0]
    gen_elem = sum(
        [vars[j] * tensor_list[j] for j in range(1, dim)], vars[0] * tensor_list[0]
    )

    params = set()

    def computeBracket(j, k, par):
        if k < j:
            return [0] * dim
        product = (tensor_list[j] * tensor_list[k]) - gen_elem
        solutions = solve_dgcv(product, vars)
        if len(solutions) > 0:
            sol_values = solutions[0]
            coeffs = []
            for var in vars:
                coeff = var.subs(sol_values)
                par |= get_free_symbols(coeff)
                coeffs.append(coeff)
            return coeffs
        else:
            clearVar(*listVar(temporary_only=True), report=False)
            raise Exception(
                f"Contraction product of tensors at positions {j} and {k} are not in the given tensor list."
            )

    structure_data = [
        [[0 for _ in tensor_list] for _ in tensor_list] for _ in tensor_list
    ]

    for j in range(dim):
        for k in range(j):
            structure_data[k][j] = computeBracket(k, j, params)  # CHECK index order!!!
            structure_data[j][k] = [-elem for elem in structure_data[k][j]]

    clearVar(*listVar(temporary_only=True), report=False)

    return structure_data, tensor_list, params  # filter independants
