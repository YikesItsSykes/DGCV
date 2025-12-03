import random
import warnings

import sympy as sp

from .._config import _cached_caller_globals, dgcv_exception_note
from .._safeguards import (
    create_key,
    get_dgcv_category,
    retrieve_passkey,
    retrieve_public_key,
)
from ..backends._caches import _is_atomic
from ..dgcv_core import VF_bracket, VFClass, addVF, allToReal, variableProcedure
from ..solvers import solve_dgcv
from ..vmf import clearVar, listVar


def _validate_structure_data(data, process_matrix_rep=False, assume_skew=False, assume_Lie_alg=False, basis_order_for_supplied_str_eqns = None, process_tensor_rep = False):
    if process_tensor_rep:
        try:
            return algebraDataFromTensorRep(data),'tensor'
        except Exception as e:
            raise dgcv_exception_note(f'{e}') from None
    if process_matrix_rep:
        if all(isinstance(sp.Matrix(obj), sp.Matrix) and len(set(sp.Matrix(obj).shape[:2]))<2 for obj in data):
            try:
                return algebraDataFromMatRep(data), 'matrix'
            except Exception as e:
                raise dgcv_exception_note(f'{e}') from None
        elif all(get_dgcv_category(elem)=='tensorProduct' for elem in data):
            warnings.warn('`_validate_structure_data` was given a list of tensorProduct instance, but `process_matrix_rep` was also marked True. The latter was ignored.')
            return _validate_structure_data(data, process_matrix_rep=False, assume_skew=assume_skew, assume_Lie_alg=assume_Lie_alg, basis_order_for_supplied_str_eqns = basis_order_for_supplied_str_eqns, process_tensor_rep = True), 'tensor'
        else:
            raise ValueError(
                f"matrix representation prcessing requires a list of square matrices. Recieved: {data}"
            )

    if isinstance(data,(list,tuple)):
        if len(data)>0: 
            if all(isinstance(obj, VFClass) for obj in data):
                return algebraDataFromVF(data)
        else:
            return tuple(), set()
    try:
        if isinstance(data, dict):
            if all(isinstance(key,tuple) and len(key)==2 and all(_is_atomic(idx) for idx in key) for key in data):
                if basis_order_for_supplied_str_eqns is None:
                    build_basis_order = True
                    basis_order_for_supplied_str_eqns = []
                else:
                    build_basis_order = False
                if not isinstance(basis_order_for_supplied_str_eqns,(list,tuple)) or not all(_is_atomic(var) for var in basis_order_for_supplied_str_eqns):
                    raise ValueError('If initializing an algebra from structure equations and supplying the `basis_order_for_supplied_str_eqns` parameter, this parameter should be a list of the atomic variables (e.g., sympy.Symbol instances) appearing in the supplied structure equations.')
                for var in set(sum([list(key) for key in data.keys()],[])):
                    if var not in basis_order_for_supplied_str_eqns:
                        if build_basis_order:
                            basis_order_for_supplied_str_eqns.append(var)
                        else:
                            raise ValueError('If initializing an algebra from structure equations and supplying the `basis_order_for_supplied_str_eqns` parameter, this parameter should be a list containing all atomic variables (e.g., sympy.Symbol instances) appearing in the supplied structure equations.')
                ordered_BV = basis_order_for_supplied_str_eqns
                zeroing = {var:0 for var in ordered_BV}
                new_data = dict()
                for idx_pair, val in data.items():
                    if val!=0:
                        v1,v2 = idx_pair
                        idx1 = ordered_BV.index(v1)
                        idx2 = ordered_BV.index(v2)
                        if hasattr(val,'subs') and val.subs(zeroing)==0:
                            coeffs = []
                            for var in ordered_BV:
                                coeffs.append(sp.simplify(val.subs({var:1}).subs(zeroing)))
                            new_data[(idx2,idx1)]=tuple(coeffs)
                        else:
                            raise ValueError('If initializing an algebra from structure equations, supplied structure equations should be a dictionary whose keys are tuples of atomic variables (e.g., `sympy.Symbol` class instances) and whose value is a linear combination of variables representing the product of the elements in the key tuple. If that is the case then you are likely getting this error because you did not supply the algebra creator with a valid value for the `basis_order_for_supplied_str_eqns` parameter. If that paremeter were omited, it is not always possible to unambiguously infer its proper value from general structure equations data, and hence this error arises.')
                data = new_data
            if all(isinstance(key,tuple) and len(key)==2 and all(isinstance(idx,int) and idx>=0 for idx in key) for key in data):
                provided_index_bound = max(sum([list(key) for key in data.keys()],[]))
            else:
                raise ValueError("Structure data must be have one of several formats: It can be a list/tuple with 3D shape of size (x, x, x). Or it can be a dictionairy of the (i,j) entries for the structure data. Set `process_matrix_rep=True` to initialize from a matrix representation, or provide a list of vector fields to initialize from a VF rep.")
            if all(isinstance(val,(tuple,list)) for val in data.values()):
                base_dims = list(len(val) for val in data.values())
                if len(set(base_dims))!=1 or base_dims[0]<provided_index_bound+1:
                    raise ValueError("If initializing an algebra algebra with structure data from a dictionairy, its keys should be (i,j) index tuples and its values should be tuples of coefficients from the product of i and j basis elements. All values tuples must have the same length in particular. Indices in the keys must not exceed the length of value tuples - 1 (as indexing starts from 0!)")
                else:
                    base_dim = base_dims[0]
                if assume_skew or assume_Lie_alg:
                    seen = []
                    initial_keys = list(data.keys())
                    for idx in initial_keys:
                        if idx in seen:
                            pass
                        else:
                            invert_idx = (idx[1],idx[0])
                            if invert_idx in data.keys():
                                if any(j+k!=0 for j,k in zip(data[idx],data[invert_idx])):
                                    raise ValueError("Either `assume_skew=True` or `assume_Lie_alg=True` was passed to the algebra contructor, but the accompanying structure data was not skew symmetric.")
                            else:
                                data[invert_idx]=[-j for j in data[idx]]
                            seen+=[idx,invert_idx]
                data = [[list(data.get((j,k),[0]*base_dim)) for j in range(base_dim)] for k in range(base_dim)]
            else:
                raise ValueError("If initializing an algebra algebra with structure data from a dictionairy, its keys should be (i,j) index tuples and its values should be tuples of coefficients from the product of i and j basis elements. All values tuples must have the same length in particular.")
        params=set()
        def _tuple_scan(elems,par:set):
            for elem in elems:
                par|=getattr(elem,"free_symbols",set())
            return tuple(elems)
        # Check that the data is a 3D list-like structure
        if isinstance(data, (list,tuple)) and len(data) > 0 and isinstance(data[0], (list,tuple)):
            if len(data) == len(data[0]) == len(data[0][0]):
                sd=tuple(tuple(_tuple_scan(inner,params) for inner in outer) for outer in data)
                return sd, params
            else:
                raise ValueError("Structure data must be a list with 3D shape of size (x, x, x). Or it can a  dictionairy of the (i,j) entries for the structure data. Set `process_matrix_rep=True` to initialize from a matrix representation, or provide a list of vector fields to initialize from a VF rep.")
        else:
            raise ValueError("Structure data must be a list with 3D shape of size (x, x, x). Or it can a  dictionairy of the (i,j) entries for the structure data. Set `process_matrix_rep=True` to initialize from a matrix representation, or provide a list of vector fields to initialize from a VF rep.")
    except Exception as e:
        raise ValueError(f"Invalid structure data format: {type(data)} - {e}")

def algebraDataFromVF(vector_fields):
    """
    Create the structure data array for a Lie algebra from a list of vector fields.

    Parameters
    ----------
    vector_fields : list
        A list of VFClass instances, all defined on the same variable space with respect to the same basis.

    Returns
    -------
    list
        A 3D array-like list of lists of lists representing the Lie algebra structure data.

    Notes
    -----
    This function dynamically chooses its approach to solve for the structure constants:
    - For smaller dimensional algebras, it substitutes pseudo-arbitrary values for the variables in `varSpaceLoc` to create a system of linear equations.
    - For larger systems, where `len(varSpaceLoc)` raised to `len(vector_fields)` exceeds a threshold, random rational numbers are used for substitution to minimize performance hits.
    """

    product_threshold = 1

    if len(set([vf.varSpace for vf in vector_fields])) != 1:
        raise Exception("algebraDataFromVF requires vector fields defined with respect to a common basis.")

    complexHandling = any(vf.dgcvType == "complex" for vf in vector_fields)
    if complexHandling:
        vector_fields = [allToReal(j) for j in vector_fields]
    varSpaceLoc = vector_fields[0].varSpace

    tempVarLabel = "T" + retrieve_public_key()
    dim=len(vector_fields)
    variableProcedure(tempVarLabel, dim, _tempVar=retrieve_passkey())
    combiVFLoc = addVF(*[_cached_caller_globals[tempVarLabel][j] * vector_fields[j] for j in range(len(_cached_caller_globals[tempVarLabel]))])
    params=set()
    def computeBracket(j, k, par):
        if k <= j:
            return [0] * dim, params

        bracket = VF_bracket(vector_fields[j], vector_fields[k]) - combiVFLoc

        if complexHandling:
            bracket = [allToReal(expr) for expr in bracket.coeffs]
        else:
            bracket = bracket.coeffs

        if len(varSpaceLoc) ** len(vector_fields) <= product_threshold:
            bracketVals = list(set(sum([[expr.subs([(varSpaceLoc[i],sp.Rational((i + 1) ** sampling_index, 32)) for i in range(len(varSpaceLoc))]) for expr in bracket] for sampling_index in range(len(vector_fields))],[])))
        else:
            # random sampling system for larger cases
            def random_rational():
                return sp.Rational(random.randint(1, 1000), random.randint(1001, 2000))            
            bracketVals = list(set(sum([[expr if not hasattr(expr,'subs') else
                                expr.subs([(varSpaceLoc[i], random_rational()) for i in range(len(varSpaceLoc))]) for expr in bracket] for _ in range(len(vector_fields))],[])))

        solutions = list(solve_dgcv(bracketVals, _cached_caller_globals[tempVarLabel]))
        if len(solutions) == 1:
            coeffs=[]
            for var in _cached_caller_globals[tempVarLabel]:
                coeff=var.subs(solutions[0])
                par|=getattr(coeff,"free_symbols",set())
                coeffs.append(coeff)
            return coeffs
        else:
            raise Exception(f"Fields at positions {j} and {k} are not closed under Lie brackets.")

    structure_data = [[[0 for _ in vector_fields] for _ in vector_fields] for _ in vector_fields]

    for j in range(len(vector_fields)):
        for k in range(j + 1, len(vector_fields)):
            structure_data[j][k] = computeBracket(j, k, params)         # CHECK index order!!!
            structure_data[k][j] = [-elem for elem in structure_data[j][k]]

    clearVar(*listVar(temporary_only=True), report=False)

    return structure_data,params

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
    if isinstance(mat_list, (list,tuple)):
        mListLoc = [sp.Matrix(j) for j in mat_list]
        shapeLoc = mListLoc[0].shape[0]
        indexRangeCap=len(mat_list)

        if all(j.shape == (shapeLoc, shapeLoc) for j in mListLoc):
            tempVarLabel = "T" + retrieve_public_key()
            vars=variableProcedure(tempVarLabel, indexRangeCap,return_created_object=True, _tempVar=retrieve_passkey())[0]
            combiMatLoc = sum([vars[j] * mListLoc[j] for j in range(indexRangeCap)],sp.zeros(shapeLoc, shapeLoc))
            params=set()
            def pairValue(j, k, par):
                """
                Compute the commutator [m_j, m_k] and match with the combination matrix.

                Returns
                -------
                list
                    The coefficients representing the structure constants.
                """
                mat = (mListLoc[j] * mListLoc[k] - mListLoc[k] * mListLoc[j] - combiMatLoc)
                bracketVals = list(set([*mat]))
                if len(bracketVals)==1 and bracketVals[0]==0:
                    return [0]*indexRangeCap
                solLoc = list(solve_dgcv(bracketVals, vars))

                if len(solLoc) == 1:
                    coeffs=[]
                    for var in vars:
                        coeff=var.subs(solLoc[0])
                        par|=getattr(coeff,"free_symbols",set())
                        coeffs.append(coeff)
                    return coeffs
                else:
                    clearVar(*listVar(temporary_only=True),report=False)
                    raise Exception(
                        f"Unable to determine if matrices are closed under commutators. "
                        f"Problem matrices are in positions {j} and {k}."
                    )

            structure_data = [[[0]*indexRangeCap if k<=j else pairValue(k, j, params) for j in range(indexRangeCap)] for k in range(indexRangeCap)]
            for k in range(indexRangeCap):
                for j in range(k+1,indexRangeCap):
                    structure_data[k][j]=[-entry for entry in structure_data[j][k]]

            clearVar(*listVar(temporary_only=True), report=False)

            return structure_data,mat_list,params ###!!! filter the mat_list for independence
        else:
            raise Exception("algorithm for extracting algebra data from matrices expects a list of square matrices of the same size.")
    else:
        raise Exception("algorithm for extracting algebra data from matrices expects a list of square matrices.")

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
    dim=len(tensor_list)
    if dim==0:
        return [[[]]],tensor_list,set()
    vars=variableProcedure(tempVarLabel, dim,return_created_object=True,_tempVar=retrieve_passkey())[0]
    gen_elem = sum([vars[j] * tensor_list[j] for j in range(1,dim)],vars[0]*tensor_list[0])

    params = set()
    def computeBracket(j, k, par):
        if k < j:
            return [0] * dim
        product = (tensor_list[j]*tensor_list[k]) - gen_elem
        solutions = solve_dgcv(product, vars)
        if len(solutions) > 0:
            sol_values = solutions[0]
            coeffs=[]
            for var in vars:
                coeff=var.subs(sol_values)
                par|=getattr(coeff,"free_symbols",set())
                coeffs.append(coeff)
            return coeffs
        else:
            clearVar(*listVar(temporary_only=True),report=False)
            raise Exception(f"Contraction product of tensors at positions {j} and {k} are not in the given tensor list.")

    structure_data = [[[0 for _ in tensor_list] for _ in tensor_list] for _ in tensor_list]

    for j in range(dim):
        for k in range(j):
            structure_data[k][j] = computeBracket(k, j, params)         # CHECK index order!!!
            structure_data[j][k] = [-elem for elem in structure_data[k][j]]

    clearVar(*listVar(temporary_only=True), report=False)

    return structure_data,tensor_list,params   # filter independants
