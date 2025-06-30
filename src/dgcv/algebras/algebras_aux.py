import random

import sympy as sp

from .._config import _cached_caller_globals, dgcv_exception_note
from .._safeguards import retrieve_passkey, retrieve_public_key
from ..dgcv_core import VF_bracket, VFClass, addVF, allToReal, variableProcedure
from ..solvers import solve_dgcv
from ..vmf import clearVar, listVar


def _validate_structure_data(data, process_matrix_rep=False, assume_skew=False, assume_Lie_alg=False, basis_order_for_supplied_str_eqns = None):
    if process_matrix_rep:
        if all(
            isinstance(sp.Matrix(obj), sp.Matrix)
            and len(set(sp.Matrix(obj).shape[:2]))<2
            for obj in data
        ):
            try:
                return algebraDataFromMatRep(data)
            except Exception as e:
                raise dgcv_exception_note(f'{e}') from None
        else:
            raise ValueError(
                f"matrix representation prcessing requires a list of square matrices. Recieved: {data}"
            )

    if isinstance(data,(list,tuple)) and all(isinstance(obj, VFClass) for obj in data):
        return algebraDataFromVF(data)
    try:
        if isinstance(data, dict):
            if all(isinstance(key,tuple) and len(key)==2 and all(isinstance(idx,sp.Symbol) for idx in key) for key in data):
                if basis_order_for_supplied_str_eqns is None:
                    basis_order_for_supplied_str_eqns = []
                if not isinstance(basis_order_for_supplied_str_eqns,(list,tuple)) or any(not isinstance(var,sp.Symbol) for var in basis_order_for_supplied_str_eqns):
                    raise ValueError('If initializing an algebra from structure equations and supplying the `basis_order_for_supplied_str_eqns` parameter, this parameter should be a list of the sympy.Symbols instances appearing in the supplied structure equations.')
                for var in set(sum([list(key) for key in data.keys()],[])):
                    if var not in basis_order_for_supplied_str_eqns:
                        raise ValueError('If initializing an algebra from structure equations and supplying the `basis_order_for_supplied_str_eqns` parameter, this parameter should be a list containing all sympy.Symbols instances appearing in the supplied structure equations.')
                ordered_BV = basis_order_for_supplied_str_eqns
                zeroing = {var:0 for var in ordered_BV}
                new_data = dict()
                for idx_pair, val in data.items():
                    v1,v2 = idx_pair
                    idx1 = ordered_BV.index(v1)
                    idx2 = ordered_BV.index(v2)
                    if hasattr(val,'subs') and val.subs(zeroing)==0:
                        coeffs = []
                        for var in ordered_BV:
                            coeffs.append(sp.simplify(val.subs({var:1}).subs(zeroing)))
                        new_data[(idx2,idx1)]=tuple(coeffs)
                    else:
                        print(val,zeroing)
                        raise ValueError('If initializing an algebra from structure equations, supplied structure equations should be a dictionary whose keys are tuples of variables (`sympy.Symbol` class instances) and whose value is a linear combination of variables representing the product of the elements in the key tuple.')
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

        # Check that the data is a 3D list-like structure
        if isinstance(data, (list,tuple)) and len(data) > 0 and isinstance(data[0], (list,tuple)):
            if len(data) == len(data[0]) == len(data[0][0]):
                return tuple(tuple(tuple(inner) for inner in outer) for outer in data)  # Return as a validated 3D list as tuples
            else:
                raise ValueError("Structure data must be a list with 3D shape of size (x, x, x). Or it can a  dictionairy of the (i,j) entries for the structure data. Set `process_matrix_rep=True` to initialize from a matrix representation, or provide a list of vector fields to initialize from a VF rep.")
        else:
            raise ValueError("Structure data must be a list with 3D shape of size (x, x, x). Or it can a  dictionairy of the (i,j) entries for the structure data. Set `process_matrix_rep=True` to initialize from a matrix representation, or provide a list of vector fields to initialize from a VF rep.")
    except Exception as e:
        raise ValueError(f"Invalid structure data format: {type(data)} - {e}")

def algebraDataFromVF(vector_fields):
    """
    Create the structure data array for a Lie algebra from a list of vector fields in *vector_fields*.

    This function computes the Lie algebra structure constants from a list of vector fields
    (instances of VFClass) defined on the same variable space. The returned structure data
    can be used to initialize an algebra instance.

    Parameters
    ----------
    vector_fields : list
        A list of VFClass instances, all defined on the same variable space with respect to the same basis.

    Returns
    -------
    list
        A 3D array-like list of lists of lists representing the Lie algebra structure data.

    Raises
    ------
    Exception
        If the vector fields do not span a Lie algebra or are not defined on a common basis.

    Notes
    -----
    This function dynamically chooses its approach to solve for the structure constants:
    - For smaller dimensional algebras, it substitutes pseudo-arbitrary values for the variables in `varSpaceLoc`
      based on a power function to create a system of linear equations.
    - For larger systems, where `len(varSpaceLoc)` raised to `len(vector_fields)` exceeds a threshold (default is 10,000),
      random rational numbers are used for substitution to avoid performance issues caused by large numbers.
    """
    # Define the product threshold for switching to random sampling
    product_threshold = 1

    # Check if all vector fields are defined on the same variable space
    if len(set([vf.varSpace for vf in vector_fields])) != 1:
        raise Exception(
            "algebraDataFromVF requires vector fields defined with respect to a common basis."
        )

    complexHandling = any(vf.dgcvType == "complex" for vf in vector_fields)
    if complexHandling:
        vector_fields = [allToReal(j) for j in vector_fields]
    varSpaceLoc = vector_fields[0].varSpace

    # Create temporary variables for solving structure constants
    tempVarLabel = "T" + retrieve_public_key()
    variableProcedure(tempVarLabel, len(vector_fields), _tempVar=retrieve_passkey())
    combiVFLoc = addVF(*[_cached_caller_globals[tempVarLabel][j] * vector_fields[j] for j in range(len(_cached_caller_globals[tempVarLabel]))])

    def computeBracket(j, k):
        """
        Compute and return the Lie bracket [vf_j, vf_k] and structure constants.

        Parameters
        ----------
        j : int
            Index of the first vector field.
        k : int
            Index of the second vector field.

        Returns
        -------
        list
            Structure constants for the Lie bracket of vf_j and vf_k.
        """
        if k <= j:
            return [0] * len(_cached_caller_globals[tempVarLabel])

        # Compute the Lie bracket
        bracket = VF_bracket(vector_fields[j], vector_fields[k]) - combiVFLoc

        if complexHandling:
            bracket = [allToReal(expr) for expr in bracket.coeffs]
        else:
            bracket = bracket.coeffs

        if len(varSpaceLoc) ** len(vector_fields) <= product_threshold:
            # Use the current system of pseudo-arbitrary substitutions
            bracketVals = list(
                set(
                    sum(
                        [
                            [
                                expr.subs(
                                    [
                                        (
                                            varSpaceLoc[i],
                                            sp.Rational((i + 1) ** sampling_index, 32),
                                        )
                                        for i in range(len(varSpaceLoc))
                                    ]
                                )
                                for expr in bracket
                            ]
                            for sampling_index in range(len(vector_fields))
                        ],
                        [],
                    )
                )
            )
        else:
            # Use random sampling system for larger cases
            def random_rational():
                return sp.Rational(random.randint(1, 1000), random.randint(1001, 2000))            
            bracketVals = list(
                set(
                    sum(
                        [
                            [
                                expr if not hasattr(expr,'subs') else
                                expr.subs(
                                    [
                                        (varSpaceLoc[i], random_rational())
                                        for i in range(len(varSpaceLoc))
                                    ]
                                )
                                for expr in bracket
                            ]
                            for _ in range(len(vector_fields))
                        ],
                        [],
                    )
                )
            )

        solutions = list(solve_dgcv(bracketVals, _cached_caller_globals[tempVarLabel]))
        if len(solutions) == 1:
            sol_values = solutions[0]
            substituted_constants = [
                expr.subs(sol_values)
                for expr in _cached_caller_globals[tempVarLabel]
            ]
            return substituted_constants
        else:
            raise Exception(
                f"Fields at positions {j} and {k} are not closed under Lie brackets."
            )

    # Precompute all necessary Lie brackets and store as 3D list
    structure_data = [[[0 for _ in vector_fields] for _ in vector_fields] for _ in vector_fields]

    for j in range(len(vector_fields)):
        for k in range(j + 1, len(vector_fields)):
            structure_data[j][k] = computeBracket(k, j)         # CHECK index order!!!
            structure_data[k][j] = [-elem for elem in structure_data[j][k]]

    # Clean up temporary variables
    clearVar(*listVar(temporary_only=True), report=False)

    return structure_data

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
    if isinstance(mat_list, list):
        mListLoc = [sp.Matrix(j) for j in mat_list]
        shapeLoc = mListLoc[0].shape[0]
        indexRangeCap=len(mat_list)

        # Check that all matrices are square and of the same size
        if all(j.shape == (shapeLoc, shapeLoc) for j in mListLoc):
            tempVarLabel = "T" + retrieve_public_key()
            variableProcedure(tempVarLabel, indexRangeCap, _tempVar=retrieve_passkey())
            combiMatLoc = sum([_cached_caller_globals[tempVarLabel][j] * mListLoc[j] for j in range(indexRangeCap)],sp.zeros(shapeLoc, shapeLoc))
            def pairValue(j, k):
                """
                Compute the commutator [m_j, m_k] and match with the combination matrix.

                Parameters
                ----------
                j : int
                    Index of the first matrix in the commutator.
                k : int
                    Index of the second matrix in the commutator.

                Returns
                -------
                list
                    The coefficients representing the structure constants.
                """
                mat = (mListLoc[j] * mListLoc[k] - mListLoc[k] * mListLoc[j] - combiMatLoc)
                bracketVals = list(set([*mat]))
                if len(bracketVals)==1 and bracketVals[0]==0:
                    return [0]*indexRangeCap
                solLoc = list(solve_dgcv(bracketVals, _cached_caller_globals[tempVarLabel]))

                if len(solLoc) == 1:
                    result = [var.subs(solLoc[0]) for var in _cached_caller_globals[tempVarLabel]]
                    return result
                else:
                    clearVar(*listVar(temporary_only=True),report=False)
                    raise Exception(
                        f"Unable to determine if matrices are closed under commutators. "
                        f"Problem matrices are in positions {j} and {k}."
                    )

            structure_data = [[[0]*indexRangeCap if k<=j else pairValue(k, j) for j in range(indexRangeCap)] for k in range(indexRangeCap)]
            for k in range(indexRangeCap):
                for j in range(k+1,indexRangeCap):
                    structure_data[k][j]=[-entry for entry in structure_data[j][k]]

            clearVar(*listVar(temporary_only=True), report=False)

            return structure_data
        else:
            raise Exception("algorithm for extracting algebra data from matrices expects a list of square matrices of the same size.")
    else:
        raise Exception("algorithm for extracting algebra data from matrices expects a list of square matrices.")

