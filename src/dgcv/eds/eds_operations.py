
from .._safeguards import _cached_caller_globals, create_key, retrieve_passkey
from ..dgcv_core import variableProcedure
from ..vmf import clearVar, listVar
from .eds import abst_coframe, abstDFAtom, abstDFMonom, abstract_DF, extDer


def transform_coframe(original_coframe:abst_coframe, transformations:list|tuple|dict, new_coframe_basis = None, new_coframe_labels = None, min_conj_rules = {}):
    original_basis = original_coframe.forms
    if new_coframe_basis is None:
        if isinstance(new_coframe_labels, (list,tuple)) and all(isinstance(term,str) for term in new_coframe_labels) and len(new_coframe_labels) == len(original_basis):
            new_coframe_basis = [abstDFAtom(1,1,label=term) for term in new_coframe_labels]
        else:
            raise ValueError('`transform_coframe` must either be given a list of DF objects for `new_coframe_basis` or a list of string labels for `new_coframe_labels` from which to create a new basis.')
    elif len(new_coframe_basis)!=len(original_basis):
        raise ValueError(f'If provided, the `new_coframe_basis` list given to `transform_coframe` must match the original coframe\'s basis size. In this case the list had length {len(new_coframe_basis)} while {len(original_basis)} was expected.')

    if isinstance (transformations,(list,tuple)) and all(isinstance(term,(list,tuple)) for term in transformations) and all(len(term)==2 for term in transformations):
        transformations = {term[0]:term[1] for term in transformations}
    if isinstance(transformations,dict):
        new_basis = tuple([df.subs(transformations) for df in original_basis])
    elif isinstance(transformations,(list,tuple)) and all(isinstance(term,(abstract_DF,abstDFMonom,abstDFAtom)) for term in transformations) and len(transformations)==len(original_basis) and min_conj_rules=={}:
        new_basis = transformations
    else:
        if len(min_conj_rules) == {}:
            raise TypeError(f'transformation rules given to `transform_coframe` must be either a dictionary (or dict-like list of pairs) whose keys belong to the old coframe basis and whose values are the new DF they transform to, or a list of new DF with proper length expressed in terms of the old basis: not {transformations}')
        else:
            raise TypeError(f'If providing `min_conj_rules`, transformation rules given to `transform_coframe` must be a dictionary (or dict-like list of pairs) whose keys belong to the old coframe basis and whose values are the new DF they transform to expressed in terms of the old basis: not {transformations}')
    new_basis_squares = []
    new_basis_squares_labels = []
    for j, el1 in enumerate(new_basis):
        el1Alt = new_coframe_basis[j]
        for k, el2 in enumerate(new_basis[j+1:]):
            el2Alt = new_coframe_basis[j+1:][k]
            new_basis_squares += [el1*el2]
            new_basis_squares_labels += [el1Alt*el2Alt]
    varLabel=create_key(prefix="coeff")             # label for temparary variables
    variableProcedure(
        varLabel,
        len(new_basis_squares),
        _tempVar=retrieve_passkey()
    )
    cVars = _cached_caller_globals[varLabel]        # pointer to tuple of coef vars

    general_elem = sum([j*k for j,k in zip(cVars[1:],new_basis_squares[1:])],cVars[0]*new_basis_squares[0])
    general_elem_new_basis = sum([j*k for j,k in zip(cVars[1:],new_basis_squares_labels[1:])],cVars[0]*new_basis_squares_labels[0])
    new_structure_eqns = dict()

    for df_atom, df in zip(new_coframe_basis,new_basis):
        from ..solvers import solve_dgcv
        new_extD = extDer(df,original_coframe)
        eqns = [new_extD - general_elem]
        solution = solve_dgcv(eqns,cVars)
        if len(solution)>1:
            raise ValueError('The given coframe transformation rule is not invertible')
        if len(solution)<1:
            raise ValueError('Unable to compute new coframe structure equations w.r.t. given transformation rule')
        new_structure_eqns |= {df_atom:general_elem_new_basis.subs(solution[0])}
    clearVar(*listVar(temporary_only=True), report=False)

    return abst_coframe(new_coframe_basis,new_structure_eqns,min_conj_rules=min_conj_rules)
