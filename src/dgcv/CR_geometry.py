"""
DGCV: Differential Geometry with Complex Variables

This module provides tools specific to CR (Cauchy-Riemann) geometry within the DGCV package. 
It includes functions for constructing CR hypersurfaces and  computing symmetries.

Key Functions:
    - tangencyObstruction(): Computes the tangency obstruction for a holomorphic vector field's 
      real part to be tangent to a CR hypersurface.
    - weightedHomogeneousVF(): Creates a general weighted homogeneous vector field in a 
      specified coordinate space.
    - findWeightedCRSymmetries(): Attempts to find infinitesimal symmetries of a weighted CR 
      hypersurface.
    - model2Nondegenerate(): Builds the defining equation for a 2-nondegenerate model 
      hypersurface.

Author: David Sykes (https://github.com/YikesItsSykes)

Dependencies:
    - sympy

License:
    MIT License
"""

############## dependencies
############## CR geometry
from functools import reduce

import sympy as sp

from ._safeguards import create_key, retrieve_passkey
from .config import _cached_caller_globals, get_variable_registry
from .dgcv_core import (
    VFClass,
    addVF,
    allToReal,
    clearVar,
    holToReal,
    holVF_coeffs,
    listVar,
    realPartOfVF,
    scaleVF,
    symToReal,
    variableProcedure,
)
from .polynomials import createPolynomial
from .solvers import solve_DGCV
from .vector_fields_and_differential_forms import assembleFromHolVFC


def tangencyObstruction(vf, CR_defining_expr, graph_variable, simplify=False, *args):
    """
    Computes the tangency obstruction for a holomorphic vector field with respect to a CR hypersurface.

    This function computes the Lie derivative of the real part of a holomorphic vector field applied
    to the defining equation of a CR hypersurface and then substitutes the defining equation into the result.
    The output is zero (up to simplification) if and only if the vector field is a symmetry of the hypersurface.

    Parameters:
    -----------
    vf : VFClass
        A holomorphic vector field in the complex coordinate 
    CR_defining_expr : sympy expression
        A defining function of the CR hypersurface, expressed in holomorphic or real variables.
    graph_variable : sympy symbol
        The real variable whose value is set equal to the defining function to define the hypersurface.
    simplify : bool, optional
        If True, applies sp.simplify to the final result.
    *args:
        Additional arguments (unused).

    Returns:
    --------
    sympy expression
        The tangency obstruction.

    Raises:
    -------
    TypeError
        If the first argument is not a VFClass instance with DGCVType='complex'.
    """
    if not (isinstance(vf, VFClass) and vf.DGCVType == "complex"):
        raise TypeError(
            "`tangencyObstruction` requires its first argument to be a VFClass instance with DGCVType='complex'"
        )
    vf_real = allToReal(vf)
    real_eval = realPartOfVF(vf_real)(holToReal(graph_variable - CR_defining_expr))
    eval_real = allToReal(real_eval)
    substituted_expr = eval_real.subs(holToReal(symToReal(graph_variable)), symToReal(CR_defining_expr))
    result_expr = symToReal(substituted_expr)
    if simplify:
        result_expr = sp.simplify(result_expr)
    return result_expr


def weightedHomogeneousVF(
    arg1, arg2, arg3, arg4, degreeCap=0, _tempVar=None, assumeReal=None
):
    """
    Creates a weighted homogeneous vector field in a given coordinate space.

    This function generates a general weighted homogeneous vector field in the space of variables provided
    in *arg1*, with weights specified in *arg3*. The polynomial degree of variables with zero weight can
    be bounded by *degreeCap*.

    Parameters:
    -----------
    arg1 : tuple or list
        A tuple or list of variables, initialized by *varWithVF* or *complexVarProc*.
    arg2 : int
        An integer specifying the weight of the vector field.
    arg3 : list of int
        A list of non-negative integer weights corresponding to the variables in *arg1*.
    arg4 : str
        A string label for the variables in the returned vector field.
    degreeCap : int, optional
        Maximum polynomial degree for zero-weight variables (default is 0).
    _tempVar : any, optional
        Internal key.
    assumeReal : bool, optional
        Whether to assume the variables are real (default is False).

    Returns:
    --------
    VFClass
        A weighted homogeneous vector field.

    Raises:
    -------
    NA
    """
    pListLoc = []
    for j in range(len(arg3)):
        pListLoc.append(
            createPolynomial(
                arg4 + "_" + str(j) + "_",
                arg2 + arg3[j],
                arg1,
                degreeCap=degreeCap,
                weightedHomogeneity=arg3,
                _tempVar=_tempVar,
                assumeReal=assumeReal,
            )
        )
    return reduce(
        addVF,
        [
            scaleVF(pListLoc[j], eval("D_" + str(arg1[j]), _cached_caller_globals))
            for j in range(len(arg1))
        ],
    )


def findWeightedCRSymmetries_old(
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    degreeCap=0,
    returnVectorFieldBasis=False,
    applyNumer=False,
    simplifyingFactor=None,
    simplify=False
):
    """
    ***This function's algorithm will be revised in future updates***

    Attempts to find all infinitesimal symmetries of a rigid CR hypersurface given by setting one variable *arg5* equal to a defining function *arg1* in the variable space *arg2* with weighted homogeneity *arg3* w.r.t. to non-negative integer weights in *arg4*. Variables in the returned vector field's coefficients are labeled by *arg6*.

    Only polynomial vector fields are searched for, so if a variable is assigned weight zero, the function cannot search across general symmetries. In such cases, it rather searches all possible symmetries with polynomial degree in the zero-weighted variables up to the specified bound *degreeCap*. If *degreeCap* is not specified, then it defaults to zero.

    The algorithm is most succesful when the function *arg1* must be a polynomial.

    The function *arg1* should not depend on the variable *arg5* (i.e., the algorithm is not intended for implicit defining equations). If there is such dependence *findWeightedCRSymmetries* may still find some but not all symmetries.


    Args:
        arg1: Defining function of a rigid CR hypersurface.
        arg2: a tuple or list of complex variables parameterizing the space that the above CR hypersurface is defined in (not including the transverse direction symmetry).
        arg3: list of non-negative integer weights corresponding to the variables in *arg1* (must have the same length as *arg1*). If 0 is among the weights, then then proceedure will only test polynomial vector fields with polynomial degree in the weight zero variables up to the weight specified in *degreeCap*. By default, degreeCap=0, and can be set to any positive integer.
        arg4: int
        arg5: The real variable that when set equal to the defining function defines the CR hypersurface.
        arg6: str
        degreeCap: (optional keyword) set this keyword argument equal to any positive integer. If not specified, it defaults to zero.
        applyNumer: (optional keyword) True or False. Set equal to true if defining equation is rational but not polynomial. It can help the internal solvers.

    Returns:
        coefficient list for a holomorphic vector field containing variables, and any set real value for these variables defines an infinitesimal symmetry. **Note, indeed only real values for the variables define actual symmetries**

    Raises:
        NA
    """
    def extractRIVar(arg1):
        return sum([list(holToReal(j).atoms(sp.Symbol)) for j in arg1], [])

    VFLoc = addVF(
        weightedHomogeneousVF(
            arg2,
            arg4,
            arg3,
            "ALoc",
            _tempVar=retrieve_passkey(),
            degreeCap=degreeCap,
            assumeReal=True,
        ),
        scaleVF(sp.I,weightedHomogeneousVF(
                arg2,
                arg4,
                arg3,
                "BLoc",
                _tempVar=retrieve_passkey(),
                degreeCap=degreeCap,
                assumeReal=True,
            ),
        ),
    )
    tanObst = tangencyObstruction(VFLoc, arg1, arg5, simplify=simplify)
    varLoc = tanObst.atoms(sp.Symbol)
    varLoc1 = {j for j in varLoc}
    varComp = set(extractRIVar(arg2))
    varLoc.difference_update(varComp)
    varLoc1.difference_update(varLoc)
    varComp.difference_update(varLoc1)
    variableProcedure(arg6, len(varLoc), assumeReal=True)
    if applyNumer:
        if varLoc1 == set():
            varLoc1 = set(arg2)
        coefList = sp.poly_from_expr(sp.expand(sp.numer(tanObst)), *varLoc1)[0].coeffs()
        solutions = sp.solve(coefList, varLoc)
    elif simplifyingFactor is None:
        if varLoc1 == set():
            varLoc1 = set(arg2)
        coefList = sp.poly_from_expr(sp.expand(tanObst), *varLoc1)[0].coeffs()
        solutions = sp.solve(coefList, varLoc)
    else:
        if varLoc1 == set():
            varLoc1 = set(arg2)

        coefList = sp.poly_from_expr(
            sp.expand(simplify(symToReal(simplifyingFactor) * tanObst)), *varLoc1
        )[0].coeffs()
        solutions = sp.solve(coefList, varLoc)
    if len(solutions) == 0:
        if tanObst!=0:
            clearVar(*listVar(temporary_only=True), report=False)
            raise ValueError(f"no solution to this system: {coefList}")
        else:
            solutions=dict()
    if type(solutions) is dict:
        VFCLoc = [j.subs(solutions) for j in holVF_coeffs(VFLoc, arg2)]
        subVar = sum(VFCLoc).atoms(sp.Symbol)
        subVar.difference_update(set(arg2))
        variableProcedure(arg6, len(subVar), assumeReal=True)
        VFCLoc = [
            j.subs(dict(zip(subVar, eval(arg6, _cached_caller_globals))))
            for j in VFCLoc
        ]
        clearVar(*listVar(temporary_only=True), report=False)
        if returnVectorFieldBasis:
            VFListLoc = []
            for j in eval(arg6, _cached_caller_globals):
                VFCLocTemp = [
                    k.subs(j, 1).subs(
                        [(ll, 0) for ll in eval(arg6, _cached_caller_globals)]
                    )
                    for k in VFCLoc
                ]
                VFListLoc.append(assembleFromHolVFC(VFCLocTemp, arg2))
            clearVar(arg6, report=False)
            return VFListLoc, VFCLoc
        else:
            return VFCLoc
    else:
        VFCLoc = holVF_coeffs(VFLoc, arg2)
        subVar = sum(VFCLoc).atoms(sp.Symbol)
        subVar.difference_update(set(arg2))
        variableProcedure(arg6, len(subVar), assumeReal=True)
        VFCLoc = [
            j.subs(dict(zip(subVar, eval(arg6, _cached_caller_globals))))
            for j in VFCLoc
        ]
        clearVar(*listVar(temporary_only=True), report=False)
        return VFCLoc, solutions


def findWeightedCRSymmetries(
    graph_function,
    holomorphic_coordinates,
    coordinate_weights,
    symmetry_weight,
    graph_variable,
    coeff_label = None,
    degreeCap = 0,
    returnVectorFieldBasis = False,
    returnAllformats = False,
    simplifyingFactor = None,
    assume_polynomial = False,
    simplify=False
):
    """
    """
    if returnAllformats:
        returnVectorFieldBasis = True
    vr = get_variable_registry()
    cd = vr['conversion_dictionaries']
    def extractRIVar(coordinate_list):
        return set(sum([[cd['real_part'].get(var,var),cd['im_part'].get(var,var)] for var in coordinate_list], []))
    VFLoc = addVF(
        weightedHomogeneousVF(
            holomorphic_coordinates,
            symmetry_weight,
            coordinate_weights,
            "ALoc",
            _tempVar=retrieve_passkey(),
            degreeCap=degreeCap,
            assumeReal=True,
        ),
        scaleVF(sp.I,weightedHomogeneousVF(
                holomorphic_coordinates,
                symmetry_weight,
                coordinate_weights,
                "BLoc",
                _tempVar=retrieve_passkey(),
                degreeCap=degreeCap,
                assumeReal=True,
            ),
        ),
    )
    tanObst = tangencyObstruction(VFLoc, graph_function, graph_variable, simplify=simplify)
    varLoc = tanObst.atoms(sp.Symbol)
    varLoc1 = varLoc.copy()
    varComp = extractRIVar(holomorphic_coordinates)
    varLoc.difference_update(varComp)
    varLoc1.difference_update(varLoc)
    varComp.difference_update(varLoc1)
    if coeff_label is None:
        coeff_label = create_key(prefix='X',key_length=5)
    variableProcedure(coeff_label, len(varLoc), assumeReal=True)
    if simplifyingFactor is not None:
        tanObst = sp.simplify(symToReal(simplifyingFactor)*tanObst)
    if not assume_polynomial:
        if varLoc1 == set():
            varLoc1 = set(holomorphic_coordinates)
        coefList = sp.poly_from_expr(sp.expand(sp.numer(tanObst)), *varLoc1)[0].coeffs()
        solutions = solve_DGCV(coefList, varLoc, method="auto")
    else:
        if varLoc1 == set():
            varLoc1 = set(holomorphic_coordinates)

        coefList = sp.poly_from_expr(
            sp.expand(tanObst), *varLoc1
        )[0].coeffs()
        solutions = solve_DGCV(coefList, varLoc, method="auto")
    if len(solutions) == 0:
        if tanObst!=0:
            clearVar(*listVar(temporary_only=True), report=False)
            raise ValueError(f"no solution to this system: {coefList}")
        else:
            solutions=dict()
            VFCLoc = [0 for _ in holomorphic_coordinates]
    else:
        if isinstance(solutions,(list,tuple)):
            solutions = solutions[0]
        VFCLoc = [j.subs(solutions) for j in holVF_coeffs(VFLoc, holomorphic_coordinates)]

    subVar = set()
    for term in VFCLoc:
        subVar |= term.atoms(sp.Symbol)
    subVar.difference_update(set(holomorphic_coordinates))
    variableProcedure(coeff_label, len(subVar), assumeReal=True)
    coeff_vars = _cached_caller_globals[coeff_label]
    VFCLoc = [
        j.subs(dict(zip(subVar, coeff_vars)))
        for j in VFCLoc
    ]
    clearVar(*listVar(temporary_only=True), report=False)
    if returnVectorFieldBasis:
        VFListLoc = []
        for j in coeff_vars:
            VFCLocTemp = [
                k.subs(j, 1).subs(
                    [(ll, 0) for ll in coeff_vars]
                )
                for k in VFCLoc
            ]
            VFListLoc.append(assembleFromHolVFC(VFCLocTemp, holomorphic_coordinates))
        clearVar(coeff_label, report=False)
        if returnAllformats:
            return VFCLoc,VFListLoc
        return VFListLoc
    else:
        return VFCLoc




def model2Nondegenerate(arg1, arg2, arg3, arg4, return_matrices=False, simplify=True):
    """
    Builds the defining equation for a 2-nondegnerate model hypersurface using the general formula from the arXiv preprint arXiv:2404.06525.

    Args:
        arg1: nondegenerate s-by-s hermitian matrix
        arg2: s-by-s symmetric matrix valued function of some complex variables whose differential at zero is injective, and whose value at zero is zero.
        arg3: a length s tuple of complex variables, different from those appearing in *arg2*
        arg4: a single complex variable, different from those appearing in *arg2* and *arg3*

    Returns:
        A sympy expression. Setting this expression equal to zero defines the 2-nondegenerate model.

    Raises:
        NA
    """

    def format_mat(mat):
        if isinstance(mat,(tuple,list)):
            if all(isinstance(elem,(tuple,list)) for elem in mat):
                if len(set(len(elem) for elem in mat))==1:
                    mat = sp.Matrix(mat)
        if isinstance(mat,sp.Matrix):
            return mat
        else:
            raise TypeError('`model2Nondegenerate` expects first to arguments to be array-like data.')

    arg1 = format_mat(arg1)
    arg2 = format_mat(arg2)
    def simplify_rules(expr):
        if simplify:
            return sp.simplify(expr)
        else:
            return expr
    BARSLoc = sp.conjugate(arg2)
    zVecLoc = sp.Matrix(arg3)
    bzVecLoc = sp.Matrix([sp.conjugate(j) for j in arg3])
    sizeLoc = arg1.shape[0]
    hFun = (sp.Rational(1, 2)) * (
        (arg1 * (sp.eye(sizeLoc) - (BARSLoc * sp.Transpose(arg1) * arg2 * arg1)) ** (-1))
        + ((sp.eye(sizeLoc) - (arg1 * BARSLoc * sp.Transpose(arg1) * arg2)) ** (-1) * arg1)
    )
    sFun = (
        arg1
        * ((sp.eye(sizeLoc) - (BARSLoc * sp.Transpose(arg1) * arg2 * arg1)) ** (-1))
        * BARSLoc
        * sp.Transpose(arg1)
    )
    bsFun = (
        sp.Transpose(arg1)
        * ((sp.eye(sizeLoc) - (arg2 * arg1 * BARSLoc * sp.Transpose(arg1))) ** (-1))
        * arg2
        * arg1
    )
    if return_matrices:
        return (
            simplify_rules(
                (
                    sp.Transpose(zVecLoc) * hFun * bzVecLoc
                    + (sp.Rational(1, 2))
                    * (
                        sp.Transpose(zVecLoc) * sFun * zVecLoc
                        + sp.Transpose(bzVecLoc) * bsFun * bzVecLoc
                    )
                )[0]
            )
            - sp.im(arg4),
            hFun,
            sFun,
        )
    else:
        return simplify_rules(
            (
                sp.Transpose(zVecLoc) * hFun * bzVecLoc
                + (sp.Rational(1, 2))
                * (
                    sp.Transpose(zVecLoc) * sFun * zVecLoc
                    + sp.Transpose(bzVecLoc) * bsFun * bzVecLoc
                )
            )[0]
        ) - sp.im(arg4)
