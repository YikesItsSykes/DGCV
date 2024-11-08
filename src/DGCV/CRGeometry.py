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
from sympy import eye, expand, Matrix, numer, poly_from_expr, solve, Transpose, Symbol
from functools import reduce
from .config import _cached_caller_globals
from .combinatorics import *
from .classesAndVariables import *
from .vectorFieldsAndDifferentialForms import *
from .polynomials import *
from ._safeguards import retrieve_passkey


############## CR geometry

def tangencyObstruction(arg1,arg2,arg3,*args):
    """
    Computes the tangency obstruction for a holomorphic vector field with respect to a CR hypersurface.
    
    This function computes the Lie derivative of the real part of a holomorphic vector field applied 
    to the defining equation of a CR hypersurface and then substitutes the defining equation into the result.
    The output is zero (up to simplification) if and only if the vector field is a symmetry of the hypersurface.
    
    Parameters:
    -----------
    arg1 : VFClass
        A holomorphic vector field in the complex coordinate system, initialized by *assembleFromHolVFC*.
    arg2 : sympy expression
        The defining function of the CR hypersurface, expressed in holomorphic or real variables.
    arg3 : sympy symbol
        The real variable whose value is set equal to the defining function to define the hypersurface.
    
    Returns:
    --------
    sympy expression
        The tangency obstruction.
    
    Raises:
    -------
    TypeError
        If the first argument is not a VFClass instance with DGCVType='complex'.
    """
    if isinstance(arg1,VFClass):
        if arg1.DGCVType == 'standard':
            raise TypeError('`tangencyObstruction` requires its first argument `vf` to be VFClass with vf.DGCVType=\'complex\'')
    else:
        raise TypeError('`tangencyObstruction` requires its first argument `vf` to be VFClass with vf.DGCVType=\'complex\'')
    arg1=allToReal(arg1)
    evaluationLoc=allToReal(realPartOfVF(arg1)(holToReal(arg3-arg2))).subs(holToReal(symToReal(arg3)),symToReal(arg2))
    return simplify(symToReal(evaluationLoc))

def weightedHomogeneousVF(arg1,arg2,arg3,arg4,degreeCap=0,_tempVar=None,assumeReal=None):
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
    pListLoc=[]
    for j in range(len(arg3)):
        pListLoc.append(createPolynomial(arg4+'_'+str(j)+'_',arg2+arg3[j],arg1,degreeCap=degreeCap,weightedHomogeneity=arg3,_tempVar=_tempVar,assumeReal=assumeReal))
    return reduce(addVF,[scaleVF(pListLoc[j],eval('D_'+str(arg1[j]),_cached_caller_globals)) for j in range(len(arg1))])
def findWeightedCRSymmetries(arg1,arg2,arg3,arg4,arg5,arg6,degreeCap=0,returnVectorFieldBasis=False,applyNumer=False,simplifyingFactor=None):
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
        return sum([list(holToReal(j).atoms(Symbol)) for j in arg1],[])
    VFLoc=addVF(weightedHomogeneousVF(arg2,arg4,arg3,'ALoc',_tempVar=retrieve_passkey(),degreeCap=degreeCap,assumeReal=True),scaleVF(I,weightedHomogeneousVF(arg2,arg4,arg3,'BLoc',_tempVar=retrieve_passkey(),degreeCap=degreeCap,assumeReal=True)))
    tOLoc=tangencyObstruction(VFLoc,arg1,arg5,arg2)
    varLoc=tOLoc.atoms(Symbol)
    varLoc1={j for j in varLoc}
    varComp=set(extractRIVar(arg2))
    varLoc.difference_update(varComp)
    varLoc1.difference_update(varLoc)
    varComp.difference_update(varLoc1)
    variableProcedure(arg6,len(varLoc),assumeReal=True)
    if applyNumer==True:
        if varLoc1==set():
            varLoc1=set(arg2)
        coefListLoc=poly_from_expr(expand(numer(tOLoc)),*varLoc1)[0].coeffs()
        solLoc=solve(coefListLoc,varLoc)
    elif simplifyingFactor==None:
        if varLoc1==set():
            varLoc1=set(arg2)
        coefListLoc=poly_from_expr(expand(tOLoc),*varLoc1)[0].coeffs()
        solLoc=solve(coefListLoc,varLoc)
    else:
        if varLoc1==set():
            varLoc1=set(arg2)
        coefListLoc=poly_from_expr(expand(simplify(symToReal(simplifyingFactor)*tOLoc)),*varLoc1)[0].coeffs()
        solLoc=solve(coefListLoc,varLoc)
    if solLoc==[]:
        clearVar(*listVar(temporary_only=True),report=False)
        return 'no solution'
    elif type(solLoc)==dict:
        VFCLoc=[j.subs(solLoc) for j in holVF_coeffs(VFLoc,arg2)]
        subVar=sum(VFCLoc).atoms(Symbol)
        subVar.difference_update(set(arg2))
        variableProcedure(arg6,len(subVar),assumeReal=True)
        VFCLoc=[j.subs(dict(zip(subVar,eval(arg6,_cached_caller_globals)))) for j in VFCLoc]
        clearVar(*listVar(temporary_only=True),report=False)
        if returnVectorFieldBasis==True:
            VFListLoc=[]
            for j in eval(arg6,_cached_caller_globals):
                VFCLocTemp=[k.subs(j,1).subs([(l,0) for l in eval(arg6,_cached_caller_globals)]) for k in VFCLoc]
                VFListLoc.append(assembleFromHolVFC(VFCLocTemp,arg2))
            clearVar(arg6,report=False)
            return VFListLoc, VFCLoc
        else:
            return VFCLoc
    else: 
        VFCLoc=holVF_coeffs(VFLoc,arg2)
        subVar=sum(VFCLoc).atoms(Symbol)
        subVar.difference_update(set(arg2))
        variableProcedure(arg6,len(subVar),assumeReal=True)
        VFCLoc=[j.subs(dict(zip(subVar,eval(arg6,_cached_caller_globals)))) for j in VFCLoc]
        clearVar(*listVar(temporary_only=True),report=False)
        return VFCLoc,solLoc

def model2Nondegenerate(arg1,arg2,arg3,arg4, return_matrices = False):
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
    BARSLoc=conjugate(arg2)
    zVecLoc=Matrix(arg3)
    bzVecLoc=Matrix([conjugate(j) for j in arg3])
    sizeLoc=arg1.shape[0]
    hFun=(Rational(1,2))*((arg1*(eye(sizeLoc)-(BARSLoc*Transpose(arg1)*arg2*arg1))**(-1))+((eye(sizeLoc)-(arg1*BARSLoc*Transpose(arg1)*arg2))**(-1)*arg1))
    sFun=arg1*((eye(sizeLoc)-(BARSLoc*Transpose(arg1)*arg2*arg1))**(-1))*BARSLoc*Transpose(arg1)
    bsFun=Transpose(arg1)*((eye(sizeLoc)-(arg2*arg1*BARSLoc*Transpose(arg1)))**(-1))*arg2*arg1
    if return_matrices == True:
        return simplify((Transpose(zVecLoc)*hFun*bzVecLoc+(Rational(1,2))*(Transpose(zVecLoc)*sFun*zVecLoc+Transpose(bzVecLoc)*bsFun*bzVecLoc))[0])-im(arg4),hFun,sFun
    else:
        return simplify((Transpose(zVecLoc)*hFun*bzVecLoc+(Rational(1,2))*(Transpose(zVecLoc)*sFun*zVecLoc+Transpose(bzVecLoc)*bsFun*bzVecLoc))[0])-im(arg4)


