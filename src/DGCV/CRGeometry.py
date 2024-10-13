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
    Computes the Lie derivative of a holomorphic vector field's real part applied to a graph defining a CR hypersurface, then substitutes the defining equation into the result.

    Returns 0 (up to simplification) if and only if the vector field is a symmetry of the hypersurface.

    Arguments must be expressions w.r.t. a complex coordinate system initialized by *complexVarProc*

    Args:
        arg1: VFClass class instance in the form of a holomorphic vector field (for example, a VF initialized by *assembleFromHolVFC*)
        arg2: Defining function of CR hypersurface in holomorphic or real variables format (i.e., a real-valued function of the complex parameter space depending on some but not all of the real and imaginaray parts of the complex variables)
        arg3: The real variable that when set equal to the defining function defines the CR hypersurface.
    
    Returns:
        A sympy expression.
    
    Raises:
        NA
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
    Creates a general weighted homogeneous vector field (i.e., VFClass instance) in the coordinate space of variables in *arg1* of weight *arg2* w.r.t. weights in *arg3*. Variables for the coefficients in the vector field are created with the label *arg4*.

    If variable weights assigned are nonzero then the returned vector field is truly general. Otherwise the returned vector field is the general one with polynomial degree in the zero-weight variables bounded by *degreeCap* which can be set to any integer. If *degreeCap* is not specified then it defaults to zero.

    Args:
        arg1: a tuple/list of variables initialized by either *varWithVF* or *complexVarProc*
        arg2: int
        arg3: list of non-negative integer weights corresponding to the variables in *arg1* (must have the same length as *arg1*). If 0 is among the weights, then then proceedure will only test polynomial vector fields with polynomial degree in the weight zero variables up to the weight specified in *degreeCap*. By default, degreeCap=0, and can be set to any positive integer.
        arg4: str
        _tempVar: (optional keyword) internal key
        assumeReal: (optional keyword) True or False
        degreeCap: (optional keyword) set this keyword argument equal to any positive integer. If not specified, it defaults to zero.

    
    Returns:
        A vector field (i.e., VFClass instance)
    
    Raises:
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

def model2Nondegenerate(arg1,arg2,arg3,arg4):
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
    return simplify((Transpose(zVecLoc)*hFun*bzVecLoc+(Rational(1,2))*(Transpose(zVecLoc)*sFun*zVecLoc+Transpose(bzVecLoc)*bsFun*bzVecLoc))[0])-im(arg4)

