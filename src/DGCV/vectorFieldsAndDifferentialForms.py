"""
DGCV: Differential Geometry with Complex Variables

This module provides tools for creating, manipulating, and decomposing vector fields and 
differential forms within the DGCV package. It includes methods for Cartan calculus like 
the exterior derivative and Lie derivative. There are some complex structure specific
functions as well, e.g., assembling holomorphic vector fields from holomorphic coefficients.

Key Functions:

Object Assembly:
    - get_VF(): Retrieves coordinate vector fields for the given coordinate variables.
    - get_DF(): Retrieves coordinate differential forms for the given coordinate variables.
    - assembleFromHolVFC(): Assembles a holomorphic vector field from holomorphic coefficients.
    - assembleFromAntiholVFC(): Assembles an antiholomorphic vector field from antiholomorphic 
    coefficients.
    - assembleFromCompVFC(): Assembles a complex vector field from both holomorphic and 
    antiholomorphic coefficients.

Differential Operators:
    - exteriorDerivative(): Computes the exterior derivative of a differential k-form.
    - interiorProduct(): Computes the interior product of a vector field with a differential k-form.
    - LieDerivative(): Computes the Lie derivative of a differential form or vector field with respect to another vector field.

Decompositions and Basis:
    - decompose(): Decomposes a vector field or differential form as a linear combination of a given basis of vector fields or differential forms.
    - get_coframe(): Constructs a coframe dual to a given list of vector fields.
    - annihilator(): Computes the annihilator (namely sub-bundle in TM/T^*M) of a list of differential forms or vector fields.

Author: David Sykes (https://github.com/YikesItsSykes)

Dependencies:
    - sympy

License:
    MIT License
"""


############## dependencies
import warnings
from .combinatorics import *
from .DGCV_display import LaTeX
from .config import get_variable_registry,_cached_caller_globals
from .classesAndVariables import DFClass, VFClass, symToReal, allToSym, addVF, addDF,  allToReal, VF_bracket, minimalVFDataDict, variableProcedure, _remove_complex_handling,clearVar,listVar,compressDGCVClass,conj_with_hol_coor,changeDFBasis, _format_complex_coordinates
from ._safeguards import create_key,retrieve_passkey
import sympy
from sympy import Basic, im, denom, Rational, re, I, diff, prod, linsolve, sympify, simplify, solve


############## retrieval
def get_VF(*coordinates):
    vr = get_variable_registry()
    VFList = []
    parentS = vr['standard_variable_systems'] 
    parentC = vr['complex_variable_systems']
    parents = parentS | parentC
    for j in coordinates:
        varStr=str(j)
        for parent in parentS | parentC:
            if parent in parentS:
                the_house = parentS[parent]['family_values']
            else:
                if isinstance(parentC[parent]['family_values'][0],tuple):
                    the_house = sum(parentC[parent]['family_values'],tuple())
                else:
                    the_house = parentC[parent]['family_values']
            if j in  the_house:
                if parents[parent]['differential_system']==True:
                    VFList += [parents[parent]['variable_relatives'][varStr]['VFClass']]
                else:
                    raise TypeError(f'`get_VF` recieved a variable {j} that was not initialized with coordinate vector fields registered in DGCV\'s variable management framework. Use functions like `createVariables(--,withVF=True)` and `createVariables(--,complex=True)` to initialize variables with VF and DF.')
    return VFList

def get_DF(*coordinates):
    vr = get_variable_registry()
    DFList = []
    parentS = vr['standard_variable_systems'] 
    parentC = vr['complex_variable_systems']
    parents = parentS | parentC
    for j in coordinates:
        varStr=str(j)
        for parent in parentS | parentC:
            if parent in parentS:
                the_house = parentS[parent]['family_values']
            else:
                if isinstance(parentC[parent]['family_values'][0],tuple):
                    the_house = sum(parentC[parent]['family_values'],tuple())
                else:
                    the_house = parentC[parent]['family_values']
            if j in  the_house:
                if parents[parent]['differential_system']==True:
                    DFList += [parents[parent]['variable_relatives'][varStr]['DFClass']]
                else:
                    raise TypeError('`get_DF` recieved variables that were not initialized with coordinate vector fields and dual 1-forms registered in DGCV\'s variable management framework. Use functions like `createVariables(--,withVF=True)` and `createVariables(--,complex=True)` to initialize variables with VF and DF.')
    return DFList
                

############## complex vector fields

def assembleFromHolVFC(arg1, arg2):
    """
    Constructs a holomorphic vector field (i.e., VFClass instance) with prescribed coefficients in chosen coordinates.

    The vector field is expressed as a linear combination of the coordinate holomorphic vector fields 
    corresponding to the variables in *arg2* with coefficients in *arg1*.

    Parameters:
    -----------
    arg1 : tuple of sympy expressions
        Coefficients for the vector field in terms of the holomorphic variables.
    arg2 : list or tuple
        A list or tuple containing Symbol objects that were initialized as holomorphic variables via DGCV variable creation functions.
    
    Returns:
    --------
    VFClass
        A holomorphic vector field expressed as a linear combination of coordinate vector fields.
    
    Raises:
    -------
    ValueError
        If the length of *arg1* does not match the number of holomorphic variables in *arg2*.
    Exception
        If variables in *arg2* were not initialized as holomorphic variables.

    Example:
    --------
    >>> from DGCV import createVariables, assembleFromHolVFC
    >>> createVariables('z', 'x', 'y', 3)
    >>> print(assembleFromHolVFC((z1 + z2, z3, 0), [z1, z2, z3]))
    (z1/2 + z2/2)*D_x1+z3/2*D_x2+(-I*(z1 + z2)/2)*D_y1-I*z3/2*D_y2
    """
    # Ensure the length of coefficients matches the variable space
    if len(arg1) != len(arg2):
        raise ValueError("The number of coefficients in arg1 must match the number of variables in arg2.")
    vr = get_variable_registry()
    if all(var in vr['conversion_dictionaries']['symToReal'] for var in arg2):
        pass
    else:
        raise TypeError('`assembleFromHolVFC` expects the provided varibles for the coordinate space to be holomorphic.')
    
    # Return the resulting vector field as a VFClass instance
    return VFClass(arg2, arg1, 'complex')

def assembleFromAntiholVFC(arg1, arg2):
    """
    Constructs an antiholomorphic vector field (i.e., VFClass instance) with prescribed coefficients in chosen coordinates.

    The vector field is expressed as a linear combination of the coordinate antiholomorphic vector fields 
    corresponding to the variables in *arg2* with coefficients in *arg1*.

    Parameters:
    -----------
    arg1 : tuple of sympy expressions
        Coefficients for the vector field in terms of the antiholomorphic variables.
    arg2 : list or tuple
        A list or tuple containing Symbol objects that were initialized as holomorphic variables via DGCV variable creation functions.
    
    Returns:
    --------
    VFClass
        An antiholomorphic vector field expressed as a linear combination of coordinate vector fields.
    
    Raises:
    -------
    ValueError
        If the length of *arg1* does not match the number of holomorphic variables in *arg2*.
    Exception
        If variables in *arg2* were not initialized as holomorphic variables.

    Example:
    --------
    >>> from DGCV import createVariables, assembleFromAntiholVFC
    >>> createVariables('z', 'x', 'y', 3)
    >>> print(assembleFromAntiholVFC((z1 + z2, z3, 0), [z1, z2, z3]))
    (z1/2 + z2/2)*D_x1+z3/2*D_x2+(I*(z1 + z2)/2)*D_y1+I*z3/2*D_y2
    """

    # Ensure the length of coefficients matches the variable space
    if len(arg1) != len(arg2):
        raise ValueError("The number of coefficients in arg1 must match the number of variables in arg2.")
    vr = get_variable_registry()
    if all(var in vr['conversion_dictionaries']['symToReal'] for var in arg2):
        pass
    else:
        raise TypeError('`assembleFromHolVFC` expects the provided varibles for the coordinate space to be holomorphic.')

    varSpaceLoc = tuple([conj_with_hol_coor(j) for j in arg2])
    
    # Return the resulting vector field as a VFClass instance
    return VFClass(varSpaceLoc, arg1, 'complex')

def assembleFromCompVFC(arg1, arg2, arg3):
    """
    Constructs a complex vector field (i.e., VFClass instance) with prescribed coefficients of the vector field's 
    holomorphic and antiholomorphic parts in chosen coordinates.

    The vector field is expressed as a linear combination of the coordinate holomorphic and antiholomorphic vector 
    fields corresponding to the variables in *arg3*, with holomorphic part coefficients in *arg1* and antiholomorphic 
    part coefficients in *arg2*.

    Parameters:
    -----------
    arg1 : tuple of sympy expressions
        Coefficients for the holomorphic part of the vector field.
    arg2 : tuple of sympy expressions
        Coefficients for the antiholomorphic part of the vector field.
    arg3 : list or tuple
        A list or tuple containing Symbol objects that were initialized as holomorphic variables via DGCV variable creation functions.
    
    Returns:
    --------
    VFClass
        A complex vector field expressed as a linear combination of holomorphic and antiholomorphic coordinate vector fields.

    Raises:
    -------
    ValueError
        If the length of *arg1* or *arg2* does not match the number of variables in *arg3*.
    Exception
        If variables in *arg3* were not initialized as holomorphic variables.

    Example:
    --------
    >>> from DGCV import createVariables, assembleFromCompVFC
    >>> createVariables('z', 'x', 'y', 3)
    >>> assembleFromCompVFC((z1 + z2, z3, 0), (z1, z2, z3), [z1, z2, z3])
    VFClass instance representing the complex vector field with holomorphic and antiholomorphic parts
    """
    # Ensure the length of both sets of coefficients matches the variable space
    if len(arg1) != len(arg3) or len(arg2) != len(arg3):
        raise ValueError("The number of coefficients in arg1 and arg2 must match the number of variables in arg3.")
    
    # Construct the holomorphic and antiholomorphic vector fields and add them
    hol_vf = assembleFromHolVFC(arg1, arg3)
    antihol_vf = assembleFromAntiholVFC(arg2, arg3)
    
    # Return the resulting complex vector field
    return addVF(hol_vf, antihol_vf)

############## differential forms 

def makeZeroForm(arg1, varSpace=None, DGCVType=None, default_var_format=None):
    """
    Constructs a 0-form (a differential form of degree 0) based on the provided input.

    If *varSpace* is not provided, the variables will be inferred from the real parts of *arg1*. If *DGCVType* is 
    not provided, the function checks if the variables belong to the complex variable system (via the variable registry) 
    to determine the differential form type.

    Parameters:
    -----------
    arg1 : sympy expression or numeric value
        The scalar expression or value for the 0-form.
    varSpace : list or tuple, optional
        The variable space associated with the form. If not provided, the real free symbols of *arg1* will be used.
    DGCVType : str, optional
        The differential form type ('complex' or 'standard'). If not provided, the type will be inferred from the 
        variable space.

    Returns:
    --------
    DFClass
        A new differential form instance of degree 0.

    Example:
    --------
    >>> from DGCV import makeZeroForm, createVariables
    >>> createVariables('a b')
    >>> createVariables('z', 'x', 'y')
    >>> df1 = makeZeroForm(5)  # constant coeff DF form with trivial varSpace
    >>> df2 = makeZeroForm(a + b)  # standard type DFClass instance representing the 0-form with expression a + b and varSpace (a, b)
    >>> df3 = makeZeroForm(z * BARz)  # complex type DFClass instance representing the 0-form with expression x**2 + y**2 and varSpace (x, y)
    """
    variable_registry = get_variable_registry()


    # Determine the variable space
    if varSpace is None:
        if isinstance(arg1, (int, float)):
            varLoc = tuple()
        elif isinstance(arg1,sympy.Expr):
            if DGCVType=='complex':
                if default_var_format == 'real':
                    varLoc = tuple([j for j in allToReal(arg1).free_symbols if j in variable_registry['conversion_dictionaries']['realToSym']])
                else:
                    default_var_format='complex'
                    varLoc = tuple([j for j in allToSym(arg1).free_symbols if j in variable_registry['conversion_dictionaries']['symToReal']])
            else:
                varLoc = tuple(allToReal(arg1).free_symbols)
    else:
        if DGCVType=='complex':
            if all(var in variable_registry['conversion_dictionaries']['realToSym'] for var in varSpace) and default_var_format!='complex':
                varLoc=tuple(varSpace)
            elif all(var in variable_registry['conversion_dictionaries']['symToReal'] for var in varSpace) and default_var_format!='real' :
                varLoc=tuple(varSpace)
            else:
                raise TypeError('`makeZeroForm` received either all real coordinates for `varSpace` while `default_var_format=\'complex\'` was set, all complex coordinates for `varSpace` while `default_var_format=\'real\'` was set, or was given a set for `varspace` with a mixture of real and complex coordinates. None of these data types are supported in `makeZeroForm`. Tip: Specifying varSpace is optional, so if trying to create a complex type 0-form with the keyword `default_var_format` specified, consider not specifying the varSpace variables.')
        else:
            varLoc = tuple(varSpace)

    # Determine the differential form type
    if DGCVType is None:
        if all(var in variable_registry['conversion_dictionaries']['realToSym'] for var in varLoc):
            typeLoc = 'complex'
        elif all(var in variable_registry['conversion_dictionaries']['symToReal'] for var in varLoc):
            typeLoc = 'complex'
        else:
            typeLoc = 'standard'  # default to standard unless found in complex variable system
    else:
        typeLoc = DGCVType

    # Return the 0-form as a DFClass instance
    return DFClass(varLoc, {tuple(): arg1}, 0, DGCVType=typeLoc)

def exteriorDerivative(arg1, forceComplexType=None):
    """
    Computes the exterior derivative of a differential form (i.e., DFClass instance).

    If *arg1* is not already a differential form, it is converted into a 0-form via makeZeroForm. The function 
    supports handling both standard and complex forms, with the optional *forceComplexType* argument ensuring the 
    resulting form is treated as complex.

    Parameters:
    -----------
    arg1 : DFClass or sympy expression
        The differential form or expression to compute the exterior derivative of.
    forceComplexType : bool, optional
        If True, the result is forced to be of complex type, even if the original form is not (default is None).

    Returns:
    --------
    DFClass
        A new differential form representing the exterior derivative of *arg1*.

    Example:
    --------
    >>> from DGCV import createVariables, makeZeroForm, exteriorDerivative
    >>> createVariables('z', 'x', 'y')
    >>> f_Form = makeZeroForm(z,DGCVType='complex')
    >>> exteriorDerivative(f_Form)
    I*d_y+d_x
    """
    # Ensure arg1 is a DFClass or convert it into a zero-form
    if not isinstance(arg1, DFClass) and isinstance(arg1, (int,float,sympy.Expr)):
        arg1 = makeZeroForm(arg1, DGCVType='complex' if forceComplexType else None)
    elif arg1.DGCVType == 'complex':
        forceComplexType = True

    if isinstance(arg1,DFClass):
        typeSet = arg1._varSpace_type
    else:
        raise TypeError('`exteriorDerivative` can only operator of DFClass or scalars like sympy.Expr')

    # Helper function to compute the exterior derivative of a zero-form
    def extDerOfZeroForm(arg1,typeSet=None):
        if arg1.DGCVType == 'complex' or forceComplexType:
            if typeSet=='real':
                sparseDataLoc = {(j,): diff(allToReal(arg1.coeffsInKFormBasis[0]), arg1.varSpace[j]) for j in range(len(arg1.varSpace))}
            else:
                sparseDataLoc = {(j,): diff(allToSym(arg1.coeffsInKFormBasis[0]), arg1.varSpace[j]) for j in range(len(arg1.varSpace))}
        else:
            sparseDataLoc = {(j,): diff(arg1.coeffsInKFormBasis[0], arg1.varSpace[j]) for j in range(len(arg1.varSpace))}
        
        return DFClass(arg1.varSpace, sparseDataLoc, 1, DGCVType='complex' if forceComplexType else arg1.DGCVType)

    # Handle zero-forms
    if arg1.degree == 0:
        return extDerOfZeroForm(arg1,typeSet=typeSet)

    # Handle higher-degree forms
    minDataLoc = arg1.DFClassDataMinimal
    coeffsTo1Forms = [extDerOfZeroForm(makeZeroForm(j[1], varSpace=arg1.varSpace),typeSet=typeSet) for j in minDataLoc]
    # Construct the corresponding basis k-forms
    basisOfCoeffs = [
        DFClass(
            arg1.varSpace, 
            {tuple(j[0]):1}, 
            arg1.degree, 
            DGCVType='complex' if forceComplexType else arg1.DGCVType
        ) for j in minDataLoc
    ]
    
    # Multiply the one-forms by the basis and sum them
    return addDF(*[coeffsTo1Forms[j] * basisOfCoeffs[j] for j in range(len(minDataLoc))])

def LieDerivativeOf1Form(vf, oneForm):
    """
    Computes the Lie derivative of a 1-form (i.e., DFClass instance) *oneForm* along a vector field (i.e., VFClass instance) *vf*.

    The Lie derivative is computed using Cartan's formula:
        L_v(ω) = i_v(dω) + d(i_v(ω))
    where L_v is the Lie derivative along the vector field *vf*, ω is the 1-form, d is the exterior derivative, and i_v is the interior product with respect to the vector field.

    Parameters:
    -----------
    vf : VFClass
        The vector field along which the Lie derivative is computed.
    oneForm : DFClass
        The 1-form to compute the Lie derivative of.

    Returns:
    --------
    DFClass
        The Lie derivative of the 1-form along the vector field.

    """
    # Compute the Lie derivative using Cartan's formula
    return interiorProductOf2Form(vf, exteriorDerivative(oneForm)) + exteriorDerivative(interiorProductOf1Form(vf, oneForm))

def interiorProductOf1Form(vf, oneForm):
    """
    Computes the interior product of a vector field (i.e., VFClass instance) *vf* and a 1-form (i.e., DFClass instance) *oneForm*.

    The interior product (or contraction) of a vector field and a 1-form produces a 0-form, which can be viewed as the result of applying the 1-form to the vector field.

    Parameters:
    -----------
    vf : VFClass
        The vector field to contract with the 1-form.
    oneForm : DFClass
        The 1-form to contract with the vector field.

    Returns:
    --------
    DFClass
        A 0-form representing the result of the interior product of *vf* and *oneForm*.
    """
    varSpaceLoc = tuple(dict.fromkeys(vf.varSpace + oneForm.varSpace))
    return makeZeroForm(oneForm(vf), varSpace=varSpaceLoc, DGCVType=oneForm.DGCVType)

def interiorProductOf2Form(vf, twoForm):
    """
    Computes the interior product of a vector field (i.e., VFClass instance) *vf* and a 2-form (i.e., DFClass instance) *twoForm*.

    The result is a 1-form, expressed as a linear combination of 1-forms in the same variable space.

    Parameters:
    -----------
    vf : VFClass
        The vector field to contract with the 2-form.
    twoForm : DFClass
        The 2-form to contract with the vector field.

    Returns:
    --------
    DFClass
        A 1-form representing the result of the interior product of *vf* and *twoForm*.
    """
    varSpaceLoc = tuple(dict.fromkeys(vf.varSpace + twoForm.varSpace))

    # Access the global symbols directly using _cached_caller_globals
    coeffsLoc = [twoForm(vf, _cached_caller_globals['D_'+str(k)]) for k in varSpaceLoc]
    oneFormBasisLoc = [_cached_caller_globals['d_'+str(j)] for j in varSpaceLoc]

    # Return the result as a sum of the contracted 1-form
    return addDF(*[coeffsLoc[j] * oneFormBasisLoc[j] for j in range(len(varSpaceLoc))])

def interiorProduct(vf, kForm):
    """
    Computes the interior product of a vector field *vf* with a differential k-form *kForm* where k > 0.

    The interior product, or contraction, reduces the degree of a differential form by 1 and represents 
    the action of the vector field on the form. That is, the vector field is "taking one for the team"! 

    Parameters:
    -----------
    vf : VFClass
        The vector field to contract with the differential form.
    kForm : DFClass
        A differential k-form (degree > 0) to contract with the vector field.

    Returns:
    --------
    DFClass
        A differential form (degree k-1) representing the result of the interior product.

    Raises:
    -------
    Exception
        If *vf* is not a VFClass instance or *kForm* is not a DFClass instance with degree > 0.
    
    Example:
    --------
    >>> from DGCV import createVariables, makeZeroForm, interiorProduct
    >>> from sympy import latex
    >>> createVariables('z', 'x', 'y')
    >>> createVariables('v')
    >>> vf = v*D_x + D_y
    >>> omega = d_x * d_y * d_v
    >>> print(latex(interiorProduct(vf, omega)))
    $vd y \wedge d v -d x \wedge d v$
    """
    if [vf.__class__.__name__, kForm.__class__.__name__] != ['VFClass', 'DFClass']:
        raise Exception('Arguments *vf* and *kForm* must have respective types *VFClass* and *DFClass*.')
    
    if kForm.degree == 0:
        raise Exception('Differential form must have degree > 0')

    # Initialize a sparse dictionary to store non-zero entries
    sparse_array_loc = {}

    # Generate the list of indices for the contraction
    indexListLoc = [list(j) for j in chooseOp(range(len(kForm.varSpace)), kForm.degree)]
    indexListFilteredLoc = [list(j) for j in chooseOp(range(len(kForm.varSpace)), kForm.degree, withoutReplacement=True)]

    # Access the vector field components from the caller's globals
    VFListLoc = ['D_' + str(j) for j in kForm.varSpace]

    # Compute the interior product for each component, storing non-zero entries in sparse_array_loc
    for j in indexListLoc:
        result = kForm(*([vf] + [_cached_caller_globals[VFListLoc[k]] for k in j[1:]]))
        if j in indexListFilteredLoc and result != 0:
            sparse_array_loc[tuple(j[1:])] = result

    # Return a differential form using the sparse dictionary
    return DFClass(kForm.varSpace, sparse_array_loc, kForm.degree - 1, DGCVType=kForm.DGCVType)

def LieDerivative(vf, arg):
    """
    The Lie derivative, because sometimes differential forms need a little nudge in the right direction.
    
    Computes the Lie derivative with respect to the vector field (i.e., VFClass instance) *vf* of either another vector field 
    or a differential form (i.e., DFClass instance) *arg*.

    The Lie derivative of a vector field is given by the Lie bracket, and for a differential form, it is computed using Cartan's formula:
        L_v(ω) = d(i_v(ω)) + i_v(dω)
    
    Parameters:
    -----------
    vf : VFClass
        The vector field with respect to which the Lie derivative is computed.
    arg : VFClass or DFClass
        The vector field or differential form to compute the Lie derivative of.

    Returns:
    --------
    VFClass or DFClass
        The Lie derivative of *arg* with respect to *vf*.

    Raises:
    -------
    Exception
        If *vf* is not of type VFClass or *arg* is not of type VFClass/DFClass.

    Example:
    --------
    >>> from DGCV import createVariables, LieDerivative
    >>> from sympy import latex
    >>> createVariables('z', 'x', 'y')
    >>> createVariables('v', withVF=True)
    >>> vf = x * D_x
    >>> omega = d_x * d_y * d_v
    >>> print(latex(LieDerivative(vf, omega)))
    $d x \wedge d y \wedge d v$
    """
    if isinstance(vf,VFClass) and isinstance(arg,(int,float,sympy.Expr)):
        return vf(arg)
    # Check for valid argument types
    if [vf.__class__.__name__, arg.__class__.__name__] not in [['VFClass', 'DFClass'], ['VFClass', 'VFClass']]:
        raise Exception('Arguments *vf* and *arg* must have respective types *VFClass* and *VFClass/DFClass*.')

    # Case: Lie derivative of a differential form
    if isinstance(arg, DFClass):
        if arg.degree > 0:
            coeffList = arg.coeffsInKFormBasis
            oneFormList = [[_cached_caller_globals[k] for k in j] for j in arg.kFormBasisGenerators]

            # Helper function to operate on each factor of a k-form
            def operateOnKFactor(formList, k):
                firstFactor = prod(formList[0:k])
                secondFactor = LieDerivativeOf1Form(vf, formList[k])
                thirdFactor = prod(formList[k+1:])
                return firstFactor * secondFactor * thirdFactor

            # Compute the Lie derivative
            oneFormDerivatives = addDF(*[coeffList[j] * operateOnKFactor(oneFormList[j], k)
                                         for k in range(arg.degree)
                                         for j in range(len(coeffList))])
            coeffDerivatives = addDF(*[vf(coeffList[j]) * prod(oneFormList[j])
                                       for j in range(len(coeffList))])
            return coeffDerivatives + oneFormDerivatives

        # Case: Lie derivative of a 0-form
        elif arg.degree == 0:
            return DFClass(arg.varSpace, {0: vf(arg)}, 0)

    # Case: Lie derivative of a vector field (Lie bracket)
    return VF_bracket(vf, arg)


############## vector field and differential forms operations with linsolve computations

def decompose(obj, basis, return_parameters=False,new_parameters_label=None, _pass_error_report=None, _hand_off = None):
    """
    Decomposes a vector field or differential form as a linear combination of a given `basis` list.

    This function attempts to express the input `obj` (a VFClass or DFClass object) as a linear combination
    of the elements in the provided `basis` list. The `basis` list does not need to be linearly independent, 
    and if the decomposition is not unique, the function will parameterize the solution space. Any parameters 
    needed are automatically initialized and registered in DGCV's variable management framework (VMF).

    The function carefully handles variable types based on the `DGCVType` attribute of the objects. For objects 
    with `DGCVType='complex'`, it dynamically selects whether to perform real coordinate computations or complex 
    coordinate computations, depending on the input data. If a canonical variable formatting decision cannot be 
    made naturally from the input, the function will return warnings with explanations.

    Parameters
    ----------
    obj : VFClass or DFClass
        The vector field or differential form to decompose.
    basis : list of VFClass or DFClass
        A list of objects (vector fields or differential forms) to decompose `obj` with respect to. 
        The class of objects in the `basis` list must match the class of `obj`.
    return_parameters : bool, optional
        If True, the function will return the parameterized solution when the decomposition is not unique 
        (default is False). Parameters are initialized with labels registered within the VMF.
    new_parameters_label : str or None, optional
        If `return_parameters` is True and the decomposition is not unique, this label will be used 
        to name the new parameter variables. If None, obscure labels will be generated automatically (default is None).
    _pass_error_report : optional
        Internal use parameter for handling error reports in certain edge cases (default is None).

    Returns
    -------
    list
        The coefficients of the linear combination that expresses `obj` in terms of the `basis` list. 
        If the decomposition is parameterized, the returned list contains the parameterized solution.
        
    Raises
    ------
    TypeError
        If the class of `obj` does not match the class of elements in the `basis` list (i.e., both must 
        be either VFClass or DFClass), or if objects in the `basis` list have inconsistent `DGCVType` attributes.
        
    Warnings
    --------
    - A warning is issued if `obj` is not in the span of the provided `basis` list.
    - If the `basis` list is not linearly independent, the decomposition is not unique, and a parameterized 
    solution will be returned. The initialized parameters are registered as 'standard variables' in the VMF.

    Remarks
    -------
    - The function dynamically handles objects based on their `DGCVType` attribute. For `DGCVType='complex'`,
    it distinguishes between real and complex coordinate computations, converting the input as needed 
    to ensure consistency in formatting. If this decision cannot be determined from the input data, 
    the function issues warnings explaining the necessary canonical formatting.
    
    Example
    -------
    # Decompose a vector field 'vf' as a linear combination of two basis vector fields 'vf1' and 'vf2'
    coeffs, basis_used = decompose(vf, [vf1, vf2])

    # Decompose a differential form 'df' with a parameterized solution due to non-uniqueness
    coeffs, basis_used = decompose(df, [df1, df2, df3], return_parameters=True, new_parameters_label='p')
    """
    if not isinstance(obj,(VFClass,DFClass)):
        raise TypeError('`decompose` only operates on `DFClass` and `VFClass` objects at this time.')
    condition1=isinstance(obj, VFClass) and all([isinstance(j,VFClass) for j in basis])
    condition2=isinstance(obj, DFClass) and all([isinstance(j,DFClass) for j in basis])
    if condition2:
        basis = [j for j in basis if j.degree==obj.degree] # discard unneeded DFCass elements
    if not condition1 and not condition2:
        raise TypeError('`decompose` needs all objects to be of the same class (i.e., all vector fields VFClass or add diff. forms DFClass)')
    

    if len(set([j.DGCVType for j in basis]))>1:
        raise TypeError('`decompose` was given vector fields or differential forms with different `DGCVType` attribute values. Reformat them to all have `DGCVType` \'standard\' or \'complex\'. The descision whether or not to track complex variable systems in this computation affects the outcome, so `decompose` will not automate attribute homogenization here by design.')

    # make _varSpace_type uniform and strip unnecessary variables from varSpace attributes
    if obj.DGCVType=='complex':
        if obj._varSpace_type=='real':
            obj = compressDGCVClass(allToReal(obj))
            basis = [compressDGCVClass(allToReal(j)) for j in basis]
        elif obj._varSpace_type=='complex':
            obj = compressDGCVClass(allToSym(obj))
            basis = [compressDGCVClass(allToSym(j)) for j in basis]
    else:
        obj = compressDGCVClass(obj)
        basis = [compressDGCVClass(j) for j in basis]

    dimLoc=len(basis)
    if dimLoc==0:
        return [tuple(),tuple()]
    tempLabel='deco'+create_key()
    variableProcedure(tempLabel,dimLoc,initialIndex=0,_tempVar=retrieve_passkey())
    tempVars=_cached_caller_globals[tempLabel]
    if isinstance(obj, VFClass):
        solObj=compressDGCVClass(addVF(*[_cached_caller_globals[tempLabel+str(j)]*basis[j] for j in range(dimLoc)]))
        if not all([j in solObj.varSpace for j in obj.varSpace]):
            warnings.warn(f"The vector field {obj} is not in the span of {basis}")
            return []
        eqns=(compressDGCVClass(obj-solObj)).coeffs
    if isinstance(obj, DFClass):
        solObj=compressDGCVClass(addDF(*[_cached_caller_globals[tempLabel+str(j)]*basis[j] for j in range(dimLoc)]))
        if not all([j in solObj.varSpace for j in obj.varSpace]):
            warnings.warn(f"The differential form {obj} is not in the span of {basis}")
            obj=changeDFBasis(obj,solObj.varSpace)
            return []
        eqns=[j[1] for j in (compressDGCVClass(obj-solObj)).DFClassDataMinimal]

    sol=linsolve(eqns,tempVars)
    if len(list(sol))==0:
        printStr=f"`decompose` rolled back its algorithm because `linsolve` failed to solve the equations {eqns} w.r.t. the variables {tempVars}"
        clearVar(*listVar(temporary_only=True),report=False)
        if _pass_error_report==retrieve_passkey():
            return printStr
        else:
            raise Exception(printStr)
    solDict=dict(zip(tempVars,list(sol)[0]))
    freeVar=[t for t in tempVars if t==solDict[t]]
    if return_parameters==True and len(freeVar)>0:
        if isinstance(new_parameters_label,str):
            newLabel=new_parameters_label
            variableProcedure(newLabel,len(freeVar))
        else:
            newLabel='NULL'+create_key()
            variableProcedure(newLabel,len(freeVar),_obscure=None)
            warnings.warn(f'The provided object list to decompose w.r.t. is not linearly independent, so variables with intintionally obscure labels where created to parameteraze the solution space. To have the new variable\'s be assigned a particular label instead, use `new_parameters_label=True`.\n The new labels are: {list(_cached_caller_globals[newLabel])} \n Use that list to substitute the obscure labels for nicer ones as needed.')
        subDict=dict(zip(freeVar,_cached_caller_globals[newLabel]))
        return_list=[sympify(j).subs(subDict) for j in list(sol)[0]]
        clearVar(*listVar(temporary_only=True),report=False)
        return return_list, basis
    else:
        if len(freeVar)>0 and _hand_off !=  retrieve_passkey():        
            warnings.warn('The provided object list to decompose w.r.t. is not linearly independent, so the decomposition might not be unique. `decompostion` will still return a solution. To get all solutions with parameters set `return_parameters=True`.')
        subsList=[(t,0) for t in freeVar]
        clearVar(*listVar(temporary_only=True),report=False)
        return [sympify(j).subs(subsList) for j in list(sol)[0]], basis

def get_coframe(VFList):
    """
    Constructs a coframe (set of differential forms) corresponding to a given list of vector fields.

    This function computes a coframe, a set of differential forms, for the input list of vector fields `VFList`. 
    Each differential form in the coframe evaluates to 1 when applied to its corresponding vector field and 
    to 0 when applied to all other vector fields in the list. The function dynamically handles both real and 
    holomorphic coordinate systems for vector fields.

    For vector fields with `DGCVType='complex'`, `get_coframe` determines whether evaluations should default 
    to real or holomorphic coordinates based on the first vector field in the list, and all other vector fields 
    are converted accordingly. If such conversions occur, `get_coframe` issues warnings
    explaining its formatting choice.

    Parameters
    ----------
    VFList : list of VFClass
        A list of vector fields for which the coframe will be constructed. All vector fields in the list must 
        have the same `DGCVType` attribute (i.e., 'standard' or 'complex'), and for the `DGCVType='complex'` 
        case coordinate systems used in their definitions should be consistent (real or holomorphic).

    Returns
    -------
    list of DFClass
        The coframe, represented as a list of differential forms corresponding to the input vector fields. 
        Each differential form evaluates to 1 on the corresponding vector field and 0 on the others.

    Raises
    ------
    TypeError
        If the vector fields in `VFList` have inconsistent `DGCVType` attributes or if the list contains 
        linearly dependent vector fields that prevent the construction of a coframe.
        
    Warnings
    --------
    - If the vector fields do not span the minimal coordinate space containing variables in their definitions, 
    the function returns a non-unique "coframe" and issues 
    a warning.
    - If some vector fields are defined with respect to real coordinates and others with respect to holomorphic 
    coordinates, the function automatically converts them all to a consistent coordinate system based on the 
    first vector field in the list, and a warning is issued.
    
    Example
    -------
    >>> from DGCV import createVariables, get_coframe
    >>> createVariables('z', 'x', 'y', 3, default_var_format='real')
    >>> vfList = [D_z1 + z1**3 * D_BARz1, D_BARz1 + BARz1**3 * D_z1]
    >>> coframe = get_coframe(vfList)
    >>> print(coframe)

    [((x1**3 - 3*I*x1**2*y1 - 3*x1*y1**2 + I*y1**3 - 1)/(x1**6 + 3*x1**4*y1**2 + 3*x1**2*y1**4 + y1**6 - 1))*d_x1 +
    ((-I*x1**3 - 3*x1**2*y1 + 3*I*x1*y1**2 + y1**3 - I)/(x1**6 + 3*x1**4*y1**2 + 3*x1**2*y1**4 + y1**6 - 1))*d_y1,
    ((x1**3 + 3*I*x1**2*y1 - 3*x1*y1**2 - I*y1**3 - 1)/(x1**6 + 3*x1**4*y1**2 + 3*x1**2*y1**4 + y1**6 - 1))*d_x1 +
    ((I*x1**3 - 3*x1**2*y1 - 3*I*x1*y1**2 + y1**3 + I)/(x1**6 + 3*x1**4*y1**2 + 3*x1**2*y1**4 + y1**6 - 1))*d_y1]
    """
    dimLoc=len(VFList)
    vr=get_variable_registry()
    if len(set([j.DGCVType for j in VFList]))>1:
        raise TypeError('`coframe` was given a set of vector fields with different `DGCVType` attribute values. Reformat the vector fields to all have `DGCVType` \'standard\' or \'complex\'')
    elif VFList[0]._varSpace_type=='real':
        varFormat='real'
        VFList=[allToReal(j) for j in VFList]
        if any([j._varSpace_type=='complex' for j in VFList[1:]]):
            warnings.warn('`coframe` was given a list of `DGCVType=\'complex\'` vector fields in which some are defined w.r.t. to real coordinates and others w.r.t. holomorphic coordinates. To find a common coordinate system `coframe` converted everything to the coordinate type of the first provided VF, which is `real`. Tip: if holomorphic coordinates were prefered then convert the provided VF to holom. format before giving them to `coframe` (it is even enough to do this for just the first VF in the list). Use functions like `allToHol` and `allToSym` to achieve this.')
    elif VFList[0]._varSpace_type=='complex':
        varFormat='complex'
        VFList=[allToSym(j) for j in VFList]
        if any([j._varSpace_type=='real' for j in VFList[1:]]):
            warnings.warn('`coframe` was given a list of `DGCVType=\'complex\'` vector fields in which some are defined w.r.t. to real coordinates and others w.r.t. holomorphic coordinates. To find a common coordinate system `coframe` converted everything to the coordinate type of the first provided VF, which is `holomorphic`. Tip: if real coordinates were prefered then convert the provided VF to real format before giving them to `coframe` (it is even enough to do this for just the first VF in the list). The function `allToReal` is one way to do such conversion.')
    else:
        varFormat='standard'
    varSpaceLoc = tuple(set.union(*[set(d.keys()) for d in [minimalVFDataDict(j) for j in VFList]]))
    if len(varSpaceLoc)>dimLoc:
        warnings.warn('Fewer vector fields were given than the minimal number of variables needed to define them in the given coordinate representations, so the coframe solution returned will not be unique. A \"coframe\" will be returned anyway. Add a complimentary set of VF to specifify a unique solution. Example: if working with holomophic VF or real VF expressions in coordinates from a complex system, toss in the coordinate antiholomorphic VF or coordinate purely imaginary (e.g., D_z-D_BARz) to fill out a basis.\n If this message comes as a surprise, it may be that you are providing real VF of complex type (like, D_x or D_y where x and y are real/im. parts of a complex variable system) but your VF were defined w.r.t. holomorphic coordinates rather than real (so internally, D_x is regarded as D_z+D_BARz etc.). Similarly, holomorphic VF defined w.r.t. real coordinates could result in the same suprise, as internally 2*D_z is then regarded D_x-I*D_y etc. Use functions like `allToReal`, `allToHol`, and `allToSym`, for fine control of these properties.')
    if len(varSpaceLoc)<dimLoc:
        raise TypeError('`coframe` cannot find a coframe because the provided vector fields are not linearly independent')
    def getDF(key,parentsDict):
        keyStr=str(key)
        for var in parentsDict:
            if keyStr in parentsDict[var]['variable_relatives']:
                if parentsDict[var]['variable_relatives'][keyStr]['DFClass'] == None:
                    raise TypeError('One of the provided vector fields was defined over coordinates including at least one variable that was initialized without a corresponding coordinate vector field registered in the DGCV variable management framework. Suggestion: use DGCV variable creation functions to initialize variables. Use `createVariables(--,withVF=True)` or `createVariables(--,complex=True)` to initialize all variables that the vector fields are defined w.r.t.')
                else:
                    return parentsDict[var]['variable_relatives'][keyStr]['DFClass']
            
        
    if varFormat == 'standard':
        dfBasis=[_remove_complex_handling(getDF(j,vr['standard_variable_systems'] | vr['complex_variable_systems'])) for j in varSpaceLoc]
    elif varFormat == 'complex':
        dfBasis=[allToSym(getDF(j,vr['complex_variable_systems'])) for j in varSpaceLoc]
    elif varFormat=='real':
        dfBasis=[allToReal(getDF(j,vr['complex_variable_systems'])) for j in varSpaceLoc]
    tempLabel='coframe'+create_key()
    tempVars=[]
    solDFs=[]
    eqns=[]
    for j in range(dimLoc):
        variableProcedure(tempLabel+str(j)+'_',len(dfBasis),_tempVar=retrieve_passkey())
        varListLoc=list(_cached_caller_globals[tempLabel+str(j)+'_'])
        tempVars=tempVars+varListLoc
        joinList=[varListLoc[k]*dfBasis[k] for k in range(len(dfBasis))]
        newDF=addDF(*joinList)
        solDFs=solDFs+[newDF]
        newEqns=[newDF(VFList[k])-1 if k==j else newDF(VFList[k]) for k in range(dimLoc)]
        eqns=eqns+newEqns
        
    sol=linsolve(eqns,tempVars)
    if len(list(sol))==0:
        warnings.warn('linsolve failed. Trying solve.')
        sol=solve(eqns,tempVars,dict=True)
        if len(sol)==0:
            printStr=f"`coframe` rolled back its algorithm because `linsolve` and `solve` failed to solve the equations {eqns} w.r.t. the variables {tempVars}"
            clearVar(*listVar(temporary_only=True),report=False)
            raise Exception(printStr)
        else:
            solDict=sol[0]
            flagSolveLogic=True
    else:
        solDict=dict(zip(tempVars,list(sol)[0]))
        flagSolveLogic=False
    if flagSolveLogic:
        freeVar=[]
    else:
        freeVar=[t for t in tempVars if t==solDict[t]]
    if len(freeVar)>0:
        subsList=[(t,0) for t in freeVar]
        coframeSol=[j.subs(solDict).subs(subsList) for j in solDFs]
        clearVar(*listVar(temporary_only=True),report=False)
    else:
        coframeSol=[j.subs(solDict) for j in solDFs]
        clearVar(*listVar(temporary_only=True),report=False)

    return coframeSol

def annihilator(objList, coordinate_Space, control_distribution=None, _pass_error_report=None, allow_div_by_zero=False):
    """
    Finds annihilators for a given list of vector fields or differential forms.

    This function computes objects that "annihilate" the provided list of vector fields or differential forms.
    An annihilator is either the span of differential forms that evaluate to zero on each vector field in the list, 
    or vector fields whose interior product annihilates each differential form in the list. `annihilator` dynamically 
    handles both real and holomorphic coordinate systems and can convert between them as needed. Additionally, 
    solutions can be constrained to a given control distribution by using the `control_distribution` keyword.

    Parameters
    ----------
    objList : list of VFClass or DFClass
        A list of vector fields or differential forms for which the annihilator will be computed. All objects 
        in the list must be of the same class (either all vector fields or all differential forms) and have 
        consistent `DGCVType` attributes (i.e., 'standard' or 'complex').
    coordinate_Space : list, tuple, or set
        A collection of variables that define the coordinate system in which the annihilator is to be computed.
    allow_div_by_zero : bool, optional
        If True, allows the annihilator to be returned without scaling to avoid division by zero (default is False).
        Scaling to avoid division has more computational overhead but typically simplifies output.
    _pass_error_report : optional
        Internal use parameter for handling error reports in certain edge cases (default is None).

    Returns
    -------
    list
        A list of differential forms (if vector fields were provided) or vector fields (if differential forms were 
        provided) that annihilate the objects in `objList`.

    Raises
    ------
    TypeError
        If the objects in `objList` are not all of the same type (i.e., all vector fields or all differential forms), 
        or if the `coordinate_Space` is not a valid list, tuple, or set.
        
    Warnings
    --------
    - If the objects in `objList` are defined with inconsistent coordinate system types (real vs. holomorphic), the 
    function converts them to a consistent coordinate system and issues a warning.
    
    Example
    -------
    >>> from DGCV import createVariables, annihilator, exteriorDerivative, complex_struct_op, Del, DelBar, allToReal
    >>> createVariables('z', 'x', 'y', 4, initialIndex=0)
    >>> rho = (x1*x2 + x1**2*x3 - x0)  # A defining equation for a real hypersurface M in C^4
    >>> d_rho = exteriorDerivative(rho)  # Its differential will annihilate TM
    >>> print(d_rho)

    (2*x1*x3 + x2)*d_x1 + x1*d_x2 + x1**2*d_x3 - 1*d_x0

    >>> dfList = [d_rho]
    >>> TMbasis = annihilator(dfList, x+y)  # Use annihilator to compute the tangent bundle TM
    >>> TMbasis

    [(16*x1*x3 + 8*x2)*D_x0 + 8*D_x1, 8*x1*D_x0 + 8*D_x2, 8*x1**2*D_x0 + 8*D_x3, 8*D_y0, 8*D_y1, 8*D_y2, 8*D_y3]

    >>> J_of_TMbasis = [complex_struct_op(vf) for vf in TMbasis]  # Get the image of TM under the complex structure operator.
    >>> J_of_TMbasis

    [(16*x1*x3 + 8*x2)*D_y0 + 8*D_y1, 8*x1*D_y0 + 8*D_y2, 8*x1**2*D_y0 + 8*D_y3, -8*D_x0, -8*D_x1, -8*D_x2, -8*D_x3]

    >>> CR_distribution = annihilator(annihilator(J_of_TMbasis, x+y) + annihilator(TMbasis, x+y), x+y)  
    >>> CR_distribution  # Use annihilator to get the CR distribution, which is the intersection of TM with CTM

    [(16*x1*x3 + 8*x2)*D_x0 + 8*D_x1, 8*x1*D_x0 + 8*D_x2, 8*x1**2*D_x0 + 8*D_x3, 
    (16*x1*x3 + 8*x2)*D_y0 + 8*D_y1, 8*x1*D_y0 + 8*D_y2, 8*x1**2*D_y0 + 8*D_y3]

    >>> LeviForm = allToReal(Del(DelBar(rho)))  # Apply Dolbeault operators to represent the Levi form
    >>> print(LeviForm._repr_latex_())

    # Output: <LeviForm in LaTeX formatted plain text>

    >>> Levi_kernel = annihilator([LeviForm], x+y, control_distribution=CR_distribution)  
    >>> Levi_kernel  # annihilator reveals that the Levi form has a real 2-d. kernel

    [-64*x1**2*D_x0 - 128*x1*D_x2 + 64*D_x3, -64*x1**2*D_y0 - 128*x1*D_y2 + 64*D_y3]

    >>> not_the_Levi_kernel = annihilator([LeviForm], x+y)  
    >>> not_the_Levi_kernel  # Without constraining annihilator to the CR distribution, it finds a kernel that is too large.

    [8*D_x0, -16*x1*D_x2 + 8*D_x3, 8*D_y0, -16*x1*D_y2 + 8*D_y3]
    """
    if not isinstance(coordinate_Space,(list,tuple,set)):
        raise TypeError('`annihilator` needs the provided coordinate space (i.e., second argument) to be a list/tuple/set containing distinct variables. Tip: Use `createVariables(--,withVF=True)` or `createVariables(--,complex=True)` to initialize all variables with corresponding coordinate VF and DF.')
    if control_distribution==None:
        control_distribution=[]
    elif not isinstance(control_distribution,(list,tuple,set)):
        raise TypeError('`annihilator` needs the (optional) control distribution to be a list/tuple/set containing distinct VFClass/DFClass objects, if provided.')
    else:
        control_distribution=list(control_distribution)
    condition1=all([isinstance(j,VFClass) for j in objList]) and all([isinstance(j,DFClass) for j in control_distribution])
    condition2=all([isinstance(j,DFClass) for j in objList]) and all([isinstance(j,VFClass) for j in control_distribution])
    if condition2:
        i=0
        while i < len(objList):
            j = objList[i]
            if j.degree==0:
                if not j.varSpace:
                    raise TypeError('`annihilator` recieved a 0-form with trivial variable space.')
                elif not j.is_zero():
                    return VFClass(coordinate_Space,[0])
                else:
                    objList.pop(i)
            else:
                i += 1

    if not condition1 and not condition2:
        raise TypeError('`annihilator` only operates on lists/tuples that contain either all vector fields VFClass or all diff. forms DFClass)If providing a control distribution (optional) to find solutions within, its elements\' class must compliment that of the objects to annihilate.')
    
    vr=get_variable_registry()

    if condition1:
        if len(set([j.DGCVType for j in objList+control_distribution]))>1:
            raise TypeError('`annihilator` was given a set of vector fields (and possibly DFClass controls) with different `DGCVType` attribute values. Reformat the vector fields (and controls if provided) to all have `DGCVType` \'standard\' or \'complex\'')
        elif objList[0]._varSpace_type=='real':
            varFormat='real'
            varSpaceLoc=_format_complex_coordinates(coordinate_Space,default_var_format='real')
            objList=[allToReal(j) for j in objList]
            control_distribution=[allToReal(j) for j in control_distribution]
            if any([j._varSpace_type=='complex' for j in objList[1:]]):
                warnings.warn('`annihilator` was given a list of `DGCVType=\'complex\'` vector fields or DFClass controls in which some are defined w.r.t. to real coordinates and others w.r.t. holomorphic coordinates. To find a common coordinate system `annihilator` converted everything to the coordinate type of the first provided VF, which is `real`. Tip: if holomorphic coordinates were prefered then convert the provided VF to holom. format before giving them to `annihilator` (it is even enough to do this for just the first VF in the list). Use functions like `allToHol` and `allToSym` to achieve this.')
        elif objList[0]._varSpace_type=='complex':
            varFormat='complex'
            varSpaceLoc=_format_complex_coordinates(coordinate_Space,default_var_format='complex')
            objList=[allToSym(j) for j in objList]
            control_distribution=[allToSym(j) for j in control_distribution]
            if any([j._varSpace_type=='real' for j in objList[1:]]):
                warnings.warn('`annihilator` was given a list of `DGCVType=\'complex\'` vector fields or DFClass controls in which some are defined w.r.t. to real coordinates and others w.r.t. holomorphic coordinates. To find a common coordinate system `annihilator` converted everything to the coordinate type of the first provided VF, which is `holomorphic`. Tip: if real coordinates were prefered then convert the provided VF to real format before giving them to `annihilator` (it is even enough to do this for just the first VF in the list). The function `allToReal` is one way to do such conversion.')
        else:
            varFormat='standard'
            varSpaceLoc=coordinate_Space

        if control_distribution==[]:
            def getDF(key,parentsDict):
                keyStr=str(key)
                for var in parentsDict:
                    if keyStr in parentsDict[var]['variable_relatives']:
                        if parentsDict[var]['variable_relatives'][keyStr]['DFClass'] == None:
                            raise TypeError('One of the provided vector fields was defined over coordinates including at least one variable that was initialized without a corresponding coordinate vector field registered in the DGCV variable management framework. Suggestion: use DGCV variable creation functions to initialize variables. Use `createVariables(--,withVF=True)` or `createVariables(--,complex=True)` to initialize all variables that the vector fields are defined w.r.t.')
                        else:
                            return parentsDict[var]['variable_relatives'][keyStr]['DFClass']
                                
            if varFormat == 'standard':
                dfBasis=[_remove_complex_handling(getDF(j,vr['standard_variable_systems'] | vr['complex_variable_systems'])) for j in varSpaceLoc]
            elif varFormat == 'complex':
                dfBasis=[allToSym(getDF(j,vr['complex_variable_systems'])) for j in varSpaceLoc]
            elif varFormat=='real':
                dfBasis=[allToReal(getDF(j,vr['complex_variable_systems'])) for j in varSpaceLoc]
        else:
            dfBasis = control_distribution

        tempLabel='anni'+create_key()

        dimLoc=len(objList)
        variableProcedure(tempLabel,len(dfBasis),_tempVar=retrieve_passkey())
        tempVars=list(_cached_caller_globals[tempLabel])
        solDF=addDF(*[tempVars[k]*dfBasis[k] for k in range(len(dfBasis))])
        eqns=[solDF(objList[k]) for k in range(dimLoc)]
            
        sol=linsolve(eqns,tempVars)

        if len(list(sol))==0:
            printStr=f"`annihilator` rolled back its algorithm because `linsolve` failed to solve the equations {eqns} w.r.t. the variables {tempVars}"
            clearVar(*listVar(temporary_only=True),report=False)
            if _pass_error_report==retrieve_passkey():
                return printStr
            else:
                raise Exception(printStr)
        solDict=dict(zip(tempVars,list(sol)[0]))
        solDF = solDF.subs(solDict)
        freeVar=[t for t in tempVars if t==solDict[t]]
        if len(freeVar)==0:
            solDFs=[solDF]
        else:
            solDFs=[solDF.subs({j:1}).subs([(t,0) for t in freeVar]) for j in freeVar]
        clearVar(*listVar(temporary_only=True),report=False)

        if allow_div_by_zero:
            return solDFs
        else:
            def rescaleCD(df):
                data=df.DFClassDataDict
                denoms=[]
                for j in data:
                    denoms += [denom(data[j])]
                scale=simplify(sum(denoms))
                return simplify(scale*df)
            return [rescaleCD(df) for df in solDFs]


    if condition2:
        if len(set([j.DGCVType for j in objList+control_distribution]))>1:
            raise TypeError('`annihilator` was given a set of differential forms (and possibly VFClass controls) with different `DGCVType` attribute values. Reformat them to all have `DGCVType` \'standard\' or \'complex\'')
        elif objList[0]._varSpace_type=='real':
            varFormat='real'
            varSpaceLoc=_format_complex_coordinates(coordinate_Space,default_var_format='real')
            control_distribution=[allToReal(j) for j in control_distribution]
            if any([j._varSpace_type=='complex' for j in objList[1:]]):
                warnings.warn('`annihilator` was given a list of `DGCVType=\'complex\'` differential forms or VFClass controls in which some are defined w.r.t. to real coordinates and others w.r.t. holomorphic coordinates. To find a common coordinate system `annihilator` converted everything to the coordinate type of the first provided DF, which is `real`. Tip: if holomorphic coordinates were prefered then convert the provided DF to holom. format before giving them to `annihilator` (it is even enough to do this for just the first DF in the list). Use functions like `allToHol` and `allToSym` to achieve this.')
                objList=[allToReal(j) for j in objList]
        elif objList[0]._varSpace_type=='complex':
            varFormat='complex'
            varSpaceLoc=_format_complex_coordinates(coordinate_Space,default_var_format='complex')
            control_distribution=[allToSym(j) for j in control_distribution]
            if any([j._varSpace_type=='real' for j in objList[1:]]):
                warnings.warn('`annihilator` was given a list of `DGCVType=\'complex\'` differential forms or VFClass controls in which some are defined w.r.t. to real coordinates and others w.r.t. holomorphic coordinates. To find a common coordinate system `annihilator` converted everything to the coordinate type of the first provided DF, which is `holomorphic`. Tip: if real coordinates were prefered then convert the provided DF to real format before giving them to `annihilator` (it is even enough to do this for just the first DF in the list). The function `allToReal` is one way to do such conversion.')
                objList=[allToSym(j) for j in objList]
        else:
            varFormat='standard'
            varSpaceLoc=coordinate_Space
        objList = [changeDFBasis(j,varSpaceLoc) for j in objList]

        if control_distribution==[]:
            def getVF(key,parentsDict):
                keyStr=str(key)
                for var in parentsDict:
                    if keyStr in parentsDict[var]['variable_relatives']:
                        if parentsDict[var]['variable_relatives'][keyStr]['VFClass'] == None:
                            raise TypeError('One of the provided differential forms was defined over coordinates including at least one variable that was initialized without a corresponding coordinate 1-form registered in the DGCV variable management framework. Suggestion: use DGCV variable creation functions to initialize variables. Use `createVariables(--,withVF=True)` or `createVariables(--,complex=True)` to initialize all variables that the differential forms are defined w.r.t.')
                        else:
                            return parentsDict[var]['variable_relatives'][keyStr]['VFClass']
                        
            if varFormat == 'standard':
                vfBasis=[_remove_complex_handling(getVF(j,vr['standard_variable_systems'] | vr['complex_variable_systems'])) for j in varSpaceLoc]
            elif varFormat == 'complex':
                vfBasis=[allToSym(getVF(j,vr['complex_variable_systems'])) for j in varSpaceLoc]
            elif varFormat=='real':
                vfBasis=[allToReal(getVF(j,vr['complex_variable_systems'])) for j in varSpaceLoc]
        else:
            vfBasis = control_distribution
        
        tempLabel='anni'+create_key()

        dimLoc=len(objList)
        variableProcedure(tempLabel,len(vfBasis),_tempVar=retrieve_passkey())
        tempVars=list(_cached_caller_globals[tempLabel])
        solVF=addVF(*[tempVars[k]*vfBasis[k] for k in range(len(vfBasis))])
        eqns=sum([[l[1] for l in interiorProduct(solVF,objList[k]).DFClassDataMinimal] for k in range(dimLoc)],[])
            
        sol=linsolve(eqns,tempVars)

        if len(list(sol))==0:
            printStr=f"`annihilator` rolled back its algorithm because `linsolve` failed to solve the equations {eqns} w.r.t. the variables {tempVars}"
            clearVar(*listVar(temporary_only=True),report=False)
            if _pass_error_report==retrieve_passkey():
                return printStr
            else:
                raise Exception(printStr)
        solDict=dict(zip(tempVars,list(sol)[0]))
        solVF = solVF.subs(solDict)
        freeVar=[t for t in tempVars if t==solDict[t]]
        if len(freeVar)==0:
            solVFs=[solVF]
        else:
            solVFs=[solVF.subs({j:1}).subs([(t,0) for t in freeVar]) for j in freeVar]
        clearVar(*listVar(temporary_only=True),report=False)
        if allow_div_by_zero:
            return solVFs
        else:
            def rescaleCD(vf):
                data=vf.coeffs
                denoms=[denom(j) for j in data]
                scale=simplify(sum(denoms))
                return simplify(scale*vf)
            return [rescaleCD(vf) for vf in solVFs]




# class structureEquations(Basic):

#     def __new__(cls, basis, structureFunctionsDict, connectionFormsDict, torsion = None):
#         # Call Basic.__new__ with only the positional arguments
#         obj = Basic.__new__(cls, basis, structureFunctionsDict)
#         return obj

#     def __init__(self, basis, structureFunctionsDict, connectionFormsDict, torsion = None):
#         self.basis = basis
#         self.structureFunctionsDict = structureFunctionsDict
#         self.connectionFormsDict = connectionFormsDict
#         self.torsion = torsion

#     def __str__(self):
#         result = "Structure Functions:\n"
#         for basis_elem, inner_dict in self.structureFunctionsDict.items():
#             result += f"{basis_elem} : {inner_dict}\n"
#         return result

#     def _repr_latex_(self):
#         latex_result = r"\begin{align*}"
#         dummy_labels = [f"\\_e_{{{i+1}}}" for i in range(len(self.basis))]  # Create dummy labels for the 1-forms
        
#         # Loop through the structure functions dict
#         for i, (basis_elem, two_form_dict) in enumerate(self.structureFunctionsDict.items()):
#             latex_result += f"d ({dummy_labels[i]}) &= "  # Start the equation with the 1-form dummy label
            
#             # Build the 2-form sum
#             two_form_sum = []
#             for (j, k), coeff in two_form_dict.items():
#                 two_form_label = f"{dummy_labels[j]} \\wedge {dummy_labels[k]}"
#                 term = f"{LaTeX(coeff)} \\cdot {two_form_label}"
#                 two_form_sum.append(term)
            
#             # Join the terms in the sum
#             latex_result += " + ".join(two_form_sum) + r" \\"
        
#         latex_result += r"\end{align*}"
#         return latex_result
    
#     def _eval_simplify(self, **kwargs):
#         # Loop through the structure functions and apply simplify to each coefficient
#         simplified_structureFunctions = {}
    
#         for basis_elem, two_form_dict in self.structureFunctionsDict.items():
#             simplified_inner_dict = {}
#             for key, val in two_form_dict.items():
#                 # Apply SymPy's simplify to the value
#                 simplified_val = simplify(val, **kwargs)
#                 simplified_inner_dict[key] = simplified_val
#             simplified_structureFunctions[basis_elem] = simplified_inner_dict
    
#         # Return a new instance of structureEquations with the simplified data
#         return structureEquations(self.basis, simplified_structureFunctions)

#     def subs(self, *args, **kwargs):
#         # Create a new dictionary to hold the substituted structure functions
#         substituted_structureFunctions = {}
        
#         # Iterate over the structure functions dictionary
#         for basis_elem, two_form_dict in self.structureFunctionsDict.items():
#             # Initialize an inner dictionary for the new substituted values
#             substituted_inner_dict = {}
            
#             # Iterate over the inner dictionary (2-form indices and their coefficients)
#             for key, val in two_form_dict.items():
#                 # Apply sympy's substitution method
#                 substituted_val = sympify(val).subs(*args, **kwargs)
#                 substituted_inner_dict[key] = substituted_val
            
#             # Store the substituted inner dictionary in the outer dictionary
#             substituted_structureFunctions[basis_elem] = substituted_inner_dict
        
#         # Return a new instance of the structureEquations class with the substituted data
#         return structureEquations(self.basis, substituted_structureFunctions)
    

# def get_structureEquations(dfList, torsion = None):
#     if not isinstance(dfList,(list,tuple)):
#         raise TypeError('`structureEquations needs the coframe to be a list/tuple of DFClass objects.')
#     elif not all([isinstance(j,DFClass) for j in dfList]):
#         raise TypeError('`structureEquations needs the coframe to be a list/tuple of DFClass objects.')
#     elif not all([j.degree==1 for j in dfList]):
#         raise TypeError('`structureEquations was given a list/tuple k-forms for the coframe whose degrees are not all one.')
#     if torsion == None:
#         df = get_DF(dfList[0].varSpace[0])[0]
#         Torsion = [0*df*df]*len(dfList)
#     elif not isinstance(torsion,(list,tuple)):
#         raise TypeError('If providing torsion `structureEquations` needs it to be a list of 2-forms.')
#     elif not all([isinstance(j,DFClass) for j in Torsion]):
#         raise TypeError('If providing torsion `structureEquations` needs it to be a list of 2-forms.')
#     elif not all([j.degree==2 for j in Torsion]):
#         raise TypeError('If providing torsion `structureEquations` needs it to be a list of 2-forms.')
#     dfList = list(dfList)
#     Torsion = list(Torsion)

#     if len(set([j.DGCVType for j in dfList]))>1:
#         raise TypeError('`get_structureEquations` was given differential forms with different `DGCVType` attribute values. Reformat them to all have `DGCVType` \'standard\' or \'complex\'. The descision whether or not to track complex variable systems in this computation affects the outcome, so `get_structureEquations` will not automate attribute homogenization here by design.')
    
#     if Torsion != None:
#         if len(set([j.DGCVType for j in dfList+Torsion]))>1:
#             raise TypeError('`get_structureEquations` was given differential forms with different `DGCVType` attribute values. Reformat them to all have `DGCVType` \'standard\' or \'complex\'. The descision whether or not to track complex variable systems in this computation affects the outcome, so `get_structureEquations` will not automate attribute homogenization here by design.')
    
#     # make _varSpace_type uniform 
#     if dfList[0].DGCVType=='complex':
#         if dfList[0]._varSpace_type=='real':
#             Torsion = [allToReal(j) for j in Torsion]
#             if any([j._varSpace_type!='real' for j in dfList]):
#                 dfList = [allToReal(j) for j in dfList]
#         elif dfList[0]._varSpace_type=='complex':
#             Torsion = [allToSym(j) for j in Torsion]
#             if any([j._varSpace_type!='complex' for j in dfList]):
#                 dfList = [allToSym(j) for j in dfList]

#     unraveler = dict()
#     twoForms = []
#     count = 0
#     if len(dfList)>1:
#         indexList=[(j,k) for j in range(len(dfList)) for k in range(len(dfList))]
#     else:
#         indexList=[(0,0)]
#     for index in indexList:
#         unraveler[count] = index
#         count = count+1
#         twoForms = twoForms +[dfList[index[0]]*dfList[index[1]]]

#     structureFunctions=dict()
#     for j in range(len(dfList)):
#         df=dfList[j]
#         extdf = Torsion[j]-exteriorDerivative(df)
#         fetchSol = decompose(extdf,twoForms,_hand_off = retrieve_passkey())
#         if isinstance(fetchSol,str):
#             raise ValueError(f'`structureEquations` failed. It appears the exterior derivative of {df} is not in the span of the provided 1-forms\' wedge products. The related error report from DGCV `decompose` is as follows:\n {fetchSol}')
#         else:
#             structureFunctions[j] = {unraveler[k]:fetchSol[0][k] for k in range(len(fetchSol[0]))}
                
#     connectionForms=dict()
#     for j in structureFunctions:
#         for k in range(len(dfList)):
#             omega = addDF([structureFunctions[j][l]*dfList[l[0]] for l in structureFunctions[j] if l[1]==k])
#             connectionForms[(j,k)] = omega

    

#     if torsion==None:
#         tval=None
#     else:
#         tval = Torsion

#     return structureEquations(dfList,structureFunctions, connectionForms, torsion = Torsion)


    



