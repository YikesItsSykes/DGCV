"""
DGCV: Polynomial Tools

This module integrates DGCV's complex variable handling features with SymPy's core polynomial functionality. 
It provides functions for creating and manipulating polynomials with variable coefficients registered 
in DGCV's Variable Management Framework (VMF). The polynomials created by these functions are compatible 
with the DGCVPolyClass, which wraps SymPy.Poly objects to support DGCV's framework for managing complex 
variable systems.

Key Functions:

Polynomial Creation:
    - createPolynomial(): Constructs a polynomial with specified degree and variables, supporting 
      homogeneous and weighted homogeneous options.
    - createBigradPolynomial(): Creates a polynomial that is weighted-homogeneous with respect 
      to two independent weight systems.

Term and Monomial Handling:
    - getWeightedTerms(): Extracts terms from a polynomial that satisfy specific weighted homogeneity 
      conditions.
    - monomialWeight(): Computes the weight of a monomial with respect to a given weight system.

Usage Notes:
- All functions in this module register created variables in the VMF for consistency across the DGCV framework. 
- These functions are designed to work seamlessly with DGCVPolyClass objects but can also operate on standard 
  SymPy expressions.

Dependencies:
    - sympy: Core symbolic mathematics library used for polynomial manipulation.
    - functools: Provides the `reduce` function for processing monomials.
    - operator: Supplies the `mul` operator for combining monomial components.
    - warnings: Used for issuing warnings in functions.
    - DGCV modules:
        - combinatorics: Handles combinatorial logic for generating polynomial terms.
        - classesAndVariables: Interfaces with the DGCV framework for variable management.
        - config: Provides `_cached_caller_globals` for managing global variable states.

Author: David Sykes (https://www.realandimaginary.com/dgcv/)

License:
    MIT License
"""

############## dependencies
import sympy
from sympy import prod, ln, exp, expand
from .combinatorics import *
from .classesAndVariables import *
from .config import _cached_caller_globals
from functools import reduce
from operator import mul
import warnings

############## creating polynomials (monomialsOfPoly is in classesAndVariables.py)

def createPolynomial(arg1, arg2, arg3, homogeneous=False, weightedHomogeneity=None, degreeCap=0, _tempVar=None, returnMonomialList=False, assumeReal=None, remove_guardrails=False):
    """
    Constructs a polynomial in the variables specified by *arg3* of degree *arg2*, with coefficients labeled by *arg1*.

    This function supports generating general polynomials, homogeneous polynomials, and weighted homogeneous polynomials. The polynomial's structure is determined based on the input parameters and optional arguments provided.

    Parameters
    ----------
    arg1 : str
        Prefix for the coefficient variables (e.g., 'alpha').
    arg2 : int
        Degree of the polynomial.
    arg3 : list or tuple
        A list or tuple of initialized variables over which the polynomial is defined.
    homogeneous : bool, optional
        If True, constructs a homogeneous polynomial of degree *arg2*. Default is False.
    weightedHomogeneity : list, optional
        A list of non-negative integers representing the weights for each variable. If provided, the polynomial will be weighted homogeneous with respect to these weights.
    degreeCap : int, optional
        Maximum degree for variables with zero weights in the weighted homogeneity case. Default is 0.
    _tempVar : bool, optional
        If True, the created coefficient variables will be treated as temporary variables. Default is False (for backend compatibility).
    returnMonomialList : bool, optional
        If True, returns a list of monomials instead of a summed polynomial expression. Default is False.
    assumeReal : bool, optional
        If provided, assumes the variables are real. Default is None.
    remove_guardrails : bool, optional
        If True, disables the DGCV guardrails system to bypass label protections. Default is False.

    Returns
    -------
    sympy.Expr or list
        The resulting polynomial expression if *returnMonomialList* is False, otherwise a list of monomials.

    Notes
    -----
    - If *homogeneous* is True, the function creates a homogeneous polynomial of degree *arg2* in the variables *arg3*.
    - If *weightedHomogeneity* is provided, the function creates a weighted homogeneous polynomial of degree *arg2* with respect to the weights specified. If any variable has a zero weight, the polynomial's degree in these variables is capped by *degreeCap*.
    - If both *homogeneous* and *weightedHomogeneity* are specified, only the *homogeneous* argument will be used, and *weightedHomogeneity* will be ignored.
    - Coefficient variables are created using *variableProcedure(arg1)*, and their labels will follow the prefix specified by *arg1*.

    Raises
    ------
    ValueError
        If the input parameters are inconsistent or invalid for polynomial creation.

    Examples
    --------
    >>> variableProcedure('x', 3)
    >>> createPolynomial('alpha', 2, (x1, x2, x3))
    alpha1 + alpha10*x1**2 + alpha2*x3 + alpha3*x3**2 + alpha4*x2 + alpha5*x2*x3 + alpha6*x2**2 + alpha7*x1 + alpha8*x1*x3 + alpha9*x1*x2

    >>> createPolynomial('alpha', 4, (x1, x2, x3), weightedHomogeneity=[1, 2, 3])
    alpha1*x2**2 + alpha2*x1*x3 + alpha3*x1**2*x2 + alpha4*x1**4
    """
    clearVar(arg1)
    
    var_symbols = arg3
    num_vars = len(var_symbols)
    
    # Use DGCV's custom combinatorial functions
    if homogeneous:
        indicesLoc = list(chooseOp(range(arg2+1), num_vars, restrictHomogeneity=arg2))
    elif isinstance(weightedHomogeneity, list):
        if 0 in weightedHomogeneity:
            zeroIndLoc = [j for j in range(len(weightedHomogeneity)) if weightedHomogeneity[j] == 0]
            nonZIndLoc = [j for j in range(len(weightedHomogeneity)) if weightedHomogeneity[j] != 0]
            arg3 = [arg3[j] for j in nonZIndLoc] + [arg3[j] for j in zeroIndLoc]
            shortenedWeightsLoc = [weightedHomogeneity[j] for j in nonZIndLoc] + [weightedHomogeneity[j] for j in zeroIndLoc]
            nonZIndicesLoc = list([j for j in chooseOp(range(arg2+1), len(nonZIndLoc))
                              if sum(k[0] * k[1] for k in zip(j, shortenedWeightsLoc)) == arg2])
            zeroIndicesLoc = list([j for j in chooseOp(range(degreeCap+1), len(zeroIndLoc))])
            indicesLoc = [j + k for j in nonZIndicesLoc for k in zeroIndicesLoc]
        else:
            indicesLoc = list([j for j in chooseOp(range(arg2+1), len(arg3))
                          if sum(k[0] * k[1] for k in zip(j, weightedHomogeneity)) == arg2])
    else:
        indicesLoc = list([j for j in chooseOp(range(arg2+1), len(arg3)) if sum(j) <= arg2])

    # Create coefficient variables
    variableProcedure(arg1, len(indicesLoc), _tempVar=_tempVar, assumeReal=assumeReal,remove_guardrails=remove_guardrails)
    
    # Construct the polynomial using the combinatorial monomials and coefficient variables
    monomials = []
    for idx in range(len(indicesLoc)):
        coeff = eval(arg1, _cached_caller_globals)[idx]
        monomial = reduce(mul, [var**exp for var, exp in zip(var_symbols, indicesLoc[idx])])
        monomials.append(coeff * monomial)
    
    if returnMonomialList:
        return monomials
    else:
        return sum(monomials)  

def createBigradPolynomial(arg1, arg2, arg3, arg4, arg5, _tempVar=None, returnMonomialList=False, remove_guardrails=False):
    """
    Creates a bigraded polynomial in the variables specified by *arg3* with coefficients labeled by *arg1*, 
    weighted-homogeneous of degree *arg2* in the weight systems of *arg4* and *arg5*.

    Parameters
    ----------
    arg1 : str
        Prefix for the coefficient variables (e.g., 'alpha').
    arg2 : list or tuple of length 2
        The degrees in the two weight systems. Both elements must be positive integers.
    arg3 : list or tuple
        A list or tuple of initialized variables over which the polynomial is defined.
    arg4 : list or tuple
        A list or tuple of weights for the first weight system (same length as *arg3*).
    arg5 : list or tuple
        A list or tuple of weights for the second weight system (same length as *arg3*).
    _tempVar : bool, optional
        If True, the created coefficient variables will be treated as temporary variables. Default is False.
    returnMonomialList : bool, optional
        If True, returns a list of monomials instead of a summed polynomial expression. Default is False.
    remove_guardrails : bool, optional
        If True, disables the DGCV guardrails system to bypass label protections. Default is False.

    Returns
    -------
    sympy.Expr or list
        The resulting bigraded polynomial expression if *returnMonomialList* is False, otherwise a list of monomials.

    Raises
    ------
    ValueError
        If the input parameters are inconsistent or invalid for bigraded polynomial creation.
    
    Notes
    -----
    - The polynomial is weighted-homogeneous in two weight systems, each specified by *arg4* and *arg5*. 
      The degrees for the two systems are specified by the two elements in *arg2*.
    - Coefficients are created using *variableProcedure(arg1)*, and their labels will follow the prefix specified by *arg1*.
    Examples
    --------
    >>> variableProcedure('x', 3)
    >>> createBigradPolynomial('alpha', (2, 4), (x1, x2, x3), (1, 1, 0), (0, 1, 1))
    alpha1*x2**2*x3**2 + alpha2*x1*x2*x3**3 + alpha3*x1**2*x3**4
    """
    
    # Clear any previous variables with the same label prefix
    clearVar(arg1)
    
    # Generate indices that satisfy the two weight system conditions
    indicesLoc = list([j for j in chooseOp(range(max(arg2) + 1), len(arg3))
                  if sum(k[0] * k[1] for k in zip(j, arg4)) == arg2[0]
                  and sum(k[0] * k[1] for k in zip(j, arg5)) == arg2[1]])
    
    # Create coefficient variables
    variableProcedure(arg1, len(indicesLoc), _tempVar=_tempVar, remove_guardrails=remove_guardrails)
    
    # Build the polynomial by summing monomials
    monomials = [eval(arg1, _cached_caller_globals)[k] * reduce(mul, [var**exp for var, exp in zip(arg3, indicesLoc[k])])
                 for k in range(len(indicesLoc))]
    
    if returnMonomialList:
        return monomials
    else:
        return sum(monomials)
    
from sympy import simplify, expand, ln, exp, Poly
import warnings

def monomialWeight(arg1, arg2, arg3):
    """
    Computes the weight of a nonzero monomial *arg1* in the variables *arg2* with respect to weights *arg3*.

    Parameters
    ----------
    arg1 : sympy.Expr
        A monomial expression (e.g., x**2*y).
    arg2 : list or tuple
        A list or tuple of variables in the monomial.
    arg3 : list
        A list of weights corresponding to each variable in *arg2*. Must be the same length as *arg2*.

    Returns
    -------
    sympy.Expr
        The computed weight of the monomial.

    Raises
    ------
    ValueError
        If the length of *arg2* and *arg3* do not match.

    Notes
    -----
    - A warning is issued if the input is not detected as a monomial.
    - The weight is computed by taking the logarithm of the monomial and substituting 
      each variable with an exponential weight as specified in *arg3*.
    """
    if len(arg2) != len(arg3):
        raise ValueError("The number of variables and weights must match.")
    
    # Expand and simplify the expression
    arg1 = simplify(expand(arg1))

    # Check if the expression is a monomial
    try:
        poly = Poly(arg1, *arg2)
        if len(poly.terms()) > 1:
            warnings.warn(f"Input {arg1} is not a monomial (contains multiple terms). Proceeding anyway.")
    except Exception:
        warnings.warn(f"Input {arg1} could not be interpreted as a polynomial in {arg2}. Proceeding anyway.")

    # Isolate the coefficient by setting all variables to 1
    coeffLoc = simplify(arg1.subs([(var, 1) for var in arg2]))
    
    # Compute the logarithmic weight of the monomial
    return simplify(ln(arg1 / coeffLoc).subs([(arg2[i], exp(arg3[i])) for i in range(len(arg2))]))

############## manipulating polynomials

def getWeightedTerms(arg1, arg2, arg3):
    """
    Extracts the weighted homogeneous terms from a DGCVPolyClass polynomial *arg1*, where the weights 
    are specified by the list *arg2* and the weight systems by the list *arg3*.

    Parameters
    ----------
    arg1 : DGCVPolyClass
        A DGCVPolyClass object representing the polynomial.
    arg2 : list
        A list of weights for weighted homogeneity conditions.
    arg3 : list of lists
        A list of weight systems (each corresponding to a list of weights for the variables).

    Returns
    -------
    DGCVPolyClass
        A DGCVPolyClass object containing the terms that satisfy the weighted homogeneity conditions.

    Notes
    -----
    - The function filters the terms of the polynomial in *arg1* that satisfy the weighted homogeneity 
      conditions specified by the weight systems in *arg3* with respect to the weights in *arg2*.
    - Each entry in *arg2* corresponds to a degree, and each list in *arg3* corresponds to the weight 
      system for the variables.

    Examples
    --------
    >>> from DGCV import variableProcedure, createPolynomial, getWeightedTerms, DGCVPolyClass
    >>> variableProcedure(['x'], 5)
    >>> largePoly = DGCVPolyClass(createPolynomial('alpha', 6, x), varSpace=x)
    >>> print(len(largePoly.poly_obj_unformatted.terms()))  # Output: 462

    >>> getWeightedTerms(largePoly, [12, 6, 12], [[1, 2, 3, 4, 5], [1, 1, 3, 2, 1], [2, 1, 3, 1, 4]])
    alpha438*x1**3*x2**2*x5
    """
    # Extract coefficients, monomials, and variables from the polynomial
    coeffsLoc = arg1.poly_obj_unformatted.coeffs()
    monLoc = arg1.poly_obj_unformatted.monoms()
    varLoc = arg1.poly_obj_unformatted.gens
    
    # Reconstruct the monomials
    monomials = [coeffsLoc[k] * prod([varLoc[j]**monLoc[k][j] for j in range(len(varLoc))]) for k in range(len(monLoc))]
    
    # Initialize the list of admissible indices
    admissibleIndex = range(len(monomials))
    
    # Filter terms based on the weighted homogeneity conditions
    for k in range(len(arg2)):
        admissibleIndex = [j for j in admissibleIndex if sum([monLoc[j][l] * arg3[k][l] for l in range(len(varLoc))]) == arg2[k]]
    
    # Return the filtered terms as a new DGCVPolyClass object
    return DGCVPolyClass(sum([monomials[j] for j in admissibleIndex]), arg1.varSpace)