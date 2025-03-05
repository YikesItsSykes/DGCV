import random
import string
from itertools import combinations
from math import prod  # requires python >=3.8

import sympy as sp

from .config import _cached_caller_globals
from .eds import abstDFAtom, abstDFMonom, abstract_DF, abstract_ZF, zeroFormAtom


def solve_carefully(eqns, vars_to_solve, dict=True):
    """
    Recursively applies sympy.solve() to handle underdetermined systems.
    If solve() fails due to "no valid subset found", it tries solving for smaller subsets of variables.

    Parameters:
    - eqns: list of sympy equations
    - vars_to_solve: list/tuple of variables to solve for
    - dict: whether to return solutions as a dictionary (default: True)

    Returns:
    - Solution from sympy.solve() if found
    - Otherwise, tries smaller variable subsets recursively
    - Raises NotImplementedError if no subset can be solved
    """

    try:
        # First, try to solve normally
        sol = sp.solve(eqns, vars_to_solve, dict=dict)
        if sol:  # Return only if non-empty
            return sol
    except NotImplementedError as e:
        if "no valid subset found" not in str(e):
            raise  # Re-raise other errors

    # If solve() fails, or returned an empty solution, try smaller subsets of variables
    num_vars = len(vars_to_solve)

    if num_vars == 1:
        raise NotImplementedError("No valid subset found, even at minimal variable count.")

    # Try subsets with one fewer variable
    subset_list = list(combinations(vars_to_solve, num_vars - 1))
    for i, subset in enumerate(subset_list):
        try:
            sol = solve_carefully(eqns, subset, dict=dict)
            if sol or i == len(subset_list) - 1:  # Only return if non-empty or last subset
                return sol
        except NotImplementedError:
            continue  # Try the next subset

    # If no subset worked, raise the error
    raise NotImplementedError(f"No valid subset found for variables {vars_to_solve}")

def DGCV_solve(eqns, vars_to_solve):
    processed_eqns, system_vars, extra_vars, variables_dict = _equations_preprocessing(eqns,vars_to_solve)
    solutions = sp.solve(processed_eqns, system_vars, dict=True)
    solutions_formatted = []
    for solution in solutions:
        solutions_formatted += [{variables_dict.get(var,var):solution[var] for var in solution}]

def _generate_str_id(base_str: str, *dicts: dict) -> str:
    """
    Generates a unique identifier based on base_str.
    Filters against the provided dictionaries to make sure the generated str is not in them.
    """
    candidate = base_str
    while any(candidate in d for d in dicts):
        random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        candidate = f"{base_str}_{random_suffix}"

    return candidate

def _equations_preprocessing(eqns:tuple|list,vars:tuple|list):
    processed_eqns = []
    variables_dict = dict()
    for eqn in eqns:
        eqn_formatted, new_var_dict = _equation_formatting(eqn,variables_dict)
        processed_eqns += eqn_formatted
        variables_dict = variables_dict | new_var_dict
    subbedValues = {variables_dict[k][0]:variables_dict[k][1] for k in variables_dict}
    pre_system_vars = [subbedValues[var] if var in subbedValues else var for var in vars]
    system_vars = []
    extra_vars = []
    for var in pre_system_vars:
        if isinstance(var,sp.Symbol):
            system_vars += [var]
        else:
            extra_vars += [var]
    return processed_eqns, system_vars, extra_vars, variables_dict

def _equation_formatting(eqn,variables_dict):
    var_dict = dict()
    if isinstance(eqn,(sp.Expr,int,float)):
         return [sp.sympify(eqn)], var_dict
    elif isinstance(eqn,zeroFormAtom):
        candidate_str = eqn.__str__()
        if candidate_str in variables_dict and variables_dict[candidate_str][0]==eqn:
            identifier = candidate_str
            eqn_formatted = variables_dict[candidate_str][1]
            # nothing new to add to var_dict here.
        else:
            identifier = _generate_str_id(candidate_str,variables_dict,_cached_caller_globals)
            eqn_formatted =  [sp.symbols(identifier)]   # The single variable is the equation
            var_dict[identifier] = (eqn,eqn_formatted)  # string label --> (original, formatted)
        return eqn_formatted,var_dict
    elif isinstance(eqn,abstract_ZF):
        eqn_formatted,var_dict= _sympify_abst_ZF(eqn_formatted,variables_dict)
        return eqn_formatted, var_dict
    elif isinstance(eqn,abstDFAtom):
        eqn_formatted,var_dict = _equation_formatting(eqn._coeff,variables_dict)
        return eqn_formatted, var_dict
    elif isinstance(eqn,abstDFMonom):
        eqn_formatted,var_dict = _equation_formatting(eqn._coeff,variables_dict)
        return eqn_formatted, var_dict
    elif isinstance(eqn,abstract_DF):
        terms = []
        var_dict = dict()
        for term in eqn.terms:
            new_term,new_var_dict = _equation_formatting(term,variables_dict|var_dict)
            var_dict = var_dict|new_var_dict
            terms += new_term
        return terms, var_dict

def _sympify_abst_ZF(zf:abstract_ZF, varDict):
    if isinstance(zf.base,(int,float,sp.Expr)):
        return [zf.base], varDict
    if isinstance(zf.base,zeroFormAtom):
        return _equation_formatting(zf.base,varDict)
    if isinstance(zf.base,tuple):
        op, args = zf.base
        new_args = []
        constructedVarDict = varDict
        for arg in args:
            new_arg, new_dict = _sympify_abst_ZF(arg, constructedVarDict)
            new_args += new_arg
            constructedVarDict |= new_dict
        if op == 'mul':
            zf_formatted = [prod(new_args)]
        if op == 'add':
            zf_formatted = [sum(new_args)]
        if op == 'pow':
            zf_formatted = [new_args[0]**new_args[1]]
        if op == 'sub':
            zf_formatted = [new_args[0]-new_args[1]]
        if op == 'div':
            zf_formatted = [sp.Rational(new_args[0],new_args[1])]
        return zf_formatted, constructedVarDict




