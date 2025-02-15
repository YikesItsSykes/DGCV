import random
import string
from itertools import combinations

import sympy as sp

from .config import _cached_caller_globals
from .eds import (
    abstDFAtom,
    abstDFMonom,
    abstract_DF,
    zeroFormAtom,
)


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
        processed_eqns += [eqn_formatted]
        variables_dict = variables_dict | new_var_dict
    return processed_eqns, variables_dict

def _equation_formatting(eqn,variables_dict):
    var_dict = dict()
    if isinstance(eqn,(sp.Expr,int,float)):
         return sp.sympify(eqn), var_dict
    elif isinstance(eqn,zeroFormAtom):
        identifier = _generate_str_id(eqn.__str__(),variables_dict,_cached_caller_globals)
        eqn_formatted =  sp.symbols(identifier)
        var_dict[identifier] = (eqn,eqn_formatted)
        return eqn_formatted,var_dict
    elif isinstance(eqn,abstDFAtom):
        eqn_formatted,var_dict = _equation_formatting(eqn.coeff,variables_dict)
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
        eqn_formatted = sum(terms)
        return eqn_formatted, var_dict



