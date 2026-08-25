from __future__ import annotations

from ..._aux._backends._engine import engine_kind
from ..._aux._backends._symbolic_router import get_free_symbols
from ..._aux._backends._types_and_constants import expr_numeric_types, is_atomic
from ...eds.eds import _equation_formatting, zeroFormAtom
from ...eds.eds_representations import DF_representation


def normalize_equations_and_vars(eqns, vars_to_solve):
    if isinstance(eqns, DF_representation):
        eqns = eqns.flatten()
    if not isinstance(eqns, (list, tuple)):
        eqns = [eqns]
    formatted = []
    for eqn in eqns:
        neqn = getattr(eqn, "__dgcv_zero_obstr__", None)
        if neqn is None:
            formatted.append(eqn)
        else:
            try:
                formatted += list(neqn[0])
            except Exception as solve_exception:
                raise RuntimeError(
                    "Equations data provided to `solve_dgcv` were in an unsupported format."
                ) from solve_exception
    eqns = formatted

    if vars_to_solve is None:
        vars_to_solve = set()
        for eqn in eqns:
            try:
                vars_to_solve |= set(get_free_symbols(eqn))
            except Exception:
                pass

    if isinstance(vars_to_solve, set):
        vars_to_solve = list(vars_to_solve)
    if not isinstance(vars_to_solve, (list, tuple)):
        vars_to_solve = [vars_to_solve]
    return eqns, vars_to_solve


def _equations_preprocessing(eqns: tuple | list, vars: tuple | list):
    processed_eqns = []
    variables_dict = dict()
    native_passthrough = engine_kind() == "sage"
    for eqn in eqns:
        if native_passthrough and (
            isinstance(eqn, expr_numeric_types()) and not isinstance(eqn, zeroFormAtom)
        ):
            processed_eqns.append(eqn)
            continue
        eqn_formatted, new_var_dict = _equation_formatting(eqn, variables_dict)
        processed_eqns += eqn_formatted
        variables_dict = variables_dict | new_var_dict

    subbedValues = {variables_dict[k][0]: variables_dict[k][1] for k in variables_dict}
    pre_system_vars = [
        subbedValues[var] if var in subbedValues else var for var in vars
    ]

    system_vars = []
    extra_vars = []
    for var in pre_system_vars:
        if isinstance(var, (list, tuple)) and len(var) == 1:
            var = var[0]
        if is_atomic(var):
            system_vars += [var]
        else:
            extra_vars += [var]
    return processed_eqns, system_vars, extra_vars, variables_dict
