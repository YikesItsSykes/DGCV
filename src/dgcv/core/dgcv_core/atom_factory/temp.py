from __future__ import annotations

from numbers import Integral

from ...._aux._vmf._safeguards import create_key, retrieve_passkey
from .atoms import variableProcedure
from .creator import createVariables


def temporaryVariables(
    variable_label: str = None,
    number_of_variables=None,
    initialIndex=1,
    multiindex_shape=None,
    return_created_object=True,
    register_in_vmf: bool = False,
    remove_guardrails=None,
    assumptions: dict = None,
    targeted_assumptions: dict = None,
):
    if isinstance(variable_label, Integral) and number_of_variables is None:
        variable_label, number_of_variables = None, variable_label
    if not isinstance(variable_label, str):
        variable_label = create_key(
            "tvar", avoid_caller_globals=register_in_vmf, key_length=6
        )
    if register_in_vmf:
        newObj = createVariables(
            variable_label=variable_label,
            number_of_variables=number_of_variables,
            initialIndex=initialIndex,
            multiindex_shape=multiindex_shape,
            return_created_object=return_created_object,
            temporary_variables=True,
            remove_guardrails=remove_guardrails,
            assumptions=assumptions,
            targeted_assumptions=targeted_assumptions,
        )
        if isinstance(newObj, (list, tuple)) and len(newObj) == 1:
            newObj = newObj[0]
        return newObj
    newObj = variableProcedure(
        variables_label=variable_label,
        number_of_variables=number_of_variables,
        initialIndex=initialIndex,
        multiindex_shape=multiindex_shape,
        return_created_object=return_created_object,
        _tempVar=retrieve_passkey(),
        remove_guardrails=remove_guardrails,
        assumptions=assumptions,
        targeted_assumptions=targeted_assumptions,
    )
    if isinstance(newObj, (list, tuple)) and len(newObj) == 1:
        newObj = newObj[0]
    return newObj
