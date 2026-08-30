from __future__ import annotations

from ..._aux._vmf._safeguards import check_dgcv_category
from ..._aux.printing.printing._dgcv_display import (
    LaTeX_eqn_system,
    LaTeX_list,
    show,
)
from .._tasks import result_message
from ._conditions_latex import conditions_str, render_value


def case_summary(node, plain_text=False):
    f = getattr(node, "summary", None)
    if callable(f):
        return f(node)
    param_count = len(node.system_parameters)
    purality_note = "parameter" if param_count == 1 else "parameters"
    print(f"The system is parameterized with {param_count} {purality_note}.")

    profile_mode = node.profile is not None
    if not profile_mode:
        eqns = node.reduced_equation_system
        if check_dgcv_category(eqns):
            eqns = {0: eqns}
        else:
            eqns = [var for var in eqns]

    param_bucket = node.parameter_conditions
    var_bucket = node.variable_constraints
    p_count = len(param_bucket["closed"]) + sum(
        len(s) for s in param_bucket["open"].values()
    )
    v_count = len(var_bucket["closed"]) + sum(
        len(s) for s in var_bucket["open"].values()
    )

    def _emit(bucket):
        if plain_text:
            print(conditions_str(bucket, plain_text=True))
        else:
            show(conditions_str(bucket, punctuation=","))

    if p_count == 0 and v_count == 0:
        if not profile_mode:
            print("Equation system:")
    else:
        if p_count > 0:
            print(
                "Restricting to the subfamily defined by the parameter condition"
                + ("" if p_count == 1 else "s")
            )
            _emit(param_bucket)
        if v_count > 0:
            print(
                ("with" if p_count > 0 else "Imposing")
                + " the variable constraint"
                + ("" if v_count == 1 else "s")
            )
            _emit(var_bucket)
        if not profile_mode:
            print("Reduced equations system:")

    if profile_mode:
        print_profile_report(node, plain_text=plain_text)
    elif plain_text:
        print(eqns)
    else:
        show(
            LaTeX_eqn_system(
                eqns,
                punctuation=".",
            )
        )
    if param_count > 0:
        fp = node.free_parameters
        if len(fp) == 0:
            print("No free parameters remain in this branch.")
        else:
            print("The remaining free parameters in this branch are")
            if plain_text is True:
                print(fp)
            else:
                show(LaTeX_list(fp, one_line=True, punctuation="."))
    if len(node.system_variables) > 0:
        fv = node.free_variables
        if len(fv) == 0:
            print("No free variables remain in this branch.")
        else:
            print("The remaining free variables in this branch are")
            if plain_text is True:
                print(fv)
            else:
                show(LaTeX_list(fv, one_line=True, punctuation="."))
    if not profile_mode and node.complete:
        print("********** The branch is complete! **********")
        if param_count > 0:
            print(f"     {len(node.free_parameters)} dim. parameter space remaining ")

def print_profile_report(node, plain_text=False):
    err = node.error
    if err is not None:
        print(f"The branch profile raised {type(err).__name__}: {err}")
        return
    fields = node.result_fields
    st = node.status
    header = "Properties profile:" if st is None else f"Properties profile: {st}"
    if not fields:
        message = result_message(node)
        if message:
            print(header)
            print(f"    {message}")
        elif st is not None:
            print(header)
        elif node.result is not None:
            print(header)
            print(node.result)
        return
    print(header)
    for k, v in fields.items():
        if plain_text:
            print(f"    {k} = {v}")
        else:
            rendered = render_value(v, True)
            show(f"{k} = {rendered}" if rendered.startswith("$") else f"{k} = {v}")
