from __future__ import annotations

from ..._aux._utilities._config import get_dgcv_settings_registry, latex_in_html
from ..._aux._utilities._styles import get_style
from ..._aux._vmf.vmf import order_coordinates
from ..._aux.printing._tables import build_plain_table
from ..._aux.printing.printing._dgcv_display import LaTeX_eqn_system, show
from .._traversal import walk
from ._conditions_latex import render_value, represents_strings


def tree_leaves_html(
    tree,
    theme=None,
    use_latex=True,
    return_displayable=False,
    sort_by: str | list = None,
    reverse=False,
    hide_variable_constraints: bool = False,
    **kwargs,
):
    if not isinstance(theme, str):
        theme = get_dgcv_settings_registry().get("theme", "dark")
    leaves = walk(tree)

    def process_conditions(conds):
        closed = conds.get("closed", {})
        open_ = conds.get("open", {})
        eqns = []
        for e in order_coordinates(list(closed)):
            v = closed[e]
            eqns.append(
                LaTeX_eqn_system({e: v}, math_mode="$") if use_latex else f"{e} = {v}"
            )
        for e in order_coordinates(list(open_)):
            for v in sorted(open_[e], key=str):
                eqns.append(
                    LaTeX_eqn_system({e: v}, relation=r" \neq ", math_mode="$")
                    if use_latex
                    else f"{e} \u2260 {v}"
                )
        return ", ".join(eqns)

    profile_mode = tree.profile is not None
    state_header = "status" if profile_mode else "equation state"

    def state(node):
        st = node.status
        if profile_mode:
            return st if st else ""
        return "solved" if st == "complete" else "unsolved"

    no_params = len(tree.free_parameters) == 0
    show_params = not no_params
    show_state = not profile_mode or any(node.status is not None for _, node in leaves)
    show_free_vars = len(tree.system_variables) > 0
    show_represents = any(node.is_sampled for _, node in leaves)
    show_var_constraints = not hide_variable_constraints and any(
        node.variable_constraints["closed"] or node.variable_constraints["open"]
        for _, node in leaves
    )

    declared = tree.report_fields
    if declared:
        field_names = list(declared)
    else:
        field_names = []
        for _, node in leaves:
            for key in node.result_fields:
                if key not in field_names:
                    field_names.append(key)

    headers = ["subcase"]
    if show_state:
        headers.append(state_header)
    headers.extend(field_names)
    if show_params:
        headers.append("free parameters")
    if show_free_vars:
        headers.append("free variables")
    if show_params:
        headers.append("parameter conditions")
    if show_represents:
        headers.append("further conditions")
    if show_var_constraints:
        headers.append("variable constraints")

    rows = []
    keys = []

    def cell(row, key, text, value=None):
        row.append(text)
        key.append((0, text if value is None else value))

    for k, node in leaves:
        row, key = [], []
        cell(row, key, (k[1:] if k.startswith(".") else k).replace("._", "."))
        if show_state:
            cell(row, key, state(node))
        fields = node.result_fields
        for name in field_names:
            if name in fields:
                cell(row, key, render_value(fields[name], use_latex), fields[name])
            else:
                row.append("")
                key.append((1, None))
        if show_params:
            cell(row, key, str(len(node.free_parameters)), len(node.free_parameters))
        if show_free_vars:
            cell(row, key, str(len(node.free_variables)), len(node.free_variables))
        if show_params:
            cell(row, key, process_conditions(node.parameter_conditions))
        if show_represents:
            cell(row, key, ", ".join(represents_strings(node, use_latex)))
        if show_var_constraints:
            cell(row, key, process_conditions(node.variable_constraints))
        rows.append(row)
        keys.append(key)

    def sort(rs, property):
        aliases = {
            "case": "subcase",
            "cases": "subcase",
            "subcases": "subcase",
            "label": "subcase",
            "state": state_header,
            "status": state_header,
            "equation state": state_header,
            "equation states": state_header,
            "states": state_header,
            "variable": "free variables",
            "variables": "free variables",
            "var": "free variables",
            "vars": "free variables",
            "par": "free parameters",
            "pars": "free parameters",
            "parameters": "free parameters",
            "parameter": "free parameters",
            "param": "free parameters",
            "params": "free parameters",
            "case_rules": "variable constraints",
            "conditions": "variable constraints",
            "constraint": "variable constraints",
            "constraints": "variable constraints",
            "variable constraint": "variable constraints",
            "var constraints": "variable constraints",
            "var constraint": "variable constraints",
            "parameter condition": "parameter conditions",
            "param conditions": "parameter conditions",
            "param condition": "parameter conditions",
            "represents": "further conditions",
        }
        idxs = {h: i for i, h in enumerate(headers)}
        tuple_sort = isinstance(property, (list, tuple))
        if tuple_sort:
            idx = [idxs.get(aliases.get(prop, prop), None) for prop in property]
        else:
            idx = [idxs.get(aliases.get(property, property), None)]
        if any(i is None for i in idx):
            if reverse:
                return rs[-1::-1]
            return rs

        def by_value(pair):
            return tuple(pair[1][i] for i in idx)

        def by_text(pair):
            return tuple((0, str(pair[0][i])) for i in idx)

        try:
            return sorted(rs, key=by_value, reverse=reverse)
        except Exception:
            # values that cannot be compared - symbolic entries, or a column
            # mixing types - fall back to the order of their rendered text
            return sorted(rs, key=by_text, reverse=reverse)

    rows = [row for row, _ in sort(list(zip(rows, keys)), sort_by)]

    label = getattr(tree, "label", None)
    caption = (
        f"leaf summary for the tree: {label}"
        if label and str(label) != "root"
        else "case tree leaf summary"
    )

    table = build_plain_table(
        columns=headers,
        rows=rows,
        theme_css_vars=get_style(theme, legacy=False),
        caption=caption,
        table_attrs='style="table-layout:auto;"',
        container_id="tree-leaves-summary",
    )
    extra_support_for_math_in_tables = bool(
        get_dgcv_settings_registry().get("extra_support_for_math_in_tables") is True
    )
    out = latex_in_html(
        table, extra_support_for_math_in_tables=extra_support_for_math_in_tables
    )
    if return_displayable:
        return out
    show(out)
