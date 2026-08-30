from __future__ import annotations

from .._aux._utilities._config import dgcv_warning
from ._traversal import walk


def label_from_path(path, prefix):
    parts = [p.lstrip("_") for p in str(path).split(".")]
    body = "_".join(p for p in parts if p)
    head = str(prefix) if prefix else ""
    out = f"{head}_{body}" if head and body else (head or body)
    if not out:
        return "unlabeled"
    return "_" + out if out[0].isdigit() else out


def selector(select):
    if select is None:
        return lambda node: True
    if callable(select):
        return select
    text = str(select).lower()
    if text in ("complete", "solved", "closed"):
        return lambda node: node.status == "complete"
    if text in ("incomplete", "unsolved", "open"):
        return lambda node: node.status == "incomplete"
    raise ValueError(
        "`select` should be None, a callable predicate, or one of the strings "
        "'complete' and 'incomplete'."
    )


def add_cases_from(
    target,
    source,
    select=None,
    prefix="",
    label_rule=None,
    transfer_open=True,
    include_interior=False,
    return_created_object=False,
    **kwargs,
):
    if hasattr(source, "_subcases"):
        items = walk(source, include_interior=include_interior)
    elif isinstance(source, dict):
        items = list(source.items())
    else:
        items = [(getattr(n, "label", str(i)), n) for i, n in enumerate(source)]

    keep = selector(select)
    made = {}
    for path, node in items:
        if not keep(node):
            continue
        if callable(label_rule):
            name = label_rule(path, node)
        else:
            name = label_from_path(path, prefix)
        known = target.system_parameters | target.system_variables
        closed, hidden = {}, {}
        for e, v in node.closed_case_rules.items():
            (closed if e in known else hidden)[e] = v
        conditions = {"closed": closed}
        if transfer_open:
            conditions["open"] = {e: set(v) for e, v in node.open_case_rules.items()}
        made[name] = target.add_case(
            name,
            defining_conditions=conditions,
            return_created_object=True,
            **{
                "source_case": node,
                "_warn_open": False,
                "_inherited_constraints": hidden,
                **kwargs,
            },
        )
    if not made:
        dgcv_warning(
            "No branches in the given source matched, so no subcases were added.",
            wc_label="dgcvOperationsNote",
        )
    return made if return_created_object else None


def seeded_from(
    cls,
    source,
    select=None,
    prefix="",
    label_rule=None,
    transfer_open=True,
    include_interior=False,
    **kwargs,
):
    root = cls(
        **{
            "variables": source.system_variables,
            "parameters": source.system_parameters,
            **kwargs,
        }
    )
    add_cases_from(
        root,
        source,
        select=select,
        prefix=prefix,
        label_rule=label_rule,
        transfer_open=transfer_open,
        include_interior=include_interior,
    )
    return root
