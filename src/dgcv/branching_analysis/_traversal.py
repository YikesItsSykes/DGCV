from __future__ import annotations

from ._tasks import leaf_annotation


def walk(node, include_interior=False, _path=""):
    out = []
    if node._subcases:
        if include_interior and _path:
            out.append((_path, node))
        for label, sc in node._subcases.items():
            out.extend(walk(sc, include_interior, f"{_path}.{label}"))
    elif _path:
        out.append((_path, node))
    return out


def tree_dict(node):
    out = {node.label: {}}
    if node._subcases:
        for _, subtree in node._subcases.items():
            out[node.label] |= tree_dict(subtree)
    else:
        out[node.label] = leaf_annotation(node)
    return out


def count_leaves(value):
    if not isinstance(value, dict):
        return 1
    descendants = value.get("descendants")
    if not isinstance(descendants, dict):
        return 1
    return sum(count_leaves(v) for v in descendants.values())
