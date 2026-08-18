from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from ..._aux._backends._types_and_constants import symbol
from ..._aux._vmf._safeguards import query_dgcv_categories
from ..dgcv_core import variableProcedure


def _local_unknowns(n: int, lbl: str = "__tem_label__") -> List[Any]:
    return [symbol(f"{lbl}{i}") for i in range(n)]


def _make_parameters(
    n: int,
    *,
    register_parameters: bool,
    parameters_label: Optional[str],
) -> Tuple[Any, ...]:
    if n <= 0:
        return tuple()

    if register_parameters:
        prefix = parameters_label if isinstance(parameters_label, str) else "param"
        params = variableProcedure(prefix, n, return_created_object=True)[0]
        return tuple(params)

    prefix = parameters_label if isinstance(parameters_label, str) else "p"
    return tuple(symbol(f"{prefix}{i}") for i in range(n))


def _require_subcategory(objs: Sequence[Any], cats: set[str], who: str) -> None:
    for o in objs:
        if not query_dgcv_categories(o, cats):
            raise TypeError(
                f"`{who}` expects dgcv objects in categories {sorted(cats)}."
            )
