from __future__ import annotations

from typing import Any, List, Sequence

from ..._aux._backends._symbolic_router import get_free_symbols
from ..._aux._vmf.vmf import vmf_lookup


def _coordinate_basis_from_vmf(
    coordinate_space: Sequence[Any],
    *,
    coorType: str,  # "df" or "vf"
) -> List[Any]:
    out: List[Any] = []
    for coordinate in coordinate_space:
        info = vmf_lookup(coordinate, differential_system=True)
        ds = info.get("differential_system")
        if not isinstance(ds, dict):
            raise TypeError(
                "`coordinate_space` must consist of VMF-registered coordinates with a differential system."
            )
        obj = ds.get(coorType)
        if obj is None:
            raise TypeError(
                "`coordinate_space` must consist of VMF-registered coordinates with a differential system."
            )
        out.append(obj)
    return out


def _infer_coordinate_space_from_objs(objs: Sequence[Any]) -> List[Any]:
    syms: set[Any] = set()
    for o in objs:
        fs = get_free_symbols(o)
        if fs:
            params = getattr(o, "parameters", None)
            if params:
                fs = set(fs) - set(params)
            syms |= set(fs)

    coordinates: List[Any] = []
    for a in syms:
        info = vmf_lookup(a, differential_system=True)
        ds = info.get("differential_system")
        if isinstance(ds, dict) and (
            ds.get("vf") is not None or ds.get("df") is not None
        ):
            coordinates.append(a)

    coordinates.sort(key=lambda x: str(x))
    return coordinates
