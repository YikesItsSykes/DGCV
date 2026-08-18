from __future__ import annotations

from ..._aux._backends._types_and_constants import is_atomic
from ..._aux._vmf.vmf import vmf_lookup
from ..dgcv_core import differential_form_class, vector_field_class


def get_VF(*coordinates):
    out = []
    for coord in coordinates:
        info = vmf_lookup(coord, relatives=True, differential_system=True)

        if info.get("type") != "coordinate":
            if info.get("type") == "unregistered" and is_atomic(coord):
                vf = vector_field_class(coeff_dict={(coord, 1, None): 1})
                out.append(vf)
            continue

        ds = info.get("differential_system")
        if not isinstance(ds, dict):
            continue

        vf = ds.get("vf")
        if vf is None:
            continue

        out.append(vf)

    return out


def get_DF(*coordinates):
    out = []
    for coord in coordinates:
        info = vmf_lookup(coord, relatives=True, differential_system=True)

        if info.get("type") != "coordinate":
            if info.get("type") == "unregistered" and is_atomic(coord):
                df = differential_form_class(coeff_dict={(coord, 0, None): 1})
                out.append(df)
            continue

        ds = info.get("differential_system")
        if not isinstance(ds, dict):
            continue

        df = ds.get("df")
        if df is None:
            continue

        out.append(df)

    return out


def coordinate_vector_field(obj):
    vf_list = get_VF(obj)
    if len(vf_list) != 1:
        return
    return vf_list[0]


def coordinate_differential_form(obj):
    df_list = get_DF(obj)
    if len(df_list) != 1:
        return
    return df_list[0]
