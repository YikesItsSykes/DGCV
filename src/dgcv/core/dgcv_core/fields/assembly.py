from __future__ import annotations

from typing import Literal, Optional

from ...._aux._vmf.vmf import vmf_lookup
from .differential_forms import differential_form_class
from .general import tensor_field_class
from .vector_fields import vector_field_class


def assemble_tensor_field(
    coordinate_space: tuple | list,
    coefficient_dict: dict,
    valence: Optional[tuple | list] = None,
    shape: Optional[Literal["general", "symmetric", "skew"]] = "general",
    subclass: Optional[Literal["vector field", "differential form"]] = None,
) -> tensor_field_class:
    """
    Assemble a dgcv tensor field from coordinate-space indices and coefficients.

    Parameters
    ----------
    coordinate_space : tuple or list
        Sequence of variables that defines the coordinate basis. Every element
        must be registered in the dgcv VMF (tip: use createVariables to
        register coordinates).

    coefficient_dict : dict
        Dictionary mapping tuples of integer indices to coefficient values.
        Each key is interpreted as a list of slots selecting elements of
        `coordinate_space` by index. Values are scalar coefficients.

    valence : tuple or list, optional
        Valence specification aligned with `coordinate_space`. If None, defaults
        to a tuple of zeros of the same length as `coordinate_space`.

    shape : {"general","symmetric","skew"}, default "general"
        Declared symmetry for the resulting tensor field.

    subclass : {"vector field","differential form"}, optional
        If provided, returns an instance of the requested subclass. If None,
        returns a `tensor_field_class`.

    Returns
    -------
    tensor_field_class
    """
    return_class = (
        vector_field_class
        if subclass == "vector field"
        else differential_form_class
        if subclass == "differential form"
        else tensor_field_class
    )
    if valence is None:
        valence = tuple(0 for _ in range(len(coordinate_space)))
    if len(valence) != len(coordinate_space):
        raise ValueError(
            "`assemble_tensor_field` expects valence list if given to match the coordinate_space list length."
        )
    variable_spaces, coordinates = dict(), dict()
    for var in coordinate_space:
        data = vmf_lookup(var, path=True, flattened_relatives=True, system_index=True)
        path = data.get("path")
        syslabel = path[1] if path else None
        if syslabel is None:
            raise LookupError(
                "`assemble_tensor_field` requires variables in the given coordinate_space to be registered in the dgcv VMF."
            )
        if syslabel not in variable_spaces:
            variable_spaces[syslabel] = data.get("flattened_relatives")
        idx = data.get("system_index")
        coordinates[var] = {"idx": idx, "sysl": syslabel}

    def new_key(key):
        f, m, L = [], [], []
        for indx in key:
            try:
                cvar, val = coordinate_space[indx], valence[indx]
            except Exception:
                raise ValueError(
                    "`assemble_tensor_field` expects cooeficient dict keys to be tuples with indices in range of the coordinate_space list length."
                )
            f.append(coordinates[cvar]["idx"])
            m.append(val)
            L.append(coordinates[cvar]["sysl"])
        return tuple(f + m + L)

    cd = {new_key(k): v for k, v in coefficient_dict.items()}
    return return_class(coeff_dict=cd, data_shape=shape)
