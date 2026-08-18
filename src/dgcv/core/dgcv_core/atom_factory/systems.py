from numbers import Integral

from ...._aux._backends._engine import engine_kind
from ...._aux._backends._types_and_constants import symbol
from ...._aux._utilities._config import (
    dgcv_warning,
    get_variable_registry,
    update_globals,
    update_globals_k_v,
)
from ...._aux._vmf._safeguards import retrieve_passkey, validate_label
from ...._aux._vmf.vmf import clearVar
from ...combinatorics.combinatorics import build_nd_array, carProd
from ..fields import differential_form_class, vector_field_class
from .atoms import indexing_places


def varWithVF(
    variables_label,
    number_of_variables=None,
    initialIndex=1,
    multiindex_shape=None,
    index_placement=None,
    _doNotUpdateVar=False,
    _calledFromCVP=None,
    _calledFromFactory=None,
    remove_guardrails=None,
    return_created_object=None,
    assumptions: dict = None,
    targeted_assumptions: dict = None,
    **kwargs,
):
    """
    Initializes one or more coordinate systems with accompanying vector fields and differential 1-forms.
    """

    variable_registry = get_variable_registry()
    passkey = retrieve_passkey()

    if multiindex_shape is not None and number_of_variables is not None:
        multiindex_shape = None
        dgcv_warning(
            "Provide at most one of `number_of_variables` and `multiindex_shape`. The given `multiindex_shape` was ignored."
        )

    labels = (
        [variables_label] if isinstance(variables_label, str) else list(variables_label)
    )
    rco = [] if return_created_object is True else None

    kind = engine_kind()
    enforce_real = kind is not None and kind != "sympy"
    enforced_real_dict = (
        variable_registry.get("dgcv_enforced_real_atoms", None)
        if enforce_real
        else None
    )

    index_placement = (
        []
        if index_placement is None
        else [indexing_places.get(place, "_") for place in index_placement]
    )

    def _valid_multiindex_shape(ms):
        return isinstance(ms, (list, tuple)) and all(
            isinstance(n, Integral) and n > 0 for n in ms
        )

    def _indices_for_multiindex(ms):
        return list(carProd(*[range(initialIndex, initialIndex + n) for n in ms]))

    for raw_label in labels:
        labelLoc = (
            validate_label(raw_label, remove_guardrails=remove_guardrails)
            if (not _calledFromCVP == passkey)
            else raw_label
        )

        if _doNotUpdateVar != passkey:
            clearVar(labelLoc, report=False)

        if number_of_variables is None and _valid_multiindex_shape(multiindex_shape):
            if len(multiindex_shape) > len(index_placement):
                index_placement += ["_"] * (
                    len(multiindex_shape) - len(index_placement)
                )
            idxs = _indices_for_multiindex(multiindex_shape)
            var_names = [
                f"{labelLoc}{''.join([a + str(b) for a, b in zip(index_placement, idx)])}"
                for idx in idxs
            ]
            if targeted_assumptions:
                merger = assumptions if assumptions else dict()
                vars = tuple(
                    symbol(
                        name,
                        assumptions=merger | targeted_assumptions.get(name, merger),
                    )
                    for name in var_names
                )
            elif assumptions:
                vars = tuple(
                    symbol(name, assumptions=assumptions) for name in var_names
                )
            else:
                vars = tuple(symbol(name) for name in var_names)
            base_vars = tuple(vars)
            var_values = build_nd_array(base_vars, multiindex_shape)
            if _doNotUpdateVar != passkey or (
                _calledFromCVP == passkey or _calledFromFactory == passkey
            ):  # CVP doesn't manage sub-parant names
                clearVar(*vars, report=False)
                update_globals(dict(zip(var_names, vars)))
                update_globals_k_v(labelLoc, var_values)

            if enforce_real and enforced_real_dict is not None:
                if (
                    assumptions.get("real", False)
                    or assumptions.get("nonnegative")
                    or assumptions.get("positive", False)
                ):
                    for v in vars:
                        enforced_real_dict[v.conjugate()] = v
                elif targeted_assumptions:
                    for v in vars:
                        t_a = targeted_assumptions.get(str(v), None)
                        if (
                            t_a
                            and t_a.get("real", False)
                            or t_a.get("nonnegative")
                            or t_a.get("positive", False)
                        ):
                            enforced_real_dict[v.conjugate()] = v

            N = len(base_vars)
            vf_instances = [
                vector_field_class(
                    coeff_dict={(j, 1, labelLoc): 1},
                    dgcvType="standard",
                    variable_spaces={labelLoc: base_vars},
                )
                for j in range(N)
            ]
            df_instances = [
                differential_form_class(
                    coeff_dict={(j, 0, labelLoc): 1},
                    dgcvType="standard",
                    variable_spaces={labelLoc: base_vars},
                )
                # DFClass(base_vars, {(j,): 1}, 1)
                for j in range(N)
            ]

            if _doNotUpdateVar != passkey:
                update_globals(dict(zip([f"D_{vn}" for vn in var_names], vf_instances)))
                update_globals(dict(zip([f"d_{vn}" for vn in var_names], df_instances)))

            def var_v_a(elem):
                if targeted_assumptions:
                    t_a = targeted_assumptions.get(elem, None)
                    return t_a if t_a else assumptions
                return assumptions

            if not (_calledFromCVP == passkey or _calledFromFactory == passkey):
                variable_registry["standard_variable_systems"][labelLoc] = {
                    "family_type": "multi_index",
                    "family_shape": tuple(multiindex_shape),
                    "family_names": tuple(var_names),
                    "family_values": var_values,
                    "differential_system": True,
                    "tempVar": None,
                    "obsVar": None,
                    "initial_index": initialIndex,
                    "variable_relatives": {
                        var_name: {
                            "VFClass": vf_instances[i],
                            "DFClass": df_instances[i],
                            "assumptions": var_v_a(var_name),
                            "system_index": i,
                        }
                        for i, var_name in enumerate(var_names)
                    },
                }
                variable_registry["_labels"][labelLoc] = {
                    "path": ("standard_variable_systems", labelLoc),
                    "children": set(
                        var_names
                        + [f"D_{v}" for v in var_names]
                        + [f"d_{v}" for v in var_names]
                    ),
                }

                paths = variable_registry.get("paths", None)
                if paths is not None:
                    base_vals = ("standard_variable_systems", labelLoc, "family_values")
                    for idx, var_name in zip(idxs, var_names):
                        offs = tuple(int(k - initialIndex) for k in idx)
                        paths[var_name] = {
                            "kind": "standard_variable",
                            "path": base_vals + offs,
                        }

        elif number_of_variables is None:
            if targeted_assumptions:
                merger = assumptions if assumptions else dict()
                sym = symbol(
                    labelLoc,
                    assumptions=merger | targeted_assumptions.get(labelLoc, merger),
                )
            elif assumptions:
                sym = symbol(labelLoc, assumptions=assumptions)
            else:
                sym = symbol(labelLoc)
            var_values = (sym,)
            update_globals_k_v(labelLoc, sym)

            if enforce_real and enforced_real_dict is not None:
                if (
                    assumptions.get("real", False)
                    or assumptions.get("nonnegative")
                    or assumptions.get("positive", False)
                ):
                    enforced_real_dict[sym.conjugate()] = sym
                elif targeted_assumptions:
                    t_a = targeted_assumptions.get(labelLoc, None)
                    if (
                        t_a
                        and t_a.get("real", False)
                        or t_a.get("nonnegative")
                        or t_a.get("positive", False)
                    ):
                        enforced_real_dict[sym.conjugate()] = sym

            vf_instance = vector_field_class(
                coeff_dict={(0, 1, labelLoc): 1},
                dgcvType="standard",
                variable_spaces={labelLoc: (sym,)},
            )
            df_instance = differential_form_class(
                coeff_dict={(0, 0, labelLoc): 1},
                dgcvType="standard",
                variable_spaces={labelLoc: (sym,)},
            )
            update_globals_k_v(f"D_{labelLoc}", vf_instance)
            update_globals_k_v(f"d_{labelLoc}", df_instance)

            if not (_calledFromCVP == passkey or _calledFromFactory == passkey):
                if targeted_assumptions:
                    t_a = targeted_assumptions.get(labelLoc, None)
                    v_a = t_a if t_a else assumptions
                else:
                    v_a = assumptions
                variable_registry["standard_variable_systems"][labelLoc] = {
                    "family_type": "single",
                    "family_values": var_values,
                    "family_names": (labelLoc,),
                    "differential_system": True,
                    "tempVar": None,
                    "initial_index": None,
                    "variable_relatives": {
                        labelLoc: {
                            "VFClass": vf_instance,
                            "DFClass": df_instance,
                            "assumptions": v_a,
                            "system_index": 0,
                        }
                    },
                }
                variable_registry["_labels"][labelLoc] = {
                    "path": ("standard_variable_systems", labelLoc),
                    "children": {f"D_{labelLoc}", f"d_{labelLoc}"},
                }
                paths = variable_registry.get("paths", None)
                if paths is not None:
                    paths[labelLoc] = {
                        "kind": "coordinate",
                        "path": (
                            "standard_variable_systems",
                            labelLoc,
                            "family_values",
                            0,
                        ),
                    }

        elif isinstance(number_of_variables, Integral) and number_of_variables >= 0:
            lengthLoc = number_of_variables
            if isinstance(index_placement, (list, tuple)):
                if len(index_placement) > 0:
                    index_placement = index_placement[0]
                else:
                    index_placement = ""
            if index_placement is None:
                index_placement = ""
            else:
                index_placement = indexing_places.get(index_placement, "_")
            var_names = [
                f"{labelLoc}{index_placement}{i}"
                for i in range(initialIndex, lengthLoc + initialIndex)
            ]
            if targeted_assumptions:
                merger = assumptions if assumptions else dict()
                vars = tuple(
                    symbol(
                        name,
                        assumptions=merger | targeted_assumptions.get(name, merger),
                    )
                    for name in var_names
                )
            elif assumptions:
                vars = tuple(
                    symbol(name, assumptions=assumptions) for name in var_names
                )
            else:
                vars = tuple(symbol(name) for name in var_names)
            var_values = vars
            update_globals(dict(zip(var_names, vars)))
            update_globals_k_v(labelLoc, tuple(vars))

            if enforce_real and enforced_real_dict is not None:
                if (
                    assumptions.get("real", False)
                    or assumptions.get("nonnegative")
                    or assumptions.get("positive", False)
                ):
                    for v in vars:
                        enforced_real_dict[v.conjugate()] = v
                elif targeted_assumptions:
                    for v in vars:
                        t_a = targeted_assumptions.get(str(v), None)
                        if (
                            t_a
                            and t_a.get("real", False)
                            or t_a.get("nonnegative")
                            or t_a.get("positive", False)
                        ):
                            enforced_real_dict[v.conjugate()] = v

            N = len(var_values)

            vf_instances = [
                vector_field_class(
                    coeff_dict={(j, 1, labelLoc): 1},
                    dgcvType="standard",
                    variable_spaces={labelLoc: var_values},
                )
                for j in range(N)
            ]
            df_instances = [
                differential_form_class(
                    coeff_dict={(j, 0, labelLoc): 1},
                    dgcvType="standard",
                    variable_spaces={labelLoc: var_values},
                )
                for j in range(N)
            ]

            update_globals(dict(zip([f"D_{vn}" for vn in var_names], vf_instances)))
            update_globals(dict(zip([f"d_{vn}" for vn in var_names], df_instances)))

            if not (_calledFromCVP == passkey or _calledFromFactory == passkey):

                def var_v_a(elem):
                    if targeted_assumptions:
                        t_a = targeted_assumptions.get(elem, None)
                        return t_a if t_a else assumptions
                    return assumptions

                variable_registry["standard_variable_systems"][labelLoc] = {
                    "family_type": "tuple",
                    "family_values": var_values,
                    "family_names": tuple(var_names),
                    "differential_system": True,
                    "tempVar": None,
                    "initial_index": initialIndex,
                    "variable_relatives": {
                        var_name: {
                            "VFClass": vf_instances[i],
                            "DFClass": df_instances[i],
                            "assumptions": var_v_a(var_name),
                            "system_index": i,
                        }
                        for i, var_name in enumerate(var_names)
                    },
                }
                variable_registry["_labels"][labelLoc] = {
                    "path": ("standard_variable_systems", labelLoc),
                    "children": set(
                        var_names
                        + [f"D_{v}" for v in var_names]
                        + [f"d_{v}" for v in var_names]
                    ),
                }
                paths = variable_registry.get("paths", None)
                if paths is not None:
                    base_vals = ("standard_variable_systems", labelLoc, "family_values")
                    for i, vn in enumerate(var_names):
                        paths[labelLoc] = {
                            "kind": "standard_variable_system",
                            "path": ("standard_variable_systems", labelLoc),
                        }
                        paths[vn] = {
                            "kind": "coordinate",
                            "path": base_vals + (i,),
                        }

        else:
            raise ValueError(
                "varWithVF expected `number_of_variables` to be a non-negative integer, if provided."
            )

        if rco is not None:
            rco.append(var_values)
    return rco
