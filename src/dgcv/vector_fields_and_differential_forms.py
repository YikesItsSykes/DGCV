"""
package: dgcv - Differential Geometry with Complex Variables
module: vector_fields_and_differential_forms

Description: This module provides tools for creating, manipulating, and decomposing vector fields and
differential forms within the dgcv package. It includes methods for Cartan calculus like
the exterior derivative and Lie derivative. There are some complex structure specific
functions as well, e.g., assembling holomorphic vector fields from holomorphic coefficients.

Key Functions:

Object Assembly:
    - get_VF(): Retrieves coordinate vector fields for the given coordinate variables.
    - get_DF(): Retrieves coordinate differential forms for the given coordinate variables.
    - assembleFromHolVFC(): Assembles a holomorphic vector field from holomorphic coefficients.
    - assembleFromAntiholVFC(): Assembles an antiholomorphic vector field from antiholomorphic
    coefficients.
    - assembleFromCompVFC(): Assembles a complex vector field from both holomorphic and
    antiholomorphic coefficients.

Differential Operators:
    - exteriorDerivative(): Computes the exterior derivative of a differential k-form.
    - interiorProduct(): Computes the interior product of a vector field with a differential k-form.
    - LieDerivative(): Computes the Lie derivative of a differential form or vector field with respect to another vector field.

Decompositions and Basis:
    - decompose(): Decomposes a vector field or differential form as a linear combination of a given basis of vector fields or differential forms.
    - get_coframe(): Constructs a coframe dual to a given list of vector fields.
    - annihilator(): Computes the annihilator (namely sub-bundle in TM/T^*M) of a list of differential forms or vector fields.

Author (of this module): David Sykes (https://realandimaginary.com/dgcv/)

License:
    MIT License
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._config import get_dgcv_settings_registry
from ._safeguards import (
    create_key,
    get_dgcv_category,
    query_dgcv_categories,
    retrieve_passkey,
)
from .backends._numeric_router import zeroish
from .backends._symbolic_router import _scalar_is_zero, get_free_symbols, simplify, subs
from .backends._types_and_constants import expr_numeric_types, symbol
from .conversions import allToReal
from .dgcv_core import (
    VF_bracket,
    differential_form_class,
    variableProcedure,
    vector_field_class,
    wedge,
)
from .solvers import solve_dgcv
from .vmf import vmf_lookup


# -----------------------------------------------------------------------------
# retrieval
# -----------------------------------------------------------------------------
def get_VF(*coordinates):
    out = []
    for coord in coordinates:
        info = vmf_lookup(coord, relatives=True, differential_system=True)

        if info.get("type") != "coordinate":
            raise TypeError(
                f"`get_VF` expects VMF-registered coordinate atoms. Got {coord!r}."
            )

        ds = info.get("differential_system")
        if not isinstance(ds, dict):
            raise TypeError(
                f"`get_VF` received {coord!r}, but it has no differential system registered. "
                "Use `createVariables(..., withVF=True)` or `createVariables(..., complex=True)`."
            )

        vf = ds.get("vf")
        if vf is None:
            raise TypeError(
                f"`get_VF` could not resolve a coordinate vector field for {coord!r}. "
                "Use `createVariables(..., withVF=True)` or `createVariables(..., complex=True)`."
            )

        out.append(vf)

    return out


def get_DF(*coordinates):
    out = []
    for coord in coordinates:
        info = vmf_lookup(coord, relatives=True, differential_system=True)

        if info.get("type") != "coordinate":
            raise TypeError(
                f"`get_DF` expects VMF-registered coordinate atoms. Got {coord!r}."
            )

        ds = info.get("differential_system")
        if not isinstance(ds, dict):
            raise TypeError(
                f"`get_DF` received {coord!r}, but it has no differential system registered. "
                "Use `createVariables(..., withVF=True)` or `createVariables(..., complex=True)`."
            )

        df = ds.get("df")
        if df is None:
            raise TypeError(
                f"`get_DF` could not resolve a coordinate 1-form for {coord!r}. "
                "Use `createVariables(..., withVF=True)` or `createVariables(..., complex=True)`."
            )

        out.append(df)

    return out


# -----------------------------------------------------------------------------
# complex vector fields
# -----------------------------------------------------------------------------
def _require_complex_coordinate_atom(atom: Any, *, context: str) -> dict:
    info = vmf_lookup(atom, path=True, relatives=True)
    if info.get("type") != "coordinate":
        raise TypeError(
            f"{context} requires variables registered in the VMF as coordinates. "
            "Suggestion: initialize complex coordinate systems with dgcv.createVariables(...)."
        )
    st = info.get("sub_type")
    if st == "standard":
        raise TypeError(
            f"{context} requires complex coordinates (holo/anti/real/imag), "
            "but received a standard coordinate. Suggestion: use dgcv.createVariables(...) "
            "to register a complex coordinate system in the VMF."
        )
    return info


def _pick_relative(atom: Any, *, want: str, context: str):
    info = _require_complex_coordinate_atom(atom, context=context)
    st = info.get("sub_type")

    rel = info.get("relatives") or {}
    if want == "holo":
        if st == "holo":
            out = atom
        else:
            out = rel.get("holo", None)
    elif want == "anti":
        if st == "anti":
            out = atom
        else:
            out = rel.get("anti", None)
    elif want == "real":
        if st == "real":
            out = atom
        else:
            out = rel.get("real", None)
    elif want == "imag":
        if st == "imag":
            out = atom
        else:
            out = rel.get("imag", None)
    else:
        raise ValueError("want must be one of: 'holo', 'anti', 'real', 'imag'")

    if out is None:
        raise TypeError(
            f"{context} requires the variable to belong to a complex coordinate system "
            "with the appropriate relatives available in the VMF."
        )

    return out, st


def _warn_if_not_holo_inputs(hol_vars: Sequence[Any], *, context: str):
    if get_dgcv_settings_registry().get("forgo_warnings", False):
        return

    bad = []
    for v in hol_vars:
        info = vmf_lookup(v, relatives=False)
        if info.get("type") != "coordinate":
            continue
        st = info.get("sub_type")
        if st not in (None, "holo"):
            bad.append((v, st))

    if bad:
        warnings.warn(
            f"{context}: holomorphic variables (sub_type='holo') are recommended for hol_vars. "
            f"Received non-holo coordinate atoms: {bad}. "
            "Proceeding by using VMF relatives to interpret them."
        )


def assembleFromHolVFC(
    coeffs: Sequence[Any],
    hol_vars: Sequence[Any],
    *,
    _warn_on_nonholo: bool = True,
) -> "vector_field_class":
    if not isinstance(hol_vars, (list, tuple)):
        raise TypeError("hol_vars must be a list or tuple.")
    if len(coeffs) != len(hol_vars):
        raise ValueError("coeffs and hol_vars must have the same length.")
    if _warn_on_nonholo:
        _warn_if_not_holo_inputs(hol_vars, context="assembleFromHolVFC")

    picked = [
        (_pick_relative(z, want="holo", context="assembleFromHolVFC")[0])
        for z in hol_vars
    ]
    Dz_list = get_VF(*picked)

    out = None
    for c, Dz in zip(coeffs, Dz_list):
        if not c:
            continue
        term = c * Dz
        out = term if out is None else (out + term)

    if out is None:
        out = vector_field_class(coeff_dict={tuple(): 0}, data_shape="all")

    return out


def assembleFromAntiholVFC(
    coeffs: Sequence[Any],
    hol_vars: Sequence[Any],
    *,
    _warn_on_nonholo: bool = True,
) -> "vector_field_class":
    if not isinstance(hol_vars, (list, tuple)):
        raise TypeError("hol_vars must be a list or tuple.")
    if len(coeffs) != len(hol_vars):
        raise ValueError("coeffs and hol_vars must have the same length.")
    if _warn_on_nonholo:
        _warn_if_not_holo_inputs(hol_vars, context="assembleFromAntiholVFC")

    picked = [
        (_pick_relative(z, want="anti", context="assembleFromAntiholVFC")[0])
        for z in hol_vars
    ]
    Dzb_list = get_VF(*picked)

    out = None
    for c, Dzb in zip(coeffs, Dzb_list):
        if not c:
            continue
        term = c * Dzb
        out = term if out is None else (out + term)

    if out is None:
        out = vector_field_class(coeff_dict={tuple(): 0}, data_shape="all")

    return out


def assembleFromCompVFC(
    holomorphic_coeffs: Sequence[Any],
    antiholomorphic_coeffs: Sequence[Any],
    hol_vars: Sequence[Any],
    *,
    _warn_on_nonholo: bool = True,
) -> "vector_field_class":
    if not isinstance(hol_vars, (list, tuple)):
        raise TypeError("hol_vars must be a list or tuple.")
    if len(holomorphic_coeffs) != len(hol_vars) or len(antiholomorphic_coeffs) != len(
        hol_vars
    ):
        raise ValueError("Coefficient lengths must match hol_vars length.")

    vf_h = assembleFromHolVFC(
        holomorphic_coeffs, hol_vars, _warn_on_nonholo=_warn_on_nonholo
    )
    vf_a = assembleFromAntiholVFC(
        antiholomorphic_coeffs, hol_vars, _warn_on_nonholo=_warn_on_nonholo
    )
    return vf_h + vf_a


# -----------------------------------------------------------------------------
# differential forms
# -----------------------------------------------------------------------------
def makeZeroForm(
    expr: Any,
    varSpace: Sequence[Any] | None = None,
    *,
    variable_spaces: dict | None = None,
    data_shape: str = "all",
    dgcvType: str = "standard",
    _simplifyKW=None,
    parameters=set(),
) -> "differential_form_class":
    if variable_spaces is None:
        rv = get_free_symbols(expr)
        if rv:
            params = getattr(expr, "parameters", None)
            if params:
                rv = set(rv) - set(params)
        else:
            rv = set()

        variable_spaces = _prep_symb_set_for_ext_der(
            rv, use_for_zero_form=True, full_dict=True
        )

    return differential_form_class(
        coeff_dict={tuple(): expr},
        valence=tuple(),
        data_shape=data_shape,
        dgcvType=dgcvType,
        _simplifyKW=_simplifyKW,
        variable_spaces=variable_spaces,
        parameters=parameters,
    )


def _prep_symb_set_for_ext_der(symbols, use_for_zero_form=False, full_dict=False):
    registry: Dict[Any, Any] = {}

    for atom in symbols:
        atom_data = vmf_lookup(
            atom,
            path=True,
            flattened_relatives=True,
            differential_system=True,
        )
        if atom_data.get("type") != "coordinate":
            continue

        path = atom_data.get("path")
        if not (isinstance(path, tuple) and len(path) >= 2):
            continue

        syslbl = path[1]

        if full_dict or use_for_zero_form:
            if syslbl in registry:
                continue

            sys_key = atom_data.get("differential_system") or syslbl
            sys_data = vmf_lookup(sys_key, flattened_relatives=True)

            flattened = sys_data.get("flattened_relatives")
            if isinstance(flattened, tuple) and flattened:
                registry[syslbl] = flattened
            continue

        sysreg = registry.setdefault(syslbl, {})

        if atom in sysreg:
            continue

        st = atom_data.get("sub_type")
        rel = atom_data.get("relatives") or {}

        if st in {"holo", "anti"}:
            sysreg[atom] = (rel.get("holo"), rel.get("anti"))
        elif st in {"real", "imag"}:
            sysreg[atom] = (rel.get("holo"), rel.get("anti"))
        else:
            ds = atom_data.get("differential_system")
            if ds is not None:
                sysreg[atom] = (atom,)

    return registry


def exteriorDerivative(form_or_scalar: Any, **kwargs) -> "differential_form_class":
    obj = form_or_scalar

    if get_dgcv_category(obj) != "tensor_field":
        if isinstance(obj, expr_numeric_types()):
            obj = makeZeroForm(obj)
        else:
            raise TypeError(
                "exteriorDerivative expects a differential form or scalar expression."
            )

    if not query_dgcv_categories(obj, {"differential_form"}):
        raise TypeError(
            "exteriorDerivative expects a differential form or scalar expression."
        )

    form = obj

    rv = get_free_symbols(form)
    if rv:
        params = getattr(form, "parameters", None)
        if params:
            rv = set(rv) - set(params)
    else:
        rv = set()

    relevant_vars = _prep_symb_set_for_ext_der(rv)

    accumulation = 0
    for system in relevant_vars.values():
        for atom in system:
            ds = vmf_lookup(atom, differential_system=True).get("differential_system")
            if ds is None:
                continue
            vf = ds.get("vf")
            df = ds.get("df")
            if vf is None or df is None:
                continue
            accumulation += df * (form.apply(vf))

    return accumulation


def interiorProduct(
    vector_field: "vector_field_class",
    differential_form: "differential_form_class",
) -> "differential_form_class":
    if not query_dgcv_categories(vector_field, {"vector_field"}):
        raise TypeError("interiorProduct expects a vector_field_class instance.")
    if not query_dgcv_categories(differential_form, {"differential_form"}):
        raise TypeError("interiorProduct expects a differential_form_class instance.")

    return differential_form(vector_field)


def LieDerivative(
    vector_field: "vector_field_class",
    obj: Any,
    **kwargs,
):
    if not query_dgcv_categories(vector_field, {"vector_field"}):
        raise TypeError(
            "LieDerivative expects the first argument to be a vector field."
        )

    if isinstance(obj, expr_numeric_types()):
        return vector_field(obj)

    if get_dgcv_category(obj) != "tensor_field":
        raise TypeError("LieDerivative expects a tensor field or scalar expression.")

    if query_dgcv_categories(obj, {"differential_form"}):
        return exteriorDerivative(interiorProduct(vector_field, obj)) + interiorProduct(
            vector_field, exteriorDerivative(obj)
        )

    if query_dgcv_categories(obj, {"vector_field"}):
        return VF_bracket(vector_field, obj)

    raise TypeError(
        "LieDerivative currently supports differential forms and vector fields only."
    )


# -----------------------------------------------------------------------------
# vector field and differential forms operations with linsolve computations
# -----------------------------------------------------------------------------
def _local_unknowns(n: int, lbl: str = "__tem_label__") -> List[Any]:
    return [symbol(f"{lbl}{i}") for i in range(n)]


def _free_vars_from_solution(
    sol: Dict[Any, Any], vars_list: Sequence[Any]
) -> List[Any]:
    out: List[Any] = []
    for v in vars_list:
        try:
            if sol.get(v, None) == v:
                out.append(v)
        except Exception:
            pass
    return out


def _sub_free_to_zero(sol: Dict[Any, Any], free: Sequence[Any]) -> Dict[Any, Any]:
    if not free:
        return sol
    out = dict(sol)
    for v in free:
        out[v] = 0
    return out


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


def _key_universe(tfs):
    K = set()
    for tf in tfs:
        if get_dgcv_category(tf) == "tensor_field":
            for k, v in tf.coeff_dict.items():
                if v != 0:
                    K.add(k)
    return tuple(sorted(K))


def _as_coeff_vector_form(tf, K, syslbl="__dgcv_par__"):
    cd = {}
    varspacPlaceholder = None
    for j, k in enumerate(K):
        v = tf.coeff_dict.get(k, 0)
        if v != 0:
            cd[(j, 0, syslbl)] = v
    if not cd:
        cd = {tuple(): 0}
    return differential_form_class(
        coeff_dict=cd, data_shape="skew", variable_spaces={syslbl: varspacPlaceholder}
    )


def _extract_basis_by_wedge_vectorized(objs, *, use_numeric_methods: bool = False):
    if not objs:
        return []

    use_numeric = use_numeric_methods or bool(
        get_dgcv_settings_registry().get("use_numeric_methods", False)
    )

    def _is_zero_scalar(x):
        if use_numeric:
            return zeroish(x)
        return _scalar_is_zero(x)

    if all(get_dgcv_category(o) == "array" for o in objs):
        K = set()
        for o in objs:
            K.update(getattr(o, "_data", {}).keys())
        if not K:
            return []

        K = tuple(sorted(K))

        vecs = []
        for o in objs:
            d = getattr(o, "_data", {})
            v = {}
            for k in K:
                val = d.get(k, 0)
                if val is None or _is_zero_scalar(val):
                    continue
                v[k] = val
            vecs.append(v)

        def _kform_is_zero(F):
            return not F or all(_is_zero_scalar(c) for c in F.values())

        def _wedge_kform_vector(F, v):
            out = {}
            if not F or not v:
                return out
            for idx, c in F.items():
                I_set = set(idx)
                for j, vj in v.items():
                    if j in I_set:
                        continue
                    pos = 0
                    for t in idx:
                        if t < j:
                            pos += 1
                        else:
                            break
                    sign = -1 if ((len(idx) - pos) % 2) else 1
                    J = idx[:pos] + (j,) + idx[pos:]
                    coeff = sign * c * vj
                    if _is_zero_scalar(coeff):
                        continue
                    out[J] = out.get(J, 0) + coeff
            if not use_numeric:
                out = {k: simplify(v) for k, v in out.items() if not _is_zero_scalar(v)}
            else:
                out = {k: v for k, v in out.items() if not _is_zero_scalar(v)}
            return out

        obstruction = None
        out = []
        for o, v in zip(objs, vecs):
            if not v:
                continue

            if obstruction is None:
                obstruction = {(k,): val for k, val in v.items()}
                if not use_numeric:
                    obstruction = {
                        idx: simplify(c)
                        for idx, c in obstruction.items()
                        if not _is_zero_scalar(c)
                    }
                out.append(o)
                continue

            w = _wedge_kform_vector(obstruction, v)
            if _kform_is_zero(w):
                continue

            obstruction = w
            out.append(o)

        return out

    K = _key_universe(objs)
    if not K:
        return []

    try:
        vec_forms = [_as_coeff_vector_form(o, K) for o in objs]
    except Exception as exc:
        raise TypeError(
            "Could not compute a linear-independence test for these objects because "
            "they do not support the required linear-combination behavior."
        ) from exc

    obstruction = None
    out = []
    for o, v in zip(objs, vec_forms):
        if use_numeric:
            if zeroish(v):
                continue
        else:
            if _scalar_is_zero(v):
                continue

        if obstruction is None:
            obstruction = v
            out.append(o)
            continue

        w = wedge(obstruction, v)
        if not use_numeric:
            w = simplify(w)
            if _scalar_is_zero(w):
                continue
        else:
            if zeroish(w):
                continue

        obstruction = w
        out.append(o)

    return out


def _coordinate_basis_from_vmf(
    coordinate_space: Sequence[Any],
    *,
    want: str,  # "df" or "vf"
) -> List[Any]:
    out: List[Any] = []
    for atom in coordinate_space:
        info = vmf_lookup(atom, differential_system=True)
        ds = info.get("differential_system")
        if not isinstance(ds, dict):
            raise TypeError(
                "`coordinate_space` must consist of VMF-registered coordinates with a differential system."
            )
        obj = ds.get(want)
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

    atoms: List[Any] = []
    for a in syms:
        info = vmf_lookup(a, differential_system=True)
        ds = info.get("differential_system")
        if isinstance(ds, dict) and (
            ds.get("vf") is not None or ds.get("df") is not None
        ):
            atoms.append(a)

    atoms.sort(key=lambda x: str(x))
    return atoms


def decompose(
    obj,
    basis,
    return_parameters: bool = False,
    new_parameters_label: str | None = None,
    only_check_decomposability: bool = False,
    _pass_error_report=None,
    _hand_off=None,
    *,
    assume_basis: bool = False,
    register_parameters: bool = False,
):
    """
    Decomposes a vector field or differential form as a linear combination of a given `basis` list.

    This function attempts to express the input `obj` (a VFClass or DFClass object) as a linear combination
    of the elements in the provided `basis` list. The `basis` list does not need to be linearly independent,
    and if the decomposition is not unique, the function will parameterize the solution space. Any parameters
    needed are automatically initialized and registered in dgcv's variable management framework (VMF).

    The function carefully handles variable types based on the `dgcvType` attribute of the objects. For objects
    with `dgcvType='complex'`, it dynamically selects whether to perform real coordinate computations or complex
    coordinate computations, depending on the input data. If a canonical variable formatting decision cannot be
    made naturally from the input, the function will return warnings with explanations.

    Parameters
    ----------
    obj : VFClass or DFClass
        The vector field or differential form to decompose.
    basis : list of VFClass or DFClass
        A list of objects (vector fields or differential forms) to decompose `obj` with respect to.
        The class of objects in the `basis` list must match the class of `obj`.
    return_parameters : bool, optional
        If True, the function will return the parameterized solution when the decomposition is not unique
        (default is False). Parameters are initialized with labels registered within the VMF.
    new_parameters_label : str or None, optional
        If `return_parameters` is True and the decomposition is not unique, this label will be used
        to name the new parameter variables. If None, obscure labels will be generated automatically (default is None).
    _pass_error_report : optional
        Internal use parameter for handling error reports in certain edge cases (default is None).

    Returns
    -------
    list
        The coefficients of the linear combination that expresses `obj` in terms of the `basis` list.
        If the decomposition is parameterized, the returned list contains the parameterized solution.

    Raises
    ------
    TypeError
        If the class of `obj` does not match the class of elements in the `basis` list (i.e., both must
        be either VFClass or DFClass), or if objects in the `basis` list have inconsistent `dgcvType` attributes.

    Warnings
    --------
    - A warning is issued if `obj` is not in the span of the provided `basis` list.
    - If the `basis` list is not linearly independent, the decomposition is not unique, and a parameterized
    solution will be returned. The initialized parameters are registered as 'standard variables' in the VMF.

    Remarks
    -------
    - The function dynamically handles objects based on their `dgcvType` attribute. For `dgcvType='complex'`,
    it distinguishes between real and complex coordinate computations, converting the input as needed
    to ensure consistency in formatting. If this decision cannot be determined from the input data,
    the function issues warnings explaining the necessary canonical formatting.

    Example
    -------
    # Decompose a vector field 'vf' as a linear combination of two basis vector fields 'vf1' and 'vf2'
    coeffs, basis_used = decompose(vf, [vf1, vf2])

    # Decompose a differential form 'df' with a parameterized solution due to non-uniqueness
    coeffs, basis_used = decompose(df, [df1, df2, df3], return_parameters=True, new_parameters_label='p')
    """
    basis = list(basis)
    original_length = len(basis)

    if not return_parameters and not assume_basis:
        basis = _extract_basis_by_wedge_vectorized(basis)

    n = len(basis)
    if n == 0:
        if only_check_decomposability is True:
            ob = getattr(obj, "__dgcv_zero_obstr__", None)
            if ob is None:
                raise TypeError(
                    f"decompose does not operate on objects of type {type(obj).__name__}"
                )
            eqns, _ = ob
            eqns = list(eqns or [])
            return bool(eqns) and all(_scalar_is_zero(e) for e in eqns)
        return ([], basis)

    label = create_key(prefix="var")
    vars = [symbol(f"{label}{i}") for i in range(n)]

    try:
        system = obj
        for v, e in zip(vars, basis):
            system = system - v * e
    except Exception as exc:
        msg = (
            "`decompose` could not form the residual `obj - sum(var_i * basis_i)` "
            f"for obj type {type(obj).__name__} and basis element types "
            f"{[type(b).__name__ for b in basis]}. "
            "This usually means the objects do not support compatible addition/subtraction "
            "and scalar multiplication."
        )
        if _pass_error_report == retrieve_passkey():
            return msg + f" Original error: {exc!r}"
        raise TypeError(msg) from exc

    ob = getattr(system, "__dgcv_zero_obstr__", None)
    if ob is None:
        raise TypeError(f"{type(system).__name__} lacks __dgcv_zero_obstr__")
    eqns, _ = ob
    eqns = list(eqns or [])

    solutions = solve_dgcv(eqns, vars, method="linsolve")

    if not solutions:
        if only_check_decomposability is True:
            return False
        if _pass_error_report == retrieve_passkey():
            return (
                f"`decompose` found no solution for equations {eqns} in variables {vars} "
                f"against a spanning family of length {original_length}."
            )
        return ([], basis)

    if only_check_decomposability is True:
        return True

    sol0 = solutions[0]
    decomp_coeffs = [sol0.get(v, v) for v in vars]

    free = set()
    for expr in decomp_coeffs:
        free |= set(get_free_symbols(expr) or ())

    if return_parameters and free:
        free = tuple(free)
        if register_parameters:
            params = variableProcedure(
                new_parameters_label
                if isinstance(new_parameters_label, str)
                else "param",
                len(free),
                return_created_object=True,
            )[0]
            params = tuple(params)
        else:
            plabel = (
                new_parameters_label
                if isinstance(new_parameters_label, str) and new_parameters_label
                else create_key(prefix="param")
            )
            params = tuple(symbol(f"{plabel}{i}") for i in range(len(free)))

        subs_dict = {v: p for v, p in zip(free, params)}
        decomp_coeffs = [subs(c, subs_dict) for c in decomp_coeffs]

    return (decomp_coeffs, basis)


def get_coframe(
    VFList: Sequence[Any],
    *,
    coordinate_space: Sequence[Any] | None = None,
    return_parameters: bool = False,
    register_parameters: bool = False,
    parameters_label: str | None = None,
):
    VFList = list(VFList)
    if not VFList:
        return []

    _require_subcategory(VFList, {"vector_field"}, "get_coframe")

    if coordinate_space is None:
        coordinate_space = _infer_coordinate_space_from_objs(VFList)
    else:
        if not isinstance(coordinate_space, (list, tuple, set)):
            raise TypeError(
                "`get_coframe` coordinate_space must be a list/tuple/set if provided."
            )
        coordinate_space = list(coordinate_space)
        for a in coordinate_space:
            ds = vmf_lookup(a, differential_system=True).get("differential_system")
            if ds is None:
                raise TypeError(
                    "`get_coframe` requires coordinates to be registered in the dgcv VMF with differential objects. "
                    "Suggestion: initialize coordinates with `createVariables(..., withVF=True)` (or `createVariables(..., complex=True)` as appropriate)."
                )

    dfBasis = _coordinate_basis_from_vmf(coordinate_space, want="df")
    n = len(dfBasis)
    m = len(VFList)

    if n == 0:
        raise TypeError("`get_coframe` could not obtain a DF basis from VMF.")

    def _as_scalar_eqns(x):
        if get_dgcv_category(x) == "tensor_field":
            return list(x.coeff_dict.values())
        return [x]

    eval_entries: List[List[Any]] = []
    for k in range(m):
        row = []
        vf = VFList[k]
        for df in dfBasis:
            val = df(vf)
            scalars = _as_scalar_eqns(val)
            if len(scalars) != 1:
                row.append(sum(scalars))
            else:
                row.append(scalars[0])
        eval_entries.append(row)

    unknowns = _local_unknowns(m * n)

    eqns: List[Any] = []
    for j in range(m):
        row_vars = unknowns[j * n : (j + 1) * n]
        for k in range(m):
            target = 1 if j == k else 0
            s = 0
            Ek = eval_entries[k]
            for u, e in zip(row_vars, Ek):
                s = s + u * e
            eqns.append(s - target)

    sols = solve_dgcv(eqns, unknowns, method="linsolve")
    if not sols:
        raise RuntimeError(
            "`get_coframe` could not solve for a coframe (system unsatisfiable)."
        )

    sol = sols[0]

    unknowns_set = set(unknowns)
    free_vars_set = set()
    for u in unknowns:
        free_vars_set |= set(get_free_symbols(sol.get(u, u))) & unknowns_set
    free = [u for u in unknowns if u in free_vars_set]

    if return_parameters and free:
        params = _make_parameters(
            len(free),
            register_parameters=register_parameters,
            parameters_label=parameters_label,
        )
        sub = dict(zip(free, params))
        sol_use = dict(sol)
        for v in free:
            sol_use[v] = v

        def _coeff(u):
            c = sol_use.get(u, u)
            return subs(c, sub)
    else:
        sol_use = dict(sol)
        for v in free:
            sol_use[v] = 0

        def _coeff(u):
            return sol_use.get(u, u)

    out: List[Any] = []
    for j in range(m):
        row_vars = unknowns[j * n : (j + 1) * n]
        omega_j = 0
        for u, df in zip(row_vars, dfBasis):
            omega_j = omega_j + _coeff(u) * df
        out.append(omega_j)

    return out


def annihilator(
    objList: Sequence[Any],
    coordinate_space: Sequence[Any] | None = None,
    control_distribution: Optional[Sequence[Any]] = None,
    _pass_error_report=None,
    allow_div_by_zero: bool = False,
    *,
    return_parameters: bool = False,
    register_parameters: bool = False,
    parameters_label: str | None = None,
    coherent_coordinates_checked: bool = False,
):
    """
    Finds annihilators for a given list of vector fields or differential forms.

    This function computes objects that "annihilate" the provided list of vector fields or differential forms.
    An annihilator is either the span of differential forms that evaluate to zero on each vector field in the list,
    or vector fields whose interior product annihilates each differential form in the list. `annihilator` dynamically
    handles both real and holomorphic coordinate systems and can convert between them as needed. Additionally,
    solutions can be constrained to a given control distribution by using the `control_distribution` keyword.

    Parameters
    ----------
    objList : list of VFClass or DFClass
        A list of vector fields or differential forms for which the annihilator will be computed. All objects
        in the list must be of the same class (either all vector fields or all differential forms) and have
        consistent `dgcvType` attributes (i.e., 'standard' or 'complex').
    coordinate_Space : list, tuple, or set
        A collection of variables that define the coordinate system in which the annihilator is to be computed.
    allow_div_by_zero : bool, optional
        If True, allows the annihilator to be returned without scaling to avoid division by zero (default is False).
        Scaling to avoid division has more computational overhead but typically simplifies output.
    _pass_error_report : optional
        Internal use parameter for handling error reports in certain edge cases (default is None).

    Returns
    -------
    list
        A list of differential forms (if vector fields were provided) or vector fields (if differential forms were
        provided) that annihilate the objects in `objList`.

    Raises
    ------
    TypeError
        If the objects in `objList` are not all of the same type (i.e., all vector fields or all differential forms),
        or if the `coordinate_Space` is not a valid list, tuple, or set.

    Warnings
    --------
    - If the objects in `objList` are defined with inconsistent coordinate system types (real vs. holomorphic), the
    function converts them to a consistent coordinate system and issues a warning.

    Example
    -------
    >>> from dgcv import createVariables, annihilator, exteriorDerivative, complex_struct_op, Del, DelBar, allToReal
    >>> createVariables('z', 'x', 'y', 4, initialIndex=0)
    >>> rho = (x1*x2 + x1**2*x3 - x0)  # A defining equation for a real hypersurface M in C^4
    >>> d_rho = exteriorDerivative(rho)  # Its differential will annihilate TM
    >>> print(d_rho)

    (2*x1*x3 + x2)*d_x1 + x1*d_x2 + x1**2*d_x3 - 1*d_x0

    >>> dfList = [d_rho]
    >>> TMbasis = annihilator(dfList, x+y)  # Use annihilator to compute the tangent bundle TM
    >>> TMbasis

    [(16*x1*x3 + 8*x2)*D_x0 + 8*D_x1, 8*x1*D_x0 + 8*D_x2, 8*x1**2*D_x0 + 8*D_x3, 8*D_y0, 8*D_y1, 8*D_y2, 8*D_y3]

    >>> J_of_TMbasis = [complex_struct_op(vf) for vf in TMbasis]  # Get the image of TM under the complex structure operator.
    >>> J_of_TMbasis

    [(16*x1*x3 + 8*x2)*D_y0 + 8*D_y1, 8*x1*D_y0 + 8*D_y2, 8*x1**2*D_y0 + 8*D_y3, -8*D_x0, -8*D_x1, -8*D_x2, -8*D_x3]

    >>> CR_distribution = annihilator(annihilator(J_of_TMbasis, x+y) + annihilator(TMbasis, x+y), x+y)
    >>> CR_distribution  # Use annihilator to get the CR distribution, which is the intersection of TM with CTM

    [(16*x1*x3 + 8*x2)*D_x0 + 8*D_x1, 8*x1*D_x0 + 8*D_x2, 8*x1**2*D_x0 + 8*D_x3,
    (16*x1*x3 + 8*x2)*D_y0 + 8*D_y1, 8*x1*D_y0 + 8*D_y2, 8*x1**2*D_y0 + 8*D_y3]

    >>> LeviForm = allToReal(Del(DelBar(rho)))  # Apply Dolbeault operators to represent the Levi form
    >>> print(LeviForm._repr_latex_())

    # Output: <LeviForm in LaTeX formatted plain text>

    >>> Levi_kernel = annihilator([LeviForm], x+y, control_distribution=CR_distribution)
    >>> Levi_kernel  # annihilator reveals that the Levi form has a real 2-d. kernel

    [-64*x1**2*D_x0 - 128*x1*D_x2 + 64*D_x3, -64*x1**2*D_y0 - 128*x1*D_y2 + 64*D_y3]

    >>> not_the_Levi_kernel = annihilator([LeviForm], x+y)
    >>> not_the_Levi_kernel  # Without constraining annihilator to the CR distribution, it finds a kernel that is too large.

    [8*D_x0, -16*x1*D_x2 + 8*D_x3, 8*D_y0, -16*x1*D_y2 + 8*D_y3]
    """
    objList = list(objList)
    if not objList:
        return []

    if not all(get_dgcv_category(o) == "tensor_field" for o in objList):
        raise TypeError("`annihilator` expects tensor_field objects.")

    is_vf_case = all(query_dgcv_categories(o, {"vector_field"}) for o in objList)
    is_df_case = all(query_dgcv_categories(o, {"differential_form"}) for o in objList)
    if not is_vf_case and not is_df_case:
        raise TypeError(
            "`annihilator` expects a list of all vector fields or all differential forms."
        )

    if control_distribution is not None:
        control_distribution = list(control_distribution)
        if is_vf_case:
            if not all(
                query_dgcv_categories(o, {"differential_form"})
                for o in control_distribution
            ):
                raise TypeError(
                    "`annihilator` control_distribution must be differential forms in the VF case."
                )
            basis_elems = control_distribution
        else:
            if not all(
                query_dgcv_categories(o, {"vector_field"}) for o in control_distribution
            ):
                raise TypeError(
                    "`annihilator` control_distribution must be vector fields in the DF case."
                )
            basis_elems = control_distribution

    elif coordinate_space is not None:
        if not isinstance(coordinate_space, (list, tuple, set)):
            raise TypeError(
                "`annihilator` coordinate_space must be a list/tuple/set if provided."
            )

        coords = list(coordinate_space)
        for a in coords:
            ds = vmf_lookup(a, differential_system=True).get("differential_system")
            if ds is None:
                raise TypeError(
                    "`annihilator` requires coordinates to be registered in the dgcv VMF with differential objects. "
                    "Suggestion: initialize coordinates with `createVariables(..., withVF=True)` (or `createVariables(..., complex=True)` as appropriate)."
                )

        basis_elems = []
        for a in coords:
            ds = vmf_lookup(a, differential_system=True).get("differential_system")
            elem = ds.get("df") if is_vf_case else ds.get("vf")
            if elem is None:
                raise TypeError(
                    "`annihilator` requires coordinates to be registered in the dgcv VMF with differential objects. "
                    "Suggestion: initialize coordinates with `createVariables(..., withVF=True)` (or `createVariables(..., complex=True)` as appropriate)."
                )
            basis_elems.append(elem)

    else:
        atoms = set()
        for o in objList:
            vs = getattr(o, "_variable_spaces", None)
            if isinstance(vs, dict):
                for tup in vs.values():
                    atoms.update(tup)

        basis_elems = []
        for a in sorted(atoms, key=lambda x: str(x)):
            ds = vmf_lookup(a, differential_system=True).get("differential_system")
            if not ds:
                continue
            elem = ds.get("df") if is_vf_case else ds.get("vf")
            if elem is None:
                continue
            basis_elems.append(elem)

        if not basis_elems:
            raise TypeError(
                "`annihilator` could not obtain a coordinate basis from VMF with differential objects. "
                "Suggestion: initialize coordinates with `createVariables(..., withVF=True)` (or `createVariables(..., complex=True)` as appropriate)."
            )

    if not coherent_coordinates_checked:
        objList = [allToReal(o) for o in objList]
        basis_elems = [allToReal(e) for e in basis_elems]

    n = len(basis_elems)
    label = create_key(prefix="var")
    vars_list = [symbol(f"{label}{i}") for i in range(n)]

    general = 0
    for u, e in zip(vars_list, basis_elems):
        general = general + u * e

    if is_vf_case:
        eqns = [general(vf) for vf in objList]
    else:
        eqns = [df(general) for df in objList]

    scalar_eqns = []
    for e in eqns:
        if get_dgcv_category(e) == "tensor_field":
            scalar_eqns.extend(e.coeff_dict.values())
        else:
            scalar_eqns.append(e)

    sols = solve_dgcv(scalar_eqns, vars_list, method="linsolve")
    if not sols:
        if _pass_error_report == retrieve_passkey():
            return "`annihilator` found no solutions."
        return []

    sol = sols[0]

    def _apply_sol(expr, sol_dict):
        return subs(expr, sol_dict) if isinstance(sol_dict, dict) else expr

    general_solution = _apply_sol(general, sol)

    free_vars = [u for u in vars_list if sol.get(u, u) == u]

    if not free_vars:
        return [general_solution]

    out = []
    for v in free_vars:
        assign = {u: 0 for u in free_vars}
        assign[v] = 1
        out.append(_apply_sol(general_solution, assign))

    out = [elem for elem in out if not _scalar_is_zero(elem)]
    return out
