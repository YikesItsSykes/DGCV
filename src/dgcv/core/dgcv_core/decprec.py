from __future__ import annotations

from numbers import Integral

from ..._aux._backends._symbolic_router import _scalar_is_zero, conjugate
from ..._aux._utilities._config import dgcv_warning, dgcvDeprecationWarning
from ..._aux._vmf._safeguards import get_dgcv_category, query_dgcv_categories
from ..._aux._vmf.vmf import vmf_lookup
from .fields import differential_form_class, vector_field_class
from .fields.assembly import assemble_tensor_field


def TFClass(
    self,
    varSpace,
    coeff_dict,
    valence=None,
    data_shape="general",
    dgcvType="standard",
    _simplifyKW=None,
):
    dgcv_warning(
        "`TFClass` is deprecated and has been replaced by the general `tensor_field_class`. The function label remains"
        "as a dispatcher to build `tensor_field_class` objects, but this may be removed in the future"
        "Please use `tensor_field_class` or `assemble_tensor_field` instead.",
        dgcvDeprecationWarning,
        stacklevel=2,
        old_kw="TFClass",
        new_kw="tensor_field_class",
        sunset="2026",
    )
    key = next(iter(coeff_dict))
    if valence is None and key is None:
        raise RuntimeError(
            "`STFClass` recieved invalid `data_dict`. `STFClass` is also deprecated. Use `tensor_field_class` instead"
        )
    val = valence if valence else key[len(key) // 2 :]  # old key format assumption
    return assemble_tensor_field(
        coordinate_space=varSpace,
        coefficient_dict=coeff_dict,
        valence=val,
        shape=data_shape,
        _simplifyKW=_simplifyKW,
    )


def tensorField(
    self,
    varSpace,
    coeff_dict,
    valence=None,
    data_shape="general",
    dgcvType="standard",
    _simplifyKW=None,
):
    dgcv_warning(
        "`tensorField` is deprecated and has been replaced by the general `tensor_field_class`. The function label remains"
        "as a dispatcher to build `tensor_field_class` objects, but this may be removed in the future.",
        dgcvDeprecationWarning,
        stacklevel=2,
        old_kw="tensorField",
        new_kw="tensor_field_class",
        sunset="2027",
    )
    key = next(iter(coeff_dict))
    if valence is None and key is None:
        raise RuntimeError(
            "`STFClass` recieved invalid `data_dict`. `STFClass` is also deprecated. Use `tensor_field_class` instead"
        )
    val = valence if valence else key[len(key) // 3 : 2 * len(key) // 3]
    return assemble_tensor_field(
        coordinate_space=varSpace,
        coefficient_dict=coeff_dict,
        valence=val,
        shape=data_shape,
        _simplifyKW=_simplifyKW,
    )


def STFClass(
    self,
    varSpace,
    data_dict,
    degree,
    dgcvType="standard",
    _simplifyKW=None,
):
    dgcv_warning(
        "`STFClass` is deprecated and has been replaced by the general `tensor_field_class`. The function label remains"
        "as a dispatcher to build `tensor_field_class` objects, but this may be removed in the future.",
        dgcvDeprecationWarning,
        stacklevel=2,
        old_kw="STFClass",
        new_kw="tensor_field_class",
        sunset="2027",
    )
    key = next(iter(data_dict))
    if key is None:
        raise RuntimeError(
            "`STFClass` recieved invalid `data_dict`. `STFClass` is also deprecated. Use `tensor_field_class` instead"
        )
    val = key[len(key) // 3 : 2 * len(key) // 3]  # old key format assumption
    return assemble_tensor_field(
        coordinate_space=varSpace,
        coefficient_dict=data_dict,
        valence=val,
        shape="symmetric",
        _simplifyKW=_simplifyKW,
    )


def VFClass(
    varSpace,
    coeffs,
    dgcvType="standard",
    _simplifyKW=None,
):
    dgcv_warning(
        "`VFClass` is deprecated and has been replaced by the general `vector_field_class`. The function label remains"
        "as a dispatcher to build `vector_field_class` objects, but this may be removed in the future.",
        dgcvDeprecationWarning,
        stacklevel=2,
        old_kw="VFClass",
        new_kw="vector_field_class",
        sunset="2027",
    )
    if _simplifyKW is None:
        _simplifyKW = {
            "simplify_rule": None,
            "simplify_ignore_list": None,
            "preferred_basis_element": None,
        }

    vs = tuple(varSpace) if varSpace is not None else tuple()
    cs = list(coeffs) if coeffs is not None else []

    return vector_field_class(
        varSpace=vs,
        coeffs=cs,
        dgcvType=dgcvType,
        _simplifyKW=_simplifyKW,
        variable_spaces=None,
    )


def DFClass(
    varSpace,
    data_dict,
    degree,
    dgcvType="standard",
    _simplifyKW=None,
):
    dgcv_warning(
        "`DFClass` is deprecated and has been replaced by the general `differential_form_class`. The function label remains"
        "as a dispatcher to build `differential_form_class` objects, but this may be removed in the future.",
        dgcvDeprecationWarning,
        stacklevel=2,
        old_kw="tensorField",
        new_kw="differential_form_class",
        sunset="2027",
    )
    if _simplifyKW is None:
        _simplifyKW = {
            "simplify_rule": None,
            "simplify_ignore_list": None,
            "preferred_basis_element": None,
        }

    if not isinstance(varSpace, (list, tuple)):
        raise TypeError("`varSpace` must be a list or tuple.")
    if not isinstance(degree, Integral) or int(degree) < 0:
        raise ValueError("`degree` must be a non-negative integer.")
    if not isinstance(data_dict, dict):
        raise TypeError("`data_dict` must be a dictionary.")

    deg = int(degree)
    vs_list = list(varSpace)

    nz = {k: v for k, v in data_dict.items() if not _scalar_is_zero(v)}
    if not nz:
        return differential_form_class(
            coeff_dict={tuple(): 0},
            dgcvType=dgcvType,
            _simplifyKW=_simplifyKW,
            variable_spaces={},
        )

    if deg == 0:
        val = nz.get(tuple(), 0)
        return differential_form_class(
            coeff_dict={tuple(): val},
            dgcvType=dgcvType,
            _simplifyKW=_simplifyKW,
            variable_spaces={},
        )

    sys_for_var = {}
    systems_used = set()

    for v in vs_list:
        info = vmf_lookup(v, path=True, relatives=False)
        p = info.get("path")
        if not (isinstance(p, tuple) and len(p) >= 2):
            raise KeyError(
                "DFClass legacy init requires variables registered in the VMF."
            )
        syslbl = p[1]
        sys_for_var[v] = syslbl
        systems_used.add(syslbl)

    variable_spaces = {}
    system_index_cache = {}

    for syslbl in systems_used:
        info = vmf_lookup(syslbl, path=True, relatives=True, flattened_relatives=True)
        flat = info.get("flattened_relatives", None)
        if isinstance(flat, tuple) and flat:
            variable_spaces[syslbl] = flat
        else:
            seen = []
            for v in vs_list:
                if sys_for_var[v] == syslbl:
                    seen.append(v)
            variable_spaces[syslbl] = tuple(seen)

        system_index_cache[syslbl] = {
            v: i for i, v in enumerate(variable_spaces[syslbl])
        }

    new_cd = {}
    valence_tuple = (0,) * deg

    for key, value in nz.items():
        if not isinstance(key, tuple):
            raise TypeError("Keys in `data_dict` must be tuples.")
        if len(key) != deg:
            raise ValueError("`data_dict` keys must have length equal to `degree`.")

        idxs = []
        syslbls = []

        for pos in key:
            if not isinstance(pos, Integral):
                raise TypeError("Old-style indices must be integers.")
            ii = int(pos)
            if ii < 0 or ii >= len(vs_list):
                raise ValueError("Old-style index out of range.")

            var = vs_list[ii]
            syslbl = sys_for_var[var]
            syslbls.append(syslbl)

            j = system_index_cache[syslbl].get(var, None)
            if j is None:
                raise KeyError(
                    f"DFClass: variable '{var}' not found in cached system '{syslbl}'."
                )
            idxs.append(j)

        nk = tuple(idxs + list(valence_tuple) + syslbls)
        new_cd[nk] = new_cd.get(nk, 0) + value

    new_cd = {k: v for k, v in new_cd.items() if not _scalar_is_zero(v)}
    if not new_cd:
        new_cd = {tuple(): 0}

    return differential_form_class(
        coeff_dict=new_cd,
        dgcvType=dgcvType,
        _simplifyKW=_simplifyKW,
        variable_spaces=variable_spaces,
    )


def _TFDictToNewBasis(data_dict, oldBasis, newBasis):
    data_list = list(data_dict.items())
    degree = len(data_list[0][0])
    try:
        dataDict = dict(
            [
                (tuple(newBasis.index(oldBasis[k]) for k in j[0]), j[1])
                for j in data_list
                if not _scalar_is_zero(j[1])
            ]
        )
    except ValueError as e:
        raise ValueError(
            f"`sparseKFormDataNewBasis` recieved bases for which an element in oldBasis {oldBasis} does not exist in newBasis {newBasis} whilst the sparseKFormData indicates this element crucial in the k-form's definition: {e}"
        )
    if not dataDict:
        dataDict = {(0,) * degree: 0}

    return dataDict


def sparseKFormDataNewBasis(sparseKFormData, oldBasis, newBasis):
    if (
        not sparseKFormData
    ):  # Maybe safe to remove following October DFClassDataDict reformat!!!
        return {tuple(): 0}
    degree = len(sparseKFormData[0][0])
    try:
        dataDict = dict(
            [
                (tuple(newBasis.index(oldBasis[k]) for k in j[0]), j[1])
                for j in sparseKFormData
                if not _scalar_is_zero(j[1])
            ]
        )
    except ValueError as e:
        raise ValueError(
            f"`sparseKFormDataNewBasis` recieved bases for which an element in oldBasis {oldBasis} does not exist in newBasis {newBasis} whilst the sparseKFormData indicates this element crucial in the k-form's definition: {e}"
        )
    if not dataDict:
        dataDict = {(0,) * degree: 0}
    return dataDict


def VF_coeffs_direct(vf, var_space, sparse=False):
    """
    Depricated: Use `VF_coeffs` instead.
    """
    if not query_dgcv_categories(vf, {"vector_field"}):
        raise TypeError("Expected first argument to be a vector field")

    if not isinstance(var_space, (list, tuple)):
        raise TypeError("Expected second argument to be a list or tuple of variables")

    # Evaluate the vector field on each element in var_space
    coeffs = [vf(var) for var in var_space]

    # Return sparse or full result
    if sparse:
        return [
            ((i,), coeffs[i])
            for i in range(len(coeffs))
            if not _scalar_is_zero(coeffs[i])
        ] or [((0,), 0)]
    return coeffs


def VF_coeffs(vf, var_list, sparse: bool = False):
    if not query_dgcv_categories(vf, {"vector_field"}):
        raise TypeError(f"VF_coeffs expects a vector_field, got {type(vf).__name__}.")

    if not isinstance(var_list, (list, tuple)):
        raise TypeError("VF_coeffs expects var_list to be a list or tuple.")

    cd = vf.coeff_dict
    vspaces = getattr(vf, "_variable_spaces", None)

    def _locate_in_vspaces(var):
        if not isinstance(vspaces, dict):
            return None, None
        for syslbl, vs in vspaces.items():
            try:
                j = vs.index(var)
            except Exception:
                continue
            return syslbl, j
        return None, None

    coeffs = []
    for var in var_list:
        info = vmf_lookup(var, relatives=True, system_index=True)
        rel = info.get("relatives") if isinstance(info, dict) else None

        syslbl = rel.get("system_label") if isinstance(rel, dict) else None
        j = info.get("system_index") if isinstance(rel, dict) else None

        if syslbl is None or j is None:
            syslbl, j = _locate_in_vspaces(var)

        if syslbl is None or j is None:
            raise KeyError(
                f"VF_coeffs could not locate a system_label/system_index for a variable {var} in var_list."
            )

        coeffs.append(cd.get((j, 1, syslbl), 0))

    if sparse:
        out = [((i,), c) for i, c in enumerate(coeffs) if not _scalar_is_zero(c)]
        return out or [((0,), 0)]

    return coeffs


def addVF(*vf_args):
    """
    Adds the given vector fields (i.e., vector_field_class instances).
    This is a superfluous function preserved for backward-compatibility.
    """
    return sum(vf_args)


def scaleVF(scalar, vector_field):
    """
    Scales the given vector field.
    This is a superfluous function preserved for backward-compatibility.
    """
    return scalar * vector_field


def addDF(*args, doNotSimplify=False):
    """
    Adds the given vector fields (i.e., vector_field_class instances).
    This is a superfluous function preserved for backward-compatibility.
    """
    return sum(args)


def scaleDF(scalar, df):
    """
    Scales the given form
    This is a superfluous function preserved for backward-compatibility.
    """
    return scalar * df


def conjComplex(arg):
    dgcv_warning("`conjComplex` has been deprecated. Use `conjugate_dgcv` instead")
    return _conjComplexVFDF(arg)


def _conjComplexVFDF(arg):
    """
    Computes the complex conjugate of a tensor_field_class.
    """

    if get_dgcv_category(arg) == "tensor_field":
        return conjugate(arg)
    else:
        raise Exception("Expected the input to be a dgcv tensor_field.")
