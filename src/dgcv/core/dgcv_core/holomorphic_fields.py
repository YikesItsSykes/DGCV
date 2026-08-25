from ..._aux._backends._symbolic_router import _scalar_is_zero, conjugate, simplify
from ..._aux._backends._types_and_constants import imag_unit
from ..._aux._vmf._safeguards import get_dgcv_category, query_dgcv_categories
from ..conversions.conversions import realToHol


def holVF_coeffs(vf, arg2: list | tuple, doNotSimplify=False) -> tuple:
    """
    Evaluates the vector field (i.e., vector_field_class instance) *arg1* on each holomorphic variable in *arg2*,
    and returns the result as a list of coefficients.

    The variables in *arg2* must be previously initialized via complexVarProc. The function returns the
    coefficients of the holomorphic part when the vector field is expressed in terms of holomorphic coordinate
    vector fields.

    Parameters:
    -----------
    arg1 : vector_field_class
        A vector field instance to evaluate on the holomorphic variables.
    arg2 : list or tuple
        A list or tuple of Symbol objects that were initialized as holomorphic variables via complexVarProc.
    doNotSimplify : bool, optional
        If True, the results are returned without simplification (default is False).

    Returns:
    --------
    list
        A list of symbolic expressions representing the coefficients in holomorphic coordinates.
    """
    if doNotSimplify:
        return [realToHol(vf(j)) for j in arg2]
    else:
        return [simplify(realToHol(vf(j))) for j in arg2]


def antiholVF_coeffs(vf, arg2: list | tuple, doNotSimplify=False) -> tuple:
    """
    Evaluates the vector field (i.e., vector_field_class instance) *arg1* on the conjugate of each holomorphic variable
    in *arg2*, and returns the result as a list of coefficients.

    The variables in *arg2* must be previously initialized via complexVarProc. The function returns the
    coefficients of the holomorphic part when the vector field is expressed in terms of holomorphic coordinate
    vector fields.

    Parameters:
    -----------
    arg1 : vector_field_class
        A vector field instance to evaluate on the holomorphic variables.
    arg2 : list or tuple
        A list or tuple of Symbol objects that were initialized as holomorphic variables via complexVarProc.
    doNotSimplify : bool, optional
        If True, the results are returned without simplification (default is False).

    Returns:
    --------
    list
        A list of symbolic expressions representing the coefficients in antiholomorphic coordinates.
    """
    if doNotSimplify:
        return [realToHol(vf(conjugate(j))) for j in arg2]
    else:
        return [simplify(realToHol(vf(conjugate(j)))) for j in arg2]


def complexVFC(vf, arg2: list | tuple, doNotSimplify=False) -> tuple:
    """
    Evaluates the vector field (i.e., vector_field_class instance) *arg1* on the holomorphic variables in *arg2*
    and their complex conjugates, returning the result as two lists of coefficients.

    The variables in *arg2* must be previously initialized via complexVarProc. The function returns the
    coefficients for both the holomorphic and antiholomorphic parts of the vector field when expressed in
    terms of the respective coordinate vector fields.

    Parameters:
    -----------
    arg1 : vector_field_class
        A vector field instance to evaluate on the holomorphic and antiholomorphic variables.
    arg2 : list or tuple
        A list or tuple of Symbol objects that were initialized as holomorphic variables via complexVarProc.
    doNotSimplify : bool, optional
        If True, the results are returned without simplification (default is False).

    Returns:
    --------
    tuple of two lists
        The first list contains the coefficients of the holomorphic part, and the second list contains
        the coefficients of the antiholomorphic part.

    """
    if query_dgcv_categories(vf, {"vector_field"}):
        hol_coeffs = holVF_coeffs(vf, arg2, doNotSimplify=doNotSimplify)
        antihol_coeffs = antiholVF_coeffs(vf, arg2, doNotSimplify=doNotSimplify)
        return hol_coeffs, antihol_coeffs
    else:
        raise Exception(
            "Expected first positional argument to be of type vector_field_class"
        )


def realPartOfVF(vf, *args):
    """
    Computes the real part of a complex vector field *vf*.
    """
    if query_dgcv_categories(vf, {"vector_field"}):
        return vf.real_part()
    else:
        raise Exception("Expected the input to be of type vector_field_class.")


def complex_struct_op(vf):
    if get_dgcv_category(vf) == "distribution":
        return vf.apply(complex_struct_op)

    if not query_dgcv_categories(vf, {"vector_field"}):
        raise TypeError(
            "complex_struct_op expects a vector_field instance or distribution."
        )

    imu = imag_unit()
    vst = vf.variable_spaces_types

    new_cd = {}

    for k, c in vf.coeff_dict.items():
        if _scalar_is_zero(c):
            continue

        if k == tuple():
            new_cd[tuple()] = new_cd.get(tuple(), 0) + c
            continue

        d = len(k) // 3
        idxs = list(k[:d])
        valence_tuple = k[d : 2 * d]
        syslbls = k[2 * d :]

        for slot in range(d):
            sys = syslbls[slot]
            sys_data = vst.get(sys)

            if sys_data is None or sys_data.get("type") != "complex":
                continue

            b0, b1, b2 = sys_data["breaks"]
            idx = idxs[slot]

            if idx < b0:  # holo
                c = imu * c
            elif idx < b1:  # anti
                c = -imu * c
            elif idx < b2:  # real
                idxs[slot] = idx + (b2 - b1)  # -> imag
            else:  # imag
                idxs[slot] = idx - (b2 - b1)  # -> real
                c = -c

        nk = tuple(idxs) + valence_tuple + syslbls
        new_cd[nk] = new_cd.get(nk, 0) + c

    if not new_cd:
        new_cd = {tuple(): 0}

    return vf.__class__(
        coeff_dict=new_cd,
        data_shape=getattr(vf, "data_shape", "all"),
        dgcvType=getattr(vf, "dgcvType", "standard"),
        _simplifyKW=getattr(vf, "_simplifyKW", None),
        variable_spaces=getattr(vf, "_variable_spaces", None),
    )
