from __future__ import annotations

from ..._backends._display import latex as _backend_latex
from ..._backends._symbolic_router import _scalar_is_zero
from ._scalars import (
    _print_style,
    _scalar_is_minus_one,
    _scalar_is_one,
    _scalar_mul,
    _shape_joiner,
)
from ._string_processing import (
    _process_var_label,
    coeff_needs_parens_latex,
    coeff_needs_parens_plain,
)


def _tf2_split_key(key: tuple):
    if not isinstance(key, tuple):
        raise TypeError("tensorField key must be a tuple")
    n = len(key)
    if n % 3 != 0:
        raise ValueError("tensorField key length must be divisible by 3")
    deg = n // 3
    return deg, key[:deg], key[deg : 2 * deg], key[2 * deg :]


def _tf2_get_varspaces(tensor):
    vs = getattr(tensor, "_variable_spaces", None)
    if isinstance(vs, dict) and vs:
        return vs
    vs = getattr(tensor, "variable_spaces", None)
    if isinstance(vs, dict) and vs:
        return vs
    return {}


def _tf2_lookup_var(varspaces: dict, sys_label, idx):
    try:
        vs = varspaces[sys_label]
    except Exception:
        raise KeyError(
            f"tensorField references coordinate system '{sys_label}' not present in its cached variable spaces."
        ) from None
    if not isinstance(vs, tuple | dict):
        vs = tuple(vs)
        varspaces[sys_label] = vs
    return vs[idx]


def _tf2_plain_coeff_prefix(scalar) -> str:
    smul = _scalar_mul("plain")
    style = _print_style()

    if _scalar_is_one(scalar):
        return ""
    if _scalar_is_minus_one(scalar):
        return "-" if style == "literal" else "- "
    s = str(scalar)
    if coeff_needs_parens_plain(s):
        return f"({s}){smul}"
    return f"{s}{smul}"


def _tf2_latex_coeff(scalar) -> str:
    if _scalar_is_one(scalar):
        return ""
    if _scalar_is_minus_one(scalar):
        return "-"
    s = _backend_latex(scalar)
    if coeff_needs_parens_latex(s):
        return rf"\left({s}\right)"
    return s


def tensor_field_printer2(tensor) -> str:
    terms = getattr(tensor, "coeff_dict", None)
    if not isinstance(terms, dict):
        raise TypeError("tensorField.coeff_dict must be a dict")

    varspaces = _tf2_get_varspaces(tensor)
    joiner = _shape_joiner(getattr(tensor, "data_shape", "general"), "plain")

    formatted_terms: list[str] = []
    for key, scalar in terms.items():
        if _scalar_is_zero(scalar):
            continue

        deg, idxs, vals, syss = _tf2_split_key(key)

        if deg == 0:
            formatted_terms.append(str(scalar))
            continue

        if not all(v in (0, 1) for v in vals):
            raise ValueError("tensorField valence entries must be 0/1")

        basis_elems: list[str] = []
        for j in range(deg):
            var = (
                f"{{dgcv_par_{idxs[j]}}}"
                if syss[j] == "__dgcv_par__"
                else _tf2_lookup_var(varspaces, syss[j], idxs[j])
            )
            basis_elems.append(f"D_{var}" if vals[j] == 1 else f"d_{var}")

        basis = joiner.join(basis_elems)
        formatted_terms.append(f"{_tf2_plain_coeff_prefix(scalar)}{basis}")

    if not formatted_terms:
        return "0"

    out = formatted_terms[0]
    for t in formatted_terms[1:]:
        out += t if t.startswith("-") else f"+{t}"
    return out


def tensor_field_latex2(tensor, raw: bool = False) -> str:
    terms = getattr(tensor, "coeff_dict", None)
    if not isinstance(terms, dict):
        raise TypeError("tensorField.coeff_dict must be a dict")

    varspaces = _tf2_get_varspaces(tensor)
    joiner = _shape_joiner(getattr(tensor, "data_shape", "general"), "latex")

    formatted_terms: list[str] = []
    for key, scalar in terms.items():
        if _scalar_is_zero(scalar):
            continue

        deg, idxs, vals, syss = _tf2_split_key(key)

        if deg == 0:
            formatted_terms.append(_backend_latex(scalar))
            continue

        if not all(v in (0, 1) for v in vals):
            raise ValueError("tensorField valence entries must be 0/1")

        basis_elems: list[str] = []

        def handle_idx(j):
            if syss[j] == "__dgcv_par__":
                lab = f"dgcv_par_{idxs[j]}"
            else:
                var = _tf2_lookup_var(varspaces, syss[j], idxs[j])
                lab = _process_var_label(var)
            basis_elems.append(
                rf"\frac{{\partial}}{{\partial {lab}}}"
                if vals[j] == 1
                else rf"\operatorname{{d}} {lab}"
            )

        for j in range(deg):
            handle_idx(j)

        basis = joiner.join(basis_elems)

        c = _tf2_latex_coeff(scalar)
        if c == "":
            formatted_terms.append(basis)
        elif c == "-":
            formatted_terms.append(rf"- {basis}")
        else:
            formatted_terms.append(rf"{c} {basis}")

    latex_str = (
        "0" if not formatted_terms else " + ".join(formatted_terms).replace("+ -", "- ")
    )
    return latex_str if raw else f"${latex_str}$"
