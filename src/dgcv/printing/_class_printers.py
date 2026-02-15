"""
package: dgcv - Differential Geometry with Complex Variables
module: printing/_class_printers

Author (of this module): David Sykes (https://realandimaginary.com/dgcv/)

License:
    MIT License
"""

# -----------------------------------------------------------------------------
# imports
# -----------------------------------------------------------------------------
from __future__ import annotations

import re
from typing import Any

from .._config import from_vsr, get_dgcv_settings_registry
from ..backends._display import latex as _backend_latex
from ._string_processing import (
    _format_label_with_hi_low,
    _latex_escape_text,
    _process_var_label,
    coeff_needs_parens_latex,
    coeff_needs_parens_plain,
    convert_to_greek,
    latex_superscript,
)

# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------

joinders = {
    "literal": {
        "general": {"latex": r"\otimes ", "plain": "@"},
        "symmetric": {"latex": r"\odot ", "plain": "&"},
        "skew": {"latex": r"\wedge ", "plain": "*"},
        "dual": {"latex": r"*", "plain": r"^''"},
        "scalar_mul": {"plain": "*", "latex": " "},
    },
    "readable": {
        "general": {"latex": r"\otimes ", "plain": " ⊗ "},
        "symmetric": {"latex": r"\odot ", "plain": " ⊙ "},
        "skew": {"latex": r"\wedge ", "plain": " ∧ "},
        "dual": {"latex": r"*", "plain": "^*"},
        "scalar_mul": {"plain": " * ", "latex": " "},
    },
}


def _print_style() -> str:
    style = get_dgcv_settings_registry().get("print_style")
    return style if style in joinders else "literal"


def _shape_joiner(shape: str, fmt: str) -> str:
    style = _print_style()
    return joinders[style].get(shape, joinders[style]["general"])[fmt]


def _scalar_mul(fmt: str = "plain") -> str:
    style = _print_style()
    return joinders[style]["scalar_mul"][fmt]


def _dual_marker(fmt: str = "plain") -> str:
    style = _print_style()
    return joinders[style]["dual"][fmt]


# -----------------------------------------------------------------------------
# Scalar helpers
# -----------------------------------------------------------------------------


def _scalar_plain_string(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        try:
            return repr(x)
        except Exception:
            return ""


def _scalar_latex_string(x: Any) -> str:
    try:
        return _backend_latex(x)
    except Exception:
        return _scalar_plain_string(x)


def _scalar_is_zero(x: Any) -> bool:
    iz = getattr(x, "is_zero", None)
    if isinstance(iz, bool):
        return iz
    if callable(iz):
        try:
            return bool(iz())
        except Exception:
            pass
    try:
        return x == 0
    except Exception:
        return False


def _scalar_is_one(x: Any) -> bool:
    io = getattr(x, "is_one", None)
    if isinstance(io, bool):
        return io
    if callable(io):
        try:
            return bool(io())
        except Exception:
            pass
    try:
        return x == 1
    except Exception:
        return False


def _scalar_is_minus_one(x: Any) -> bool:
    im1 = getattr(x, "is_minus_one", None)
    if isinstance(im1, bool):
        return im1
    if callable(im1):
        try:
            return bool(im1())
        except Exception:
            pass
    try:
        return x == -1
    except Exception:
        return False


# -----------------------------------------------------------------------------
# vector space and tensorProduct printers
# -----------------------------------------------------------------------------


def tensor_VS_printer(tp) -> str:
    terms = tp.coeff_dict
    smul = _scalar_mul("plain")
    joiner = _shape_joiner("general", "plain")
    dual_plain = _dual_marker("plain")

    def coeff_prefix(scalar) -> str:
        if _scalar_is_one(scalar):
            return ""
        if _scalar_is_minus_one(scalar):
            return "-"
        s = _scalar_plain_string(scalar)
        if coeff_needs_parens_plain(s):
            return f"({s}){smul}"
        return f"{s}{smul}"

    BL: dict[Any, list[str]] = {}

    def labler(idx, vsidx) -> str:
        if vsidx not in BL:
            vsl = from_vsr(vsidx)
            BL[vsidx] = vsl.basis_labels or [
                f"VS{vsl.vector_spaces.index(vsidx)}_{j + 1}"
                for j in range(vsl.dimension)
            ]
        return BL[vsidx][idx]

    formatted_terms: list[str] = []
    for vec, scalar in terms.items():
        if _scalar_is_zero(scalar):
            continue
        n = len(vec) // 3
        idx = vec[:n]
        valence = vec[n : 2 * n]
        vs = vec[2 * n :]

        basis_elements = [
            labler(idx[j], vs[j])
            if valence[j] == 1
            else f"{labler(idx[j], vs[j])}{dual_plain}"
            for j in range(n)
        ]
        basis = joiner.join(basis_elements)
        formatted_terms.append(f"{coeff_prefix(scalar)}{basis}")

    if not formatted_terms:
        vec = next(iter(terms.keys()))
        n = len(vec) // 3
        idx = vec[:n]
        vs = vec[2 * n :]
        basis = joiner.join(labler(j, k) for j, k in zip(idx, vs))
        return f"0{basis}"

    out = formatted_terms[0]
    for t in formatted_terms[1:]:
        out += t if t.startswith("-") else f"+{t}"
    return out


def tensor_latex_helper(tp) -> str:
    terms = tp.coeff_dict
    joiner = _shape_joiner("general", "latex")
    dual_latex = _dual_marker("latex")

    def coeff_latex(scalar, bypass=None) -> str:
        if _scalar_is_one(scalar):
            return "" if bypass != "" else "1"
        if _scalar_is_minus_one(scalar):
            return "-" if bypass != "" else "-1"
        s = _scalar_latex_string(scalar)
        if coeff_needs_parens_latex(s):
            return rf"\left({s}\right)"
        return s

    def labler(idx, vsidx) -> str:
        vsl = from_vsr(vsidx)
        return vsl.basis[idx]._repr_latex_(raw=True)

    def _pwrap_for_dual(base: str) -> str:
        # only for things like x^2, so x^2^* becomes (x^2)^*
        # intentionally minimal: basis labels are close to atomic
        if "^" in base and not base.rstrip().endswith(")"):
            return rf"\left({base}\right)"
        return base

    formatted_terms: list[str] = []
    for vec, scalar in terms.items():
        if _scalar_is_zero(scalar):
            continue
        n = len(vec) // 3
        idx = vec[:n]
        valence = vec[n : 2 * n]
        vs = vec[2 * n :]

        basis_elements = [
            labler(idx[j], vs[j])
            if valence[j] == 1
            else rf"{_pwrap_for_dual(labler(idx[j], vs[j]))}^{dual_latex}"
            for j in range(n)
        ]
        basis = joiner.join(basis_elements)

        c = coeff_latex(scalar, basis)
        if c == "":
            formatted_terms.append(basis)
        elif c == "-":
            formatted_terms.append(rf"- {basis}")
        else:
            formatted_terms.append(rf"{c} {basis}")

    if not formatted_terms:
        vec = next(iter(terms.keys()))
        n = len(vec) // 3
        idx = vec[:n]
        vs = vec[2 * n :]
        basis = joiner.join(labler(j, k) for j, k in zip(idx, vs))
        return "0" + basis

    out = formatted_terms[0]
    for t in formatted_terms[1:]:
        out += t if t.startswith("-") else f" + {t}"
    return out


def lincomb_plain(
    coeffs,
    labels,
    *,
    valence: int = 1,
    label_transform=None,
    fallback_label: str = "e_1",
    include_zero_term: bool = True,
) -> str:
    style = _print_style()
    smul = _scalar_mul("plain")
    dual_plain = _dual_marker("plain")

    def _bl(lbl: str) -> str:
        s = label_transform(lbl) if callable(label_transform) else lbl
        return s if valence == 1 else f"{s}{dual_plain}"

    def _coeff_prefix(c) -> str:
        if _scalar_is_one(c):
            return ""
        if _scalar_is_minus_one(c):
            return "-"
        s = _scalar_plain_string(c)
        if coeff_needs_parens_plain(s):
            return f"({s}){smul}"
        return f"{s}{smul}"

    terms: list[str] = []
    for c, lbl in zip(coeffs, labels):
        if _scalar_is_zero(c):
            continue

        bl = _bl(str(lbl))

        if _scalar_is_one(c):
            terms.append(bl)
        elif _scalar_is_minus_one(c):
            terms.append(f"-{bl}")
        else:
            terms.append(f"{_coeff_prefix(c)}{bl}")

    if not terms:
        if not include_zero_term:
            return "0"
        return f"0{smul}{_bl(str(fallback_label))}"

    if style == "literal":
        out = terms[0]
        for t in terms[1:]:
            out += t if t.startswith("-") else f"+{t}"
        return out

    return " + ".join(terms).replace("+ -", "- ")


def lincomb_latex(
    coeffs,
    labels=None,
    *,
    vectorSpace=None,
    valence: int = 1,
    fallback_label: str | None = None,
    verbose: bool = False,
    raw: bool = False,
    apply_vlp_trim: bool = False,
):
    if labels is None:
        if vectorSpace is None:
            raise ValueError("lincomb_latex expects either labels or vectorSpace")
        if getattr(vectorSpace, "_tex_basis_labels", None) is not None:
            labels = vectorSpace._tex_basis_labels
            proDone = True
        elif getattr(vectorSpace, "basis_labels", None) is not None:
            labels = vectorSpace.basis_labels
            proDone = False
        else:
            labels = [
                f"_e_{{{i + 1}}}" for i in range(getattr(vectorSpace, "dimension", 0))
            ]
            proDone = False
    else:
        proDone = False

    if fallback_label is None:
        bl = (
            getattr(vectorSpace, "basis_labels", None)
            if vectorSpace is not None
            else None
        )
        fallback_label = str(bl[0]) if isinstance(bl, (list, tuple)) and bl else "e_1"

    def _process_label(lbl: str) -> str:
        if proDone:
            return lbl

        if vectorSpace is not None and getattr(
            vectorSpace, "_basis_labels_parent", False
        ):
            m = re.search(r"(_\d+)$", lbl)
            if m:
                tail = f"_{{{m.group(1)[1:]}}}"
                return (
                    f"\\left({vectorSpace._repr_latex_(abbrev=True, raw=True)}\\right)"
                    + tail
                )

        m = re.search(r"(\d+)$", lbl)
        if m and "_" not in lbl:
            lbl = lbl[: m.start(1)] + "_" + m.group(1)
        return _format_label_with_hi_low(lbl)

    def _coeff_tex(c) -> str:
        if _scalar_is_one(c):
            return ""
        if _scalar_is_minus_one(c):
            return "-"
        s = _scalar_latex_string(c)
        if coeff_needs_parens_latex(s):
            return rf"\left({s}\right)"
        return s

    def _star(lbl: str) -> str:
        return latex_superscript(lbl, "*")

    terms: list[str] = []
    for c, lbl in zip(coeffs, labels):
        if _scalar_is_zero(c):
            continue

        bl = _process_label(str(lbl))
        bl = bl if valence == 1 else _star(bl)

        if _scalar_is_one(c):
            terms.append(bl)
            continue
        if _scalar_is_minus_one(c):
            terms.append(rf"-{bl}")
            continue

        ct = _coeff_tex(c)
        terms.append(rf"{ct} {bl}")

    if not terms:
        if verbose:
            out0 = rf"0 {fallback_label}"
            return out0 if raw else rf"${out0}$"
        return "0" if raw else "$0$"

    result = " + ".join(terms).replace("+ -", "- ")

    if apply_vlp_trim:
        reg = get_dgcv_settings_registry()
        if reg.get("verbose_label_printing") is False:
            m = reg["VLP"].match(result)
            if m and m.group("content") is not None:
                result = m.group("content")
            elif m:
                result = result[: result.rfind("_{\\operatorname{v.}")]

    return result if raw else rf"$\displaystyle {result}$"


def space_display(
    *,
    fmt: str,
    basis_tokens,
    dim: int | None = None,
    label: str | None = None,
    label_tex: str | None = None,
    mathfrak_label: bool = False,
    unlabeled_plain: str = "Unnamed",
    unlabeled_tex: str = r"\text{Unnamed}",
    max_dim: int = 20,
    raw: bool = False,
    abbrev: bool = False,
    use_displaystyle: bool = True,
    plain_wrapper: str = "<{}>",
    latex_wrapper: str = r"\langle {}\rangle",
    label_only_if_not_vlp: bool = False,
):
    reg = get_dgcv_settings_registry()
    vlp = bool(reg.get("verbose_label_printing", False))

    if fmt not in ("plain", "latex"):
        raise ValueError("fmt must be 'plain' or 'latex'")

    def _truncate(tokens: list[str]) -> list[str]:
        d = len(tokens) if dim is None else dim
        if d <= max_dim or len(tokens) <= max_dim:
            return tokens
        k = max_dim // 2
        return tokens[:k] + ["..."] + tokens[-k:]

    def _label_tex(label0: str | None, *, mathfrak: bool) -> str:
        if not label0:
            return unlabeled_tex

        s = str(label0)

        if "_" in s:
            main, sub = s.split("_", 1)
            main_conv = convert_to_greek(main)
            if mathfrak and main.islower() and main_conv == main:
                main_conv = rf"\mathfrak{{{main}}}"
            return rf"{main_conv}_{{{sub}}}"

        if s[-1].isdigit():
            head = "".join(ch for ch in s if ch.isalpha())
            tail = "".join(ch for ch in s if ch.isdigit())
            head_conv = convert_to_greek(head)
            if mathfrak and head.islower() and head_conv == head:
                head_conv = rf"\mathfrak{{{head}}}"
            return rf"{head_conv}_{{{tail}}}"

        head_conv = convert_to_greek(s)
        if mathfrak and s.islower() and head_conv == s:
            return rf"\mathfrak{{{s}}}"
        return head_conv

    if fmt == "latex":
        lab = (
            label_tex
            if label_tex is not None
            else _label_tex(label, mathfrak=mathfrak_label)
        )
    else:
        lab = None

    if abbrev:
        if fmt == "plain":
            return str(label) if label else unlabeled_plain
        return lab if raw else f"${lab}$"

    if (not vlp) and label_only_if_not_vlp:
        if fmt == "plain":
            return str(label) if label else unlabeled_plain
        return (
            lab
            if raw
            else (rf"$\displaystyle {lab}$" if use_displaystyle else f"${lab}$")
        )

    toks = list(basis_tokens() if callable(basis_tokens) else (basis_tokens or []))
    toks = [str(t) for t in toks]
    toks = _truncate(toks)

    if fmt == "plain":
        core = plain_wrapper.format(", ".join(toks))
        if vlp and label:
            return f"{label}={core}"
        return core

    inner = ", ".join(toks).replace("...", r"\dots")
    core = latex_wrapper.format(inner)

    if vlp and (label or label_tex):
        core = f"{lab}={core}"

    if raw:
        return core
    return rf"$\displaystyle {core}$" if use_displaystyle else f"${core}$"


# -----------------------------------------------------------------------------
# tensor_field_class printers (v2 key format)
# -----------------------------------------------------------------------------


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
    if not isinstance(vs, tuple):
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
    s = _scalar_plain_string(scalar)
    if coeff_needs_parens_plain(s):
        return f"({s}){smul}"
    return f"{s}{smul}"


def _tf2_latex_coeff(scalar) -> str:
    if _scalar_is_one(scalar):
        return ""
    if _scalar_is_minus_one(scalar):
        return "-"
    s = _scalar_latex_string(scalar)
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
            formatted_terms.append(_scalar_plain_string(scalar))
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
            formatted_terms.append(_scalar_latex_string(scalar))
            continue

        if not all(v in (0, 1) for v in vals):
            raise ValueError("tensorField valence entries must be 0/1")

        basis_elems: list[str] = []
        for j in range(deg):
            if syss[j] == "__dgcv_par__":
                lab = f"dgcv_par_{idxs[j]}"
            else:
                var = _tf2_lookup_var(varspaces, syss[j], idxs[j])
                lab = _process_var_label(var)
            basis_elems.append(
                rf"\frac{{\partial}}{{\partial {lab}}}" if vals[j] == 1 else rf"d {lab}"
            )

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


# -----------------------------------------------------------------------------
# arrays
# -----------------------------------------------------------------------------


def array_VS_printer(A, max_rows=12, max_cols=12):
    shape = getattr(A, "shape", None)
    if not shape:
        return "array_dgcv(?)"
    if len(shape) == 2:
        r, c = shape
        rr = min(r, max_rows)
        cc = min(c, max_cols)
        lines = []
        for i in range(rr):
            row = [A[i, j] for j in range(cc)]
            row_str = ", ".join("∅" if x is None else str(x) for x in row)
            if cc < c:
                row_str += ", …"
            lines.append("[" + row_str + "]")
        if rr < r:
            lines.append("…")
        head = f"{A.__class__.__name__}(shape={shape})"
        return head + "\n" + "\n".join(lines)
    return f"{A.__class__.__name__}(shape={shape}, ndim={len(shape)})"


def _array_latex_2d(A, env="bmatrix"):
    from dgcv._dgcv_display import LaTeX

    r, c = A.shape
    rows = []
    for i in range(r):
        entries = []
        for j in range(c):
            x = A[i, j]
            entries.append("" if x is None else LaTeX(x))
        rows.append(" & ".join(entries))
    body = r" \\ ".join(rows)
    return rf"\begin{{{env}}}{body}\end{{{env}}}"


def _array_latex_nd(A, max_total_entries=800):
    shape = A.shape
    total = 1
    for s in shape:
        total *= s
    if total > max_total_entries:
        sh = _latex_escape_text(str(shape))
        return rf"\text{{array}}(\text{{shape}}={sh})"

    def rec(prefix, axis):
        from dgcv._dgcv_display import LaTeX

        if axis == len(shape) - 1:
            parts = []
            for i in range(shape[axis]):
                x = A[tuple(prefix + [i])]
                parts.append("" if x is None else LaTeX(x))
            inner = r",\, ".join(parts)
            return rf"\left[{inner}\right]"
        parts = []
        for i in range(shape[axis]):
            parts.append(rec(prefix + [i], axis + 1))
        inner = r",\, ".join(parts)
        return rf"\left[{inner}\right]"

    return rec([], 0)


def array_latex_helper(A, env="bmatrix"):
    if getattr(A, "shape", None) is None:
        return r"\text{array}(\text{shape}=? )"
    if len(A.shape) == 2:
        return _array_latex_2d(A, env=env)
    return _array_latex_nd(A)
