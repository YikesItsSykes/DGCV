from __future__ import annotations

import textwrap

from ..._aux._backends._display_engine import is_rich_displaying_available
from ..._aux._utilities._config import (
    dgcv_warning,
    get_dgcv_settings_registry,
    latex_in_html,
)
from ..._aux._utilities._styles import get_style
from ..._aux._vmf._safeguards import (
    retrieve_passkey,
)
from ..._aux.printing._tables import build_matrix_table
from ..._aux.printing.printing._dgcv_display import show
from ..aec import algebra_element_class
from ..saec import subalgebra_element


def multiplication_table(
    target_alg,
    elements=None,
    restrict_to_subspace=False,
    theme=None,
    use_latex=None,
    plain_text: bool | None = None,
    return_displayable: bool = False,
    col_number_limit: int = 10,
    row_number_limit: int = 15,
    cell_char_lim: int = 20,
    table_css_properties: str = None,
    _called_from_subalgebra=None,
    **kwargs,
):
    if elements is None:
        elements = target_alg.basis
    elif not all(
        isinstance(elem, algebra_element_class) and elem.algebra == target_alg
        for elem in elements
    ):
        raise ValueError("All elements must be instances of algebraElement.") from None

    if restrict_to_subspace is True:
        basis_elements = elements
    elif isinstance(restrict_to_subspace, (list, tuple)) and all(
        isinstance(elem, algebra_element_class) and elem.algebra == target_alg
        for elem in restrict_to_subspace
    ):
        basis_elements = restrict_to_subspace
    elif (
        isinstance(_called_from_subalgebra, dict)
        and _called_from_subalgebra.get("internalLock", None) == retrieve_passkey()
    ):
        basis_elements = _called_from_subalgebra["basis"]
    else:
        basis_elements = target_alg.basis

    c_limited, r_limited = False, False
    if col_number_limit < len(elements):
        c_limited = True
        elements = elements[:col_number_limit]
    if row_number_limit < len(basis_elements):
        r_limited = True
        basis_elements = basis_elements[:row_number_limit]

    dgcvSR = get_dgcv_settings_registry()

    if not is_rich_displaying_available():
        plain_text = True

    if plain_text:
        from dgcv._aux.printing.printing._data_structures import (
            format_unicode_table,
        )

        c_aug = ["⋯"] if c_limited else []
        r_aug = ["⋮"] if r_limited else []
        headers = [str(e) for e in elements] + c_aug
        index_headers = [str(e) for e in basis_elements] + r_aug
        corner_aug = [" "] if c_limited else []
        data = []
        for left in basis_elements:
            data.append([str(left * right) for right in elements] + c_aug)
        if r_limited:
            data.append(["⋮" for _ in range(len(elements))] + corner_aug)

        out = format_unicode_table(
            data,
            row_labels=index_headers,
            column_labels=headers,
            caption="Multiplication Table",
            cell_char_lim=cell_char_lim,
            align="center",
            header_align="center",
            row_label_align="left",
        )

        if return_displayable:
            return out
        print(out)
        return

    if use_latex is None:
        use_latex = dgcvSR.get("use_latex", False)
    if not isinstance(theme, str):
        style_key = kwargs.get("style", None) or dgcvSR.get("theme", "dark")
    else:
        style_key = theme

    def _to_string(element, ul=False):
        if ul:
            s = element._repr_latex_(verbose=False)
            if s.startswith("$") and s.endswith("$"):
                s = s[1:-1]
            s = s.replace(r"\\displaystyle", "").replace(r"\displaystyle", "").strip()
            return f"${s}$"
        return str(element)

    headers = [_to_string(e, ul=use_latex) for e in elements]
    if c_limited:
        headers += [r"$\cdots$"] if use_latex else ["⋯"]
    index_headers = [_to_string(e, ul=use_latex) for e in basis_elements]
    if r_limited:
        index_headers += [r"$\vdots$"] if use_latex else ["︙"]

    data = []
    for left in basis_elements:
        row = [_to_string(left * right, ul=use_latex) for right in elements]
        if c_limited:
            row += [r"$\cdots$"] if use_latex else ["⋯"]
        data.append(row)
    if r_limited:
        corner_aug = [] if not c_limited else [r"$\ddots$"] if use_latex else ["⋱"]
        vdots = r"$\vdots$" if use_latex else "⋮"
        data.append([vdots for _ in range(len(elements))] + corner_aug)

    theme_string = get_style(style_key)

    extra_css_override = textwrap.dedent("""
        .dgcv-data-table { 
            table-layout: auto; 
        }
    """).strip()

    table = build_matrix_table(
        index_labels=index_headers,
        columns=headers,
        rows=data,
        caption="Multiplication Table",
        theme_css_vars=theme_string,
        extra_css=extra_css_override
        if table_css_properties is None
        else table_css_properties,
        mirror_header_to_index=True,
        dashed_corner=True,
        header_underline_exclude_index=True,
        cell_align="center",
        escape_cells=False,
        escape_headers=False,
        escape_index=False,
        table_scroll=True,
        nowrap=True,
        hover_mode="cell",
        ul=0,
        ur=0,
        ll=0,
        lr=0,
    )
    out = (
        latex_in_html(
            table,
            container_id=table.container_id,
            katex_selector=".dgcv-data-table",
        )
        if use_latex
        else latex_in_html(table, extra_support_for_math_in_tables=False)
    )
    if return_displayable:
        return out
    show(out)
    return


def _sa_multiplication_table(
    target_alg,
    elements=None,
    restrict_to_subspace=False,
    theme=None,
    use_latex=None,
    plain_text: bool | None = None,
    return_displayable: bool = False,
    col_number_limit: int = 10,
    row_number_limit: int = 15,
    cell_char_lim: int = 20,
    table_css_properties: str = None,
    **kwargs,
):
    if elements is None:
        newElements = [elem.ambient_rep for elem in target_alg.basis]
    elif isinstance(elements, (list, tuple)):
        warningMessage = ""
        newElements = []
        for elem in elements:
            elemTest = (
                elem.ambient_rep if isinstance(elem, subalgebra_element) else elem
            )
            if target_alg.contains(elemTest) is False:
                if warningMessage == "":
                    warningMessage += "Some elements in the `elements` list were not in the span of the subalgebra's basis, so they were omitted from the multiplication table."
            else:
                newElements.append(elemTest)
        if warningMessage != 0 and len(newElements) > 0:
            dgcv_warning(warningMessage)
        else:
            raise TypeError(
                "No elements from the provided `elements` list belong to the subalgebra, so a multiplication table will not be produced."
            ) from None
    else:
        raise TypeError(
            "If provided, the `elements` parameter in `subalgebra_class.multiplication_table` must be a list."
        ) from None

    return target_alg.ambient.multiplication_table(
        elements=newElements,
        restrict_to_subspace=restrict_to_subspace,
        theme=theme,
        use_latex=use_latex,
        plain_text=plain_text,
        return_displayable=return_displayable,
        col_number_limit=col_number_limit,
        row_number_limit=row_number_limit,
        cell_char_lim=cell_char_lim,
        table_css_properties=table_css_properties,
        _called_from_subalgebra={
            "internalLock": retrieve_passkey(),
            "basis": target_alg.basis,
        },
    )
