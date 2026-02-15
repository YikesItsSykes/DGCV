"""
package: dgcv - Differential Geometry with Complex Variables
module: _dgcv_display

Author (of this module): David Sykes (https://realandimaginary.com/dgcv/)

License:
    MIT License
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
from __future__ import annotations

import numbers
import warnings
from typing import Any

from ._safeguards import check_dgcv_category, get_dgcv_category
from .backends._display import latex as _backend_latex
from .backends._display_engine import is_rich_displaying_available
from .backends._engine import _get_sympy_module, is_sympy_available
from .backends._notebooks import (
    can_rich_display_latex,
    is_ipython_available,
)
from .backends._notebooks import display_html as _nb_display_html
from .backends._notebooks import display_latex as _nb_display_latex
from .backends._types_and_constants import expr_types
from .base import dgcv_class
from .conversions import symToHol, symToReal
from .printing._string_processing import (
    _coerce_to_str,
    _strip_display_dollars,
    _unwrap_math_delims,
)

__all__ = ["LaTeX", "LaTeX_eqn_system", "LaTeX_list", "show"]


# -----------------------------------------------------------------------------
# body
# -----------------------------------------------------------------------------
def _is_engine_expr(x: Any) -> bool:
    try:
        return isinstance(x, expr_types())
    except Exception:
        return False


def _try_symToHol(x: Any, removeBARs: bool) -> Any:
    if removeBARs:
        return x
    try:
        return symToHol(x, simplify_everything=False)
    except Exception:
        return x


def _latex_from_engine_expr(x: Any) -> str:
    s = _backend_latex(x)
    if not isinstance(s, str):
        return _coerce_to_str(x)
    return _unwrap_math_delims(s)


def _is_tensorish_dgcv(x: Any) -> bool:
    return get_dgcv_category(x) == "tensor_field"


def _has_varSpace_type(x: Any) -> bool:
    return getattr(x, "_varSpace_type", None) is not None


def LaTeX(obj: Any, removeBARs: bool = False) -> str:
    """
    Custom LaTeX function for dgcv. Attempts to produce a LaTeX-ish string for "mathy" objects.
    """

    def _latex_of(x: Any) -> str:
        if x is None:
            return ""

        if isinstance(x, list):
            elems = [_strip_display_dollars(_latex_of(e)) for e in x]
            return r"\left[ " + ", ".join(elems) + r" \right]"
        if isinstance(x, tuple):
            elems = [_strip_display_dollars(_latex_of(e)) for e in x]
            return r"\left( " + ", ".join(elems) + r" \right)"
        if isinstance(x, set):
            elems = [_strip_display_dollars(_latex_of(e)) for e in x]
            return r"\left\{ " + ", ".join(elems) + r" \right\}"

        if _is_tensorish_dgcv(x):
            if removeBARs:
                x2 = x
            else:
                vft = getattr(x, "_varSpace_type", None)
                if vft == "real":
                    x2 = symToReal(x)
                elif vft == "complex":
                    x2 = symToHol(x)
                else:
                    x2 = x

            if _is_engine_expr(x2):
                try:
                    return _latex_from_engine_expr(x2)
                except Exception:
                    return _coerce_to_str(x2)

            f = getattr(x2, "_repr_latex_", None)
            if callable(f):
                try:
                    s = f()
                    if isinstance(s, str):
                        return _unwrap_math_delims(s)
                except Exception:
                    pass
            return _coerce_to_str(x2)

        if check_dgcv_category(x):
            if not _has_varSpace_type(x):
                f = getattr(x, "_repr_latex_", None)
                if callable(f):
                    try:
                        s = f()
                        if isinstance(s, str):
                            return _unwrap_math_delims(s)
                    except Exception:
                        pass

            x2 = _try_symToHol(x, removeBARs)

            if _is_engine_expr(x2):
                try:
                    return _latex_from_engine_expr(x2)
                except Exception:
                    return _coerce_to_str(x2)

            f = getattr(x2, "_repr_latex_", None)
            if callable(f):
                try:
                    s = f()
                    if isinstance(s, str):
                        return _unwrap_math_delims(s)
                except Exception:
                    pass
            return _coerce_to_str(x2)

        x2 = _try_symToHol(x, removeBARs)

        if _is_engine_expr(x2):
            try:
                return _latex_from_engine_expr(x2)
            except Exception:
                return _coerce_to_str(x2)

        f = getattr(x2, "_repr_latex_", None)
        if callable(f):
            try:
                s = f()
                if isinstance(s, str):
                    return _unwrap_math_delims(s)
            except Exception:
                pass

        return _coerce_to_str(x2)

    return _strip_display_dollars(_latex_of(obj)) or ""


def LaTeX_eqn_system(
    eqn_dict,
    math_mode="$$",
    left_prefix="",
    left_suffix="",
    right_prefix="",
    right_suffix="",
    one_line=False,
    bare_latex=False,
    punctuation=None,
    add_period=False,
):
    if isinstance(eqn_dict, (list, tuple)):
        eqn_dict = {k: 0 for k in eqn_dict}
        list_format = True
    else:
        list_format = False

    if add_period is True:
        punct = "."
    elif isinstance(punctuation, str):
        punct = punctuation
    else:
        punct = ""

    if bare_latex is True:
        joiner = r", "
        boundary = ""
        penultim = r",\quad\text{and}\quad "
    elif math_mode == "$":
        joiner = "$, $"
        boundary = "$"
        penultim = "$, and $"
    elif one_line is True:
        joiner = r", \quad "
        boundary = "$$"
        penultim = r",\quad\text{and}\quad "
    else:
        joiner = r",$$ $$ "
        boundary = "$$"
        penultim = r",$$ and $$"

    if list_format is True:
        kv_pairs = [
            f"0={right_prefix}{LaTeX(k)}{right_suffix}" for k in eqn_dict.keys()
        ]
    else:
        kv_pairs = [
            f"{left_prefix}{LaTeX(k)}{left_suffix}={right_prefix}{LaTeX(v)}{right_suffix}"
            for k, v in eqn_dict.items()
        ]

    if len(kv_pairs) == 0:
        return punct
    if len(kv_pairs) == 1:
        return boundary + kv_pairs[0] + punct + boundary
    if len(kv_pairs) == 2:
        if bare_latex is True:
            return (
                boundary
                + kv_pairs[0]
                + r"\quad\text{and}\quad "
                + kv_pairs[1]
                + punct
                + boundary
            )
        if math_mode == "$":
            return (
                boundary
                + kv_pairs[0]
                + boundary
                + "and"
                + boundary
                + kv_pairs[1]
                + punct
                + boundary
            )
        if one_line is True:
            return (
                boundary
                + kv_pairs[0]
                + r" \quad \text{ and }\quad "
                + kv_pairs[1]
                + punct
                + boundary
            )
        return (
            boundary
            + kv_pairs[0]
            + boundary
            + "and"
            + boundary
            + kv_pairs[1]
            + punct
            + boundary
        )

    return (
        boundary
        + joiner.join(kv_pairs[:-1])
        + penultim
        + kv_pairs[-1]
        + punct
        + boundary
    )


def LaTeX_list(
    list_to_print,
    math_mode="$$",
    prefix="",
    suffix="",
    one_line=False,
    items_per_line=1,
    bare_latex=False,
    punctuation=None,
    item_labels=None,
):
    if not isinstance(list_to_print, (list, tuple)):
        if bare_latex is not True and (math_mode == "$" or math_mode == "$$"):
            return f"{math_mode}{LaTeX(list_to_print)}{math_mode}"
        return LaTeX(list_to_print)

    if (
        one_line is True
        or math_mode == "$"
        or not isinstance(items_per_line, numbers.Integral)
        or items_per_line < 1
    ):
        items_per_line = len(list_to_print)

    if not isinstance(item_labels, (list, tuple)):
        item_labels = []
    item_labels = [
        str(label) + " = "
        for label in item_labels[: min(len(item_labels), len(list_to_print))]
    ] + ([""]) * max(0, len(list_to_print) - len(item_labels))

    punct = punctuation if isinstance(punctuation, str) else ""

    if bare_latex is True:
        joiner = r", "
        boundary = ""
        penultim = r",\quad\text{and}\quad "
    elif math_mode == "$":
        joiner = "$, $"
        boundary = "$"
        penultim = "$, and $"
    elif items_per_line != 1:
        joiner = r", \quad "
        boundary = "$$"
        penultim = r",\quad\text{and}\quad "
    else:
        joiner = r",$$ $$ "
        boundary = "$$"
        penultim = r",$$ and $$"

    formatted_elems = [
        f"{j}{prefix}{LaTeX(k)}{suffix}" for j, k in zip(item_labels, list_to_print)
    ]
    formatted_chunks = [
        formatted_elems[j : j + items_per_line]
        for j in range(0, len(formatted_elems), items_per_line)
    ]

    def line_printer(formatted_items, conjunction=False, pun=","):
        if len(formatted_items) == 0:
            return pun
        if len(formatted_items) == 1:
            return boundary + formatted_items[0] + pun + boundary
        if len(formatted_items) == 2:
            if conjunction is False:
                insert = joiner
            else:
                insert = r"\quad\text{and}\quad "
            if bare_latex is True:
                return (
                    boundary
                    + formatted_items[0]
                    + insert
                    + formatted_items[1]
                    + pun
                    + boundary
                )
            if math_mode == "$":
                insert2 = ", " if conjunction is False else "and"
                return (
                    boundary
                    + formatted_items[0]
                    + boundary
                    + insert2
                    + boundary
                    + formatted_items[1]
                    + pun
                    + boundary
                )
            insert2 = joiner if conjunction is False else r" \quad \text{ and }\quad "
            return (
                boundary
                + formatted_items[0]
                + insert2
                + formatted_items[1]
                + pun
                + boundary
            )

        if conjunction is False:
            return boundary + joiner.join(formatted_items) + pun + boundary
        return (
            boundary
            + joiner.join(formatted_items[:-1])
            + penultim
            + formatted_items[-1]
            + pun
            + boundary
        )

    to_print = ""
    for fc in formatted_chunks[:-1]:
        to_print += line_printer(fc) + " "
    conjuction = (
        " and " if len(formatted_chunks) > 1 and len(formatted_chunks[-1]) == 1 else ""
    )
    return (
        to_print
        + conjuction
        + line_printer(formatted_chunks[-1], conjunction=True, pun=punct)
    )


def display_dgcv(*args):
    warnings.warn(
        "`display_dgcv` has been deprecated as part of a shift toward standardizing naming styles in the dgcv library."
        "`It` will be removed in 2026. Use the command `show` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return show(*args)


def display_DGCV(*args):
    warnings.warn(
        "`display_DGCV` has been deprecated as part of a shift toward standardizing naming styles in the dgcv library."
        "`It` will be removed in 2026. Use the command `show` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return show(*args)


def show(*args):
    """
    Display dgcv objects with IPython if available.

    If IPython rich display is not available/relevant, this function falls back to:
        print(str(obj))
    """
    if not is_rich_displaying_available():
        print(*args)
        return
    for j in args:
        _display_dgcv_single(j)


def _display_dgcv_single(arg: Any) -> None:
    if not can_rich_display_latex():
        print(_coerce_to_str(arg))
        return

    if isinstance(arg, str):
        _nb_display_latex(arg)
        return

    # If already an HTML-capable object, dispatch IPython
    if is_ipython_available():
        try:
            from IPython.display import display

            if hasattr(arg, "_repr_html_") or hasattr(arg, "to_html"):
                display(arg)
                return
        except Exception:
            pass

    if _is_engine_expr(arg):
        _complexDisplay(arg)
        return

    if isinstance(arg, dgcv_class) or check_dgcv_category(arg):
        _complexDisplay(arg)
        return

    if is_ipython_available():
        try:
            from IPython.display import Math, display

            display(Math(f"$\\displaystyle {LaTeX(arg)}$"))
            return
        except Exception:
            try:
                from IPython.display import display

                display(arg)
                return
            except Exception:
                pass

    print(_coerce_to_str(arg))


def _complexDisplay(*args):
    if not can_rich_display_latex():
        for j in args:
            print(_coerce_to_str(j))
        return

    def _to_math_payload(x: Any) -> str:
        if getattr(x, "_varSpace_type", None) in ("real", "complex"):
            try:
                return (
                    f"$\\displaystyle {LaTeX(symToHol(x, simplify_everything=False))}$"
                )
            except Exception:
                return f"$\\displaystyle {LaTeX(x)}$"
        return f"$\\displaystyle {LaTeX(x)}$"

    converted = [_to_math_payload(j) for j in args]

    if is_ipython_available():
        try:
            from IPython.display import Math, display

            for expr in converted:
                display(Math(expr))
            return
        except Exception:
            pass

    for j in converted:
        print(_coerce_to_str(j))


def load_fonts():
    """
    Inject google fonts in an IPython environment.

    If IPython isn't available, do nothing (no error).
    """
    if not can_rich_display_latex():
        return
    font_links = """
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Press+Start+2P&family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    <style>
    body {
        font-family: 'Roboto', sans-serif;
    }
    </style>
    """
    _nb_display_html(font_links)


def dgcv_init_printing(minimal_scope: bool = False, *args, **kwargs):
    """
    Initialize notebook display formatting for dgcv objects.

    With dgcv classes implementing `_repr_mimebundle_` via `dgcv_class`, there is
    nothing to register with IPython. This function optionally keeps SymPy's
    init_printing for users who want it.
    """
    if is_sympy_available() and minimal_scope is False:
        sp = _get_sympy_module()
        sp.init_printing(*args, **kwargs)
    return


def DGCV_init_printing(*args, **kwargs):
    warnings.warn(
        "`DGCV_init_printing` has been deprecated, as its functionality has been consolidated into the `set_dgcv_settings` function."
        "`It` will be removed in 2026. Run `set_dgcv_settings(format_displays=True)` instead to apply dgcv formatting in Jupyter notebooks.",
        DeprecationWarning,
        stacklevel=2,
    )
    return dgcv_init_printing(*args, **kwargs)
