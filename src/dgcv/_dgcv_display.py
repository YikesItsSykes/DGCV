"""
_display.py

This module provides functions for adjusting display formatting in Jupyter notebooks.

"""


############# for printing
import numbers
import warnings

import sympy as sp
from IPython.display import HTML, Latex, display
from sympy.printing.latex import LatexPrinter

############# dgcv classes to format
from ._safeguards import check_dgcv_category, get_dgcv_category
from .dgcv_core import (
    DFClass,
    STFClass,
    VFClass,
    dgcvPolyClass,
    symToHol,
    symToReal,
    tensorField,
)
from .filtered_structures import Tanaka_symbol
from .Riemannian_geometry import metricClass


def LaTeX(obj, removeBARs=False):
    """
    Custom LaTeX function for dgcv. Extends sympy.latex() to support application to dgcv classes.

    Parameters
    ----------
    obj : any
        The object to convert to LaTeX. Can be a SymPy expression, dgcv object, or a list/tuple of such objects.

    Returns
    -------
    str
        The LaTeX-formatted string.
    """

    def filter(term):
        if removeBARs:
            return sp.latex(term)
        if isinstance(term, (DFClass, VFClass, STFClass, tensorField, dgcvPolyClass)):
            if term._varSpace_type == "real":
                return sp.latex(symToReal(term))
            elif term._varSpace_type == "complex":
                return sp.latex(symToHol(term))
            else:
                return sp.latex(term)
        elif get_dgcv_category(term) in {'algebra','algebra_element'}:
            return term._repr_latex_()
        elif get_dgcv_category(term)=='algebra_element':
            return _alglabeldisplayclass(term.algebra.label, term)._repr_latex_()
        elif isinstance(term, Tanaka_symbol):
            return "Tanaka_symbol Class"
        else:
            return sp.latex(symToHol(term))

    def strip_dollar_signs(latex_str):
        if latex_str is None:
            return latex_str
        return latex_str.replace("$", "")

    if isinstance(obj, list):
        latex_elements = [strip_dollar_signs(filter(elem)) for elem in obj]
        return r"\left[ " + ", ".join(latex_elements) + r" \right]"
    elif isinstance(obj, tuple):
        latex_elements = [strip_dollar_signs(filter(elem)) for elem in obj]
        return r"\left( " + ", ".join(latex_elements) + r" \right)"
    elif isinstance(obj, set):
        latex_elements = [strip_dollar_signs(filter(elem)) for elem in obj]
        return r"\left\{ " + ", ".join(latex_elements) + r" \right\}"

    else:
        return strip_dollar_signs(filter(obj))

def LaTeX_eqn_system(eqn_dict, math_mode = '$$', left_prefix = "", left_suffix = "", right_prefix = "", right_suffix = "", one_line = False, bare_latex = False, punctuation=None, add_period = False):
    if isinstance(eqn_dict,(list,tuple)):
        eqn_dict = {k:0 for k in eqn_dict}
        list_format = True
    else:
        list_format = False
    if add_period is True:
        punct = '.'
    elif isinstance(punctuation,str):
        punct = punctuation
    else:
        punct = ''
    if bare_latex is True:
        joiner = r', '
        boundary = ''
        penultim = r',\quad\text{and}\quad '
    elif math_mode == '$':
        joiner = '$, $'
        boundary = '$'
        penultim = '$, and $'

    elif one_line is True:
        joiner = r', \quad '
        boundary = '$$'
        penultim = r',\quad\text{and}\quad '
    else:
        joiner = r',$$ $$ '
        boundary = '$$'
        penultim = r',$$ and $$'

    if list_format is True:
        kv_pairs = [f'0={right_prefix}{LaTeX(k)}{right_suffix}' for k in eqn_dict.keys()]
    else:
        kv_pairs = [f'{left_prefix}{LaTeX(k)}{left_suffix}={right_prefix}{LaTeX(v)}{right_suffix}' for k,v in eqn_dict.items()]
    if len(kv_pairs)==0:
        return punct
    elif len(kv_pairs)==1:
        return boundary + kv_pairs[0] + punct + boundary
    elif len(kv_pairs)==2:
        if bare_latex is True:
            return boundary + kv_pairs[0] + r'\quad\text{and}\quad ' + kv_pairs[1] + punct + boundary
        if math_mode == '$':
            return boundary + kv_pairs[0] + boundary + 'and' + boundary + kv_pairs[1] + punct + boundary
        if one_line is True:
            return boundary + kv_pairs[0] + r' \quad \text{ and }\quad ' + kv_pairs[1] + punct + boundary
        return boundary + kv_pairs[0] + boundary + 'and' + boundary + kv_pairs[1] + punct + boundary
    return boundary+joiner.join(kv_pairs[:-1])+penultim + kv_pairs[-1] + punct + boundary

def LaTeX_list(list_to_print, math_mode = '$$', prefix = "", suffix = "", one_line = False, items_per_line = 1, bare_latex = False, punctuation = None, item_labels = None):
    if not isinstance(list_to_print,(list,tuple)):
        if bare_latex is not True and (math_mode == '$' or math_mode=='$$'):
            return f'{math_mode}{LaTeX(list_to_print)}{math_mode}'
        return LaTeX(list_to_print)
    if one_line is True or math_mode=='$' or not isinstance(items_per_line,numbers.Integral) or items_per_line<1:
        items_per_line = len(list_to_print)
    if not isinstance(item_labels,(list,tuple)):
        item_labels=[]
    item_labels = [str(label)+' = ' for label in item_labels[:min(len(item_labels),len(list_to_print))]]+([''])*max(0,len(list_to_print)-len(item_labels))
    if isinstance(punctuation,str):
        punct = punctuation
    else:
        punct = ''
    if bare_latex is True:
        joiner = r', '
        boundary = ''
        penultim = r',\quad\text{and}\quad '
    elif math_mode == '$':
        joiner = '$, $'
        boundary = '$'
        penultim = '$, and $'
    elif items_per_line!=1:
        joiner = r', \quad '
        boundary = '$$'
        penultim = r',\quad\text{and}\quad '
    else:
        joiner = r',$$ $$ '
        boundary = '$$'
        penultim = r',$$ and $$'

    formatted_elems = [f'{j}{prefix}{LaTeX(k)}{suffix}' for j,k in zip(item_labels,list_to_print)]
    formatted_chunks = [formatted_elems[j:j+items_per_line] for j in range(0, len(formatted_elems), items_per_line)]
    def line_printer(formatted_items, conjunction = False, pun = ','):
        if len(formatted_items)==0:
            return pun
        elif len(formatted_items)==1:
            return boundary + formatted_items[0] + pun + boundary
        elif len(formatted_items)==2:
            if conjunction is False:
                insert = joiner
            else:
                insert = r'\quad\text{and}\quad '
            if bare_latex is True:
                return boundary + formatted_items[0] + insert + formatted_items[1] + pun + boundary
            if math_mode == '$':
                if conjunction is False:
                    insert = ', '
                else:
                    insert = 'and'
                return boundary + formatted_items[0] + boundary + insert + boundary + formatted_items[1] + pun + boundary
            if conjunction is False:
                insert = joiner
            else:
                insert = r' \quad \text{ and }\quad '
            return boundary + formatted_items[0] + insert + formatted_items[1] + pun + boundary
        if conjunction is False:
            return boundary+joiner.join(formatted_items) + pun + boundary
        else:
            return boundary+joiner.join(formatted_items[:-1])+penultim + formatted_items[-1] + pun + boundary
    to_print = ''
    for fc in formatted_chunks[:-1]:
        to_print+=line_printer(fc)+' '
    if len(formatted_chunks)>1 and len(formatted_chunks[-1])==1:
        conjuction = ' and '
    else:
        conjuction = ''
    return to_print + conjuction +line_printer(formatted_chunks[-1],conjunction=True,pun=punct)

def display_DGCV(*args):
    warnings.warn(
        "`display_DGCV` has been deprecated as part of a shift toward standardizing naming styles in the dgcv library."
        "`It` will be removed in 2026. Use the command `show` instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return show(*args)

def show(*args):
    for j in args:
        _display_DGCV_single(j)

def _display_DGCV_single(arg):
    if isinstance(arg, str):
        to_print=Latex(arg)
        display(to_print)
    elif isinstance(
        arg,
        (sp.Expr, metricClass, DFClass, VFClass, STFClass, tensorField, dgcvPolyClass, Tanaka_symbol) or check_dgcv_category(arg),
    ):
        _complexDisplay(arg)
    else:
        display(arg)

def _complexDisplay(*args):
    """
    Taking dgcv expressions in *args* written in terms of symbolic conjugate variables, displays them with actual complex conjugates
    """
    display(*[symToHol(j, simplify_everything=False) for j in args])

class _alglabeldisplayclass(sp.Basic):

    def __new__(cls, label, ae=None):
        obj = sp.Basic.__new__(cls, label)
        return obj

    def __init__(self, label, ae=None):
        self.label = str(label)
        self.ae = ae

    @staticmethod
    def format_algebra_label(label):
        r"""Wrap the algebra label in \mathfrak{} if all characters are lowercase, and subscript any numeric suffix."""
        if label[-1].isdigit():
            # Split into text and number parts for subscript formatting
            label_text = "".join(filter(str.isalpha, label))
            label_number = "".join(filter(str.isdigit, label))
            if label_text.islower():
                return rf"\mathfrak{{{label_text}}}_{{{label_number}}}"
            return rf"{label_text}_{{{label_number}}}"
        elif label.islower():
            return rf"\mathfrak{{{label}}}"
        return label

    @staticmethod
    def format_ae(ae):
        if ae is not None:
            terms = []
            for coeff, basis_label in zip(ae.coeffs, ae.algebra.basis_labels):
                if coeff == 0:
                    continue  # Skip zero terms
                elif coeff == 1:
                    terms.append(rf"{basis_label}")  # Suppress 1 as coefficient
                elif coeff == -1:
                    terms.append(
                        rf"-{basis_label}"
                    )  # Suppress 1 but keep the negative sign
                else:
                    # Check if the coefficient has more than one term (e.g., 1 + I)
                    if isinstance(coeff, sp.Expr) and len(coeff.args) > 1:
                        terms.append(
                            rf"({sp.latex(coeff)}) \cdot {basis_label}"
                        )  # Wrap multi-term coefficient in parentheses
                    else:
                        terms.append(
                            rf"{sp.latex(coeff)} \cdot {basis_label}"
                        )  # Single-term coefficient

            # Handle special case: all zero coefficients
            if not terms:
                return rf"$0 \cdot {ae.algebra.basis_labels[0]}$"

            # Join terms with proper LaTeX sign handling
            result = " + ".join(terms).replace("+ -", "- ")
            return rf"{result}"

    def _repr_latex_(self):
        if self.ae:
            return _alglabeldisplayclass.format_ae(self.ae)
        else:
            return _alglabeldisplayclass.format_algebra_label(self.label)

    def __str__(self):
        return self.label

def load_fonts():
    font_links = """
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Press+Start+2P&family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    <style>
    body {
        font-family: 'Roboto', sans-serif;
    }
    </style>
    """
    display(HTML(font_links))

class DGCVLatexPrinter(LatexPrinter):
    def _print_VFClass(self, expr):
        return expr._repr_latex_()

    def _print_DFClass(self, expr):
        return expr._repr_latex_()

def DGCV_collection_latex_printer(obj):
    if isinstance(obj, (tuple, list)):
        return tuple(
            Latex(element._repr_latex_() if hasattr(element, "_repr_latex_") else sp.latex(element))
            for element in obj
        )
    return None

def DGCV_latex_printer(obj, **kwargs):
    if obj is None:
        return ''
    if LaTeX(obj) is None:
        return ''
    return LaTeX(obj).strip("$")

def DGCV_init_printing(*args, **kwargs):
    warnings.warn(
        "`DGCV_init_printing` has been deprecated, as its functionality has been consolidated into the `set_dgcv_settings` function."
        "`It` will be removed in 2026. Run `update_dgcv_settings(format_displays=True)` instead to apply dgcv formatting in Jupyter notebooks.",
        DeprecationWarning,
        stacklevel=2
    )
    return dgcv_init_printing(*args, **kwargs)

def dgcv_init_printing(*args, **kwargs):
    from sympy import init_printing

    kwargs["latex_printer"] = DGCV_latex_printer
    init_printing(*args, **kwargs)
