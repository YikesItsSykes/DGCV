"""
_display.py

This module provides functions for customizing the display in Jupyter notebooks.
The `load_fonts()` function loads web fonts and applies them to the notebook
environment using HTML and CSS.

Functions
---------
- load_fonts: Injects HTML and CSS to load Google Fonts and apply custom fonts to the notebook.
"""


############# for printing
import sympy as sp
from IPython.display import HTML, Latex, display
from sympy.printing.latex import LatexPrinter

############# DGCV classes to format
from .dgcv_core import (
    DFClass,
    DGCVPolyClass,
    STFClass,
    VFClass,
    symToHol,
    symToReal,
    tensorField,
)
from .filtered_structures import Tanaka_symbol
from .finite_dim_algebras import AlgebraElement, FAClass
from .Riemannian_geometry import metricClass


def LaTeX(obj, removeBARs=False):
    """
    Custom LaTeX function for DGCV. Extends sympy.latex() to support application to DGCV classes.

    Parameters
    ----------
    obj : any
        The object to convert to LaTeX. Can be a SymPy expression, DGCV object, or a list/tuple of such objects.

    Returns
    -------
    str
        The LaTeX-formatted string.
    """

    def filter(term):
        if removeBARs:
            return sp.latex(term)
        if isinstance(term, (DFClass, VFClass, STFClass,tensorField, DGCVPolyClass)):
            if term._varSpace_type == "real":
                return sp.latex(symToReal(term))
            elif term._varSpace_type == "complex":
                return sp.latex(symToHol(term))
            else:
                return sp.latex(term)
        elif isinstance(term, FAClass):
            return _alglabeldisplayclass(term.label)._repr_latex_()
        elif isinstance(term, AlgebraElement):
            return _alglabeldisplayclass(term.algebra.label, term)._repr_latex_()
        elif isinstance(term, Tanaka_symbol):
            return "Tanaka_symbol Class"
        else:
            return sp.latex(symToHol(term))

    def strip_dollar_signs(latex_str):
        """Strip leading and trailing $ or $$ signs from a LaTeX string."""
        # if latex_str == None:
        #     return ''
        if latex_str is None:
            return latex_str
        if latex_str.startswith("$$") and latex_str.endswith("$$"):
            return latex_str[2:-2]
        if latex_str.startswith("$") and latex_str.endswith("$"):
            return latex_str[1:-1]
        return latex_str

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
        # Apply sympy.latex() for non-list/tuple objects
        return strip_dollar_signs(filter(obj))


def display_DGCV(*args):
    for j in args:
        _display_DGCV_single(j)


def _display_DGCV_single(arg):
    if isinstance(arg, str):
        display(Latex(arg))
    elif isinstance(
        arg,
        (sp.Expr, metricClass, DFClass, VFClass, STFClass,tensorField, DGCVPolyClass,Tanaka_symbol),
    ):
        _complexDisplay(arg)
    elif isinstance(arg, FAClass):
        _complexDisplay(_alglabeldisplayclass(arg.label))
    elif isinstance(arg, AlgebraElement):
        _complexDisplay(_alglabeldisplayclass(arg.algebra.label, ae=arg))
    else:
        display(arg)


def _complexDisplay(*args):
    """
    Taking DGCV expressions in *args* written in terms of symbolic conjugate variables, displays them with actual complex conjugates
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
            return _alglabeldisplayclass.format_algebra_label(self.label)
        else:
            return _alglabeldisplayclass.format_ae(self.ae)

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



# DGCV-specific SymPy LatexPrinter for VFClass and DFClass
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


# def DGCV_latex_printer(obj, **kwargs):
#     if isinstance(
#         obj,
#         (
#             VFClass,
#             DFClass,
#             STFClass,
#             tensorField,
#             metricClass,
#             FAClass,
#             AlgebraElement,
#             DGCVPolyClass,
#             Tanaka_symbol
#         ),
#     ):
#         latex_str = obj._repr_latex_()
#         return latex_str.strip("$")
#     elif isinstance(obj, (list, tuple)):
#         latex_elements = [DGCV_latex_printer(elem) for elem in obj]
#         return r"\left( " + r" , ".join(latex_elements) + r" \right)"
#     return latex(obj, **kwargs)

def DGCV_latex_printer(obj, **kwargs):
    if obj is None:
        return ''
    if LaTeX(obj) is None:
        return ''
    return LaTeX(obj).strip("$")

def DGCV_init_printing(*args, **kwargs):
    from sympy import init_printing

    kwargs["latex_printer"] = DGCV_latex_printer
    init_printing(*args, **kwargs)
