
"""
display.py

This module provides functions for customizing font display in Jupyter notebooks.
The `load_fonts()` function loads web fonts and applies them to the notebook
environment using HTML and CSS.

Functions
---------
- load_fonts: Injects HTML and CSS to load Google Fonts and apply custom fonts to the notebook.
"""
import sympy
from .classesAndVariables import DFClass,VFClass,STFClass,FAClass,DGCVPolyClass,symToHol,symToReal,TFClass,AlgebraElement
from .RiemannianGeometry import metricClass

from IPython.display import display, HTML, Latex



from sympy import latex, Basic

def LaTeX(obj,removeBARs=False):
    """
    Custom LaTeX function for DGCV.
    Applies sympy.latex() to the input object, with special handling for lists and tuples.
    
    For lists, the elements are LaTeX-processed, stripped of surrounding $ or $$ symbols if present,
    and wrapped in \left[ ... \right].
    
    For tuples, the elements are LaTeX-processed similarly but wrapped in \left( ... \right).
    
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
        if removeBARs==True:
            return latex(term)
        if isinstance(term,(DFClass,VFClass,STFClass,DGCVPolyClass,TFClass)):
            if term._varSpace_type == 'real':
                return latex(symToReal(term))
            elif term._varSpace_type == 'complex':
                return latex(symToHol(term))
            else:
                return latex(term)
        elif isinstance(term, FAClass):
            return _alglabeldisplayclass(term.label)._repr_latex_()
        elif isinstance(term, AlgebraElement):
            return _alglabeldisplayclass(term.algebra.label,term)._repr_latex_()
        else:
            return latex(symToHol(term))
        
    def strip_dollar_signs(latex_str):
        """Strip leading and trailing $ or $$ signs from a LaTeX string."""
        # if latex_str == None:
        #     return ''
        if latex_str.startswith("$$") and latex_str.endswith("$$"):
            return latex_str[2:-2]
        elif latex_str.startswith("$") and latex_str.endswith("$"):
            return latex_str[1:-1]
        return latex_str

    if isinstance(obj, list):
        # Handle lists by applying LaTeX and wrapping with \left[ ... \right]
        latex_elements = [strip_dollar_signs(filter(elem)) for elem in obj]
        return r"\left[ " + ", ".join(latex_elements) + r" \right]"
    
    elif isinstance(obj, tuple):
        # Handle tuples by applying LaTeX and wrapping with \left( ... \right)
        latex_elements = [strip_dollar_signs(filter(elem)) for elem in obj]
        return r"\left( " + ", ".join(latex_elements) + r" \right)"
    
    else:
        # Apply sympy.latex() for non-list/tuple objects
        return strip_dollar_signs(filter(obj))


def display_DGCV(*args):
    for j in args:
        _display_DGCV_single(j)

def _display_DGCV_single(arg):
    if isinstance(arg,str):
        display(Latex(arg))
    elif isinstance(arg,(sympy.Expr,metricClass,DFClass,VFClass,TFClass,STFClass,DGCVPolyClass)):
        _complexDisplay(arg)
    elif isinstance(arg,FAClass):
        _complexDisplay(_alglabeldisplayclass(arg.label))
    elif isinstance(arg,AlgebraElement):
        _complexDisplay(_alglabeldisplayclass(arg.algebra.label,ae=arg))
    else:
        display(arg)

def _complexDisplay(*args):
    """
    Taking DGCV expressions in *args* written in terms of symbolic conjugate variables, displays them with actual complex conjugates
    """
    display(*[symToHol(j,simplify_everything=False) for j in args])


class _alglabeldisplayclass(Basic):

    def __new__(cls, label, ae = None):
        obj = Basic.__new__(cls, label)
        return obj

    def __init__(self, label, ae = None):
        self.label = str(label)
        self.ae = ae

    @staticmethod
    def format_algebra_label(label):
        """Wrap the algebra label in \mathfrak{} if all characters are lowercase, and subscript any numeric suffix."""
        if label[-1].isdigit():
            # Split into text and number parts for subscript formatting
            label_text = ''.join(filter(str.isalpha, label))
            label_number = ''.join(filter(str.isdigit, label))
            if label_text.islower():
                return rf"\mathfrak{{{label_text}}}_{{{label_number}}}"
            return rf"{label_text}_{{{label_number}}}"
        elif label.islower():
            return rf"\mathfrak{{{label}}}"
        return label
    
    @staticmethod
    def format_ae(ae):
        if ae != None:
            terms = []
            for coeff, basis_label in zip(ae.coeffs, ae.algebra.basis_labels):
                if coeff == 0:
                    continue  # Skip zero terms
                elif coeff == 1:
                    terms.append(rf"{basis_label}")  # Suppress 1 as coefficient
                elif coeff == -1:
                    terms.append(rf"-{basis_label}")  # Suppress 1 but keep the negative sign
                else:
                    # Check if the coefficient has more than one term (e.g., 1 + I)
                    if isinstance(coeff, sympy.Expr) and len(coeff.args) > 1:
                        terms.append(rf"({sympy.latex(coeff)}) \cdot {basis_label}")  # Wrap multi-term coefficient in parentheses
                    else:
                        terms.append(rf"{sympy.latex(coeff)} \cdot {basis_label}")  # Single-term coefficient

            # Handle special case: all zero coefficients
            if not terms:
                return rf"$0 \cdot {ae.algebra.basis_labels[0]}$"

            # Join terms with proper LaTeX sign handling
            result = " + ".join(terms).replace("+ -", "- ")
            return rf"{result}"
    
    def _repr_latex_(self):
        if self.ae == None:
            return _alglabeldisplayclass.format_algebra_label(self.label)
        else:
            return _alglabeldisplayclass.format_ae(self.ae)
    
    def __str__(self):
        return self.label


def load_fonts():
    """
    Load custom web fonts into the Jupyter notebook.
    
    This function injects HTML and CSS to load fonts from Google Fonts and sets 
    the body font to 'Roboto'. It is intended for use in Jupyter notebooks to 
    customize the appearance of text.

    Fonts Loaded
    ------------
    - Orbitron
    - Press Start 2P
    - Roboto

    Examples
    --------
    >>> load_fonts()

    Notes
    -----
    The fonts are applied globally to the notebook interface via CSS.
    """
    font_links = """
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Press+Start+2P&family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    <style>
    body {
        font-family: 'Roboto', sans-serif;
    }
    </style>
    """
    display(HTML(font_links))