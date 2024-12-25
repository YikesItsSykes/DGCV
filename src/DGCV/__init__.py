
from .config import cache_globals, get_variable_registry, configure_warnings

############# Variable Management Framework (VMF) tools
# Initialize the globals cache when DGCV is imported
cache_globals()

# Initialize variable_registry when DGCV is imported
_ = get_variable_registry()

############# library
from .classesAndVariables import DFClass, VFClass, STFClass, TFClass, DGCVPolyClass, FAClass, AlgebraElement, createVariables, holToReal, realToSym, symToHol, holToSym, realToHol, symToReal, allToReal, allToHol, allToSym, complex_struct_op, conjugate_DGCV, conj_with_real_coor, re_with_real_coor, im_with_real_coor, conj_with_hol_coor, re_with_hol_coor, im_with_hol_coor, cleanUpConjugation, compressDGCVClass, VF_coeffs, changeVFBasis, addVF, scaleVF, VF_bracket, changeDFBasis, changeTFBasis, changeSTFBasis, scaleDF, addDF, exteriorProduct, addSTF, addTF, scaleTF, tensorProduct, holVF_coeffs, antiholVF_coeffs, complexVFC, conjComplex, realPartOfVF, createFiniteAlg, algebraDataFromVF, algebraDataFromMatRep, killingForm, adjointRepresentation, listVar, clearVar, DGCV_snapshot, variableSummary
from .combinatorics import chooseOp, permSign, carProd
from .complexStructures import Del, DelBar, KahlerStructure
from .CRGeometry import tangencyObstruction, weightedHomogeneousVF, findWeightedCRSymmetries, model2Nondegenerate
from .polynomials import createPolynomial, createBigradPolynomial, monomialWeight, getWeightedTerms
from .RiemannianGeometry import metricClass, metric_from_matrix, LeviCivitaConnectionClass
from .vectorFieldsAndDifferentialForms import get_VF, get_DF, assembleFromHolVFC, assembleFromAntiholVFC, assembleFromCompVFC, makeZeroForm, exteriorDerivative, interiorProduct, LieDerivative, decompose, get_coframe, annihilator
from .coordinateMaps import coordinate_map
from .DGCV_display import load_fonts, display_DGCV, LaTeX
from .styles import get_DGCV_themes
from sympy import I, re, im, conjugate, simplify

############# for printing
from sympy import latex, Basic  # Rename SymPy's latex
from IPython.display import Latex, display  # Rename IPython's Latex
from sympy.printing.latex import LatexPrinter

############# warnings formatting
configure_warnings()



############# DGCV-specific SymPy LatexPrinter for VFClass and DFClass
class DGCVLatexPrinter(LatexPrinter):
    def _print_VFClass(self, expr):
        return expr._repr_latex_()  # Keep the surrounding $

    def _print_DFClass(self, expr):
        return expr._repr_latex_()  # Keep the surrounding $


def DGCV_collection_latex_printer(obj):
    if isinstance(obj, (tuple, list)):
        latex_elements = []
        for element in obj:
            if isinstance(element, VFClass):
                latex_elements.append(Latex(element._repr_latex_()))  # Use Latex for Jupyter rendering
            elif isinstance(element, DFClass):
                latex_elements.append(Latex(element._repr_latex_()))
            elif isinstance(element, FAClass):
                latex_elements.append(Latex(element._repr_latex_()))
            elif isinstance(element, AlgebraElement):
                latex_elements.append(Latex(element._repr_latex_()))
            elif isinstance(element, DGCVPolyClass):
                latex_elements.append(Latex(element._repr_latex_()))
            else:
                latex_elements.append(Latex(latex(element)))  
         
        # Return a tuple of individually rendered LaTeX objects
        return tuple(latex_elements)
    return None

############# DGCV latex_printer function that uses DGCVLatexPrinter and accepts keyword arguments
def DGCV_latex_printer(obj, **kwargs):
    # If the object is a DGCV class, use its custom LaTeX method
    if isinstance(obj, (VFClass, DFClass, TFClass,STFClass,metricClass, FAClass, AlgebraElement, DGCVPolyClass)):
        latex_str = obj._repr_latex_()
        # Remove surrounding $ signs
        if latex_str.startswith("$") and latex_str.endswith("$"):
            latex_str = latex_str[1:-1]
        if latex_str.startswith("$") and latex_str.endswith("$"):
            latex_str = latex_str[1:-1]
        return latex_str
    # If the object is a list or tuple, process each element separately
    elif isinstance(obj, (list, tuple)):
        latex_elements = [DGCV_latex_printer(elem) for elem in obj]  # Recursively process each element
        # Join with commas and enclose with \left( ... \right) for proper LaTeX wrapping
        return r"\left( " + r" , ".join(latex_elements) + r" \right)"
    # Otherwise, use SymPy's default LaTeX printer
    return latex(obj, **kwargs)

def DGCV_init_printing(*args, **kwargs):
    """Initialize DGCV's custom printing and load fonts."""
    load_fonts()  # Load custom fonts
    from sympy import init_printing

    # Register the DGCV printer with SymPy's init_printing system
    kwargs['latex_printer'] = DGCV_latex_printer  # Use DGCV custom LaTeX printer
    init_printing(*args, **kwargs)  # Call SymPy's init_printing
