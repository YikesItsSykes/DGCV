"""
DGCV: Package Initialization

The DGCV package integrates tools for differential geometry with a framework for conveniently working with complex variables. The `__init__.py` module initializes core components of the package.

Initialization:
    - Global Cache and Variable Management Framework: Automatically sets up the global cache and variable registry
      systems that underly DGCV's Variable Management Framework (VMF). The VMF tracks and caches 
      relationships between variables (of coordinate systems) and related objects, and it is fundamental
      in much of the library's functionalities.
    - Warnings Configuration: Configures DGCV-specific warning behaviors.

Usage Notes:
    - To start using DGCV, simply import the package:
    ```python
    import DGCV

DGCV: Package Initialization

The DGCV package integrates tools for differential geometry with a framework for conveniently working with complex variables. The `__init__.py` module initializes core components of the package.


Initialization:
    - Global Cache and Variable Management: Automatically sets up the global cache and variable registry
      systems that underly DGCV's Variable Management Framework (VMF). The VMF tracks and caches 
      relationships between variables (of coordinate systems) and related objects, and it is fundamental
      in much of the library's functionalities.
    - Warnings Configuration: Configures DGCV-specific warning behaviors.

Usage Notes:
- To start using DGCV, simply import the package:
  ```python
  import DGCV
  ```
- SymPy utilities frequently used with DGCV (e.g., `I`, `conjugate`, `im`, `re`, `simplify`) should be imported 
  directly from SymPy:
  ```python
  from sympy import I, conjugate, im, re, simplify
  ```
- The package initialization automatically configures the VMF and integrates DGCV-specific warnings.

Dependencies:
    - sympy: Provides foundational symbolic computation tools.
    - IPython: Supports enhanced output display for Jupyter notebooks.

Author: David Sykes (https://www.realandimaginary.com/dgcv/)

License:
    MIT License

"""

# Imports
from IPython.display import Latex
from sympy import latex
from sympy.printing.latex import LatexPrinter

from ._DGCV_display import LaTeX, display_DGCV, load_fonts
from .combinatorics import carProd, chooseOp, permSign
from .complexStructures import Del, DelBar, KahlerStructure
from .config import cache_globals, configure_warnings, get_variable_registry
from .coordinateMaps import coordinate_map
from .CRGeometry import (
    findWeightedCRSymmetries,
    model2Nondegenerate,
    tangencyObstruction,
    weightedHomogeneousVF,
)
from .DGCVCore import (
    DFClass,
    DGCV_snapshot,
    DGCVPolyClass,
    STFClass,
    TFClass,
    VF_bracket,
    VF_coeffs,
    VFClass,
    addDF,
    addSTF,
    addTF,
    addVF,
    allToHol,
    allToReal,
    allToSym,
    antiholVF_coeffs,
    changeDFBasis,
    changeSTFBasis,
    changeTFBasis,
    changeVFBasis,
    cleanUpConjugation,
    clearVar,
    complex_struct_op,
    complexVFC,
    compressDGCVClass,
    conj_with_hol_coor,
    conj_with_real_coor,
    conjComplex,
    conjugate_DGCV,
    createVariables,
    exteriorProduct,
    holToReal,
    holToSym,
    holVF_coeffs,
    im_with_hol_coor,
    im_with_real_coor,
    listVar,
    re_with_hol_coor,
    re_with_real_coor,
    realPartOfVF,
    realToHol,
    realToSym,
    scaleDF,
    scaleTF,
    scaleVF,
    symToHol,
    symToReal,
    tensorProduct,
    variableSummary,
)
from .finiteDimAlgebras import (
    AlgebraElement,
    FAClass,
    adjointRepresentation,
    algebraDataFromMatRep,
    algebraDataFromVF,
    createFiniteAlg,
    killingForm,
)
from .polynomials import (
    createBigradPolynomial,
    createPolynomial,
    getWeightedTerms,
    monomialWeight,
)
from .RiemannianGeometry import (
    LeviCivitaConnectionClass,
    metric_from_matrix,
    metricClass,
)
from .styles import get_DGCV_themes
from .vectorFieldsAndDifferentialForms import (
    LieDerivative,
    annihilator,
    assembleFromAntiholVFC,
    assembleFromCompVFC,
    assembleFromHolVFC,
    decompose,
    exteriorDerivative,
    get_coframe,
    get_DF,
    get_VF,
    interiorProduct,
    makeZeroForm,
)

# Initialize global caches and variable registry
cache_globals()
_ = get_variable_registry()

# Configure warnings
configure_warnings()

# Default functions/classes
__all__ = [
    ############ DGCV default functions/classes ####
    # From config
    "cache_globals",        # Initialize global caches
    "configure_warnings",   # Configure DGCV warnings
    "get_variable_registry",# Get the DGCV variable registry

    # From _DGCV_display
    "LaTeX",                # Custom LaTeX renderer
    "load_fonts",           # Load custom fonts for LaTeX output
    "display_DGCV",         # Augments IPython.display.display
                            # with support for DGCV object like
                            # custom latex rendering

    # From combinatorics
    "carProd",              # Cartesian product
    "chooseOp",             # Choose operation
    "permSign",             # Permutation sign

    # From complexStructures
    "Del",                  # Holomorphic derivative operator
    "DelBar",               # Anti-holomorphic derivative operator
    "KahlerStructure",      # Represents a Kähler structure

    # From coordinateMaps
    "coordinate_map",       # Transforms coordinates systems

    # From CRGeometry
    "findWeightedCRSymmetries", # Find weighted CR symmetries
    "model2Nondegenerate",  # Produces a 2-nond. model structure
    "tangencyObstruction",  # Obstruction for VF to be tangent to submanifold
    "weightedHomogeneousVF",# Produce general weighted homogeneous vector fields

    # From DGCVCore
    "DFClass",              # Differential form class
    "DGCVPolyClass",        # DGCV polynomial class
    "DGCV_snapshot",        # Summarize initialized DGCV objects
    "STFClass",             # Symmetric tensor field class
    "TFClass",              # Tensor field class
    "VFClass",              # Vector field class
    "VF_bracket",           # Lie bracket of vector fields
    "VF_coeffs",            # Coefficients of vector fields
    "addDF",                # Add differential forms
    "addSTF",               # Add symmetric tensor fields
    "addTF",                # Add tensor fields
    "addVF",                # Add vector fields
    "allToHol",             # Convert DGCV expressions to holomorphic
                            # coordinate format
    "allToReal",            # Convert all fields to real
                            # coordinate format
    "allToSym",             # Convert all fields to symbolic
                            # conjugate coordinate format
    "antiholVF_coeffs",     # Anti-holomorphic coefficients of vector field
    "changeDFBasis",        # Change basis for differential forms
    "changeSTFBasis",       # Change basis for symmetric tensor fields
    "changeTFBasis",        # Change basis for tensor fields
    "changeVFBasis",        # Change basis for vector fields
    "cleanUpConjugation",   # Cleanup conjugation operations
    "clearVar",             # Clear DGCV objects from globals()
    "complexVFC",           # Complex coordingate vector field coefficients
    "complex_struct_op",    # Complex structure operator
    "compressDGCVClass",    # 
    "conjComplex",          # Conjugate complex variables
    "conj_with_hol_coor",   # Conjugate with holomorphic coordinate formatting
    "conj_with_real_coor",  # Conjugate with real coordinate formatting
    "conjugate_DGCV",       # Conjugate DGCV objects
    "createVariables",      # Initialize variables in DGCV's VMF
    "exteriorProduct",      # Compute exterior product
    "holToReal",            # Convert holomorphic to real format
    "holToSym",             # Convert holomorphic to symbolic conjugates format
    "holVF_coeffs",         # Holomorphic coefficients of vector field
    "im_with_hol_coor",     # Imaginary part with holomorphic coordinate format
    "im_with_real_coor",    # Imaginary part with real coordinate format
    "listVar",              # List objects from the DGCV VMF
    "realPartOfVF",         # Real part of vector fields
    "realToHol",            # Convert real to holomorphic fomrat
    "realToSym",            # Convert real to symbolic conjugates format
    "re_with_hol_coor",     # Real part with holomorphic coordinate format
    "re_with_real_coor",    # Real part with real coordinate format
    "scaleDF",              # Scale differential forms
    "scaleTF",              # Scale tensor fields
    "scaleVF",              # Scale vector fields
    "symToHol",             # Convert symbolic conjugates to holomorphic format
    "symToReal",            # Convert symbolic conjugates to real format
    "tensorProduct",        # Compute tensor product
    "variableSummary",      # Depricated - use DGCV_snapshot instead

    # From finiteDimAlgebras
    "AlgebraElement",       # Algebra element class
    "FAClass",              # Finite dimensional algebra class
    "adjointRepresentation",# Adjoint representation of algebra
    "algebraDataFromMatRep",# Algebra data from matrix representation
    "algebraDataFromVF",    # Algebra data from vector fields
    "createFiniteAlg",      # Create a finite dimensional algebra
    "killingForm",          # Compute the Killing form

    # From polynomials
    "createBigradPolynomial",# Create bigraded polynomial
    "createPolynomial",     # Create polynomial
    "getWeightedTerms",     # Get weighted terms of a polynomial
    "monomialWeight",       # Compute monomial weights

    # From RiemannianGeometry
    "LeviCivitaConnectionClass", # Levi-Civita connection class
    "metric_from_matrix",   # Create metric from matrix
    "metricClass",          # Metric class

    # From styles
    "get_DGCV_themes",      # Get DGCV themes for various output styles

    # From vectorFieldsAndDifferentialForms
    "LieDerivative",        # Compute Lie derivative
    "annihilator",          # Compute annihilator
    "assembleFromAntiholVFC",# Assemble VF from anti-holomorphic VF coefficients
    "assembleFromCompVFC",  # Assemble VF from complex VF coefficients
    "assembleFromHolVFC",   # Assemble VF from holomorphic VF coefficients
    "decompose",            # Decompose objects into linear combinations
    "exteriorDerivative",   # Compute exterior derivative
    "get_coframe",          # Get coframe from frame
    "get_DF",               # Get differential form from label in VMF
    "get_VF",               # Get vector field from label in VMF
    "interiorProduct",      # Compute interior product
    "makeZeroForm",         # Create zero-form from scalar
]

# DGCV-specific SymPy LatexPrinter for VFClass and DFClass
class DGCVLatexPrinter(LatexPrinter):
    def _print_VFClass(self, expr):
        return expr._repr_latex_()

    def _print_DFClass(self, expr):
        return expr._repr_latex_()


def DGCV_collection_latex_printer(obj):
    if isinstance(obj, (tuple, list)):
        return tuple(
            Latex(element._repr_latex_() if hasattr(element, "_repr_latex_") else latex(element))
            for element in obj
        )
    return None


def DGCV_latex_printer(obj, **kwargs):
    if isinstance(
        obj,
        (
            VFClass,
            DFClass,
            TFClass,
            STFClass,
            metricClass,
            FAClass,
            AlgebraElement,
            DGCVPolyClass,
        ),
    ):
        latex_str = obj._repr_latex_()
        return latex_str.strip("$")
    elif isinstance(obj, (list, tuple)):
        latex_elements = [DGCV_latex_printer(elem) for elem in obj]
        return r"\left( " + r" , ".join(latex_elements) + r" \right)"
    return latex(obj, **kwargs)


def DGCV_init_printing(*args, **kwargs):
    load_fonts()
    from sympy import init_printing

    kwargs["latex_printer"] = DGCV_latex_printer
    init_printing(*args, **kwargs)
