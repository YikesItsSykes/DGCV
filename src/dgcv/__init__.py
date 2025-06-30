"""
dgcv: Package Initialization

The dgcv package integrates tools for differential geometry with a framework for conveniently working with complex variables. The `__init__.py` module initializes core components of the package.

Initialization:
    - Global Cache and Variable Management Framework: Automatically sets up the global cache and variable registry systems that underly dgcv's Variable Management Framework (VMF). The VMF tracks and caches relationships between variables (of coordinate systems) and related objects, and it is fundamental in much of the library's functionalities.
    - Warnings Configuration: Configures dgcv-specific warning behaviors.

Dependencies:
    - sympy: Provides foundational symbolic computation tools.
    - pandas: Used for displaying tables of data
    - IPython: Supports enhanced output display for Jupyter notebooks.

Author: David Sykes (https://www.realandimaginary.com/dgcv/)

License:
    MIT License

"""

# Imports

# getting current version for dgcv settings defaults
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dgcv")
except PackageNotFoundError:
    __version__ = "unknown"




from ._config import cache_globals, configure_warnings, get_variable_registry

############# Variable Management Framework (VMF) tools
# Initialize the globals pointer cache when dgcv is imported
cache_globals()

# Initialize variable_registry when dgcv is imported
_ = get_variable_registry()
import numbers

from ._config import canonicalize
from ._dgcv_display import (
    DGCV_init_printing,
    LaTeX,
    LaTeX_eqn_system,
    LaTeX_list,
    display_DGCV,
    show,
)
from ._settings import set_dgcv_settings
from .algebras.algebras_aux import algebraDataFromMatRep, algebraDataFromVF
from .algebras.algebras_core import (
    adjointRepresentation,
    algebra_class,
    algebra_element_class,
    algebra_subspace_class,
    killingForm,
)
from .algebras.algebras_secondary import (
    createAlgebra,
    createFiniteAlg,  # deprecated
    createSimpleLieAlgebra,
    subalgebra_class,
)
from .backends._sage_backend import get_sage_module, is_sage_available
from .combinatorics import carProd, chooseOp, permSign
from .complex_structures import Del, DelBar, KahlerStructure
from .coordinate_maps import coordinate_map
from .CR_geometry import (
    findWeightedCRSymmetries,
    model2Nondegenerate,
    tangencyObstruction,
    weightedHomogeneousVF,
)
from .dgcv_core import (
    DFClass,
    DGCVPolyClass,  # deprecated
    STFClass,
    VF_bracket,
    VF_coeffs,
    VFClass,
    addDF,
    addSTF,
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
    complex_struct_op,
    complexVFC,
    compress_dgcv_class,
    compressDGCVClass,  # deprecated
    conj_with_hol_coor,
    conj_with_real_coor,
    conjComplex,
    conjugate_DGCV,  # deprecated
    conjugate_dgcv,
    createVariables,
    dgcvPolyClass,
    exteriorProduct,
    holToReal,
    holToSym,
    holVF_coeffs,
    im_with_hol_coor,
    im_with_real_coor,
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
    tensor_product,
    tensorField,
)
from .eds import (
    DF_representation,
    abst_coframe,
    abstract_DF,
    abstract_ZF,
    coframe_derivative,
    createCoframe,
    createDiffForm,
    createZeroForm,
    expand_dgcv,
    extDer,
    factor_dgcv,
    simplify_with_PDEs,
    transform_coframe,
    zeroFormAtom,
)
from .filtered_structures import Tanaka_symbol, distribution
from .polynomials import (
    createBigradPolynomial,
    createPolynomial,
    getWeightedTerms,
    monomialWeight,
)
from .Riemannian_geometry import (
    LeviCivitaConnectionClass,
    metric_from_matrix,
    metricClass,
)
from .solvers import solve_dgcv
from .styles import get_DGCV_themes, get_dgcv_themes  # get_DGCV_themes is deprecated
from .tensors import (
    createVectorSpace,
    tensorProduct,
    vector_space_class,
    vector_space_element,
)
from .vector_fields_and_differential_forms import (
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
from .vmf import DGCV_snapshot, clearVar, listVar, variableSummary, vmf_summary

# Default functions/classes
__all__ = [
    ############ DGCV default functions/classes ####

    # From _dgcv_display
    "LaTeX",                # Custom LaTeX renderer for DGCV objects
    "LaTeX_eqn_system",     # Custom LaTeX renderer for dictionaries 
                            # or lists representing equation systems     
    "LaTeX_list",
    "display_DGCV",         # deprecated
    "show",                 # Augments IPython.display.display
                            # with support for DGCV object like
                            # custom latex rendering
    "DGCV_init_printing",   # Augments SymPy.init_printing for DGCV
                            # objects

    # From _settings
    "set_dgcv_settings",

    # From combinatorics
    "carProd",              # Cartesian product
    "chooseOp",             # Choose operation
    "permSign",             # Permutation sign

    # From complexStructures
    "Del",                  # Holomorphic derivative operator
    "DelBar",               # Anti-holomorphic derivative operator
    "KahlerStructure",      # Represents a Kähler structure

    # From _config
    "canonicalize",         # Reformat supported objects canonically

    # From algebras
    "algebra_element_class",# Algebra element class
    "algebra_subspace_class",     # Algebra subspace class
    "algebra_class",        # Finite dimensional algebra
    "adjointRepresentation",# Adjoint representation of algebra
    "algebraDataFromMatRep",# Algebra data from matrix representation
    "algebraDataFromVF",    # Algebra data from vector fields
    "createFiniteAlg",      # deprecated
    "createAlgebra",        # Create a finite dimensional algebra
    "createSimpleLieAlgebra",
    "killingForm",          # Compute the Killing form
    "subalgebra_class",

    # From coordinateMaps
    "coordinate_map",       # Transforms coordinates systems

    # From CRGeometry
    "findWeightedCRSymmetries", # Find weighted CR symmetries
    "model2Nondegenerate",  # Produces a 2-nond. model structure
    "tangencyObstruction",  # Obstruction for VF to be tangent to submanifold
    "weightedHomogeneousVF",# Produce general weighted homogeneous vector fields

    # From dgcv_core
    "tensorField",          # Tensor field class
    "DFClass",              # Differential form class
    "DGCVPolyClass",        # deprecated 
    "dgcvPolyClass",        # dgcv polynomial class
    "DGCV_snapshot",        # deprecated
    "vmf_summary",          # Summarize initialized dgcv objects
    "STFClass",             # Symmetric tensor field class
    "VFClass",              # Vector field class
    "VF_bracket",           # Lie bracket of vector fields
    "VF_coeffs",            # Coefficients of vector fields
    "addDF",                # Add differential forms
    "addSTF",               # Add symmetric tensor fields
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
    "compressDGCVClass",    # deprecated
    "compress_dgcv_class",  # Removes superfluous variables from tensorField variable spaces
    "conjComplex",          # Conjugate complex variables
    "conj_with_hol_coor",   # Conjugate with holomorphic coordinate formatting
    "conj_with_real_coor",  # Conjugate with real coordinate formatting
    "conjugate_DGCV",       # deprecated
    "conjugate_dgcv",       # Conjugate DGCV objects
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
    "tensor_product",        # Compute tensor product of tensorField instances
    "variableSummary",      # Depricated - use DGCV_snapshot instead

    # From eds
    "zeroFormAtom",
    "factor_dgcv",
    "expand_dgcv",
    "createZeroForm",
    "createDiffForm",
    "abst_coframe",
    "createCoframe",
    "abstract_DF",
    "abstract_ZF",
    "extDer",
    "simplify_with_PDEs",
    "coframe_derivative",
    "DF_representation",
    "transform_coframe",

    # From filtered_structures
    "distribution", 
    "Tanaka_symbol",

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
    "get_DGCV_themes",      # deprecated
    "get_dgcv_themes",       # Get DGCV themes for various output styles

    # From solvers
    "solve_dgcv",           # supports solving equations with various DGCV types

    # From tensors
    "vector_space_class",          # Class representing vector spaces
    "vector_space_element",   # Class representing elements in a vector space
    "tensorProduct",       # Class representing elements in tensor products (of VS elements)
                            # of vector space and their dual spaces
    "createVectorSpace",    # Create vector_space_class class instances with labeling

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


# Configure warnings
configure_warnings()


# Register Sage’s numeric types with numbers ABCs for isinstance checks
if is_sage_available():
    try:
        sage = get_sage_module()
        from sage.all import Integer as SageInteger  # type: ignore
        from sage.all import Rational as SageRational  # type: ignore
        from sage.all import RealNumber as SageFloat  # type: ignore
        numbers.Integral.register(SageInteger)
        numbers.Real.register(SageFloat)
        numbers.Rational.register(SageRational)
    except Exception:
        pass
