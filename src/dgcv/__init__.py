"""
dgcv: Package Initialization

The dgcv package (Differential Geometry with Complex Variables) provides
tools for differential geometry together a framework for conveniently
working with complex variables.

The top-level ``__init__.py`` initializes core infrastructure used
throughout the library, principally building light-weight backend
dictionaries for dgcv's variable management framework (VMF).

Initialization
--------------
- Variable Management Framework (VMF):
  Automatically initializes the an active namespace hook and variable registry
  systems underlying dgcv's Variable Management Framework. The VMF tracks
  coordinate systems, algebraic objects, differential forms, tensor fields,
  and related structures, and coordinates their interaction across modules.

- Backend Detection and Routing:
  Detects available symbolic engines (e.g., SymPy or Sage) and configures
  dgcv's symbolic routing layer according to the active settings. there are
  no hard dependencies on a single backend engine; compatible engines are
  detected and used if available.

- Display Integration:
  Detects rich display environments (such as IPython/Jupyter) and enables
  LaTeX-aware rendering when supported and enabled in dgcv settings.

Dependencies
------------
dgcv has no mandatory hard dependency , but at least one supported backend
is recommended (currently supported: SymPy or Sage).
Optional integrations (e.g., IPython) are used automatically if present.

Author: David Sykes
https://www.realandimaginary.com/dgcv/

License:
    MIT License
"""

# -----------------------------------------------------------------------------
# preliminary imports
# -----------------------------------------------------------------------------
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dgcv")
except PackageNotFoundError:
    __version__ = "unknown"


from ._config import (
    cache_globals,
    configure_convenient_labels,
    configure_warnings,
    get_variable_registry,
)

# -----------------------------------------------------------------------------
# Variable Management Framework (VMF) tools
# -----------------------------------------------------------------------------
cache_globals()
_ = get_variable_registry()


# -----------------------------------------------------------------------------
# remaining imports
# -----------------------------------------------------------------------------
from ._config import canonicalize
from ._dgcv_display import (
    DGCV_init_printing,
    LaTeX,
    LaTeX_eqn_system,
    LaTeX_list,
    display_DGCV,
    show,
)
from ._settings import reset_dgcv_settings, set_dgcv_settings, view_dgcv_settings
from .algebras.algebras_aux import algebraDataFromMatRep, algebraDataFromVF
from .algebras.algebras_core import (
    adjointRepresentation,
    algebra_class,
    algebra_element_class,
    algebra_subspace_class,
    killingForm,
    linear_representation,
    vector_space_endomorphisms,
)
from .algebras.algebras_secondary import (
    createAlgebra,
    createFiniteAlg,  # deprecated
    createSimpleLieAlgebra,
    subalgebra_class,
    subalgebra_element,
)
from .arrays import array_dgcv, matrix_dgcv
from .backends import (
    expand_dgcv,
    factor_dgcv,
    get_free_symbols,
    simplify_dgcv,
    subs_dgcv,
)
from .combinatorics import carProd, chooseOp, permSign, split_number
from .complex_structures import Del, DelBar, KahlerStructure
from .conversions import (
    allToHol,
    allToReal,
    allToSym,
    cleanUpConjugation,
    holToReal,
    holToSym,
    realToHol,
    realToSym,
    symToHol,
    symToReal,
)
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
    addVF,
    antiholVF_coeffs,
    assemble_tensor_field,
    complex_struct_op,
    complexVFC,
    conj_with_hol_coor,
    conj_with_real_coor,
    conjComplex,
    conjugate_DGCV,  # deprecated
    conjugate_dgcv,
    createVariables,
    dgcvPolyClass,
    differential_form_class,
    exteriorProduct,
    holVF_coeffs,
    im_with_hol_coor,
    im_with_real_coor,
    polynomial_dgcv,
    re_with_hol_coor,
    re_with_real_coor,
    realPartOfVF,
    scaleDF,
    scaleVF,
    temporaryVariables,
    tensor_field_class,
    tensor_product,
    tensorField,
    vector_field_class,
    wedge,
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
    extDer,
    simplify_with_PDEs,
    transform_coframe,
    zeroFormAtom,
)
from .filtered_structures import Tanaka_symbol, distribution
from .light_wrappers import function_dgcv
from .morphisms import homomorphism
from .polynomials import (
    createBigradPolynomial,
    createPolynomial,
    getWeightedTerms,
    monomialWeight,
)
from .printing._string_processing import clean_LaTeX
from .Riemannian_geometry import (
    metric_from_matrix,
    metricClass,
)
from .solvers import solve_dgcv
from .styles import get_DGCV_themes, get_dgcv_themes  # get_DGCV_themes is deprecated
from .tensors import (
    createVectorSpace,
    multi_tensor_product,
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
from .vmf import (
    DGCV_snapshot,
    clear_vmf,
    clearVar,
    listVar,
    variableSummary,
    vmf_lookup,
    vmf_summary,
)

# -----------------------------------------------------------------------------
# broadcasting
# -----------------------------------------------------------------------------
__all__ = [
    ############ dgcv default functions/classes ####
    # From _config
    "configure_convenient_labels",
    # From _dgcv_display
    "LaTeX",  # Custom LaTeX renderer for dgcv objects
    "LaTeX_eqn_system",  # Custom LaTeX renderer for dictionaries
    # or lists representing equation systems
    "LaTeX_list",
    "display_DGCV",  # deprecated
    "show",  # Augments IPython.display.display
    # with support for dgcv object like
    # custom latex rendering
    "DGCV_init_printing",  # Augments SymPy.init_printing for dgcv
    # objects
    # From _settings
    "reset_dgcv_settings",
    "set_dgcv_settings",
    "view_dgcv_settings",
    # From algebras
    "algebra_element_class",  # Algebra element class
    "subalgebra_element",
    "algebra_subspace_class",  # Algebra subspace class
    "algebra_class",  # Finite dimensional algebra
    "adjointRepresentation",  # Adjoint representation of algebra
    "algebraDataFromMatRep",  # Algebra data from matrix representation
    "algebraDataFromVF",  # Algebra data from vector fields
    "createFiniteAlg",  # deprecated
    "createAlgebra",  # Create a finite dimensional algebra
    "createSimpleLieAlgebra",
    "killingForm",  # Compute the Killing form
    "vector_space_endomorphisms",
    "subalgebra_class",
    # From arrays
    "array_dgcv",  # light-weight array representation
    "matrix_dgcv",  # light-weight matrix representation
    # From backends
    "expand_dgcv",
    "factor_dgcv",
    "get_free_symbols",
    "subs_dgcv",
    # From combinatorics
    "carProd",  # Cartesian product
    "chooseOp",  # Choose operation
    "permSign",  # Permutation sign
    "split_number",
    # From complexStructures
    "Del",  # Holomorphic derivative operator
    "DelBar",  # Anti-holomorphic derivative operator
    "KahlerStructure",  # Represents a Kähler structure
    # From _config
    "canonicalize",  # Reformat supported objects canonically
    # From coordinateMaps
    "coordinate_map",  # Transforms coordinates systems
    # From CRGeometry
    "findWeightedCRSymmetries",  # Find weighted CR symmetries
    "model2Nondegenerate",  # Produces a 2-nond. model structure
    "linear_representation",
    "tangencyObstruction",  # Obstruction for VF to be tangent to submanifold
    "weightedHomogeneousVF",  # Produce general weighted homogeneous vector fields
    # From dgcv_core
    "assemble_tensor_field",
    "DFClass",  # Depricated - old differential form class (now just a dispatch shim)
    "differential_form_class",
    "DGCVPolyClass",  # deprecated
    "dgcvPolyClass",  # dgcv polynomial class
    "DGCV_snapshot",  # deprecated
    "STFClass",  # Depricated - old symmetric tensor field class (now just a dispatch shim)
    "vector_field_class",
    "VFClass",  # # Depricated - old vector field class (now just a dispatch shim)
    "VF_bracket",  # Lie bracket of vector fields
    "VF_coeffs",  # Coefficients of vector fields
    "addDF",  # Depricated -  Add differential forms
    "addVF",  # Depricated - Add vector fields
    "allToHol",  # Convert dgcv expressions to holomorphic
    "tensor_field_class",
    # coordinate format
    "allToReal",  # Convert all fields to real
    # coordinate format
    "allToSym",  # Convert all fields to symbolic
    # conjugate coordinate format
    "antiholVF_coeffs",  # Anti-holomorphic coefficients of vector field
    "cleanUpConjugation",  # Cleanup conjugation operations
    "clearVar",  # Clear dgcv objects from globals()
    "clear_vmf",
    "complexVFC",  # Complex coordingate vector field coefficients
    "complex_struct_op",  # Complex structure operator
    "conjComplex",  # Conjugate complex variables
    "conj_with_hol_coor",  # Conjugate with holomorphic coordinate formatting
    "conj_with_real_coor",  # Conjugate with real coordinate formatting
    "conjugate_DGCV",  # deprecated
    "conjugate_dgcv",  # Conjugate dgcv objects
    "createVariables",  # Initialize variables in dgcv's VMF
    "temporaryVariables",
    "exteriorProduct",  # Compute exterior product
    "holToReal",  # Convert holomorphic to real format
    "holToSym",  # Convert holomorphic to symbolic conjugates format
    "holVF_coeffs",  # Holomorphic coefficients of vector field
    "im_with_hol_coor",  # Imaginary part with holomorphic coordinate format
    "im_with_real_coor",  # Imaginary part with real coordinate format
    "listVar",  # List objects from the dgcv VMF
    "polynomial_dgcv",  # dgcv polynomial class
    "realPartOfVF",  # Real part of vector fields
    "realToHol",  # Convert real to holomorphic fomrat
    "realToSym",  # Convert real to symbolic conjugates format
    "re_with_hol_coor",  # Real part with holomorphic coordinate format
    "re_with_real_coor",  # Real part with real coordinate format
    "scaleDF",  # Depricated - Scale differential forms
    "scaleVF",  # Depricated - Scale vector fields
    "symToHol",  # Convert symbolic conjugates to holomorphic format
    "symToReal",  # Convert symbolic conjugates to real format
    "tensorField",  # Tensor field class
    "tensor_product",  # Depricated - Compute tensor product of tensorField instances
    "variableSummary",  # Depricated - use vmf_summary instead
    "wedge",  # wedge product of tensor field classes
    # From eds
    "zeroFormAtom",
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
    # From light_wrappers
    "function_dgcv",
    # From morphisms
    "homomorphism",
    # From printing/
    "clean_LaTeX",
    # From polynomials
    "createBigradPolynomial",  # Create bigraded polynomial
    "createPolynomial",  # Create polynomial
    "getWeightedTerms",  # Get weighted terms of a polynomial
    "monomialWeight",  # Compute monomial weights
    # From RiemannianGeometry
    "metric_from_matrix",  # Create metric from matrix
    "metricClass",  # Metric class
    # From styles
    "get_DGCV_themes",  # deprecated
    "get_dgcv_themes",  # Get dgcv themes for various output styles
    # From solvers
    "solve_dgcv",  # supports solving equations with various dgcv types
    "simplify_dgcv",  #
    # From tensors
    "createVectorSpace",  # Create vector_space_class class instances with labeling
    "multi_tensor_product",  # Form tensorProduct from multiple factors
    # of vector space and their dual spaces
    "vector_space_class",  # Class representing vector spaces
    "vector_space_element",  # Class representing elements in a vector space
    "tensorProduct",  # Class representing elements in tensor products (of VS elements)
    # From vectorFieldsAndDifferentialForms
    "LieDerivative",  # Compute Lie derivative
    "annihilator",  # Compute annihilator
    "assembleFromAntiholVFC",  # Assemble VF from anti-holomorphic VF coefficients
    "assembleFromCompVFC",  # Assemble VF from complex VF coefficients
    "assembleFromHolVFC",  # Assemble VF from holomorphic VF coefficients
    "decompose",  # Decompose objects into linear combinations
    "exteriorDerivative",  # Compute exterior derivative
    "get_coframe",  # Get coframe from frame
    "get_DF",  # Get differential form from label in VMF
    "get_VF",  # Get vector field from label in VMF
    "interiorProduct",  # Compute interior product
    "makeZeroForm",  # Create zero-form from scalar
    # From vmf
    "vmf_lookup",  # find details about an object stored in the VMF
    "vmf_summary",  # Summarize initialized dgcv objects
]


# -----------------------------------------------------------------------------
# additional configurations
# -----------------------------------------------------------------------------
configure_warnings()
