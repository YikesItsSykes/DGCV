"""
package: dgcv - Differential Geometry with Complex Variables

Description:
------------
The dgcv package (Differential Geometry with Complex Variables) provides
general tools for differential geometry together a framework for efficiently
working with complex variables.

The library is oriented toward supporting convenient syntax that is intuitive
in the context of its mathematics applications.

Dependencies:
-------------
There is no mandatory hard dependency, but using dgcv along side at least one
supported CAS is recommended (currently supported: SymPy and Sage).
Optional integrations (e.g., IPython) are used automatically if present.

---
Author: David Gamble Sykes,

Project page for help and documentation: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# -----------------------------------------------------------------------------
# preliminary imports
# -----------------------------------------------------------------------------
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dgcv")
except PackageNotFoundError:
    __version__ = "unknown"


from ._aux._utilities._config import (
    configure_convenient_labels,
    configure_warnings,
    get_variable_registry,
    set_up_globals,
)

# -----------------------------------------------------------------------------
# Variable Management Framework (VMF) tools
# -----------------------------------------------------------------------------
set_up_globals()
_ = get_variable_registry()


# -----------------------------------------------------------------------------
# remaining imports
# -----------------------------------------------------------------------------
from ._aux._backends import (
    expand_dgcv,
    factor_dgcv,
    get_free_symbols,
    simplify_dgcv,
    subs_dgcv,
)
from ._aux._utilities._config import canonicalize
from ._aux._utilities._settings import (
    reset_dgcv_settings,
    set_dgcv_settings,
    view_dgcv_settings,
)
from ._aux._utilities._styles import (
    get_dgcv_themes,
)
from ._aux._vmf.vmf import (
    DGCV_snapshot,
    clear_vmf,
    clearVar,
    listVar,
    variableSummary,
    vmf_lookup,
    vmf_summary,
)
from ._aux.printing.printing._dgcv_display import (
    LaTeX,
    LaTeX_eqn_system,
    LaTeX_list,
    show,
)
from ._aux.printing.printing._string_processing import clean_LaTeX
from .algebras import algebra_tools
from .algebras.algebras_aux import algebraDataFromMatRep
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
from .ancillary.light_wrappers import function_dgcv
from .core.arrays.arrays import array_dgcv, assemble_block_matrix, matrix_dgcv
from .core.combinatorics.combinatorics import (
    Baker_Campbell_Hausdorff,
    carProd,
    chooseOp,
    permSign,
    split_number,
)
from .core.conversions.conversions import (
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
from .core.dgcv_core.dgcv_core import (
    DFClass,
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
    symmetric_product,
    temporaryVariables,
    tensor_field_class,
    tensor_product,
    tensorField,
    vector_field_class,
    wedge,
)
from .core.morphisms.coordinate_maps import coordinate_map
from .core.morphisms.morphisms import homomorphism
from .core.polynomials.polynomials import (
    createBigradPolynomial,
    createPolynomial,
    createRational,
    getWeightedTerms,
    monomialWeight,
)
from .core.solvers.solvers import solve_dgcv
from .core.tensors.tensors import (
    createVectorSpace,
    multi_tensor_product,
    tensorProduct,
    vector_space_class,
    vector_space_element,
)
from .core.vector_fields_and_differential_forms.vector_fields_and_differential_forms import (
    LieDerivative,
    annihilator,
    assembleFromAntiholVFC,
    assembleFromCompVFC,
    assembleFromHolVFC,
    coordinate_differential_form,
    coordinate_vector_field,
    decompose,
    exteriorDerivative,
    get_coframe,
    get_DF,
    get_VF,
    interiorProduct,
    makeZeroForm,
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
from .special_fields.complex_structures import Del, DelBar, KahlerStructure
from .special_fields.CR_geometry import (
    CR_structure,
    findWeightedCRSymmetries,
    model2Nondegenerate,
    tangencyObstruction,
    weightedHomogeneousVF,
)
from .special_fields.filtered_structures import Tanaka_symbol, distribution
from .special_fields.Riemannian_geometry import (
    metric_from_matrix,
    metricClass,
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
    "show",  # Augments IPython.display.display
    # with support for dgcv object like
    # custom latex rendering
    "reset_dgcv_settings",
    "set_dgcv_settings",
    "view_dgcv_settings",
    # From algebras
    "algebra_element_class",  # Algebra element class
    "subalgebra_element",
    "algebra_subspace_class",  # Algebra subspace class
    "algebra_class",  # Finite dimensional algebra
    "algebra_tools",
    "adjointRepresentation",  # Adjoint representation of algebra
    "algebraDataFromMatRep",  # Algebra data from matrix representation
    "createFiniteAlg",  # deprecated
    "createAlgebra",  # Create a finite dimensional algebra
    "createSimpleLieAlgebra",
    "killingForm",  # Compute the Killing form
    "vector_space_endomorphisms",
    "subalgebra_class",
    # From arrays
    "array_dgcv",  # light-weight array representation
    "matrix_dgcv",  # light-weight matrix representation
    "assemble_block_matrix",
    # From backends
    "expand_dgcv",
    "factor_dgcv",
    "get_free_symbols",
    "subs_dgcv",
    # From combinatorics
    "Baker_Campbell_Hausdorff",
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
    "CR_structure",
    "findWeightedCRSymmetries",  # Find weighted CR symmetries
    "model2Nondegenerate",  # Produces a 2-nond. model structure
    "linear_representation",
    "tangencyObstruction",  # Obstruction for VF to be tangent to submanifold
    "weightedHomogeneousVF",  # Produce general weighted homogeneous vector fields
    # From dgcv_core
    "assemble_tensor_field",
    "DFClass",  # deprecated - old differential form class (now just a dispatch shim)
    "differential_form_class",
    "dgcvPolyClass",  # dgcv polynomial class
    "DGCV_snapshot",  # deprecated
    "STFClass",  # deprecated - old symmetric tensor field class (now just a dispatch shim)
    "symmetric_product",
    "vector_field_class",
    "VFClass",  # # deprecated - old vector field class (now just a dispatch shim)
    "VF_bracket",  # Lie bracket of vector fields
    "VF_coeffs",  # Coefficients of vector fields
    "addDF",  # deprecated -  Add differential forms
    "addVF",  # deprecated - Add vector fields
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
    "scaleDF",  # deprecated - Scale differential forms
    "scaleVF",  # deprecated - Scale vector fields
    "symToHol",  # Convert symbolic conjugates to holomorphic format
    "symToReal",  # Convert symbolic conjugates to real format
    "tensorField",  # Tensor field class
    "tensor_product",  # deprecated - Compute tensor product of tensorField instances
    "variableSummary",  # deprecated - use vmf_summary instead
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
    "createRational",
    "getWeightedTerms",  # Get weighted terms of a polynomial
    "monomialWeight",  # Compute monomial weights
    # From RiemannianGeometry
    "metric_from_matrix",  # Create metric from matrix
    "metricClass",  # Metric class
    # From styles
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
    "coordinate_differential_form",
    "coordinate_vector_field",
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
