"""
package: dgcv - Differential Geometry with Complex Variables
module: algebras/__init__

Author (of this module): David Sykes (https://realandimaginary.com/dgcv/)

License:
    MIT License
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
from .algebras_core import (
    adjointRepresentation,
    algebra_class,
    algebra_dual,
    algebra_element_class,
    algebra_subspace_class,
    killingForm,
    linear_representation,
    vector_space_endomorphisms,
)
from .algebras_secondary import (
    createAlgebra,
    createSimpleLieAlgebra,
    simple_Lie_algebra,
    subalgebra_class,
    subalgebra_element,
)

__all__ = [
    # core
    "adjointRepresentation",
    "algebra_class",
    "algebra_dual",
    "algebra_element_class",
    "algebra_subspace_class",
    "killingForm",
    "linear_representation",
    "vector_space_endomorphisms",
    # secondary
    "createAlgebra",
    "createSimpleLieAlgebra",
    "simple_Lie_algebra",
    "subalgebra_class",
    "subalgebra_element",
]
