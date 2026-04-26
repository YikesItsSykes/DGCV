"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.algebras


Description:
------------
This sub-package provides tools for representing and analyzing algebras

---
Author (of this sub-package): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
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
