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

from .aec import algebra_element_class
from .algebra_tools import intersection
from .algebras import algebra_class, killingForm
from .composition.composites import vector_space_endomorphisms
from .creators import createAlgebra, createSimpleLieAlgebra
from .dual import algebra_dual
from .format_support import algebraDataFromMatRep
from .linear_algebra import linear_representation
from .saec import subalgebra_element
from .specialized import simple_Lie_algebra
from .subspaces import algebra_subspace_class
from .subspaces.subalgebras import subalgebra_class
from .threads import _extract_basis, adjointRepresentation

__all__ = [
    # core
    "adjointRepresentation",
    "algebra_class",
    "algebra_dual",
    "algebra_element_class",
    "algebra_subspace_class",
    "algebraDataFromMatRep",
    "intersection",
    "killingForm",
    "linear_representation",
    "vector_space_endomorphisms",
    # secondary
    "createAlgebra",
    "createSimpleLieAlgebra",
    "simple_Lie_algebra",
    "subalgebra_class",
    "subalgebra_element",
    # misc
    "_extract_basis",
]
