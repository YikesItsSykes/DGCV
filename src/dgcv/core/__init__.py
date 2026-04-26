"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.core

Description:
------------
Defining dgcv's core classes and utilities.

Dependency note: Positioned upstream from all non-utility subpackages in the library.


---
Author (of this sub-package): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from .arrays import array_dgcv, frozen_array_dgcv, frozen_matrix_dgcv, matrix_dgcv
from .dgcv_core import (
    complex_struct_op,
    createVariables,
    differential_form_class,
    exteriorProduct,
    polynomial_dgcv,
    realPartOfVF,
    symmetric_product,
    tensor_field_class,
    tensor_product,
    vector_field_class,
)
from .vector_fields_and_differential_forms import (
    LieDerivative,
    annihilator,
    coordinate_differential_form,
    coordinate_vector_field,
    decompose,
    exteriorDerivative,
    get_coframe,
    get_DF,
    get_VF,
    interiorProduct,
)

__all__ = [
    "LieDerivative",
    "annihilator",
    "array_dgcv",
    "complex_struct_op",
    "coordinate_differential_form",
    "coordinate_vector_field",
    "createVariables",
    "decompose",
    "differential_form_class",
    "exteriorDerivative",
    "exteriorProduct",
    "frozen_array_dgcv",
    "frozen_matrix_dgcv",
    "get_DF",
    "get_VF",
    "get_coframe",
    "interiorProduct",
    "matrix_dgcv",
    "polynomial_dgcv",
    "realPartOfVF",
    "symmetric_product",
    "tensor_field_class",
    "tensor_product",
    "vector_field_class",
]
