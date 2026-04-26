"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.core.dgcv_core


---
Author (of this sub-package): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

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

__all__ = [
    "complex_struct_op",
    "createVariables",
    "differential_form_class",
    "exteriorProduct",
    "polynomial_dgcv",
    "realPartOfVF",
    "symmetric_product",
    "tensor_field_class",
    "tensor_product",
    "vector_field_class",
]
