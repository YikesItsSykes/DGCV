"""
package: dgcv - Differential Geometry with Complex Variables

module: dgcv.secondary_library


---
Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from ._aux._backends._symbolic_router import defloat
from .core.arrays.arrays import frozen_array_dgcv, frozen_matrix_dgcv
from .core.vector_fields_and_differential_forms.vector_fields_and_differential_forms import (
    _decompose_over_number_field as decompose_over_number_field,
)
from .core.vector_fields_and_differential_forms.vector_fields_and_differential_forms import (
    _extract_basis_by_wedge_vectorized as extract_basis_over_function_ring,
)
from .core.vector_fields_and_differential_forms.vector_fields_and_differential_forms import (
    _extract_basis_over_number_field as extract_basis_over_number_field,
)

__all__ = [
    "decompose_over_number_field",
    "extract_basis_over_function_ring",
    "extract_basis_over_number_field",
    "defloat",
    "frozen_matrix_dgcv",
    "frozen_array_dgcv",
]
