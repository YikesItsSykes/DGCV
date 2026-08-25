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

from .atom_factory import createMatrixCoordinates, createVariables, variableProcedure
from .combinations import (
    VF_bracket,
    exteriorProduct,
    sum_dgcv,
    symmetric_product,
    tensor_product,
    wedge,
)
from .coordinate_formats import (
    conj_with_hol_coor,
    conj_with_real_coor,
    conjugate_dgcv,
    im_with_hol_coor,
    im_with_real_coor,
    re_with_hol_coor,
    re_with_real_coor,
)
from .fields import (
    assemble_tensor_field,
    differential_form_class,
    tensor_field_class,
    vector_field_class,
)
from .holomorphic_fields import (
    antiholVF_coeffs,
    complex_struct_op,
    complexVFC,
    holVF_coeffs,
    realPartOfVF,
)
from .polynomials.poly import polynomial_dgcv

__all__ = [
    "VF_bracket",
    "antiholVF_coeffs",
    "assemble_tensor_field",
    "complexVFC",
    "complex_struct_op",
    "conj_with_hol_coor",
    "conj_with_real_coor",
    "conjugate_dgcv",
    "conjugate_dgcv",
    "createMatrixCoordinates",
    "createVariables",
    "differential_form_class",
    "exteriorProduct",
    "holVF_coeffs",
    "im_with_hol_coor",
    "im_with_real_coor",
    "polynomial_dgcv",
    "re_with_hol_coor",
    "re_with_real_coor",
    "realPartOfVF",
    "sum_dgcv",
    "symmetric_product",
    "tensor_field_class",
    "tensor_product",
    "variableProcedure",
    "vector_field_class",
    "wedge",
]
