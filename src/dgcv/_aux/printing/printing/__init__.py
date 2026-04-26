"""
package: dgcv - Differential Geometry with Complex Variables

module: dgcv._aux.printing.printing.__init__

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
from ._class_printers import (
    array_latex_helper,
    array_VS_printer,
    lincomb_latex,
    lincomb_plain,
    space_display,
    tensor_field_latex2,
    tensor_field_printer2,
    tensor_latex_helper,
    tensor_VS_printer,
)
from ._string_processing import _unwrap_math_delims

__all__ = [
    "_unwrap_math_delims",
    "array_VS_printer",
    "array_latex_helper",
    "lincomb_latex",
    "lincomb_plain",
    "space_display",
    "tensor_field_latex2",
    "tensor_field_printer2",
    "tensor_latex_helper",
    "tensor_VS_printer",
]
