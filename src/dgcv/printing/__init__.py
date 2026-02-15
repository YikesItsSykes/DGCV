"""
package: dgcv - Differential Geometry with Complex Variables
module: printing/__init__

Author (of this module): David Sykes (https://realandimaginary.com/dgcv/)

License:
    MIT License
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
