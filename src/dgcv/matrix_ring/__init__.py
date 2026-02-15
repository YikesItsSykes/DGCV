"""
package: dgcv - Differential Geometry with Complex Variables
module: matrix_ring/__init__

Author (of this module): David Sykes (https://realandimaginary.com/dgcv/)

License:
    MIT License
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
from .matrix_ring import matrix_atom, symbolic_matrix

__all__ = ["matrix_atom", "symbolic_matrix"]
