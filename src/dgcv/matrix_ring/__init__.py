"""
package: dgcv - Differential Geometry with Complex Variables

module: matrix_ring/__init__


Description:
------------
A ring structure for symbolic linear algebra.


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
from .matrix_ring import matrix_atom, symbolic_matrix

__all__ = ["matrix_atom", "symbolic_matrix"]
