"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.branching_analysis


Description:
------------
This sub-package defines tools for organizing branching analysis of equation systems

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
from .branching_analysis import case_tree

__all__ = ["case_tree"]
