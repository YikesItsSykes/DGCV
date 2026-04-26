"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.core.combinatorics


---
Author (of this sub-package): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from .combinatorics import (
    Baker_Campbell_Hausdorff,
    carProd,
    chooseOp,
    permSign,
    shufflings,
    weightedPermSign,
)

__all__ = [
    "Baker_Campbell_Hausdorff",
    "carProd",
    "chooseOp",
    "permSign",
    "weightedPermSign",
    "shufflings",
]
