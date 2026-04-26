"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.core.morphisms


---
Author (of this sub-package): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from .coordinate_maps import coordinate_map
from .morphisms import homomorphism

__all__ = ["coordinate_map", "homomorphism"]
