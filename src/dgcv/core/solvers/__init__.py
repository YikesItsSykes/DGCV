"""
package: dgcv - Differential Geometry with Complex Variables
sub-package: dgcv.core
module: dgcv.core.solvers

Author (of this module): David Gamble Sykes
Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes
Licensed under the Apache License, Version 2.0
SPDX-License-Identifier: Apache-2.0
"""

from ._driver import solve_dgcv, solve_knowing_solution_exists
from ._filters import linear_filter

__all__ = ["solve_dgcv", "solve_knowing_solution_exists", "linear_filter"]
