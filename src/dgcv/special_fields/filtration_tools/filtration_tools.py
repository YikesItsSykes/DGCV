"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.special_fields.filtration_tools

module: dgcv.special_fields.filtration_tools.algebra_tools


---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

SPDX-License-Identifier: Apache-2.0


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
from ..filtered_structures import distribution

__all__ = [
    "distribution_sum",
]


def distribution_sum(
    *distributions: distribution, assume_transversal: bool = False
) -> distribution:
    if any(not isinstance(d, distribution) for d in distributions):
        raise TypeError(
            "`filtered_structures.distribution` requires arguments to be dgcv distribution classes"
        )
    if assume_transversal:
        vf_basis = sum([list(d.vf_basis) for d in distributions], [])
        return distribution(vf_basis, assume_spanning_sections_linearly_indep=True)
    return sum(distributions)
