"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds - Exterior Differential Systems

Description:
------------
This sub-package provides functionality for abstract exterior differential systems (EDS).

Modules included:
    - eds: Core EDS definitions and functions.
    - eds_representations: Classes and methods to handle EDS matrix representations.
    - eds_operations: Additional operations for EDS objects.

---
Author (of this sub-package): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from .eds import (
    abst_coframe,
    abstract_DF,
    abstract_ZF,
    coframe_derivative,
    createCoframe,
    createDiffForm,
    createZeroForm,
    extDer,
    simplify_with_PDEs,
    zeroFormAtom,
)
from .eds_operations import transform_coframe
from .eds_representations import DF_representation

__all__ = [
    "zeroFormAtom",
    "createZeroForm",
    "createDiffForm",
    "abst_coframe",
    "createCoframe",
    "abstract_DF",
    "abstract_ZF",
    "extDer",
    "simplify_with_PDEs",
    "coframe_derivative",
    "DF_representation",
    "transform_coframe",
]
