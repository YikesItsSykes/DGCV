"""
dgcv EDS Subpackage

This subpackage provides functionality for abstract exterior differential systems (EDS).

Modules included:
    - eds: Core EDS definitions and functions.
    - eds_representations: Classes and methods to handle EDS matrix representations.
    - eds_operations: Additional operations for EDS objects.

"""

from .eds import (
    abst_coframe,
    abstract_DF,
    abstract_ZF,
    coframe_derivative,
    createCoframe,
    createDiffForm,
    createZeroForm,
    expand_dgcv,
    extDer,
    factor_dgcv,
    simplify_with_PDEs,
    zeroFormAtom,
)
from .eds_operations import transform_coframe
from .eds_representations import DF_representation

__all__ = [
    "zeroFormAtom",
    "factor_dgcv",
    "expand_dgcv",
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
