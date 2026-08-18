"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.core.arrays

---

Author (of this sub-package): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from ..base import dgcv_class
from . import _array_core as _mod_array_core
from . import _block_matrix as _mod_block_matrix
from . import _coercion as _mod_coercion
from . import _frozen_array as _mod_frozen_array
from . import _frozen_matrix as _mod_frozen_matrix
from . import _matrix_core as _mod_matrix_core
from . import _matrix_engine as _mod_matrix_engine
from . import _matrix_linalg as _mod_matrix_linalg
from ._array_core import _array_core
from ._array_transforms import _array_transforms
from ._block_matrix import assemble_block_matrix
from ._coercion import _as_matrix_dgcv
from ._frozen_array import _frozen_array, freeze_array
from ._frozen_matrix import _frozen_matrix, freeze_matrix
from ._indexing import _spool, _unspool
from ._matrix_arithmetic import _matrix_arithmetic
from ._matrix_constructors import _matrix_constructors
from ._matrix_core import _matrix_core
from ._matrix_engine import _matrix_engine
from ._matrix_linalg import _matrix_linalg
from ._matrix_structure import _matrix_structure
from ._matrix_symmetric import _matrix_symmetric
from ._printing import _array_printing, _matrix_printing

__all__ = [
    "_as_matrix_dgcv",
    "_spool",
    "_unspool",
    "array_dgcv",
    "assemble_block_matrix",
    "freeze_array",
    "freeze_matrix",
    "frozen_array_dgcv",
    "frozen_matrix_dgcv",
    "matrix_dgcv",
]


class array_dgcv(_array_core, _array_printing, _array_transforms, dgcv_class):
    _dgcv_category = "array"


class frozen_array_dgcv(_frozen_array, array_dgcv):
    _dgcv_category = "array"


class matrix_dgcv(
    _matrix_core,
    _matrix_printing,
    _matrix_constructors,
    _matrix_structure,
    _matrix_arithmetic,
    _matrix_linalg,
    _matrix_engine,
    _matrix_symmetric,
    array_dgcv,
):
    """
    A general 2-d array structure with convenient properties for dgcv. It can be used for storing and displaying data, or more linear algebra intensive applications with standard matrix arithmetic. Although built-in methods for the latter applications are limited as no assumptions are enforced about object types in the array's entries.

    Parameters
    ----------
    array_data : list of lists, tuple of tuples, various matrix/array classes
        Data defining a 2-dimensional array

    Notes
    -----
    Types for entries are not restricted, but several class methods (matrix multiplication, scalar multiplication, addition, etc.) are written with the assumption that entry types behave as elements in some algebra, i.e., they need to have methods __add__, __mul__, etc. enabeling scalar multiplication and multiplication and addition between them.
    """

    _dgcv_categories = {"matrix"}


class frozen_matrix_dgcv(_frozen_matrix, matrix_dgcv):
    _dgcv_categories = {"matrix"}
    _inertia_cache = None


_mod_array_core.array_dgcv = array_dgcv
_mod_block_matrix.matrix_dgcv = matrix_dgcv
_mod_coercion.matrix_dgcv = matrix_dgcv
_mod_frozen_array.array_dgcv = array_dgcv
_mod_frozen_array.frozen_array_dgcv = frozen_array_dgcv
_mod_frozen_matrix.frozen_matrix_dgcv = frozen_matrix_dgcv
_mod_frozen_matrix.matrix_dgcv = matrix_dgcv
_mod_matrix_core.matrix_dgcv = matrix_dgcv
_mod_matrix_engine.matrix_dgcv = matrix_dgcv
_mod_matrix_linalg.matrix_dgcv = matrix_dgcv
