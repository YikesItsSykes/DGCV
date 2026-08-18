from __future__ import annotations

from ...core.base import dgcv_class
from ._brackets import _symbol_brackets
from ._core import _symbol_core
from ._export import _symbol_export
from ._printing import _symbol_printing
from ._prolongation import _symbol_prolongation
from ._prolongation_stages import _symbol_prolongation_stages
from ._prolongation_step import _symbol_prolongation_step


class Tanaka_symbol(
    _symbol_core,
    _symbol_prolongation,
    _symbol_prolongation_step,
    _symbol_prolongation_stages,
    _symbol_brackets,
    _symbol_export,
    _symbol_printing,
    dgcv_class,
):
    """
    Graded Lie algebra data and related structures prepared for Tanaka prolongation.

    Parameters
    ----------
    GLA : dgcv algebra class (algebra_class, subalgebra_class)
        Graded Lie algebra containing the negative part.
    nonnegParts : list or dict, default []
        Weighted homogeneous elements of nonnegative degree, or a dict keying
        them by degree. dict formatting is only intended for fast init with
        pre-validated data; use the list formatting if unsure.
    assume_FGLA : bool, default False
        Permit assuming the negative part is fundamental (i.e., generated
        by -1 component), which allows some performance optimizations in
        prolong and algebra conversion methods
    subspace : subalgebra, optional
        Negative part to use. Defaults to the negative part of `GLA`.
    distinguished_subspaces : list of list, optional
        Subspaces the prolongation must preserve.
    prolongation_label_prefix : str, optional
        Prefix for labels of computed prolongation levels.
    assume_linear_independence : bool, default False
        Skip basis extraction on distinguished subspaces. Only set True if
        supplied distinguished subspace element are known to be linearly
        independant
    assume_NNP_linear_indep : bool, default False
        Skip basis extraction on `nonnegParts`.
    index_threshold : int, optional
        Lowest degree the level indexing recognizes.
    precompute_generators : bool, default False
        Compute generators of the negative part at initialization.

    Methods
    -------
    prolong
        Compute prolongation levels, optionally returning a new symbol.
    summary
        Display the levels and their labels.
    export_algebra_data
        Return structure data for abstract algebra isomorphic to the symbol
        with its computed prolongation levels.
    """
