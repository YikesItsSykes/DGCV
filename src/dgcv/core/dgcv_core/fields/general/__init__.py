from __future__ import annotations

from ....base import dgcv_class
from .algebra import _tensor_field_algebra
from .attributes import _tensor_field_attributes
from .construction import _tensor_field_construction
from .contraction import _tensor_field_contraction
from .coordinates import _tensor_field_coordinates
from .formats import _tensor_field_formats
from .printing import _tensor_field_printing
from .reduction import _tensor_field_reduction

__all__ = ["tensor_field_class"]


class tensor_field_class(
    _tensor_field_algebra,
    _tensor_field_attributes,
    _tensor_field_construction,
    _tensor_field_contraction,
    _tensor_field_coordinates,
    _tensor_field_formats,
    _tensor_field_printing,
    _tensor_field_reduction,
    dgcv_class,
):
    """
    The primary class for representing tensor fields in dgcv.

    Parameters
    ----------
    coeff_dict : dict, optional
        Coefficient dictionary describing the tensor field

    data_shape : {"general","symmetric","skew","all"}, default "general"
        Declared symmetry of the tensor coefficients. If an invalid value is
        provided, it is treated as "general".

    variable_spaces : dict, optional
        Optional mapping to local coordinate spaces from thier labels. If not
        provided, it is automatically inferred from `coeff_dict`. The only
        reason to provide it is bypassing the inference overhead, but this
        must be given in very precise formatting. I.e., Ignore if unsure.

    parameters : set-like, optional
        Free symbols treated by dgcv as parameters (i.e. not part of relevant
        coordinate systems). Stored as `self.parameters`.

    varSpace : optional
        depricated

    valence : optional
        depricated

    _simplifyKW : dict, optional
        Internal simplification configuration. If None, defaults are used.

    _inheritance : dict, optional
        Internal initialization hook used when constructing an instance from
        another tensor field and inheriting cached/validated format metadata.

    Attributes
    ----------
    coeff_dict : dict
        Canonical coefficient dictionary for the tensor field. Its format
        is shape-aware sparse. e.g., "skew" or "symmetric" shapes store
        at most one index tuple from a multi-index permutation class.
    data_shape : str
        Shape tag: "general", "symmetric", "skew", or "all".
    parameters : set
        Set of atoms regarded as non-coordinate variables
    max_degree : int
        Maximum tensor degree present in `coeff_dict`

    min_degree : int
        Minimum tensor degree present in `coeff_dict`

    total_degree : int
        Alias for `max_degree`

    free_symbols : set
        Union of symbols appearing in the coefficient expressions and symbols
        appearing in the registered variable spaces referenced by this tensor
        field

    coeff_free_symbols : set
        Set of symbols appearing in the coefficient expressions only.

    expanded_coeff_dict : dict
        Coefficient dictionary expanded to general shape

    homogeneous_parts : list[tensor_field_class]
        Decomposition of the tensor into homogeneous parts

    varSpace : tuple
        deprecated

    valence : tuple
        deprecated

    Methods
    -------
    infer_varSpace(formatting="complex", *, return_dict=False)
        Infer an ordered variable space from the variable systems referenced by
        the tensor. `formatting` may be "real", "complex", or "any". If
        `return_dict=True`, also returns a dictionary mapping variables to
        (system_label, local_index) pairs.

    infer_minimal_varSpace(*, return_dict=False)
        Infer a minimal variable space containing only variables whose system
        indices actually occur in `coeff_dict`. If `return_dict=True`, also
        returns the same variable-to-location dictionary format as
        `infer_varSpace`.

    apply(func, *, drop_zeros=True, data_shape=None, dgcvType=None, ...)
        Apply a callable to every coefficient value and return a new tensor
        field. By default, zero coefficients are dropped.

    holomorphic_part()
    antiholomorphic_part()
    mixed_term_component()
    pure_standard_coordinate_terms()
    real_part()
    imaginary_part()
        Extract components by coordinate-format type when the tensor is stored
        in (or convertible to) complex/real mixed formats.

    swap_tensor_valence()
        Toggle covariant/contravariant valence per tensor index. E.g., vector
        fields become differential forms, type (p,q) tensors become (q,p), etc.
        Returns `vector_field_class` or `differential_form_class`
        when applicable.

    tensor_product(*others)
    symmetric_product(*others)
    skew_product(*others) / wedge(*others)
        Tensor products with shape control. `wedge` is an alias for
        `skew_product`.

    __call__(*args, strict_left_to_right=False)
        Contract the tensor against dgcv tensor fields (and scalars). Supports
        iterative contraction when multiple arguments are provided.

    is_zero : bool
        True if the tensor is identically zero (including the scalar case).

    subs(substitutions)
        Substitute into coefficient expressions and return a new tensor field.
    """
