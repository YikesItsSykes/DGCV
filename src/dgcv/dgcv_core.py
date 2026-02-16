"""
package: dgcv - Differential Geometry with Complex Variables
module: dgcv_core

Description: This module defines the core classes and functions for the dgcv package, handling the
creation and manipulation of vector fields, differential forms, tensor fields, algebras, and more.
It includes tools for managing relationships between real and complex coordinate systems, object
creation function, and basic operations for its classes.

Author (of this module): David Sykes (https://realandimaginary.com/dgcv/)

License:
    MIT License
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
from __future__ import annotations

import itertools
import warnings
from math import gcd, prod
from numbers import Integral, Number
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from ._config import (
    _cached_caller_globals,
    get_dgcv_settings_registry,
    get_variable_registry,
)
from ._safeguards import (
    check_dgcv_category,
    create_key,
    get_dgcv_category,
    query_dgcv_categories,
    retrieve_passkey,
    retrieve_public_key,
    validate_label,
)
from .arrays import _spool, array_dgcv
from .backends._calculus import diff, integrate
from .backends._display import latex as _backend_latex
from .backends._engine import engine_kind
from .backends._exact_arith import exact_reciprocal
from .backends._polynomials import (
    _stable_dedupe,
    make_poly,
    poly_coeffs,
    poly_gens,
    poly_monoms,
    poly_total_degree,
)
from .backends._symbolic_router import (
    _scalar_is_zero,
    as_numer_denom,
    cancel,
    conjugate,
    expand,
    factor,
    get_free_symbols,
    ilcm,
    im,
    ratio,
    re,
    simplify,
    subs,
)
from .backends._types_and_constants import (
    check_dgcv_scalar,
    imag_unit,
    integer,
    is_atomic,
    one,
    rational,
    symbol,
    verify_conjugate_re_im_free,
    zero,
)
from .base import dgcv_class
from .combinatorics import build_nd_array, carProd, permSign
from .conversions import (
    _coeff_dict_formatter,
    allToHol,
    allToReal,
    allToSym,
    holToReal,
    realToHol,
    realToSym,
    symToHol,
    symToReal,
)
from .printing import _unwrap_math_delims, tensor_field_latex2, tensor_field_printer2
from .vmf import clearVar, vmf_lookup

__all__ = [
    "tensor_field_class",
    "vector_field_class",
    "differential_form_class",
    "assemble_tensor_field",
    "polynomial_dgcv",
    "createVariables",
    "complex_struct_op",
    "exteriorProduct",
    "realPartOfVF",
    "tensor_product",
]


# -----------------------------------------------------------------------------
# constants
# -----------------------------------------------------------------------------
half = rational(1, 2)


# -----------------------------------------------------------------------------
# tensor fields
# -----------------------------------------------------------------------------
class tensor_field_class(dgcv_class):
    """
    The primary class for representing tensor fields in dgcv. Vector fields
    and differential forms are defined as subclass, for example.

    Parameters
    ----------
    coeff_dict : dict, optional
        Coefficient dictionary describing the tensor field. In the standard
        format, keys are “tripled” tuples:

            (local_coordinate_indices, valence_indices, local_coordinate_labels)

        where each component is a tuple of equal length. Values are scalar
        expressions.

        Special case: a scalar may be provided via a scalar-style coeff_dict
        (e.g. `{(): scalar}`), which initializes a degree-0 tensor field with
        `data_shape="all"`.

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
        DO NOT USE (depricated). Legacy initialization input. If `varSpace`
        or `valence` is provided, the initializer may switch to old-format
        parsing.

    valence : optional
        DO NOT USE (depricated). Legacy initialization input. If `varSpace`
        or `valence` is provided, the initializer may switch to old-format
        parsing.

    _simplifyKW : dict, optional
        Internal simplification configuration. If None, defaults are used.

    _inheritance : dict, optional
        Internal initialization hook used when constructing an instance from
        another tensor field and inheriting cached/validated format metadata.

    Notes
    -----
    - If neither `varSpace` nor `valence` is provided, initialization proceeds
      in the new standard format.
    - Mixed-format coefficient dictionaries (using both holomorphic and real
      parts of complex coordinate systems) will be converted to a single
      preferred format (real or complex) based on dgcv settings.

    Attributes
    ----------
    coeff_dict : dict
        Canonical coefficient dictionary for the tensor field. Its format
        is shape-aware sparse. e.g., "skew" or "symmetric" shapes store
        at most one index tuple from a multi-index permutation class.
    data_shape : str
        Shape tag: "general", "symmetric", "skew", or "all".
    parameters : set
        Set of dgcv parameter symbols regarded as non-coordinate variables.
    max_degree : int
        Maximum tensor degree present in `coeff_dict` (cached). For homogeneous
        objects this matches `total_degree`.

    min_degree : int
        Minimum tensor degree present in `coeff_dict` (cached).

    total_degree : int
        Alias for `max_degree`.

    free_symbols : set
        Union of symbols appearing in the coefficient expressions and symbols
        appearing in the registered variable spaces referenced by this tensor
        field.

    coeff_free_symbols : set
        Set of symbols appearing in the coefficient expressions only.

    expanded_coeff_dict : dict
        Coefficient dictionary expanded to general shape. I.e., a shape-unaware
        space coefficient dictionary.

    homogeneous_parts : list[tensor_field_class]
        Decomposition of the tensor into homogeneous parts, grouped by valence
        profile induced by the tripled-key structure.
    varSpace : tuple
        ONLY FOR DEPRICATED COMPATIBILITY. Associated variable space.
        Not meaningful in general
    valence : tuple
        ONLY FOR DEPRICATED COMPATIBILITY. Tensor valence.
        Not meaningful in general

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

    format_combinations = {
        ("standard", "standard"): "standard",
        ("complex", "complex"): "complex",
        ("standard", "complex"): "complex",
        ("complex", "standard"): "complex",
        ("real", "real"): "real",
        ("standard", "real"): "real",
        ("real", "standard"): "real",
        ("complex", "real"): "mixed",
        ("real", "complex"): "mixed",
    }
    format_filter = {
        "ss": "standard",
        "ch": "complex",
        "ca": "complex",
        "rr": "real",
        "ri": "real",
        "standard": "standard",
        "complex": "complex",
        "real": "real",
    }

    def __init__(
        self,
        varSpace=None,
        coeff_dict=None,
        valence=None,
        data_shape: str = "general",
        dgcvType: str = "standard",
        _simplifyKW=None,
        variable_spaces: Optional[dict] = None,
        parameters=set(),
        _inheritance=None,
    ):
        if data_shape not in ("general", "symmetric", "skew", "all"):
            data_shape = "general"
        if coeff_dict is None:
            coeff_dict = {}
        if not isinstance(coeff_dict, dict):
            raise TypeError("`coeff_dict` must be a dictionary.")

        if _simplifyKW is None:
            _simplifyKW = {
                "simplify_rule": None,
                "simplify_ignore_list": None,
                "preferred_basis_element": None,
            }

        self.dgcvType = dgcvType
        self._simplifyKW = _simplifyKW
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "tensor_field"
        self.parameters = set(parameters)

        self._variable_spaces = (
            dict(variable_spaces) if isinstance(variable_spaces, dict) else {}
        )
        self._system_index_cache: Dict[str, Dict[Any, int]] = {}
        self._shape_checked = False

        self._preferred_basis_element = None
        self._expanded_coeff_dict = None
        self._coeffArray = None

        self._cd_formats = None
        self._realVarSpace = None
        self._holVarSpace = None
        self._antiholVarSpace = None
        self._imVarSpace = None
        self._varSpace_type = "standard"
        self._variable_spaces_types = None
        self._key_profiles = {}  # str values "s", "r", "c" tuples for standard, real, complex
        self._conj_key_profiles = {}  # numeric value index shif for conjugation
        self._coordinate_format = None
        self._free_symbols = None
        self._coeff_free_symbols = None
        self._max_degree = None
        self._min_degree = None
        self._validated_format: Literal[
            "open", "standard", "complex", "real", "mixed"
        ] = "open"

        if coeff_dict == {}:
            coeff_dict = {tuple(): 0}

        old_mode = (varSpace is not None) or (valence is not None)

        if (not old_mode) and self._is_scalar_coeff_dict(coeff_dict):
            self.varSpace = tuple() if varSpace is None else tuple(varSpace)
            self.valence = tuple()
            self.coeff_dict = {tuple(): coeff_dict.get(tuple(), 0)}
            self.data_shape = "all"
            self._shape_checked = True
        else:
            if old_mode:
                self.varSpace, self.valence, self.coeff_dict, self.data_shape = (
                    self._init_from_old_format(
                        varSpace, coeff_dict, valence, data_shape
                    )
                )
            else:
                self._variable_spaces = self._infer_variable_spaces_from_coeff_dict(
                    coeff_dict, self._variable_spaces
                )
                (
                    self.varSpace,
                    self.valence,
                    self.coeff_dict,
                    self.data_shape,
                    self._validated_format,
                ) = self._init_from_standard_data(
                    varSpace, coeff_dict, data_shape, self._variable_spaces
                )
                if _inheritance:
                    self._validated_format = _inheritance.get(
                        "_validated_format", self._validated_format
                    )

                if self.coeff_dict and self._validated_format == "mixed":
                    if (
                        get_dgcv_settings_registry().get(
                            "preferred_variable_format", None
                        )
                        == "real"
                    ):
                        new_cd = tensor_field_class._to_real_algo(
                            cd=self.coeff_dict,
                            vst=tensor_field_class._variable_spaces_types_algo(
                                self._variable_spaces
                            ),
                        )
                    else:
                        new_cd = tensor_field_class._to_complex_algo(
                            cd=self.coeff_dict,
                            vst=tensor_field_class._variable_spaces_types_algo(
                                self._variable_spaces
                            ),
                        )

                    self.coeff_dict, self._validated_format = new_cd, "complex"
            self._shape_checked = True

    def _set_degrees(self):
        m, mm = None, None
        for key in self.coeff_dict:
            lk = len(key) // 3
            if m is None:
                m = lk
                mm = lk
            else:
                m = max(m, lk)
                mm = min(mm, lk)
        self._max_degree, self._min_degree = m, mm

    @property
    def max_degree(self):
        if self._max_degree is None:
            self._set_degrees()
        return self._max_degree

    @property
    def min_degree(self):
        if self._min_degree is None:
            self._set_degrees()
        return self._min_degree

    @property
    def total_degree(self):
        return self.max_degree

    @staticmethod
    def _is_scalar_coeff_dict(d: Dict[Any, Any]) -> bool:
        return isinstance(d, dict) and (
            not d or (tuple() in d and all(k == tuple() for k in d))
        )

    def _init_from_old_format(self, varSpace, coeff_dict, valence, data_shape):
        if varSpace is None:
            raise TypeError("`varSpace` is required when `valence` is provided.")
        varSpace_t = tuple(varSpace)

        if valence is None:
            if not coeff_dict:
                valence_t = tuple()
            else:
                first_key = next(iter(coeff_dict))
                if not isinstance(first_key, tuple):
                    raise TypeError("Keys in `coeff_dict` must be tuples.")
                valence_t = (0,) * len(first_key)
        else:
            valence_t = tuple(valence)

        if not all(v in (0, 1) for v in valence_t):
            raise ValueError("`valence` must contain only 0s and 1s.")

        if len(set(valence_t)) > 1 and data_shape in ("symmetric", "skew"):
            raise ValueError(
                "Symmetry constraints require pure covariant or pure contravariant valence."
            )

        new_cd, inferred_type = self._convert_old_coeff_dict(
            varSpace_t, coeff_dict, valence_t
        )

        if self.dgcvType == "complex" and inferred_type != "complex":
            self.dgcvType = "standard"

        processed_cd, eff_shape, formatting = self._process_coeffs_dict_new(
            new_cd, data_shape, formatting=True
        )
        return varSpace_t, valence_t, processed_cd, eff_shape

    def _init_from_standard_data(
        self, varSpace, coeff_dict, data_shape, variable_spaces
    ):
        varSpace_t = tuple() if varSpace is None else tuple(varSpace)

        if not coeff_dict:
            return varSpace_t, tuple(), {tuple(): 0}, "all"

        first_key = next(iter(coeff_dict))
        deg = len(first_key) // 3
        valence_t = tuple(first_key[deg : 2 * deg])

        processed_cd, eff_shape, formatting = self._process_coeffs_dict_new(
            coeff_dict,
            data_shape,
            variable_spaces,
            formatting=True,
        )
        return varSpace_t, valence_t, processed_cd, eff_shape, formatting

    @staticmethod
    def _infer_variable_spaces_from_coeff_dict(coeff_dict: dict, seed: dict) -> dict:
        out = dict(seed) if isinstance(seed, dict) else {}
        for key in coeff_dict.keys():
            if not isinstance(key, tuple):
                continue
            kl = len(key)
            if kl == 0 or kl % 3 != 0:
                continue
            deg = kl // 3
            coord_ids = key[2 * deg :]
            for cid in coord_ids:
                if not isinstance(cid, str):
                    continue
                if cid in out:
                    continue
                info = vmf_lookup(
                    cid, path=True, relatives=True, flattened_relatives=True
                )
                if info.get("type") != "coordinate":
                    raise KeyError(
                        f"tensorField: coord_id '{cid}' is not registered in the VMF."
                    )
                flat = info.get("flattened_relatives", None)
                if not isinstance(flat, tuple) or len(flat) == 0:
                    raise KeyError(
                        f"tensorField: coord_id '{cid}' did not yield a usable variable space."
                    )
                out[cid] = flat
        return out

    def _require_cached_system(self, system_label: str) -> Tuple[Any, ...]:
        cached = self._variable_spaces.get(system_label, None)
        if isinstance(cached, tuple):
            self._system_index_cache.setdefault(
                system_label, {v: i for i, v in enumerate(cached)}
            )
            return cached
        if isinstance(cached, list):
            vs = tuple(cached)
            self._variable_spaces[system_label] = vs
            self._system_index_cache[system_label] = {v: i for i, v in enumerate(vs)}
            return vs
        raise KeyError(self._missing_system_msg(system_label))

    @property
    def free_symbols(self):
        if self._free_symbols is None:
            fs = set()
            for atoms in self._variable_spaces.values():
                fs.update(atoms)

            for v in self.coeff_dict.values():
                fs |= get_free_symbols(v)

            self._free_symbols = fs
        return self._free_symbols

    @property
    def coeff_free_symbols(self):
        if self._coeff_free_symbols is None:
            fs = set()
            for v in self.coeff_dict.values():
                fs |= get_free_symbols(v)
            self._coeff_free_symbols = fs
        return self._coeff_free_symbols

    @property
    def homogeneous_parts(self):
        new_dicts = dict()
        for k, v in self.coeff_dict.items():
            deg = len(k) // 3
            valence = tuple(k[deg : 2 * deg])
            new_dicts[valence] = new_dicts.get(deg, dict())
            new_dicts[valence][k] = v
        return [
            self.__class__(
                coeff_dict=cd,
                data_shape=self.data_shape,
                _simplifyKW=self._simplifyKW,
                variable_spaces=self._variable_spaces,
                parameters=self.parameters,
            )
            for cd in new_dicts.values()
        ]

    @staticmethod
    def _missing_system_msg(system_label: str) -> str:
        return (
            f"tensor_field_class: coordinate system '{system_label}' is not available in cached `variable_spaces`. "
            "Re-initialize with `variable_spaces={...}`."
        )

    @staticmethod
    def _format_combinator(formats, seed=None):
        fc = tensor_field_class.format_combinations
        ff = tensor_field_class.format_filter
        if len(formats) == 0:
            return seed if seed else "open"
        if seed:
            formatting = ff.get(seed, formats[0])
        else:
            formatting = formats[0]
        for format2 in formats:
            f1 = ff.get(formatting, "mixed")
            f2 = ff.get(format2, "mixed")
            formatting = fc.get((f1, f2), "mixed")
        return formatting

    def _convert_old_coeff_dict(
        self,
        varSpace_t: Tuple[Any, ...],
        old_cd: Dict[Tuple[int, ...], Any],
        valence_t: Tuple[int, ...],
    ) -> Tuple[Dict[Tuple[Any, ...], Any], str]:
        inferred = "standard"
        sys_for_var = {}
        for v in varSpace_t:
            info = vmf_lookup(v, path=True, relatives=False)
            p = info.get("path")
            if not (isinstance(p, tuple) and len(p) >= 2):
                raise KeyError(
                    f"tensor_field_class: variable '{v}' is not registered in the VMF."
                )
            branch, system_label = p[0], p[1]
            if branch == "complex_variable_systems":
                inferred = "complex"
            sys_for_var[v] = system_label

        for system_label in set(sys_for_var.values()):
            if system_label not in self._variable_spaces:
                info = vmf_lookup(
                    system_label, path=True, relatives=True, flattened_relatives=True
                )
                flat = info.get("flattened_relatives", None)
                if not isinstance(flat, tuple) or len(flat) == 0:
                    raise KeyError(self._missing_system_msg(system_label))
                self._variable_spaces[system_label] = flat
            self._require_cached_system(system_label)

        new_cd: Dict[Tuple[Any, ...], Any] = {}
        for key, value in old_cd.items():
            if _scalar_is_zero(value):
                continue
            if not isinstance(key, tuple):
                raise TypeError("Keys in `coeff_dict` must be tuples.")
            if len(key) != len(valence_t):
                raise ValueError("`coeff_dict` keys must match tensor degree.")

            idxs = []
            syslbls = []

            for i in key:
                if not isinstance(i, Integral):
                    raise TypeError("Old-style indices must be integers.")
                ii = int(i)
                if ii < 0 or ii >= len(varSpace_t):
                    raise ValueError("Old-style index out of range.")
                var = varSpace_t[ii]
                sys_label = sys_for_var[var]
                syslbls.append(sys_label)

                idx_map = self._system_index_cache.get(sys_label)
                if idx_map is None:
                    self._require_cached_system(sys_label)
                    idx_map = self._system_index_cache[sys_label]

                j = idx_map.get(var)
                if j is None:
                    raise KeyError(
                        f"tensor_field_class: variable '{var}' not found in cached system '{sys_label}'."
                    )
                idxs.append(j)

            nk = tuple(idxs + list(valence_t) + syslbls)
            new_cd[nk] = new_cd.get(nk, 0) + value

        new_cd = {k: v for k, v in new_cd.items() if not _scalar_is_zero(v)}
        if not new_cd:
            return {tuple(): 0}, inferred

        return new_cd, inferred

    @staticmethod
    def _process_coeffs_dict_new(
        data: Dict[Tuple[Any, ...], Any],
        shape: str,
        variable_spaces=None,
        formatting: bool = False,
    ) -> Tuple[Dict[Tuple[Any, ...], Any], str]:
        if shape not in ("general", "symmetric", "skew", "all"):
            shape = "general"

        if not isinstance(data, dict):
            raise TypeError("`coeff_dict` must be a dictionary.")

        def _parse_key(k, find_format=True, seed=None, inference_dict=None):
            deg = len(k) // 3
            if find_format:
                kf = tensor_field_class._profile_key_full_inference(
                    k, inference_dict=inference_dict
                )
                out_format = tensor_field_class._format_combinator(kf, seed=seed)
                return (deg, k, out_format)
            return (deg, k)

        def _sort_slot_key(slot):
            idx, valence, syslbl = slot
            idx_key = int(idx)
            return (str(syslbl), idx_key, int(valence))

        def _slotify(k, deg):
            idxs = k[:deg]
            valence_tuple = k[deg : 2 * deg]
            syslbls = k[2 * deg :]
            return tuple((idxs[i], valence_tuple[i], syslbls[i]) for i in range(deg))

        def _unslotify(slots):
            idxs = tuple(s[0] for s in slots)
            valence_tuple = tuple(s[1] for s in slots)
            syslbls = tuple(s[2] for s in slots)
            return idxs + valence_tuple + syslbls

        canon: Dict[Tuple[Any, ...], Any] = {}
        if formatting:
            out_format = "open"
        for k, v in data.items():
            if _scalar_is_zero(v):
                continue
            if formatting is False or out_format == "mixed":
                deg, kk = _parse_key(
                    k, find_format=False, inference_dict=variable_spaces
                )
            else:
                deg, kk, out_format = _parse_key(
                    k, seed=out_format, inference_dict=variable_spaces
                )
            if deg <= 1 or shape == "general":
                canon[kk] = canon.get(kk, 0) + v
                continue

            slots = _slotify(kk, deg)

            if shape == "skew" and len(set(slots)) < len(slots):
                continue

            if shape == "symmetric":
                sorted_slots = tuple(sorted(slots, key=_sort_slot_key))
                nk = _unslotify(sorted_slots)
                vv = v
            else:
                sign, sorted_slots = tuple(
                    permSign(slots, returnSorted=True, key=_sort_slot_key)
                )
                nk = _unslotify(sorted_slots)
                vv = sign * v

            canon[nk] = canon.get(nk, 0) + vv

        canon = {k: v for k, v in canon.items() if not _scalar_is_zero(v)}
        if not canon:
            if formatting:
                return {tuple(): 0}, "all", "open"
            return {tuple(): 0}, "all"

        if formatting:
            return canon, shape, out_format
        return canon, shape

    @staticmethod
    def _expand_special_to_general(
        data: Dict[Tuple[Any, ...], Any], shape: str
    ) -> Dict[Tuple[Any, ...], Any]:
        if shape not in ("general", "symmetric", "skew", "all"):
            raise ValueError("Invalid data_shape.")

        if not isinstance(data, dict):
            raise TypeError("`coeff_dict` must be a dictionary.")

        if shape in ("general", "all"):
            return {k: v for k, v in data.items() if not _scalar_is_zero(v)}

        nz = {k: v for k, v in data.items() if not _scalar_is_zero(v)}
        if not nz:
            return {}

        out: Dict[Tuple[Any, ...], Any] = {}

        first_key = next(iter(nz))
        if not isinstance(first_key, tuple) or len(first_key) % 3 != 0:
            raise ValueError("Invalid coeff_dict key format.")

        deg = len(first_key) // 3

        def slotify(k):
            idxs = k[:deg]
            valence_tuple = k[deg : 2 * deg]
            syslbls = k[2 * deg :]
            return tuple((idxs[i], valence_tuple[i], syslbls[i]) for i in range(deg))

        def unslotify(slots):
            idxs = tuple(s[0] for s in slots)
            valence_tuple = tuple(s[1] for s in slots)
            syslbls = tuple(s[2] for s in slots)
            return idxs + valence_tuple + syslbls

        for k, v in nz.items():
            slots = slotify(k)

            for perm in itertools.permutations(range(deg)):
                perm_slots = tuple(slots[i] for i in perm)
                nk = unslotify(perm_slots)

                if shape == "symmetric":
                    vv = v
                else:
                    vv = permSign(perm) * v

                out[nk] = out.get(nk, 0) + vv

        return {k: v for k, v in out.items() if not _scalar_is_zero(v)}

    @staticmethod
    def _variable_spaces_types_algo(vs={}):
        out = {}
        if not isinstance(vs, dict):
            vs = {}
        for system, coordinates in vs.items():
            info = vmf_lookup(system)
            sys_type = info.get("sub_type", "standard")
            if sys_type == "complex":
                L = len(coordinates)
                fourth = L // 4
                breaks = [fourth, 2 * fourth, 3 * fourth]
                out[system] = {"type": sys_type, "breaks": breaks}
            else:
                out[system] = {"type": sys_type}
        return out

    @property
    def variable_spaces_types(self):
        if self._variable_spaces_types is None:
            self._variable_spaces_types = (
                tensor_field_class._variable_spaces_types_algo(
                    getattr(self, "_variable_spaces", {})
                )
            )
        return self._variable_spaces_types

    @staticmethod
    def _slot_allowed(plan, syslbl, idx):
        if not plan:
            return True

        if syslbl in plan.get("skip_systems", ()):
            return False

        si = plan.get("skip_indices", {}).get(syslbl)
        if si and idx in si:
            return False

        scoped = plan.get("systems", {}).get(syslbl, None)
        if scoped is None:
            return True
        return idx in scoped

    def infer_varSpace(
        self,
        formatting: Literal["real", "complex", "any"] = "complex",
        *,
        return_dict: bool = False,
    ):
        cache = getattr(self, "_infer_varSpace_cache", None)
        if cache is None:
            cache = self._infer_varSpace_cache = {}

        key = (formatting, bool(return_dict))
        hit = cache.get(key, None)
        if hit is not None:
            return hit

        if formatting not in ("real", "complex", "any"):
            raise ValueError("formatting must be one of: 'real', 'complex', 'any'")

        systems = getattr(self, "_variable_spaces", None)
        if not isinstance(systems, dict) or not systems:
            out = (tuple(), {}) if return_dict else tuple()
            cache[key] = out
            return out

        std_bucket = []
        holo_bucket = []
        anti_bucket = []
        real_bucket = []
        imag_bucket = []

        sys_info_cache: Dict[Any, dict] = {}
        for syslbl in sorted(systems.keys(), key=lambda x: str(x)):
            info = vmf_lookup(
                syslbl, path=True, relatives=True, flattened_relatives=True
            )
            sys_info_cache[syslbl] = info

            if info.get("type") != "coordinate":
                continue

            sub = info.get("sub_type", None)
            rel = info.get("relatives") or {}

            if sub != "complex":
                st = rel.get("standard", None)
                if isinstance(st, tuple) and st:
                    std_bucket.append((syslbl, st))
                continue

            h = rel.get("holo", None)
            a = rel.get("anti", None)
            r = rel.get("real", None)
            i = rel.get("imag", None)

            if isinstance(h, tuple) and h:
                holo_bucket.append((syslbl, h))
            if isinstance(a, tuple) and a:
                anti_bucket.append((syslbl, a))
            if isinstance(r, tuple) and r:
                real_bucket.append((syslbl, r))
            if isinstance(i, tuple) and i:
                imag_bucket.append((syslbl, i))

        out_vars = []

        if formatting == "any":
            for _, t in std_bucket:
                out_vars.extend(t)
            for _, t in holo_bucket:
                out_vars.extend(t)
            for _, t in anti_bucket:
                out_vars.extend(t)
            for _, t in real_bucket:
                out_vars.extend(t)
            for _, t in imag_bucket:
                out_vars.extend(t)

        elif formatting == "complex":
            for _, t in std_bucket:
                out_vars.extend(t)
            for _, t in holo_bucket:
                out_vars.extend(t)
            for _, t in anti_bucket:
                out_vars.extend(t)

        else:  # "real"
            for _, t in std_bucket:
                out_vars.extend(t)
            for _, t in real_bucket:
                out_vars.extend(t)
            for _, t in imag_bucket:
                out_vars.extend(t)

        out_t = tuple(out_vars)

        if not return_dict:
            cache[key] = out_t
            return out_t

        loc: Dict[Any, Tuple[Any, int]] = {}
        for syslbl, info in sys_info_cache.items():
            if info.get("type") != "coordinate":
                continue
            flat = info.get("flattened_relatives", None)
            if not isinstance(flat, tuple) or not flat:
                continue

            rel = info.get("relatives") or {}
            sub = info.get("sub_type", None)

            if sub == "complex":
                h = rel.get("holo") or tuple()
                a = rel.get("anti") or tuple()
                r = rel.get("real") or tuple()
                i = rel.get("imag") or tuple()

                off_h = 0
                off_a = off_h + len(h)
                off_r = off_a + len(a)
                off_i = off_r + len(r)

                for j, v in enumerate(h):
                    loc[v] = (syslbl, off_h + j)
                for j, v in enumerate(a):
                    loc[v] = (syslbl, off_a + j)
                for j, v in enumerate(r):
                    loc[v] = (syslbl, off_r + j)
                for j, v in enumerate(i):
                    loc[v] = (syslbl, off_i + j)
            else:
                st = rel.get("standard") or flat
                if isinstance(st, tuple) and st:
                    for j, v in enumerate(st):
                        loc[v] = (syslbl, j)

        loc_out = {v: loc[v] for v in out_t if v in loc}
        out = (out_t, loc_out)
        cache[key] = out
        return out

    def infer_minimal_varSpace(
        self,
        *,
        return_dict: bool = False,
    ):
        vs_all, loc = self.infer_varSpace(formatting="any", return_dict=True)

        present = set()
        cd = getattr(self, "coeff_dict", None)
        if isinstance(cd, dict):
            for k in cd.keys():
                if not isinstance(k, tuple):
                    continue
                kl = len(k)
                if kl == 0 or kl % 3 != 0:
                    continue
                deg = kl // 3
                idxs = k[:deg]
                syslbls = k[2 * deg :]
                for idx, sys in zip(idxs, syslbls):
                    present.add((sys, idx))

        out = tuple(
            v
            for v in vs_all
            if (loc.get(v) in (None,)) is False and (loc[v][0], loc[v][1]) in present
        )

        if not return_dict:
            return out

        out_loc = {v: loc[v] for v in out if v in loc}
        return out, out_loc

    def _to_real(self, plan=None):
        new_dict = tensor_field_class._to_real_algo(
            plan=plan, cd=self.coeff_dict, vst=self.variable_spaces_types
        )
        return self.__class__(
            coeff_dict=new_dict,
            data_shape=self.data_shape,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=self._variable_spaces,
        )

    @staticmethod
    def _to_real_algo(plan=None, cd={}, vst={}):
        new_dict = {}

        for k_seed, v_seed in cd.items():
            current_contribution = {k_seed: v_seed}
            degree = len(k_seed) // 3
            valences = k_seed[degree : 2 * degree]
            systems = k_seed[2 * degree :]

            for idx in range(degree):
                new_contribution = {}

                for k, v in current_contribution.items():
                    idxs = k[:degree]

                    br1, br2, br3 = (
                        idxs[idx],
                        valences[idx],
                        systems[idx],
                    )  # break point data
                    sys_data = vst[br3]

                    if sys_data["type"] != "complex":
                        new_contribution[k] = new_contribution.get(k, 0) + v
                        continue

                    if br1 >= sys_data["breaks"][1]:
                        new_contribution[k] = new_contribution.get(k, 0) + v
                        continue

                    if not tensor_field_class._slot_allowed(plan, br3, br1):
                        new_contribution[k] = new_contribution.get(k, 0) + v
                        continue

                    lead = list(idxs[:idx])
                    tail = list(idxs[idx + 1 :])
                    term_tail = list(valences) + list(systems)

                    if br1 >= sys_data["breaks"][0]:  # antiholomorphic variable
                        real_idx = br1 + sys_data["breaks"][0]
                        im_idx = br1 + sys_data["breaks"][1]

                        if br2 == 1:  # contravariant slot
                            real_term = lead + [real_idx] + tail + term_tail
                            im_term = lead + [im_idx] + tail + term_tail
                            new_contribution[tuple(real_term)] = (
                                new_contribution.get(tuple(real_term), 0) + half * v
                            )
                            new_contribution[tuple(im_term)] = (
                                new_contribution.get(tuple(im_term), 0)
                                + half * imag_unit() * v
                            )
                        else:  # covariant slot
                            real_term = lead + [real_idx] + tail + term_tail
                            im_term = lead + [im_idx] + tail + term_tail
                            new_contribution[tuple(real_term)] = (
                                new_contribution.get(tuple(real_term), 0) + v
                            )
                            new_contribution[tuple(im_term)] = (
                                new_contribution.get(tuple(im_term), 0)
                                - imag_unit() * v
                            )

                    else:  # holomorphic variable
                        real_idx = br1 + sys_data["breaks"][1]
                        im_idx = br1 + sys_data["breaks"][2]

                        if br2 == 1:  # contravariant slot
                            real_term = lead + [real_idx] + tail + term_tail
                            im_term = lead + [im_idx] + tail + term_tail
                            new_contribution[tuple(real_term)] = (
                                new_contribution.get(tuple(real_term), 0) + half * v
                            )
                            new_contribution[tuple(im_term)] = (
                                new_contribution.get(tuple(im_term), 0)
                                - half * imag_unit() * v
                            )
                        else:  # covariant slot
                            real_term = lead + [real_idx] + tail + term_tail
                            im_term = lead + [im_idx] + tail + term_tail
                            new_contribution[tuple(real_term)] = (
                                new_contribution.get(tuple(real_term), 0) + v
                            )
                            new_contribution[tuple(im_term)] = (
                                new_contribution.get(tuple(im_term), 0)
                                + imag_unit() * v
                            )

                current_contribution = new_contribution

            for key, val in current_contribution.items():
                if not _scalar_is_zero(val):
                    new_dict[key] = new_dict.get(key, 0) + val

        if not new_dict:
            new_dict = {tuple(): 0}

        return new_dict

    def _to_complex(self, plan=None):
        new_dict = tensor_field_class._to_complex_algo(
            plan=plan, cd=self.coeff_dict, vst=self.variable_spaces_types
        )
        return self.__class__(
            coeff_dict=new_dict,
            data_shape=self.data_shape,
            dgcvType="complex",
            _simplifyKW=self._simplifyKW,
            variable_spaces=self._variable_spaces,
        )

    @staticmethod
    def _to_complex_algo(plan=None, cd={}, vst={}):
        new_dict = {}

        for k_seed, v_seed in cd.items():
            current_contribution = {k_seed: v_seed}
            degree = len(k_seed) // 3
            valences = k_seed[degree : 2 * degree]
            systems = k_seed[2 * degree :]

            for idx in range(degree):
                new_contribution = {}

                for k, v in current_contribution.items():
                    idxs = k[:degree]

                    vb = valences[idx]
                    syslbl = systems[idx]
                    sys_data = vst[syslbl]

                    if sys_data["type"] != "complex":
                        new_contribution[k] = new_contribution.get(k, 0) + v
                        continue

                    # determine which real/imag block we are in
                    br0, br1, br2 = sys_data["breaks"]

                    if idxs[idx] < br1:
                        new_contribution[k] = (
                            new_contribution.get(k, 0) + v
                        )  # holomorphic / antiholomorphic already
                        continue

                    if not tensor_field_class._slot_allowed(plan, syslbl, idxs[idx]):
                        new_contribution[k] = new_contribution.get(k, 0) + v
                        continue

                    lead = list(idxs[:idx])
                    tail = list(idxs[idx + 1 :])
                    rest = list(valences) + list(systems)

                    j = idxs[idx]

                    # real variable
                    if j < br2:
                        holo_idx = j - br1
                        anti_idx = holo_idx + br0

                        if vb == 1:  # contravariant
                            holo_term = lead + [holo_idx] + tail + rest
                            anti_term = lead + [anti_idx] + tail + rest
                            new_contribution[tuple(holo_term)] = (
                                new_contribution.get(tuple(holo_term), 0) + v
                            )
                            new_contribution[tuple(anti_term)] = (
                                new_contribution.get(tuple(anti_term), 0) + v
                            )
                        else:
                            holo_term = lead + [holo_idx] + tail + rest
                            anti_term = lead + [anti_idx] + tail + rest
                            new_contribution[tuple(holo_term)] = (
                                new_contribution.get(tuple(holo_term), 0) + v * half
                            )
                            new_contribution[tuple(anti_term)] = (
                                new_contribution.get(tuple(anti_term), 0) + v * half
                            )

                    # imaginary variable
                    else:
                        holo_idx = j - br2
                        anti_idx = holo_idx + br0

                        if vb == 1:  # contravariant
                            holo_term = lead + [holo_idx] + tail + rest
                            anti_term = lead + [anti_idx] + tail + rest
                            new_contribution[tuple(holo_term)] = (
                                new_contribution.get(tuple(holo_term), 0)
                                + imag_unit() * v
                            )
                            new_contribution[tuple(anti_term)] = (
                                new_contribution.get(tuple(anti_term), 0)
                                - imag_unit() * v
                            )
                        else:
                            holo_term = lead + [holo_idx] + tail + rest
                            anti_term = lead + [anti_idx] + tail + rest
                            new_contribution[tuple(holo_term)] = (
                                new_contribution.get(tuple(holo_term), 0)
                                - imag_unit() * v * half
                            )
                            new_contribution[tuple(anti_term)] = (
                                new_contribution.get(tuple(anti_term), 0)
                                + imag_unit() * v * half
                            )

                current_contribution = new_contribution

            for key, val in current_contribution.items():
                if not _scalar_is_zero(val):
                    new_dict[key] = new_dict.get(key, 0) + val

        if not new_dict:
            new_dict = {tuple(): 0}

        return new_dict

    def _get_parts(self, *, type: str):
        t = str(type).lower()

        if t in {"holo", "holomorphic"}:
            want = "ch"
            do_all_to_sym = True
            mode = "pure"
        elif t in {"anti", "antiholo", "antiholomorphic"}:
            want = "ca"
            do_all_to_sym = True
            mode = "pure"
        elif t == "mixed":
            want = None
            do_all_to_sym = True
            mode = "mixed"
        elif t == "standard":
            want = "ss"
            do_all_to_sym = False
            mode = "pure"
        elif t in {"real", "imag", "imaginary"}:
            cfg = get_dgcv_settings_registry()
            if not cfg.get("forgo_warnings", False):
                info_f = getattr(self, "_coordinate_format_info", None)
                info = info_f() if callable(info_f) else None
                if not (isinstance(info, dict) and info.get("dgcv_type") == "complex"):
                    warnings.warn(
                        "Requested real/imaginary part of a tensor field whose variables are not registered as dgcv complex coordinate systems. "
                        "The result is only well-defined if such variables are assumed real.",
                        UserWarning,
                        stacklevel=2,
                    )

            half = rational(1, 2)
            if t == "real":
                return half * (self + conjugate(self))
            return imag_unit() * half * (-self + conjugate(self))

        else:
            raise ValueError(
                "type must be one of: 'holomorphic', 'antiholomorphic', 'real', 'imaginary', 'mixed', 'standard'."
            )

        obj = self
        if do_all_to_sym:
            from .conversions import allToSym

            obj = allToSym(obj)

        new_cd = {}
        for k, v in obj.coeff_dict.items():
            if not v:
                continue
            prof = obj._profile_key(k)

            if mode == "pure":
                if all(tag == want for tag in prof):
                    new_cd[k] = new_cd.get(k, 0) + v
            else:  # "mixed"
                if len(set(prof)) > 1:
                    new_cd[k] = new_cd.get(k, 0) + v

        if not new_cd:
            new_cd = {tuple(): 0}

        return self.__class__(
            coeff_dict=new_cd,
            data_shape=getattr(self, "data_shape", "general"),
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=self._variable_spaces,
        )

    @staticmethod
    def _profile_key_full_inference(k, _variable_dict=None, inference_dict={}):
        degree = len(k) // 3
        idxs = k[:degree]
        systems = k[2 * degree :]

        out = []
        if _variable_dict is None:
            _variable_dict = tensor_field_class._variable_spaces_types_algo(
                inference_dict
            )

        for idx, sys in zip(idxs, systems):
            sys_data = _variable_dict.get(sys)
            if sys_data is None:
                raise KeyError(
                    "At least one system label in a coeff_dict key is not registered in the VMF."
                )

            if sys_data is None or sys_data.get("type") != "complex":
                out.append("ss")
                continue

            b0, b1, b2 = sys_data["breaks"]

            if idx < b0:
                out.append("ch")
            elif idx < b1:
                out.append("ca")
            elif idx < b2:
                out.append("rr")
            else:
                out.append("ri")

        return tuple(out)

    def _profile_key(self, k):
        prof = self._key_profiles.get(k)
        if prof is not None:
            return prof
        prof = self._profile_key_full_inference(
            k, _variable_dict=self.variable_spaces_types
        )
        self._key_profiles[k] = prof
        return prof

    def _flip_format(self, slot: int, k, v, *, to_kind: str):
        degree = len(k) // 3
        idxs = list(k[:degree])
        valences = k[degree : 2 * degree]
        systems = k[2 * degree :]

        sys = systems[slot]
        sys_data = self.variable_spaces_types.get(sys)
        if sys_data is None or sys_data.get("type") != "complex":
            return {k: v}

        b0, b1, b2 = sys_data["breaks"]
        idx = idxs[slot]
        vb = valences[slot]

        half = rational(1, 2)
        imU = imag_unit()

        def _mk(new_idx, coeff):
            new_idxs = list(idxs)
            new_idxs[slot] = new_idx
            nk = tuple(new_idxs) + tuple(valences) + tuple(systems)
            return nk, coeff

        if to_kind == "r":
            if idx >= b1:
                return {k: v}

            if idx >= b0:
                real_idx = idx + b0
                imag_idx = idx + b1
                if vb == 1:
                    nk1, c1 = _mk(real_idx, half * v)
                    nk2, c2 = _mk(imag_idx, half * imU * v)
                else:
                    nk1, c1 = _mk(real_idx, v)
                    nk2, c2 = _mk(imag_idx, -imU * v)
                return {nk1: c1, nk2: c2}

            real_idx = idx + b1
            imag_idx = idx + b2
            if vb == 1:
                nk1, c1 = _mk(real_idx, half * v)
                nk2, c2 = _mk(imag_idx, -half * imU * v)
            else:
                nk1, c1 = _mk(real_idx, v)
                nk2, c2 = _mk(imag_idx, imU * v)
            return {nk1: c1, nk2: c2}

        if to_kind == "c":
            if idx < b1:
                return {k: v}

            if idx < b2:
                holo_idx = idx - b1
                anti_idx = idx - b1
                if vb == 1:
                    nk1, c1 = _mk(holo_idx, v)
                    nk2, c2 = _mk(anti_idx, v)
                else:
                    nk1, c1 = _mk(holo_idx, half * v)
                    nk2, c2 = _mk(anti_idx, half * v)
                return {nk1: c1, nk2: c2}

            holo_idx = idx - b2
            anti_idx = idx - b2
            if vb == 1:
                nk1, c1 = _mk(holo_idx, imU * v)
                nk2, c2 = _mk(anti_idx, -imU * v)
            else:
                nk1, c1 = _mk(holo_idx, -imU * half * v)
                nk2, c2 = _mk(anti_idx, imU * half * v)
            return {nk1: c1, nk2: c2}

        raise ValueError("to_kind must be 'r' or 'c'")

    def __dgcv_apply__(self, fun, **kwargs):
        return self.apply(fun, **kwargs)

    def apply(
        self,
        func,
        *,
        drop_zeros: bool = True,
        data_shape: str | None = None,
        dgcvType: str | None = None,
        _simplifyKW=None,
        variable_spaces=None,
        **func_kwargs,
    ):
        if not callable(func):
            raise TypeError("apply(func): `func` must be callable.")

        out = {}
        for k, v in self.coeff_dict.items():
            vv = func(v, **func_kwargs)
            if drop_zeros and _scalar_is_zero(vv):
                continue
            out[k] = vv

        if not out:
            out = {tuple(): 0}

        return self.__class__(
            coeff_dict=out,
            data_shape=self.data_shape if data_shape is None else data_shape,
            dgcvType=self.dgcvType if dgcvType is None else dgcvType,
            _simplifyKW=self._simplifyKW if _simplifyKW is None else _simplifyKW,
            variable_spaces=self._variable_spaces
            if variable_spaces is None
            else variable_spaces,
        )

    def __dgcv_conjugate__(self):
        from .backends._symbolic_router import conjugate

        new_cd = {}
        vst = self.variable_spaces_types
        cache = self._conj_key_profiles

        for k, v in self.coeff_dict.items():
            if k == tuple():
                nk = tuple()
                nv = conjugate(v)
                new_cd[nk] = new_cd.get(nk, 0) + nv
                continue

            shifts = cache.get(k)
            if shifts is None:
                L = len(k)
                d = L // 3
                idxs = k[:d]
                systems = k[2 * d :]

                out = []
                for idx, sys in zip(idxs, systems):
                    sys_data = vst.get(sys)
                    if sys_data is None or sys_data.get("type") != "complex":
                        out.append(0)
                        continue

                    b0, b1, b2 = sys_data["breaks"]

                    if idx < b0:
                        out.append(b0)
                    elif idx < b1:
                        out.append(-b0)
                    else:
                        out.append(0)

                shifts = tuple(out)
                cache[k] = shifts

            d = len(shifts)
            idxs = k[:d]
            tail = k[d:]  # valences + systems

            new_idxs = tuple(i + s for i, s in zip(idxs, shifts))
            nk = new_idxs + tail
            nv = conjugate(v)
            new_cd[nk] = new_cd.get(nk, 0) + nv

        return self.__class__(
            coeff_dict=new_cd,
            data_shape=self.data_shape,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=self._variable_spaces,
        )

    def __str__(self):
        return tensor_field_printer2(self)

    def _repr_latex_(self, raw: bool = False, **kwargs):
        return tensor_field_latex2(self, raw=raw)

    def _latex(self, printer=None, raw: bool = True, **kwargs):
        return self._repr_latex_(raw=raw)

    def _latex_(self):
        return self._repr_latex_(raw=True)

    def _legacy_system_label(self):
        syslbl = None
        for k, v in self.coeff_dict.items():
            if _scalar_is_zero(v):
                continue
            if not isinstance(k, tuple) or len(k) == 0 or len(k) % 3 != 0:
                return None
            deg = len(k) // 3
            sys = k[2 * deg :]
            if not sys:
                continue
            s0 = sys[0]
            if any(s != s0 for s in sys):
                return None
            if syslbl is None:
                syslbl = s0
            elif syslbl != s0:
                return None
        return syslbl

    def _legacy_varSpace(self):
        syslbl = self._legacy_system_label()
        if syslbl is None:
            return None
        vs = self._variable_spaces.get(syslbl, None)
        if not isinstance(vs, tuple):
            return None
        if self.total_degree == 0:
            return vs
        n = len(vs)
        if n == 0:
            return None
        for k, v in self.coeff_dict.items():
            if _scalar_is_zero(v):
                continue
            deg = len(k) // 3
            idxs = k[:deg]
            if any((not isinstance(i, Integral)) or i < 0 or i >= n for i in idxs):
                return None
        return vs

    def _legacy_coeff_dict(self):
        vs = self._legacy_varSpace()
        if vs is None:
            return None
        if self.total_degree == 0:
            return {tuple(): self.coeff_dict.get(tuple(), 0)}
        out = {}
        deg = self.total_degree
        for k, v in self.coeff_dict.items():
            if _scalar_is_zero(v):
                continue
            idxs = k[:deg]
            out[tuple(idxs)] = v
        if not out:
            return {(0,) * deg: 0}
        return out

    def _require_legacy_view(self, what: str):
        if self._legacy_varSpace() is None:
            raise ValueError(
                f"{what} is only available when the tensor is supported on a single cached coordinate system."
            )

    def holomorphic_part(self):
        return self._get_parts(type="holomorphic")

    def antiholomorphic_part(self):
        return self._get_parts(type="antiholomorphic")

    def mixed_term_component(self):
        return self._get_parts(type="mixed")

    def pure_standard_coordinate_terms(self):
        return self._get_parts(type="standard")

    def real_part(self):
        return self._get_parts(type="real")

    def imaginary_part(self):
        return self._get_parts(type="imaginary")

    @property
    def cd_formats(self):
        if self._cd_formats is not None:
            return self._cd_formats
        if self.dgcvType == "standard":
            return self._cd_formats
        vs = self._legacy_varSpace()
        cd = self._legacy_coeff_dict()
        if vs is None or cd is None:
            return self._cd_formats
        (
            populate,
            self._realVarSpace,
            self._holVarSpace,
            self._antiholVarSpace,
            self._imVarSpace,
        ) = _coeff_dict_formatter(
            vs,
            cd,
            self.valence,
            self.total_degree,
            getattr(self, "_varSpace_type", "standard"),
            self.data_shape if self.data_shape != "all" else "general",
        )
        self._cd_formats = populate
        return populate

    @property
    def realVarSpace(self):
        if self.dgcvType == "standard":
            return self._realVarSpace
        if self._realVarSpace is None or self._imVarSpace is None:
            self.cd_formats
        if self._realVarSpace is None or self._imVarSpace is None:
            return None
        return self._realVarSpace + self._imVarSpace

    @property
    def holVarSpace(self):
        if self.dgcvType == "standard":
            return self._holVarSpace
        if self._holVarSpace is None:
            self.cd_formats
        return self._holVarSpace

    @property
    def antiholVarSpace(self):
        if self.dgcvType == "standard":
            return self._antiholVarSpace
        if self._antiholVarSpace is None:
            self.cd_formats
        return self._antiholVarSpace

    @property
    def compVarSpace(self):
        if self.dgcvType == "standard":
            if self._holVarSpace is None or self._antiholVarSpace is None:
                return None
            return self._holVarSpace + self._antiholVarSpace
        if self._holVarSpace is None or self._antiholVarSpace is None:
            self.cd_formats
        if self._holVarSpace is None or self._antiholVarSpace is None:
            return None
        return self._holVarSpace + self._antiholVarSpace

    @property
    def coeffArray(self):
        if self._coeffArray is not None:
            return self._coeffArray

        self._require_legacy_view("coeffArray")

        vs = self._legacy_varSpace()
        deg = self.total_degree
        n = len(vs)
        shape = (n,) * deg

        def entry_rule(indexTuple):
            if self.data_shape == "symmetric":
                indexTuple = tuple(sorted(indexTuple))
            cd = self.expanded_coeff_dict
            return cd.get(indexTuple, 0)

        def generate_indices(sh):
            if len(sh) == 1:
                return [(i,) for i in range(sh[0])]
            return [(i,) + t for i in range(sh[0]) for t in generate_indices(sh[1:])]

        sparse_data = {idx: entry_rule(idx) for idx in generate_indices(shape)}
        flat = {_spool(idx_tup, shape): v for idx_tup, v in sparse_data.items()}

        arr = array_dgcv.__new__(array_dgcv)
        arr._data = flat
        arr.shape = shape
        arr.ndim = len(shape)
        arr._dgcv_class_check = retrieve_passkey()
        arr._dgcv_category = "array"

        self._coeffArray = arr
        return self._coeffArray

    @property
    def expanded_coeff_dict(self):
        if self._expanded_coeff_dict is not None:
            return self._expanded_coeff_dict

        if self.data_shape in ("general", "all"):
            if self.data_shape == "all":
                self._expanded_coeff_dict = {tuple(): self.coeff_dict.get(tuple(), 0)}
            else:
                self._expanded_coeff_dict = {
                    k: v for k, v in self.coeff_dict.items() if not _scalar_is_zero(v)
                }
            return self._expanded_coeff_dict

        self._expanded_coeff_dict = self._expand_special_to_general(
            self.coeff_dict, self.data_shape
        )
        return self._expanded_coeff_dict

    def swap_tensor_valence(self):
        def key_change(key):
            deg = len(key) // 3
            new_k = tuple(
                j if c < deg or 2 * deg <= c else 1 - j for c, j in enumerate(key)
            )
            return new_k

        cd = {key_change(key): value for key, value in self.coeff_dict.items()}
        if query_dgcv_categories(self, "differential_form"):
            return vector_field_class(
                coeff_dict=cd, _simplifyKW=self._simplifyKW, parameters=self.parameters
            )
        if query_dgcv_categories(self, "vector_field"):
            return differential_form_class(
                coeff_dict=cd, _simplifyKW=self._simplifyKW, parameters=self.parameters
            )
        return tensor_field_class(
            coeff_dict=cd, _simplifyKW=self._simplifyKW, parameters=self.parameters
        )

    def __dgcv_re__(self):
        return self.real_part()

    def __dgcv_im__(self):
        return self.imaginary_part()

    def _is_scalar(self) -> bool:
        return self.data_shape == "all" and self.total_degree == 0

    def _scalar_value(self):
        return self.coeff_dict.get(tuple(), 0)

    def _with_same_meta(self, *, coeff_dict, data_shape=None, variable_spaces=None):
        return self.__class__(
            coeff_dict=coeff_dict,
            data_shape=self.data_shape if data_shape is None else data_shape,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=self._variable_spaces
            if variable_spaces is None
            else variable_spaces,
        )

    def _merged_variable_spaces(self, other):
        out = dict(self._variable_spaces)
        ov = getattr(other, "_variable_spaces", {})
        for k, v in ov.items():
            if k in out and out[k] != v:
                raise ValueError(
                    f"Incompatible cached variable spaces {out[k]} and {v} for system '{k}'."
                )
            out[k] = v
        return out

    def _coerce_to_general(self):
        if self.data_shape == "general":
            return self
        if self.data_shape == "all":
            return self
        g = self._expand_special_to_general(self.coeff_dict, self.data_shape)
        out = self._with_same_meta(coeff_dict=g, data_shape="general")
        out._shape_checked = True
        return out

    def _maybe_promote_general_to(self, target_shape: str):
        if self._shape_checked:
            return
        self._shape_checked = True
        if self.data_shape != "general":
            return
        if target_shape not in ("symmetric", "skew"):
            return
        if not self.valence or len(set(self.valence)) != 1:
            return
        new_cd, eff_shape = self._process_coeffs_dict_new(self.coeff_dict, target_shape)
        if eff_shape == target_shape:
            self.coeff_dict = new_cd
            self.data_shape = eff_shape
            self._expanded_coeff_dict = None
            self._coeffArray = None
            self._cd_formats = None

    def __add__(self, other):
        if _scalar_is_zero(other):
            return self
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self._is_scalar() or other._is_scalar():
            raise TypeError("Cannot add tensors of different degrees.")

        return self._add_tensor(other, coerce_shapes=True)

    def __radd__(self, other):
        if _scalar_is_zero(other):
            return self
        return NotImplemented

    def __neg__(self):
        if self._is_scalar():
            return self.__class__(
                coeff_dict={tuple(): -self._scalar_value()},
                data_shape="all",
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=self._variable_spaces,
            )
        return self._with_same_meta(
            coeff_dict={k: -v for k, v in self.coeff_dict.items()},
            variable_spaces=self._variable_spaces,
        )

    def __sub__(self, other):
        if _scalar_is_zero(other):
            return self
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self + (-other)

    def __matmul__(self, other):
        tf = other
        if check_dgcv_category(tf):
            coerce = getattr(tf, "as_tensor_field", None)
            if callable(coerce):
                tf = coerce()
        if not isinstance(tf, self.__class__):
            return NotImplemented
        return self._shape_product(tf, kind="general")

    def __mul__(self, other):
        if check_dgcv_scalar(other):
            if self._is_scalar():
                return self.__class__(
                    coeff_dict={tuple(): other * self._scalar_value()},
                    data_shape="all",
                    dgcvType=self.dgcvType,
                    _simplifyKW=self._simplifyKW,
                    variable_spaces=self._variable_spaces,
                )
            return self._with_same_meta(
                coeff_dict={k: other * v for k, v in self.coeff_dict.items()},
                variable_spaces=self._variable_spaces,
            )

        tf = other
        if check_dgcv_category(tf):
            coerce = getattr(tf, "as_tensor_field", None)
            if callable(coerce):
                tf = coerce()
        if not isinstance(tf, self.__class__):
            return NotImplemented

        return self._shape_product(tf, kind="general")

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        if check_dgcv_scalar(scalar):
            return self * exact_reciprocal(scalar)
        return NotImplemented

    def _add_tensor(self, other, *, coerce_shapes: bool, _return_raw: bool = False):
        a = self
        b = other

        a_shape = a.data_shape
        b_shape = b.data_shape

        if coerce_shapes:
            if a_shape == "general" and b_shape in ("symmetric", "skew"):
                a._maybe_promote_general_to(b_shape)
                a_shape = a.data_shape
            if b_shape == "general" and a_shape in ("symmetric", "skew"):
                b._maybe_promote_general_to(a_shape)
                b_shape = b.data_shape

        if a_shape == "all":
            out_shape = b_shape
        elif b_shape == "all":
            out_shape = a_shape
        elif a_shape == b_shape:
            out_shape = a_shape
        else:
            out_shape = "general"

        aa = a if a_shape == out_shape or a_shape == "all" else a._coerce_to_general()
        bb = b if b_shape == out_shape or b_shape == "all" else b._coerce_to_general()

        new_cd = {}
        for k, v in aa.coeff_dict.items():
            if not _scalar_is_zero(v):
                new_cd[k] = v
        for k, v in bb.coeff_dict.items():
            if not _scalar_is_zero(v):
                new_cd[k] = new_cd.get(k, 0) + v

        new_cd, eff_shape = self._process_coeffs_dict_new(new_cd, out_shape)
        merged_vs = a._merged_variable_spaces(b)

        if _return_raw:
            return new_cd, eff_shape, merged_vs

        return self.__class__(
            coeff_dict=new_cd,
            data_shape=eff_shape,
            dgcvType=a.dgcvType,
            _simplifyKW=a._simplifyKW,
            variable_spaces=merged_vs,
        )

    def _tp_concat_cd_fast(self, other, shape=None):
        a = self
        b = other
        out = {}

        def _parity_sign(order):
            n = len(order)
            sign = 1
            seen = [False] * n
            for i in range(n):
                if seen[i]:
                    continue
                j = i
                cycle_len = 0
                while not seen[j]:
                    seen[j] = True
                    j = order[j]
                    cycle_len += 1
                if cycle_len and (cycle_len % 2 == 0):
                    sign = -sign
            return sign

        for ka, va in a.coeff_dict.items():
            if va == 0:
                continue
            if ka:
                da = len(ka) // 3
                ia = ka[:da]
                va_bits = ka[da : 2 * da]
                sa = ka[2 * da :]
            else:
                ia = va_bits = sa = tuple()

            for kb, vb in b.coeff_dict.items():
                if vb == 0:
                    continue
                if kb:
                    db = len(kb) // 3
                    ib = kb[:db]
                    vb_bits = kb[db : 2 * db]
                    sb = kb[2 * db :]
                else:
                    ib = vb_bits = sb = tuple()

                inds = ia + ib
                bits = va_bits + vb_bits
                sys = sa + sb
                n = len(inds)

                if n == 0:
                    nk = tuple()
                    out[nk] = out.get(nk, 0) + va * vb
                    continue

                if shape in ("skew", "symmetric"):
                    if shape == "skew":
                        order = sorted(
                            range(n), key=lambda k: (inds[k], bits[k], sys[k])
                        )
                        sign = _parity_sign(order)
                    else:
                        order = sorted(
                            range(n), key=lambda k: (inds[k], bits[k], sys[k])
                        )
                        sign = 1

                    inds2 = tuple(inds[k] for k in order)
                    bits2 = tuple(bits[k] for k in order)
                    sys2 = tuple(sys[k] for k in order)

                    if shape == "skew":
                        seen = set()
                        for t in zip(inds2, bits2, sys2):
                            if t in seen:
                                sign = 0
                                break
                            seen.add(t)
                        if sign == 0:
                            continue

                    nk = inds2 + bits2 + sys2
                    out[nk] = out.get(nk, 0) + sign * va * vb
                    continue

                nk = inds + bits + sys
                out[nk] = out.get(nk, 0) + va * vb

        return out

    def _shape_product(self, other, *, kind: str):
        a = self
        b = other

        if kind == "general":
            aa = a if a.data_shape in ("general", "all") else a._coerce_to_general()
            bb = b if b.data_shape in ("general", "all") else b._coerce_to_general()
            cd = aa._tp_concat_cd_fast(bb)
            cd, eff_shape = self._process_coeffs_dict_new(cd, "general")
            return self.__class__(
                coeff_dict=cd,
                data_shape=eff_shape,
                dgcvType=a.dgcvType,
                _simplifyKW=a._simplifyKW,
                variable_spaces=a._merged_variable_spaces(b),
            )

        if kind == "skew":
            if a.data_shape in ("skew", "all") and b.data_shape in ("skew", "all"):
                cd = a._tp_concat_cd_fast(b, shape="skew")
                cd, eff_shape = self._process_coeffs_dict_new(cd, "skew")
                return self.__class__(
                    coeff_dict=cd,
                    data_shape=eff_shape,
                    dgcvType=a.dgcvType,
                    _simplifyKW=a._simplifyKW,
                    variable_spaces=a._merged_variable_spaces(b),
                )
            ab = a._shape_product(b, kind="general")
            ba = b._shape_product(a, kind="general")
            cd = dict(ab.coeff_dict)
            for k, v in ba.coeff_dict.items():
                if not _scalar_is_zero(v):
                    cd[k] = cd.get(k, 0) - v
            cd, eff_shape = self._process_coeffs_dict_new(cd, "skew")
            return self.__class__(
                coeff_dict=cd,
                data_shape=eff_shape,
                dgcvType=a.dgcvType,
                _simplifyKW=a._simplifyKW,
                variable_spaces=a._merged_variable_spaces(b),
            )

        if kind == "symmetric":
            if a.data_shape in ("symmetric", "all") and b.data_shape in (
                "symmetric",
                "all",
            ):
                cd = a._tp_concat_cd_fast(b)
                cd, eff_shape = self._process_coeffs_dict_new(cd, "symmetric")
                return self.__class__(
                    coeff_dict=cd,
                    data_shape=eff_shape,
                    dgcvType=a.dgcvType,
                    _simplifyKW=a._simplifyKW,
                    variable_spaces=a._merged_variable_spaces(b),
                )

            ab = a._shape_product(b, kind="general")
            ba = b._shape_product(a, kind="general")
            cd = dict(ab.coeff_dict)
            for k, v in ba.coeff_dict.items():
                if not _scalar_is_zero(v):
                    cd[k] = cd.get(k, 0) + v
            cd, eff_shape = self._process_coeffs_dict_new(cd, "symmetric")
            return self.__class__(
                coeff_dict=cd,
                data_shape=eff_shape,
                dgcvType=a.dgcvType,
                _simplifyKW=a._simplifyKW,
                variable_spaces=a._merged_variable_spaces(b),
            )

        raise ValueError(f"Unknown product kind '{kind}'.")

    def tp(self, *others):
        return self.tensor_product(*others)

    def tensor_product(self, *others):
        out = self
        for o in others:
            if check_dgcv_category(o):
                coerce = getattr(o, "as_tensor_field", None)
                if callable(coerce):
                    o = coerce()
            if not isinstance(o, self.__class__):
                return NotImplemented
            out = out._shape_product(o, kind="general")
        return out

    def skew_product(self, *others):
        out = self
        df = query_dgcv_categories(out, {"differential_form"})  # bool
        for o in others:
            if check_dgcv_scalar(o):
                out = o * out
                continue
            if not get_dgcv_category(o) == "tensor_field":
                return NotImplemented

            if not (df and query_dgcv_categories(o, {"differential_form"})):
                if check_dgcv_category(out):
                    coerce = getattr(out, "as_tensor_field", None)
                    if callable(coerce):
                        out = coerce()
                if check_dgcv_category(o):
                    coerce = getattr(o, "as_tensor_field", None)
                    if callable(coerce):
                        o = coerce()
            out = out._shape_product(o, kind="skew")
        return out

    def wedge(self, *others):
        # alias
        return self.skew_product(*others)

    def symmetric_product(self, *others):
        out = self
        for o in others:
            if not get_dgcv_category(o) == "tensor_field":
                return NotImplemented
            coerce = getattr(o, "as_tensor_field", None)
            if callable(coerce):
                o = coerce()
            coerce = getattr(self, "as_tensor_field", None)
            s = coerce() if callable(coerce) else self
            out = s._shape_product(o, kind="symmetric")
        return out

    def __call__(self, *args, strict_left_to_right: bool = False):
        if len(args) == 0:
            return self
        if len(args) > 1:
            contracted = self.__call__(args[0])
            if contracted == 0:
                return 0
            if get_dgcv_category(contracted) == "tensor_field":
                coerce = getattr(contracted, "as_tensor_field", None)
                if callable(coerce):
                    contracted = coerce()
                return contracted(*args[1:])
            first_tf = None
            for idx, arg in enumerate(args[1:]):
                if get_dgcv_category(arg) == "tensor_field":
                    first_tf = arg
                    new_args = args[idx + 2 :]
                    break
                elif check_dgcv_scalar(arg):
                    contracted *= arg
                else:
                    raise TypeError(
                        "tensor_field_class only contracts with dgcv tensor_field classes and scalars, "
                        f"not {type(arg).__name__}."
                    )
            if first_tf:
                tail = first_tf(*new_args) if new_args else first_tf
                return contracted * tail
            return contracted

        other = args[0]
        if check_dgcv_category(other):
            coerce = getattr(other, "as_tensor_field", None)
            if callable(coerce):
                other = coerce()

        if get_dgcv_category(other) == "tensor_field":
            if self._validated_format == "complex":
                if other._validated_format in {"mixed", "real"}:
                    return self.__call__(other._to_complex())
            elif self._validated_format == "real":
                if other._validated_format in {"mixed", "complex"}:
                    return self.__call__(other._to_real())
            elif self._validated_format == "mixed":
                if other._validated_format == "real":
                    return self.__call__(other._to_real())
                if other._validated_format == "complex":
                    return self.__call__(other._to_complex())
                if other._validated_format == "mixed":
                    pref = get_dgcv_settings_registry()["preferred_variable_format"]
                    if pref == "real":
                        return self._to_real().__call__(other._to_real())
                    else:
                        return self._to_complex().__call__(other._to_complex())
            if self.data_shape == "symmetric" or self.data_shape == "skew":
                scale = -1 if self.data_shape == "skew" else 1

                shape_cd = {}
                abort = False

                for k2, v2 in other.coeff_dict.items():
                    if not v2:
                        continue

                    lk2 = len(k2)
                    if lk2 not in (0, 3):
                        abort = True
                        break

                    if lk2 == 0:
                        for k1, v1 in self.coeff_dict.items():
                            if not v1:
                                continue
                            shape_cd[k1] = shape_cd.get(k1, 0) + v1 * v2
                        continue

                    a, b, c = k2
                    for k1, v1 in self.coeff_dict.items():
                        if not v1:
                            continue
                        sign = 1
                        deg = len(k1) // 3
                        for idx in range(deg):
                            idx2, idx3 = deg + idx, 2 * deg + idx
                            if a == k1[idx] and b + k1[idx2] == 1 and c == k1[idx3]:
                                new_key = tuple(
                                    elem
                                    for count, elem in enumerate(k1)
                                    if count not in (idx, idx2, idx3)
                                )
                                shape_cd[new_key] = (
                                    shape_cd.get(new_key, 0) + sign * v1 * v2
                                )
                                break
                            sign *= scale

                if not abort:
                    if not shape_cd:
                        return 0
                    if tuple() in shape_cd and len(shape_cd) == 1:
                        return shape_cd[tuple()]

                    return tensor_field_class(
                        coeff_dict=shape_cd,
                        data_shape=self.data_shape,
                        dgcvType=self.dgcvType,
                        _simplifyKW=self._simplifyKW,
                        variable_spaces=self._merged_variable_spaces(other),
                        parameters=self.parameters
                        | getattr(other, "parameters", set()),
                    )
            if self.data_shape == "general":
                gen_cd = {}
                abort = False

                for k2, v2 in other.coeff_dict.items():
                    if not v2:
                        continue

                    lk2 = len(k2)
                    if lk2 not in (0, 3):
                        abort = True
                        break

                    if lk2 == 0:
                        for k1, v1 in self.coeff_dict.items():
                            if not v1:
                                continue
                            gen_cd[k1] = gen_cd.get(k1, 0) + v1 * v2
                        continue

                    a, b, c = k2
                    for k1, v1 in self.coeff_dict.items():
                        if not v1:
                            continue
                        deg = len(k1) // 3
                        idx, idx2, idx3 = 0, deg, 2 * deg
                        if a == k1[idx] and b + k1[idx2] == 1 and c == k1[idx3]:
                            new_key = tuple(
                                elem
                                for count, elem in enumerate(k1)
                                if count not in (idx, idx2, idx3)
                            )
                            gen_cd[new_key] = gen_cd.get(new_key, 0) + v1 * v2

                if not abort:
                    if not gen_cd:
                        return 0
                    if tuple() in gen_cd and len(gen_cd) == 1:
                        return gen_cd[tuple()]

                    return tensor_field_class(
                        coeff_dict=gen_cd,
                        data_shape="general",
                        dgcvType=self.dgcvType,
                        _simplifyKW=self._simplifyKW,
                        variable_spaces=self._merged_variable_spaces(other),
                        parameters=self.parameters
                        | getattr(other, "parameters", set()),
                    )

        if get_dgcv_category(other) != "tensor_field":
            raise TypeError(
                "tensor_field_class.__call__ only supports contraction against tensor_field instances."
            )

        shape_a = getattr(self, "data_shape", "general")
        shape_b = getattr(other, "data_shape", "general")
        if shape_a != shape_b and shape_a != "all" and shape_b != "all":
            ###!!! optimize later: shape-aware contraction can avoid general expansion by sorting argument keys
            self = self._coerce_to_general()
            other = other._coerce_to_general()

        def _split_tripled(k):
            d = len(k) // 3
            return k[:d], k[d : 2 * d], k[2 * d :]

        def _join_tripled(idxs, valence_tuple, syslbls):
            return tuple(idxs) + tuple(valence_tuple) + tuple(syslbls)

        def _complementary(vb1, vb2):
            for a, b in zip(vb1, vb2):
                if a + b != 1:
                    return False
            return True

        vst = self.variable_spaces_types

        def _profile_from_parts(idxs, syslbls):
            out = []
            for idx, sys in zip(idxs, syslbls):
                sys_data = vst.get(sys)
                if sys_data is None or sys_data.get("type") != "complex":
                    out.append("s")
                    continue
                b0, b1, b2 = sys_data["breaks"]
                if idx < b1:
                    out.append("c")
                elif idx < b2:
                    out.append("r")
                else:
                    out.append("s")
            return tuple(out)

        def _expand_to_profile(k, v, want_profile):
            have = self._profile_key(k)
            have0 = tuple(t[0] for t in have)
            want0 = tuple(t[0] for t in want_profile)

            for a0, b0 in zip(have0, want0):
                if (a0 == "s") != (b0 == "s"):
                    return None

            swap_slots = [i for i, (a0, b0) in enumerate(zip(have0, want0)) if a0 != b0]
            if not swap_slots:
                return {k: v}

            flip = getattr(self, "_flip_format", None)
            if not callable(flip):
                return None

            terms = {k: v}
            for slot in swap_slots:
                kind0 = want0[slot]
                if kind0 not in ("r", "c"):
                    return None

                new_terms = {}
                for kk, vv in terms.items():
                    out = flip(slot, kk, vv, to_kind=kind0)
                    if not out:
                        continue
                    for nk, nv in out.items():
                        if nv:
                            new_terms[nk] = new_terms.get(nk, 0) + nv

                terms = new_terms
                if not terms:
                    return None

            return terms

        def _pick_by_idxs(terms, want_idxs):
            out = {}
            for k, v in terms.items():
                if not v:
                    continue
                idxs, _, _ = _split_tripled(k)
                if idxs == want_idxs:
                    out[k] = out.get(k, 0) + v
            return out or None

        new_cd = {}

        for k1, v1 in self.coeff_dict.items():
            if not v1:
                continue

            i1, val1, s1 = _split_tripled(k1)
            d1 = len(i1)

            for k2, v2 in other.coeff_dict.items():
                if not v2:
                    continue

                i2, val2, s2 = _split_tripled(k2)
                d2 = len(i2)

                strict_left_to_right = (
                    True  # hard override -- releasing this has not been decided
                )
                if d2 > d1:
                    if strict_left_to_right:
                        continue
                    lead_i2, tail_i2 = i2[:d1], i2[d1:]
                    lead_val2, tail_val2 = val2[:d1], val2[d1:]
                    lead_s2, tail_s2 = s2[:d1], s2[d1:]

                    if lead_s2 != s1:
                        continue
                    if not _complementary(val1, lead_val2):
                        continue

                    want_profile = _profile_from_parts(lead_i2, lead_s2)

                    terms = _expand_to_profile(k1, v1, want_profile)
                    if not terms:
                        continue
                    terms = _pick_by_idxs(terms, lead_i2)
                    if not terms:
                        continue

                    nk = _join_tripled(tail_i2, tail_val2, tail_s2)
                    acc = new_cd.get(nk, 0)
                    for v1a in terms.values():
                        acc += v1a * v2
                    if acc:
                        new_cd[nk] = acc
                    continue

                lead_i1, tail_i1 = i1[:d2], i1[d2:]
                lead_val1, tail_val1 = val1[:d2], val1[d2:]
                lead_s1, tail_s1 = s1[:d2], s1[d2:]

                if lead_s1 != s2:
                    continue
                if not _complementary(lead_val1, val2):
                    continue

                want_profile = _profile_from_parts(lead_i1, lead_s1)

                terms = _expand_to_profile(k2, v2, want_profile)
                if not terms:
                    continue
                terms = _pick_by_idxs(terms, lead_i1)
                if not terms:
                    continue

                nk = _join_tripled(tail_i1, tail_val1, tail_s1)
                acc = new_cd.get(nk, 0)
                for v2a in terms.values():
                    acc += v1 * v2a
                if acc:
                    new_cd[nk] = acc

        if not new_cd:
            return 0

        if tuple() in new_cd and len(new_cd) == 1:
            return new_cd[tuple()]

        return tensor_field_class(
            coeff_dict=new_cd,
            data_shape="general",
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=self._merged_variable_spaces(other),
        )

    def _eval_simplify(self, **kwargs):
        rule = self._simplifyKW.get("simplify_rule", None)
        ign = self._simplifyKW.get("simplify_ignore_list", None)

        if rule is None:
            simplified = {k: simplify(v, **kwargs) for k, v in self.coeff_dict.items()}
        elif rule == "holomorphic":
            simplified = {
                k: simplify(allToHol(v, skipVar=ign), **kwargs)
                for k, v in self.coeff_dict.items()
            }
        elif rule == "real":
            simplified = {
                k: simplify(allToReal(v, skipVar=ign), **kwargs)
                for k, v in self.coeff_dict.items()
            }
        elif rule == "symbolic_conjugate":
            simplified = {
                k: simplify(allToSym(v, skipVar=ign), **kwargs)
                for k, v in self.coeff_dict.items()
            }
        else:
            warnings.warn(f"Unsupported simplify_rule: {rule}.")
            simplified = {k: simplify(v, **kwargs) for k, v in self.coeff_dict.items()}

        return self.__class__(
            coeff_dict=simplified,
            data_shape=self.data_shape,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=self._variable_spaces,
        )

    def __dgcv_simplify__(self, method=None, **kwargs):
        return self._eval_simplify(**kwargs)

    @property
    def __dgcv_zero_obstr__(self):
        return self.coeff_dict.values(), self.coeff_free_symbols

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if (
            self.varSpace != other.varSpace
            or self.valence != other.valence
            or self.dgcvType != other.dgcvType
        ):
            return False

        if self.data_shape != other.data_shape:
            a = self._expand_special_to_general(self.coeff_dict, self.data_shape)
            b = self._expand_special_to_general(other.coeff_dict, other.data_shape)
            keys = set(a.keys()).union(b.keys())
            for k in keys:
                if simplify(allToReal(a.get(k, 0))) != simplify(allToReal(b.get(k, 0))):
                    return False
            return True

        keys = set(self.coeff_dict.keys()).union(other.coeff_dict.keys())
        for k in keys:
            if simplify(allToReal(self.coeff_dict.get(k, 0))) != simplify(
                allToReal(other.coeff_dict.get(k, 0))
            ):
                return False
        return True

    def __hash__(self):
        g = self._expand_special_to_general(self.coeff_dict, self.data_shape)
        items = tuple(sorted((k, simplify(allToReal(v))) for k, v in g.items()))
        return hash(
            (self.varSpace, self.valence, self.data_shape, self.dgcvType, items)
        )

    @property
    def is_zero(self) -> bool:
        cd = getattr(self, "coeff_dict", None)
        if not isinstance(cd, dict) or not cd:
            return True

        if self._is_scalar_coeff_dict(cd):
            return cd.get(tuple(), 0) == 0

        for v in cd.values():
            if not _scalar_is_zero(v):
                return False
        return True

    def subs(self, substitutions):
        substituted = {k: subs(v, substitutions) for k, v in self.coeff_dict.items()}
        return self.__class__(
            coeff_dict=substituted,
            data_shape=self.data_shape,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=self._variable_spaces,
        )

    def _coordinate_format_info(self) -> dict:
        cached = getattr(self, "_coordinate_format", None)
        if cached is not None:
            return cached

        vst = self.variable_spaces_types

        saw_standard = False
        saw_complex = False

        saw_holo = False
        saw_anti = False
        saw_real = False
        saw_imag = False

        for k in self.coeff_dict:
            d = len(k) // 3
            if d == 0:
                continue

            idxs = k[:d]
            systems = k[2 * d :]

            for idx, sys in zip(idxs, systems):
                sys_data = vst[sys]
                if sys_data["type"] != "complex":
                    saw_standard = True
                    continue

                saw_complex = True
                b0, b1, b2 = sys_data["breaks"]

                if idx < b0:
                    saw_holo = True
                elif idx < b1:
                    saw_anti = True
                elif idx < b2:
                    saw_real = True
                else:
                    saw_imag = True

        if not saw_complex:
            out = {"dgcv_type": "standard", "sub_type": "standard", "role": "standard"}
            self._coordinate_format = out
            return out

        if saw_standard:
            out = {"dgcv_type": "mixed", "sub_type": "mixed", "role": "mixed"}
            self._coordinate_format = out
            return out

        has_complex_block = saw_holo or saw_anti
        has_real_block = saw_real or saw_imag

        if has_complex_block and not has_real_block:
            role = "mixed"
            if saw_holo and not saw_anti:
                role = "holo"
            elif saw_anti and not saw_holo:
                role = "anti"
            out = {"dgcv_type": "complex", "sub_type": "complex", "role": role}
            self._coordinate_format = out
            return out

        if has_real_block and not has_complex_block:
            role = "mixed"
            if saw_real and not saw_imag:
                role = "real"
            elif saw_imag and not saw_real:
                role = "imag"
            out = {"dgcv_type": "complex", "sub_type": "real", "role": role}
            self._coordinate_format = out
            return out

        out = {"dgcv_type": "complex", "sub_type": "mixed", "role": "mixed"}
        self._coordinate_format = out
        return out


class vector_field_class(tensor_field_class):
    _dgcv_categories = {"vector_field"}

    def __init__(
        self,
        varSpace=None,
        coeffs=None,
        *,
        coeff_dict=None,
        valence=None,
        data_shape: str = "all",
        dgcvType: str = "standard",
        _simplifyKW=None,
        variable_spaces=None,
        parameters=set(),
        _inheritance=None,
    ):
        self._vf_cache = None
        if _simplifyKW is None:
            _simplifyKW = {
                "simplify_rule": None,
                "simplify_ignore_list": None,
                "preferred_basis_element": None,
            }
        self.parameters = parameters
        if coeff_dict is not None:
            if varSpace is not None or coeffs is not None:
                raise TypeError(
                    "Provide either `coeff_dict=...` or (`varSpace`, `coeffs`), not both."
                )

            super().__init__(
                coeff_dict=coeff_dict,
                data_shape=data_shape,
                dgcvType=dgcvType,
                _simplifyKW=_simplifyKW,
                variable_spaces=variable_spaces,
                _inheritance=_inheritance,
            )

            if self.valence != (1,) and self.valence != tuple():
                raise ValueError(
                    f"vector_field expects valence=(1,), not {self.valence}"
                )

            self._vf_cache = None
            return

        if varSpace is None or coeffs is None:
            raise TypeError(
                "Provide either `coeff_dict=...` or (`varSpace`, `coeffs`)."
            )

        vs = tuple(varSpace)
        cs = list(coeffs)

        if len(vs) != len(cs):
            raise ValueError("`varSpace` and `coeffs` must have the same length.")
        if len(vs) != len(set(vs)):
            raise TypeError("`varSpace` must not contain repeated variables.")

        syslbl = None
        if len(vs) > 0:
            if isinstance(variable_spaces, dict) and variable_spaces:
                if len(variable_spaces) == 1:
                    syslbl = next(iter(variable_spaces.keys()))
                else:
                    raise ValueError(
                        "vector_field legacy init requires a single system in `variable_spaces`."
                    )
            else:
                info0 = vmf_lookup(vs[0], path=True, relatives=False)
                p0 = info0.get("path")
                if not (isinstance(p0, tuple) and len(p0) >= 2):
                    raise KeyError(
                        "vector_field legacy init requires variables registered in the VMF or `variable_spaces={...}`."
                    )
                syslbl = p0[1]

        if syslbl is None:
            syslbl = "__anon__"

        cd = {(i, 1, syslbl): c for i, c in enumerate(cs) if not _scalar_is_zero(c)}
        if not cd:
            cd = {tuple(): 0}

        if variable_spaces is None:
            variable_spaces = {syslbl: vs}
        else:
            variable_spaces = dict(variable_spaces)
            variable_spaces.setdefault(syslbl, vs)
        super().__init__(
            coeff_dict=cd,
            data_shape=data_shape,
            dgcvType=dgcvType,
            _simplifyKW=_simplifyKW,
            variable_spaces=variable_spaces,
            _inheritance=_inheritance,
        )

    def _vf_view(self):
        cache = getattr(self, "_vf_cache", None)
        if cache is not None:
            return cache
        if self._is_scalar_coeff_dict(self.coeff_dict):
            self._vf_cache = (tuple(), [])
            return self._vf_cache

        syslbl = None
        n = 0
        items = []

        for k, v in self.coeff_dict.items():
            if _scalar_is_zero(v) or not isinstance(k, tuple) or len(k) != 3:
                continue
            i, vb, s = k
            if vb != 1:
                raise ValueError(
                    "vector_field expects contravariant bit 1 in coeff_dict keys."
                )
            if syslbl is None:
                syslbl = s
            elif syslbl != s:
                raise ValueError(
                    "vector_field view requires a single system label in coeff_dict."
                )
            if isinstance(i, Integral):
                ii = int(i)
                if ii >= n:
                    n = ii + 1
                items.append((ii, v))
            else:
                raise TypeError(
                    "vector_field expects integer indices in coeff_dict keys."
                )

        if syslbl is None:
            self._vf_cache = (tuple(), [])
            return self._vf_cache

        vs = self._variable_spaces.get(syslbl)
        if not isinstance(vs, tuple):
            raise KeyError(f"vector_field missing `variable_spaces[{syslbl!r}]`.")

        coeffs = [0] * max(n, len(vs))
        for ii, v in items:
            if ii >= len(coeffs):
                coeffs.extend([0] * (ii + 1 - len(coeffs)))
            coeffs[ii] = coeffs[ii] + v

        self._vf_cache = (vs, coeffs)
        return self._vf_cache

    def as_tensor_field(self, data_shape: Optional[str] = None) -> tensor_field_class:
        vs = getattr(self, "_variable_spaces", None)
        if not isinstance(vs, dict):
            vs = None
        data_shape = data_shape if data_shape else self.data_shape
        return tensor_field_class(
            coeff_dict=self.coeff_dict,
            data_shape=data_shape,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=vs,
            parameters=self.parameters,
        )

    @property
    def coeffs(self):
        return self._vf_view()[1]

    def simplify_format(self, format_type=None, skipVar=None):
        if format_type not in {None, "holomorphic", "real", "symbolic_conjugate"}:
            warnings.warn(
                "simplify_format() received an unsupported first argument. Try None, 'holomorphic', 'real', or 'symbolic_conjugate'."
            )
        return self.__class__(
            coeff_dict=self.coeff_dict,
            dgcvType=self.dgcvType,
            _simplifyKW={"simplify_rule": format_type, "simplify_ignore_list": skipVar},
            variable_spaces=self._variable_spaces,
        )

    def _eval_simplify(self, **kwargs):
        rule = self._simplifyKW.get("simplify_rule", None)
        ign = self._simplifyKW.get("simplify_ignore_list", None)

        if rule is None:

            def f(c):
                return simplify(c, **kwargs)
        elif rule == "holomorphic":

            def f(c):
                return simplify(allToHol(c, skipVar=ign), **kwargs)
        elif rule == "real":

            def f(c):
                return simplify(allToReal(c, skipVar=ign), **kwargs)
        elif rule == "symbolic_conjugate":

            def f(c):
                return simplify(allToSym(c, skipVar=ign), **kwargs)
        else:

            def f(c):
                return simplify(c, **kwargs)

        cd = {}
        for k, v in self.coeff_dict.items():
            if _scalar_is_zero(v):
                continue
            nv = f(v)
            if not _scalar_is_zero(nv):
                cd[k] = cd.get(k, 0) + nv

        if not cd:
            cd = {tuple(): 0}

        return self.__class__(
            coeff_dict=cd,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=self._variable_spaces,
            data_shape=self.data_shape,
        )

    def subs(self, subsData):
        cd = {}
        for k, v in self.coeff_dict.items():
            if _scalar_is_zero(v):
                continue
            nv = subs(v, subsData)
            if not _scalar_is_zero(nv):
                cd[k] = cd.get(k, 0) + nv

        if not cd:
            cd = {tuple(): 0}

        return self.__class__(
            coeff_dict=cd,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=self._variable_spaces,
            data_shape=self.data_shape,
        )

    def __neg__(self):
        cd = {k: -v for k, v in self.coeff_dict.items() if not _scalar_is_zero(v)}
        if not cd:
            cd = {tuple(): 0}
        return self.__class__(
            coeff_dict=cd,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=getattr(self, "_variable_spaces", None),
        )

    def _as_tensor_maybe(self, obj):
        if isinstance(obj, tensor_field_class):
            return obj
        if check_dgcv_category(obj):
            coerce = getattr(obj, "as_tensor_field", None)
            if callable(coerce):
                return coerce()
        return None

    def __add__(self, other):
        if _scalar_is_zero(other):
            return self

        if isinstance(other, self.__class__):
            vs = self._merged_variable_spaces(other)
            new_cd = dict(self.coeff_dict)
            for k, v in other.coeff_dict.items():
                if not _scalar_is_zero(v):
                    new_cd[k] = new_cd.get(k, 0) + v
            new_cd = {k: v for k, v in new_cd.items() if not _scalar_is_zero(v)} or {
                tuple(): 0
            }
            return self.__class__(
                coeff_dict=new_cd,
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=vs,
                data_shape=self.data_shape,
            )

        if check_dgcv_scalar(other):
            return self.as_tensor_field().__add__(other)

        tf = self._as_tensor_maybe(other)
        if tf is not None:
            return self.as_tensor_field().__add__(tf)

        return NotImplemented

    def __sub__(self, other):
        if _scalar_is_zero(other):
            return self

        if isinstance(other, self.__class__):
            vs = self._merged_variable_spaces(other)
            new_cd = dict(self.coeff_dict)
            for k, v in other.coeff_dict.items():
                if not _scalar_is_zero(v):
                    new_cd[k] = new_cd.get(k, 0) - v
            new_cd = {k: v for k, v in new_cd.items() if not _scalar_is_zero(v)} or {
                tuple(): 0
            }
            return self.__class__(
                coeff_dict=new_cd,
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=vs,
                data_shape=self.data_shape,
            )

        if check_dgcv_scalar(other):
            return self.as_tensor_field().__sub__(other)

        tf = self._as_tensor_maybe(other)
        if tf is not None:
            return self.as_tensor_field().__sub__(tf)

        return NotImplemented

    def __radd__(self, other):
        if _scalar_is_zero(other):
            return self
        return self.__add__(other)

    def __rsub__(self, other):
        if check_dgcv_scalar(other):
            return self.as_tensor_field().__rsub__(other)

        tf = self._as_tensor_maybe(other)
        if tf is not None:
            return tf.__sub__(self.as_tensor_field())

        return NotImplemented

    def __mul__(self, other):
        if check_dgcv_scalar(other):
            return super().__mul__(other)

        tf = other
        if check_dgcv_category(tf):
            coerce = getattr(tf, "as_tensor_field", None)
            if callable(coerce):
                tf = coerce()

        if not (get_dgcv_category(tf) == "tensor_field"):
            return NotImplemented

        return self.as_tensor_field(data_shape="skew")._shape_product(
            tf,
            kind="skew",
        )

    def __matmul__(self, other):
        tf = other
        if check_dgcv_category(tf):
            coerce = getattr(tf, "as_tensor_field", None)
            if callable(coerce):
                tf = coerce()

        if not (get_dgcv_category(tf) == "tensor_field"):
            return NotImplemented

        return self.as_tensor_field(data_shape="all")._shape_product(
            tf,
            kind="general",
        )

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __call__(self, *args, ignore_complex_handling=None):
        if len(args) != 1:
            raise ValueError("vector_field expects exactly one argument.")
        other = args[0]

        if get_dgcv_category(other) == "array":
            return other.apply(self.__call__)

        if get_dgcv_category(other) == "tensor_field":
            if self._validated_format == "complex":
                if other._validated_format in {"mixed", "real"}:
                    return self.__call__(other._to_complex())
            elif self._validated_format == "real":
                if other._validated_format in {"mixed", "complex"}:
                    return self.__call__(other._to_real())
            elif self._validated_format == "mixed":
                if other._validated_format == "real":
                    return self.__call__(other._to_real())
                if other._validated_format == "complex":
                    return self.__call__(other._to_complex())
                if other._validated_format == "mixed":
                    pref = get_dgcv_settings_registry()["preferred_variable_format"]
                    if pref == "real":
                        return self._to_real().__call__(other._to_real())
                    else:
                        return self._to_complex().__call__(other._to_complex())
            if other._is_scalar():
                key, val = next(iter(other.coeff_dict.items()))
                return self(val)
            return super().__call__(other)

        if (
            get_dgcv_category(other) == "differential_form"
            and getattr(other, "degree", None) == 0
        ):
            c0 = getattr(other, "coeffsInKFormBasis", None)
            if isinstance(c0, (list, tuple)) and c0:
                other = c0[0]

        diff_local = diff
        half = rational(1, 2)
        imu = imag_unit()
        mIhalf = -imu * half

        fmt = self._coordinate_format_info()

        if ignore_complex_handling or fmt.get("dgcv_type") == "standard":
            out = 0
            for k, c in self.coeff_dict.items():
                if not c:
                    continue
                d = len(k) // 3
                if d != 1:
                    continue
                idx = k[0]
                sys = k[2]
                vs = self._variable_spaces.get(sys)
                if not isinstance(vs, tuple):
                    continue
                v = vs[idx]
                out += c * diff_local(other, v)
            return out

        has_conj = not verify_conjugate_re_im_free(other)
        a = allToSym(other) if has_conj else other

        out = 0

        for k, c in self.coeff_dict.items():
            if not c:
                continue

            d = len(k) // 3
            if d != 1:
                continue

            idx = k[0]
            sys = k[2]

            vs = self._variable_spaces.get(sys)
            if not isinstance(vs, tuple):
                continue

            v = vs[idx]

            info = vmf_lookup(v, flattened_relatives=True)
            st = info.get("sub_type")
            rel = info.get("flattened_relatives")

            if st is None or rel is None or not isinstance(rel, tuple) or len(rel) != 4:
                out += c * diff_local(a, v)
                continue

            z, zb, x, y = rel

            if has_conj:
                if st == "holo":
                    out += c * diff_local(a, z)
                elif st == "anti":
                    out += c * diff_local(a, zb)
                elif st == "real":
                    out += c * half * (diff_local(a, z) + diff_local(a, zb))
                elif st == "imag":
                    out += c * mIhalf * (diff_local(a, z) - diff_local(a, zb))
                else:
                    out += c * diff_local(a, v)
                continue

            if st == "holo":
                out += c * (
                    diff_local(a, z)
                    + half * (diff_local(a, x) - imu * diff_local(a, y))
                )
            elif st == "anti":
                out += c * (
                    diff_local(a, zb)
                    + half * (diff_local(a, x) + imu * diff_local(a, y))
                )
            elif st == "real":
                out += c * (diff_local(a, x) + diff_local(a, z) + diff_local(a, zb))
            elif st == "imag":
                out += c * (
                    diff_local(a, y) + imu * (diff_local(a, z) - diff_local(a, zb))
                )
            else:
                out += c * diff_local(a, v)

        return out

    def tensor_product(self, *others, coerce_shapes: bool = False):
        return self.as_tensor_field().tensor_product(*others)


class differential_form_class(tensor_field_class):
    _dgcv_categories = {"differential_form"}

    def __init__(
        self,
        varSpace=None,
        data_dict=None,
        degree=None,
        *,
        coeff_dict=None,
        valence=None,
        data_shape: str = "all",
        dgcvType: str = "standard",
        _simplifyKW=None,
        variable_spaces=None,
        parameters=set(),
        _inheritance=None,
    ):
        if _simplifyKW is None:
            _simplifyKW = {
                "simplify_rule": None,
                "simplify_ignore_list": None,
                "preferred_basis_element": None,
            }

        self.parameters = parameters
        if coeff_dict is not None:
            if varSpace is not None or data_dict is not None or degree is not None:
                raise TypeError(
                    "Provide either `coeff_dict=...` or (`varSpace`, `data_dict`, `degree`), not both."
                )

            if data_shape == "all":
                max_deg = 0
                all_covariant = True
                for k, c in coeff_dict.items():
                    if not c:
                        continue
                    d = len(k) // 3
                    if d > max_deg:
                        max_deg = d
                    valence_tuple = k[d : 2 * d]
                    for vb in valence_tuple:
                        if vb != 0:
                            all_covariant = False
                            break
                    if not all_covariant:
                        break
                if all_covariant and max_deg > 1:
                    data_shape = "skew"

            super().__init__(
                coeff_dict=coeff_dict,
                data_shape=data_shape,
                dgcvType=dgcvType,
                _simplifyKW=_simplifyKW,
                variable_spaces=variable_spaces,
                _inheritance=_inheritance,
            )
            return

        if varSpace is None or data_dict is None or degree is None:
            raise TypeError(
                "Provide either `coeff_dict=...` or (`varSpace`, `data_dict`, `degree`)."
            )

        vs = tuple(varSpace)
        if len(vs) != len(set(vs)):
            raise ValueError("`varSpace` must not have duplicate entries.")

        syslbl = None
        if len(vs) > 0:
            if isinstance(variable_spaces, dict) and variable_spaces:
                if len(variable_spaces) == 1:
                    syslbl = next(iter(variable_spaces.keys()))
                else:
                    raise ValueError(
                        "differential_form legacy init requires a single system in `variable_spaces`."
                    )
            else:
                info0 = vmf_lookup(vs[0], path=True, relatives=False)
                p0 = info0.get("path")
                if not (isinstance(p0, tuple) and len(p0) >= 2):
                    raise KeyError(
                        "differential_form legacy init requires variables registered in the VMF or `variable_spaces={...}`."
                    )
                syslbl = p0[1]

        if syslbl is None:
            syslbl = "__anon__"

        if variable_spaces is None:
            variable_spaces = {syslbl: vs}
        else:
            variable_spaces = dict(variable_spaces)
            variable_spaces.setdefault(syslbl, vs)

        vtuple = variable_spaces.get(syslbl)
        idx_map = {v: i for i, v in enumerate(vtuple)}

        deg = int(degree)
        if data_shape == "all" and deg > 1:
            data_shape = "skew"

        zeros = (0,) * deg
        syslbls = (syslbl,) * deg

        cd = {}
        for k, c in data_dict.items():
            if not c:
                continue

            if deg == 0:
                if k == tuple() or k == 0:
                    cd[tuple()] = cd.get(tuple(), 0) + c
                    continue
                raise TypeError("degree=0 forms require scalar key tuple().")

            idxs = []
            for jj in k:
                var = vs[int(jj)]
                idxs.append(idx_map[var])

            nk = tuple(idxs) + zeros + syslbls
            cd[nk] = cd.get(nk, 0) + c

        if not cd:
            cd = {tuple(): 0}

        super().__init__(
            coeff_dict=cd,
            data_shape=data_shape,
            dgcvType=dgcvType,
            _simplifyKW=_simplifyKW,
            variable_spaces=variable_spaces,
            _inheritance=_inheritance,
        )

    @property
    def degree(self):
        return self._df_degree

    def simplify_format(self, format_type=None, skipVar=None):
        if format_type not in {None, "holomorphic", "real", "symbolic_conjugate"}:
            warnings.warn(
                "simplify_format() received an unsupported first argument. Try None, 'holomorphic', 'real', or 'symbolic_conjugate'."
            )
        return self.__class__(
            coeff_dict=self.coeff_dict,
            dgcvType=self.dgcvType,
            _simplifyKW={"simplify_rule": format_type, "simplify_ignore_list": skipVar},
            variable_spaces=getattr(self, "_variable_spaces", None),
        )

    def _eval_simplify(self, **kwargs):
        rule = self._simplifyKW.get("simplify_rule", None)
        ign = self._simplifyKW.get("simplify_ignore_list", None)

        if self._is_scalar():
            return self.__class__(
                coeff_dict={tuple(): simplify(self._scalar_value(), **kwargs)},
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=getattr(self, "_variable_spaces", None),
            )

        if rule is None:
            new_cd = {k: simplify(v, **kwargs) for k, v in self.coeff_dict.items()}
        elif rule == "holomorphic":
            new_cd = {
                k: simplify(allToHol(v, skipVar=ign), **kwargs)
                for k, v in self.coeff_dict.items()
            }
        elif rule == "real":
            new_cd = {
                k: simplify(allToReal(v, skipVar=ign), **kwargs)
                for k, v in self.coeff_dict.items()
            }
        elif rule == "symbolic_conjugate":
            new_cd = {
                k: simplify(allToSym(v, skipVar=ign), **kwargs)
                for k, v in self.coeff_dict.items()
            }
        else:
            warnings.warn(f"Unsupported simplify_rule: {rule}.")
            new_cd = {k: simplify(v, **kwargs) for k, v in self.coeff_dict.items()}

        new_cd, _ = self._process_coeffs_dict_new(new_cd, "all")
        return self.__class__(
            coeff_dict=new_cd,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=getattr(self, "_variable_spaces", None),
        )

    def subs(self, subsData):
        new_cd = {k: subs(v, subsData) for k, v in self.coeff_dict.items()}
        new_cd, _ = self._process_coeffs_dict_new(new_cd, "all")
        return self.__class__(
            coeff_dict=new_cd,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=getattr(self, "_variable_spaces", None),
        )

    def as_tensor_field(self, data_shape: Optional[str] = None) -> tensor_field_class:
        vs = getattr(self, "_variable_spaces", None)
        if not isinstance(vs, dict):
            vs = None
        return tensor_field_class(
            coeff_dict=self.coeff_dict,
            data_shape=(data_shape if data_shape is not None else self.data_shape),
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=vs,
            parameters=self.parameters,
        )

    def _as_tensor_maybe(self, obj):
        if isinstance(obj, tensor_field_class):
            return obj
        if check_dgcv_category(obj):
            coerce = getattr(obj, "as_tensor_field", None)
            if callable(coerce):
                return coerce()
        return None

    def __add__(self, other):
        if _scalar_is_zero(other):
            return self

        if check_dgcv_scalar(other):
            new_cd = dict(self.coeff_dict)
            new_cd[tuple()] = new_cd.get(tuple(), 0) + other
            new_cd = {k: v for k, v in new_cd.items() if not _scalar_is_zero(v)} or {
                tuple(): 0
            }
            return self.__class__(
                coeff_dict=new_cd,
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=getattr(self, "_variable_spaces", None),
                data_shape=self.data_shape,
            )

        if isinstance(other, self.__class__):
            vs = self._merged_variable_spaces(other)
            new_cd = dict(self.coeff_dict)
            for k, v in other.coeff_dict.items():
                if not _scalar_is_zero(v):
                    new_cd[k] = new_cd.get(k, 0) + v
            new_cd = {k: v for k, v in new_cd.items() if not _scalar_is_zero(v)} or {
                tuple(): 0
            }
            return self.__class__(
                coeff_dict=new_cd,
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=vs,
                data_shape=self.data_shape,
            )

        tf = self._as_tensor_maybe(other)
        if tf is not None:
            return self.as_tensor_field().__add__(tf)

        return NotImplemented

    def __sub__(self, other):
        if _scalar_is_zero(other):
            return self

        if check_dgcv_scalar(other):
            new_cd = dict(self.coeff_dict)
            new_cd[tuple()] = new_cd.get(tuple(), 0) - other
            new_cd = {k: v for k, v in new_cd.items() if not _scalar_is_zero(v)} or {
                tuple(): 0
            }
            return self.__class__(
                coeff_dict=new_cd,
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=getattr(self, "_variable_spaces", None),
                data_shape=self.data_shape,
            )

        if isinstance(other, self.__class__):
            vs = self._merged_variable_spaces(other)
            new_cd = dict(self.coeff_dict)
            for k, v in other.coeff_dict.items():
                if not _scalar_is_zero(v):
                    new_cd[k] = new_cd.get(k, 0) - v
            new_cd = {k: v for k, v in new_cd.items() if not _scalar_is_zero(v)} or {
                tuple(): 0
            }
            return self.__class__(
                coeff_dict=new_cd,
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=vs,
                data_shape=self.data_shape,
            )

        tf = self._as_tensor_maybe(other)
        if tf is not None:
            return self.as_tensor_field().__sub__(tf)

        return NotImplemented

    def __radd__(self, other):
        if _scalar_is_zero(other):
            return self
        return self.__add__(other)

    def __rsub__(self, other):
        if check_dgcv_scalar(other):
            new_cd = {
                k: (-v) for k, v in self.coeff_dict.items() if not _scalar_is_zero(v)
            }
            new_cd[tuple()] = new_cd.get(tuple(), 0) + other
            new_cd = {k: v for k, v in new_cd.items() if not _scalar_is_zero(v)} or {
                tuple(): 0
            }
            return self.__class__(
                coeff_dict=new_cd,
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=getattr(self, "_variable_spaces", None),
                data_shape=self.data_shape,
            )

        tf = self._as_tensor_maybe(other)
        if tf is not None:
            return tf.__sub__(self.as_tensor_field())

        return NotImplemented

    def __mul__(self, other):
        if check_dgcv_scalar(other):
            cd = {
                k: other * v
                for k, v in self.coeff_dict.items()
                if not _scalar_is_zero(v)
            }
            if not cd:
                cd = {tuple(): 0}
            return self.__class__(
                coeff_dict=cd,
                data_shape=self.data_shape,
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=self._variable_spaces,
                parameters=getattr(self, "parameters", set()),
            )

        tf = self._as_tensor_maybe(other)
        if tf is None:
            return NotImplemented

        if query_dgcv_categories(tf, {"differential_form"}):
            cd = self._tp_concat_cd_fast(tf, shape="skew")
            return self.__class__(
                coeff_dict=cd,
                data_shape="skew",
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=self._merged_variable_spaces(tf),
                parameters=getattr(self, "parameters", set()),
            )

        if get_dgcv_category(tf) == "tensor_field":
            if tf.data_shape in {"skew", "all"}:
                return self.as_tensor_field().wedge(tf)
            return NotImplemented

        return NotImplemented

    def __matmul__(self, other):
        tf = self._as_tensor_maybe(other)
        if tf is None:
            return NotImplemented
        return self.as_tensor_field()._shape_product(
            tf.as_tensor_field() if hasattr(tf, "as_tensor_field") else tf,
            kind="general",
        )

    def tensor_product(self, *others, coerce_shapes: bool = False):
        return self.as_tensor_field().tensor_product(*others)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __call__(self, *args, **kwargs):
        if args and all(query_dgcv_categories(a, {"vector_field"}) for a in args):
            if len(args) > 1:
                contract = self(args[0])
                if query_dgcv_categories(contract, "differential_form"):
                    return contract(*args[1:])
                else:
                    return 0

            other = args[0]
            if self._validated_format == "complex":
                if other._validated_format in {"mixed", "real"}:
                    return self.__call__(other._to_complex())
            elif self._validated_format == "real":
                if other._validated_format in {"mixed", "complex"}:
                    return self.__call__(other._to_real())
            elif self._validated_format == "mixed":
                if other._validated_format == "real":
                    return self.__call__(other._to_real())
                if other._validated_format == "complex":
                    return self.__call__(other._to_complex())
                if other._validated_format == "mixed":
                    pref = get_dgcv_settings_registry()["preferred_variable_format"]
                    if pref == "real":
                        return self._to_real().__call__(other._to_real())
                    else:
                        return self._to_complex().__call__(other._to_complex())

            shape_cd = {}

            for k2, v2 in other.coeff_dict.items():
                if not v2:
                    continue

                a, b, c = k2
                for k1, v1 in self.coeff_dict.items():
                    if not v1:
                        continue
                    sign = 1
                    deg = len(k1) // 3
                    for idx in range(deg):
                        idx2, idx3 = deg + idx, 2 * deg + idx
                        if a == k1[idx] and b + k1[idx2] == 1 and c == k1[idx3]:
                            new_key = tuple(
                                elem
                                for count, elem in enumerate(k1)
                                if count not in (idx, idx2, idx3)
                            )
                            shape_cd[new_key] = (
                                shape_cd.get(new_key, 0) + sign * v1 * v2
                            )
                            break
                        sign *= -1

            if not shape_cd:
                return 0
            if tuple() in shape_cd and len(shape_cd) == 1:
                return shape_cd[tuple()]

            return differential_form_class(
                coeff_dict=shape_cd,
                data_shape=self.data_shape,
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=self._merged_variable_spaces(other),
                parameters=self.parameters | getattr(other, "parameters", set()),
            )

        return super().__call__(*args, **kwargs)


def assemble_tensor_field(
    coordinate_space: tuple | list,
    coefficient_dict: dict,
    valence: Optional[tuple | list] = None,
    shape: Optional[Literal["general", "symmetric", "skew"]] = "general",
    subclass: Optional[Literal["vector field", "differential form"]] = None,
) -> tensor_field_class:
    """
    Assemble a dgcv tensor field from coordinate-space indices and coefficients.

    This is a convenience constructor that converts an index-based coefficient
    dictionary into dgcv's internal "tripled-key" `coeff_dict` format by using
    VMF metadata for the variables in `coordinate_space`.

    Parameters
    ----------
    coordinate_space : tuple or list
        Sequence of variables that defines the coordinate basis. Every element
        must be registered in the dgcv VMF (tip: use createVariables to
        register coordinates).

    coefficient_dict : dict
        Dictionary mapping tuples of integer indices to coefficient values.
        Each key is interpreted as a list of slots selecting elements of
        `coordinate_space` by index. Values are scalar coefficients.

    valence : tuple or list, optional
        Valence specification aligned with `coordinate_space`. If None, defaults
        to a tuple of zeros of the same length as `coordinate_space`.

    shape : {"general","symmetric","skew"}, default "general"
        Declared symmetry for the resulting tensor field.

    subclass : {"vector field","differential form"}, optional
        If provided, returns an instance of the requested subclass. If None,
        returns a `tensor_field_class`.

    Returns
    -------
    tensor_field_class
        A tensor field instance with coefficients stored in dgcv's internal
        tripled-key format.

    Raises
    ------
    ValueError
        If `valence` is provided and its length does not match `coordinate_space`,
        or if `coefficient_dict` keys contain indices outside the range of
        `coordinate_space`.

    LookupError
        If any variable in `coordinate_space` is not registered in the dgcv VMF.
    """
    return_class = (
        vector_field_class
        if subclass == "vector field"
        else differential_form_class
        if subclass == "differential form"
        else tensor_field_class
    )
    if valence is None:
        valence = tuple(0 for _ in range(len(coordinate_space)))
    if len(valence) != len(coordinate_space):
        raise ValueError(
            "`assemble_tensor_field` expects valence list if given to match the coordinate_space list length."
        )
    variable_spaces, coordinates = dict(), dict()
    for var in coordinate_space:
        data = vmf_lookup(var, path=True, flattened_relatives=True, system_index=True)
        path = data.get("path")
        syslabel = path[1] if path else None
        if syslabel is None:
            raise LookupError(
                "`assemble_tensor_field` requires variables in the given coordinate_space to be registered in the dgcv VMF."
            )
        if syslabel not in variable_spaces:
            variable_spaces[syslabel] = data.get("flattened_relatives")
        idx = data.get("system_index")
        coordinates[var] = {"idx": idx, "sysl": syslabel}

    def new_key(key):
        f, m, L = [], [], []
        for indx in key:
            try:
                cvar, val = coordinate_space[indx], valence[indx]
            except Exception:
                raise ValueError(
                    "`assemble_tensor_field` expects cooeficient dict keys to be tuples with indices in range of the coordinate_space list length."
                )
            f.append(coordinates[cvar]["idx"])
            m.append(val)
            L.append(coordinates[cvar]["sysl"])
        return tuple(f + m + L)

    cd = {new_key(k): v for k, v in coefficient_dict.items()}
    return return_class(coeff_dict=cd, data_shape=shape)


# -----------------------------------------------------------------------------
# polynomials
# -----------------------------------------------------------------------------
class polynomial_dgcv(dgcv_class):
    """
    A class representing polynomial expressions in the dgcv package, providing a light,
    wrapper around the polynomial functionality of the active symbolic backend.

    This class interprets a symbolic expression as a polynomial in a specified set of
    generators, with all other atomic symbols treated as parameters (unknown scalar
    constants). It integrates with dgcv's complex variable systems and the VMF, allowing
    efficient computation of complex-structure-related properties such as holomorphic,
    antiholomorphic, pluriharmonic, and mixed terms.

    Parameters
    ----------
    polyExpr : symbolic expression
        The symbolic expression to be interpreted as a polynomial.

    varSpace : list or tuple, optional
        Variables to be treated as polynomial generators. If not provided, the generators
        are inferred from the free symbols of `polyExpr`.

    parameters : list or tuple, optional
        Symbols to be treated as parameters (coefficients) rather than polynomial
        generators. If not provided and `varSpace` is specified, parameters are inferred
        lazily as the free symbols of `polyExpr` not contained in `varSpace`.

    degreeUpperBound : int, optional
        An optional upper bound on the total degree of the polynomial, used in applications
        where only terms up to a given degree are relevant.

    Methods
    -------
    get_monomials(min_degree=0, max_degree=None, format='unformatted', return_coeffs=False)
        Returns the monomials (or coefficients) of the polynomial within the specified
        degree range and representation.

    holomorphic_part
        Returns the holomorphic part of the polynomial with respect to dgcv's complex
        variable systems.

    antiholomorphic_part
        Returns the antiholomorphic part of the polynomial.

    pluriharmonic_part
        Returns the pluriharmonic part of the polynomial, defined as the sum of the
        holomorphic and antiholomorphic parts with the constant term counted once.

    mixed_terms
        Returns the mixed terms of the polynomial, involving nonzero powers of both
        holomorphic and antiholomorphic variables.

    simplify_poly(method=None, **kwargs)
        Simplifies the underlying polynomial expression using the active symbolic backend.

    subs(substitutions)
        Substitutes variables or expressions into the polynomial and returns a new
        `polynomial_dgcv` instance.

    diff(*symbols)
        Differentiates the polynomial with respect to the given symbols and returns a new
        `polynomial_dgcv` instance.
    """

    def __init__(
        self,
        polyExpr: Any,
        varSpace: Optional[Sequence[Any]] = None,
        *,
        parameters: Optional[Iterable[Any]] = None,
        degreeUpperBound: Optional[int] = None,
        conjugate_free: Optional[bool] = None,
    ):
        src = polyExpr if isinstance(polyExpr, polynomial_dgcv) else None

        if src is not None:
            if varSpace is None:
                varSpace = src.varSpace
            if degreeUpperBound is None:
                degreeUpperBound = src.degreeUpperBound
            if parameters is None:
                parameters = src._parameters
            polyExpr = src.polyExpr

        hard_filter = tuple(parameters) if parameters is not None else tuple()
        self.polyExpr = (
            polyExpr
            if conjugate_free
            else (
                polyExpr
                if verify_conjugate_re_im_free(polyExpr)
                else allToSym(polyExpr, skipVar=hard_filter)
            )
        )

        if varSpace is None:
            self._parameters = hard_filter
            if isinstance(polyExpr, Number) and not isinstance(polyExpr, bool):
                self.varSpace = ()
            else:
                self.varSpace = tuple(get_free_symbols(polyExpr))
        else:
            self.varSpace = polynomial_dgcv._normalize_polynomial_varspace_via_vmf(
                tuple(varSpace)
            )
            self._parameters = hard_filter if parameters is not None else None

        self.degreeUpperBound = degreeUpperBound

        self._degree = None
        self._poly_obj_unformatted = None
        self._poly_obj_complex = None
        self._poly_obj_real = None
        self._holomorphic_part = None
        self._antiholomorphic_part = None
        self._pluriharmonic_part = None
        self._mixed_terms_part = None
        self._is_zero = None
        self._is_one = None
        self._is_minus_one = None
        self._is_constant = None
        self._constant_term = None
        self._is_monomial = None
        self._complex_terms_cache = None
        self._complex_holo_anti_idx_cache = None

        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "expression"
        self._dgcv_categories = {"polynomial"}

        if src is not None:
            same_varspace = self.varSpace == src.varSpace
            same_deg_ub = self.degreeUpperBound == src.degreeUpperBound
            same_params_semantics = self._parameters == src._parameters

            if same_varspace and same_deg_ub and same_params_semantics:
                self._degree = src._degree
                self._poly_obj_unformatted = src._poly_obj_unformatted
                self._poly_obj_complex = src._poly_obj_complex
                self._poly_obj_real = src._poly_obj_real
                self._holomorphic_part = src._holomorphic_part
                self._antiholomorphic_part = src._antiholomorphic_part
                self._pluriharmonic_part = src._pluriharmonic_part
                self._mixed_terms_part = src._mixed_terms_part
                self._is_zero = src._is_zero
                self._is_one = src._is_one
                self._is_minus_one = src._is_minus_one
                self._is_constant = src._is_constant
                self._constant_term = src._constant_term
                self._is_monomial = src._is_monomial

    @property
    def parameters(self) -> Tuple[Any, ...]:
        if self._parameters is None:
            fs = tuple(get_free_symbols(self.polyExpr))
            vs = set(self.varSpace)
            self._parameters = tuple(x for x in _stable_dedupe(fs) if x not in vs)
        return self._parameters

    @property
    def free_symbols(self) -> set:
        return set(self.parameters) | set(self.varSpace)

    @property
    def degree(self) -> Optional[int]:
        if self._degree is None:
            try:
                self._degree = poly_total_degree(
                    self.polyExpr, self.varSpace, parameters=self.parameters
                )
            except Exception:
                self._degree = None
        return self._degree

    @staticmethod
    def _vmf_coordinate_info(x: Any):
        try:
            info = vmf_lookup(x, relatives=True, system_index=True)
        except Exception:
            return None
        if not isinstance(info, dict):
            return None
        t = info.get("type", None)
        if t == "unregistered":
            return None
        if t != "coordinate":
            return None
        rel = info.get("relatives", None)
        if not isinstance(rel, dict):
            return None
        sys_label = rel.get("system_label", None)
        if not isinstance(sys_label, str) or not sys_label:
            return None
        si = info.get("system_index", None)
        try:
            si_int = int(si)
        except Exception:
            return None
        return sys_label, si_int, rel

    @staticmethod
    def _normalize_polynomial_varspace_via_vmf(
        varSpace_in: Sequence[Any],
    ) -> Tuple[Any, ...]:
        blocks: List[tuple] = []
        block_index_for_slot: Dict[tuple, int] = {}
        seen_standard: set = set()

        def _append_standard(a: Any) -> None:
            if a in seen_standard:
                return
            seen_standard.add(a)
            if blocks and blocks[0][0] == "standard":
                blocks[0][1].append(a)
            else:
                blocks.insert(0, ("standard", [a]))

        for v in varSpace_in:
            if not is_atomic(v):
                raise TypeError(
                    "polynomial_dgcv(varSpace=...): all entries must be atomic symbols; "
                    f"got {v!r} of type {type(v).__name__}"
                )

            info = polynomial_dgcv._vmf_coordinate_info(v)
            if info is None:
                _append_standard(v)
                continue

            sys_label, sys_i, rel = info

            holo = rel.get("holo", None)
            anti = rel.get("anti", None)
            real = rel.get("real", None)
            imag = rel.get("imag", None)

            has_symb_pair = (holo is not None) and (anti is not None)
            has_real_pair = (real is not None) and (imag is not None)
            if not (has_symb_pair or has_real_pair):
                _append_standard(v)
                continue

            if has_symb_pair and (v == holo or v == anti):
                fmt_seen = "symb"
                pair = (holo, anti)
            elif has_real_pair and (v == real or v == imag):
                fmt_seen = "real"
                pair = (real, imag)
            else:
                _append_standard(v)
                continue

            slot_key = (sys_label, sys_i)

            if slot_key not in block_index_for_slot:
                blocks.append(("complex", slot_key, fmt_seen, list(pair)))
                block_index_for_slot[slot_key] = len(blocks) - 1
                continue

            idx = block_index_for_slot[slot_key]
            _tag, _slot, fmt_current, _pair_current = blocks[idx]

            if fmt_current == fmt_seen:
                continue

            blocks[idx] = ("complex", slot_key, fmt_seen, list(pair))

        out: List[Any] = []
        seen_out: set = set()

        for blk in blocks:
            if blk[0] == "standard":
                for a in blk[1]:
                    if a in seen_out:
                        continue
                    seen_out.add(a)
                    out.append(a)
            else:
                _tag, _slot, _fmt, pair_list = blk
                for a in pair_list:
                    if a in seen_out:
                        continue
                    seen_out.add(a)
                    out.append(a)

        return tuple(out)

    def __dgcv_converter__(
        self,
        conv,
        *,
        skipVar=None,
        simplify_everything=True,
        conversion_dict=None,
    ):
        conv_map = {
            "holToReal": holToReal,
            "realToSym": realToSym,
            "symToHol": symToHol,
            "realToHol": realToHol,
            "symToReal": symToReal,
            "allToReal": allToReal,
            "allToHol": allToHol,
            "allToSym": allToSym,
        }

        fn_expr = conv_map.get(conv)
        if fn_expr is None:
            return None

        if conv == "realToHol":
            fn_basis = realToSym
        elif conv == "allToHol":
            fn_basis = allToSym
        elif conv == "symToHol":
            fn_basis = None
        else:
            fn_basis = fn_expr

        kw_expr = {"skipVar": skipVar, "simplify_everything": simplify_everything}
        if conversion_dict is not None and conv in {
            "holToReal",
            "realToSym",
            "symToHol",
            "realToHol",
            "symToReal",
        }:
            kw_expr["_conversion_dict"] = conversion_dict

        new_expr = fn_expr(self.polyExpr, **kw_expr)

        if fn_basis is None:
            new_varSpace = self.varSpace
            new_params = None if self._parameters is None else self._parameters
        else:
            kw_basis = {"skipVar": skipVar, "simplify_everything": simplify_everything}

            vs_atoms = []
            for v in self.varSpace:
                vv = fn_basis(v, **kw_basis)
                vs_atoms.extend(get_free_symbols(vv))
            new_varSpace = tuple(_stable_dedupe(vs_atoms))

            if self._parameters is None:
                new_params = None
            else:
                p_atoms = []
                for p in self._parameters:
                    pp = fn_basis(p, **kw_basis)
                    p_atoms.extend(get_free_symbols(pp))
                new_params = tuple(_stable_dedupe(p_atoms))

        return polynomial_dgcv(
            new_expr,
            varSpace=new_varSpace,
            parameters=new_params,
            degreeUpperBound=self.degreeUpperBound,
        )

    def _complex_view(self, *, skipVar=None, simplify_everything=True):
        expr_c = allToSym(
            self.polyExpr, skipVar=skipVar, simplify_everything=simplify_everything
        )

        gens = []
        for v in self.varSpace:
            vv = allToSym(v, skipVar=skipVar, simplify_everything=simplify_everything)
            gens.extend(get_free_symbols(vv))
        gens_c = _stable_dedupe(gens)

        params_c = self.parameters
        return expr_c, gens_c, params_c

    def _real_view(self, *, skipVar=None, simplify_everything=True):
        expr_r = allToReal(
            self.polyExpr, skipVar=skipVar, simplify_everything=simplify_everything
        )

        gens = []
        for v in self.varSpace:
            vv = allToReal(v, skipVar=skipVar, simplify_everything=simplify_everything)
            gens.extend(get_free_symbols(vv))
        gens_r = _stable_dedupe(gens)

        params_r = self.parameters
        return expr_r, gens_r, params_r

    def __dgcv_simplify__(self, *, method=None, **kwargs):
        new_expr = simplify(self.polyExpr, method=method, **kwargs)
        return polynomial_dgcv(
            new_expr,
            varSpace=self.varSpace,
            parameters=self._parameters if self._parameters is not None else None,
            degreeUpperBound=self.degreeUpperBound,
        )

    def __dgcv_conjugate__(self):
        params = None if self._parameters is None else self._parameters
        return polynomial_dgcv(
            conjugate(self.polyExpr),
            varSpace=self.varSpace,
            parameters=params,
            degreeUpperBound=self.degreeUpperBound,
        )

    @property
    def poly_obj_unformatted(self):
        if self._poly_obj_unformatted is None:
            self._poly_obj_unformatted = make_poly(
                self.polyExpr, self.varSpace, parameters=self.parameters
            )
        return self._poly_obj_unformatted

    @property
    def poly_obj_complex(self):
        if self._poly_obj_complex is None:
            expr_c, gens_c, params_c = self._complex_view()
            self._poly_obj_complex = make_poly(expr_c, gens_c, parameters=params_c)
        return self._poly_obj_complex

    @property
    def poly_obj_real(self):
        if self._poly_obj_real is None:
            expr_r, gens_r, params_r = self._real_view()
            self._poly_obj_real = make_poly(expr_r, gens_r, parameters=params_r)
        return self._poly_obj_real

    @property
    def is_zero(self) -> bool:
        v = self._is_zero
        if v is not None:
            return v

        expr = self.polyExpr

        if isinstance(expr, Number) and not isinstance(expr, bool):
            self._is_zero = expr == 0
            return self._is_zero

        iz = getattr(expr, "is_zero", None)
        if iz is True:
            self._is_zero = True
            return True
        if callable(iz):
            try:
                self._is_zero = bool(iz())
                return self._is_zero
            except Exception:
                pass

        try:
            P = self.poly_obj_unformatted
            self._is_zero = all(c == 0 for c in poly_coeffs(P))
        except Exception:
            try:
                self._is_zero = expr == 0
            except Exception:
                self._is_zero = False

        return self._is_zero

    @property
    def is_constant(self) -> bool:
        v = self._is_constant
        if v is not None:
            return v

        try:
            P = self.poly_obj_unformatted
            self._is_constant = all(all(int(e) == 0 for e in m) for m in poly_monoms(P))
        except Exception:
            try:
                self._is_constant = len(self.varSpace) == 0
            except Exception:
                self._is_constant = False

        return self._is_constant

    @property
    def constant_term(self):
        v = self._constant_term
        if v is not None:
            return v

        expr = self.polyExpr

        try:
            P = self.poly_obj_unformatted
            for m, c in zip(poly_monoms(P), poly_coeffs(P)):
                if all(int(e) == 0 for e in m):
                    self._constant_term = c
                    return c
            self._constant_term = zero()
        except Exception:
            self._constant_term = expr

        return self._constant_term

    @property
    def is_one(self) -> bool:
        v = self._is_one
        if v is not None:
            return v

        expr = self.polyExpr

        if isinstance(expr, Number) and not isinstance(expr, bool):
            self._is_one = expr == 1
            return self._is_one

        try:
            self._is_one = bool(self.is_constant and (self.constant_term == 1))
        except Exception:
            try:
                self._is_one = expr == 1
            except Exception:
                self._is_one = False

        return self._is_one

    @property
    def is_minus_one(self) -> bool:
        v = self._is_minus_one
        if v is not None:
            return v

        expr = self.polyExpr

        if isinstance(expr, Number) and not isinstance(expr, bool):
            self._is_minus_one = expr == -1
            return self._is_minus_one

        try:
            self._is_minus_one = bool(self.is_constant and (self.constant_term == -1))
        except Exception:
            try:
                self._is_minus_one = expr == -1
            except Exception:
                self._is_minus_one = False

        return self._is_minus_one

    @property
    def is_monomial(self) -> bool:
        v = self._is_monomial
        if v is not None:
            return v

        try:
            coeffs = self.get_monomials(format="unformatted", return_coeffs=True)
            nz = 0
            for c in coeffs:
                if not _scalar_is_zero(c):
                    nz += 1
                    if nz > 1:
                        self._is_monomial = False
                        return False
            self._is_monomial = True
            return True
        except Exception:
            expr = self.polyExpr
            if isinstance(expr, Number) and not isinstance(expr, bool):
                self._is_monomial = True
                return True
            self._is_monomial = bool(getattr(expr, "is_Atom", False))
            return self._is_monomial

    def _complex_terms(self):
        """
        Return (gens, monoms, coeffs) for the complex view.
        """
        if self._complex_terms_cache is not None:
            return self._complex_terms_cache

        expr_c, gens_c, params_c = self._complex_view()

        gens = tuple(gens_c)

        from .backends._polynomials import poly_terms

        _gens_out, monoms, coeffs = poly_terms(
            expr_c,
            gens,
            assume_polynomial=False,
            parameters=params_c,
        )

        self._complex_terms_cache = (tuple(_gens_out), monoms, coeffs)
        return self._complex_terms_cache

    def _complex_holo_anti_indices(self):
        if self._complex_holo_anti_idx_cache is not None:
            return self._complex_holo_anti_idx_cache

        gens, _monoms, _coeffs = self._complex_terms()

        holo_idx = []
        anti_idx = []
        conjugation_prefix = get_dgcv_settings_registry().get(
            "conjugation_prefix", "BAR"
        )
        for i, g in enumerate(gens):
            s = str(g)
            if s.startswith(conjugation_prefix):
                anti_idx.append(i)
            else:
                holo_idx.append(i)

        self._complex_holo_anti_idx_cache = (tuple(holo_idx), tuple(anti_idx))
        return self._complex_holo_anti_idx_cache

    def get_monomials(
        self,
        min_degree: int = 0,
        max_degree: Optional[int] = None,
        *,
        format: str = "unformatted",
        return_coeffs: bool = False,
        separate_coeffs: bool = False,
        as_dict: bool = False,
    ):
        if format not in ("unformatted", "complex", "real"):
            raise ValueError("format must be one of: 'unformatted', 'complex', 'real'")

        if return_coeffs and (separate_coeffs or as_dict):
            raise ValueError(
                "return_coeffs cannot be combined with separate_coeffs or as_dict"
            )

        if as_dict:
            separate_coeffs = True

        if format == "unformatted":
            P = self.poly_obj_unformatted
        elif format == "complex":
            P = self.poly_obj_complex
        else:
            P = self.poly_obj_real

        gens_t = poly_gens(P)
        monoms = poly_monoms(P)
        coeffs = poly_coeffs(P)

        if max_degree is None:
            max_degree = self.degree
        if max_degree is None:
            max_degree = max(sum(int(e) for e in m) for m in monoms) if monoms else 0

        if return_coeffs:
            return [
                c
                for m, c in zip(monoms, coeffs)
                if min_degree <= sum(int(e) for e in m) <= max_degree
            ]

        if separate_coeffs:
            out_m = []
            out_c = []
            for m, c in zip(monoms, coeffs):
                d = sum(int(e) for e in m)
                if d < min_degree or d > max_degree:
                    continue
                mon = 1
                for g, e in zip(gens_t, m):
                    ee = int(e)
                    if ee:
                        mon *= g**ee
                out_m.append(mon)
                out_c.append(c)

            if as_dict:
                return dict(zip(out_m, out_c))

            return out_m, out_c

        out = []
        for m, c in zip(monoms, coeffs):
            d = sum(int(e) for e in m)
            if d < min_degree or d > max_degree:
                continue
            term = c
            for g, e in zip(gens_t, m):
                ee = int(e)
                if ee:
                    term *= g**ee
            out.append(term)

        return out

    def get_coeffs(
        self,
        min_degree: int = 0,
        max_degree: Optional[int] = None,
        *,
        format: str = "unformatted",
    ):
        return self.get_monomials(
            min_degree=min_degree,
            max_degree=max_degree,
            format=format,
            return_coeffs=True,
        )

    def _constant_term_in_complex_view(self):
        _gens, monoms, coeffs = self._complex_terms()
        for m, c in zip(monoms, coeffs):
            if all(int(e) == 0 for e in m):
                return c
        return zero()

    @property
    def holomorphic_part(self):
        if self._holomorphic_part is None:
            gens, monoms, coeffs = self._complex_terms()
            _holo_idx, anti_idx = self._complex_holo_anti_indices()

            terms = []
            for m, c in zip(monoms, coeffs):
                if anti_idx and any(int(m[i]) != 0 for i in anti_idx):
                    continue
                terms.append(_term_from_monom(gens, m, c))

            self._holomorphic_part = sum(terms, zero()) if terms else zero()
        return self._holomorphic_part

    @property
    def antiholomorphic_part(self):
        if self._antiholomorphic_part is None:
            gens, monoms, coeffs = self._complex_terms()
            holo_idx, _anti_idx = self._complex_holo_anti_indices()

            terms = []
            for m, c in zip(monoms, coeffs):
                if holo_idx and any(int(m[i]) != 0 for i in holo_idx):
                    continue
                terms.append(_term_from_monom(gens, m, c))

            self._antiholomorphic_part = sum(terms, zero()) if terms else zero()
        return self._antiholomorphic_part

    @property
    def mixed_terms(self):
        if self._mixed_terms_part is None:
            gens, monoms, coeffs = self._complex_terms()
            holo_idx, anti_idx = self._complex_holo_anti_indices()

            terms = []
            for m, c in zip(monoms, coeffs):
                has_holo = holo_idx and any(int(m[i]) != 0 for i in holo_idx)
                has_anti = anti_idx and any(int(m[i]) != 0 for i in anti_idx)
                if has_holo and has_anti:
                    terms.append(_term_from_monom(gens, m, c))

            self._mixed_terms_part = sum(terms, zero()) if terms else zero()
        return self._mixed_terms_part

    @property
    def pluriharmonic_part(self):
        if self._pluriharmonic_part is None:
            c0 = self._constant_term_in_complex_view()
            self._pluriharmonic_part = (
                self.holomorphic_part + self.antiholomorphic_part - c0
            )
        return self._pluriharmonic_part

    def simplify_poly(self, method: Optional[str] = None, **kwargs):
        return polynomial_dgcv(
            simplify(self.polyExpr, method=method, **kwargs),
            varSpace=self.varSpace,
            parameters=self.parameters,
            degreeUpperBound=self.degreeUpperBound,
        )

    def evaluate(self, **values):
        return subs(self.polyExpr, values)

    def subs(self, substitutions, **kwargs):
        new_expr = subs(self.polyExpr, substitutions, **kwargs)
        return polynomial_dgcv(
            new_expr,
            varSpace=self.varSpace,
            parameters=self._parameters if self._parameters is not None else None,
            degreeUpperBound=self.degreeUpperBound,
        )

    def diff(self, *args, **kwargs):
        new_expr = diff(self.polyExpr, *args, **kwargs)
        return polynomial_dgcv(
            new_expr,
            varSpace=self.varSpace,
            parameters=self._parameters if self._parameters is not None else None,
            degreeUpperBound=self.degreeUpperBound,
        )

    def scale_to_have_int_coeffs(
        self,
        return_scale_only: bool = False,
        *,
        reduce_by_gcd: bool = True,
        balance_sign: bool = True,
    ):
        coeffs = list(self.get_coeffs())

        def _is_zero(x):
            z = getattr(x, "is_zero", None)
            if z is True:
                return True
            if callable(z):
                try:
                    return bool(z())
                except Exception:
                    pass
            return x == 0

        nz_coeffs = [c for c in coeffs if not _is_zero(c)]
        if not nz_coeffs:
            scale = one()
            return scale if return_scale_only else scale * self

        def _sign_real(x):
            if _is_zero(x):
                return 0
            neg = getattr(x, "is_negative", None)
            pos = getattr(x, "is_positive", None)
            if neg is True:
                return -1
            if pos is True:
                return 1
            try:
                v = float(x)
                return -1 if v < 0 else (1 if v > 0 else 0)
            except Exception:
                return 0

        def _sign_quadrant(c):
            if _is_zero(c):
                return 0

            rr = re(c)
            ii = im(c)

            rr0 = _is_zero(rr)
            ii0 = _is_zero(ii)

            if ii0:
                return _sign_real(rr)

            if rr0:
                return _sign_real(ii)

            rr_nonpos = getattr(rr, "is_nonpositive", None)
            ii_nonpos = getattr(ii, "is_nonpositive", None)
            rr_nonneg = getattr(rr, "is_nonnegative", None)
            ii_nonneg = getattr(ii, "is_nonnegative", None)

            rr_neg = getattr(rr, "is_negative", None)
            ii_neg = getattr(ii, "is_negative", None)
            rr_pos = getattr(rr, "is_positive", None)
            ii_pos = getattr(ii, "is_positive", None)

            neg_corner = (
                rr_nonpos is True
                and ii_nonpos is True
                and (rr_neg is True or ii_neg is True)
            )
            pos_corner = (
                rr_nonneg is True
                and ii_nonneg is True
                and (rr_pos is True or ii_pos is True)
            )

            if neg_corner:
                return -1
            if pos_corner:
                return 1

            try:
                zc = complex(float(rr), float(ii))
                if zc.real <= 0 and zc.imag <= 0 and (zc.real < 0 or zc.imag < 0):
                    return -1
                if zc.real >= 0 and zc.imag >= 0 and (zc.real > 0 or zc.imag > 0):
                    return 1
            except Exception:
                pass

            return 0

        denoms = []
        for c in nz_coeffs:
            _, d = as_numer_denom(c)
            try:
                di = int(d)
            except Exception:
                continue
            if not _scalar_is_zero(di):
                denoms.append(abs(di))

        base = ilcm(*denoms) if denoms else one()
        base = integer(base) if base != one() else base

        if not reduce_by_gcd and not balance_sign:
            scale = base
            return scale if return_scale_only else scale * self

        scaled_coeffs = [base * c for c in nz_coeffs]

        def _as_int(x):
            if _is_zero(x):
                return 0
            try:
                xi = int(x)
            except Exception:
                return None
            try:
                if x == xi:
                    return xi
            except Exception:
                pass
            return None

        ints = []
        for c in scaled_coeffs:
            ci = _as_int(c)
            if ci is None:
                ints = []
                break
            if not _scalar_is_zero(ci):
                ints.append(abs(ci))

        g = 1
        if reduce_by_gcd and ints:
            g = ints[0]
            for u in ints[1:]:
                g = gcd(g, u)
                if g == 1:
                    break

        content = integer(g) if g != 1 else one()
        scale0 = base / content if content != one() else base

        if not balance_sign:
            scale = scale0
            return scale if return_scale_only else scale * self

        def _sign_count_after_scale(s):
            pos = 0
            neg = 0
            for c in nz_coeffs:
                q = _sign_quadrant(s * c)
                if q == 1:
                    pos += 1
                elif q == -1:
                    neg += 1
            return pos, neg

        pos0, neg0 = _sign_count_after_scale(scale0)
        pos1, neg1 = _sign_count_after_scale(-scale0)

        if pos1 > pos0 or (pos1 == pos0 and neg1 < neg0):
            scale = -scale0
        else:
            scale = scale0

        return scale if return_scale_only else scale * self

    def __add__(self, other):
        if _scalar_is_zero(other):
            return self
        if isinstance(other, polynomial_dgcv):
            new_vs = _stable_dedupe(self.varSpace + other.varSpace)
            return polynomial_dgcv(
                self.polyExpr + other.polyExpr,
                varSpace=new_vs,
                parameters=None,
                degreeUpperBound=self.degreeUpperBound,
            )
        if check_dgcv_scalar(other):
            inferred_param = self._parameters if self._parameters is not None else None
            OIP = get_free_symbols(other) - set(self.varSpace)
            if OIP:
                inferred_param = (
                    tuple(set(inferred_param) | OIP) if inferred_param else tuple(OIP)
                )
            return polynomial_dgcv(
                self.polyExpr + other,
                varSpace=self.varSpace,
                parameters=inferred_param,
                degreeUpperBound=self.degreeUpperBound,
            )
        return NotImplemented

    def __truediv__(self, other):
        if check_dgcv_scalar(other):
            return ratio(1, other) * self

    def __radd__(self, other):
        if _scalar_is_zero(other):
            return self
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, polynomial_dgcv):
            new_vs = _stable_dedupe(self.varSpace + other.varSpace)
            return polynomial_dgcv(
                self.polyExpr - other.polyExpr,
                varSpace=new_vs,
                parameters=None,
                degreeUpperBound=self.degreeUpperBound,
            )
        if check_dgcv_scalar(other):
            inferred_param = self._parameters if self._parameters is not None else None
            OIP = get_free_symbols(other) - set(self.varSpace)
            if OIP:
                inferred_param = (
                    tuple(set(inferred_param) | OIP) if inferred_param else tuple(OIP)
                )
            return polynomial_dgcv(
                self.polyExpr - other,
                varSpace=self.varSpace,
                parameters=inferred_param,
                degreeUpperBound=self.degreeUpperBound,
            )
        return NotImplemented

    def __rsub__(self, other):
        if check_dgcv_scalar(other):
            inferred_param = self._parameters if self._parameters is not None else None
            OIP = get_free_symbols(other) - set(self.varSpace)
            if OIP:
                inferred_param = (
                    tuple(set(inferred_param) | OIP) if inferred_param else tuple(OIP)
                )
            return polynomial_dgcv(
                other - self.polyExpr,
                varSpace=self.varSpace,
                parameters=inferred_param,
                degreeUpperBound=self.degreeUpperBound,
            )
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, polynomial_dgcv):
            new_vs = _stable_dedupe(self.varSpace + other.varSpace)

            if self.degreeUpperBound is None and other.degreeUpperBound is None:
                new_bound = None
            else:
                vals = [
                    v
                    for v in (self.degreeUpperBound, other.degreeUpperBound)
                    if v is not None
                ]
                new_bound = min(vals) if vals else None

            return polynomial_dgcv(
                self.polyExpr * other.polyExpr,
                varSpace=new_vs,
                parameters=None,
                degreeUpperBound=new_bound,
            )

        if check_dgcv_scalar(other):
            inferred_param = self._parameters if self._parameters is not None else None
            OIP = get_free_symbols(other) - set(self.varSpace)
            if OIP:
                inferred_param = (
                    tuple(set(inferred_param) | OIP) if inferred_param else tuple(OIP)
                )
            return polynomial_dgcv(
                self.polyExpr * other,
                varSpace=self.varSpace,
                parameters=inferred_param,
                degreeUpperBound=self.degreeUpperBound,
            )
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __dgcv_apply__(self, fun, **kwargs):
        new_expr = fun(self.polyExpr, **kwargs)
        if engine_kind() == "sympy":
            return polynomial_dgcv(
                make_poly(
                    new_expr, self.varSpace, parameters=self.parameters
                ).as_expr(),
                varSpace=self.varSpace,
                parameters=self._parameters,
                degreeUpperBound=self.degreeUpperBound,
            )
        return polynomial_dgcv(
            new_expr,
            varSpace=self.varSpace,
            parameters=self._parameters if self._parameters is not None else None,
            degreeUpperBound=self.degreeUpperBound,
        )

    def __dgcv_expand__(self, **kwargs):
        return self.expand(**kwargs)

    def expand(self, **kwargs):
        if engine_kind() == "sympy":
            return polynomial_dgcv(
                make_poly(
                    self.polyExpr, self.varSpace, parameters=self.parameters
                ).as_expr(),
                varSpace=self.varSpace,
                parameters=self._parameters,
                degreeUpperBound=self.degreeUpperBound,
            )
        return polynomial_dgcv(
            expand(self.polyExpr, **kwargs),
            varSpace=self.varSpace,
            parameters=self._parameters if self._parameters is not None else None,
            degreeUpperBound=self.degreeUpperBound,
        )

    def factor(self, **kwargs):
        new_expr = factor(self.polyExpr, **kwargs)
        return polynomial_dgcv(
            new_expr,
            varSpace=self.varSpace,
            parameters=self._parameters if self._parameters is not None else None,
            degreeUpperBound=self.degreeUpperBound,
        )

    def cancel(self, **kwargs):
        new_expr = cancel(self.polyExpr, **kwargs)
        return polynomial_dgcv(
            new_expr,
            varSpace=self.varSpace,
            parameters=self._parameters if self._parameters is not None else None,
            degreeUpperBound=self.degreeUpperBound,
        )

    def integrate(self, *args, **kwargs):
        new_expr = integrate(self.polyExpr, *args, **kwargs)
        return polynomial_dgcv(
            new_expr,
            varSpace=self.varSpace,
            parameters=self._parameters if self._parameters is not None else None,
            degreeUpperBound=self.degreeUpperBound,
        )

    def get_degree(self):
        return self.degree

    def is_homogeneous(self, *, format: str = "unformatted") -> bool:
        P = (
            self.poly_obj_unformatted
            if format == "unformatted"
            else (self.poly_obj_complex if format == "complex" else self.poly_obj_real)
        )
        monoms = poly_monoms(P)
        if not monoms:
            return True
        degs = {sum(int(e) for e in m) for m in monoms}
        return len(degs) <= 1

    def leading_term(self, *, format: str = "unformatted"):
        terms = self.get_monomials(format=format, return_coeffs=False)
        if not terms:
            return zero()

        def _deg_of_term(t):
            try:
                p = polynomial_dgcv(
                    t, varSpace=self.varSpace, parameters=self.parameters
                )
                d = p.degree
                return -1 if d is None else int(d)
            except Exception:
                return -1

        return max(terms, key=_deg_of_term)

    def latex_representation(self, removeBARs: bool = False) -> str:
        x = getattr(self, "polyExpr", None)
        if x is None:
            return ""

        if not removeBARs:
            try:
                x = symToHol(x, simplify_everything=False)
            except Exception:
                pass

        s = _backend_latex(x)
        if isinstance(s, str) and s.strip():
            return _unwrap_math_delims(s)

        return str(x)

    def _repr_latex_(self, raw: bool = False, removeBARs: bool = False, **kwargs):
        s = self.latex_representation(removeBARs=removeBARs)
        s = _unwrap_math_delims(s)

        if raw:
            return s
        return f"$\\displaystyle {s}$"

    def _latex(
        self, printer=None, raw: bool = True, removeBARs: bool = False, **kwargs
    ):
        return self._repr_latex_(raw=raw, removeBARs=removeBARs, **kwargs)

    def __str__(self):
        try:
            return str(self.polyExpr)
        except Exception:
            return self.__repr__()

    def pretty_print(self):
        f = getattr(self.polyExpr, "pretty", None)
        if callable(f):
            try:
                return f()
            except Exception:
                pass
        return str(self.polyExpr)


def _vmf_atoms_for_view(v, *, view: str):
    data = vmf_lookup(
        v,
        path=True,
        relatives=True,
        flattened_relatives=False,
        system_index=True,
    )
    rel = data.get("relatives") or {}

    st = rel.get("standard") or ()
    if st:
        return tuple(st)

    keys = ("holo", "anti") if view == "complex" else ("real", "imag")
    out = []
    for k in keys:
        w = rel.get(k)
        if w is None:
            continue
        if isinstance(w, tuple):
            out.extend(w)
        else:
            out.append(w)
    return tuple(out)


def _term_from_monom(gens, m, c):
    term = c
    for g, e in zip(gens, m):
        if e:
            term = term * (g ** int(e))
    return term


def _build_holo_anti_index_sets(P):
    gens = tuple(poly_gens(P))

    vr = get_variable_registry()
    conv = vr.get("conversion_dictionaries", {})

    holo_keys = set(conv.get("holToReal", {}).keys())  # strings like "z1"
    anti_keys = set(conv.get("symToHol", {}).keys())  # strings like "BARz1"

    holo_idx = [i for i, g in enumerate(gens) if str(g) in holo_keys]
    anti_idx = [i for i, g in enumerate(gens) if str(g) in anti_keys]

    return gens, holo_idx, anti_idx


def DGCVPolyClass(polyExpr, varSpace=None, degreeUpperBound=None):
    warnings.warn(
        "`DGCVPolyClass` has been deprecated as part of a shift toward standardized "
        "naming conventions in the `dgcv` library. It will be removed in 2026. "
        "Please use `polynomial_dgcv` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return polynomial_dgcv(
        polyExpr,
        varSpace=varSpace,
        degreeUpperBound=degreeUpperBound,
    )


class dgcvPolyClass(polynomial_dgcv):
    """
    Deprecated alias for `polynomial_dgcv`.

    This class is retained for backward compatibility and will be removed in 2026.
    """

    def __init__(self, polyExpr, varSpace=None, degreeUpperBound=None):
        warnings.warn(
            "`dgcvPolyClass` has been deprecated and will be removed in 2026. "
            "Please use `polynomial_dgcv` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            polyExpr,
            varSpace=varSpace,
            degreeUpperBound=degreeUpperBound,
        )


# -----------------------------------------------------------------------------
# coordinate system creation
# -----------------------------------------------------------------------------
def createVariables(
    variable_label,
    real_label=None,
    imaginary_label=None,
    number_of_variables=None,
    initialIndex=1,
    withVF=None,
    complex=None,
    multiindex_shape=None,
    assumeReal=None,
    return_created_object=None,
    remove_guardrails=None,
    default_var_format=None,
    temporary_variables=False,
):
    """
    This function serves as the default interface for creating variables within the dgcv package. It supports creating
    both standard coordinate systems and complex coordinate systems, with options for initializing coordinate vector fields
    and differential forms. Variables created through `createVariables` are automatically tracked within dgcv's Variable
    Management Framework (VMF) and are assigned labels validated through a safeguards routine that prevents overwriting important labels (e.g., standard Python built-ins).

    Parameters
    ----------
    variable_label : str
        The label for the primary variable or system of variables to be created. If creating a complex variable system,
        this will correspond to the holomorphic variable(s), whilst antiholomorphic variable(s) recieve this label
        pre-pended with "BAR".

    real_label : str, optional
        The label for the real part of the complex variable system. Required only when creating complex variable systems.

    imaginary_label : str, optional
        The label for the imaginary part of the complex variable system. Required only when creating complex variable
        systems.

    number_of_variables : int, optional
        The number of variables to be created, used to initialize a tuple of variables rather than a single variable
        (e.g., x=(x1, x2, x3) rather than just x).

    initialIndex : int, optional, default=1
        The starting index for tuple variables, allowing for flexible indexing when initializing variable systems
        as tuples (e.g., x=(x0, x1, x2) with `initialIndex=0`).

    withVF : bool, optional
        If set to True, creates associated coordinate vector fields and differential forms for the variable(s) in the
        system.

    complex : bool, optional
        Specifies whether to create a complex variable system. If not provided, the function will infer whether to
        create a complex system based on whether `real_label` or `imaginary_label` is provided. If provided a complex
        variable system will be created regardless of `real_label` and `imaginary_label` settings, and string labels
        are automatically created for `real_label` and `imaginary_label` if they are not provided.

    multiindex_shape : tuple[int, ...], optional
        If provided, creates a multi-index variable system with shape given by `multiindex_shape`, using index values
        starting at `initialIndex`. For standard systems this creates an N-dimensional array handle stored in the global
        namespace under `variable_label` (or each label if a list is provided). For complex systems, this creates a
        multi-index complex variable system with holomorphic/antiholomorphic and real/imaginary parts.

    assumeReal : bool, optional
        If set to True, specifies that the variables being created are real-valued. This is only relevant for standard
        variable systems and is ignored for complex systems.

    remove_guardrails : bool, optional
        If set to True, bypasses dgcv's safeguard system for variable labeling, allowing one to overwrite certain
        reserved labels. Use with caution, as it may overwrite important variables in the global namespace.

    default_var_format : {'complex', 'real'}, optional
        Relevant only for complex variable systems. Specifies whether the system's vector fields and differential forms
        default to real coordinate expressions (`real`) or holomorphic coordinate expressions (`complex`). If not
        provided, the default is holomorphic coordinates.

    temporary_variables : bool, optional
        If True, variables are created as temporary variables under dgcv's VMF conventions.

    Returns
    -------
    None or list
        If `return_created_object=True`, returns a list of created handles (for standard systems) or created labels (for
        complex systems, consistent with dgcv conventions). Otherwise returns None.

    Functionality
    -------------
    - Creates standard or complex variable systems.
    - Automatically registers all created variables, vector fields, and differential forms in dgcv's VMF.
    - Safeguards are applied to prevent overwriting critical Python or dgcv internal functions by default.

    Notes
    -----
    - For complex variable systems, dgcv initializes associated differential objects by default.
    - For multi-index standard systems, `withVF=True` is supported and will create VF/DF objects for each atomic entry.
    - For multi-index complex systems, VF/DF objects are created for each atomic entry in the system.
    - Use `vmf_summary()` for a clear summary of variables created and tracked within the dgcv VMF.

    Examples
    --------
    # Creating a single standard variable 'alpha'
    >>> createVariables('alpha')

    # Creating a tuple of 3 standard variables with associated vector fields and differential forms
    >>> createVariables('alpha', 3, withVF=True)

    # Creating a single complex variable system with real part 'beta' and imaginary part 'gamma'
    >>> createVariables('alpha', 'beta', 'gamma')

    # Creating a tuple of 3 complex variables
    >>> createVariables('alpha', 'beta', 'gamma', 3)

    # Creating a tuple of 3 standard variables with custom initial index
    >>> createVariables('alpha', 3, initialIndex=0)

    # Creating a complex variable system without specifying real and imaginary part labels (obscure labels are created)
    >>> createVariables('alpha', complex=True)

    # Creating a multi-index standard variable system
    >>> createVariables('X', multiindex_shape=(2, 3))

    # Creating a multi-index complex variable system
    >>> createVariables('z', 'x', 'y', multiindex_shape=(2, 3))

    Warnings
    --------
    If incompatible keywords are provided (e.g., `complex=True` with `withVF=False`, or `complex=False` while also
    providing `real_label`/`imaginary_label`), the function resolves conflicts internally and issues a warning about the
    resolution.
    """

    def _valid_multiindex_shape(ms):
        return isinstance(ms, (list, tuple)) and all(
            isinstance(n, Integral) and n > 0 for n in ms
        )

    def reformat_string(input_string: str):
        substrings = input_string.replace(",", " ").split()
        return [s for s in substrings if len(s) > 0]

    if not isinstance(variable_label, str):
        raise TypeError(
            "`createVariables` requires its first argument to be a string, which will be used in labels for the created variables."
        )

    if (
        isinstance(real_label, Integral)
        and imaginary_label is None
        and number_of_variables is None
    ):
        number_of_variables = real_label
        real_label = None

    if real_label is not None and not isinstance(real_label, str):
        raise TypeError(
            "A non-string value cannot be assigned to the `real_label` keyword of `createVariables`."
        )
    if imaginary_label is not None and not isinstance(imaginary_label, str):
        raise TypeError(
            "A non-string value cannot be assigned to the `imaginary_label` keyword of `createVariables`."
        )

    if multiindex_shape is not None and not _valid_multiindex_shape(multiindex_shape):
        raise TypeError("`multiindex_shape` must be a tuple/list of positive integers.")

    if multiindex_shape is not None and number_of_variables is not None:
        raise ValueError(
            "Provide at most one of `number_of_variables` (tuple system) and `multiindex_shape` (multi-index system)."
        )

    complex_requested = (
        complex is True or real_label is not None or imaginary_label is not None
    )

    if default_var_format is not None and not complex_requested:
        warnings.warn(
            "`default_var_format` is only relevant for complex variable systems; it was disregarded."
        )
        default_var_format = None

    if complex and not withVF:
        warnings.warn(
            "`createVariables` was called with `complex=True` and `withVF=False`. The latter keyword was disregarded because "
            "dgcv initializes associated differential objects whenever complex variable systems are created."
        )

    if complex and assumeReal:
        warnings.warn(
            "`createVariables` was called with `complex=True` and `assumeReal=True`. The latter keyword was disregarded because "
            "dgcv has fixed variable assumptions for elements in its complex variable systems."
        )
        assumeReal = None

    if complex is False and (real_label is not None or imaginary_label is not None):
        warnings.warn(
            "`createVariables` received `complex=False` and values for `real_label` and/or `imaginary_label`. Honoring "
            "`complex=False`, only a standard variable system was created and the latter labels were disregarded."
        )
        real_label = None
        imaginary_label = None
        complex_requested = False

    if complex is True or complex_requested:
        if (
            complex is not True
            and complex is not None
            and (real_label or imaginary_label)
        ):
            warnings.warn(
                "The keyword `complex` was set to a non-bool value. Since a string value was also assigned to either "
                "`real_label` or `imaginary_label`, `createVariables` proceeded under the assumption that it should "
                "create a complex variable system. Set `complex=False` to force a standard variable system."
            )
        complex = True

        if real_label is None and imaginary_label is None:
            key_string = retrieve_public_key()
            real_label = variable_label + "REAL" + key_string
            imaginary_label = variable_label + "IM" + key_string
            warnings.warn(
                "`createVariables` received `complex=True` and did not receive assignments for `real_label` or "
                "`imaginary_label`, so intentionally obscure labels were created for both."
            )
        else:
            if real_label is None:
                real_label = variable_label + "REAL" + retrieve_public_key()
                warnings.warn(
                    "`createVariables` received a value for `imaginary_label` but not `real_label`, so an intentionally "
                    "obscure label was created for the real variables."
                )
            if imaginary_label is None:
                imaginary_label = variable_label + "IM" + retrieve_public_key()
                warnings.warn(
                    "`createVariables` received a value for `real_label` but not `imaginary_label`, so an intentionally "
                    "obscure label was created for the imaginary variables."
                )

    variable_label = reformat_string(variable_label)
    if isinstance(real_label, str):
        real_label = reformat_string(real_label)
    if isinstance(imaginary_label, str):
        imaginary_label = reformat_string(imaginary_label)

    if complex:
        rv = complexVarProc(
            variable_label,
            real_label,
            imaginary_label,
            number_of_variables=number_of_variables,
            initialIndex=initialIndex,
            multiindex_shape=multiindex_shape,
            default_var_format=default_var_format,
            remove_guardrails=remove_guardrails,
            return_created_object=return_created_object,
        )
        return rv if rv is not None else None

    if withVF:
        rv = varWithVF(
            variable_label,
            number_of_variables=number_of_variables,
            initialIndex=initialIndex,
            multiindex_shape=multiindex_shape,
            _doNotUpdateVar=False,
            assumeReal=assumeReal,
            _calledFromCVP=None,
            remove_guardrails=remove_guardrails,
            return_created_object=return_created_object,
        )
        return rv if rv is not None else None

    pk = retrieve_passkey()
    tv = pk if temporary_variables else None
    rv = variableProcedure(
        variable_label,
        number_of_variables=number_of_variables,
        initialIndex=initialIndex,
        assumeReal=assumeReal,
        multiindex_shape=multiindex_shape,
        _tempVar=tv,
        _doNotUpdateVar=None,
        _calledFromCVP=pk,
        remove_guardrails=remove_guardrails,
        return_created_object=return_created_object,
    )
    return rv if rv is not None else None


############## variable factories


class coordinate_system(tuple):  ###!!! TO FIX
    def __new__(
        cls, *elements, label=None, shape=None, dgcv_type="standard", rich=False
    ):
        obj = super().__new__(cls, elements)
        return obj

    def __init__(
        self, *elements, label=None, shape=None, dgcv_type="standard", rich=False
    ):
        if shape is None:
            self.shape = (len(self),)
        else:
            self.shape = tuple(shape)
        self.dgcv_type = dgcv_type
        self.rich = rich
        self.label = label
        self._spooled_id = {}

    def _spool_modifier(self, midx, val):
        self._spooled_id[midx] = val
        return val

    def _spool(self, midx):
        return sum(midx[j] * prod(self.shape[j + 1 :]) for j in range(len(self.shape)))

    def _unspool(self, idx):
        midx = []
        p = idx
        for j in range(len(self.shape)):
            q = prod(self.shape[j + 1 :]) if j + 1 < len(self.shape) else 1
            midx1 = p // q
            midx.append(midx1)
            p -= midx1 * q
        return tuple(midx)

    def __getitem__(self, idx):
        """
        Supports:
          - cs[i]                 -> flat index like a normal tuple
          - cs[i, j, k, ...]     -> multi-index matching self.shape (spooled to flat)
          - (optional) partial multi-index cs[i, j]  -> TODO (not implemented here)
        """
        if not isinstance(idx, tuple):
            if isinstance(idx, Integral):
                return tuple.__getitem__(self, idx)
            elif isinstance(idx, slice):
                return tuple.__getitem__(self, idx)
            else:
                raise TypeError(f"Invalid index type: {type(idx).__name__}")

        if any(isinstance(x, slice) for x in idx):
            raise TypeError("Slicing with slices in multi-index is not implemented.")

        if len(idx) == len(self.shape):
            key = tuple(idx)
            flat = self._spooled_id.get(
                key, self._spool_modifier(key, self._spool(key))
            )
            return tuple.__getitem__(self, flat)

        if len(idx) < len(self.shape):
            raise NotImplementedError(
                f"Partial indexing with {len(idx)} components for shape {self.shape} not implemented."
            )

        raise TypeError(
            "Multi-index has too many components for this coordinate_system."
        )


def variableProcedure(
    variables_label,
    number_of_variables=None,
    initialIndex=1,
    multiindex_shape=None,
    assumeReal=None,
    return_created_object=None,
    _tempVar=None,
    _doNotUpdateVar=None,
    _calledFromCVP=None,
    remove_guardrails=None,
    _obscure=None,
):
    """
    Initializes one or more standard variable systems (single or tuples) and integrates them into dgcv's Variable Management Framework.
    """

    globals_namespace = _cached_caller_globals
    passkey = retrieve_passkey()
    variable_registry = get_variable_registry()
    settings = get_dgcv_settings_registry()

    if (
        settings["ask_before_overwriting_objects_in_vmf"]
        and not _calledFromCVP == passkey
    ):
        labels_iter = (
            tuple(variables_label)
            if isinstance(variables_label, (list, tuple))
            else (variables_label,)
        )
        protected = variable_registry.get("protected_variables", set())
        for j in labels_iter:
            if j in protected:
                raise Exception(
                    f"{variables_label} is already assigned to the real or imaginary part of a complex variable system, "
                    "so dgcv variable creation functions will not reassign it as a standard variable. Instead, use the "
                    "clearVar function to remove the conflicting CV system first before implementing such reassignments."
                )

    kind = engine_kind()
    enforce_real = kind is not None and kind != "sympy"
    enforced_real_dict = (
        variable_registry.get("dgcv_enforced_real_atoms", None)
        if enforce_real
        else None
    )

    labels = (
        tuple(variables_label)
        if isinstance(variables_label, (list, tuple))
        else (variables_label,)
    )

    rco = [] if return_created_object is True else None

    paths = variable_registry.get("paths", None)

    for j in labels:
        labelLoc = (
            validate_label(j)
            if (not _calledFromCVP == passkey and not remove_guardrails)
            else j
        )

        if _doNotUpdateVar != passkey:
            clearVar(labelLoc, report=False)

        temp_flag = True if _tempVar == passkey else None
        obscure_flag = True if _obscure == passkey else None

        if temp_flag is True:
            variable_registry["temporary_variables"].add(labelLoc)
        if obscure_flag is True:
            variable_registry["obscure_variables"].add(labelLoc)
        _tuple_flag = False
        if isinstance(multiindex_shape, (list, tuple)) and all(
            isinstance(n, Integral) and n > 0 for n in multiindex_shape
        ):
            _tuple_flag = True
            indices = list(
                carProd(
                    *[range(initialIndex, initialIndex + n) for n in multiindex_shape]
                )
            )
            var_names = [f"{labelLoc}_{'_'.join(map(str, idx))}" for idx in indices]
            if (
                _doNotUpdateVar != passkey or _calledFromCVP == passkey
            ):  # CVP doesn't manage sub-parant names
                clearVar(*var_names, report=False)
            vars = tuple(symbol(name, assumeReal=assumeReal) for name in var_names)

            new_globals = dict(zip(var_names, vars))
            new_globals[labelLoc] = build_nd_array(vars, multiindex_shape)
            globals_namespace.update(new_globals)

            if enforce_real and assumeReal is True and enforced_real_dict is not None:
                for v in vars:
                    enforced_real_dict[v.conjugate()] = v

            if _doNotUpdateVar != passkey:
                variable_registry["standard_variable_systems"][labelLoc] = {
                    "family_type": "multi_index",
                    "family_shape": multiindex_shape,
                    "family_names": tuple(var_names),
                    "family_values": globals_namespace[labelLoc],
                    "differential_system": None,
                    "tempVar": temp_flag,
                    "obsVar": obscure_flag,
                    "initial_index": initialIndex,
                    "variable_relatives": {
                        var_name: {
                            "VFClass": None,
                            "DFClass": None,
                            "assumeReal": assumeReal,
                            "system_index": idx,
                        }
                        for idx, var_name in enumerate(var_names)
                    },
                }
                variable_registry["_labels"][labelLoc] = {
                    "path": ("standard_variable_systems", labelLoc),
                    "children": set(var_names),
                }

                if paths is not None:
                    paths[labelLoc] = {
                        "kind": "standard_variable_system",
                        "path": ("standard_variable_systems", labelLoc),
                    }

                    base = ("standard_variable_systems", labelLoc, "family_values")
                    for idx, var_name in zip(indices, var_names):
                        offs = tuple(int(k - initialIndex) for k in idx)
                        paths[var_name] = {
                            "kind": "standard_variable",
                            "path": base + offs,
                        }

        elif number_of_variables is None:
            sym = symbol(labelLoc, assumeReal=assumeReal)
            globals_namespace[labelLoc] = sym

            if enforce_real and assumeReal is True and enforced_real_dict is not None:
                enforced_real_dict[sym.conjugate()] = sym

            if _doNotUpdateVar != passkey:
                variable_registry["standard_variable_systems"][labelLoc] = {
                    "family_type": "single",
                    "family_names": (labelLoc,),
                    "family_values": (globals_namespace[labelLoc],),
                    "differential_system": None,
                    "tempVar": temp_flag,
                    "obsVar": obscure_flag,
                    "initial_index": None,
                    "variable_relatives": {
                        labelLoc: {
                            "VFClass": None,
                            "DFClass": None,
                            "assumeReal": assumeReal,
                            "system_index": 0,
                        }
                    },
                }
                variable_registry["_labels"][labelLoc] = {
                    "path": ("standard_variable_systems", labelLoc),
                    "children": set(),
                }

                if paths is not None:
                    paths[labelLoc] = {
                        "kind": "standard_variable",
                        "path": (
                            "standard_variable_systems",
                            labelLoc,
                            "family_values",
                            0,
                        ),
                    }

        elif isinstance(number_of_variables, Integral) and number_of_variables >= 0:
            _tuple_flag = True
            lengthLoc = number_of_variables
            var_names = [
                f"{labelLoc}{i}" for i in range(initialIndex, lengthLoc + initialIndex)
            ]
            vars = [symbol(name, assumeReal=assumeReal) for name in var_names]
            if (
                _doNotUpdateVar != passkey or _calledFromCVP == passkey
            ):  # CVP doesn't manage sub-parant names
                clearVar(*vars, report=False)
                new_globals = dict(zip(var_names, vars))
                new_globals[labelLoc] = tuple(vars)
                globals_namespace.update(new_globals)

            if enforce_real and assumeReal is True and enforced_real_dict is not None:
                for v in vars:
                    enforced_real_dict[v.conjugate()] = v

            vtuple = tuple(vars)

            vf_instances = [
                vector_field_class(
                    coeff_dict={(j, 1, labelLoc): 1},
                    dgcvType="standard",
                    variable_spaces={labelLoc: vtuple},
                )
                for j in range(len(vtuple))
            ]
            globals_namespace.update(
                dict(zip([f"D_{vn}" for vn in var_names], vf_instances))
            )

            df_instances = [
                differential_form_class(
                    coeff_dict={(j, 0, labelLoc): 1},
                    dgcvType="standard",
                    variable_spaces={labelLoc: vtuple},
                )
                for j in range(len(vtuple))
            ]
            globals_namespace.update(
                dict(zip([f"d_{vn}" for vn in var_names], df_instances))
            )

            if _doNotUpdateVar != passkey:
                variable_registry["standard_variable_systems"][labelLoc] = {
                    "family_type": "tuple",
                    "family_values": vtuple,
                    "family_names": tuple(var_names),
                    "differential_system": True,
                    "tempVar": temp_flag,
                    "initial_index": initialIndex,
                    "obsVar": obscure_flag,
                    "variable_relatives": {
                        var_name: {
                            "VFClass": vf_instances[i],
                            "DFClass": df_instances[i],
                            "assumeReal": assumeReal,
                            "system_index": i,
                        }
                        for i, var_name in enumerate(var_names)
                    },
                }
                variable_registry["_labels"][labelLoc] = {
                    "path": ("standard_variable_systems", labelLoc),
                    "children": set(var_names),
                }

                if paths is not None:
                    paths[labelLoc] = {
                        "kind": "standard_variable_system",
                        "path": ("standard_variable_systems", labelLoc),
                    }

                    base_vals = ("standard_variable_systems", labelLoc, "family_values")
                    base_rel = (
                        "standard_variable_systems",
                        labelLoc,
                        "variable_relatives",
                    )

                    for i, var_name in enumerate(var_names):
                        paths[var_name] = {
                            "kind": "standard_variable",
                            "path": base_vals + (i,),
                        }
                        paths[f"D_{var_name}"] = {
                            "kind": "vector_field",
                            "path": base_rel + (var_name, "VFClass"),
                        }
                        paths[f"d_{var_name}"] = {
                            "kind": "differential_form",
                            "path": base_rel + (var_name, "DFClass"),
                        }

        else:
            raise ValueError(
                "variableProcedure expected its second argument number_of_variables (optional) to be a positive integer, if provided."
            )

        if rco is not None:
            if _tuple_flag:
                rco.append(globals_namespace[labelLoc])
            else:
                rco.append((globals_namespace[labelLoc],))
    return rco


def varWithVF(
    variables_label,
    number_of_variables=None,
    initialIndex=1,
    multiindex_shape=None,
    _doNotUpdateVar=False,
    assumeReal=None,
    _calledFromCVP=None,
    remove_guardrails=None,
    return_created_object=None,
):
    """
    Initializes one or more standard variable systems with accompanying vector fields and differential 1-forms.
    Supports single variables, tuple systems, and multi-index systems.
    Updates the internal variable_registry dict accordingly.
    """

    variable_registry = get_variable_registry()
    globals_namespace = _cached_caller_globals
    passkey = retrieve_passkey()

    if multiindex_shape is not None and number_of_variables is not None:
        multiindex_shape = None
        warnings.warn(
            "Provide at most one of `number_of_variables` (tuple system) and `multiindex_shape` (multi-index system). The given `multiindex_shape` was ignored."
        )

    labels = (
        [variables_label] if isinstance(variables_label, str) else list(variables_label)
    )
    rco = [] if return_created_object is True else None

    kind = engine_kind()
    enforce_real = kind is not None and kind != "sympy"
    enforced_real_dict = (
        variable_registry.get("dgcv_enforced_real_atoms", None)
        if enforce_real
        else None
    )

    def _valid_multiindex_shape(ms):
        return isinstance(ms, (list, tuple)) and all(
            isinstance(n, Integral) and n > 0 for n in ms
        )

    def _indices_for_multiindex(ms):
        return list(carProd(*[range(initialIndex, initialIndex + n) for n in ms]))

    for raw_label in labels:
        labelLoc = (
            validate_label(raw_label)
            if (not _calledFromCVP == passkey and not remove_guardrails)
            else raw_label
        )

        if _doNotUpdateVar != passkey:
            clearVar(labelLoc, report=False)

        _tuple_flag = False

        if number_of_variables is None and _valid_multiindex_shape(multiindex_shape):
            _tuple_flag = True
            idxs = _indices_for_multiindex(multiindex_shape)
            var_names = [f"{labelLoc}_{'_'.join(map(str, idx))}" for idx in idxs]
            vars = [symbol(name, assumeReal=assumeReal) for name in var_names]
            base_vars = tuple(vars)
            if (
                _doNotUpdateVar != passkey or _calledFromCVP == passkey
            ):  # CVP doesn't manage sub-parant names
                clearVar(*vars, report=False)
                globals_namespace.update(dict(zip(var_names, vars)))
                globals_namespace[labelLoc] = build_nd_array(
                    base_vars, multiindex_shape
                )

            if enforce_real and assumeReal is True and enforced_real_dict is not None:
                for v in vars:
                    enforced_real_dict[v.conjugate()] = v

            N = len(base_vars)
            vf_instances = [
                vector_field_class(
                    coeff_dict={(j, 1, labelLoc): 1},
                    dgcvType="standard",
                    variable_spaces={labelLoc: base_vars},
                )
                for j in range(N)
            ]
            df_instances = [
                differential_form_class(
                    coeff_dict={(j, 0, labelLoc): 1},
                    dgcvType="standard",
                    variable_spaces={labelLoc: base_vars},
                )
                # DFClass(base_vars, {(j,): 1}, 1)
                for j in range(N)
            ]

            if _doNotUpdateVar != passkey:
                globals_namespace.update(
                    dict(zip([f"D_{vn}" for vn in var_names], vf_instances))
                )
                globals_namespace.update(
                    dict(zip([f"d_{vn}" for vn in var_names], df_instances))
                )

            if not _calledFromCVP == passkey:
                variable_registry["standard_variable_systems"][labelLoc] = {
                    "family_type": "multi_index",
                    "family_shape": tuple(multiindex_shape),
                    "family_names": tuple(var_names),
                    "family_values": globals_namespace[labelLoc],
                    "differential_system": True,
                    "tempVar": None,
                    "obsVar": None,
                    "initial_index": initialIndex,
                    "variable_relatives": {
                        var_name: {
                            "VFClass": vf_instances[i],
                            "DFClass": df_instances[i],
                            "assumeReal": assumeReal,
                            "system_index": i,
                        }
                        for i, var_name in enumerate(var_names)
                    },
                }
                variable_registry["_labels"][labelLoc] = {
                    "path": ("standard_variable_systems", labelLoc),
                    "children": set(
                        var_names
                        + [f"D_{v}" for v in var_names]
                        + [f"d_{v}" for v in var_names]
                    ),
                }

                paths = variable_registry.get("paths", None)
                if paths is not None:
                    base_vals = ("standard_variable_systems", labelLoc, "family_values")
                    for idx, var_name in zip(idxs, var_names):
                        offs = tuple(int(k - initialIndex) for k in idx)
                        paths[var_name] = {
                            "kind": "standard_variable",
                            "path": base_vals + offs,
                        }

        elif number_of_variables is None:
            sym = symbol(labelLoc, assumeReal=assumeReal)
            globals_namespace[labelLoc] = sym

            if enforce_real and assumeReal is True and enforced_real_dict is not None:
                enforced_real_dict[sym.conjugate()] = sym

            vf_instance = vector_field_class(
                coeff_dict={(0, 1, labelLoc): 1},
                dgcvType="standard",
                variable_spaces={labelLoc: (sym,)},
            )
            df_instance = differential_form_class(
                coeff_dict={(0, 0, labelLoc): 1},
                dgcvType="standard",
                variable_spaces={labelLoc: (sym,)},
            )
            globals_namespace[f"D_{labelLoc}"] = vf_instance
            globals_namespace[f"d_{labelLoc}"] = df_instance

            if not _calledFromCVP == passkey:
                variable_registry["standard_variable_systems"][labelLoc] = {
                    "family_type": "single",
                    "family_values": (sym,),
                    "family_names": (labelLoc,),
                    "differential_system": True,
                    "tempVar": None,
                    "initial_index": None,
                    "variable_relatives": {
                        labelLoc: {
                            "VFClass": vf_instance,
                            "DFClass": df_instance,
                            "assumeReal": assumeReal,
                            "system_index": 0,
                        }
                    },
                }
                variable_registry["_labels"][labelLoc] = {
                    "path": ("standard_variable_systems", labelLoc),
                    "children": {f"D_{labelLoc}", f"d_{labelLoc}"},
                }
                paths = variable_registry.get("paths", None)
                if paths is not None:
                    paths[labelLoc] = {
                        "kind": "coordinate",
                        "path": (
                            "standard_variable_systems",
                            labelLoc,
                            "family_values",
                            0,
                        ),
                    }

        elif isinstance(number_of_variables, Integral) and number_of_variables >= 0:
            _tuple_flag = True
            lengthLoc = number_of_variables
            var_names = [
                f"{labelLoc}{i}" for i in range(initialIndex, lengthLoc + initialIndex)
            ]
            vars = [symbol(name, assumeReal=assumeReal) for name in var_names]
            globals_namespace.update(dict(zip(var_names, vars)))
            globals_namespace[labelLoc] = tuple(vars)

            if enforce_real and assumeReal is True and enforced_real_dict is not None:
                for v in vars:
                    enforced_real_dict[v.conjugate()] = v

            vtuple = tuple(vars)
            N = len(vtuple)

            vf_instances = [
                vector_field_class(
                    coeff_dict={(j, 1, labelLoc): 1},
                    dgcvType="standard",
                    variable_spaces={labelLoc: vtuple},
                )
                for j in range(N)
            ]
            df_instances = [
                differential_form_class(
                    coeff_dict={(j, 0, labelLoc): 1},
                    dgcvType="standard",
                    variable_spaces={labelLoc: vtuple},
                )
                for j in range(N)
            ]

            globals_namespace.update(
                dict(zip([f"D_{vn}" for vn in var_names], vf_instances))
            )
            globals_namespace.update(
                dict(zip([f"d_{vn}" for vn in var_names], df_instances))
            )

            if not _calledFromCVP == passkey:
                variable_registry["standard_variable_systems"][labelLoc] = {
                    "family_type": "tuple",
                    "family_values": vtuple,
                    "family_names": tuple(var_names),
                    "differential_system": True,
                    "tempVar": None,
                    "initial_index": initialIndex,
                    "variable_relatives": {
                        var_name: {
                            "VFClass": vf_instances[i],
                            "DFClass": df_instances[i],
                            "assumeReal": assumeReal,
                            "system_index": i,
                        }
                        for i, var_name in enumerate(var_names)
                    },
                }
                variable_registry["_labels"][labelLoc] = {
                    "path": ("standard_variable_systems", labelLoc),
                    "children": set(
                        var_names
                        + [f"D_{v}" for v in var_names]
                        + [f"d_{v}" for v in var_names]
                    ),
                }
                paths = variable_registry.get("paths", None)
                if paths is not None:
                    base_vals = ("standard_variable_systems", labelLoc, "family_values")
                    for i, vn in enumerate(var_names):
                        paths[labelLoc] = {
                            "kind": "standard_variable_system",
                            "path": ("standard_variable_systems", labelLoc),
                        }
                        paths[vn] = {
                            "kind": "coordinate",
                            "path": base_vals + (i,),
                        }

        else:
            raise ValueError(
                "varWithVF expected `number_of_variables` (optional) to be a non-negative integer, if provided."
            )

        if rco is not None:
            if _tuple_flag:
                rco.append(globals_namespace[labelLoc])
            else:
                rco.append((globals_namespace[labelLoc],))

    return rco if rco is not None else None


def complexVarProc(
    holom_label,
    real_label,
    im_label,
    number_of_variables=None,
    initialIndex=1,
    multiindex_shape=None,
    default_var_format="complex",
    remove_guardrails=None,
    return_created_object=True,
):
    """
    Initializes a complex variable system, linking a holomorphic variable with its real and imaginary parts and
    a symbolic representative of its complex conjugate.
    """
    if default_var_format not in ("real", "complex"):
        if default_var_format is not None:
            warnings.warn(
                "`default_var_format` was set to an unsupported value, so it was reset to the default 'complex'."
            )
        default_var_format = "complex"

    variable_registry = get_variable_registry()
    conv = variable_registry["conversion_dictionaries"]
    find_parents = conv["find_parents"]
    # protected_vars = variable_registry["protected_variables"]

    conj_updates = {}
    holToReal_updates = {}
    realToSym_updates = {}
    symToHol_updates = {}
    symToReal_updates = {}
    realToHol_updates = {}
    real_part_updates = {}
    im_part_updates = {}
    complex_system_updates = {}

    # For tuple systems, store data for deferred differential object creation
    # Each entry: (labelLoc1, var_names1, var_namesBAR, var_names2, var_names3, lengthLoc)
    tuple_system_data = []

    def _register_complex_paths(system_label: str) -> None:
        paths = variable_registry.get("paths", None)
        if paths is None:
            return

        sys = variable_registry.get("complex_variable_systems", {}).get(system_label)
        if not isinstance(sys, dict):
            return

        paths[system_label] = {
            "kind": "complex_variable_system",
            "path": ("complex_variable_systems", system_label),
        }

        houses = sys.get("family_houses")
        if isinstance(houses, (tuple, list)):
            base = ("complex_variable_systems", system_label, "family_houses")
            for i, house in enumerate(houses):
                if isinstance(house, str):
                    paths[house] = {
                        "kind": "complex_variable_house",
                        "path": base + (i,),
                    }

        rel = sys.get("variable_relatives")
        if isinstance(rel, dict):
            base = ("complex_variable_systems", system_label, "variable_relatives")
            for member_label in rel.keys():
                if isinstance(member_label, str):
                    paths[member_label] = {
                        "kind": "coordinate",
                        "path": base + (member_label, "variable_value"),
                    }

    def validate_variable_labels(*labels, remove_guardrails=False):
        reformatted_labels = []
        seen_labels = set()
        for label in labels:
            reformatted_label = validate_label(
                label, remove_guardrails=remove_guardrails
            )
            if reformatted_label in seen_labels:
                raise ValueError(
                    f"Duplicate label found while formatting '{labels}'. Each label must be unique."
                )
            seen_labels.add(reformatted_label)
            reformatted_labels.append(reformatted_label)
        return tuple(reformatted_labels)

    if isinstance(holom_label, str):
        holom_label = [holom_label]
        real_label = [real_label]
        im_label = [im_label]

    rco = [] if return_created_object is True else None

    pref = get_dgcv_settings_registry().get("conjugation_prefix", "BAR")
    for j in range(len(holom_label)):
        if remove_guardrails:
            labelLoc1 = holom_label[j]
            labelLoc2 = real_label[j]
            labelLoc3 = im_label[j]
        else:
            labelLoc1, labelLoc2, labelLoc3 = validate_variable_labels(
                holom_label[j], real_label[j], im_label[j]
            )
        labelLocBAR = f"{pref}{labelLoc1}"

        clearVar(labelLoc1, report=False)
        clearVar(labelLoc2, report=False)
        clearVar(labelLoc3, report=False)
        clearVar(labelLocBAR, report=False)

        def _valid_multiindex_shape(ms):
            return isinstance(ms, (list, tuple)) and all(
                isinstance(n, Integral) and n > 0 for n in ms
            )

        def _multiindex_indices(ms):
            return list(carProd(*[range(initialIndex, initialIndex + n) for n in ms]))

        def _multiindex_names(base, idxs):
            return [f"{base}_{'_'.join(map(str, idx))}" for idx in idxs]

        # protected_vars.update({labelLoc2, labelLoc3})

        # Multi-index System Case
        if number_of_variables is None and _valid_multiindex_shape(multiindex_shape):
            idxs = _multiindex_indices(multiindex_shape)

            variableProcedure(
                labelLoc1,
                initialIndex=initialIndex,
                multiindex_shape=multiindex_shape,
                _doNotUpdateVar=retrieve_passkey(),
                _calledFromCVP=retrieve_passkey(),
            )
            variableProcedure(
                labelLocBAR,
                initialIndex=initialIndex,
                multiindex_shape=multiindex_shape,
                _doNotUpdateVar=retrieve_passkey(),
                _calledFromCVP=retrieve_passkey(),
            )
            variableProcedure(
                labelLoc2,
                initialIndex=initialIndex,
                multiindex_shape=multiindex_shape,
                _doNotUpdateVar=retrieve_passkey(),
                assumeReal=True,
                _calledFromCVP=retrieve_passkey(),
            )
            variableProcedure(
                labelLoc3,
                initialIndex=initialIndex,
                multiindex_shape=multiindex_shape,
                _doNotUpdateVar=retrieve_passkey(),
                assumeReal=True,
                _calledFromCVP=retrieve_passkey(),
            )

            var_arr1 = _cached_caller_globals[labelLoc1]
            var_arrBAR = _cached_caller_globals[labelLocBAR]
            var_arr2 = _cached_caller_globals[labelLoc2]
            var_arr3 = _cached_caller_globals[labelLoc3]
            if rco is not None:
                rco += [var_arr1, var_arrBAR, var_arr2, var_arr3]

            var_str1 = tuple(_multiindex_names(labelLoc1, idxs))
            var_strBAR = tuple(_multiindex_names(labelLocBAR, idxs))
            var_str2 = tuple(_multiindex_names(labelLoc2, idxs))
            var_str3 = tuple(_multiindex_names(labelLoc3, idxs))

            complex_system_updates[labelLoc1] = {
                "family_type": "multi_index",
                "family_shape": tuple(multiindex_shape),
                "family_names": (var_str1, var_strBAR, var_str2, var_str3),
                "family_values": (var_arr1, var_arrBAR, var_arr2, var_arr3),
                "family_houses": (labelLoc1, labelLocBAR, labelLoc2, labelLoc3),
                "differential_system": True,
                "initial_index": initialIndex,
                "variable_relatives": {},
            }

            all_var_strs = list(var_str1 + var_strBAR + var_str2 + var_str3)
            variable_registry["_labels"][labelLoc1] = {
                "path": ("complex_variable_systems", labelLoc1),
                "children": set(
                    all_var_strs
                    + [f"D_{v}" for v in all_var_strs]
                    + [f"d_{v}" for v in all_var_strs]
                ),
            }

            flat1 = [_cached_caller_globals[nm] for nm in var_str1]
            flatBAR = [_cached_caller_globals[nm] for nm in var_strBAR]
            flat2 = [_cached_caller_globals[nm] for nm in var_str2]
            flat3 = [_cached_caller_globals[nm] for nm in var_str3]

            totalVarListLoc = list(zip(flat1, flatBAR, flat2, flat3))
            for comp_var, bar_comp_var, real_var, imag_var in totalVarListLoc:
                find_parents[real_var] = (comp_var, bar_comp_var)
                find_parents[imag_var] = (comp_var, bar_comp_var)

                conj_updates[comp_var] = bar_comp_var
                conj_updates[bar_comp_var] = comp_var
                holToReal_updates[comp_var] = real_var + imag_unit() * imag_var
                realToSym_updates[real_var] = half * (comp_var + bar_comp_var)
                realToSym_updates[imag_var] = (
                    -imag_unit() * half * (comp_var - bar_comp_var)
                )
                symToHol_updates[bar_comp_var] = conjugate(comp_var)
                symToReal_updates[comp_var] = real_var + imag_unit() * imag_var
                symToReal_updates[bar_comp_var] = real_var - imag_unit() * imag_var
                realToHol_updates[real_var] = half * (comp_var + conjugate(comp_var))
                realToHol_updates[imag_var] = (
                    imag_unit() * half * (conjugate(comp_var) - comp_var)
                )
                real_part_updates[comp_var] = real_var
                real_part_updates[bar_comp_var] = real_var
                im_part_updates[comp_var] = imag_var
                im_part_updates[bar_comp_var] = -imag_var

            tuple_system_data.append(
                (
                    labelLoc1,
                    flat1,
                    flatBAR,
                    flat2,
                    flat3,
                    len(flat1),
                )
            )

            conv["conjugation"].update(conj_updates)
            conv["holToReal"].update(holToReal_updates)
            conv["realToSym"].update(realToSym_updates)
            conv["symToHol"].update(symToHol_updates)
            conv["symToReal"].update(symToReal_updates)
            conv["realToHol"].update(realToHol_updates)
            conv["real_part"].update(real_part_updates)
            conv["im_part"].update(im_part_updates)

        # Single Variable System
        elif number_of_variables is None:
            variableProcedure(
                labelLoc1,
                _doNotUpdateVar=retrieve_passkey(),
                _calledFromCVP=retrieve_passkey(),
            )
            variableProcedure(
                labelLocBAR,
                _doNotUpdateVar=retrieve_passkey(),
                _calledFromCVP=retrieve_passkey(),
            )
            variableProcedure(
                labelLoc2,
                _doNotUpdateVar=retrieve_passkey(),
                assumeReal=True,
                _calledFromCVP=retrieve_passkey(),
            )
            variableProcedure(
                labelLoc3,
                _doNotUpdateVar=retrieve_passkey(),
                assumeReal=True,
                _calledFromCVP=retrieve_passkey(),
            )

            # Retrieve created variables from the caller's globals.
            var_hol = _cached_caller_globals[labelLoc1]
            var_bar = _cached_caller_globals[labelLocBAR]
            var_real = _cached_caller_globals[labelLoc2]
            var_im = _cached_caller_globals[labelLoc3]
            if rco is not None:
                rco += [(var_hol,), (var_bar,), (var_real,), (var_im,)]

            # Accumulate conversion updates.
            conj_updates[var_hol] = var_bar
            conj_updates[var_bar] = var_hol
            holToReal_updates[var_hol] = var_real + imag_unit() * var_im
            realToSym_updates[var_real] = half * (var_hol + var_bar)
            realToSym_updates[var_im] = -imag_unit() * half * (var_hol - var_bar)
            symToHol_updates[var_bar] = conjugate(var_hol)
            symToReal_updates[var_hol] = var_real + imag_unit() * var_im
            symToReal_updates[var_bar] = var_real - imag_unit() * var_im
            realToHol_updates[var_real] = half * (var_hol + conjugate(var_hol))
            realToHol_updates[var_im] = (
                imag_unit() * half * (conjugate(var_hol) - var_hol)
            )
            real_part_updates[var_hol] = var_real
            real_part_updates[var_bar] = var_real
            im_part_updates[var_hol] = var_im
            im_part_updates[var_bar] = -var_im

            conv["conjugation"].update(conj_updates)
            conv["holToReal"].update(holToReal_updates)
            conv["realToSym"].update(realToSym_updates)
            conv["symToHol"].update(symToHol_updates)
            conv["symToReal"].update(symToReal_updates)
            conv["realToHol"].update(realToHol_updates)
            conv["real_part"].update(real_part_updates)
            conv["im_part"].update(im_part_updates)

            # Register this single variable system in the registry.
            variable_registry["complex_variable_systems"][labelLoc1] = {
                "family_type": "single",
                "family_names": (
                    (labelLoc1,),
                    (labelLocBAR,),
                    (labelLoc2,),
                    (labelLoc3,),
                ),
                "family_values": (var_hol, var_bar, var_real, var_im),
                "family_houses": (labelLoc1, labelLocBAR, labelLoc2, labelLoc3),
                "differential_system": True,
                "initial_index": None,
                "variable_relatives": {
                    labelLoc1: {
                        "complex_positioning": "holomorphic",
                        "complex_family": (var_hol, var_bar, var_real, var_im),
                        "variable_value": var_hol,
                        "VFClass": None,
                        "DFClass": None,
                        "assumeReal": None,
                        "system_index": 0,
                    },
                    labelLocBAR: {
                        "complex_positioning": "antiholomorphic",
                        "complex_family": (var_hol, var_bar, var_real, var_im),
                        "variable_value": var_bar,
                        "VFClass": None,
                        "DFClass": None,
                        "assumeReal": None,
                        "system_index": 1,
                    },
                    labelLoc2: {
                        "complex_positioning": "real",
                        "complex_family": (var_hol, var_bar, var_real, var_im),
                        "variable_value": var_real,
                        "VFClass": None,
                        "DFClass": None,
                        "assumeReal": True,
                        "system_index": 2,
                    },
                    labelLoc3: {
                        "complex_positioning": "imaginary",
                        "complex_family": (var_hol, var_bar, var_real, var_im),
                        "variable_value": var_im,
                        "VFClass": None,
                        "DFClass": None,
                        "assumeReal": True,
                        "system_index": 3,
                    },
                },
            }
            _register_complex_paths(labelLoc1)

            variable_registry["_labels"][labelLoc1] = {
                "path": ("complex_variable_systems", labelLoc1),
                "children": {
                    labelLocBAR,
                    labelLoc2,
                    labelLoc3,
                    f"D_{labelLoc1}",
                    f"d_{labelLoc1}",
                    f"D_{labelLocBAR}",
                    f"d_{labelLocBAR}",
                    f"D_{labelLoc2}",
                    f"d_{labelLoc2}",
                    f"D_{labelLoc3}",
                    f"d_{labelLoc3}",
                },
            }

            def create_differential_objects_single(
                var_hol, var_bar, var_real, var_im, default_var_format
            ):
                vs = (var_hol, var_bar, var_real, var_im)
                sys = labelLoc1  # the complex system label
                if default_var_format == "real":
                    # Differential objects using the real/imaginary parts.
                    inh_dict = {"_validated_format": "real"}
                    vf_instance_hol = vector_field_class(
                        coeff_dict={(2, 1, sys): half, (3, 1, sys): -imag_unit() / 2},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    vf_instance_aHol = vector_field_class(
                        coeff_dict={(2, 1, sys): half, (3, 1, sys): imag_unit() / 2},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    vf_instance_real = vector_field_class(
                        coeff_dict={(2, 1, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    vf_instance_im = vector_field_class(
                        coeff_dict={(3, 1, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    df_instance_hol = differential_form_class(
                        coeff_dict={(2, 0, sys): 1, (3, 0, sys): imag_unit()},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    df_instance_aHol = differential_form_class(
                        coeff_dict={(2, 0, sys): 1, (3, 0, sys): -imag_unit()},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    df_instance_real = differential_form_class(
                        coeff_dict={(2, 0, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    df_instance_im = differential_form_class(
                        coeff_dict={(3, 0, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                else:  # default_var_format == "complex"
                    inh_dict = {"_validated_format": "complex"}
                    vf_instance_hol = vector_field_class(
                        coeff_dict={(0, 1, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    vf_instance_aHol = vector_field_class(
                        coeff_dict={(1, 1, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    vf_instance_real = vector_field_class(
                        coeff_dict={(0, 1, sys): 1, (1, 1, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    vf_instance_im = vector_field_class(
                        coeff_dict={
                            (0, 1, sys): imag_unit(),
                            (1, 1, sys): -imag_unit(),
                        },
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    df_instance_hol = differential_form_class(
                        coeff_dict={(0, 0, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    df_instance_aHol = differential_form_class(
                        coeff_dict={(1, 0, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    df_instance_real = differential_form_class(
                        coeff_dict={(0, 0, sys): half, (1, 0, sys): half},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    df_instance_im = differential_form_class(
                        coeff_dict={
                            (0, 0, sys): -imag_unit() / 2,
                            (1, 0, sys): imag_unit() / 2,
                        },
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                return (
                    vf_instance_hol,
                    vf_instance_aHol,
                    vf_instance_real,
                    vf_instance_im,
                    df_instance_hol,
                    df_instance_aHol,
                    df_instance_real,
                    df_instance_im,
                )

            # Create differential objects for single systems.
            (
                vf_instance_hol,
                vf_instance_aHol,
                vf_instance_real,
                vf_instance_im,
                df_instance_hol,
                df_instance_aHol,
                df_instance_real,
                df_instance_im,
            ) = create_differential_objects_single(
                var_hol, var_bar, var_real, var_im, default_var_format
            )

            # Register the differential objects
            _cached_caller_globals.update(
                {
                    f"D_{labelLoc1}": vf_instance_hol,
                    f"D_{labelLocBAR}": vf_instance_aHol,
                    f"d_{labelLoc1}": df_instance_hol,
                    f"d_{labelLocBAR}": df_instance_aHol,
                    f"D_{labelLoc2}": vf_instance_real,
                    f"D_{labelLoc3}": vf_instance_im,
                    f"d_{labelLoc2}": df_instance_real,
                    f"d_{labelLoc3}": df_instance_im,
                }
            )

            # Update find_parents.
            find_parents[var_real] = (var_hol, var_bar)
            find_parents[var_im] = (var_hol, var_bar)

            # Register VFClass and DFClass
            address = variable_registry["complex_variable_systems"][labelLoc1][
                "variable_relatives"
            ]
            address[labelLoc1] |= {
                "VFClass": _cached_caller_globals[f"D_{labelLoc1}"],
                "DFClass": _cached_caller_globals[f"d_{labelLoc1}"],
            }
            address[labelLocBAR] |= {
                "VFClass": _cached_caller_globals[f"D_{labelLocBAR}"],
                "DFClass": _cached_caller_globals[f"d_{labelLocBAR}"],
            }
            address[labelLoc2] |= {
                "VFClass": _cached_caller_globals[f"D_{labelLoc2}"],
                "DFClass": _cached_caller_globals[f"d_{labelLoc2}"],
            }
            address[labelLoc3] |= {
                "VFClass": _cached_caller_globals[f"D_{labelLoc3}"],
                "DFClass": _cached_caller_globals[f"d_{labelLoc3}"],
            }

        # Tuple System Case
        elif isinstance(number_of_variables, Number) and number_of_variables > 0:
            lengthLoc = number_of_variables
            variableProcedure(
                labelLoc1,
                lengthLoc,
                initialIndex=initialIndex,
                _doNotUpdateVar=retrieve_passkey(),
                _calledFromCVP=retrieve_passkey(),
            )
            variableProcedure(
                labelLocBAR,
                lengthLoc,
                initialIndex=initialIndex,
                _doNotUpdateVar=retrieve_passkey(),
                _calledFromCVP=retrieve_passkey(),
            )
            variableProcedure(
                labelLoc2,
                lengthLoc,
                initialIndex=initialIndex,
                _doNotUpdateVar=retrieve_passkey(),
                assumeReal=True,
                _calledFromCVP=retrieve_passkey(),
            )
            variableProcedure(
                labelLoc3,
                lengthLoc,
                initialIndex=initialIndex,
                _doNotUpdateVar=retrieve_passkey(),
                assumeReal=True,
                _calledFromCVP=retrieve_passkey(),
            )

            # Retrieve lists of created variables from the caller's globals.
            var_names1 = _cached_caller_globals[labelLoc1]
            var_namesBAR = _cached_caller_globals[labelLocBAR]
            var_names2 = _cached_caller_globals[labelLoc2]
            var_names3 = _cached_caller_globals[labelLoc3]
            if rco is not None:
                rco += [var_names1, var_namesBAR, var_names2, var_names3]

            # Build string labels for registry.
            var_str1 = tuple(
                f"{labelLoc1}{i}" for i in range(initialIndex, lengthLoc + initialIndex)
            )
            var_strBAR = tuple(
                f"{labelLocBAR}{i}"
                for i in range(initialIndex, lengthLoc + initialIndex)
            )
            var_str2 = tuple(
                f"{labelLoc2}{i}" for i in range(initialIndex, lengthLoc + initialIndex)
            )
            var_str3 = tuple(
                f"{labelLoc3}{i}" for i in range(initialIndex, lengthLoc + initialIndex)
            )

            # Register the tuple system (the differential objects will be added in Phase 2).
            complex_system_updates[labelLoc1] = {
                "family_type": "tuple",
                "family_names": (var_str1, var_strBAR, var_str2, var_str3),
                "family_values": (var_names1, var_namesBAR, var_names2, var_names3),
                "family_houses": (labelLoc1, labelLocBAR, labelLoc2, labelLoc3),
                "differential_system": True,
                "initial_index": initialIndex,
                "variable_relatives": {},
            }

            all_var_strs = list(var_str1 + var_strBAR + var_str2 + var_str3)
            variable_registry["_labels"][labelLoc1] = {
                "path": ("complex_variable_systems", labelLoc1),
                "children": set(
                    all_var_strs
                    + [f"D_{v}" for v in all_var_strs]
                    + [f"d_{v}" for v in all_var_strs]
                ),
            }

            # Accumulate conversion updates for each tuple element.
            totalVarListLoc = list(
                zip(var_names1, var_namesBAR, var_names2, var_names3)
            )
            for idx, (comp_var, bar_comp_var, real_var, imag_var) in enumerate(
                totalVarListLoc
            ):
                find_parents[real_var] = (comp_var, bar_comp_var)
                find_parents[imag_var] = (comp_var, bar_comp_var)

                conj_updates[comp_var] = bar_comp_var
                conj_updates[bar_comp_var] = comp_var
                holToReal_updates[comp_var] = real_var + imag_unit() * imag_var
                realToSym_updates[real_var] = half * (comp_var + bar_comp_var)
                realToSym_updates[imag_var] = (
                    -imag_unit() * half * (comp_var - bar_comp_var)
                )
                symToHol_updates[bar_comp_var] = conjugate(comp_var)
                symToReal_updates[comp_var] = real_var + imag_unit() * imag_var
                symToReal_updates[bar_comp_var] = real_var - imag_unit() * imag_var
                realToHol_updates[real_var] = half * (comp_var + conjugate(comp_var))
                realToHol_updates[imag_var] = (
                    imag_unit() * half * (conjugate(comp_var) - comp_var)
                )
                real_part_updates[comp_var] = real_var
                real_part_updates[bar_comp_var] = real_var
                im_part_updates[comp_var] = imag_var
                im_part_updates[bar_comp_var] = -imag_var

            # Save tuple system info for Phase 2.
            tuple_system_data.append(
                (labelLoc1, var_names1, var_namesBAR, var_names2, var_names3, lengthLoc)
            )

            # Batch update the conversion dictionaries.
            conv["conjugation"].update(conj_updates)
            conv["holToReal"].update(holToReal_updates)
            conv["realToSym"].update(realToSym_updates)
            conv["symToHol"].update(symToHol_updates)
            conv["symToReal"].update(symToReal_updates)
            conv["realToHol"].update(realToHol_updates)
            conv["real_part"].update(real_part_updates)
            conv["im_part"].update(im_part_updates)
        else:
            raise ValueError(
                "variableProcedure expected its second argument number_of_variables (optional) to be a positive integer, if provided."
            )

    variable_registry["complex_variable_systems"].update(complex_system_updates)

    # Create differential objects for tuple systems.
    for (
        labelLoc1,
        var_names1,
        var_namesBAR,
        var_names2,
        var_names3,
        lengthLoc,
    ) in tuple_system_data:
        relatives = variable_registry["complex_variable_systems"][labelLoc1][
            "variable_relatives"
        ]
        totalVarListLoc = list(zip(var_names1, var_namesBAR, var_names2, var_names3))

        # Batch conversion dictionary updates
        conj_updates_batch = {comp: anti for comp, anti, _, _ in totalVarListLoc}
        conj_updates_batch.update({anti: comp for comp, anti, _, _ in totalVarListLoc})

        holToReal_updates_batch = {
            comp: real + imag_unit() * imag for comp, _, real, imag in totalVarListLoc
        }

        realToSym_updates_batch = {}
        for comp, anti, real, imag in totalVarListLoc:
            realToSym_updates_batch[real] = half * (comp + anti)
            realToSym_updates_batch[imag] = -imag_unit() * half * (comp - anti)

        symToHol_updates_batch = {
            anti: conjugate(comp) for comp, anti, _, _ in totalVarListLoc
        }

        symToReal_updates_batch = {}
        for comp, anti, real, imag in totalVarListLoc:
            symToReal_updates_batch[comp] = real + imag_unit() * imag
            symToReal_updates_batch[anti] = real - imag_unit() * imag

        realToHol_updates_batch = {}
        for comp, _, real, _ in totalVarListLoc:
            realToHol_updates_batch[real] = half * (comp + conjugate(comp))
        for comp, _, _, imag in totalVarListLoc:
            realToHol_updates_batch[imag] = (
                imag_unit() * half * (conjugate(comp) - comp)
            )

        real_part_updates_batch = {}
        im_part_updates_batch = {}
        for comp, anti, real, imag in totalVarListLoc:
            real_part_updates_batch[comp] = real
            real_part_updates_batch[anti] = real
            im_part_updates_batch[comp] = imag
            im_part_updates_batch[anti] = -imag

        # Create differential objects
        for idx, (comp_var, bar_comp_var, real_var, imag_var) in enumerate(
            totalVarListLoc
        ):
            sys = labelLoc1
            N = lengthLoc

            vs = (
                tuple(var_names1)
                + tuple(var_namesBAR)
                + tuple(var_names2)
                + tuple(var_names3)
            )
            if default_var_format == "real":
                i_real = idx + 2 * N
                i_im = idx + 3 * N

                inh_dict = {"_validated_format": "real"}
                D_comp = vector_field_class(
                    coeff_dict={
                        (i_real, 1, sys): half,
                        (i_im, 1, sys): -imag_unit() / 2,
                    },
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                D_bar_comp = vector_field_class(
                    coeff_dict={
                        (i_real, 1, sys): half,
                        (i_im, 1, sys): imag_unit() / 2,
                    },
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                D_real = vector_field_class(
                    coeff_dict={(i_real, 1, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                D_im = vector_field_class(
                    coeff_dict={(i_im, 1, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                d_comp = differential_form_class(
                    coeff_dict={(i_real, 0, sys): 1, (i_im, 0, sys): imag_unit()},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                d_bar_comp = differential_form_class(
                    coeff_dict={(i_real, 0, sys): 1, (i_im, 0, sys): -imag_unit()},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                d_real = differential_form_class(
                    coeff_dict={(i_real, 0, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                d_im = differential_form_class(
                    coeff_dict={(i_im, 0, sys): imag_unit()},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
            else:  # default_var_format == "complex"
                sys = labelLoc1
                N = lengthLoc

                vs = (
                    tuple(var_names1)
                    + tuple(var_namesBAR)
                    + tuple(var_names2)
                    + tuple(var_names3)
                )

                i_hol = idx
                i_anti = idx + N
                inh_dict = {"_validated_format": "complex"}

                D_comp = vector_field_class(
                    coeff_dict={(i_hol, 1, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                D_bar_comp = vector_field_class(
                    coeff_dict={(i_anti, 1, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                D_real = vector_field_class(
                    coeff_dict={
                        (i_hol, 1, sys): 1,
                        (i_anti, 1, sys): 1,
                    },
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                D_im = vector_field_class(
                    coeff_dict={
                        (i_hol, 1, sys): imag_unit(),
                        (i_anti, 1, sys): -imag_unit(),
                    },
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                d_comp = differential_form_class(
                    coeff_dict={(i_hol, 0, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                d_bar_comp = differential_form_class(
                    coeff_dict={(i_anti, 0, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                d_real = differential_form_class(
                    coeff_dict={(i_hol, 0, sys): half, (i_anti, 0, sys): half},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                d_im = differential_form_class(
                    coeff_dict={
                        (i_hol, 0, sys): -imag_unit() / 2,
                        (i_anti, 0, sys): imag_unit() / 2,
                    },
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )

            # Register the differential objects in VMF
            _cached_caller_globals[f"D_{comp_var}"] = D_comp
            _cached_caller_globals[f"D_{bar_comp_var}"] = D_bar_comp
            _cached_caller_globals[f"d_{comp_var}"] = d_comp
            _cached_caller_globals[f"d_{bar_comp_var}"] = d_bar_comp
            _cached_caller_globals[f"D_{real_var}"] = D_real
            _cached_caller_globals[f"D_{imag_var}"] = D_im
            _cached_caller_globals[f"d_{real_var}"] = d_real
            _cached_caller_globals[f"d_{imag_var}"] = d_im

            # Update variable_relatives in VMF
            relatives[str(comp_var)] = {
                "complex_positioning": "holomorphic",
                "complex_family": (comp_var, bar_comp_var, real_var, imag_var),
                "variable_value": comp_var,
                "VFClass": D_comp,
                "DFClass": d_comp,
                "assumeReal": None,
                "system_index": idx,
            }
            relatives[str(bar_comp_var)] = {
                "complex_positioning": "antiholomorphic",
                "complex_family": (comp_var, bar_comp_var, real_var, imag_var),
                "variable_value": bar_comp_var,
                "VFClass": D_bar_comp,
                "DFClass": d_bar_comp,
                "assumeReal": None,
                "system_index": idx + N,
            }
            relatives[str(real_var)] = {
                "complex_positioning": "real",
                "complex_family": (comp_var, bar_comp_var, real_var, imag_var),
                "variable_value": real_var,
                "VFClass": _cached_caller_globals[f"D_{real_var}"],
                "DFClass": _cached_caller_globals[f"d_{real_var}"],
                "assumeReal": True,
                "system_index": idx + 2 * N,
            }
            relatives[str(imag_var)] = {
                "complex_positioning": "imaginary",
                "complex_family": (comp_var, bar_comp_var, real_var, imag_var),
                "variable_value": imag_var,
                "VFClass": _cached_caller_globals[f"D_{imag_var}"],
                "DFClass": _cached_caller_globals[f"d_{imag_var}"],
                "assumeReal": True,
                "system_index": idx + 3 * N,
            }
            _register_complex_paths(labelLoc1)
    if tuple_system_data:
        conv["conjugation"].update(conj_updates_batch)
        conv["holToReal"].update(holToReal_updates_batch)
        conv["realToSym"].update(realToSym_updates_batch)
        conv["symToHol"].update(symToHol_updates_batch)
        conv["symToReal"].update(symToReal_updates_batch)
        conv["realToHol"].update(realToHol_updates_batch)
        conv["real_part"].update(real_part_updates_batch)
        conv["im_part"].update(im_part_updates_batch)

    rv = rco if return_created_object is True else None
    return rv


def temporaryVariables(
    variable_label: str = None,
    number_of_variables=None,
    initialIndex=1,
    multiindex_shape=None,
    assumeReal=None,
    return_created_object=True,
    register_in_vmf: bool = False,
    remove_guardrails=None,
):
    if isinstance(variable_label, Integral) and number_of_variables is None:
        variable_label, number_of_variables = None, variable_label
    if not isinstance(variable_label, str):
        variable_label = create_key(
            "tvar", avoid_caller_globals=register_in_vmf, key_length=6
        )
    if register_in_vmf:
        newObj = createVariables(
            variable_label=variable_label,
            number_of_variables=number_of_variables,
            initialIndex=initialIndex,
            multiindex_shape=multiindex_shape,
            assumeReal=assumeReal,
            return_created_object=return_created_object,
            temporary_variables=True,
            remove_guardrails=remove_guardrails,
        )
        if isinstance(newObj, (list, tuple)) and len(newObj) == 1:
            newObj = newObj[0]
        return newObj
    newObj = variableProcedure(
        variables_label=variable_label,
        number_of_variables=number_of_variables,
        initialIndex=initialIndex,
        multiindex_shape=multiindex_shape,
        assumeReal=assumeReal,
        return_created_object=return_created_object,
        _tempVar=retrieve_passkey(),
        remove_guardrails=remove_guardrails,
    )
    if isinstance(newObj, (list, tuple)) and len(newObj) == 1:
        newObj = newObj[0]
    return newObj


def _format_complex_coordinates(
    coordinate_tuple, default_var_format="complex", pass_error_report=None
):
    """
    Format var. lists consisting of variables within dgcv complex variable systems, formatting as real or holomorphic and completeing the basis as needed (i.e., adding BARz if only z is present, adding y if only x, etc.)
    """
    vr = get_variable_registry()
    exaustList = list(coordinate_tuple)
    newList1 = []
    newList2 = []
    try:
        for var in coordinate_tuple:
            if var in exaustList:
                varStr = str(var)
                for parent in vr["complex_variable_systems"]:
                    if (
                        varStr
                        in vr["complex_variable_systems"][parent]["variable_relatives"]
                    ):
                        foundVars = vr["complex_variable_systems"][parent][
                            "variable_relatives"
                        ][varStr]["complex_family"]
                        if default_var_format == "complex":
                            newList1 = newList1 + [foundVars[0]]
                            newList2 = newList2 + [foundVars[1]]
                        else:
                            newList1 = newList1 + [foundVars[2]]
                            newList2 = newList2 + [foundVars[3]]
                        for j in foundVars:
                            if j in exaustList:
                                exaustList.remove(j)
    except KeyError:
        if pass_error_report == retrieve_passkey():
            return "At least one element in the given variable list is not registered as part of a complex variable system in the dgcv variable management framework."
    return tuple(newList1 + newList2)


# -----------------------------------------------------------------------------
# variable format conversion
# -----------------------------------------------------------------------------
def complex_struct_op(vf):
    if not query_dgcv_categories(vf, {"vector_field"}):
        raise TypeError("complex_struct_op expects a vector_field instance.")

    imu = imag_unit()
    vst = vf.variable_spaces_types

    new_cd = {}

    for k, c in vf.coeff_dict.items():
        if not c:
            continue

        if k == tuple():
            new_cd[tuple()] = new_cd.get(tuple(), 0) + c
            continue

        d = len(k) // 3
        idxs = list(k[:d])
        valence_tuple = k[d : 2 * d]
        syslbls = k[2 * d :]

        for slot in range(d):
            sys = syslbls[slot]
            sys_data = vst.get(sys)

            if sys_data is None or sys_data.get("type") != "complex":
                continue

            b0, b1, b2 = sys_data["breaks"]
            idx = idxs[slot]

            if idx < b0:  # holo
                c = imu * c
            elif idx < b1:  # anti
                c = -imu * c
            elif idx < b2:  # real
                idxs[slot] = idx + (b2 - b1)  # -> imag
            else:  # imag
                idxs[slot] = idx - (b2 - b1)  # -> real
                c = -c

        nk = tuple(idxs) + valence_tuple + syslbls
        new_cd[nk] = new_cd.get(nk, 0) + c

    if not new_cd:
        new_cd = {tuple(): 0}

    return vf.__class__(
        coeff_dict=new_cd,
        data_shape=getattr(vf, "data_shape", "all"),
        dgcvType=getattr(vf, "dgcvType", "standard"),
        _simplifyKW=getattr(vf, "_simplifyKW", None),
        variable_spaces=getattr(vf, "_variable_spaces", None),
    )


def conjugate_DGCV(expr):
    warnings.warn(
        "`conjugate_DGCV` has been deprecated as part of the shift toward standardized naming conventions in the `dgcv` library. "
        "It will be removed in 2026. Please use `conjugate_dgcv` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return conjugate_dgcv(expr)


def conjugate_dgcv(expr):
    return conjugate(expr)


def conj_with_real_coor(expr):
    return allToReal(expr).subs({imag_unit(): -imag_unit()})


def re_with_real_coor(expr):
    expr = allToReal(expr)
    s = simplify(half * (expr + conj_with_real_coor(expr)))
    return s


def im_with_real_coor(expr):
    expr = allToReal(expr)
    s = simplify(-imag_unit() * half * (expr - conj_with_real_coor(expr)))
    return s


def conj_with_hol_coor(expr):
    vr = get_variable_registry()
    subsDictA = vr["conversion_dictionaries"]["conjugation"]
    subsDict = subsDictA | {imag_unit(): -imag_unit()}
    return allToSym(expr).subs(subsDict, simultaneous=True)


def re_with_hol_coor(expr):
    expr = allToSym(expr)
    s = simplify(half * (expr + conj_with_hol_coor(expr)))
    return s


def im_with_hol_coor(expr):
    expr = allToSym(expr)
    s = simplify(-imag_unit() * half * (expr - conj_with_hol_coor(expr)))
    return s


# -----------------------------------------------------------------------------
# basic vector field and differential forms operations
# -----------------------------------------------------------------------------
def VF_coeffs_direct(vf, var_space, sparse=False):
    """
    Depricated: Use `VF_coeffs` instead.
    """
    if not query_dgcv_categories(vf, {"vector_field"}):
        raise TypeError("Expected first argument to be a vector field")

    if not isinstance(var_space, (list, tuple)):
        raise TypeError("Expected second argument to be a list or tuple of variables")

    # Evaluate the vector field on each element in var_space
    coeffs = [vf(var) for var in var_space]

    # Return sparse or full result
    if sparse:
        return [
            ((i,), coeffs[i])
            for i in range(len(coeffs))
            if not _scalar_is_zero(coeffs[i])
        ] or [((0,), 0)]
    return coeffs


def VF_coeffs(vf, var_list, sparse: bool = False):
    if not query_dgcv_categories(vf, {"vector_field"}):
        raise TypeError(f"VF_coeffs expects a vector_field, got {type(vf).__name__}.")

    if not isinstance(var_list, (list, tuple)):
        raise TypeError("VF_coeffs expects var_list to be a list or tuple.")

    cd = vf.coeff_dict
    vspaces = getattr(vf, "_variable_spaces", None)

    def _locate_in_vspaces(var):
        if not isinstance(vspaces, dict):
            return None, None
        for syslbl, vs in vspaces.items():
            try:
                j = vs.index(var)
            except Exception:
                continue
            return syslbl, j
        return None, None

    coeffs = []
    for var in var_list:
        info = vmf_lookup(var, relatives=True, system_index=True)
        rel = info.get("relatives") if isinstance(info, dict) else None

        syslbl = rel.get("system_label") if isinstance(rel, dict) else None
        j = info.get("system_index") if isinstance(rel, dict) else None

        if syslbl is None or j is None:
            syslbl, j = _locate_in_vspaces(var)

        if syslbl is None or j is None:
            raise KeyError(
                f"VF_coeffs could not locate a system_label/system_index for a variable {var} in var_list."
            )

        coeffs.append(cd.get((j, 1, syslbl), 0))

    if sparse:
        out = [((i,), c) for i, c in enumerate(coeffs) if not _scalar_is_zero(c)]
        return out or [((0,), 0)]

    return coeffs


def addVF(*vf_args):
    """
    Adds the given vector fields (i.e., vector_field_class instances).
    This is a superfluous function preserved for backward-compatibility.
    """
    return sum(vf_args)


def scaleVF(scalar, vector_field):
    """
    Scales the given vector field.
    This is a superfluous function preserved for backward-compatibility.
    """
    return scalar * vector_field


def VF_bracket(
    X: Any,
    Y: Any,
    *,
    doNotSimplify: bool = False,
    **_ignored,
):
    if not query_dgcv_categories(X, {"vector_field"}):
        raise TypeError(
            f"VF_bracket expects X to be a vector field, got {type(X).__name__}."
        )
    if not query_dgcv_categories(Y, {"vector_field"}):
        raise TypeError(
            f"VF_bracket expects Y to be a vector field, got {type(Y).__name__}."
        )

    cd_X = getattr(X, "coeff_dict", None)
    cd_Y = getattr(Y, "coeff_dict", None)
    if not isinstance(cd_X, dict) or not isinstance(cd_Y, dict):
        raise TypeError("VF_bracket expects vector fields with a coeff_dict attribute.")

    merged_vs = None
    mv = getattr(X, "_merged_variable_spaces", None)
    if callable(mv):
        merged_vs = mv(Y)
    else:
        merged_vs = getattr(X, "_variable_spaces", None)

    out: Dict[tuple, Any] = {}

    for k, v in cd_Y.items():
        if v:
            w = X(v)
            if w:
                out[k] = out.get(k, 0) + w

    for k, v in cd_X.items():
        if v:
            w = Y(v)
            if w:
                out[k] = out.get(k, 0) - w

    if not out:
        out = {tuple(): 0}
    else:
        if not doNotSimplify:
            for k, v in list(out.items()):
                if v:
                    out[k] = simplify(v)
        out = {k: v for k, v in out.items() if v}
        if not out:
            out = {tuple(): 0}

    out_cls = X.__class__ if (X.__class__ is Y.__class__) else vector_field_class

    return out_cls(
        coeff_dict=out,
        data_shape="all",
        dgcvType=getattr(X, "dgcvType", "standard"),
        _simplifyKW=getattr(X, "_simplifyKW", None),
        variable_spaces=merged_vs,
    )


def addDF(*args, doNotSimplify=False):
    """
    Adds the given vector fields (i.e., vector_field_class instances).
    This is a superfluous function preserved for backward-compatibility.
    """
    return sum(args)


def scaleDF(scalar, df):
    """
    Scales the given form
    This is a superfluous function preserved for backward-compatibility.
    """
    return scalar * df


def exteriorProduct(*args):
    if len(args) == 0:
        raise TypeError("exteriorProduct expects at least one argument.")

    for a in args:
        if not query_dgcv_categories(a, {"differential_form"}):
            raise TypeError(
                "exteriorProduct expects differential_form objects. "
                "Use dgcv creator functions to build differential forms."
            )

    out = args[0]
    for a in args[1:]:
        out = out._shape_product(a, kind="skew")
    return out


def _TFDictToNewBasis(data_dict, oldBasis, newBasis):
    data_list = list(data_dict.items())
    degree = len(data_list[0][0])
    try:
        dataDict = dict(
            [
                (tuple(newBasis.index(oldBasis[k]) for k in j[0]), j[1])
                for j in data_list
                if not _scalar_is_zero(j[1])
            ]
        )
    except ValueError as e:
        raise ValueError(
            f"`sparseKFormDataNewBasis` recieved bases for which an element in oldBasis {oldBasis} does not exist in newBasis {newBasis} whilst the sparseKFormData indicates this element crucial in the k-form's definition: {e}"
        )
    if not dataDict:
        dataDict = {(0,) * degree: 0}

    return dataDict


def sparseKFormDataNewBasis(sparseKFormData, oldBasis, newBasis):
    """
    Converts the indices of a k-form's sparse data representation from an old basis to a new basis.

    This function is primarily a helper for dgcv's exterior product calculus, allowing sparse data representing
    differential forms to be transformed into a new variable basis.

    Parameters
    ----------
    sparseKFormData : list of tuples
        A list where each tuple consists of the indices of variables in the old basis and the corresponding coefficient.

    oldBasis : list
        A list of variables representing the current basis.

    newBasis : list
        A list of variables representing the new basis, into which the k-form will be transformed.

    Returns
    -------
    list of tuples
        A transformed list where the indices are mapped from the old basis to the new basis, preserving the
        corresponding coefficients.

    Raises
    ------
    ValueError
        If a variable in the old basis is not found in the new basis.

    Examples
    --------
    >>> oldBasis = ['x', 'y', 'z']
    >>> newBasis = ['y', 'x', 'z']
    >>> sparseKFormData = [((0, 1), 1), ((1, 2), -1)]
    >>> sparseKFormDataNewBasis(sparseKFormData, oldBasis, newBasis)
    {(0, 2):−1, (1, 0):1}
    """
    if (
        not sparseKFormData
    ):  # Maybe safe to remove following October DFClassDataDict reformat!!!
        return {tuple(): 0}
    degree = len(sparseKFormData[0][0])
    try:
        dataDict = dict(
            [
                (tuple(newBasis.index(oldBasis[k]) for k in j[0]), j[1])
                for j in sparseKFormData
                if not _scalar_is_zero(j[1])
            ]
        )
    except ValueError as e:
        raise ValueError(
            f"`sparseKFormDataNewBasis` recieved bases for which an element in oldBasis {oldBasis} does not exist in newBasis {newBasis} whilst the sparseKFormData indicates this element crucial in the k-form's definition: {e}"
        )
    if not dataDict:
        dataDict = {(0,) * degree: 0}
    return dataDict


def wedge(*tfs):
    types = {get_dgcv_category(tf) for tf in tfs if not check_dgcv_scalar(tf)}
    if types != {"tensor_field"} and not all(
        dgcv_type in {"algebra_element", "vector_space_element", "tensorProduct"}
        for dgcv_type in types
    ):
        raise TypeError(
            "`wedge` only operates on scalars and dgcv tensor field, tensor algebra, and vector space classes. The field and vector space classes moreover cannot mix."
        )
    if len(tfs) == 0:
        return
    if len(tfs) == 1:
        return tfs[0]
    if len(tfs) > 2:
        return wedge(wedge(tfs[0], tfs[1]), *tfs[2:])
    tf1, tf2 = tfs
    if check_dgcv_scalar(tf1) or check_dgcv_scalar(tf2):
        return tf1 * tf2
    elif get_dgcv_category(tf1) == "tensor_field":
        return tf1.skew_product(tf2)
    elif get_dgcv_category(tf1) == [
        "algebra_element",
        "vector_space_element",
        "tensorProduct",
    ]:
        return tf1 @ tf2 - tf2 @ tf1
    else:
        raise TypeError(
            "`wedge` only operates on dgcv tensor field, tensor algebra, and vector space classes."
        )


# -----------------------------------------------------------------------------
# tensor fields
# -----------------------------------------------------------------------------
def tensor_product(*args, doNotSimplify=False):
    """
    Computes the tensor product of tensor_field_class instances by dispatching the tensor_field_class product method.
    """
    if not all(
        check_dgcv_scalar(arg) or get_dgcv_category(arg) == "tensor_field"
        for arg in args
    ):
        bad_types = []
        for arg in args:
            if not (check_dgcv_scalar(arg) or isinstance(arg, tensorField)):
                bad_types += [type(arg)]
        bad_types = list(set(bad_types))
        bt_str = ", ".join(bad_types)
        raise Exception(
            f"Expected all arguments to be instances of tensorField or scalar-like objects, not type: {bt_str}"
        )
    if len(args) == 0:
        return
    if len(args) == 1:
        return args[0]
    if check_dgcv_scalar(args[0]):
        return args[0] * tensor_product(*args[1:])
    if len(args) == 2:
        return args[0].tensor_product(args[1])
    return tensor_product(args[0].tensor_product(args[1]), *args[2:])


# -----------------------------------------------------------------------------
# complex vector fields
# -----------------------------------------------------------------------------
def holVF_coeffs(
    vf: vector_field_class, arg2: list | tuple, doNotSimplify=False
) -> tuple:
    """
    Evaluates the vector field (i.e., vector_field_class instance) *arg1* on each holomorphic variable in *arg2*,
    and returns the result as a list of coefficients.

    The variables in *arg2* must be previously initialized via complexVarProc. The function returns the
    coefficients of the holomorphic part when the vector field is expressed in terms of holomorphic coordinate
    vector fields.

    Parameters:
    -----------
    arg1 : vector_field_class
        A vector field instance to evaluate on the holomorphic variables.
    arg2 : list or tuple
        A list or tuple of Symbol objects that were initialized as holomorphic variables via complexVarProc.
    doNotSimplify : bool, optional
        If True, the results are returned without simplification (default is False).

    Returns:
    --------
    list
        A list of symbolic expressions representing the coefficients in holomorphic coordinates.
    """
    if doNotSimplify:
        return [realToHol(vf(j)) for j in arg2]
    else:
        return [simplify(realToHol(vf(j))) for j in arg2]


def antiholVF_coeffs(
    vf: vector_field_class, arg2: list | tuple, doNotSimplify=False
) -> tuple:
    """
    Evaluates the vector field (i.e., vector_field_class instance) *arg1* on the conjugate of each holomorphic variable
    in *arg2*, and returns the result as a list of coefficients.

    The variables in *arg2* must be previously initialized via complexVarProc. The function returns the
    coefficients of the holomorphic part when the vector field is expressed in terms of holomorphic coordinate
    vector fields.

    Parameters:
    -----------
    arg1 : vector_field_class
        A vector field instance to evaluate on the holomorphic variables.
    arg2 : list or tuple
        A list or tuple of Symbol objects that were initialized as holomorphic variables via complexVarProc.
    doNotSimplify : bool, optional
        If True, the results are returned without simplification (default is False).

    Returns:
    --------
    list
        A list of symbolic expressions representing the coefficients in antiholomorphic coordinates.
    """
    if doNotSimplify:
        return [realToHol(vf(conjugate(j))) for j in arg2]
    else:
        return [simplify(realToHol(vf(conjugate(j)))) for j in arg2]


def complexVFC(
    vf: vector_field_class, arg2: list | tuple, doNotSimplify=False
) -> tuple:
    """
    Evaluates the vector field (i.e., vector_field_class instance) *arg1* on the holomorphic variables in *arg2*
    and their complex conjugates, returning the result as two lists of coefficients.

    The variables in *arg2* must be previously initialized via complexVarProc. The function returns the
    coefficients for both the holomorphic and antiholomorphic parts of the vector field when expressed in
    terms of the respective coordinate vector fields.

    Parameters:
    -----------
    arg1 : vector_field_class
        A vector field instance to evaluate on the holomorphic and antiholomorphic variables.
    arg2 : list or tuple
        A list or tuple of Symbol objects that were initialized as holomorphic variables via complexVarProc.
    doNotSimplify : bool, optional
        If True, the results are returned without simplification (default is False).

    Returns:
    --------
    tuple of two lists
        The first list contains the coefficients of the holomorphic part, and the second list contains
        the coefficients of the antiholomorphic part.

    """
    if query_dgcv_categories(vf, {"vector_field"}):
        hol_coeffs = holVF_coeffs(vf, arg2, doNotSimplify=doNotSimplify)
        antihol_coeffs = antiholVF_coeffs(vf, arg2, doNotSimplify=doNotSimplify)
        return hol_coeffs, antihol_coeffs
    else:
        raise Exception(
            "Expected first positional argument to be of type vector_field_class"
        )


def conjComplex(arg):
    warnings.warn("`conjComplex` has been deprecated. Use `conjugate_dgcv` instead")
    return _conjComplexVFDF(arg)


def _conjComplexVFDF(arg: tensor_field_class) -> tensor_field_class:
    """
    Computes the complex conjugate of a tensor_field_class.
    """

    if get_dgcv_category(arg) == "tensor_field":
        return conjugate(arg)
    else:
        raise Exception("Expected the input to be a dgcv tensor_field.")


def realPartOfVF(vf: vector_field_class, *args) -> vector_field_class:
    """
    Computes the real part of a complex vector field *vf*.
    """
    if query_dgcv_categories(vf, {"vector_field"}):
        return vf.real_part()
    else:
        raise Exception("Expected the input to be of type vector_field_class.")


# -----------------------------------------------------------------------------
# depricated
# -----------------------------------------------------------------------------
def TFClass(
    self,
    varSpace,
    coeff_dict,
    valence=None,
    data_shape="general",
    dgcvType="standard",
    _simplifyKW=None,
):
    warnings.warn(
        "`TFClass` is deprecated and has been replaced by the general `tensor_field_class`. The function label remains"
        "as a dispatcher to build `tensor_field_class` objects, but this may be removed in the future"
        "Please use `tensor_field_class` or `assemble_tensor_field` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    key = next(iter(coeff_dict))
    if valence is None and key is None:
        raise RuntimeError(
            "`STFClass` recieved invalid `data_dict`. `STFClass` is also deprecated. Use `tensor_field_class` instead"
        )
    val = valence if valence else key[len(key) // 2 :]  # old key format assumption
    return assemble_tensor_field(
        coordinate_space=varSpace,
        coefficient_dict=coeff_dict,
        valence=val,
        shape=data_shape,
        _simplifyKW=_simplifyKW,
    )


def tensorField(
    self,
    varSpace,
    coeff_dict,
    valence=None,
    data_shape="general",
    dgcvType="standard",
    _simplifyKW=None,
):
    warnings.warn(
        "`tensorField` is deprecated and has been replaced by the general `tensor_field_class`. The function label remains"
        "as a dispatcher to build `tensor_field_class` objects, but this may be removed in the future"
        "Please use `tensor_field_class` or `assemble_tensor_field` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    key = next(iter(coeff_dict))
    if valence is None and key is None:
        raise RuntimeError(
            "`STFClass` recieved invalid `data_dict`. `STFClass` is also deprecated. Use `tensor_field_class` instead"
        )
    val = valence if valence else key[len(key) // 3 : 2 * len(key) // 3]
    return assemble_tensor_field(
        coordinate_space=varSpace,
        coefficient_dict=coeff_dict,
        valence=val,
        shape=data_shape,
        _simplifyKW=_simplifyKW,
    )


def STFClass(
    self,
    varSpace,
    data_dict,
    degree,
    dgcvType="standard",
    _simplifyKW=None,
):
    warnings.warn(
        "`STFClass` is deprecated and has been replaced by the general `tensor_field_class`. The function label remains"
        "as a dispatcher to build `tensor_field_class` objects, but this may be removed in the future"
        "Please use `tensor_field_class` or `assemble_tensor_field` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    key = next(iter(data_dict))
    if key is None:
        raise RuntimeError(
            "`STFClass` recieved invalid `data_dict`. `STFClass` is also deprecated. Use `tensor_field_class` instead"
        )
    val = key[len(key) // 3 : 2 * len(key) // 3]  # old key format assumption
    return assemble_tensor_field(
        coordinate_space=varSpace,
        coefficient_dict=data_dict,
        valence=val,
        shape="symmetric",
        _simplifyKW=_simplifyKW,
    )


def VFClass(
    varSpace,
    coeffs,
    dgcvType="standard",
    _simplifyKW=None,
):
    warnings.warn(
        "`VFClass` is deprecated and has been replaced by the general `vector_field_class`. The function label remains"
        "as a dispatcher to build `vector_field_class` objects, but this may be removed in the future"
        "Please use `vector_field_class` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if _simplifyKW is None:
        _simplifyKW = {
            "simplify_rule": None,
            "simplify_ignore_list": None,
            "preferred_basis_element": None,
        }

    vs = tuple(varSpace) if varSpace is not None else tuple()
    cs = list(coeffs) if coeffs is not None else []

    return vector_field_class(
        varSpace=vs,
        coeffs=cs,
        dgcvType=dgcvType,
        _simplifyKW=_simplifyKW,
        variable_spaces=None,
    )


def DFClass(
    varSpace,
    data_dict,
    degree,
    dgcvType="standard",
    _simplifyKW=None,
):
    warnings.warn(
        "`DFClass` is deprecated and has been replaced by the general `differential_form_class`. The function label remains"
        "as a dispatcher to build `differential_form_class` objects, but this may be removed in the future"
        "Please use `differential_form_class` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if _simplifyKW is None:
        _simplifyKW = {
            "simplify_rule": None,
            "simplify_ignore_list": None,
            "preferred_basis_element": None,
        }

    if not isinstance(varSpace, (list, tuple)):
        raise TypeError("`varSpace` must be a list or tuple.")
    if not isinstance(degree, Integral) or int(degree) < 0:
        raise ValueError("`degree` must be a non-negative integer.")
    if not isinstance(data_dict, dict):
        raise TypeError("`data_dict` must be a dictionary.")

    deg = int(degree)
    vs_list = list(varSpace)

    nz = {k: v for k, v in data_dict.items() if not _scalar_is_zero(v)}
    if not nz:
        return differential_form_class(
            coeff_dict={tuple(): 0},
            dgcvType=dgcvType,
            _simplifyKW=_simplifyKW,
            variable_spaces={},
        )

    if deg == 0:
        val = nz.get(tuple(), 0)
        return differential_form_class(
            coeff_dict={tuple(): val},
            dgcvType=dgcvType,
            _simplifyKW=_simplifyKW,
            variable_spaces={},
        )

    sys_for_var = {}
    systems_used = set()

    for v in vs_list:
        info = vmf_lookup(v, path=True, relatives=False)
        p = info.get("path")
        if not (isinstance(p, tuple) and len(p) >= 2):
            raise KeyError(
                "DFClass legacy init requires variables registered in the VMF."
            )
        syslbl = p[1]
        sys_for_var[v] = syslbl
        systems_used.add(syslbl)

    variable_spaces = {}
    system_index_cache = {}

    for syslbl in systems_used:
        info = vmf_lookup(syslbl, path=True, relatives=True, flattened_relatives=True)
        flat = info.get("flattened_relatives", None)
        if isinstance(flat, tuple) and flat:
            variable_spaces[syslbl] = flat
        else:
            seen = []
            for v in vs_list:
                if sys_for_var[v] == syslbl:
                    seen.append(v)
            variable_spaces[syslbl] = tuple(seen)

        system_index_cache[syslbl] = {
            v: i for i, v in enumerate(variable_spaces[syslbl])
        }

    new_cd = {}
    valence_tuple = (0,) * deg

    for key, value in nz.items():
        if not isinstance(key, tuple):
            raise TypeError("Keys in `data_dict` must be tuples.")
        if len(key) != deg:
            raise ValueError("`data_dict` keys must have length equal to `degree`.")

        idxs = []
        syslbls = []

        for pos in key:
            if not isinstance(pos, Integral):
                raise TypeError("Old-style indices must be integers.")
            ii = int(pos)
            if ii < 0 or ii >= len(vs_list):
                raise ValueError("Old-style index out of range.")

            var = vs_list[ii]
            syslbl = sys_for_var[var]
            syslbls.append(syslbl)

            j = system_index_cache[syslbl].get(var, None)
            if j is None:
                raise KeyError(
                    f"DFClass: variable '{var}' not found in cached system '{syslbl}'."
                )
            idxs.append(j)

        nk = tuple(idxs + list(valence_tuple) + syslbls)
        new_cd[nk] = new_cd.get(nk, 0) + value

    new_cd = {k: v for k, v in new_cd.items() if not _scalar_is_zero(v)}
    if not new_cd:
        new_cd = {tuple(): 0}

    return differential_form_class(
        coeff_dict=new_cd,
        dgcvType=dgcvType,
        _simplifyKW=_simplifyKW,
        variable_spaces=variable_spaces,
    )
