"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.algebras

module: algebras.algebras_core


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
from __future__ import annotations

import numbers
import random
import textwrap
import uuid
from collections.abc import Mapping
from functools import lru_cache
from html import escape as _esc
from typing import List, Literal, Optional

from .._aux._backends._display import fast_printable, latex
from .._aux._backends._display_engine import is_rich_displaying_available
from .._aux._backends._engine import engine_kind, engine_module
from .._aux._backends._polynomials import expr_union_primitives
from .._aux._backends._symbolic_router import (
    _scalar_is_zero,
    as_numer_denom,
    get_free_symbols,
    ratio,
    simplify,
    subs,
)
from .._aux._backends._types_and_constants import (
    expr_numeric_types,
    fast_scalar_types,
    rational,
    symbol,
)
from .._aux._utilities._config import (
    dgcv_exception_note,
    dgcv_warning,
    dgcvDeprecationWarning,
    from_vsr,
    get_dgcv_settings_registry,
    get_vs_registry,
    latex_in_html,
)
from .._aux._utilities._misc import linear_combination, zip_sum
from .._aux._utilities._styles import get_style
from .._aux._vmf._safeguards import (
    create_key,
    get_dgcv_category,
    query_dgcv_categories,
    retrieve_passkey,
    unique_label,
)
from .._aux._vmf.vmf import clearVar, listVar, order_coordinates
from .._aux.printing._tables import build_matrix_table, panel_view
from .._aux.printing.printing import lincomb_latex, lincomb_plain, space_display
from .._aux.printing.printing._dgcv_display import show
from ..core.arrays.arrays import _as_matrix_dgcv, array_dgcv, freeze_matrix, matrix_dgcv
from ..core.base import annotated_container, dgcv_class
from ..core.dgcv_core.dgcv_core import wedge
from ..core.morphisms.morphisms import homomorphism
from ..core.solvers.solvers import solve_dgcv
from ..core.tensors.tensors import tensorProduct
from .algebras_aux import _validate_structure_data

__all__ = [
    "adjointRepresentation",
    "algebra_class",
    "algebra_dual",
    "algebra_element_class",
    "algebra_subspace_class",
    "killingForm",
    "linear_representation",
    "vector_space_endomorphisms",
]


# -----------------------------------------------------------------------------
# Algebra classes
# -----------------------------------------------------------------------------


class _vector_space_methods:
    """
    Shared basis and grading-level operations for `dgcv` vector-space-like
    classes.

    Notes
    -----
    Host classes must provide `basis`, `dimension`, `grading`, `_gradingNumber`,
    `ambient` and `_verbose_subject`.
    Cache attributes default to None, so a host that does not initialize them
    still resolves.
    """

    _basis_index_cache = None
    _graded_components = None

    @property
    def _basis_index(self):
        if self._basis_index_cache is None:
            index_map = dict()
            for idx, elem in enumerate(self.basis):
                if elem not in index_map:
                    index_map[elem] = idx
            self._basis_index_cache = index_map
        return self._basis_index_cache

    def __contains__(self, item):
        try:
            return item in self._basis_index
        except TypeError:
            return False

    def _weight_coordinates(self, element):
        coeffs = self.contains(element, return_basis_coeffs=True)
        if coeffs is False:
            raise TypeError(
                f"Input to `check_element_weight` must belong to the {self._dgcv_category} instance whose `check_element_weight` is being called."
            ) from None
        return coeffs

    def check_element_weight(self, element, test_weights=None, flatten_weights=False):
        """
        Determines the weight vector of an element with respect to the grading vectors. Weight can instead be computed against another grading vector passed as a list of weights via the keyword `test_weights`.

        Parameters
        ----------
        element : (sub)algebra_element_class
            The element to analyze, given in the host's own basis coordinates.
        test_weights : list of scalars, optional (default: None)
        flatten_weights : bool, optional (default: False)
            If True, returns the contents of a length-1 list rather than the list.

        Returns
        -------
        tuple or weight value
            Weights corresponding to the grading vectors of this instance (or to `test_weights` if provided). Each entry is a scalar, the string 'AllW' if the element is zero, or 'NoW' if the element is not homogeneous.

        Notes
        -----
        - 'AllW' (All Weights) is returned for zero elements, which are compatible with all weights.
        - 'NoW' (No Weights) is returned for non-homogeneous elements.
        """
        if not test_weights and self._gradingNumber == 0:
            raise ValueError(
                f"This {self._dgcv_category} instance has no assigned grading vectors to test weighting w.r.t.."
            ) from None
        if element.is_zero:
            return tuple(["AllW"] * self._gradingNumber)
        if not test_weights and getattr(element, "_known_weight", None) is not None:
            if flatten_weights is True and len(element._known_weight) == 1:
                return element._known_weight[0]
            return element._known_weight
        if test_weights:
            numeric_types = expr_numeric_types()
            if not isinstance(test_weights, (list, tuple)):
                raise TypeError(
                    f"`check_element_weight` expects `test_weights` to be None or a list/tuple of lists/tuples of weight values (int, float, etc.). Received {test_weights}"
                ) from None
            for weight in test_weights:
                if not isinstance(weight, (list, tuple)):
                    raise TypeError(
                        f"`check_element_weight` expects `test_weights` to be None or a list/tuple of lists/tuples of weight values (int, float, etc.). Received {test_weights}"
                    ) from None
                if self.dimension != len(weight) or not all(
                    isinstance(j, numeric_types) for j in weight
                ):
                    raise TypeError(
                        f"`check_element_weight` expects `test_weights` to be None or a list/tuple of lists/tuples of weight values (int, float, etc.) of length {self.dimension}. Received {test_weights}"
                    ) from None
            GVs = test_weights
        else:
            GVs = self.grading
        coeff_indices = tuple(self._weight_coordinates(element))
        if not coeff_indices:
            weights = ["NoW"] * len(GVs)
        else:
            weights = []
            for grading_vector in GVs:
                first = grading_vector[coeff_indices[0]]
                homogeneous = True
                for i in coeff_indices:
                    if grading_vector[i] != first:
                        homogeneous = False
                        break
                weights.append(first if homogeneous else "NoW")
        if not test_weights:
            element._known_weight = tuple(weights)
        if flatten_weights and len(weights) == 1:
            return weights[0]
        return tuple(weights)

    def _verbose_subject(self):
        raise NotImplementedError

    def update_grading(self, new_weight_vectors_list, replace_instead_of_add=False):
        if isinstance(new_weight_vectors_list, (list, tuple)):
            if all(isinstance(elem, (list, tuple)) for elem in new_weight_vectors_list):
                if replace_instead_of_add is True:
                    self.grading = [tuple(elem) for elem in new_weight_vectors_list]
                else:
                    grad = list(self.grading) + [
                        tuple(elem) for elem in new_weight_vectors_list
                    ]
                    self.grading = grad
            else:
                raise TypeError(
                    f"update_grading expects first parameter to be a list of lists. The inner lists should have length {self.dimension}"
                )
        else:
            raise TypeError(
                f"update_grading expects first parameter to be a list of lists. The inner lists should have length {self.dimension}"
            )

    def compute_weight(self, element, test_weights=None, flatten_weights=False):
        return self.check_element_weight(
            element, test_weights=test_weights, flatten_weights=flatten_weights
        )

    def weighted_component(
        self,
        weights,
        test_weights=None,
        trust_test_weight_format=False,
    ):
        numeric_types = expr_numeric_types()
        if isinstance(weights, (set, dict)):
            weights = list(weights)
        if isinstance(weights, (list, tuple)):
            if all(isinstance(weight, numeric_types) for weight in weights):
                weights = [(weight,) for weight in weights]
            elif not all(isinstance(weight, (list, tuple)) for weight in weights):
                raise ValueError(
                    "The `weights` parameter in `algebra_class.weighted_component` must be a list/tuple of weights/multi-weights. If giving a single multi-weight, it should be a length-1 list/tuple of lists/tuples, as otherwise a bare mult-weight tuple will be interpreted as a list of singleton weights."
                ) from None
            else:
                weights = [tuple(weight) for weight in weights]
        else:
            raise ValueError(
                f"The `weights` parameter in `algebra_class.weighted_component` must be a list/tuple of weights/multi-weights. If giving a single multi-weight, it should be a length-1 list/tuple of lists/tuples, as otherwise a bare mult-weight tuple will be interpreted as a list of singleton weights. Instead received{weights}"
            ) from None
        try:
            weight_lookup = set(weights)
        except TypeError:
            weight_lookup = weights
        if test_weights is None:
            test_weights = self.grading
        elif trust_test_weight_format is False:
            if not isinstance(test_weights, (list, tuple)) or not all(
                isinstance(j, (list, tuple)) and len(j) == self.dimension
                for j in test_weights
            ):
                raise TypeError(
                    "The `test_weights` parameter in `algebra_class.weighted_component` must be a list/tuple of lists/tuples that length matches `algebra_class.dimension` and whose elements are weight values representing weights elements in `algebra_class.basis`."
                )
        component = []
        for idx, elem in enumerate(self.basis):
            elem_weight = tuple(vector[idx] for vector in test_weights)
            if elem_weight in weight_lookup or all(w == "AllW" for w in elem_weight):
                component.append(elem)
        return algebra_subspace_class(component, parent_algebra=self)

    @property
    def graded_components(self):
        if self._graded_components is None:
            basis = self.basis
            buckets = dict()
            for idx, weight in enumerate(zip(*self.grading)):
                key = tuple(weight)
                elem = basis[idx]
                elem._known_weight = key
                bucket = buckets.get(key)
                if bucket is None:
                    buckets[key] = [elem]
                else:
                    bucket.append(elem)
            try:
                keys = sorted(buckets)
            except TypeError:
                keys = sorted(buckets, key=lambda k: tuple(str(w) for w in k))
            self._graded_components = {
                key: algebra_subspace_class(buckets[key], parent_algebra=self)
                for key in keys
            }
        return self._graded_components

    def compute_graded_component_wrt_weight_index(self, idx=0):
        if idx not in range(len(self.grading)):
            dgcv_warning(
                "The provided index is out of range. `compute_graded_component_wrt_weight_index` is using 0 instead."
            )
            idx = 0
        wc = dict()
        for idxs, comp in self.graded_components.items():
            key = idxs[idx]
            if key in wc:
                wc[key] = wc[key] + comp
            else:
                wc[key] = self.subspace([]) + comp
        return wc

    def grading_summary(self):
        from .._aux.printing.printing._dgcv_display import show

        gradingNumber = len(self.grading)
        graded_components = self.graded_components
        pref = self._repr_latex_(abbrev=True).replace("$", "")
        if "_" in pref:
            prefi = f"\\left({pref} \\right)_"
        else:
            prefi = f"{pref}_"
        strings = []
        for k, v in graded_components.items():
            inner = "".join(str(j) for j in k)
            latex = v._repr_latex_().replace("$", "").replace(r"\displaystyle", "")
            strings.append(f"$$ {prefi}{{{inner}}} = {latex},$$")
        if gradingNumber == 0 or not strings:
            show(f"The algebra ${pref}$ has no assigned grading.")
        else:
            if len(strings) > 1:
                strings.insert(-1, "and")
            strings[-1] = strings[-1][:-3] + ".$$"
            if gradingNumber == 1:
                gradPhrase = "graded"
            elif gradingNumber == 2:
                gradPhrase = "bi-graded"
            elif gradingNumber == 3:
                gradPhrase = "tri-graded"
            else:
                gradPhrase = f"{gradingNumber}-graded"
            show(
                f"The algebra ${pref}$ has {gradPhrase} components: {' '.join(strings)}"
            )

    def filter_independent_elements(
        self,
        elements,
        apply_light_basis_simplification=False,
        return_indices: bool = False,
        surface_singularities: bool = False,
        simplify_singularities: bool = None,
        force_heavy_solve: bool = False,
    ):
        """
        Filters a set of elements to retain only a linearly independent subset.

        Parameters
        ----------
        elements : list of algebra_element_class
            The set of elements to filter.

        Returns
        -------
        list of algebra_element_class
            A subset of the input elements that are linearly independent and unique.
        """
        warning_message = ""
        remain_subalg = True
        subalg = None
        if not isinstance(elements, (list, tuple)):
            warning_message += (
                "\n The given value for `elements` is not a list or tuple"
            )
        else:
            nonAE, wrongAlgebra, correct = [], [], []
            typeCheck = {"algebra_element", "subalgebra_element"}
            for elem in elements:
                if elem == 0:
                    continue
                if remain_subalg is True:
                    if get_dgcv_category(elem) == "algebra_element":
                        remain_subalg = False
                    elif get_dgcv_category(elem) == "subalgebra_element":
                        if subalg is None:
                            subalg = elem.algebra
                        elif subalg != elem.algebra:
                            remain_subalg = False
                if get_dgcv_category(elem) not in typeCheck:
                    nonAE.append(elem)
                elif (
                    get_dgcv_category(elem) == "algebra_element"
                    and elem.algebra != self
                ) or (
                    get_dgcv_category(elem) == "subalgebra_element"
                    and elem.algebra.ambient != self
                ):
                    wrongAlgebra.append(elem)
                else:
                    correct.append(elem)
            elements = correct
            if len(nonAE) > 0:
                warning_message += f"\n • These list elements are not `algebra_element` or `subalgebra_element` type: {nonAE}"
            if len(wrongAlgebra) > 0:
                warning_message += f"\n • These list elements are `algebra_element` or `subalgebra_element` type, but belong to a different, unrelated algebra: {wrongAlgebra}"
        if warning_message:
            raise ValueError(
                "The `algebra` method `filter_independent_elements` can only be applied to lists of elements belong to the parent algebra the method is called from or any its subalgebras. Given data has the following problems:"
                + warning_message
            ) from None

        if remain_subalg is False:
            elements = [
                (
                    elem.ambient_rep
                    if get_dgcv_category(elem) == "subalgebra_element"
                    else elem
                )
                for elem in elements
            ]
        else:
            elements = list(elements)

        if return_indices is True:
            out = _extract_basis(
                elements,
                ALBS=apply_light_basis_simplification,
                return_indices=True,
                surface_singularities=surface_singularities,
                simplify_singularities=simplify_singularities,
                force_heavy_solve=force_heavy_solve,
            )
            if surface_singularities:
                _, idxs, sing = out
            else:
                _, idxs = out
            return (idxs, sing) if surface_singularities else idxs
        return _extract_basis(
            elements,
            ALBS=apply_light_basis_simplification,
            surface_singularities=surface_singularities,
            simplify_singularities=simplify_singularities,
            force_heavy_solve=force_heavy_solve,
        )

    def is_in_span(self, element, subspace_elements, assume_basis=False):
        """
        Checks if a given algebra_element_class is in the span of subspace_elements.

        Parameters
        ----------
        element : algebra_element_class
            The element to check.
        subspace_elements : list
            A list of algebra_element_class instances representing the subspace they span.
        assume_basis : bool, optional
            If True, returns the wedge product rather than a bool. Note a *zero*
            wedge indicates the element is in the span.

        Returns
        -------
        bool
            True if the element is in the span of subspace_elements, False otherwise.
        """
        if (
            not isinstance(subspace_elements, (list, tuple))
            or len(subspace_elements) == 0
        ):
            return _scalar_is_zero(element)
        if assume_basis:
            return wedge(element, *subspace_elements)
        combo, variables = linear_combination(subspace_elements)
        diff = element - combo
        eqns = list(diff.coeff_dict.values())
        sol2 = solve_dgcv(eqns, variables, method="linsolve", simplify_result=False)
        return bool(sol2)

    def subspace(self, basis: list | tuple = [], grading=None, span_warning=True):
        if grading is None:
            grading = self.grading
        return algebra_subspace_class(
            basis, parent_algebra=self, test_weights=grading, span_warning=span_warning
        )


class _algebra_methods(_vector_space_methods):
    """
    Shared method implementations for `dgcv` algebra-like classes.

    Notes
    -----
    Host classes must provide `dimension`, `basis`, `structureData`,
    `structureDataDict`, `grading`, `_gradingNumber`, `_educed_properties`,
    `_parameters`, `_registered`, `ambient`, `_dgcv_category` and
    `_verbose_subject`. Cache attributes default to None here, so a host that
    does not initialize them still resolves.
    """

    _skew_symmetric_cache = None
    _jacobi_identity_cache = None
    _lie_algebra_cache = None
    _is_semisimple_cache = None
    _is_simple_cache = None
    _is_nilpotent_cache = None
    _is_solvable_cache = None
    _is_abelian_cache = None
    _killing_form = None
    _Levi_deco_cache = None
    _lower_central_series_cache = None
    _lower_central_series_terminated = None
    _lower_central_series_depth = None
    _derived_series_cache = None
    _derived_series_terminated = None
    _derived_series_depth = None
    _derived_subalg_cache = None
    _radical_cache = None
    _center_cache = None
    _grading_compatible = None
    _grading_report = None
    _rank_approximation = None
    _structure_data_slices = None
    _structure_rows_cache = None

    @property
    def _structure_rows(self):
        if self._structure_rows_cache is None:
            rows = dict()
            for (i, j, k), v in self.structureDataDict.items():
                row = rows.get((i, j))
                if row is None:
                    rows[(i, j)] = {k: v}
                else:
                    row[k] = v
            self._structure_rows_cache = rows
        return self._structure_rows_cache

    def is_skew_symmetric(
        self,
        verbose=False,
        _return_proof_path=False,
        _ignore_caches=False,
        *,
        _timed_reporting: bool | None = None,
        _reporting_threshold_s: float = 10,
        _progress_message: str | None = None,
        _on_timed_update=None,
    ):
        """
        Checks if the algebra is skew-symmetric.
        """
        if verbose and not self._registered:
            if self.ambient._callLock == retrieve_passkey() and isinstance(
                self.ambient._print_warning, str
            ):
                print(self.ambient._print_warning)
            else:
                print(
                    "Warning: This algebra instance is unregistered. Initialize algebra objects with createFiniteAlg instead to register them."
                )

        educed = self._educed_properties.get("is_skew", None)
        if isinstance(educed, str) and _ignore_caches is False:
            t_message = educed
            self._skew_symmetric_cache = (True, None)
        else:
            t_message = ""

        timed = bool(_timed_reporting) if _timed_reporting is not None else False

        cached = self._skew_symmetric_cache
        if cached is not None and _ignore_caches is False:
            result, failure = cached
        else:
            result, failure = _timed_progress_call(
                self._check_skew_symmetric,
                timed=timed,
                threshold_s=float(_reporting_threshold_s),
                step_desc="checking skew symmetry of the structure constants",
                continue_desc=_progress_message,
                progress_message=None,
                _on_timed_update=_on_timed_update,
            )
            self._skew_symmetric_cache = (result, failure)

        if verbose and not timed:
            if result:
                print(f"{self._verbose_subject()} is skew-symmetric.")
            else:
                i, j, k = failure
                print(
                    f"Skew symmetry fails for basis elements {i}, {j}, at coefficient index {k}."
                )
        if _return_proof_path is True:
            return result, t_message
        return result

    def _check_skew_symmetric(self):
        sdd = self.structureDataDict
        candidates = {(i, j, k) if i <= j else (j, i, k) for i, j, k in sdd}
        for i, j, k in sorted(candidates):
            expr = sdd.get((i, j, k), 0) + sdd.get((j, i, k), 0)
            if _scalar_is_zero(expr):
                continue
            if get_free_symbols(expr) and _scalar_is_zero(simplify(expr)):
                continue
            return False, (i, j, k)
        return True, None

    def satisfies_jacobi_identity(
        self,
        verbose=False,
        _return_proof_path=False,
        _ignore_caches=False,
        *,
        _timed_reporting: bool | None = None,
        _reporting_threshold_s: float = 10,
        _progress_message: str | None = None,
        _on_timed_update=None,
    ):
        """
        Checks if the algebra satisfies the Jacobi identity.
        Includes a warning for unregistered instances only if verbose=True.
        """
        if not self._registered and verbose:
            if self.ambient._callLock == retrieve_passkey() and isinstance(
                self.ambient._print_warning, str
            ):
                print(self.ambient._print_warning)
            else:
                print(
                    "Warning: This algebra instance is unregistered. Initialize algebra objects with createFiniteAlg instead to register them."
                )

        if (
            isinstance(self._educed_properties.get("satisfies_Jacobi_ID", None), str)
            and _ignore_caches is False
        ):
            t_message = self._educed_properties.get("satisfies_Jacobi_ID", None)
            self._jacobi_identity_cache = (True, None)
        else:
            t_message = ""

        timed = bool(_timed_reporting) if _timed_reporting is not None else False
        threshold = float(_reporting_threshold_s)

        if self._jacobi_identity_cache is None or _ignore_caches is True:
            result, fail_list = _timed_progress_call(
                self._check_jacobi_identity,
                timed=timed,
                threshold_s=threshold,
                step_desc="checking the Jacobi identity",
                continue_desc=_progress_message,
                progress_message=None,
                _on_timed_update=_on_timed_update,
            )
            self._jacobi_identity_cache = (result, fail_list)
        else:
            result, fail_list = self._jacobi_identity_cache

        if verbose and not timed:
            if result:
                print(f"{self._verbose_subject()} satisfies the Jacobi identity.")
            else:
                print(f"Jacobi identity fails for the following triples: {fail_list}")

        if _return_proof_path is True:
            return result, t_message
        return result

    def Jacobi_identities(self):
        skew, dim, basis = self.is_skew_symmetric(), self.dimension, self.basis
        JI_list = []
        for i in range(dim):
            lower_j = i + 1 if skew else 0
            for j in range(lower_j, dim):
                lower_k = j + 1 if skew else 0
                for k in range(lower_k, dim):
                    ai, aj, ak = basis[i], basis[j], basis[k]
                    JI_list.append(ai * aj * ak + aj * ak * ai + ak * ai * aj)
        return JI_list

    def _check_jacobi_identity(self):
        skew, dim = self.is_skew_symmetric(), self.dimension
        rows = self._structure_rows
        if skew:
            candidates = set()
            for a, b in rows:
                lo, hi = (a, b) if a < b else (b, a)
                if lo == hi:
                    continue
                for c in range(dim):
                    if c == lo or c == hi:
                        continue
                    candidates.add(tuple(sorted((lo, hi, c))))
            triples = sorted(candidates)
        else:
            triples = (
                (i, j, k)
                for i in range(dim)
                for j in range(dim)
                for k in range(dim)
                if (i, j) in rows or (j, k) in rows or (k, i) in rows
            )
        fail_list = []
        for i, j, k in triples:
            acc = dict()
            for a, b, c in ((i, j, k), (j, k, i), (k, i, j)):
                left = rows.get((a, b))
                if not left:
                    continue
                for m, cm in left.items():
                    right = rows.get((m, c))
                    if not right:
                        continue
                    for n, cn in right.items():
                        acc[n] = acc.get(n, 0) + cm * cn
            for expr in acc.values():
                if _scalar_is_zero(expr):
                    continue
                if get_free_symbols(expr) and _scalar_is_zero(simplify(expr)):
                    continue
                fail_list.append((i, j, k))
                break
        if fail_list:
            return False, fail_list
        return True, None

    def _warn_associativity_assumption(self, method_name):
        """
        Issues a warning that the method assumes the algebra is associative.

        Parameters
        ----------
        method_name : str
            The name of the method assuming associativity.

        Notes
        -----
        - helper method intended for internal use
        """
        dgcv_warning(
            f"{method_name} assumes the algebra is associative. "
            "If it is not then unexpected results may occur."
        )

    def is_lie_algebra(self, verbose=False, return_bool=True):
        dgcv_warning(
            "`is_lie_algebra` has been deprecated as part of the shift toward standardized naming conventions in the `dgcv` library.",
            dgcvDeprecationWarning,
            stacklevel=2,
            old_kw="is_lie_algebra",
            new_kw="is_Lie_algebra",
            sunset="2026",
        )
        return self.is_Lie_algebra(verbose=verbose, return_bool=return_bool)

    def is_Lie_algebra(
        self,
        verbose=False,
        return_bool=True,
        _return_proof_path=False,
        _ignore_caches=False,
        *,
        _timed_reporting: bool | None = None,
        _reporting_threshold_s: float = 10,
        _progress_message: str | None = None,
        _on_timed_update=None,
    ):
        if not self._registered and verbose:
            if self.ambient._callLock == retrieve_passkey() and isinstance(
                self.ambient._print_warning, str
            ):
                print(self.ambient._print_warning)
            else:
                print(
                    "Warning: This algebra instance is unregistered. Initialize algebra objects with createFiniteAlg instead to register them."
                )

        if isinstance(self._educed_properties.get("is_Lie_algebra", None), str):
            t_message = self._educed_properties.get("is_Lie_algebra", None)
            self._lie_algebra_cache = True
            self._jacobi_identity_cache = (True, None)
            self._skew_symmetric_cache = (True, None)
        else:
            t_message = ""

        timed = bool(_timed_reporting) if _timed_reporting is not None else False
        threshold = float(_reporting_threshold_s)

        if self._lie_algebra_cache is not None and _ignore_caches is False:
            if verbose and not timed:
                print(
                    f"Cached result: Previously verified "
                    f"{self._verbose_subject()} is"
                    f"{'' if self._lie_algebra_cache else ' not'} a Lie algebra."
                )
            if _return_proof_path is True:
                return self._lie_algebra_cache, t_message
            return self._lie_algebra_cache

        ok_skew = self.is_skew_symmetric(
            verbose=verbose,
            _ignore_caches=_ignore_caches,
            _timed_reporting=timed,
            _reporting_threshold_s=threshold,
            _progress_message="check the Jacobi identity",
            _on_timed_update=_on_timed_update,
        )
        if not ok_skew:
            self._lie_algebra_cache = False
            if return_bool is True:
                if _return_proof_path is True:
                    return False, t_message
                return False
            return

        ok_jacobi = self.satisfies_jacobi_identity(
            verbose=verbose,
            _ignore_caches=_ignore_caches,
            _timed_reporting=timed,
            _reporting_threshold_s=threshold,
            _progress_message=_progress_message,
            _on_timed_update=_on_timed_update,
        )
        if not ok_jacobi:
            self._lie_algebra_cache = False
            if return_bool is True:
                if _return_proof_path is True:
                    return False, t_message
                return False
            return

        if self._lie_algebra_cache is None or _ignore_caches is True:
            self._lie_algebra_cache = True

        if verbose and not timed:
            print(f"{self._verbose_subject()} is a Lie algebra.")

        if return_bool is True:
            if _return_proof_path is True:
                return self._lie_algebra_cache, t_message
            return self._lie_algebra_cache

    def _require_lie_algebra(self, method_name):
        """
        Checks that the algebra is a Lie algebra before proceeding.

        Parameters
        ----------
        method_name : str
            The name of the method requiring a Lie algebra.

        Raises
        ------
        ValueError
            If the algebra is not a Lie algebra.
        """
        if not self.is_Lie_algebra():
            raise ValueError(
                f"{method_name} can only be applied to Lie algebras."
            ) from None

    def is_semisimple(
        self,
        verbose=False,
        return_bool=True,
        _return_proof_path=False,
        _ignore_caches=False,
        *,
        _timed_reporting: bool | None = None,
        _reporting_threshold_s: float = 10,
        _progress_message: str | None = None,
        _on_timed_update=None,
    ):
        """
        Checks if the algebra is semisimple.
        Nothing is returned if return_bool=False is set.
        """
        if not self._registered and verbose:
            if self.ambient._callLock == retrieve_passkey() and isinstance(
                self.ambient._print_warning, str
            ):
                print(self.ambient._print_warning)
            else:
                print(
                    "Warning: This algebra instance is unregistered. Initialize algebra objects with createFiniteAlg instead to register them."
                )

        if (
            isinstance(self._educed_properties.get("is_simple", None), str)
            and _ignore_caches is False
        ):
            t_message = self._educed_properties.get("is_simple", None)
            self._is_simple_cache = True
            self._is_semisimple_cache = True
            self._educed_properties["special_type"] = "simple"
            self._is_nilpotent_cache = False
            self._is_solvable_cache = False
        elif (
            isinstance(self._educed_properties.get("is_semisimple", None), str)
            and _ignore_caches is False
        ):
            t_message = self._educed_properties.get("is_semisimple", None)
            self._is_semisimple_cache = True
            self._educed_properties["special_type"] = (
                self._educed_properties.get("special_type", None) or "semisimple"
            )
            self._is_nilpotent_cache = False
            self._is_solvable_cache = False
        else:
            t_message = ""

        timed = bool(_timed_reporting) if _timed_reporting is not None else False
        threshold = float(_reporting_threshold_s)

        if self._is_semisimple_cache is None and _ignore_caches is False:
            if self._is_simple_cache is True:
                self._is_semisimple_cache = True
                self._is_solvable_cache = False
                self._is_abelian_cache = False
                self._is_nilpotent_cache = False
            elif self._Levi_deco_cache is not None:
                LC, MSI = self._Levi_deco_cache["LD_components"]
                if getattr(MSI, "dimension", None) == 0 and self.dimension > 0:
                    self._is_semisimple_cache = True
                    self._is_solvable_cache = False
                    self._is_abelian_cache = False
                    self._is_nilpotent_cache = False
                elif getattr(MSI, "dimension", None) != 0:
                    self._is_semisimple_cache = False
                    self._is_simple_cache = False
                    if getattr(LC, "dimension", None) == 0:
                        self._is_solvable_cache = True
                        self._educed_properties["special_type"] = "solvable"
                    else:
                        self._is_solvable_cache = False
                        self._is_nilpotent_cache = False
                        self._is_abelian_cache = False

        if self._is_semisimple_cache is not None and _ignore_caches is False:
            if verbose and not timed:
                print(
                    f"Cached result: Previously verified "
                    f"{self._verbose_subject()} is"
                    f"{'' if self._is_semisimple_cache else ' not'} a semisimple Lie algebra."
                )
            if return_bool is True:
                if _return_proof_path is True:
                    return self._is_semisimple_cache, t_message
                return self._is_semisimple_cache
            if _return_proof_path is True:
                return t_message
            return

        ok_lie = self.is_Lie_algebra(
            verbose=verbose,
            _ignore_caches=_ignore_caches,
            _timed_reporting=timed,
            _reporting_threshold_s=threshold,
            _progress_message=_progress_message,
            _on_timed_update=_on_timed_update,
        )

        if not ok_lie:
            self._is_semisimple_cache = False
            if return_bool is True:
                if _return_proof_path is True:
                    return False, "not a Lie algebra"
                return False
            if _return_proof_path is True:
                return "not a Lie algebra"
            return

        def _killing_det():
            if self._killing_form is None:
                self._killing_form = killingForm(self)
            return simplify(self._killing_form.det())

        det = _timed_progress_call(
            _killing_det,
            timed=timed,
            threshold_s=threshold,
            step_desc="computing determinant of the Killing form",
            continue_desc=_progress_message,
            progress_message=None,
            _on_timed_update=_on_timed_update,
        )

        iz = getattr(det, "is_zero", None)
        if iz is True:
            det_is_zero = True
        elif callable(iz):
            try:
                det_is_zero = bool(iz())
            except Exception:
                det_is_zero = _scalar_is_zero(det)
        else:
            det_is_zero = _scalar_is_zero(det)

        det_is_nonzero = not det_is_zero

        if det_is_nonzero:
            self._is_semisimple_cache = True
            self._educed_properties["special_type"] = "semisimple"
            self._is_nilpotent_cache = False
            self._is_solvable_cache = False
        else:
            self._is_semisimple_cache = False
            self._is_simple_cache = False

        if verbose and not timed:
            print(
                f"{self._verbose_subject()} is"
                f"{'' if det_is_nonzero else ' not'} semisimple."
            )

        if return_bool is True:
            if _return_proof_path is True:
                return det_is_nonzero, t_message
            return det_is_nonzero

    def is_simple(
        self,
        verbose=False,
        bypass_semisimple_check=False,
        _return_proof_path=False,
        _ignore_caches=False,
        *,
        surface_singularities=False,
        _timed_reporting: bool | None = None,
        _reporting_threshold_s: float = 10,
        _progress_message: str | None = None,
        _on_timed_update=None,
    ):
        if isinstance(self._educed_properties.get("is_simple", None), str):
            t_message = self._educed_properties.get("is_simple", None)
            self._is_simple_cache = True
            self._is_semisimple_cache = True
            self._educed_properties["special_type"] = "simple"
            self._is_nilpotent_cache = False
            self._is_solvable_cache = False
        else:
            t_message = ""

        timed = bool(_timed_reporting) if _timed_reporting is not None else False
        threshold = float(_reporting_threshold_s)

        if bypass_semisimple_check is False and self._is_semisimple_cache is None:
            self.is_semisimple(
                verbose=verbose,
                _ignore_caches=_ignore_caches,
                _timed_reporting=timed,
                _reporting_threshold_s=threshold,
                _progress_message=_progress_message,
                _on_timed_update=_on_timed_update,
            )

        if self._is_simple_cache is None:
            self.compute_simple_subalgebras(
                verbose=verbose,
                surface_singularities=surface_singularities,
                _timed_reporting=timed,
                _reporting_threshold_s=threshold,
                _progress_message=_progress_message,
                _on_timed_update=_on_timed_update,
            )

            if self._Levi_deco_cache["LD_components"][1].dimension == 0:
                self._is_semisimple_cache = True
                self._is_nilpotent_cache = False
                self._is_solvable_cache = False
                if len(self._Levi_deco_cache["simple_ideals"]) == 1:
                    self._is_simple_cache = True
                    self._educed_properties["special_type"] = "simple"
                else:
                    self._is_simple_cache = False
                    self._educed_properties["special_type"] = "semisimple"
            else:
                self._is_semisimple_cache = False
                self._is_simple_cache = False
                if self._Levi_deco_cache["LD_components"][0].dimension == 0:
                    self._is_solvable_cache = True
                    if self._educed_properties.get("special_type", None) is None:
                        self._educed_properties["special_type"] = "solvable"

        if _return_proof_path is True:
            return self._is_simple_cache, t_message
        return self._is_simple_cache

    def is_nilpotent(
        self,
        *,
        _timed_reporting: bool | None = None,
        _reporting_threshold_s: float = 10,
        _progress_message: str | None = None,
        _on_timed_update=None,
        **kwargs,
    ):
        """
        Checks if the algebra is nilpotent.

        Returns
        -------
        bool
            True if the algebra is nilpotent, False otherwise.
        """
        if kwargs:
            dgcv_warning(
                f"`{type(self).__name__}.is_nilpotent` received unexpected keyword "
                f"argument(s) {sorted(kwargs)}, which were ignored."
            )
        if self._is_nilpotent_cache is None and self._is_abelian_cache is True:
            self._is_nilpotent_cache = True
        if self._is_nilpotent_cache is None:
            _timed_progress_call(
                self.lower_central_series,
                timed=bool(_timed_reporting) if _timed_reporting is not None else False,
                threshold_s=float(_reporting_threshold_s),
                step_desc="computing the lower central series",
                continue_desc=_progress_message,
                progress_message=None,
                _on_timed_update=_on_timed_update,
            )
            if getattr(self, "_lower_central_series_terminated", None) is True:
                self._is_nilpotent_cache = True
                self._educed_properties["special_type"] = "nilpotent"
                self._is_semisimple_cache = False
                self._is_simple_cache = False
            else:
                self._is_nilpotent_cache = False
                self._is_abelian_cache = False
        return self._is_nilpotent_cache

    def is_solvable(
        self,
        *,
        _timed_reporting: bool | None = None,
        _reporting_threshold_s: float = 10,
        _progress_message: str | None = None,
        _on_timed_update=None,
        **kwargs,
    ):
        """
        Checks if the algebra is solvable.

        Returns
        -------
        bool
            True if the algebra is solvable, False otherwise.
        """
        if kwargs:
            dgcv_warning(
                f"`{type(self).__name__}.is_solvable` received unexpected keyword "
                f"argument(s) {sorted(kwargs)}, which were ignored."
            )
        if self._is_solvable_cache is None:
            if self._is_nilpotent_cache is None or self._is_nilpotent_cache is False:
                _timed_progress_call(
                    self.derived_series,
                    timed=bool(_timed_reporting)
                    if _timed_reporting is not None
                    else False,
                    threshold_s=float(_reporting_threshold_s),
                    step_desc="computing the derived series",
                    continue_desc=_progress_message,
                    progress_message=None,
                    _on_timed_update=_on_timed_update,
                )
                if getattr(self, "_derived_series_terminated", None) is True:
                    self._is_solvable_cache = True
                    self._is_semisimple_cache = False
                    self._is_simple_cache = False
                    self._educed_properties["special_type"] = "solvable"
                else:
                    self._is_solvable_cache = False
                    self._is_abelian_cache = False
                    self._is_nilpotent_cache = False
            else:
                self._is_solvable_cache = self._is_nilpotent_cache
        return self._is_solvable_cache

    def is_abelian(
        self,
        *,
        _timed_reporting: bool | None = None,
        _reporting_threshold_s: float = 10,
        _progress_message: str | None = None,
        _on_timed_update=None,
        **kwargs,
    ):
        if kwargs:
            dgcv_warning(
                f"`{type(self).__name__}.is_abelian` received unexpected keyword "
                f"argument(s) {sorted(kwargs)}, which were ignored."
            )
        if self._is_abelian_cache is None:
            if self._educed_properties.get("special_type", None) == "abelian":
                self._is_abelian_cache = True
                self._is_nilpotent_cache = True
                self._is_solvable_cache = True
                self._is_semisimple_cache = False
                self._is_simple_cache = False
            else:
                self._is_abelian_cache = _timed_progress_call(
                    lambda: all(
                        _scalar_is_zero(elem)
                        for elem in self.structureDataDict.values()
                    ),
                    timed=bool(_timed_reporting)
                    if _timed_reporting is not None
                    else False,
                    threshold_s=float(_reporting_threshold_s),
                    step_desc="checking whether every structure constant vanishes",
                    continue_desc=_progress_message,
                    progress_message=None,
                    _on_timed_update=_on_timed_update,
                )
                if self._is_abelian_cache is True:
                    self._educed_properties["special_type"] = "abelian"
                    self._is_nilpotent_cache = True
                    self._is_solvable_cache = True
                    self._is_semisimple_cache = False
                    self._is_simple_cache = False
        return self._is_abelian_cache

    def compute_simple_subalgebras(
        self,
        verbose: bool = False,
        *,
        surface_singularities=False,
        _timed_reporting: bool | None = None,
        _reporting_threshold_s: float = 10,
        _progress_message: str | None = None,
        _on_timed_update=None,
    ):
        timed = bool(_timed_reporting) if _timed_reporting is not None else False
        threshold = float(_reporting_threshold_s)
        self.Levi_decomposition(
            decompose_semisimple_fully=True,
            verbose=verbose,
            _timed_reporting=timed,
            _reporting_threshold_s=threshold,
            _progress_message=_progress_message,
            _on_timed_update=_on_timed_update,
            surface_singularities=surface_singularities,
        )
        return self._Levi_deco_cache["simple_ideals"]

    def compute_derived_algebra(self):
        """
        Computes the derived algebra (commutator subalgebra) for Lie algebras.

        Returns
        -------
        algebra
            A new algebra instance representing the derived algebra.

        Raises
        ------
        ValueError
            If the algebra is not a Lie algebra or if the derived algebra cannot be computed.

        Notes
        -----
        - This method only applies to Lie algebras.
        - The derived algebra is generated by all products [x, y] = x * y, where * is the Lie bracket.
        """
        self._set_product_protocol()

        ###!!!
        # self._require_lie_algebra("compute_derived_algebra")

        if self._derived_subalg_cache is None:
            commutators = []
            basis = self.basis
            dim = len(basis)
            skew = self.is_skew_symmetric()
            for j in range(dim):
                el1 = basis[j]
                lIdx = j + 1 if skew else 0
                for k in range(lIdx, dim):
                    commutators.append(el1 * basis[k])
            self._derived_subalg_cache = self.subalgebra(
                commutators, span_warning=False, simplify_basis=True
            )
        return self._derived_subalg_cache

    def lower_central_series(
        self,
        max_depth=None,
        format_as_subalgebras=False,
        align_nested_bases=False,
    ):
        """
        Computes the lower central series of the algebra (or given subalgebra).

        Parameters
        ----------
        max_depth : int, optional
            Maximum depth to compute the series. Defaults to the dimension of the algebra.

        Returns
        -------
        list of lists
            A list where each entry contains the basis for that level of the lower central series.

        Notes
        -----
        - The lower central series is defined as:
            g_1 = g,
            g_{k+1} = [g_k, g]
        """
        self._set_product_protocol()
        scoped_basis = list(self.basis)
        requested_depth = (
            max(self.dimension, 1) if max_depth is None else int(max_depth)
        )
        cached_depth = getattr(self, "_lower_central_series_depth", None)
        cache_usable = self._lower_central_series_cache is not None and (
            getattr(self, "_lower_central_series_terminated", None) is True
            or (cached_depth is not None and cached_depth >= requested_depth)
        )
        if not cache_usable:
            series = []
            current_basis = scoped_basis
            previous_length = len(current_basis)
            terminated = False

            for _ in range(requested_depth):
                series.append(current_basis)

                lower_central = []
                for el1 in current_basis:
                    for el2 in scoped_basis:
                        commutator = el1 * el2
                        lower_central.append(commutator)
                independent_generators = self.filter_independent_elements(
                    lower_central, apply_light_basis_simplification=True
                )
                if len(independent_generators) == 0:
                    if len(scoped_basis) > 0:
                        series.append([])
                    terminated = True
                    break
                if len(independent_generators) == previous_length:
                    break
                current_basis = independent_generators
                previous_length = len(independent_generators)
            if len(series) > 1 and self._derived_subalg_cache is None:
                self._derived_subalg_cache = self.subalgebra(
                    series[1], span_warning=False, simplify_basis=True
                )
            self._lower_central_series_cache = (
                series,
                False,
            )  # series, alignment bool
            self._lower_central_series_terminated = terminated
            self._lower_central_series_depth = requested_depth
        if align_nested_bases is True and self._lower_central_series_cache[1] is False:
            if len(self._lower_central_series_cache[0]) > 0 and get_dgcv_category(
                self._lower_central_series_cache[0][0]
            ) in {"algebra", "subalgebra"}:
                ser = [list(alg.basis) for alg in self._lower_central_series_cache[0]]
            else:
                ser = self._lower_central_series_cache[0]
            new_series = [ser[-1]]
            depth = len(ser)
            for idx in range(1, depth):
                old_level = ser[depth - 1 - idx]
                discrep = len(old_level) - len(ser[depth - idx])
                new_level = list(new_series[0])
                for idx2 in range(len(old_level)):
                    if discrep == 0:
                        break
                    elem = old_level[-1 - idx2]
                    if _indep_check(ser[depth - idx], elem):
                        new_level.insert(0, elem)
                        discrep += -1
                new_series.insert(0, new_level)
            self._lower_central_series_cache = (
                new_series,
                True,
            )  # series, alignment bool
        if format_as_subalgebras:
            if len(self._lower_central_series_cache[0]) > 0 and isinstance(
                self._lower_central_series_cache[0][0], list
            ):
                self._lower_central_series_cache = (
                    [
                        self.subalgebra(sa, span_warning=False)
                        for sa in self._lower_central_series_cache[0]
                    ],
                    self._lower_central_series_cache[1],
                )
            returnSer = self._lower_central_series_cache[0]
        else:
            if len(self._lower_central_series_cache[0]) > 0 and get_dgcv_category(
                self._lower_central_series_cache[0][0]
            ) in {"algebra", "subalgebra"}:
                returnSer = [
                    list(alg.basis) for alg in self._lower_central_series_cache[0]
                ]
            else:
                returnSer = self._lower_central_series_cache[0]
        return returnSer

    def derived_series(
        self,
        max_depth=None,
        format_as_subalgebras=False,
        align_nested_bases=False,
        surface_singularities=False,
        simplify_singularities=None,
        force_heavy_solve=False,
    ):
        """
        Computes the derived series of the algebra.

        Parameters
        ----------
        max_depth : int, optional
            Maximum depth to compute the series. Defaults to the dimension of the algebra.

        Returns
        -------
        list of lists
            A list where each entry contains the basis for that level of the derived series.

        Notes
        -----
        - The derived series is defined as:
            g^{(1)} = g,
            g^{(k+1)} = [g^{(k)}, g^{(k)}]
        """

        self._set_product_protocol()
        scoped_basis = list(self.basis)
        requested_depth = (
            max(self.dimension, 1) if max_depth is None else int(max_depth)
        )
        cached_depth = getattr(self, "_derived_series_depth", None)
        cached_heavy = getattr(self, "_derived_series_heavy", False)
        cache_usable = (
            self._derived_series_cache is not None
            and (cached_heavy or not force_heavy_solve)
            and (
                getattr(self, "_derived_series_terminated", None) is True
                or (cached_depth is not None and cached_depth >= requested_depth)
            )
        )
        if not cache_usable:
            series = []
            current_basis = scoped_basis
            previous_length = len(current_basis)
            total_sing = []
            terminated = False
            for _ in range(requested_depth):
                series.append(list(current_basis))

                derived = []
                level_len = len(current_basis)
                for count in range(level_len):
                    el1 = current_basis[count]
                    start = count + 1 if self.is_skew_symmetric() else 0
                    for idx2 in range(start, level_len):
                        derived.append(el1 * current_basis[idx2])
                out = self.filter_independent_elements(
                    derived,
                    apply_light_basis_simplification=True,
                    surface_singularities=surface_singularities,
                    simplify_singularities=simplify_singularities,
                    force_heavy_solve=force_heavy_solve,
                )
                if surface_singularities:
                    independent_generators, sing = out
                    total_sing += sing
                else:
                    independent_generators = out
                if len(independent_generators) == 0:
                    if len(scoped_basis) > 0:
                        series.append([])
                    terminated = True
                    break
                if len(independent_generators) == previous_length:
                    break

                if force_heavy_solve:
                    independent_generators = [
                        simplify(gen) for gen in independent_generators
                    ]
                current_basis = list(independent_generators)
                previous_length = len(independent_generators)
            if surface_singularities:
                if get_dgcv_settings_registry().get(
                    "simplify_singularity_ideals_by_default", True
                ):
                    self._singularities["derived_series"] = expr_union_primitives(
                        [v for v in total_sing if get_free_symbols(v)],
                        order_coordinates(self._parameters),
                        process_rationals=True,
                        fail_quietly=True,
                    )
                else:
                    self._singularities["derived_series"] = [
                        v for v in total_sing if get_free_symbols(v)
                    ]
            if len(series) > 1 and self._derived_subalg_cache is None:
                self._derived_subalg_cache = self.subalgebra(
                    series[1], span_warning=False, simplify_basis=True
                )
            self._derived_series_cache = (series, False)  # series, alignment bool
            self._derived_series_terminated = terminated
            self._derived_series_depth = requested_depth
            self._derived_series_heavy = bool(force_heavy_solve)
        if align_nested_bases is True and self._derived_series_cache[1] is False:
            if len(self._derived_series_cache[0]) > 0 and get_dgcv_category(
                self._derived_series_cache[0][0]
            ) in {"algebra", "subalgebra"}:
                ser = [list(alg.basis) for alg in self._derived_series_cache[0]]
            else:
                ser = self._derived_series_cache[0]
            depth = len(ser)
            new_series = [] if depth == 0 else [ser[-1]]
            build_step = 1
            if (
                len(new_series) == 1
                and len(new_series[0]) == 1
                and getattr(new_series[0][0], "is_zero", False)
            ):
                new_series.insert(0, ser[-2])
                build_step = 2
            for idx in range(build_step, depth):
                old_level = ser[depth - 1 - idx]
                discrep = len(old_level) - len(ser[depth - idx])
                new_level = list(new_series[0])
                for idx2 in range(len(old_level)):
                    if discrep == 0:
                        break
                    elem = old_level[-1 - idx2]
                    if _indep_check(
                        new_level,
                        elem,
                        force_heavy_solve=force_heavy_solve,
                    ):
                        new_level.insert(0, elem)
                        discrep += -1
                new_series.insert(0, new_level)
            self._derived_series_cache = (new_series, True)  # series, alignment bool
        if format_as_subalgebras:
            if len(self._derived_series_cache[0]) > 0 and isinstance(
                self._derived_series_cache[0][0], list
            ):
                self._derived_series_cache = (
                    [
                        self.subalgebra(sa, span_warning=False)
                        for sa in self._derived_series_cache[0]
                    ],
                    self._derived_series_cache[1],
                )
            returnSer = self._derived_series_cache[0]
        else:
            if len(self._derived_series_cache[0]) > 0 and get_dgcv_category(
                self._derived_series_cache[0][0]
            ) in {"algebra", "subalgebra"}:
                returnSer = [list(alg.basis) for alg in self._derived_series_cache[0]]
            else:
                returnSer = self._derived_series_cache[0]
        return returnSer

    def radical(
        self,
        assume_Lie_algebra=False,
        surface_singularities=False,
        simplify_singularities=None,
        force_heavy_solve=False,
    ):
        if (
            self._radical_cache is not None
            and force_heavy_solve
            and not getattr(self, "_radical_heavy", False)
        ):
            self._radical_cache = None
        if self._radical_cache is None and self.dimension == 0:
            self._radical_cache = self.subalgebra([], span_warning=False)
            self._radical_heavy = True
        elif self._radical_cache is None:
            da = self.compute_derived_algebra()
            genElem, variables = linear_combination(self.basis_in_ambient_alg)
            amb = self.ambient
            if amb._killing_form is None:
                amb._killing_form = killingForm(
                    amb, assume_Lie_algebra=assume_Lie_algebra
                )
            amb_dim = amb.dimension
            kf_gen = amb._killing_form * matrix_dgcv(
                genElem.coeff_dict, shape=(amb_dim, 1)
            )
            eqns = [
                (matrix_dgcv(elem.coeff_dict, shape=(1, amb_dim)) * kf_gen)[0]
                for elem in da.basis_in_ambient_alg
            ]
            solve_kwargs = _solve_weight_kwargs(
                force_heavy_solve, surface_singularities, simplify_singularities
            )
            if surface_singularities:
                sol, singularities = solve_dgcv(eqns, variables, **solve_kwargs)
            else:
                sol = solve_dgcv(eqns, variables, **solve_kwargs)
            if len(sol) == 0:
                raise RuntimeError("failed to compute radical.")
            else:
                genSol = subs(genElem, sol[0])
                if surface_singularities:
                    sing = [subs(v, sol[0]) for v in singularities]
                    sing = [v for v in sing if get_free_symbols(v)]
                    if get_dgcv_settings_registry().get(
                        "simplify_singularity_ideals_by_default", True
                    ):
                        sing = expr_union_primitives(
                            sing,
                            order_coordinates(self._parameters),
                            process_rationals=True,
                            fail_quietly=True,
                        )
                    self._singularities["radical"] = sing
            freeVars = get_free_symbols(genSol)
            if self._parameters:
                freeVars = {v for v in freeVars if v not in self._parameters}
            if len(freeVars) != 0:
                freeVars = sorted(freeVars, key=str)
                zeroing = {v: 0 for v in freeVars}
                radSpanners = [genSol.subs({**zeroing, var: 1}) for var in freeVars]
            else:
                radSpanners = []
            if force_heavy_solve:
                radSpanners = [simplify(sp) for sp in radSpanners]
            self._radical_cache = self.subalgebra(radSpanners, span_warning=False)
            self._radical_heavy = bool(force_heavy_solve)
            clearVar(*listVar(temporary_only=True), report=False)
        return self._radical_cache

    def Levi_decomposition(
        self,
        decompose_semisimple_fully=False,
        _bust_cache=False,
        assume_Lie_algebra=False,
        verbose=False,
        surface_singularities=None,
        simplify_singularities=None,
        force_heavy_solve=False,
        _timed_reporting: bool | None = None,
        _reporting_threshold_s: float = 10,
        _progress_message: str | None = None,
        _on_timed_update=None,
    ):
        timed = bool(_timed_reporting) if _timed_reporting is not None else False
        threshold = float(_reporting_threshold_s)

        def _time_call(fn, step_desc: str, continue_desc: str | None):
            return _timed_progress_call(
                fn,
                timed=timed,
                threshold_s=threshold,
                step_desc=step_desc,
                continue_desc=continue_desc,
                progress_message=None,
                _on_timed_update=_on_timed_update,
            )

        if _bust_cache:
            self._radical_cache = None
            self._derived_series_cache = None
            self._lower_central_series_cache = None
            self._derived_subalg_cache = None
        if surface_singularities is None:
            surface_singularities = True if self._parameters else False
        surface_singularities = bool(surface_singularities)
        if surface_singularities:
            sing = []
        if self._Levi_deco_cache is None:
            if self._educed_properties.get("special_type", None) in {
                "simple",
                "semisimple",
            }:
                self._Levi_deco_cache = {
                    "LD_components": (self, self.subalgebra([])),
                    "simple_ideals": None,
                }
            elif self._educed_properties.get("special_type", None) in {
                "nilpotent",
                "solvable",
                "abelian",
            }:
                self._Levi_deco_cache = {
                    "LD_components": (self.subalgebra([]), self),
                    "simple_ideals": None,
                }
            else:
                if verbose is True:
                    print("Deriving (or retrieving) maximal solvable ideal...")

                rad = _time_call(
                    lambda: self.radical(
                        assume_Lie_algebra=assume_Lie_algebra,
                        surface_singularities=surface_singularities,
                        simplify_singularities=simplify_singularities,
                        force_heavy_solve=force_heavy_solve,
                    ),
                    "deriving the maximal solvable ideal",
                    "compute the max. solvable ideal's derived series",
                )
                if surface_singularities:
                    sing += getattr(self, "_singularities", {}).get("radical", [])
                    new_sing = self._singularities.get("LD", []) + [
                        v for v in sing if get_free_symbols(v)
                    ]
                    if get_dgcv_settings_registry().get(
                        "simplify_singularity_ideals_by_default", True
                    ):
                        new_sing = expr_union_primitives(
                            new_sing,
                            order_coordinates(self._parameters),
                            process_rationals=True,
                            fail_quietly=True,
                        )
                    self._singularities["LD"] = new_sing
                if len(rad.basis) > 0:
                    if verbose is True:
                        print(
                            "Finding a semisimple complement to the max. solvable ideal..."
                        )

                    rad_seq = _time_call(
                        lambda: rad.derived_series(
                            align_nested_bases=True,
                            surface_singularities=surface_singularities,
                            simplify_singularities=simplify_singularities,
                            force_heavy_solve=force_heavy_solve,
                        ),
                        "computing the max. solvable ideal's derived series",
                        "compute a semisimple complement to the maximal solvable ideal",
                    )
                    if surface_singularities:
                        sing += getattr(rad, "_singularities", {}).get(
                            "derived_series", set()
                        )
                        new_sing = self._singularities.get("LD", []) + [
                            v for v in sing if get_free_symbols(v)
                        ]
                        if get_dgcv_settings_registry().get(
                            "simplify_singularity_ideals_by_default", True
                        ):
                            new_sing = expr_union_primitives(
                                new_sing,
                                order_coordinates(self._parameters),
                                process_rationals=True,
                                fail_quietly=True,
                            )
                        self._singularities["LD"] = new_sing

                    def _compute_complement():
                        local_rad_seq = list(rad_seq) if rad_seq else []
                        if local_rad_seq and local_rad_seq[-1] == []:
                            local_rad_seq = local_rad_seq[:-1]  ###!!! note convention
                        local_rad_seq.append([])

                        discrep = self.dimension - len(local_rad_seq[0])
                        naiveBasis = []
                        augment_NB = list(local_rad_seq[0])
                        for elem in self.basis:
                            if len(naiveBasis) == discrep:
                                break
                            indep = _indep_check(
                                augment_NB,
                                elem,
                                surface_singularities=surface_singularities,
                                force_heavy_solve=force_heavy_solve,
                            )
                            if surface_singularities:
                                indep, sing = indep
                                new_sing = self._singularities.get("LD", []) + [
                                    v for v in sing if get_free_symbols(v)
                                ]
                                if get_dgcv_settings_registry().get(
                                    "simplify_singularity_ideals_by_default", True
                                ):
                                    new_sing = expr_union_primitives(
                                        new_sing,
                                        order_coordinates(self._parameters),
                                        process_rationals=True,
                                        fail_quietly=True,
                                    )
                                self._singularities["LD"] = new_sing
                            if indep:
                                augment_NB.append(elem)
                                naiveBasis.append(elem)
                        ss_dim = len(naiveBasis)

                        for idx in range(len(local_rad_seq)):
                            if idx == len(local_rad_seq) - 1:
                                compare_set = local_rad_seq[idx]
                                quot_set = []
                                rad_discrep = len(local_rad_seq[idx])
                            else:
                                rad_discrep = len(local_rad_seq[idx]) - len(
                                    local_rad_seq[idx + 1]
                                )
                                compare_set = local_rad_seq[idx][:rad_discrep]
                                quot_set = local_rad_seq[idx][rad_discrep:]
                            compLen = len(compare_set)

                            variables = []
                            basis_modifiers = []
                            for count in range(len(naiveBasis)):
                                if compLen > 0:
                                    w_sum, w_vars = linear_combination(
                                        compare_set, prefix=f"_v_{count}_"
                                    )
                                    variables += w_vars
                                    basis_modifiers.append(w_sum)
                                else:
                                    basis_modifiers.append(0 * naiveBasis[0])

                            leading_coeffs = {}
                            trailing_coeffs = {}
                            eqns = []
                            for idx1 in range(ss_dim):
                                for idx2 in range(idx1 + 1, ss_dim):
                                    w1, w2 = naiveBasis[idx1], naiveBasis[idx2]
                                    lb = w1 * w2
                                    surfacing = (
                                        True
                                        if self._parameters or surface_singularities
                                        else False
                                    )
                                    lb_decomp = _indep_check(
                                        naiveBasis + local_rad_seq[idx],
                                        lb,
                                        return_decomp_coeffs=True,
                                        surface_singularities=surfacing,
                                        force_heavy_solve=force_heavy_solve,
                                    )
                                    if lb_decomp[0] is True and not force_heavy_solve:
                                        dgcv_warning(
                                            "The Levi decomposition algorithm encountered a bug caused by solver failing to recognize a zero. Retrying now with the heavier solve algorithm.",
                                            wc_label="debug_log",
                                        )
                                        lb_decomp = _indep_check(
                                            naiveBasis + local_rad_seq[idx],
                                            lb,
                                            return_decomp_coeffs=True,
                                            surface_singularities=surfacing,
                                            _force_eqn_simiplify=True,
                                            force_heavy_solve=True,
                                        )

                                    if surfacing:
                                        new_sing = self._singularities.get("LD", []) + [
                                            v
                                            for v in lb_decomp[2]
                                            if get_free_symbols(v)
                                        ]
                                        if get_dgcv_settings_registry().get(
                                            "simplify_singularity_ideals_by_default",
                                            True,
                                        ):
                                            new_sing = expr_union_primitives(
                                                new_sing,
                                                order_coordinates(self._parameters),
                                                process_rationals=True,
                                                fail_quietly=True,
                                            )
                                        self._singularities["LD"] = new_sing
                                    if lb_decomp[0] is True:
                                        raise RuntimeError(
                                            "the dgcv Levi decomposition algorithm could "
                                            "not express a bracket of complement basis "
                                            f"elements {idx1} and {idx2} within the span "
                                            f"of the naive complement and level {idx} of "
                                            f"{len(local_rad_seq) - 1} of the radical's "
                                            "derived series. Either the linear solver "
                                            "failed to recognize a vanishing expression, "
                                            "or an earlier step produced a naive "
                                            "complement that does not complement the "
                                            f"radical (complement dimension {ss_dim}, "
                                            f"expected {discrep}; comparison set size "
                                            f"{compLen}, level size {rad_discrep})."
                                        )
                                    lb_decomp = lb_decomp[1][0]
                                    leading_coeffs[(idx1, idx2)] = [
                                        lb_decomp.get(idx, 0) for idx in range(ss_dim)
                                    ]
                                    trailing_coeffs[(idx1, idx2)] = [
                                        lb_decomp.get(idx, 0)
                                        for idx in range(ss_dim, ss_dim + compLen)
                                    ]

                            for idxs in leading_coeffs:
                                oldV_sum = zip_sum(trailing_coeffs[idxs], compare_set)
                                vTerms_sum = -zip_sum(
                                    leading_coeffs[idxs], basis_modifiers
                                )
                                newV = (
                                    naiveBasis[idxs[0]] * basis_modifiers[idxs[1]]
                                    - naiveBasis[idxs[1]] * basis_modifiers[idxs[0]]
                                )
                                qTerms_sum, t_vars = linear_combination(
                                    quot_set, prefix=f"tv_{idxs[0]}_{idxs[1]}_"
                                )
                                variables += t_vars
                                eqns.append(oldV_sum + vTerms_sum + qTerms_sum + newV)
                            if force_heavy_solve:
                                eqns = [simplify(eqn) for eqn in eqns]
                            solve_kwargs = _solve_weight_kwargs(
                                force_heavy_solve,
                                surface_singularities,
                                simplify_singularities,
                            )
                            if surface_singularities:
                                sol, _ = solve_dgcv(eqns, variables, **solve_kwargs)
                            else:
                                sol = solve_dgcv(eqns, variables, **solve_kwargs)
                            if len(sol) == 0:
                                if not all(
                                    getattr(eqn, "is_zero", False) for eqn in eqns
                                ):
                                    dgcv_warning(
                                        f"eqn: {eqns},\\n variables{variables},\\n sol: {sol}",
                                        wc_label="debug_log",
                                    )
                                    raise RuntimeError(
                                        "solver failed during the dgcv Levi decomposition algorithm."
                                    )
                                new_basis = list(naiveBasis)
                            else:
                                new_basis = [
                                    (w + v).subs(sol[0])
                                    for w, v in zip(naiveBasis, basis_modifiers)
                                ]
                            free_variables = set()
                            for nb in new_basis:
                                for j in nb.coeff_dict.values():
                                    free_variables |= set(get_free_symbols(j))
                            free_variables = {
                                x for x in free_variables if x in variables
                            }
                            if len(free_variables) > 0:
                                zeroing = {v: 0 for v in free_variables}
                                target = next(iter(free_variables))
                                new_basis = [
                                    subs(bv, {**zeroing, target: 1}) for bv in new_basis
                                ]
                            if force_heavy_solve:
                                new_basis = [simplify(bv) for bv in new_basis]
                            naiveBasis = new_basis
                        return self.ambient.subalgebra(
                            naiveBasis, span_warning=True, simplify_basis=True
                        )

                    Levi_component = _time_call(
                        _compute_complement,
                        "computing a semisimple complement to the max. solvable ideal",
                        "decompose the semisimple component into simple ideals"
                        if decompose_semisimple_fully
                        else _progress_message,
                    )
                else:
                    Levi_component = self

                self._Levi_deco_cache = {
                    "LD_components": (Levi_component, rad),
                    "simple_ideals": None,
                }

        if (
            decompose_semisimple_fully is True
            and self._Levi_deco_cache.get("LD_components", None) is not None
            and self._Levi_deco_cache.get("simple_ideals", 1) is None
        ):
            if verbose is True:
                print("Decomposing semisimple subalgebra into simple subalgebras...")

            Levi_component, rad = self._Levi_deco_cache.get("LD_components", None)

            def _decompose_semisimple():
                simples = decompose_semisimple_algebra(
                    Levi_component,
                    format_as_lists_of_elements=True,
                    surface_singularities=surface_singularities,
                    simplify_singularities=simplify_singularities,
                )
                if surface_singularities:
                    simples, sing = simples
                new_basis = []
                simple_ideals = []
                for comp in simples:
                    new_basis += comp
                    simple_ideals.append(
                        Levi_component.subalgebra(comp, simplify_basis=True)
                    )
                new_Levi = Levi_component.subalgebra(new_basis)
                if surface_singularities:
                    return new_Levi, tuple(simple_ideals), sing
                return new_Levi, tuple(simple_ideals)

            out = _time_call(
                _decompose_semisimple,
                "decomposing algebra into simple ideals",
                _progress_message,
            )
            if surface_singularities:
                new_Levi, simple_ideals, sing = out
                new_sing = self._singularities.get("simple_ideals", []) + [
                    v for v in sing if get_free_symbols(v)
                ]
                if get_dgcv_settings_registry().get(
                    "simplify_singularity_ideals_by_default", True
                ):
                    new_sing = expr_union_primitives(
                        new_sing,
                        order_coordinates(self._parameters),
                        process_rationals=True,
                        fail_quietly=True,
                    )
                self._singularities["simple_ideals"] = new_sing
            else:
                new_Levi, simple_ideals = out
            self._Levi_deco_cache["LD_components"] = (new_Levi, rad)
            self._Levi_deco_cache["simple_ideals"] = simple_ideals

        return self._Levi_deco_cache.get("LD_components", None)

    def center(
        self,
        surface_singularities: bool = None,
        simplify_singularities: bool = None,
        format_as_subalgebra=True,
    ):
        if surface_singularities is None:
            surface_singularities = True if self._parameters else False
        if self._center_cache is None:
            if self.dimension == 0:
                self._center_cache = self.subalgebra([])
                if format_as_subalgebra:
                    return self._center_cache
                return self._center_cache.basis
            gene, variables = linear_combination(self.basis)
            eqns = [gene * elem for elem in self.basis]
            if not self.is_skew_symmetric():
                eqns += [elem * gene for elem in self.basis]
            if surface_singularities is True:
                sol, sing = solve_dgcv(
                    eqns,
                    variables,
                    return_divisors=True,
                    pass_to_symbolic_engine=False,
                    simplify_pivots=simplify_singularities
                    if simplify_singularities is not None
                    else True,
                    simplify_result=False,
                )
                if not sol:
                    raise RuntimeError("failed to compute the center.") from None
                sol = sol[0]
                if get_dgcv_settings_registry().get(
                    "simplify_singularity_ideals_by_default", True
                ):
                    self._singularities["center"] = expr_union_primitives(
                        [v for v in sing if get_free_symbols(v)],
                        order_coordinates(self._parameters),
                        process_rationals=True,
                        fail_quietly=True,
                    )
                else:
                    self._singularities["center"] = [
                        v for v in sing if get_free_symbols(v)
                    ]
            else:
                sol = solve_dgcv(eqns, variables, simplify_result=False)
                if not sol:
                    raise RuntimeError("failed to compute the center.") from None
                sol = sol[0]
            gsol = subs(gene, sol)
            fv = set()
            vset = set(variables)
            for v in variables:
                fv |= {x for x in get_free_symbols(sol.get(v)) if x in vset}
            if len(fv) == 0:
                self._center_cache = self.subalgebra([])
            else:
                fv = sorted(fv, key=str)
                zeroing = {v: 0 for v in fv}
                self._center_cache = self.subalgebra(
                    [subs(gsol, {**zeroing, v: 1}) for v in fv]
                )
        if format_as_subalgebra:
            return self._center_cache
        return self._center_cache.basis

    def approximate_rank(
        self,
        check_semisimple=False,
        assume_semisimple=False,
        _use_cache=False,
        surface_singularities=False,
        simplify_singularities=None,
    ):
        if self.dimension == 0:
            self._rank_approximation = 0
            if surface_singularities:
                return 0, []
            return 0
        if check_semisimple is True:
            ssc = self.is_semisimple()
            if ssc is True:
                assume_semisimple = True
            elif assume_semisimple is True:
                print(
                    "approximate_rank received parameters `check_semisimple=True` and `assume_semisimple=True`, but the semisimple check returned false. The algorithm is proceeding with the `assume_semisimple` logic applied, but this is likely not wanted, and should be prevented by setting those parameters differently. Note, just setting `check_semisimple=True` is enough to use optimized algorithms in the event that the semisimple check returns true, whereas `assume_semisimple` should only be used in applications where forgoing the semisimple check entirely is wanted."
                )
        if _use_cache and self._rank_approximation is not None:
            if surface_singularities:
                return self._rank_approximation, []
            return self._rank_approximation
        power = (
            1
            if (assume_semisimple or self._is_semisimple_cache is True)
            else self.dimension
        )
        get_slice = self._structure_data_slice
        elem = matrix_dgcv(get_slice(0), shape=self.structureData.shape)  # test element
        bound = max(100, 10 * self.dimension)
        for idx in range(1, self.dimension):
            elem2 = get_slice(idx)
            elem += random.randint(1, bound) * matrix_dgcv(
                elem2, shape=self.structureData.shape
            )
        rank_result = fast_rank(
            elem**power,
            surface_singularities=surface_singularities,
            simplify_singularities=simplify_singularities,
        )
        if surface_singularities:
            rank, divisors = rank_result
        else:
            rank = rank_result
        rank = self.dimension - rank
        if not isinstance(rank, numbers.Integral):
            dgcv_warning(
                "`approximate_rank` obtained a non-integral rank "
                f"({rank}); the cached rank approximation was left unchanged."
            )
        elif (
            not isinstance(self._rank_approximation, numbers.Integral)
            or self._rank_approximation > rank
        ):
            self._rank_approximation = rank
        if surface_singularities:
            return self._rank_approximation, divisors
        return self._rank_approximation

    def summary(
        self,
        generate_full_report: bool = False,
        generate_partial_report: bool = False,
        theme=None,
        use_latex=None,
        *,
        plain_text: bool = False,
        return_displayable: bool = False,
        show_singularities: bool | None = None,
        interrupt_to_partial_report: bool = True,
        force_heavy_solve: bool = False,
        _reporting_threshold_s: float = 7.0,
        **kwargs,
    ):
        dgcvSR = get_dgcv_settings_registry()

        if not isinstance(theme, str):
            theme = kwargs.get("style", None)
            if theme is None:
                theme = dgcvSR.get("theme", "dark")
        if use_latex is None:
            use_latex = dgcvSR.get("use_latex")

        if (plain_text is False) and (not is_rich_displaying_available()):
            plain_text = True

        extra_support_for_math_in_tables = bool(
            dgcvSR.get("extra_support_for_math_in_tables")
        )

        subAlg = get_dgcv_category(self) == "subalgebra"
        parentAlg = self.ambient

        if use_latex and not plain_text:
            algebra_name, algebra_name_cap = _alg_name_latex(parentAlg)
        else:
            algebra_name = (
                parentAlg.label if getattr(parentAlg, "label", None) else "the algebra"
            )
            algebra_name_cap = (
                parentAlg.label if getattr(parentAlg, "label", None) else "The algebra"
            )

        reporting = bool(generate_full_report or generate_partial_report)
        threshold = float(_reporting_threshold_s)
        updates_printed = 0
        interrupted = False

        def _on_update():
            nonlocal updates_printed
            updates_printed += 1

        if reporting:
            try:
                _summary_warm_caches(
                    self,
                    subAlg=subAlg,
                    reporting_threshold_s=threshold,
                    progress_message="finish building the summary",
                    full=generate_full_report,
                    force_heavy_solve=force_heavy_solve,
                    _on_timed_update=_on_update,
                )
            except KeyboardInterrupt:
                if interrupt_to_partial_report is False:
                    raise
                interrupted = True
                updates_printed += 1
                try:
                    clearVar(*listVar(temporary_only=True), report=False)
                except Exception:
                    pass
                print(
                    "\nInterrupted. Rendering the report from results computed so far. "
                    "Results already cached are retained, so re-running summary resumes "
                    "from where this left off."
                )

        report_full = generate_full_report and not interrupted

        if plain_text:
            out = _timed_progress_call(
                lambda: _summary_render_plain(
                    parentAlg,
                    self,
                    subAlg=subAlg,
                    algebra_name=algebra_name,
                    algebra_name_cap=algebra_name_cap,
                ),
                timed=reporting,
                threshold_s=threshold,
                step_desc="rendering the summary",
                continue_desc=None,
                progress_message=None,
                _on_timed_update=_on_update,
            )
            if updates_printed:
                print()
            if return_displayable:
                return out
            print(out)
            return

        out = _timed_progress_call(
            lambda: _summary_render_rich(
                refAlg=self,
                subAlg=subAlg,
                algebra_name=algebra_name,
                algebra_name_cap=algebra_name_cap,
                style=theme,
                use_latex=use_latex,
                extra_support_for_math_in_tables=extra_support_for_math_in_tables,
                show_singularities=show_singularities,
                full=report_full,
            ),
            timed=reporting,
            threshold_s=threshold,
            step_desc="rendering the summary",
            continue_desc=None,
            progress_message=None,
            _on_timed_update=_on_update,
        )
        if updates_printed:
            print()
        if return_displayable:
            return out
        show(out)

    def _structure_data_slice(self, idx):
        slices = self._structure_data_slices
        if slices is None:
            slices = dict()
            for (i, j, k), v in self.structureDataDict.items():
                slot = slices.get(i)
                if slot is None:
                    slices[i] = {(j, k): v}
                else:
                    slot[(j, k)] = v
            self._structure_data_slices = slices
        slot = slices.get(idx)
        return dict(slot) if slot else dict()

    def _weight_coordinates(self, element):
        if (
            get_dgcv_category(element) == "subalgebra_element"
            and element.algebra != self
            and element.algebra.ambient == self
        ):
            element = element.ambient_rep
        if get_dgcv_category(element) not in {"algebra_element", "subalgebra_element"}:
            raise TypeError(
                f"Input to `check_element_weight` must be an algebra element belonging to the {self._dgcv_category} instance whose `check_element_weight` is being called."
            ) from None
        if element.algebra != self:
            raise TypeError(
                f"Input to `check_element_weight` must be an algebra element belonging to the {self._dgcv_category} instance whose `check_element_weight` is being called."
            ) from None
        return element.coeff_dict


class algebra_class(_algebra_methods, dgcv_class):
    def __init__(
        self,
        structure_data,
        grading=None,
        format_sparse=False,
        process_matrix_rep=False,
        preferred_representation=None,
        simplify_products_by_default=None,
        assume_skew=False,
        matrix_representation=None,
        tensor_representation=None,
        _basis_labels_parent=None,
        _label=None,
        _basis_labels=None,
        _calledFromCreator=None,
        _callLock=None,
        _print_warning=None,
        _child_print_warning=None,
        _exclude_from_VMF=None,
        _markers={},
    ):
        if isinstance(structure_data, numbers.Integral):
            if structure_data >= 0:
                structure_data = _structure_array(dict(), structure_data)
        if _calledFromCreator == retrieve_passkey():
            validated_structure_data = structure_data
            params = _markers.get("parameters", set())
        else:
            try:
                vsd = _validate_structure_data(
                    structure_data,
                    process_matrix_rep=process_matrix_rep,
                    assume_skew=assume_skew,
                    assume_Lie_alg=False,
                    basis_order_for_supplied_str_eqns=False,
                )
                if process_matrix_rep is True:
                    if matrix_representation is not None:
                        dgcv_warning(
                            "The `algebra_class` initializer disregarded the optional parameter value given for `matrix_representation` because `process_matrix_rep` was set to `True`, which forces automated computation of the representation."
                        )
                    validated_structure_data, matrix_representation, params = (
                        vsd[0][0],
                        vsd[0][1],
                        vsd[0][2],
                    )
                else:
                    validated_structure_data, params = vsd
                    if not isinstance(params, set):
                        params = set()
                    params |= get_free_symbols(
                        validated_structure_data
                    )  ###!!! fix vsd to remove redundancy

            except dgcv_exception_note as e:
                raise SystemExit(e)
        self.structureData = validated_structure_data
        self.dimension = self.structureData.shape[0]
        self._parameters = params
        self._tex_label = None
        self._tex_basis_labels = None
        self._educed_properties = dict()

        def _assign_composite_labels():
            # If not registered, pick collision-free labels here; otherwise trust provided labels
            if _markers.get("registered", None) is False:
                incoming_tex_label = _markers.get("_tex_label", None)
                if incoming_tex_label is None:
                    self.label = unique_label(_label)
                    self._tex_label = None
                else:
                    self.label, self._tex_label = unique_label(
                        _label, tex_label=incoming_tex_label
                    )

                incoming_basis = list(_basis_labels or [])
                incoming_tex_basis = list(_markers.get("_tex_basis_labels", []) or [])
                have_tex_basis = len(incoming_basis) > 0 and len(
                    incoming_tex_basis
                ) == len(incoming_basis)

                new_basis = []
                new_tex_basis = [] if have_tex_basis else None
                batch_protected = set()
                if isinstance(self.label, str):
                    batch_protected.add(self.label)
                for idx, base_lbl in enumerate(incoming_basis):
                    candidate = base_lbl
                    if have_tex_basis:
                        base_tex = incoming_tex_basis[idx]
                        final_lbl, final_tex = unique_label(
                            candidate, tex_label=base_tex, protected=batch_protected
                        )
                        new_basis.append(final_lbl)
                        new_tex_basis.append(final_tex)
                        batch_protected.add(final_lbl)
                    else:
                        final_lbl = unique_label(candidate, protected=batch_protected)
                        new_basis.append(final_lbl)
                        batch_protected.add(final_lbl)

                self.basis_labels = new_basis
                if have_tex_basis:
                    self._tex_basis_labels = new_tex_basis
                elif self._tex_label is not None:
                    self._tex_basis_labels = [
                        f"{self._tex_label}_{{{i + 1}}}" for i in range(self.dimension)
                    ]
            else:
                self.label = _label
                self.basis_labels = _basis_labels
                if not self.basis_labels:
                    base = (
                        self.label
                        if isinstance(self.label, str) and self.label
                        else "_e"
                    )
                    self.basis_labels = [
                        f"{base}{i + 1}" for i in range(self.dimension)
                    ]
                if _markers.get("_tex_label", None) is not None:
                    self._tex_label = _markers["_tex_label"]
                if _markers.get("_tex_basis_labels", None) is not None:
                    self._tex_basis_labels = _markers["_tex_basis_labels"]
                elif self._tex_label is not None and self._tex_basis_labels is None:
                    self._tex_basis_labels = [
                        f"{self._tex_label}_{{{i + 1}}}" for i in range(self.dimension)
                    ]

        if _calledFromCreator == retrieve_passkey():
            if isinstance(_markers.get("_educed_properties", None), dict):
                self._educed_properties = _markers.get("_educed_properties", dict())
            if _markers.get("endo", False):
                if _label is None:
                    self.label = f"gl_{_markers.get('endo_label', '')}"
                    self._tex_label = (
                        f"\\mathfrak{{gl}}\\left({_markers.get('endo_tex', '')}\\right)"
                    )
                    self.basis_labels = [
                        f"{self.label}{i + 1}" for i in range(self.dimension)
                    ]
                    self._tex_basis_labels = [
                        f"{self._tex_label}_{{{i + 1}}}" for i in range(self.dimension)
                    ]
                else:
                    self.label = _label
                    self.basis_labels = _basis_labels
            elif _markers.get("sum", False) or _markers.get("prod", False):
                _assign_composite_labels()
            else:
                self.label = _label
                self.basis_labels = _basis_labels
            self._registered = True
        else:
            self.label = "Alg_" + create_key()
            if _basis_labels_parent is True:
                self.basis_labels = [
                    f"{self.label}{i + 1}" for i in range(self.dimension)
                ]
            else:
                self.basis_labels = [f"_e{i + 1}" for i in range(self.dimension)]
            self._registered = False
        self._basis_labels_parent = _basis_labels_parent
        self._callLock = _callLock
        self._print_warning = _print_warning
        self._child_print_warning = _child_print_warning
        self._exclude_from_VMF = _exclude_from_VMF
        self.is_sparse = format_sparse
        self.structureDataDict = _flatten_structure_data(
            self.structureData, _source="algebra_class"
        )
        self._built_from_matrices = process_matrix_rep
        self.simplify_products_by_default = simplify_products_by_default
        self.semidirect_decomposition = _markers.get("semidirect_decomposition", None)
        self.tensor_decomposition = _markers.get("tensor_decomposition", None)
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "algebra"
        if self._parameters:
            self._singularities = {
                "structure": _harvest_structure_singularities(
                    self.structureData, self._parameters
                )
            }
        else:
            self._singularities = {}

        numeric_types = expr_numeric_types()

        def validate_and_adjust_grading_vector(vector, dimension):
            vector = list(vector)
            if len(vector) < dimension:
                dgcv_warning(
                    f"Grading vector is shorter than the dimension ({len(vector)} < {dimension}). "
                    f"Padding with zeros to match the dimension."
                )
                vector += [0] * (dimension - len(vector))
            elif len(vector) > dimension:
                dgcv_warning(
                    f"Grading vector is longer than the dimension ({len(vector)} > {dimension}). "
                    f"Truncating to match the dimension.",
                )
                vector = vector[:dimension]

            for i, component in enumerate(vector):
                if not isinstance(component, numeric_types):
                    raise ValueError(
                        f"Invalid component in grading vector at index {i}: {component}. "
                        f"Expected scalar"
                    ) from None

            return tuple(vector)

        if grading is None:
            self.grading = [tuple([0] * self.dimension)]
        else:
            if isinstance(grading, (list, tuple)) and all(
                isinstance(g, (list, tuple)) for g in grading
            ):
                self.grading = [
                    validate_and_adjust_grading_vector(vector, self.dimension)
                    for vector in grading
                ]
            else:
                self.grading = [
                    validate_and_adjust_grading_vector(grading, self.dimension)
                ]

        self._gradingNumber = len(self.grading)

        for val, lab in zip(
            [matrix_representation, preferred_representation, tensor_representation],
            [
                "matrix_representation",
                "preferred_representation",
                "tensor_representation",
            ],
        ):
            if val is not None and (
                not isinstance(val, (list, tuple)) or len(val) != self.dimension
            ):
                raise TypeError(f"unsupported format for {lab}.") from None
        if preferred_representation is not None and all(
            isinstance(elem, matrix_dgcv) for elem in preferred_representation
        ):
            self._preferred_rep_type = "matrix"
            self._preferred_representation = preferred_representation
        elif preferred_representation is not None and all(
            isinstance(elem, tensorProduct) for elem in preferred_representation
        ):
            self._preferred_rep_type = "tensor"
            self._preferred_representation = preferred_representation
        elif preferred_representation is not None and all(
            isinstance(elem, (list, tuple)) for elem in preferred_representation
        ):
            self._preferred_rep_type = "matrix"
            self._preferred_representation = [
                matrix_dgcv(elem) for elem in preferred_representation
            ]
        elif preferred_representation is not None:
            raise TypeError(
                "unsupported format for `preferred_representation`."
            ) from None
        else:
            self._preferred_rep_type = None
            self._preferred_representation = None

        if matrix_representation is not None and all(
            isinstance(elem, matrix_dgcv) for elem in matrix_representation
        ):
            self._mat_rep = matrix_representation
        elif matrix_representation is not None and all(
            isinstance(elem, (list, tuple)) for elem in matrix_representation
        ):
            self._mat_rep = [matrix_dgcv(elem) for elem in matrix_representation]
        elif matrix_representation is not None:
            raise TypeError("unsupported format for `matrix_representation`.") from None
        else:
            self._mat_rep = None

        if tensor_representation is not None and all(
            isinstance(elem, tensorProduct) for elem in tensor_representation
        ):
            self._tensor_rep = tensor_representation
        elif tensor_representation is not None:
            raise TypeError("unsupported format for `tensor_representation`.") from None
        else:
            self._tensor_rep = None

        vsr = get_vs_registry()
        self.dgcv_vs_id = len(vsr)
        vsr.append(self)

        self.basis = tuple(
            [
                algebra_element_class(
                    self,
                    [1 if i == j else 0 for j in range(self.dimension)],
                    1,
                )
                for i in range(self.dimension)
            ]
        )
        # immutables
        self._basis_labels = tuple(_basis_labels) if _basis_labels else None
        self._grading = tuple(map(tuple, self.grading))
        # Caches
        self._skew_symmetric_cache = None
        self._jacobi_identity_cache = None
        self._lie_algebra_cache = None
        self._is_semisimple_cache = None
        self._is_simple_cache = None
        self._is_nilpotent_cache = None
        self._is_abelian_cache = None
        self._is_solvable_cache = None
        self._rank_approximation = None
        self._center_cache = None
        self._lower_central_series_cache = None
        self._lower_central_series_terminated = None
        self._lower_central_series_depth = None
        self._derived_series_cache = None
        self._derived_series_terminated = None
        self._derived_series_depth = None
        self._grading_compatible = None
        self._grading_report = None
        self._killing_form = None
        self._derived_subalg_cache = None
        self._radical_cache = None
        self._Levi_deco_cache = None
        self._graded_components = None
        self._endomorphisms = None
        self._coproduct = {idx: None for idx in range(self.dimension)}

    def _class_builder(self, coeff_dict, valence, format_sparse=False):
        ### build algebra element
        return algebra_element_class(
            self, coeff_dict, valence, format_sparse=format_sparse
        )

    @property
    def preferred_representation(self):
        if self._preferred_representation is None:
            if self._mat_rep is not None:
                dgcv_warning(
                    "A preferred representation format for this algebra was never set up, but a cached matrix representation was found and has been set as the default for `preferred_representation`."
                )
                self._preferred_rep_type = "matrix"
                self._preferred_representation = self._mat_rep
            elif self._tensor_rep is not None:
                dgcv_warning(
                    "A preferred representation format for this algebra was never set up, but a cached tensor product representation was found and has been set as the default for `preferred_representation`."
                )
                self._preferred_rep_type = "tensor"
                self._preferred_representation = self._tensor_rep
            else:
                dgcv_warning(
                    "A preferred representation format for this algebra was not specified, so it has been set to its adjoint representation."
                )
                self._preferred_rep_type = "matrix"
                self._preferred_representation = adjointRepresentation(self)
        return self._preferred_representation

    @property
    def tensor_representation(self):
        return self._tensor_rep

    @property
    def matrix_representation(self):
        return self._mat_rep

    @property
    def ambient(self):
        return self

    @property
    def basis_in_ambient_alg(self):
        return self.basis

    def _verbose_subject(self):
        return "The algebra" if self.label is None else str(self.label)

    @property
    def endomorphism_algebra(self):
        if self._endomorphisms is None:
            self._endomorphisms = vector_space_endomorphisms(self)
        return self._endomorphisms

    @property
    def zero_element(self):
        return algebra_element_class(self, (0,) * self.dimension, 1)

    def structure_equations(
        self,
        formatting: Optional[Literal["dict", "list"]] = "dict",
        new_basis_labels: Optional[str | List[str]] = None,
        abbreviate_for_skew_struct: bool = None,
        initial_index: int = 1,
        list_symbols_as_strings: bool = False,
    ):
        if new_basis_labels is not None:
            if (
                isinstance(new_basis_labels, (list, tuple))
                and len(new_basis_labels) >= self.dimension
                and len(set(new_basis_labels)) == len(new_basis_labels)
            ):
                atoms = [symbol(lab) for lab in new_basis_labels]
            elif isinstance(new_basis_labels, str):
                atoms = [
                    symbol(f"{new_basis_labels}{i + initial_index}")
                    for i in range(self.dimension)
                ]
            else:
                atoms = [symbol(str(lab)) for lab in self.basis]
        else:
            atoms = [symbol(str(lab)) for lab in self.basis]
        str_eqns = dict()
        if list_symbols_as_strings:
            atoms = [str(atom) for atom in atoms]

        if abbreviate_for_skew_struct is None:
            abbreviate_for_skew_struct = True if self.is_Lie_algebra() else False
        for i in range(self.dimension):
            start = (
                i + 1 if abbreviate_for_skew_struct and self.is_skew_symmetric else 0
            )
            for j in range(start, self.dimension):
                val = sum(
                    c * atoms[idx] for idx, c in self.structureData[i, j]._data.items()
                )
                if not _scalar_is_zero(val):
                    str_eqns[(atoms[i], atoms[j])] = val
        if formatting == "list":
            str_eqns = [[[k[0], k[1]], v] for k, v in str_eqns.items()]
        return annotated_container(
            [str_eqns, atoms],
            _dgcv_notes={
                "signature": "algebra_str_eqns",
                "skew_aware_sparse": abbreviate_for_skew_struct,
            },
        )

    def contains(self, items, return_basis_coeffs=False, strict_types=False):
        if isinstance(items, (list, tuple)):
            return [
                self.contains(item, return_basis_coeffs=return_basis_coeffs)
                for item in items
            ]

        if strict_types is False and items == 0:
            if return_basis_coeffs is True:
                return {}
            return True
        if strict_types is False and get_dgcv_category(items) == "tensorProduct":
            if (
                next(iter(items._vs_spring)) == self.dgcv_vs_id
                and len(items.vector_spaces) == 1
            ):
                k, v = next(iter(items.coeff_dict.items()))
                if len(k) == 3:
                    ne = v * (from_vsr(k[2]).basis[k[0]])
                    if _scalar_is_zero(k[1]):
                        ne = ne.dual()
                    return self.contains(ne)
        if get_dgcv_category(items) == "subalgebra_element":
            items = items.ambient_rep
        if (
            get_dgcv_category(items) == "algebra_element"
            and items.dgcv_vs_id == self.dgcv_vs_id
        ):
            if return_basis_coeffs:
                return dict(items.coeff_dict)
            else:
                return True
        return False

    def _set_product_protocol(self):
        if self.simplify_products_by_default is None:
            fast_types = fast_scalar_types()
            if any(
                not isinstance(j, fast_types) for j in self.structureDataDict.values()
            ):
                self.simplify_products_by_default = True
            else:
                self.simplify_products_by_default = False
        elif self.simplify_products_by_default is not True:
            self.simplify_products_by_default = False

    def __eq__(self, other):
        if not isinstance(other, algebra_class):
            return NotImplemented
        return self.dgcv_vs_id == other.dgcv_vs_id

    def __hash__(self):
        return hash(self.dgcv_vs_id)

    def __iter__(self):
        return iter(self.basis)

    def __getitem__(self, indices):
        if isinstance(indices, numbers.Integral):
            return self.basis[indices]
        elif isinstance(indices, (list, tuple)):
            if len(indices) == 1:
                return self.basis[indices[0]]
            elif len(indices) == 2:
                return self.structureData[indices[0], indices[1]]
            elif len(indices) == 3:
                return self.structureData[indices[0], indices[1]][indices[2]]
            raise TypeError(
                f"Expected one, two, or three indices. Received {len(indices)}: {indices}"
            ) from None
        else:
            raise TypeError(
                f"To access an algebra element or structure data component, provide one index for an element from the basis, two indices for a list of coefficients from the product  of two basis elements, or 3 indices for the corresponding entry in the structure array. Instead of an integer of list of integers, the following was given: {indices}"
            ) from None

    def _structure_data_summary(self):
        if self.dimension <= 4:
            return self.structureData
        return "Structure data array is too large to print. Access the `structureData` attribute for details."

    def __str__(self, VLP=None):
        if not self._registered:
            if (
                self._exclude_from_VMF == retrieve_passkey()
                or get_dgcv_settings_registry()["forgo_warnings"] is True
            ):
                pass
            elif self._callLock == retrieve_passkey() and isinstance(
                self._print_warning, str
            ):
                dgcv_warning(self._print_warning)
            else:
                dgcv_warning(
                    "This algebra instance was initialized without an assigned label. "
                    "It is recommended to initialize algebra objects with dgcv creator functions like `createFiniteAlg` instead -- or set `label` parameter if creating it via a dgcv class method."
                )

        reg = get_dgcv_settings_registry()
        VLP = (
            bool(VLP)
            if VLP is not None
            else bool(reg.get("verbose_label_printing", False))
        )
        if VLP is False:
            return str(self.label) if self.label else "Unnamed Algebra"

        return space_display(
            fmt="plain",
            basis_words=(lambda: [str(e) for e in self.basis]),
            dim=self.dimension,
            label=self.label,
            unlabeled_plain="Unnamed Algebra",
            max_dim=20,
            raw=True,
            abbrev=False,
            plain_wrapper="<{}>",
        )

    def _repr_latex_(self, verbose=False, abbrev=False, raw=False, **kwargs):
        if not self._registered:
            if (
                self._exclude_from_VMF == retrieve_passkey()
                or get_dgcv_settings_registry()["forgo_warnings"] is True
            ):
                pass
            elif self._callLock == retrieve_passkey() and isinstance(
                self._print_warning, str
            ):
                dgcv_warning(self._print_warning)
            else:
                dgcv_warning(
                    "This algebra instance was initialized without an assigned label. "
                    "It is recommended to initialize algebra objects with dgcv creator functions like `createFiniteAlg` instead -- or set `label` parameter if creating it via a dgcv class method."
                )

        reg = get_dgcv_settings_registry()
        if bool(reg.get("verbose_label_printing", False)) is False:
            return space_display(
                fmt="latex",
                basis_words=(),
                dim=self.dimension,
                label=self.label,
                label_tex=(
                    self._tex_label
                    if getattr(self, "_tex_label", None) is not None
                    else None
                ),
                mathfrak_label=True,
                unlabeled_tex=r"\text{Unnamed Algebra}",
                max_dim=20,
                raw=raw,
                abbrev=True,
                use_displaystyle=False,
                latex_wrapper=r"\langle {}\rangle",
            )

        if abbrev:
            return space_display(
                fmt="latex",
                basis_words=(),
                dim=self.dimension,
                label=self.label,
                label_tex=(
                    self._tex_label
                    if getattr(self, "_tex_label", None) is not None
                    else None
                ),
                mathfrak_label=True,
                unlabeled_tex=r"\text{Unnamed Algebra}",
                max_dim=20,
                raw=raw,
                abbrev=True,
                use_displaystyle=False,
                latex_wrapper=r"\langle {}\rangle",
            )

        return space_display(
            fmt="latex",
            basis_words=(lambda: [e._repr_latex_(raw=True) for e in self.basis]),
            dim=self.dimension,
            label=self.label,
            label_tex=(
                self._tex_label
                if getattr(self, "_tex_label", None) is not None
                else None
            ),
            mathfrak_label=True,
            unlabeled_tex=r"\text{Unnamed Algebra}",
            max_dim=20,
            raw=raw,
            abbrev=False,
            use_displaystyle=True,
            latex_wrapper=r"\langle {}\rangle",
        )

    def _latex(self, printer=None, raw=True, **kwargs):
        return self._repr_latex_(raw=raw)

    def _display_DGCV_hook(self):
        if not self._registered:
            if (
                self._exclude_from_VMF == retrieve_passkey()
                or get_dgcv_settings_registry()["forgo_warnings"] is True
            ):
                pass
            elif self._callLock == retrieve_passkey() and isinstance(
                self._print_warning, str
            ):
                dgcv_warning(self._print_warning)
            else:
                dgcv_warning(
                    "This algebra instance was initialized without an assigned label. "
                    "It is recommended to initialize algebra objects with dgcv creator functions like `createFiniteAlg` instead -- or set `label` parameter if creating it via a dgcv class method."
                )

        def format_algebra_label(label):
            r"""Wrap the algebra label in \mathfrak{} if all characters are lowercase, and subscript any numeric suffix."""
            if label and label[-1].isdigit():
                label_text = "".join(filter(str.isalpha, label))
                label_number = "".join(filter(str.isdigit, label))
                if label_text.islower():
                    return rf"\mathfrak{{{label_text}}}_{{{label_number}}}"
                return rf"{label_text}_{{{label_number}}}"
            elif label and label.islower():
                return rf"\mathfrak{{{label}}}"
            return label or "Unnamed Algebra"

        return format_algebra_label(self.label)

    def is_subspace_subalgebra(
        self,
        elements,
        return_structure_data=False,
        check_linear_independence=False,
        surface_singularities=None,
    ):
        """
        Checks if a set of elements is a subspace is a subalgebra. `check_linear_independence` will additional verify if provided spanning elements are a basis.

        Parameters
        ----------
        elements : list
            A list of algebra_element_class instances.
        return_structure_data : bool, optional
            If True, returns the structure constants for the subalgebra. Returned
            data becomes a dictionary
        check_linear_independence : bool, optional
            If True, a check of linear independence of basis elements is also performed

        Returns
        -------
        dict or bool
            - If return_structure_data=True, returns a dictionary with keys:
            - 'linearly_independent': True/False
            - 'closed_under_product': True/False
            - 'structure_data': 3D list of structure constants
            - Otherwise, returns True if the elements form a subspace subalgebra, False otherwise.
        """

        if surface_singularities is None and self._parameters:
            surface_singularities = True
        filtered_elem = self.filter_independent_elements(
            elements, surface_singularities=surface_singularities
        )
        if surface_singularities:
            filtered_elem, sing = filtered_elem
        new_dim = len(filtered_elem)
        linearly_independent = len(elements) == len(filtered_elem)
        closed_under_product = True
        skew = self.is_skew_symmetric()
        if return_structure_data is True:
            structure_data = _structure_array(dict(), new_dim)
        if not isinstance(return_structure_data, bool):
            return_structure_data = False
        for count, elem in enumerate(filtered_elem):
            if closed_under_product is False:
                break
            lIdx = count + 1 if skew else 0
            for j in range(lIdx, new_dim):
                if closed_under_product is False:
                    break
                product = elem * filtered_elem[j]
                ic = _indep_check(
                    filtered_elem,
                    product,
                    return_decomp_coeffs=return_structure_data,
                    surface_singularities=surface_singularities,
                )
                if surface_singularities:
                    passCheck, new_sing = (
                        ic if return_structure_data is False else (ic[0], ic[2])
                    )
                    if passCheck is True:
                        ic = _indep_check(
                            filtered_elem,
                            product,
                            return_decomp_coeffs=return_structure_data,
                            surface_singularities=True,
                            _force_eqn_simiplify=True,
                        )
                        passCheck, new_sing = (
                            ic if return_structure_data is False else (ic[0], ic[2])
                        )
                    sing += new_sing
                else:
                    passCheck = ic if return_structure_data is False else ic[0]
                if passCheck is True:
                    closed_under_product = False
                    structure_data = None
                elif return_structure_data:
                    coeff_array = matrix_dgcv(
                        {
                            idx: coeff
                            for idx, coeff in ic[1][0].items()
                            if not _scalar_is_zero(coeff)
                        },
                        shape=(new_dim, 1),
                    )
                    structure_data[count, j] = coeff_array
                    if skew:
                        structure_data[j, count] = -coeff_array
        if return_structure_data:
            out = {
                "linearly_independent": linearly_independent,
                "closed_under_product": closed_under_product,
                "structure_data": structure_data,
            }
        elif check_linear_independence:
            out = linearly_independent and closed_under_product
        else:
            out = closed_under_product
        if surface_singularities:
            return out, sing
        return out

    def check_grading_compatibility(
        self, verbose=False, test_weights=None, trust_test_weight_format=False
    ):
        """
        Checks if the algebra's structure constants are compatible with the assigned grading. Compatibility with alternate grading assignements can be tested by supplying alternate basis weights in the `test_weights` parameter

        Parameters
        ----------
        verbose : bool, optional (default=False)
            If True, prints detailed information about incompatibilities.
        test_weights : list of (lists of length self.dimension), optional
            elements in the inner lists represent weight values for basis elements to test compatibility against
        trust_test_weight_format : bool, optional (default=False)
            Set to True to forgo safeguard checks that test_weights is correctly formatted

        Returns
        -------
        bool
            True if the algebra is compatible with all assigned grading vectors or given test weights, False otherwise.

        Notes
        -----
        - The algebra's zero element (weights labeled as 'AllW') are treated as compatible with all grading vectors.
        - Non-homogeneous elements (weights labeled as 'NoW') are treated as incompatible.
        """
        default_check = False
        if test_weights is None:
            default_check = True
            test_weights = self.grading
        elif trust_test_weight_format is False:
            if not isinstance(test_weights, (list, tuple)) or not all(
                isinstance(j, (list, tuple)) and len(j) == self.dimension
                for j in test_weights
            ):
                raise TypeError(
                    "The `test_weights` parameter in `algebra_class.weighted_component` must be a list/tuple of lists/tuples that length matches `algebra_class.dimension` and whose elements are weight values representing weights elements in `algebra_class.basis`."
                )

        if default_check is True and not self._gradingNumber:
            raise ValueError(
                "No grading vectors are assigned to this algebra instance."
            ) from None
        if (
            default_check is True
            and isinstance(self._grading_compatible, bool)
            and self._grading_report
        ):
            compatible = self._grading_compatible
            failure_details = self._grading_report
        else:
            compatible = True
            failure_details = []

            for i, el1 in enumerate(self.basis):
                for j, el2 in enumerate(self.basis):
                    product = el1 * el2
                    product_weights = self.check_element_weight(
                        product,
                        test_weights=None if default_check else test_weights,
                    )

                    for g, grading_vector in enumerate(test_weights):
                        expected_weight = grading_vector[i] + grading_vector[j]

                        if product_weights[g] == "AllW":
                            continue  # Zero product is compatible with all weights

                        if (
                            product_weights[g] == "NoW"
                            or product_weights[g] != expected_weight
                        ):
                            compatible = False
                            failure_details.append(
                                {
                                    "grading_vector_index": g + 1,
                                    "basis_elements": (i + 1, j + 1),
                                    "weights": (grading_vector[i], grading_vector[j]),
                                    "expected_weight": expected_weight,
                                    "actual_weight": product_weights[g],
                                }
                            )
            if default_check is True:
                self._grading_compatible = compatible
                self._grading_report = failure_details

        if verbose:
            if not compatible:
                print("Grading Compatibility Check Failed:")
                for failure in failure_details:
                    print(
                        f"- Grading Vector {failure['grading_vector_index']}: "
                        f"Basis elements {failure['basis_elements'][0]} and {failure['basis_elements'][1]} "
                        f"(weights: {failure['weights'][0]}, {failure['weights'][1]}) "
                        f"produced weight {failure['actual_weight']}, expected {failure['expected_weight']}."
                    )
            else:
                if default_check:
                    ps = "all of its assigned Z-gradings."
                else:
                    ps = "the given weight system."
                print(f"The algebra structure of {self.label} is compatible with " + ps)
        return compatible

    def compute_center(self, for_associative_alg=False, assume_Lie_algebra=False):
        """
        Computes the center of the algebra as a subspace.

        Parameters
        ----------
        for_associative_alg : bool, optional
            If True, computes the center for an associative algebra. Defaults to False (assumes Lie algebra).

        Returns
        -------
        list
            A list of algebra_element_class instances that span the center of the algebra.

        Raises
        ------
        ValueError
            If `for_associative_alg` is False and the algebra is not a Lie algebra.

        Notes
        -----
        - For Lie algebras, the center is the set of elements `z` such that `z * x = 0` for all `x` in the algebra.
        - For associative algebras, the center is the set of elements `z` such that `z * x = x * z` for all `x` in the algebra.
        """
        self._set_product_protocol()

        if for_associative_alg is True:
            assume_Lie_algebra = False
        elif assume_Lie_algebra is False and not self.is_Lie_algebra():
            raise ValueError(
                "This algebra is not a Lie algebra. To compute the center for an associative algebra, set for_associative_alg=True."
            ) from None

        el, temp_vars = linear_combination(self.basis)
        if for_associative_alg:
            eqns = [
                v
                for other in self.basis
                for v in (el * other - other * el).coeff_dict.values()
            ]
        else:
            eqns = [v for other in self.basis for v in (el * other).coeff_dict.values()]

        solutions = solve_dgcv(
            eqns, temp_vars, method="linsolve", simplify_result=False
        )
        if not solutions:
            dgcv_warning(
                "The internal solver (determined by whichever symbolic engine set in defaults) returned no solutions, indicating that this computation of the center failed, as solutions do exist. An empty list is being returned."
            )
            return []

        el_sol = el.subs(solutions[0])

        free_variable_set = set()
        for j in el_sol.coeff_dict.values():
            free_variable_set |= set(get_free_symbols(j))
        if not free_variable_set:
            return []
        free_variables = tuple(sorted(free_variable_set, key=str))
        zeroing = {v: 0 for v in free_variables}

        return_list = []
        for var in free_variables:
            return_list.append(el_sol.subs({**zeroing, var: 1}))

        return return_list  ###!!! return subalgebra instead

    def get_structure_matrix(self, table_format=True, style=None):
        """
        Computes the structure matrix for the algebra.

        Parameters
        ----------
        table_format : bool, optional
            If True (default), returns a nicely formatted table.
            If False, returns a raw list of lists.
        style : str, optional
            table themes.

        Returns
        -------
        list of lists
            The structure matrix, whose (j, k)-entry is `basis[j] * basis[k]`.

        Notes
        -----
        - `table_format` and `style` are retained for backwards compatibility
          and have no effect. Use `multiplication_table` for rendered output.
        - If `basis_labels` is None, defaults to "_e1", "_e2", ..., "_e{d}".
        """
        if table_format is not True or style is not None:
            dgcv_warning(
                "`algebra_class.get_structure_matrix` ignores its `table_format` "
                "and `style` parameters and always returns a list of lists. Use "
                "`algebra_class.multiplication_table` for rendered output."
            )

        dimension = self.dimension
        structure_matrix = [
            [(self.basis[j] * self.basis[k]) for k in range(dimension)]
            for j in range(dimension)
        ]
        return structure_matrix

    def is_ideal(self, subspace_elements, assume_basis=False):
        """
        Checks if the given list of algebra elements spans an ideal.

        Parameters
        ----------
        subspace_elements : list
            A list of algebra_element_class instances representing the subspace
            they span.

        Returns
        -------
        bool
            True if the subspace is an ideal, False otherwise.

        Raises
        ------
        ValueError
            If the provided elements do not belong to this algebra.
        """
        if subspace_elements == self:
            return True
        if (
            isinstance(subspace_elements, algebra_subspace_class)
            and subspace_elements.ambient == self
        ):
            subspace_elements = subspace_elements.basis
            assume_basis = True
        else:
            for el in subspace_elements:
                if not isinstance(el, algebra_element_class) or el.algebra != self:
                    raise ValueError(
                        "All elements in subspace_elements must belong to this algebra."
                    ) from None

        if assume_basis:
            b_product = wedge(*subspace_elements)
        skew = self.is_skew_symmetric()
        for el in subspace_elements:
            for other in self.basis:
                products = [el * other] if skew else [el * other, other * el]
                for product in products:
                    if assume_basis:
                        if wedge(product, b_product).is_zero:
                            return False
                    else:
                        if not self.is_in_span(
                            product, subspace_elements, assume_basis=False
                        ):
                            return False
        return True

    def multiplication_table(
        self,
        elements=None,
        restrict_to_subspace=False,
        theme=None,
        use_latex=None,
        plain_text: bool | None = None,
        return_displayable: bool = False,
        col_number_limit: int = 10,
        row_number_limit: int = 15,
        cell_char_lim: int = 20,
        table_css_properties: str = None,
        _called_from_subalgebra=None,
        **kwargs,
    ):
        if elements is None:
            elements = self.basis
        elif not all(
            isinstance(elem, algebra_element_class) and elem.algebra == self
            for elem in elements
        ):
            raise ValueError(
                "All elements must be instances of algebraElement."
            ) from None

        if restrict_to_subspace is True:
            basis_elements = elements
        elif isinstance(restrict_to_subspace, (list, tuple)) and all(
            isinstance(elem, algebra_element_class) and elem.algebra == self
            for elem in restrict_to_subspace
        ):
            basis_elements = restrict_to_subspace
        elif (
            isinstance(_called_from_subalgebra, dict)
            and _called_from_subalgebra.get("internalLock", None) == retrieve_passkey()
        ):
            basis_elements = _called_from_subalgebra["basis"]
        else:
            basis_elements = self.basis

        c_limited, r_limited = False, False
        if col_number_limit < len(elements):
            c_limited = True
            elements = elements[:col_number_limit]
        if row_number_limit < len(basis_elements):
            r_limited = True
            basis_elements = basis_elements[:row_number_limit]

        dgcvSR = get_dgcv_settings_registry()

        if not is_rich_displaying_available():
            plain_text = True

        if plain_text:
            from dgcv._aux.printing.printing._data_structures import (
                format_unicode_table,
            )

            c_aug = ["⋯"] if c_limited else []
            r_aug = ["⋮"] if r_limited else []
            headers = [str(e) for e in elements] + c_aug
            index_headers = [str(e) for e in basis_elements] + r_aug
            corner_aug = [" "] if c_limited else []
            data = []
            for left in basis_elements:
                data.append([str(left * right) for right in elements] + c_aug)
            if r_limited:
                data.append(["⋮" for _ in range(len(elements))] + corner_aug)

            out = format_unicode_table(
                data,
                row_labels=index_headers,
                column_labels=headers,
                caption="Multiplication Table",
                cell_char_lim=cell_char_lim,
                align="center",
                header_align="center",
                row_label_align="left",
            )

            if return_displayable:
                return out
            print(out)
            return

        if use_latex is None:
            use_latex = dgcvSR.get("use_latex", False)
        if not isinstance(theme, str):
            style_key = kwargs.get("style", None) or dgcvSR.get("theme", "dark")
        else:
            style_key = theme

        def _to_string(element, ul=False):
            if ul:
                s = element._repr_latex_(verbose=False)
                if s.startswith("$") and s.endswith("$"):
                    s = s[1:-1]
                s = (
                    s.replace(r"\\displaystyle", "")
                    .replace(r"\displaystyle", "")
                    .strip()
                )
                return f"${s}$"
            return str(element)

        headers = [_to_string(e, ul=use_latex) for e in elements]
        if c_limited:
            headers += [r"$\cdots$"] if use_latex else ["⋯"]
        index_headers = [_to_string(e, ul=use_latex) for e in basis_elements]
        if r_limited:
            index_headers += [r"$\vdots$"] if use_latex else ["︙"]

        data = []
        for left in basis_elements:
            row = [_to_string(left * right, ul=use_latex) for right in elements]
            if c_limited:
                row += [r"$\cdots$"] if use_latex else ["⋯"]
            data.append(row)
        if r_limited:
            corner_aug = [] if not c_limited else [r"$\ddots$"] if use_latex else ["⋱"]
            vdots = r"$\vdots$" if use_latex else "⋮"
            data.append([vdots for _ in range(len(elements))] + corner_aug)

        theme_string = get_style(style_key)

        extra_css_override = textwrap.dedent("""
            .dgcv-data-table { 
                table-layout: auto; 
            }
        """).strip()

        table = build_matrix_table(
            index_labels=index_headers,
            columns=headers,
            rows=data,
            caption="Multiplication Table",
            theme_css_vars=theme_string,
            extra_css=extra_css_override
            if table_css_properties is None
            else table_css_properties,
            mirror_header_to_index=True,
            dashed_corner=True,
            header_underline_exclude_index=True,
            cell_align="center",
            escape_cells=False,
            escape_headers=False,
            escape_index=False,
            table_scroll=True,
            nowrap=True,
            hover_mode="cell",
            ul=0,
            ur=0,
            ll=0,
            lr=0,
        )
        out = (
            latex_in_html(
                table,
                container_id=table.container_id,
                katex_selector=".dgcv-data-table",
            )
            if use_latex
            else latex_in_html(table, extra_support_for_math_in_tables=False)
        )
        if return_displayable:
            return out
        show(out)
        return

    def subalgebra(
        self,
        basis,
        grading=None,
        span_warning=True,
        simplify_basis=False,
        simplify_products_by_default=None,
        surface_singularities=None,
    ):
        from .algebras_secondary import subalgebra_class

        if surface_singularities is None and self._parameters:
            surface_singularities = True

        if simplify_products_by_default is None:
            simplify_products_by_default = self.simplify_products_by_default
        if get_dgcv_category(basis) in {"algebra_subspace", "algebra"}:
            basis = basis.basis
        use_slices = True
        subIndices = set()
        index_map = dict()
        pos = self._basis_index
        for count, elem in enumerate(basis):
            try:
                idx = pos.get(elem)
            except TypeError:
                idx = None
            if idx is None:
                use_slices = False
                break
            index_map[idx] = count
            subIndices.add(idx)
        if use_slices:
            sub_dim = len(subIndices)
            if sub_dim != len(basis):
                raise ValueError(
                    "The basis provided to `algebra_class.subalgebra` contains "
                    f"repeated elements ({len(basis)} given, {sub_dim} distinct)."
                ) from None
            sub_order = sorted(subIndices, key=index_map.get)

            def truncateBySubInd(li):
                return [li[j] for j in sub_order]

            def restrict_structure_data(data):
                new_data = dict()
                inner_shape = (sub_dim, 1)
                for (i, j, k), v in data.items():
                    if i in subIndices and j in subIndices:
                        if k in subIndices:
                            outer_key = (index_map[i], index_map[j])
                            if outer_key in new_data:
                                new_data[outer_key][index_map[k]] = v
                            else:
                                new_data[outer_key] = matrix_dgcv(
                                    {index_map[k]: v}, shape=inner_shape
                                )
                        elif v is not None and not _scalar_is_zero(v):
                            raise TypeError(
                                "The basis provided to the `algebra_class.subalgebra` method does not span a subalgebra."
                            )
                return _structure_array(new_data, sub_dim)

            if isinstance(grading, (list, tuple)) and all(
                isinstance(elem, (list, tuple)) for elem in grading
            ):
                gradings = grading
            else:
                if grading is not None:
                    dgcv_warning(
                        "The `gradings` keyword given to `algebra_class.subalgebra` was in an unsupported format (i.e., not list of lists), so a valid alternate gradings vector was computed instead inherited from the parent algebra."
                    )
                gradings = [truncateBySubInd(vector) for vector in self.grading]
            structureData = restrict_structure_data(self.structureDataDict)
            return subalgebra_class(
                basis,
                self,
                grading=gradings,
                _compressed_structure_data=structureData,
                _internal_lock=retrieve_passkey(),
            )
        if simplify_basis:
            basis = list(
                self.filter_independent_elements(
                    basis,
                    apply_light_basis_simplification=True,
                    surface_singularities=False,
                )
            )
        testStruct = self.is_subspace_subalgebra(
            basis,
            return_structure_data=True,
            surface_singularities=surface_singularities,
        )
        if surface_singularities:
            testStruct, sing = testStruct
            new_sing = [v for v in sing if get_free_symbols(v)]
            if get_dgcv_settings_registry().get(
                "simplify_singularity_ideals_by_default", True
            ):
                new_sing = expr_union_primitives(
                    new_sing,
                    order_coordinates(self._parameters),
                    process_rationals=True,
                    fail_quietly=True,
                )
            ks = {"basis": new_sing}
        else:
            ks = None
        if testStruct["closed_under_product"] is not True:
            raise TypeError(
                "The basis provided to the `algebra_class.subalgebra` method does not span a subalgebra. Suggestion: use `algebra_class.subspace` instead."
            ) from None
        return subalgebra_class(
            basis,
            self,
            grading=grading,
            _compressed_structure_data=testStruct["structure_data"],
            _internal_lock=retrieve_passkey(),
            span_warning=span_warning,
            simplify_basis=False,
            simplify_products_by_default=simplify_products_by_default,
            _known_singularities=ks,
        )

    def new_alg_from_subalgebra(
        self,
        basis,
        grading=None,
        span_warning=True,
        simplify_basis=False,
        label=None,
        basis_labels=None,
        register_in_vmf=False,
        initial_basis_index=None,
        simplify_products_by_default=None,
    ):
        if simplify_products_by_default is None:
            simplify_products_by_default = self.simplify_products_by_default
        if get_dgcv_category(basis) == "subalgebra" and basis.ambient == self:
            alg = basis
        else:
            alg = self.subalgebra(
                basis,
                grading=grading,
                span_warning=span_warning,
                simplify_basis=simplify_basis,
                simplify_products_by_default=simplify_products_by_default,
            )
        return alg.copy(
            label=label,
            basis_labels=basis_labels,
            register_in_vmf=register_in_vmf,
            initial_basis_index=initial_basis_index,
            simplify_products_by_default=simplify_products_by_default,
        )

    def killing_form_product(self, elem1, elem2, assume_Lie_algebra=False):
        if not self.contains(elem1, strict_types=True) or not self.contains(
            elem2, strict_types=True
        ):
            raise TypeError(
                "algebra_class.killing_form_product only operates on algebra elements from the dispatching algebra"
            )
        if get_dgcv_category(elem1) == "subalgebra_element":
            elem1 = elem1.ambient_rep
        if get_dgcv_category(elem2) == "subalgebra_element":
            elem2 = elem2.ambient_rep
        if self._killing_form is None:
            self._killing_form = killingForm(
                self, assume_Lie_algebra=assume_Lie_algebra
            )
        kf = self._killing_form
        vec1 = matrix_dgcv(elem1.coeff_dict, shape=(self.dimension, 1))
        vec2 = matrix_dgcv(elem2.coeff_dict, shape=(1, self.dimension))
        return (vec2 * kf * vec1)[0]

    def subalgebra_from_grading_conditions(
        self,
        callable_bool_condition,
        sort_basis_by_grading_weights: bool = False,
        index_priority_for_lex_sort: list | tuple = [],
        reverse_sort_order: bool = False,
        label=None,
        basis_labels=None,
        register_in_vmf=False,
        initial_basis_index=None,
        simplify_products_by_default=None,
    ):

        weight_of = {}
        basis = []
        for elem in self.basis:
            weight = elem.check_element_weight()
            if callable_bool_condition(weight) is True:
                weight_of[id(elem)] = weight
                basis.append(elem)
        if sort_basis_by_grading_weights is True:
            grad_len = len(self.grading)
            idx_order = [j for j in index_priority_for_lex_sort if j < grad_len]
            if len(idx_order) < len(index_priority_for_lex_sort):
                dgcv_warning(
                    "Some indices provided in the `index_priority_for_lex_sort` parameter were out of range, and ignored."
                )
            if len(idx_order) == 0:
                idx_order = range(grad_len)
            basis = sorted(
                basis,
                key=lambda elem: [weight_of[id(elem)][idx] for idx in idx_order],
                reverse=reverse_sort_order,
            )
        grad = list(zip(*[weight_of[id(elem)] for elem in basis]))
        return self.new_alg_from_subalgebra(
            basis,
            grading=grad,
            span_warning=False,
            label=label,
            basis_labels=basis_labels,
            register_in_vmf=register_in_vmf,
            initial_basis_index=initial_basis_index,
            simplify_products_by_default=simplify_products_by_default,
        )

    def representation(
        self,
        rep_space=None,
        representation_basis=None,
        use_matrix_rep_instead_of_tensor=None,
    ):
        if rep_space is None:
            rep_space = self
        elif get_dgcv_category(rep_space) not in {
            "vector_space",
            "algebra",
            "subalgebra",
        }:
            raise TypeError(
                "`rep_space` must be a `dgcv` class type representing a vector space or algebra."
            ) from None
        if representation_basis is not None and any(
            isinstance(elem, matrix_dgcv) for elem in representation_basis
        ):
            use_matrix_rep_instead_of_tensor = True
        if use_matrix_rep_instead_of_tensor is None and representation_basis is None:
            representation_basis = self.preferred_representation
            use_matrix_rep_instead_of_tensor = (
                True if self._preferred_rep_type == "matrix" else False
            )
        if use_matrix_rep_instead_of_tensor is True:
            if representation_basis is None:
                if isinstance(self._mat_rep, (list, tuple)):
                    representation_basis = self.matrix_representation
                elif self._preferred_rep_type == "matrix":
                    representation_basis = self.preferred_representation
                else:
                    representation_basis = adjointRepresentation(self)
            elif isinstance(representation_basis, (list, tuple)):
                if len(representation_basis) != self.dimension:
                    raise TypeError(
                        "`representation_basis` should be a list of matrix/tensor elements matching the length of the represented algebra's basis."
                    )
                for elem in representation_basis:
                    if not isinstance(elem, matrix_dgcv):
                        raise TypeError(
                            f"If setting `use_matrix_rep_instead_of_tensor==True` and providing `representation_basis`, it should be a list of matrices. But an element in the given list was of type {type(elem)}"
                        )
                    if elem.shape[0] != elem.shape[1]:
                        raise TypeError(
                            f"If setting `use_matrix_rep_instead_of_tensor==True` and providing `representation_basis`, it should be a list of square matrices. Received a matrix of shape {elem.shape}"
                        )
                    if rep_space.dimension != elem.shape[0]:
                        raise TypeError(
                            f"If setting `use_matrix_rep_instead_of_tensor==True` and providing `representation_basis`, it should be a list of (d,d) matrices where d is the dimension of the reprentation space (defaults to `self`). Received a matrix of shape {elem.shape} and rep. space of dimension {rep_space.dimension}"
                        )
            t_rep = [
                _mat_to_tensor(j, rep_space.dual(), rep_space)
                for j in representation_basis
            ]
        else:
            if representation_basis is None:
                if isinstance(self._tensor_rep, (list, tuple)):
                    representation_basis = self.tensor_representation
                elif self._preferred_rep_type == "tensor":
                    representation_basis = self.preferred_representation
                else:
                    raise TypeError(
                        "`representation_basis` was not provided and no cached representation is currently stored in the algebra to fall back to."
                    )
            if len(representation_basis) != self.dimension:
                raise TypeError(
                    "`representation_basis` should be a list of matrix/tensor elements matching the length of the represented algebra's basis."
                )
            for elem in representation_basis:
                if not get_dgcv_category(elem) == "tensorProduct":
                    raise TypeError(
                        f"If not setting `representation_basis` to a list of matrices or setting `use_matrix_rep_instead_of_tensor==True` then `representation_basis` should be a list of tensor products. But an element in the given list was of type {type(elem)}"
                    )
            t_rep = representation_basis
        hom = homomorphism(self, [rep_space, rep_space.dual()], t_rep)
        return linear_representation(hom)

    def direct_sum(
        self,
        other,
        grading=None,
        label=None,
        basis_labels=None,
        register_in_vmf=False,
        initial_basis_index=None,
        simplify_products_by_default=None,
        build_all_gradings=False,
    ):
        if get_dgcv_category(other) in {
            "algebra",
            "vectorspace",
            "subalgebra",
            "algebra_subspace",
            "vector_subspace",
        }:
            _markers = {"sum": True, "lockKey": retrieve_passkey()}
            if build_all_gradings is not True:
                grad1 = self.grading[:1] or [[0] * self.dimension]
                grad2 = other.grading[:1] or [[0] * other.dimension]
            else:
                grad1 = self.grading or [[0] * self.dimension]
                grad2 = other.grading or [[0] * other.dimension]
            builtG = []
            for gl1 in grad1:
                for gl2 in grad2:
                    builtG.append(list(gl1) + list(gl2))
            if not isinstance(grading, (list, tuple)):
                grading = []
            if isinstance(grading, (list, tuple)):
                if all(isinstance(elem, (list, tuple)) for elem in grading):
                    grading = [list(elem) for elem in grading] + builtG
                elif all(isinstance(elem, expr_numeric_types()) for elem in grading):
                    grading = [list(grading)] + builtG
                elif grading is not None:
                    dgcv_warning(
                        "The supplied grading data format is incompatible, and was ignored."
                    )
                    grading = builtG
                else:
                    grading = builtG

            if label is None:
                label = f"{getattr(self, 'label', 'algebra_instance')}_plus_{getattr(other, 'label', 'algebra_instance')}"
                _markers["_tex_label"] = (
                    f"{self._repr_latex_(raw=True, abbrev=True)}\\oplus {other._repr_latex_(raw=True, abbrev=True)}"
                )
            if basis_labels is None:
                basis_labels = [elem.__repr__() for elem in self.basis] + [
                    elem.__repr__() for elem in other.basis
                ]
                _markers["_tex_basis_labels"] = [
                    elem._repr_latex_(raw=True) for elem in self.basis
                ] + [elem._repr_latex_(raw=True) for elem in other.basis]

            if register_in_vmf is not True:
                _markers["registered"] = False

            return linear_representation(
                homomorphism(self, other.endomorphism_algebra)
            ).semidirect_sum(
                grading=grading,
                label=label,
                basis_labels=basis_labels,
                register_in_vmf=register_in_vmf,
                initial_basis_index=initial_basis_index,
                simplify_products_by_default=simplify_products_by_default,
                _markers=_markers,
            )
        else:
            return NotImplemented

    def copy(
        self,
        label: str = None,
        basis_labels: str | list[str] = None,
        register_in_vmf: bool = False,
        initial_basis_index: int = None,
        simplify_products_by_default: bool = None,
        parameter_sub_rules: dict = None,
    ) -> algebra_class:
        if simplify_products_by_default is None:
            simplify_products_by_default = self.simplify_products_by_default
        if not isinstance(label, str) or label == "":
            label = "Alg_" + create_key()
        if isinstance(basis_labels, (tuple, list)):
            if (
                not all(isinstance(elem, str) for elem in basis_labels)
                or len(basis_labels) != self.dimension
            ):
                dgcv_warning(
                    "`basis_labels` is in an unsupported format and was ignored"
                )
                basis_labels = None
        if not isinstance(basis_labels, (tuple, list)):
            pref = (
                basis_labels
                if (isinstance(basis_labels, str) and basis_labels != "")
                else "_e"
            )
            IIdx = (
                initial_basis_index
                if isinstance(initial_basis_index, numbers.Integral)
                else 1
            )
            basis_labels = [f"{pref}{i + IIdx}" for i in range(self.dimension)]
        if not isinstance(self._grading, (list, tuple)) or len(self._grading) == 0:
            grad = None
        else:
            grad = self._grading
        substituted = isinstance(parameter_sub_rules, dict) and bool(
            parameter_sub_rules
        )
        sd = (
            subs(self.structureData, parameter_sub_rules)
            if substituted
            else self.structureData
        )
        if substituted:
            educed = dict()
            params = set(get_free_symbols(sd))
        else:
            educed = dict(self._educed_properties)
            params = set(self._parameters)
        _markers = {
            "parameters": params,
            "_educed_properties": educed,
            "semidirect_decomposition": self.semidirect_decomposition,
            "tensor_decomposition": self.tensor_decomposition,
        }
        if register_in_vmf is True:
            from .algebras_secondary import createAlgebra

            return createAlgebra(
                sd,
                label,
                basis_labels=basis_labels,
                grading=grad,
                return_created_object=True,
                simplify_products_by_default=simplify_products_by_default,
                _markers=_markers,
            )
        return algebra_class(
            sd,
            grading=grad,
            simplify_products_by_default=simplify_products_by_default,
            _label=label,
            _basis_labels=basis_labels,
            _calledFromCreator=retrieve_passkey(),
            _markers=_markers,
        )

    def __add__(self, other):
        if _scalar_is_zero(other):
            return self
        if get_dgcv_category(other) in {
            "algebra",
            "vectorspace",
            "subalgebra",
            "algebra_subspace",
            "vector_subspace",
        }:
            return self.direct_sum(other)
        return NotImplemented

    def __radd__(self, other):
        if _scalar_is_zero(other):
            return self
        return NotImplemented

    def tensor_product(
        self,
        other,
        grading=None,
        label=None,
        basis_labels=None,
        register_in_vmf=False,
        initial_basis_index=None,
        simplify_products_by_default=None,
        build_all_gradings=False,
    ):
        if get_dgcv_category(other) in {
            "algebra",
            "vectorspace",
            "subalgebra",
            "algebra_subspace",
            "vector_subspace",
        }:
            if simplify_products_by_default is None:
                simplify_products_by_default = getattr(
                    self, "simplify_products_by_default", False
                )
            if build_all_gradings is not True:
                grad1 = self.grading[:1] or [[0] * self.dimension]
                grad2 = other.grading[:1] or [[0] * other.dimension]
            else:
                grad1 = self.grading or [[0] * self.dimension]
                grad2 = other.grading or [[0] * other.dimension]
            builtG = []
            for gl1 in grad1:
                for gl2 in grad2:
                    builtG.append([w1 + w2 for w1 in gl1 for w2 in gl2])
            if not isinstance(grading, (list, tuple)):
                grading = []
            if isinstance(grading, (list, tuple)):
                if all(isinstance(elem, (list, tuple)) for elem in grading):
                    grading = [list(elem) for elem in grading] + builtG
                elif all(isinstance(elem, expr_numeric_types()) for elem in grading):
                    grading = [list(grading)] + builtG
                elif grading is not None:
                    dgcv_warning(
                        "The supplied grading data format is incompatible, and was ignored."
                    )
                    grading = builtG
                else:
                    grading = builtG

            if isinstance(basis_labels, (tuple, list)):
                if (
                    not all(isinstance(elem, str) for elem in basis_labels)
                    or len(basis_labels) != self.dimension * other.dimension
                ):
                    dgcv_warning(
                        f"`basis_labels` is in an unsupported format and was ignored. Received {basis_labels}, types: {[type(lab) for lab in basis_labels]}, target length {self.dimension}*{other.dimension}"
                    )
                    basis_labels = None
            _markers = {
                "prod": True,
                "lockKey": retrieve_passkey(),
                "tensor_decomposition": (self, other),
            }
            if label is None:
                label = f"{self.label}_tensor_{other.label}"
                _markers["_tex_label"] = (
                    f"{self._repr_latex_(raw=True, abbrev=True)}\\otimes {other._repr_latex_(raw=True, abbrev=True)}"
                )
            if basis_labels is None or not isinstance(basis_labels, str):
                basis_labels = [
                    f"{elem1.__repr__()}_tensor_{elem2.__repr__()}"
                    for elem1 in self.basis
                    for elem2 in other.basis
                ]
                _markers["_tex_basis_labels"] = [
                    f"{elem1._repr_latex_(raw=True)}\\otimes {elem2._repr_latex_(raw=True)}"
                    for elem1 in self.basis
                    for elem2 in other.basis
                ]
            if isinstance(basis_labels, str):
                pref = basis_labels
                IIdx = (
                    initial_basis_index
                    if isinstance(initial_basis_index, numbers.Integral)
                    else 1
                )
                basis_labels = [
                    f"{pref}{i + IIdx}" for i in range(self.dimension * other.dimension)
                ]
            if not isinstance(label, str) or label == "":
                label = "Alg_" + create_key()

            if register_in_vmf is True:
                from .algebras_secondary import createAlgebra

                return createAlgebra(
                    self.dimension * other.dimension,
                    label,
                    basis_labels=basis_labels,
                    grading=grading,
                    return_created_object=True,
                    simplify_products_by_default=simplify_products_by_default,
                    _markers=_markers,
                )
            else:
                _markers["registered"] = False
                return algebra_class(
                    self.dimension * other.dimension,
                    grading=grading,
                    simplify_products_by_default=simplify_products_by_default,
                    _label=label,
                    _basis_labels=basis_labels,
                    _calledFromCreator=retrieve_passkey(),
                    _markers=_markers,
                )
        elif isinstance(other, expr_numeric_types()):
            return self._convert_to_tp().__matmul__(other)
        else:
            return NotImplemented

    def __matmul__(self, other):
        return self.tensor_product(other)

    def __rmatmul__(self, other):
        if _scalar_is_zero(other):
            return algebra_class({})
        if isinstance(other, expr_numeric_types()):
            return self._convert_to_tp().__rmatmul__(other)
        return NotImplemented

    def dual(self, invert_grad_weights=True):
        return algebra_dual(self, invert_grad_weights=invert_grad_weights)


class algebra_dual(dgcv_class):
    def __init__(self, alg, invert_grad_weights=True):
        object.__setattr__(self, "dual_algebra", alg)
        object.__setattr__(self, "basis", tuple([elem.dual() for elem in alg.basis]))
        object.__setattr__(self, "label", alg.label + "_dual")
        if invert_grad_weights is not False:
            object.__setattr__(
                self, "grading", [tuple(-j for j in elem) for elem in alg.grading]
            )
        object.__setattr__(self, "_dgcv_categories", {"algebra_dual"})

    def __getattr__(self, name):
        return getattr(self.dual_algebra, name)

    def __setattr__(self, name, value):
        if name == "dual_algebra":
            object.__setattr__(self, name, value)
        else:
            setattr(self.dual_algebra, name, value)

    def __delattr__(self, name):
        if name == "dual_algebra":
            raise AttributeError("Cannot delete 'dual_algebra'")
        delattr(self.dual_algebra, name)

    def __dir__(self):
        # Merge proxy attributes with algebra_class attributes
        return sorted(
            set(dir(type(self)))
            | set(self.__dict__.keys())
            | set(dir(self.dual_algebra))
        )

    def dual(self):
        return self.dual_algebra

    def __str__(self):
        reg = get_dgcv_settings_registry()
        vlp = bool(reg.get("verbose_label_printing", False))

        alg = getattr(self, "dual_algebra", None)
        if alg is None:
            return "<>"

        if vlp is False:
            lab = getattr(alg, "label", None)
            return f"{lab}^*" if lab else "Unnamed^*"

        nm = str(getattr(alg, "label", None) or "Unnamed")
        b = getattr(self, "basis", None) or []
        core = "<" + ", ".join(str(e) for e in b) + ">"
        return f"{nm}^*={core}"

    def _repr_latex_(self, raw: bool = False, abbrev: bool = False, **kwargs):
        reg = get_dgcv_settings_registry()
        vlp = bool(reg.get("verbose_label_printing", False))

        alg = getattr(self, "dual_algebra", None)
        texS = (
            alg._repr_latex_(raw=True, abbrev=True) if alg is not None else r"\text{?}"
        )
        texS = str(texS).replace("$", "").replace(r"\displaystyle", "")
        if "^" in texS:
            texS = f"\\left({texS}\\right)"
        texS = f"{texS}^{{*}}"

        if abbrev or (vlp is False):
            out = texS
            return out if raw else f"$\\displaystyle {out}$"

        b = getattr(self, "basis", None) or []
        inner = ", ".join(e._repr_latex_(raw=True) for e in b)
        inner = str(inner).replace("$", "").replace(r"\displaystyle", "")

        out = texS if not inner else texS + rf"=\langle {inner}\rangle"
        return out if raw else f"$\\displaystyle {out}$"

    def _latex(self, printer=None, raw: bool = True, **kwargs):
        return self._repr_latex_(raw=raw)

    def direct_sum(
        self,
        other,
        grading=None,
        label=None,
        basis_labels=None,
        register_in_vmf=False,
        initial_basis_index=None,
        simplify_products_by_default=None,
        build_all_gradings=False,
    ):
        if get_dgcv_category(other) in {
            "algebra",
            "vectorspace",
            "subalgebra",
            "algebra_subspace",
            "vector_subspace",
        }:
            _markers = {"sum": True, "lockKey": retrieve_passkey()}
            if build_all_gradings is not True:
                grad1 = self.grading[:1] or [[0] * self.dimension]
                grad2 = other.grading[:1] or [[0] * other.dimension]
            else:
                grad1 = self.grading or [[0] * self.dimension]
                grad2 = other.grading or [[0] * other.dimension]
            builtG = []
            for gl1 in grad1:
                for gl2 in grad2:
                    builtG.append(list(gl1) + list(gl2))
            if not isinstance(grading, (list, tuple)):
                grading = []
            if isinstance(grading, (list, tuple)):
                if all(isinstance(elem, (list, tuple)) for elem in grading):
                    grading = [list(elem) for elem in grading] + builtG
                elif all(isinstance(elem, expr_numeric_types()) for elem in grading):
                    grading = [list(grading)] + builtG
                elif grading is not None:
                    dgcv_warning(
                        "The supplied grading data format is incompatible, and was ignored."
                    )
                    grading = builtG
                else:
                    grading = builtG

            if label is None:
                label = f"{self.label}_plus_{other.label}"
                _markers["_tex_label"] = (
                    f"{self._repr_latex_(raw=True, abbrev=True)}\\oplus {other._repr_latex_(raw=True, abbrev=True)}"
                )
            if basis_labels is None:
                basis_labels = [elem.__repr__() for elem in self.basis] + [
                    elem.__repr__() for elem in other.basis
                ]
                _markers["_tex_basis_labels"] = [
                    elem._repr_latex_(raw=True) for elem in self.basis
                ] + [elem._repr_latex_(raw=True) for elem in other.basis]

            return linear_representation(
                homomorphism(self, other.endomorphism_algebra)
            ).semidirect_sum(
                grading=grading,
                label=label,
                basis_labels=basis_labels,
                register_in_vmf=register_in_vmf,
                initial_basis_index=initial_basis_index,
                simplify_products_by_default=simplify_products_by_default,
                _markers=_markers,
            )
        else:
            return NotImplemented

    def __add__(self, other):
        return self.direct_sum(other)

    def tensor_product(
        self,
        other,
        grading=None,
        label=None,
        basis_labels=None,
        register_in_vmf=False,
        initial_basis_index=None,
        simplify_products_by_default=None,
        build_all_gradings=False,
    ):
        if get_dgcv_category(other) in {
            "algebra",
            "vectorspace",
            "subalgebra",
            "algebra_subspace",
            "vector_subspace",
        }:
            if simplify_products_by_default is None:
                simplify_products_by_default = getattr(
                    self, "simplify_products_by_default", False
                )
            if build_all_gradings is not True:
                grad1 = self.grading[:1] or [[0] * self.dimension]
                grad2 = other.grading[:1] or [[0] * other.dimension]
            else:
                grad1 = self.grading or [[0] * self.dimension]
                grad2 = other.grading or [[0] * other.dimension]
            builtG = []
            for gl1 in grad1:
                for gl2 in grad2:
                    builtG.append([w1 + w2 for w1 in gl1 for w2 in gl2])
            if not isinstance(grading, (list, tuple)):
                grading = []
            if isinstance(grading, (list, tuple)):
                if all(isinstance(elem, (list, tuple)) for elem in grading):
                    grading = [list(elem) for elem in grading] + builtG
                elif all(isinstance(elem, expr_numeric_types()) for elem in grading):
                    grading = [list(grading)] + builtG
                elif grading is not None:
                    dgcv_warning(
                        "The supplied grading data format is incompatible, and was ignored."
                    )
                    grading = builtG
                else:
                    grading = builtG

            if isinstance(basis_labels, (tuple, list)):
                if (
                    not all(isinstance(elem, str) for elem in basis_labels)
                    or len(basis_labels) != self.dimension * other.dimension
                ):
                    dgcv_warning(
                        f"`basis_labels` is in an unsupported format and was ignored. Recieved {basis_labels}, types: {[type(lab) for lab in basis_labels]}, target length {self.dimension}*{other.dimension}"
                    )
                    basis_labels = None
            _markers = {
                "prod": True,
                "lockKey": retrieve_passkey(),
                "tensor_decomposition": (self, other),
            }
            if label is None:
                label = f"{self.label}_tensor_{other.label}"
                _markers["_tex_label"] = (
                    f"{self._repr_latex_(raw=True, abbrev=True)}\\otimes {other._repr_latex_(raw=True, abbrev=True)}"
                )
            if basis_labels is None or not isinstance(basis_labels, str):
                basis_labels = [
                    f"{elem1.__repr__()}_tensor_{elem2.__repr__()}"
                    for elem1 in self.basis
                    for elem2 in other.basis
                ]
                _markers["_tex_basis_labels"] = [
                    f"{elem1._repr_latex_(raw=True)}\\otimes {elem2._repr_latex_(raw=True)}"
                    for elem1 in self.basis
                    for elem2 in other.basis
                ]
            if isinstance(basis_labels, str):
                pref = basis_labels
                IIdx = (
                    initial_basis_index
                    if isinstance(initial_basis_index, numbers.Integral)
                    else 1
                )
                basis_labels = [
                    f"{pref}{i + IIdx}" for i in range(self.dimension * other.dimension)
                ]
            if not isinstance(label, str) or label == "":
                label = "Alg_" + create_key()

            if register_in_vmf is True:
                from .algebras_secondary import createAlgebra

                return createAlgebra(
                    self.dimension * other.dimension,
                    label,
                    basis_labels=basis_labels,
                    grading=grading,
                    return_created_object=True,
                    simplify_products_by_default=simplify_products_by_default,
                    _markers=_markers,
                )
            else:
                _markers["registered"] = False
                return algebra_class(
                    self.dimension * other.dimension,
                    grading=grading,
                    simplify_products_by_default=simplify_products_by_default,
                    _label=label,
                    _basis_labels=basis_labels,
                    _calledFromCreator=retrieve_passkey(),
                    _markers=_markers,
                )

        else:
            return NotImplemented

    def __matmul__(self, other):
        return self.tensor_product(other)


class algebra_element_class(dgcv_class):
    def __init__(self, alg, coeff_dict, valence, format_sparse=False):
        if not isinstance(alg, algebra_class):
            raise TypeError(
                "`algebra_element_class` expects the first argument to be an instance of the `algebra` class."
            ) from None
        if valence not in {0, 1}:
            raise TypeError(
                "vector_space_element expects third argument to be 0 or 1."
            ) from None
        if isinstance(coeff_dict, dict):
            coeff_dict = {k: v for k, v in coeff_dict.items() if not _scalar_is_zero(v)}
        elif isinstance(coeff_dict, (list, tuple)):
            coeff_dict = {
                k: v for k, v in enumerate(coeff_dict) if not _scalar_is_zero(v)
            }
        elif get_dgcv_category(coeff_dict) == "array":
            coeff_dict = coeff_dict._data
        else:
            raise TypeError(
                "algebra_element_class recieved unsupports coeffs parameter format."
            ) from None
        self.algebra = alg
        self.vectorSpace = alg
        self.valence = valence
        self.is_sparse = format_sparse
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "algebra_element"
        self.dgcv_vs_id = self.vectorSpace.dgcv_vs_id
        self.coeff_dict = coeff_dict
        self._coeffs = None  # deprecated
        self._coeffs_hash_cache = None
        self._tensor_rep = None
        self._known_weight = None

    @property
    def coeffs(self):
        if self._coeffs is None:
            self._coeffs = tuple(
                self.coeff_dict.get(x, 0) for x in range(self.algebra.dimension)
            )
        return self._coeffs

    @property
    def _coeffs_hash(self):
        if self._coeffs_hash_cache is None:
            self._coeffs_hash_cache = frozenset(self.coeff_dict.items())
        return self._coeffs_hash_cache

    @property
    def tensor_representation(self):
        if self._tensor_rep is None and self.algebra.tensor_representation is not None:
            trep = self.algebra.tensor_representation
            self._tensor_rep = sum(c * trep[idx] for idx, c in self.coeff_dict.items())
        return self._tensor_rep

    def __eq__(self, other):
        if not isinstance(other, algebra_element_class):
            return NotImplemented
        return (
            self.algebra == other.algebra
            and self._coeffs_hash == other._coeffs_hash
            and self.valence == other.valence
            and self.is_sparse == other.is_sparse
        )

    def __hash__(self):
        return hash((self.algebra, self._coeffs_hash, self.valence, self.is_sparse))

    def _class_builder(self, coeff_dict, valence, format_sparse=False):
        return algebra_element_class(
            self.algebra, coeff_dict, valence, format_sparse=format_sparse
        )

    def __str__(self):
        if self.algebra.basis_labels is None:
            return "elem"

        if not self.algebra._registered:
            if (
                self.algebra._exclude_from_VMF == retrieve_passkey()
                or get_dgcv_settings_registry()["forgo_warnings"] is True
            ):
                pass
            elif self.algebra._callLock == retrieve_passkey() and isinstance(
                self.algebra._child_print_warning, str
            ):
                dgcv_warning(self.algebra._child_print_warning)
            else:
                dgcv_warning(
                    "This algebra_element_class's parent vector space (algebra_class) was initialized without an assigned label. "
                    "It is recommended to initialize `algebra_class` objects with dgcv creator functions like `createAlgebra` instead."
                )

        return lincomb_plain(
            self.coeff_dict,
            self.algebra.basis_labels,
            valence=self.valence,
            label_transform=None,
            fallback_label=self.algebra.basis_labels[0]
            if self.algebra.basis_labels
            else "e_1",
            include_zero_term=False,
        )

    def _repr_latex_(self, verbose=False, raw=False, **kwargs):
        if not self.vectorSpace._registered:
            if (
                self.vectorSpace._exclude_from_VMF == retrieve_passkey()
                or get_dgcv_settings_registry()["forgo_warnings"] is True
            ):
                pass
            elif self.vectorSpace._callLock == retrieve_passkey() and isinstance(
                self.vectorSpace._child_print_warning, str
            ):
                dgcv_warning(self.vectorSpace._child_print_warning)
            else:
                dgcv_warning(
                    "This algebra_element_class's parent vector space (algebra_class) was initialized without an assigned label. "
                    "It is recommended to initialize `algebra_class` objects with dgcv creator functions like `createAlgebra` instead."
                )

        return lincomb_latex(
            self.coeff_dict,
            vectorSpace=self.vectorSpace,
            valence=self.valence,
            verbose=verbose,
            raw=raw,
            apply_vlp_trim=True,
        )

    def _latex(self, printer=None, raw=True, **kwargs):
        return self._repr_latex_(raw=raw)

    def _latex_verbose(self, printer=None):
        """deprecated"""
        if not self.algebra._registered:
            if (
                self.algebra._exclude_from_VMF == retrieve_passkey()
                or get_dgcv_settings_registry()["forgo_warnings"] is True
            ):
                pass
            elif self.algebra._callLock == retrieve_passkey() and isinstance(
                self.algebra._child_print_warning, str
            ):
                dgcv_warning(self.algebra._child_print_warning)
            else:
                dgcv_warning(
                    "This algebra_element_class's parent vector space (an `algebra` class instance) was initialized without an assigned label. "
                    "It is recommended to initialize `algebra` class objects with dgcv creator functions like `createFiniteAlg` instead."
                )

        terms = []
        labels = self.algebra.basis_labels or [
            f"e_{i + 1}" for i in range(self.algebra.dimension)
        ]
        for idx, coeff in self.coeff_dict.items():
            basis_label = labels[idx]
            if _scalar_is_zero(coeff - 1):
                if self.valence == 1:
                    terms.append(rf"{basis_label}")
                else:
                    terms.append(rf"{basis_label}^*")
            elif _scalar_is_zero(coeff + 1):
                if self.valence == 1:
                    terms.append(rf"-{basis_label}")
                else:
                    terms.append(rf"-{basis_label}^*")
            else:
                if isinstance(coeff, expr_numeric_types()) and len(coeff.args) > 1:
                    if self.valence == 1:
                        terms.append(rf"({latex(coeff)}) {basis_label}")
                    else:
                        terms.append(rf"({latex(coeff)}) {basis_label}^*")
                else:
                    if self.valence == 1:
                        terms.append(rf"{latex(coeff)} {basis_label}")
                    else:
                        terms.append(rf"{latex(coeff)} {basis_label}^*")

        if not terms:
            return rf"0 {self.algebra.basis_labels[0] if self.algebra.basis_labels else 'e_1'}"

        result = " + ".join(terms).replace("+ -", "- ")

        def format_algebra_label(label):
            r"""
            Wrap the vector space label in \mathfrak{} if lowercase, and add subscripts for numeric suffixes or parts.
            """
            if "_" in label:
                main_part, subscript_part = label.split("_", 1)
                if main_part.islower():
                    return rf"\mathfrak{{{main_part}}}_{{{subscript_part}}}"
                return rf"{main_part}_{{{subscript_part}}}"
            elif label[-1].isdigit():
                label_text = "".join(filter(str.isalpha, label))
                label_number = "".join(filter(str.isdigit, label))
                if label_text.islower():
                    return rf"\mathfrak{{{label_text}}}_{{{label_number}}}"
                return rf"{label_text}_{{{label_number}}}"
            elif label.islower():
                return rf"\mathfrak{{{label}}}"
            return label

        return rf"\text{{Element of }} {format_algebra_label(self.algebra.label)}: {result}"

    @property
    def label(self):
        return self.__repr__()

    @property
    def is_zero(self):
        for j in self.coeff_dict.values():
            if not _scalar_is_zero(simplify(j)):
                return False
        return True

    @property
    def is_literal_zero(self):
        for j in self.coeff_dict.values():
            if not _scalar_is_zero(j):
                return False
        return True

    @property
    def __dgcv_zero_obstr__(self):
        cfs = []
        cfvars = set()
        for cf in self.coeff_dict.values():
            cfs.append(cf)
            cfvars |= get_free_symbols(cf)
        return cfs, cfvars

    def subs(self, subsData):
        newCoeffs = {idx: subs(j, subsData) for idx, j in self.coeff_dict.items()}
        return algebra_element_class(self.algebra, newCoeffs, self.valence)

    @property
    def ambient_rep(self):
        return self

    def __dgcv_simplify__(self, *args, **kwargs):
        return algebra_element_class(
            self.algebra,
            {idx: simplify(j) for idx, j in self.coeff_dict.items()},
            self.valence,
        )

    def _eval_simplify(self, *args, **kwargs):
        return algebra_element_class(
            self.algebra,
            {idx: simplify(j) for idx, j in self.coeff_dict.items()},
            self.valence,
        )

    def dual(self):
        return algebra_element_class(
            self.algebra,
            self.coeff_dict,
            (self.valence + 1) % 2,
        )

    def _convert_to_tp(self):
        return tensorProduct(
            (self.dgcv_vs_id,),
            {
                (idx, self.valence, self.dgcv_vs_id): j
                for idx, j in self.coeff_dict.items()
            },
            shape="all",
        )

    def _recursion_contract_hom(self, other):
        return self._convert_to_tp()._recursion_contract_hom(other)

    def _si_wrap(self, obj):
        if self.algebra.simplify_products_by_default is True:
            return simplify(obj)
        else:
            return obj

    def _fast_add(self, other):
        """
        Internal-only: assumes `other` is an algebra_element_class
        with the same algebra and valence. No type or safety checks etc.
        """
        new_dict = dict(self.coeff_dict)
        for k, v in other.coeff_dict.items():
            new_dict[k] = new_dict.get(k, 0) + v
        return algebra_element_class(
            self.algebra,
            new_dict,
            self.valence,
        )

    @classmethod
    def _dgcv_multiadd(cls, terms, start=0):
        if not isinstance(terms, (list, tuple)):
            terms = list(terms)
        if not terms:
            return start
        acc = {}
        alg = None
        valence = None
        residual = []
        if isinstance(start, cls):
            acc.update(start.coeff_dict)
            alg = start.algebra
            valence = start.valence
        elif not _scalar_is_zero(start):
            residual.append(start)
        for t in terms:
            if isinstance(t, cls):
                if alg is None:
                    alg = t.algebra
                    valence = t.valence
                if t.algebra == alg and t.valence == valence:
                    for k, v in t.coeff_dict.items():
                        acc[k] = acc.get(k, 0) + v
                    continue
            residual.append(t)
        if alg is None:
            return sum(terms, start)
        out = cls(alg, acc, valence)
        if residual:
            return sum(residual, out)
        return out

    @classmethod
    def _dgcv_multiadd_scaled(cls, pairs, start=0):
        if not isinstance(pairs, (list, tuple)):
            pairs = list(pairs)
        if not pairs:
            return start
        acc = {}
        alg = None
        valence = None
        spbd = False
        residual = []
        if isinstance(start, cls):
            acc.update(start.coeff_dict)
            alg = start.algebra
            valence = start.valence
            spbd = alg.simplify_products_by_default is True
        elif not _scalar_is_zero(start):
            residual.append(start)
        for c, t in pairs:
            if isinstance(t, cls):
                if alg is None:
                    alg = t.algebra
                    valence = t.valence
                    spbd = alg.simplify_products_by_default is True
                if t.algebra == alg and t.valence == valence:
                    if not _scalar_is_zero(c):
                        for k, v in t.coeff_dict.items():
                            acc[k] = acc.get(k, 0) + (
                                simplify(c * v) if spbd else c * v
                            )
                    continue
            residual.append(c * t)
        if alg is None:
            return sum([c * t for c, t in pairs], start)
        out = cls(alg, acc, valence)
        if residual:
            return sum(residual, out)
        return out

    def __add__(self, other):
        if _scalar_is_zero(other):
            return self
        if get_dgcv_category(other) == "subalgebra_element":
            if (
                other.algebra.ambient.dgcv_vs_id == self.dgcv_vs_id
                and self.valence == other.valence
            ):
                other = other.ambient_rep
            else:
                other = other._convert_to_tp()
        if get_dgcv_category(other) == "algebra_element":
            if self.algebra == other.algebra and self.valence == other.valence:
                new_dict = dict(self.coeff_dict)
                for k, v in other.coeff_dict.items():
                    new_dict[k] = new_dict.get(k, 0) + v
                return algebra_element_class(
                    self.algebra,
                    new_dict,
                    self.valence,
                )
            else:
                other = other._convert_to_tp()
        if get_dgcv_category(other) == "vector_space_element":
            other = other._convert_to_tp()
        if isinstance(other, expr_numeric_types()):
            other = tensorProduct("_", {tuple(): other})
        if isinstance(other, tensorProduct):
            return self._convert_to_tp() + other
        return NotImplemented

    def __radd__(self, other):
        if _scalar_is_zero(other):
            return self
        if isinstance(other, expr_numeric_types()):
            return tensorProduct("_", {tuple(): other}) + self
        return NotImplemented

    def __sub__(self, other):
        if _scalar_is_zero(other):
            return self
        if get_dgcv_category(other) == "subalgebra_element":
            if (
                other.algebra.ambient.dgcv_vs_id == self.dgcv_vs_id
                and self.valence == other.valence
            ):
                other = other.ambient_rep
            else:
                other = other._convert_to_tp()
        if get_dgcv_category(other) == "algebra_element":
            if self.algebra == other.algebra and self.valence == other.valence:
                new_dict = dict(self.coeff_dict)
                for k, v in other.coeff_dict.items():
                    new_dict[k] = new_dict.get(k, 0) - v
                return algebra_element_class(
                    self.algebra,
                    new_dict,
                    self.valence,
                )
            else:
                other = other._convert_to_tp()
        if get_dgcv_category(other) == "vector_space_element":
            other = other._convert_to_tp()
        if isinstance(other, expr_numeric_types()):
            other = tensorProduct("_", {tuple(): other})
        if isinstance(other, tensorProduct):
            return self._convert_to_tp() - other
        return NotImplemented

    def __rsub__(self, other):
        if _scalar_is_zero(other):
            return -self
        if isinstance(other, expr_numeric_types()):
            return tensorProduct("_", {tuple(): other}) - self
        return NotImplemented

    def __mul__(self, other):
        if get_dgcv_category(other) == "subalgebra_element":
            if (
                other.algebra.ambient.dgcv_vs_id == self.dgcv_vs_id
                and self.valence == other.valence
            ):
                other = other.ambient_rep
            else:
                other = other._convert_to_tp()
        if isinstance(other, algebra_element_class):
            if self.algebra == other.algebra and self.valence == other.valence:
                sign = 1 if self.valence == 1 else -1
                alg = self.algebra
                struct = alg.structureData
                spbd = self.algebra.simplify_products_by_default
                new_coeffs = dict()
                for idx1, c1 in self.coeff_dict.items():
                    for idx2, c2 in other.coeff_dict.items():
                        scalar = sign * c1 * c2
                        row = struct[idx1, idx2]
                        for idx3, c3 in row._data.items():
                            new_coeffs[idx3] = new_coeffs.get(idx3, 0) + (
                                self._si_wrap(scalar * c3) if spbd else scalar * c3
                            )

                return algebra_element_class(
                    self.algebra,
                    new_coeffs,
                    self.valence,
                )
            else:
                other = other._convert_to_tp()
        elif isinstance(other, tensorProduct):
            return self._si_wrap((self._convert_to_tp()) * other)
        elif isinstance(other, expr_numeric_types()):
            new_coeffs = {
                idx: self._si_wrap(j * other) for idx, j in self.coeff_dict.items()
            }
            return algebra_element_class(self.algebra, new_coeffs, self.valence)
        return NotImplemented

    def __rmul__(self, other):
        if get_dgcv_category(other) == "subalgebra_element":
            if (
                other.algebra.ambient.dgcv_vs_id == self.dgcv_vs_id
                and self.valence == other.valence
            ):
                return other.ambient_rep * self
        if isinstance(other, expr_numeric_types()) or get_dgcv_category(other) in {
            "subalgebra_element",
            "algebra_element",
            "tensorProduct",
        }:
            return self._si_wrap(self * other)
        return NotImplemented

    def __matmul__(self, other):
        """Overload @ operator for tensor product."""
        if get_dgcv_category(other) == "tensorProduct":
            return self._convert_to_tp() @ other
        if isinstance(other, expr_numeric_types()):
            return other * self
        if get_dgcv_category(other) not in {
            "algebra_element",
            "subalgebra_element",
            "vector_space_element",
        }:
            raise TypeError(
                f"unsuported operand types for `@`. Types {type(self)} and {type(other)}"
            ) from None
        new_dict = {
            (
                idx1,
                idx2,
                self.valence,
                other.valence,
                self.dgcv_vs_id,
                other.dgcv_vs_id,
            ): self._si_wrap(c1 * c2)
            for idx1, c1 in self.coeff_dict.items()
            for idx2, c2 in other.coeff_dict.items()
        }
        return tensorProduct(
            [], new_dict
        )  ###!!! first keyword is deprecation placeholder

    def __rmatmul__(self, other):
        if isinstance(other, expr_numeric_types()):
            return other * self
        return self._convert_to_tp().__rmatmul__(other)

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return self._si_wrap(ratio(1, other) * self)
        elif isinstance(other, expr_numeric_types()):
            return self._si_wrap((1 / other) * self)
        else:
            raise TypeError(
                f"True division `/` of algebra elements by another object is only supported for scalars, not {type(other)}"
            ) from None

    def __neg__(self):
        return -1 * self

    def __xor__(self, other):
        if other == "":
            return self.dual()
        raise ValueError("Invalid operation. Use `^''` to denote the dual.") from None

    def __call__(self, other, **kwds):
        if get_dgcv_category(other) == "subalgebra_element":
            other = other.ambient_rep
        if get_dgcv_category(other) == "algebra_element":
            if other.algebra == self.algebra and other.valence != self.valence:
                cd = other.coeff_dict
                return sum(c * cd.get(idx, 0) for idx, c in self.coeff_dict.items())
            elif self.tensor_representation is not None:
                return self.tensor_representation(other)
            else:
                raise TypeError(
                    f"`algebra_element_class` call can only be applied to elements from the same algebra pairing one element with another of complementary valence, or applying elements from an endomorphism_space subclass. Recieved self: {self} and other: {other} belonging to {self.algebra} and {other.algebra} with valences {self.valence} and {other.valence}"
                )
        else:
            raise TypeError(
                f"`algebra_element_class` call cannot be applies objects of type {type(other)}"
            )

    def compute_weight(self, test_weights=None, flatten_weights=False):
        return self.check_element_weight(
            test_weights=test_weights, flatten_weights=flatten_weights
        )

    def check_element_weight(self, test_weights=None, flatten_weights=False):
        """
        Determines the weight vector of this algebra_element_class with respect to its algebra' grading vectors.

        Returns
        -------
        list
            A list of weights corresponding to the grading vectors of the parent algebra.
            Each entry is either an integer, variable representing a weight, the string 'AllW' if the element is the zero element,
            or 'NoW' if the element is not homogeneous.

        Notes
        -----
        - This method calls the parent algebra' check_element_weight method.
        - 'AllW' is returned for zero elements, which are compaible with all weights.
        - 'NoW' is returned for non-homogeneous elements that do not satisfy the grading constraints.
        """

        return self.algebra.check_element_weight(
            self, test_weights=test_weights, flatten_weights=flatten_weights
        )

    def weighted_decomposition(self, test_weights=None, flatten_weights=False):
        weighted_components = {}
        for idx, coeff in self.coeff_dict.items():
            elem = self.algebra.basis[idx]
            w = elem.check_element_weight(
                test_weights=test_weights, flatten_weights=flatten_weights
            )
            if isinstance(w, list):
                w = tuple(w)
            weighted_components[w] = weighted_components.get(w, 0) + coeff * elem
        return weighted_components

    def coproduct(self):
        if self.valence != 0:
            return print(
                "The algebra co-product is only defined on dual Lie algebra elements as it is dual to the algebra product map."
            )
        terms = []
        for idx, c in self.coeff_dict.items():
            elem = self.algebra.basis[idx]
            if self.algebra._coproduct.get(elem, None) is None:
                tensor_terms = []
                for idx, e1 in enumerate(self.algebra.basis):
                    if self.algebra.is_skew_symmetric():
                        skew = True
                        start = idx + 1
                    else:
                        skew = False
                        start = 0
                    for e2 in self.algebra.basis[start:]:
                        if skew:
                            tensor_terms.append(
                                self(e1 * e2)
                                * (e1.dual() @ e2.dual() - e2.dual() @ e1.dual())
                            )
                        else:
                            tensor_terms.append(self(e1 * e2) * (e1.dual() @ e2.dual()))
                self.algebra._coproduct[elem] = sum(tensor_terms)
            terms.append(c * self.algebra._coproduct[elem])
        return sum(terms)

    @property
    def free_symbols(self):
        fs = set()
        for c in self.coeff_dict.values():
            fs |= get_free_symbols(c)
        return fs

    def dual_pairing(self, other):
        return self._convert_to_tp().dual_pairing(other)

    def decompose(self, format_as_list=True, return_basis=True):
        if self.valence == 1:
            out = self.coeffs if format_as_list else self.coeff_dict
            return (out, self.algebra.basis) if return_basis else out
        out = self.coeffs if format_as_list else self.coeff_dict
        return (out, [j.dual() for j in self.algebra.basis]) if return_basis else out

    def terms(self):
        return [c * self.algebra.basis[idx] for idx, c in self.coeff_dict.items()]


class algebra_subspace_class(_vector_space_methods, dgcv_class):
    def __init__(
        self,
        basis,
        parent_algebra=None,
        test_weights=None,
        _grading=None,
        _internal_lock=None,
        span_warning=True,
        simplify_basis=False,
        **kwargs,
    ):
        # From former __new__: validate inputs and compute subspace attributes
        if not isinstance(basis, (list, tuple)):
            raise TypeError(
                "algebra_subspace_class expects first argument to a be a list or tuple of algebra_element_class instances"
            ) from None
        typeCheck = {"subalgebra_element", "algebra_element"}
        if not all(get_dgcv_category(j) in typeCheck or j == 0 for j in basis):
            raise TypeError(
                "algebra_subspace_class expects first argument to a be a list or tuple of algebra_element_class instances"
            ) from None
        if parent_algebra is None:
            if len(basis) > 0:
                if get_dgcv_category(basis[0].algebra) != "algebra":
                    if all(j.algebra == basis[0].algebra for j in basis[1:]):
                        parent_alg = basis[0].algebra.ambient
                    else:
                        parent_alg = basis[0].algebra.ambient
                else:
                    parent_alg = basis[0].algebra
            else:
                parent_alg = None
        elif get_dgcv_category(parent_algebra) in {"subalgebra", "algebra_subspace"}:
            parent_alg = parent_algebra.ambient
        elif get_dgcv_category(parent_algebra) == "algebra":
            parent_alg = parent_algebra
        else:
            raise TypeError(
                "algebra_subspace_class expects second argument to be an algebra instance or algebra subspace or subalgebra."
            ) from None

        filtered_basis = parent_alg.filter_independent_elements(
            basis,
            apply_light_basis_simplification=simplify_basis,
            surface_singularities=False,
        )
        if len(filtered_basis) < len(basis):
            basis = filtered_basis
            if span_warning:
                wmessage = (
                    " This can result in incorrect weighting assignements from the manual assignment provided to the `test_weights` parameter. To avoid this issue, provided a linearly independent spanning set instead."
                    if test_weights is None
                    else ""
                )
                dgcv_warning(
                    "The given list for `basis` was not linearly independent, so the algebra_subspace_class initializer computed a basis for its span to use instead."
                    + wmessage
                )

        self.filtered_basis = tuple(filtered_basis)
        self.basis = tuple(filtered_basis)
        self.dimension = len(filtered_basis)
        self.ambient: algebra_class = parent_alg
        grading_per_elem = []
        if (
            _internal_lock == retrieve_passkey()
            and test_weights is None
            and _grading is not None
        ):
            self.grading = _grading
        else:
            for elem in filtered_basis:
                weight = parent_alg.check_element_weight(
                    elem, test_weights=test_weights
                )
                grading_per_elem.append(weight)
            self.grading = [
                elem for elem in zip(*grading_per_elem) if "NoW" not in elem
            ]
        self.original_basis = basis
        self._parameters = self.ambient._parameters
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "algebra_subspace"

        # immutables
        self._grading = tuple(self.grading)
        self._gradingNumber = len(self._grading)
        self._singularities = {}

        vsr = get_vs_registry()
        self.dgcv_vs_id = len(vsr)
        vsr.append(self)

        # attribute caches
        self._endomorphisms = None
        self._is_subalgebra = None

    @property
    def zero_element(self):
        return algebra_element_class(self, {}, 1)

    def __eq__(self, other):
        if not isinstance(other, algebra_subspace_class):
            return NotImplemented
        return self.dgcv_vs_id == other.dgcv_vs_id

    def __hash__(self):
        return hash(self.dgcv_vs_id)

    def contains(self, items, return_basis_coeffs=False, strict_types=False):
        if isinstance(items, (list, tuple)):
            return [
                self.contains(
                    item,
                    return_basis_coeffs=return_basis_coeffs,
                    strict_types=strict_types,
                )
                for item in items
            ]
        if strict_types is False and items == 0:
            if return_basis_coeffs is True:
                return {}
            return True
        item = items
        if get_dgcv_category(item) == "subalgebra_element":
            if item.dgcv_vs_id == self.dgcv_vs_id:
                if return_basis_coeffs is True:
                    return dict(item.coeff_dict)
                else:
                    return True
            item = item.ambient_rep
        if not isinstance(item, algebra_element_class) or item.algebra != self.ambient:
            return False

        pos = self._basis_index
        try:
            found = pos.get(item)
        except TypeError:
            found = None
        if found is None:
            if self.dimension == 0:
                return False
            genElement, variables = linear_combination(self.basis)
            sol = solve_dgcv(
                item - genElement, variables, method="linsolve", simplify_result=False
            )
            if len(sol) == 0:
                return False
        else:
            if return_basis_coeffs is True:
                return {found: 1}
        if return_basis_coeffs is True:
            s = sol[0]
            out = dict()
            for c, var in enumerate(variables):
                coeff = s.get(var, 0)
                if not _scalar_is_zero(coeff):
                    out[c] = coeff
            return out
        return True

    def __iter__(self):
        return iter(self.basis)

    def __getitem__(self, index):
        return self.basis[index]

    def is_subalgebra(
        self, return_structure_data=False, surface_singularities=None
    ):  ###!!! add structure data branch
        if surface_singularities is None and self._parameters:
            surface_singularities = True
        if self._is_subalgebra is None:
            out = self.ambient.is_subspace_subalgebra(
                self.filtered_basis, surface_singularities=surface_singularities
            )
            if surface_singularities:
                self._is_subalgebra, _ = out
            else:
                self._is_subalgebra = out

        return self._is_subalgebra

    def __str__(self):
        b = getattr(self, "basis", None) or []
        return "span{" + ", ".join(str(e) for e in b) + "}"

    def _repr_latex_(self, raw: bool = False, **kwargs):
        """verbose=True keyword will override elision formatting in output."""
        b = getattr(self, "basis", None) or []
        if len(b) > 6 and not kwargs.pop("verbose", False):
            inner = [
                b[idx]._repr_latex_(raw=True)
                if idx < 2
                else b[-1]._repr_latex_(raw=True)
                if idx > 2
                else r"\ldots"
                for idx in range(4)
            ]
            inner = ", ".join(inner)
        else:
            inner = ", ".join(e._repr_latex_(raw=True) for e in b)
        inner = str(inner).replace("$", "").replace(r"\displaystyle", "")
        out = rf"\left\langle {inner} \right\rangle"
        return out if raw else rf"$\displaystyle {out}$"

    def _latex(self, printer=None, raw: bool = True, **kwargs):
        return self._repr_latex_(raw=raw)

    def __add__(self, other):
        if _scalar_is_zero(other):
            return self
        if get_dgcv_category(other) in {"algebra_subspace", "subalgebra"}:
            if other.dimension == 0:
                return self
            if self.dimension == 0:
                return other
            new_basis = list(self.basis)
            for elem in getattr(other, "basis_in_ambient_alg", other.basis):
                new_basis = _basis_builder(
                    new_basis, elem
                )  ###!!! optimize with wedge method
            return algebra_subspace_class(new_basis, self.ambient)
        return NotImplemented

    def __radd__(self, other):
        if _scalar_is_zero(other):
            return self
        return NotImplemented

    def generate_subalgebra(
        self, simplify_basis=False, simplify_products_by_default=None
    ):
        if get_dgcv_category(self) == "subalgebra":
            return self
        basis = list(self.basis)
        in_dim = len(basis)
        amb = self.ambient
        amb_dim = amb.dimension
        skew = amb.is_skew_symmetric()
        variables = [symbol(f"_indep_check_{idx}") for idx in range(amb_dim - 1)]
        sd_out = dict()
        previous_basis = []
        for _ in range(in_dim):  # it should never reach this bound
            prev_dim = len(previous_basis)
            previous_basis = list(basis)
            current_dim = len(previous_basis)
            if prev_dim == current_dim:
                break

            for idx1, e1 in enumerate(previous_basis):
                if idx1 < prev_dim:
                    start = prev_dim
                else:
                    start = idx1 + 1 if skew else 0
                for idx2 in range(start, current_dim):
                    e2 = previous_basis[idx2]
                    product = e1 * e2
                    result, cd = _indep_check(
                        basis,
                        product,
                        return_decomp_coeffs=True,
                        _solve_variables=variables,
                    )
                    if result is True:
                        out_int = len(basis)
                        sd_out[(idx1, idx2)] = {out_int: 1}
                        if skew:
                            sd_out[(idx2, idx1)] = {out_int: -1}
                        basis.append(product)
                    else:
                        out_coeffs = cd[0]
                        sd_out[(idx1, idx2)] = out_coeffs
                        if skew:
                            sd_out[(idx2, idx1)] = {
                                i: -coef for i, coef in out_coeffs.items()
                            }
        dim = len(basis)
        sd_out = _structure_array(
            {k: matrix_dgcv(v, shape=(dim, 1)) for k, v in sd_out.items()}, dim
        )

        sd_out = array_dgcv(
            {k: matrix_dgcv(v, shape=(dim, 1)) for k, v in sd_out.items()},
            shape=(dim, dim),
            null_return=freeze_matrix(matrix_dgcv.zeros(dim, 1)),
        )
        from .algebras_secondary import subalgebra_class

        return subalgebra_class(
            basis,
            amb,
            _compressed_structure_data=sd_out,
            _internal_lock=retrieve_passkey(),
            span_warning=False,
            simplify_basis=simplify_basis,
            simplify_products_by_default=simplify_products_by_default,
        )

    @property
    def endomorphism_algebra(self):
        if self._endomorphisms is None:
            self._endomorphisms = vector_space_endomorphisms(self)
        return self._endomorphisms

    def append(self, item, recompute_gradings_and_return_new=False):
        if _scalar_is_zero(item):
            pass
        elif recompute_gradings_and_return_new:
            bas = list(self.filtered_basis)
            bas.append(item)
            return self.ambient.subspace(bas)
        elif not self.contains(item):
            self.original_basis = list(self.original_basis) + [item]
            self.basis = tuple(self.basis) + (item,)
            self.filtered_basis = tuple(self.filtered_basis) + (item,)
            self.dimension += 1
            self._basis_index_cache = None
            self.grading = [(0,) * self.dimension]


# -----------------------------------------------------------------------------
# utilities
# -----------------------------------------------------------------------------
def _commutant_eigenspace_vectors_old(
    solMat,
    *,
    tries=30,
    bound=None,
):
    n = solMat.nrows

    free_vars = set()
    for v in solMat._data.values():
        if v is None:
            continue
        free_vars |= get_free_symbols(v)
    free_vars = list(free_vars)

    if bound is None:
        bound = max(100, 10 * n)

    last_err = None

    for _ in range(max(1, int(tries))):
        if free_vars:
            spec = {var: random.randint(1, bound) for var in free_vars}
            M = subs(solMat, spec)
        else:
            M = solMat

        try:
            if engine_kind() == "sympy":
                sp = engine_module()
                lam = sp.Symbol(create_key(prefix="lam"))
                Id = matrix_dgcv.identity(n)

                try:
                    cp = sp.Matrix(M.to_list()).charpoly(lam).as_poly(lam)
                except Exception:
                    cp = sp.Poly(simplify((M - lam * Id).det()), lam)

                try:
                    evals = list(cp.all_roots())
                except Exception:
                    evals = []

                evals = [r for r in evals if r is not None]
                evals_u = []
                for r in evals:
                    if r not in evals_u:
                        evals_u.append(r)
                if len(evals_u) < 2:
                    continue

                eigspaces = []
                for r in evals_u:
                    ns = (M - r * Id).nullspace()
                    if ns:
                        eigspaces.append((r, ns))

                if len(eigspaces) < 2:
                    continue

            else:
                eigdata = M._eigenvects_by_engine()
                eigspaces = [(lam, vecs) for (lam, _mult, vecs) in eigdata if vecs]
                if len(eigspaces) < 2:
                    continue

            cols = []
            for _, vecs in eigspaces:
                for v in vecs:
                    if isinstance(v, matrix_dgcv):
                        cols.append([v[i, 0] for i in range(v.nrows)])
                    else:
                        cols.append(list(v))

            basis_cols = []
            for c in cols:
                if not basis_cols:
                    basis_cols.append(c)
                    if len(basis_cols) == n:
                        break
                    continue

                r0 = matrix_dgcv.from_cols(basis_cols).rank()
                r1 = matrix_dgcv.from_cols(basis_cols + [c]).rank()
                if r1 > r0:
                    basis_cols.append(c)

                if len(basis_cols) == n:
                    break

            if len(basis_cols) != n:
                continue

            return M, [r for r, _ in eigspaces], eigspaces

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        "Unable to obtain a commutant specialization yielding >= 2 eigenspaces and a full spanning set of eigenvectors."
    ) from last_err


def _commutant_eigenspace_vectors(mat, free_vars, max_attempts=6):
    ordered_vars = sorted(free_vars, key=str)
    expected = len(ordered_vars)
    dim = mat.shape[0]
    for attempt in range(max_attempts):
        rng = random.Random(9176 + attempt)
        weights = rng.sample(range(1, 16 * (attempt + 2)), expected)
        specialized = mat.subs(dict(zip(ordered_vars, weights)))
        try:
            packets = specialized._eigenvects_by_engine()
        except Exception:
            continue
        if len(packets) != expected:
            continue
        vectors = [list(packet[-1]) for packet in packets]
        if sum(len(block) for block in vectors) != dim:
            continue
        return vectors
    return None


def decompose_semisimple_algebra(
    alg,
    assume_semisimple=False,
    format_as_lists_of_elements=False,
    surface_singularities=False,
    simplify_singularities=None,
):
    assert get_dgcv_category(alg) in {"algebra", "subalgebra"}
    if alg.dimension == 0:
        return ([alg], []) if surface_singularities else [alg]
    if assume_semisimple is False and not alg.is_semisimple():
        raise TypeError(
            "decompose_semisimple_algebra was given a non-semisimple algebra to decompose."
        )

    n = alg.dimension
    get_slice = alg._structure_data_slice
    slice_shape = (n, n)
    mbasis = [
        matrix_dgcv(get_slice(idx), shape=slice_shape).transpose()
        for idx in range(alg.dimension)
    ]

    pref = create_key("_var")
    variables = [symbol(f"{pref}{j}") for j in range(n * n)]
    vMat = matrix_dgcv(dict(enumerate(variables)), shape=(n, n))

    mats = []
    for mat in mbasis:
        comm = (vMat @ mat) - (mat @ vMat)
        mats += list(comm._data.values())
    if surface_singularities is True:
        sol, sing = solve_dgcv(
            mats,
            variables,
            method="linsolve",
            return_divisors=True,
            pass_to_symbolic_engine=False,
            simplify_pivots=simplify_singularities
            if simplify_singularities is not None
            else True,
            simplify_result=False,
        )
    else:
        sol = solve_dgcv(mats, variables, method="linsolve", simplify_result=False)
    if not sol:
        raise RuntimeError("solve_dgcv failed in decompose_semisimple_algebra.")

    solMat = vMat.subs(sol[0])

    free_vars = set()
    for v in solMat._data.values():
        if v is None:
            continue
        free_vars |= get_free_symbols(v)
    free_vars &= set(variables)
    if len(free_vars) < 2:
        out = [list(alg.basis)] if format_as_lists_of_elements else [alg]
        if surface_singularities is True:
            return out, sing
        return out

    eigspaces = _commutant_eigenspace_vectors(solMat, free_vars)
    if eigspaces is None:
        dgcv_warning(
            "decompose_semisimple_algebra could not separate the simple ideals of a "
            f"{n}-dimensional semisimple algebra whose commutant has dimension "
            f"{len(free_vars)}. Returning the algebra undecomposed.",
            wc_label="debug_log",
        )
        out = [list(alg.basis)] if format_as_lists_of_elements else [alg]
        if surface_singularities is True:
            return out, sing
        return out

    simples = []
    for vecs in eigspaces:
        new_basis = []
        for v in vecs:
            new_basis.append(zip_sum(v, alg.basis))

        if not new_basis:
            continue

        if format_as_lists_of_elements is True:
            simples.append(new_basis)
        else:
            simples.append(alg.subalgebra(new_basis, simplify_basis=True))

    if not simples:
        out = [list(alg.basis)] if format_as_lists_of_elements else [alg]
        if surface_singularities is True:
            return out, sing
        return out
    if surface_singularities is True:
        return simples, sing
    return simples


def killingForm(alg, assume_Lie_algebra=False):
    if get_dgcv_category(alg) not in {"algebra", "subalgebra"}:
        raise Exception(
            "killingForm expected to receive an algebra instance."
        ) from None
    if alg._killing_form is None:
        if assume_Lie_algebra is False and not alg.is_Lie_algebra():
            raise Exception(
                "killingForm expects argument to be a Lie algebra instance of the algebra"
            ) from None
        aRepLoc = adjointRepresentation(alg, assume_Lie_algebra=assume_Lie_algebra)
        alg._killing_form = matrix_dgcv(
            [
                [(aRepLoc[j] * aRepLoc[k]).trace() for k in range(alg.dimension)]
                for j in range(alg.dimension)
            ]
        )

    return alg._killing_form


def adjointRepresentation(alg, list_format=False, assume_Lie_algebra=False):
    if get_dgcv_category(alg) in {"algebra", "subalgebra"}:
        if assume_Lie_algebra is False and not alg.is_Lie_algebra():
            dgcv_warning(
                "The algebra passed to `adjointRepresentation` is not a Lie algebra; there is likely a mistake if applying  `adjointRepresentation`."
            )
        get_slice = alg._structure_data_slice
        shp = (alg.dimension, alg.dimension)
        return [
            matrix_dgcv(get_slice(idx), shape=shp).transpose()
            for idx in range(alg.dimension)
        ]
    else:
        raise Exception(
            "adjointRepresentation expected to receive an algebra instance."
        ) from None


###### summary helpers
def _alg_name_plain(alg) -> str:
    try:
        return alg.__str__(VLP=False)
    except Exception:
        return str(getattr(alg, "label", None) or "Unnamed Algebra")


def _alg_name_latex(alg) -> tuple[str, str]:
    try:
        s = alg._repr_latex_(abbrev=True, raw=True)
        s = str(s).replace("$", "").replace(r"\displaystyle", "").strip()
        if not s:
            raise RuntimeError
        return f"${s}$", f"${s}$"
    except Exception:
        nm = _alg_name_plain(alg)
        cap = nm if nm[:1].isupper() else (nm[:1].upper() + nm[1:])
        return nm, cap


def _fmt_bool_cache(v):
    return "true" if v is True else ("false" if v is False else "not yet evaluated")


def _ellide(str_list, *, max_items: int):
    str_list = list(str_list or [])
    if len(str_list) <= max_items:
        return str_list
    k = max_items // 2
    return str_list[:k] + ["..."] + str_list[-k:]


def _fmt_angle_list(xs, *, max_items: int = 12) -> str:
    toks = [str(x) for x in _ellide(xs, max_items=max_items)]
    return "<" + ", ".join(toks) + ">"


def _fmt_grading_plain(grading, *, max_items: int = 12) -> str:
    if not isinstance(grading, (list, tuple)) or not grading:
        return "None"
    out = []
    for g in grading:
        if not isinstance(g, (list, tuple)):
            out.append(str(g))
            continue
        toks = [str(x) for x in _ellide(list(g), max_items=max_items)]
        out.append("(" + ", ".join(toks) + ")")
    return "[" + ", ".join(out) + "]"


def _basic_items_plain(refAlg, *, subAlg: bool, algebra_name: str) -> list[str]:
    items = []
    if subAlg:
        items.append(f"Subalgebra contained in {algebra_name}")
    items.append(f"Dimension: {refAlg.dimension}")

    lie = getattr(refAlg, "_lie_algebra_cache", None)
    if lie is True:
        items.append("Lie algebra: true")
        st = getattr(refAlg, "_educed_properties", dict()).get("special_type", None)
        if st is not None:
            items.append(f"special properties: {st}")
        elif (
            getattr(refAlg, "_is_semisimple_cache", None) is False
            and getattr(refAlg, "_is_solvable_cache", None) is False
        ):
            items.append("special properties: neither solvable nor semisimple")
        else:
            items.append("special properties: not yet evaluated")
    elif lie is False:
        items.append("Lie algebra: false")
        items.append(
            f"Skew symmetric: {_fmt_bool_cache(getattr(refAlg, '_skew_symmetric_cache', None))}"
        )
        items.append(
            f"Jacobi identity satisfied: {_fmt_bool_cache(getattr(refAlg, '_jacobi_identity_cache', None))}"
        )
    else:
        items.append("Lie algebra: not yet evaluated")

    return items


def _timed_progress_call(
    fn,
    *,
    timed: bool,
    threshold_s: float,
    step_desc: str,
    continue_desc: str | None,
    progress_message: str | None,
    _on_timed_update=None,
):
    if not timed:
        return fn()

    fired = {"v": False}
    timer = {"obj": None}
    use_signal = False

    try:
        import threading as _threading

        use_signal = _threading.current_thread() is _threading.main_thread()
    except Exception:
        use_signal = False

    def _emit_update():
        if fired["v"]:
            return
        fired["v"] = True
        if callable(_on_timed_update):
            try:
                _on_timed_update()
            except Exception:
                pass
        print(f"Update: {step_desc}.")
        if progress_message:
            print(progress_message)

    if use_signal:
        prev_handler = None
        prev_itimer = None

        def _handler(signum, frame):
            _emit_update()

        try:
            import signal

            prev_handler = signal.getsignal(signal.SIGALRM)
            prev_itimer = signal.getitimer(signal.ITIMER_REAL)
        except Exception:
            prev_handler = None
            prev_itimer = None

        try:
            import signal

            signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, max(0.0, float(threshold_s)))
        except Exception:
            use_signal = False
            try:
                import signal

                if prev_handler is not None:
                    signal.signal(signal.SIGALRM, prev_handler)
                if prev_itimer is not None:
                    signal.setitimer(signal.ITIMER_REAL, prev_itimer[0], prev_itimer[1])
            except Exception:
                pass

    if not use_signal:
        try:
            import threading

            t = threading.Timer(max(0.0, float(threshold_s)), _emit_update)
            timer["obj"] = t
            t.daemon = True
            t.start()
        except Exception:
            timer["obj"] = None

    try:
        out = fn()
    finally:
        if use_signal:
            try:
                import signal

                signal.setitimer(signal.ITIMER_REAL, 0.0)
                if prev_handler is not None:
                    signal.signal(signal.SIGALRM, prev_handler)
                if prev_itimer is not None:
                    signal.setitimer(signal.ITIMER_REAL, prev_itimer[0], prev_itimer[1])
            except Exception:
                pass
        else:
            try:
                t = timer["obj"]
                if t is not None:
                    t.cancel()
            except Exception:
                pass

    if fired["v"] and continue_desc:
        print(f"Continuing to {continue_desc}.")
    return out


def _merge_rank_singularities(alg, refAlg, divisors):
    terms = [v for v in divisors if get_free_symbols(v)]
    if not terms:
        return
    hosts = (alg,) if alg is refAlg else (alg, refAlg)
    for host in hosts:
        merged = list(host._singularities.get("subalgebra_ranks", [])) + terms
        if get_dgcv_settings_registry().get(
            "simplify_singularity_ideals_by_default", True
        ):
            merged = expr_union_primitives(
                merged,
                order_coordinates(host._parameters),
                process_rationals=True,
                fail_quietly=True,
            )
        host._singularities["subalgebra_ranks"] = merged


def _summary_warm_caches(
    refAlg,
    *,
    subAlg: bool,
    reporting_threshold_s: float = 10.0,
    progress_message: str | None = None,
    full=False,
    force_heavy_solve: bool = False,
    _on_timed_update=None,
):
    thr = float(reporting_threshold_s)
    heavy = bool(force_heavy_solve)

    def _timed_kwargs(continue_desc):
        return {
            "_timed_reporting": True,
            "_reporting_threshold_s": thr,
            "_progress_message": continue_desc,
            "_on_timed_update": _on_timed_update,
        }

    def _timed_step(fn, step_desc, continue_desc):
        try:
            return _timed_progress_call(
                fn,
                timed=True,
                threshold_s=thr,
                step_desc=step_desc,
                continue_desc=continue_desc,
                progress_message=None,
                _on_timed_update=_on_timed_update,
            )
        except Exception:
            return None

    def _warm_ideal_ranks():
        ld = getattr(refAlg, "_Levi_deco_cache", None)
        simples = ld.get("simple_ideals", None) if isinstance(ld, dict) else None
        if not simples:
            return
        surfacing = bool(getattr(refAlg, "_parameters", None))
        total = len(simples)
        for idx, ideal in enumerate(simples, start=1):
            if getattr(ideal, "_rank_approximation", None) is not None:
                continue
            out = _timed_step(
                lambda a=ideal: a.approximate_rank(
                    _use_cache=True,
                    assume_semisimple=True,
                    surface_singularities=surfacing,
                ),
                f"estimating the rank of simple ideal {idx} of {total}",
                progress_message
                if idx == total
                else "finish estimating the ranks of the simple ideals",
            )
            if surfacing and isinstance(out, tuple) and len(out) == 2:
                try:
                    _merge_rank_singularities(ideal, refAlg, out[1])
                except Exception:
                    pass

    is_lie = refAlg.is_Lie_algebra(
        verbose=False,
        **_timed_kwargs(progress_message),
    )

    if not is_lie:
        return

    recovered = False
    try:
        refAlg.Levi_decomposition(
            decompose_semisimple_fully=full,
            verbose=False,
            force_heavy_solve=heavy,
            **_timed_kwargs(progress_message),
        )
    except Exception:
        if not heavy and refAlg._parameters:
            print(
                "A decomposition subroutine failed. Since the algebra's structure "
                "equations involve parameters, retrying with the heavier solve "
                "algorithm."
            )
            try:
                refAlg.Levi_decomposition(
                    decompose_semisimple_fully=full,
                    verbose=False,
                    force_heavy_solve=True,
                    _bust_cache=True,
                    **_timed_kwargs(progress_message),
                )
                recovered = True
                heavy = True
            except Exception:
                recovered = False
        if not recovered:
            if subAlg and not refAlg._parameters:
                print(
                    "A decomposition subroutine failed; proceeding with a partial report."
                    "Currently, summary is not fully tested for subalgebras, and that may be the reason."
                    "Suggestion: convert to an algebra_class via the subalgebra copy method."
                )
            else:
                addon = (
                    ", likely due to a presence of parameters in the algebra structure equations which is not fully tested across the algebra_class methods"
                    if refAlg._parameters
                    else ""
                )
                print(
                    f"A decomposition subroutine failed{addon}; proceeding with a partial report."
                )

    rad = None
    try:
        ld = getattr(refAlg, "_Levi_deco_cache", None)
        comps = ld.get("LD_components", None) if isinstance(ld, dict) else None
        if isinstance(comps, (list, tuple)) and len(comps) > 1:
            rad = comps[1]
    except Exception:
        rad = None

    if rad is None:
        try:
            rad = getattr(refAlg, "_radical_cache", None)
        except Exception:
            rad = None

    if rad is not None and getattr(rad, "dimension", 0) != 0:
        _timed_step(
            lambda: rad.derived_series(force_heavy_solve=heavy),
            "computing the maximal solvable ideal's derived series",
            "compute the maximal solvable ideal's lower central series",
        )
        _timed_step(
            lambda: rad.lower_central_series(),
            "computing the maximal solvable ideal's lower central series",
            "compute the center" if full else progress_message,
        )

    if full:
        _timed_step(
            lambda: refAlg.center(),
            "computing the center",
            progress_message,
        )

    try:
        abelian = refAlg.is_abelian(**_timed_kwargs(progress_message))
    except Exception:
        abelian = None

    if not abelian:
        try:
            is_ss = refAlg.is_semisimple(
                verbose=False,
                **_timed_kwargs(progress_message),
            )
        except Exception:
            is_ss = False

        if is_ss:
            try:
                refAlg.is_simple(
                    verbose=False,
                    **_timed_kwargs(progress_message),
                )
            except Exception:
                pass
        else:
            try:
                is_sol = refAlg.is_solvable(**_timed_kwargs(progress_message))
            except Exception:
                is_sol = False

            if is_sol:
                try:
                    refAlg.is_nilpotent(**_timed_kwargs(progress_message))
                except Exception:
                    pass

    _warm_ideal_ranks()


def _summary_render_plain(
    parentAlg,
    refAlg,
    *,
    subAlg: bool,
    algebra_name: str,
    algebra_name_cap: str,
) -> str:
    nm = _alg_name_plain(parentAlg)
    alg_dim = getattr(refAlg, "dimension", None)

    lines = [f"=== Algebra Summary: {nm} ({alg_dim} dimensional) ==="]

    if getattr(refAlg, "dimension", None) == 0:
        if subAlg:
            lines.append(
                f"  - This is the trivial 0-dimensional subalgebra in {algebra_name}."
            )
        else:
            lines.append("  - This is the trivial 0-dimensional algebra.")
        return "\n".join(lines).rstrip()

    lines.append("Basic properties:")
    for it in _basic_items_plain(refAlg, subAlg=subAlg, algebra_name=algebra_name):
        lines.append(f"  - {it}")

    basis = getattr(refAlg, "basis", ()) or ()
    lines.append("Basis and grading:")
    lines.append(f"  - basis: {_fmt_angle_list(basis, max_items=12)}")
    grad = getattr(refAlg, "grading", None)
    lines.append(f"  - grading: {_fmt_grading_plain(grad, max_items=12)}")

    ld = getattr(refAlg, "_Levi_deco_cache", None)
    if getattr(refAlg, "_lie_algebra_cache", None) is True and isinstance(ld, dict):
        comps = ld.get("LD_components", None)
        Levi_component = (
            comps[0] if isinstance(comps, (list, tuple)) and len(comps) > 0 else None
        )
        rad = comps[1] if isinstance(comps, (list, tuple)) and len(comps) > 1 else None
        simples = ld.get("simple_ideals", None)

        lines.append("Levi decomposition:")
        if refAlg.is_solvable():
            lines.append("  - The algebra equals its own maximal solvable ideal.")
        elif refAlg.is_semisimple():
            if simples is None:
                lines.append(
                    "  - The algebra is semisimple; the simple-ideal decomposition is not yet evaluated."
                )
            elif len(simples) == 1:
                lines.append("  - The algebra is simple.")
            else:
                lines.append(
                    "  - The algebra is a direct sum of the following simple ideals:"
                )
                for idx, alg in enumerate(simples, start=1):
                    adim = getattr(alg, "dimension", None)

                    try:
                        rank = alg.approximate_rank(
                            _use_cache=True, assume_semisimple=True
                        )
                    except Exception:
                        rank = "?"

                    typ = _classify_simple_by_dim_rank(adim, rank)
                    if typ is not None:
                        lines.append(f"      - Ideal {idx}: Type {typ}")
                    else:
                        lines.append(f"      - Ideal {idx}: dim={adim}, rank={rank}")
        else:
            ss_dim = (
                getattr(Levi_component, "dimension", "?") if Levi_component else "?"
            )
            rad_dim = getattr(rad, "dimension", "?") if rad else "?"
            lines.append("  - Semidirect sum of semisimple and solvable components:")
            lines.append(f"      - semisimple part: {ss_dim} dimensional")
            lines.append(f"      - max. solvable ideal: {rad_dim} dimensional")

        if (
            Levi_component is not None
            and getattr(Levi_component, "dimension", 0) != 0
            and simples is not None
            and len(simples) >= 2
        ):
            lines.append("Simple ideals in semisimple complement:")
            for idx, alg in enumerate(simples, start=1):
                adim = getattr(alg, "dimension", None)
                try:
                    rank = alg.approximate_rank(_use_cache=True, assume_semisimple=True)
                except Exception:
                    rank = "?"
                typ = _classify_simple_by_dim_rank(adim, rank)
                if typ is not None:
                    lines.append(f"  - Ideal {idx}: Type {typ}")
                else:
                    lines.append(f"  - Ideal {idx}: dim={adim}, rank={rank}")

        if rad is not None and getattr(rad, "dimension", 0) != 0:
            ds = getattr(rad, "_derived_series_cache", None)
            if ds is not None:
                lines.append("Derived series of the maximal solvable ideal:")
                seq = ds[0]
                for idx, level in enumerate(seq, start=1):
                    if not level or (
                        isinstance(level, (list, tuple))
                        and len(level) == 1
                        and bool(getattr(level[0], "is_zero", False))
                    ):
                        lines.append(f"  - Level {idx}: empty")
                    else:
                        lines.append(
                            f"  - Level {idx}: {_fmt_angle_list(level, max_items=12)}"
                        )

            lcs = getattr(rad, "_lower_central_series_cache", None)
            if lcs is not None:
                lines.append("Lower central series of the maximal solvable ideal:")
                seq = lcs[0]
                for idx, level in enumerate(seq, start=1):
                    if not level or (
                        isinstance(level, (list, tuple))
                        and len(level) == 1
                        and bool(getattr(level[0], "is_zero", False))
                    ):
                        lines.append(f"  - Level {idx}: empty")
                    else:
                        lines.append(
                            f"  - Level {idx}: {_fmt_angle_list(level, max_items=12)}"
                        )

    return "\n".join(lines).rstrip()


def _summary_render_rich(
    *,
    refAlg,
    subAlg: bool,
    algebra_name: str,
    algebra_name_cap: str,
    style,
    use_latex: bool,
    extra_support_for_math_in_tables: bool,
    show_singularities: bool | None = None,
    full=False,
):
    theme_vars, theme_data = get_style(style, return_theme_data=True)
    border_radius = int(
        theme_data.custom_css_vars.get("--dgcv-border-radius", "12px").replace("px", "")
    )
    container_id = f"dgcv-alg-summary-{uuid.uuid4().hex[:8]}"
    scoped_theme = theme_vars.replace(":root", f"#{container_id}")

    class _HTMLWrapper:
        def __init__(self, html):
            self._html = html

        def to_html(self, *args, **kwargs):
            return self._html

        def _repr_html_(self):
            return self._html

    uses_plaque = "--plaque-fill" in theme_data.custom_css_vars
    uses_plaque_border = "--plaque-border" in theme_data.custom_css_vars

    # for _stack_many
    if theme_data.custom_css_vars.get("--dgcv-special-background", None):
        panel_bg = "var(--dgcv-special-background)"
        panel_hd = "none"
        text_bg = "var(--dgcv-special-text,var(--dgcv-text-heading))"
    else:
        panel_bg = "var(--dgcv-bg-primary)"
        panel_hd = "var(--dgcv-bg-surface)"
        text_bg = "var(--dgcv-text-main)"

    def _stack_many(blocks) -> str:
        inner = "\n".join(f'<div class="section">{b}</div>' for b in blocks)
        return textwrap.dedent(f"""
            <div id="{container_id}">
            <style>
            {scoped_theme}
            #{container_id} .stack {{ display: flex; flex-direction: column; gap: 16px; align-items: stretch; width: 100%; margin: 0; }}
            #{container_id} .section {{ width: 100%; }}
            #{container_id} .dgcv-panel {{
                background: {panel_bg};
                box-shadow: var(--dgcv-table-shadow, none);
                border: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main);
                border-image: var(--dgcv-border-image, none);
                color: {text_bg};
                font-family: var(--dgcv-font-family, inherit);
                overflow: hidden;
                padding: 4px 4px;
                margin: 0;
            }}
            #{container_id} .dgcv-panel-head {{ padding: 0.75rem 1rem; background: {panel_hd}; }}
            #{container_id} .dgcv-panel-title {{ margin: 0; font-size: 1rem; font-weight: 600; color: var(--dgcv-special-text, var(--dgcv-text-heading)); text-shadow: var(--dgcv-text-shadow, none); }}
            #{container_id} .dgcv-panel-rule {{ border: 0; height: 2px; background: var(--dgcv-border-main); margin: 0; }}
            #{container_id} .dgcv-panel-body {{ padding: 0.75rem 1rem; overflow-x: auto; width: 100%; box-sizing: border-box; }}
            #{container_id} .dgcv-panel-list ul {{ margin: 0.25rem 0 0 1.25rem; color: {text_bg}; }}
            #{container_id} .dgcv-panel-footer {{ padding: 0.5rem 1rem; background: var(--dgcv-bg-alt); color: var(--dgcv-text-alt); border-top: 1px solid var(--dgcv-border-alt); }}
            #{container_id} .dgcv-data-table {{ width: 100%; border-collapse: collapse; background: var(--dgcv-bg-primary); color: var(--dgcv-text-main); margin: 0; }}
            #{container_id} .dgcv-data-table td, #{container_id} .dgcv-data-table th {{ border-right: 1px solid var(--dgcv-border-main); padding: 8px 12px; }}
            #{container_id} .dgcv-data-table thead th {{ background-color: var(--dgcv-bg-surface); color: var(--dgcv-text-heading); border-bottom: 3px solid var(--dgcv-border-main); }}
            #{container_id} .dgcv-data-table th.row_heading {{ background-color: var(--dgcv-bg-surface) !important; color: var(--dgcv-text-heading) !important; font-weight: bold; }}
            #{container_id} .dgcv-data-table tr:nth-child(even) {{ background-color: var(--dgcv-bg-alt); color: var(--dgcv-text-alt); }}
            #{container_id} .dgcv-table-wrap {{ overflow-x: auto; max-width: 100%; box-sizing: border-box; padding: 0; margin: 0; }}
            #{container_id} .dgcv-table-wrap > table.dgcv-data-table {{ min-width: 40rem; width: 100%; table-layout: fixed; }}
            #{container_id} .dgcv-data-table tbody tr {{
                transition: var(--dgcv-hover-transition, transform 0.2s, box-shadow 0.2s, background-color 0.2s);
            }}
            #{container_id} .dgcv-data-table tbody tr:hover {{
                background-color: var(--dgcv-bg-hover) !important;
                color: var(--dgcv-text-hover) !important;
                transform: var(--dgcv-hover-transform, none);
            }}
            #{container_id} .dgcv-data-table tbody tr:hover th.row_heading {{
                background-color: var(--dgcv-bg-surface) !important;
                color: var(--dgcv-text-heading) !important;
                transform: none !important;
            }}
            </style>
            <div class="stack">
            {inner}
            </div>
            </div>
        """).strip()

    def _corners_for(i: int, total: int):
        r = border_radius
        if total <= 1:
            return {"ul": r, "ur": r, "ll": r, "lr": r}
        if i == 0:
            return {"ul": r, "ur": r, "ll": 0, "lr": 0}
        if i == total - 1:
            return {"ul": 0, "ur": 0, "ll": r, "lr": r}
        return {"ul": 0, "ur": 0, "ll": 0, "lr": 0}

    def _fmt_bool_cache(v):
        return "true" if v is True else ("false" if v is False else "not yet evaluated")

    empty_tok = r"$\varnothing$" if use_latex else "empty"

    def _is_trivial_level(level) -> bool:
        if not level:
            return True
        if isinstance(level, (list, tuple)) and len(level) == 1:
            z = level[0]
            return bool(getattr(z, "is_zero", False))
        return False

    def _fmt_basis_list(elems):
        if _is_trivial_level(elems):
            return empty_tok
        if use_latex:
            out = []
            for elem in elems:
                try:
                    out.append(f"${elem._repr_latex_(raw=True)}$")
                except Exception:
                    out.append(repr(elem))
            return ", ".join(out)
        return ", ".join(repr(elem) for elem in elems)

    def _level_dim(elems):
        if _is_trivial_level(elems):
            return 0
        try:
            return len(elems)
        except Exception:
            return 0

    params_check = list(getattr(refAlg, "_parameters", []))
    if use_latex:
        try:
            from .._aux.printing.printing._dgcv_display import LaTeX_list

            params = LaTeX_list(params_check, math_mode="$")
        except Exception:
            params = [repr(b) for b in params_check]
    else:
        params = [repr(b) for b in params_check]

    if params:
        items = (
            [
                f"Subalgebra family contained in {algebra_name}",
                f"Dimension: {refAlg.dimension}",
                f"Parameters: {params}",
            ]
            if subAlg
            else [f"Dimension: {refAlg.dimension}", f"Parameters: {params}"]
        )
    else:
        items = (
            [
                f"Subalgebra contained in {algebra_name}",
                f"Dimension: {refAlg.dimension}",
            ]
            if subAlg
            else [f"Dimension: {refAlg.dimension}"]
        )

    lie = getattr(refAlg, "_lie_algebra_cache", None)
    if lie is True:
        items.append("Lie algebra: true")
        special_property = getattr(refAlg, "_educed_properties", dict()).get(
            "special_type", None
        )
        if special_property is not None:
            items.append(f"special properties: {special_property}")
        elif (
            getattr(refAlg, "_is_semisimple_cache", None) is False
            and getattr(refAlg, "_is_solvable_cache", None) is False
        ):
            items.append("special properties: neither solvable nor semisimple")
        else:
            items.append("special properties: not yet evaluated")
    elif lie is False:
        items.append("Lie algebra: false")
        items.append(
            f"Skew symmetric: {_fmt_bool_cache(getattr(refAlg, '_skew_symmetric_cache', None))}"
        )
        items.append(
            f"Jacobi identity satisfied: {_fmt_bool_cache(getattr(refAlg, '_jacobi_identity_cache', None))}"
        )
    else:
        items.append("Lie algebra: not yet evaluated")

    if refAlg.dimension == 0:
        pv0 = panel_view(
            header="Basic properties of the subalgebra"
            if subAlg
            else f"Basic properties of {algebra_name}",
            itemized_text=[
                f"This is the trivial 0-dimensional subalgebra in {algebra_name}."
            ]
            if subAlg
            else ["This is the trivial 0-dimensional algebra."],
            theme_css_vars=theme_vars,
            extra_css="",
            slim=True,
        ).to_html()
        return latex_in_html(
            _HTMLWrapper(_stack_many([pv0])),
            extra_support_for_math_in_tables=extra_support_for_math_in_tables,
        )

    basis_elems = getattr(refAlg, "basis", ()) or ()
    if use_latex:
        try:
            basis_labels = [f"${b._repr_latex_(raw=True)}$" for b in basis_elems]
        except Exception:
            basis_labels = [repr(b) for b in basis_elems]
    else:
        basis_labels = [repr(b) for b in basis_elems]

    rows = [list(basis_labels)]
    grad_index_labels = ["Basis"]
    warn_msgs = []
    grad = getattr(refAlg, "grading", None)

    def _fmt_weight(x):
        if use_latex and hasattr(x, "_repr_latex_"):
            try:
                s = str(x._repr_latex_())
                if s.startswith("$") and s.endswith("$"):
                    s = s[1:-1]
                reduced = (
                    s.replace(r"\\displaystyle", "")
                    .replace(r"\displaystyle", "")
                    .strip()
                )
                return f"${reduced}$"
            except Exception:
                pass
        return str(x)

    if isinstance(grad, (list, tuple)) and grad:
        for gi, g in enumerate(grad, start=1):
            if isinstance(g, (list, tuple)) and len(g) == len(basis_labels):
                rows.append([_fmt_weight(x) for x in g])
                grad_index_labels.append(f"Grading {gi}")
            else:
                warn_msgs.append(f"grading {gi} invalid or length mismatch")

    footer_rows = (
        [
            [
                {
                    "html": f"<em>{_esc(' | '.join(warn_msgs))}</em>",
                    "attrs": {"colspan": len(basis_labels)},
                }
            ]
        ]
        if warn_msgs
        else None
    )
    sections = []

    def _build_basic_panel(corner_kwargs):
        label = (
            ("the subalgebra family" if subAlg else "the algebra family")
            if params
            else ("the subalgebra" if subAlg else algebra_name)
        )
        return panel_view(
            header=f"Basic properties of {label}",
            itemized_text=items,
            theme_css_vars="",
            extra_css="",
            slim=True,
            plaque_fill=uses_plaque,
            plaque_border=uses_plaque_border,
            plaque_content=True,
            **corner_kwargs,
        ).to_html()

    sections.append(("panel", _build_basic_panel))

    def _build_basis_panel(corner_kwargs):
        table_obj = build_matrix_table(
            show_headers=False,
            index_labels=grad_index_labels,
            columns=[],
            rows=rows,
            caption="",
            theme_css_vars="",
            extra_css="",
            footer_rows=footer_rows,
            table_attrs='style="table-layout:auto;"',
            cell_align=None,
            escape_cells=False,
            escape_headers=True,
            nowrap=False,
            dashed_corner=False,
            slim=True,
            panel_content=True,
        )
        return panel_view(
            header="Basis and assigned grading(s)",
            primary_text=table_obj,
            theme_css_vars="",
            extra_css="",
            slim=True,
            plaque_fill=uses_plaque,
            plaque_border=uses_plaque_border,
            plaque_content=False,
            **corner_kwargs,
        ).to_html()

    sections.append(("panel", _build_basis_panel))

    if getattr(refAlg, "_center_cache", None):

        def _center_panel(corner_kwargs):
            IT = []
            center = getattr(refAlg, "_center_cache", None)
            PT = center._repr_latex_(raw=False, verbose=True)
            return panel_view(
                header="Center",
                primary_text=PT,
                itemized_text=IT,
                theme_css_vars="",
                extra_css="",
                slim=True,
                plaque_fill=uses_plaque,
                plaque_border=uses_plaque_border,
                plaque_content=True,
                **corner_kwargs,
            ).to_html()

        sections.append(("panel", _center_panel))

    ld = getattr(refAlg, "_Levi_deco_cache", None)
    if getattr(refAlg, "_lie_algebra_cache", None) is True and isinstance(ld, dict):
        simples = ld.get("simple_ideals", None)
        Levi_component, rad = ld.get("LD_components", (None, None))

        def _LD_panel(corner_kwargs):
            IT = []
            solv, semi = (
                getattr(refAlg, "_is_solvable_cache", None),
                getattr(refAlg, "_is_semisimple_cache", None),
            )
            if solv is True:
                PT = (
                    "The subalgebra equals its own maximal solvable ideal."
                    if subAlg
                    else f"{algebra_name_cap} equals its own maximal solvable ideal."
                )
            elif semi is True:
                if simples is None:
                    PT = (
                        "The subalgebra is semisimple"
                        if subAlg
                        else f"{algebra_name_cap} is semisimple"
                    )
                elif len(simples) == 1:
                    PT = (
                        "The subalgebra is simple"
                        if subAlg
                        else f"{algebra_name_cap} is simple"
                    )
                    if full is False:
                        PT += "."
                    else:
                        alg = next(iter(simples))
                        rank = "?"
                        try:
                            ss = True if len(params) > 0 else False
                            rankout = alg.approximate_rank(
                                _use_cache=True,
                                assume_semisimple=True,
                                surface_singularities=ss,
                            )
                            if ss:
                                rank, divisors = rankout
                            else:
                                rank = rankout
                            if divisors:
                                new_sing = list(
                                    alg._singularities.get("subalgebra_ranks", [])
                                ) + [v for v in divisors if get_free_symbols(v)]
                                if get_dgcv_settings_registry().get(
                                    "simplify_singularity_ideals_by_default", True
                                ):
                                    new_sing = expr_union_primitives(
                                        new_sing,
                                        order_coordinates(alg._parameters),
                                        process_rationals=True,
                                        fail_quietly=True,
                                    )
                                alg._singularities["subalgebra_ranks"] = new_sing
                                if alg != refAlg:
                                    new_sing = list(
                                        refAlg._singularities.get(
                                            "subalgebra_ranks", []
                                        )
                                    ) + [v for v in divisors if get_free_symbols(v)]
                                    if get_dgcv_settings_registry().get(
                                        "simplify_singularity_ideals_by_default", True
                                    ):
                                        new_sing = expr_union_primitives(
                                            new_sing,
                                            order_coordinates(refAlg._parameters),
                                            process_rationals=True,
                                            fail_quietly=True,
                                        )
                                    refAlg._singularities["subalgebra_ranks"] = new_sing
                        except Exception:
                            pass
                        IC = (
                            _simple_iso_label(
                                getattr(alg, "dimension", None),
                                rank,
                                use_latex=use_latex,
                            )
                            or "?"
                        )
                        PT += f" and isomorphic to {IC}." if IC != "?" else "."
                else:
                    PT = (
                        "The subalgebra is a direct sum"
                        if subAlg
                        else f"{algebra_name_cap} is a direct sum"
                    )
                    for a in simples:
                        IT.append(
                            f"${a._repr_latex_(raw=True, abbrev=True)}$"
                            if use_latex
                            else repr(a)
                        )
            else:
                PT = (
                    "The subalgebra is a semidirect sum"
                    if subAlg
                    else f"{algebra_name_cap} is a semidirect sum"
                )
                if isinstance(ld.get("LD_components", None), (list, tuple)):
                    for a in ld["LD_components"]:
                        IT.append(
                            f"${a._repr_latex_(raw=True, abbrev=True)}$"
                            if use_latex
                            else repr(a)
                        )
            return panel_view(
                header="Levi decomposition",
                primary_text=PT,
                itemized_text=IT,
                theme_css_vars="",
                extra_css="",
                slim=True,
                plaque_fill=uses_plaque,
                plaque_border=uses_plaque_border,
                plaque_content=True,
                **corner_kwargs,
            ).to_html()

        sections.append(("panel", _LD_panel))

        if getattr(refAlg, "_is_simple_cache", None) is not True and (
            Levi_component is not None
            and getattr(Levi_component, "dimension", 0) != 0
            and simples is not None
        ):

            def _ss_compl_panel(corner_kwargs):
                rows2 = []
                for idx, a in enumerate(simples):
                    rank = "?"
                    try:
                        ss = True if len(params) > 0 else False
                        rankout = a.approximate_rank(
                            _use_cache=True,
                            assume_semisimple=True,
                            surface_singularities=ss,
                        )
                        if ss:
                            rank, divisors = rankout
                        else:
                            rank = rankout
                        if divisors:
                            new_sing = list(
                                refAlg._singularities.get("subalgebra_ranks", [])
                            ) + [v for v in divisors if get_free_symbols(v)]
                            if get_dgcv_settings_registry().get(
                                "simplify_singularity_ideals_by_default", True
                            ):
                                new_sing = expr_union_primitives(
                                    new_sing,
                                    order_coordinates(refAlg._parameters),
                                    process_rationals=True,
                                    fail_quietly=True,
                                )
                            refAlg._singularities["subalgebra_ranks"] = new_sing
                            if refAlg != a:
                                new_sing = list(
                                    a._singularities.get("subalgebra_ranks", [])
                                ) + [v for v in divisors if get_free_symbols(v)]
                                if get_dgcv_settings_registry().get(
                                    "simplify_singularity_ideals_by_default", True
                                ):
                                    new_sing = expr_union_primitives(
                                        new_sing,
                                        order_coordinates(a._parameters),
                                        process_rationals=True,
                                        fail_quietly=True,
                                    )
                                a._singularities["subalgebra_ranks"] = new_sing
                    except Exception:
                        pass
                    rows2.append(
                        [
                            f"subalgebra {idx + 1}",
                            f"{getattr(a, 'dimension', None)}",
                            f"{rank}",
                            _simple_iso_label(
                                getattr(a, "dimension", None), rank, use_latex=use_latex
                            )
                            or "?",
                            _fmt_basis_list(getattr(a, "basis", ()) or ()),
                        ]
                    )
                table_obj = build_matrix_table(
                    index_labels=None,
                    columns=["Ideal #", "Dimension", "Rank", "Iso. Class", "Basis"],
                    rows=rows2,
                    theme_css_vars="",
                    extra_css="",
                    table_attrs='style="table-layout:auto;"',
                    escape_cells=False,
                    escape_headers=True,
                    dashed_corner=False,
                    slim=True,
                )
                return panel_view(
                    header="Simple ideals in semisimple complement",
                    primary_text=table_obj,
                    theme_css_vars="",
                    extra_css="",
                    slim=True,
                    plaque_fill=uses_plaque,
                    plaque_border=uses_plaque_border,
                    plaque_content=False,
                    **corner_kwargs,
                ).to_html()

            sections.append(("panel", _ss_compl_panel))

        if rad is not None and getattr(rad, "dimension", 0) != 0:
            for cache_attr, title in [
                ("_lower_central_series_cache", "Lower central series of radical"),
                ("_derived_series_cache", "Derived series of radical"),
            ]:
                cache = getattr(rad, cache_attr, None)
                if isinstance(cache, (list, tuple)) and cache and cache[0] is not None:

                    def _series_panel(corner_kwargs, c=cache[0], t=title):
                        rows2 = [
                            [
                                f"Level {idx + 1}",
                                f"{_level_dim(getattr(lvl, 'basis', lvl))}",
                                _fmt_basis_list(getattr(lvl, "basis", lvl)),
                            ]
                            for idx, lvl in enumerate(c)
                        ]
                        table_obj = build_matrix_table(
                            index_labels=None,
                            columns=["Filtration Level", "Dimension", "Basis"],
                            rows=rows2,
                            theme_css_vars="",
                            extra_css="",
                            table_attrs='style="table-layout:auto;"',
                            dashed_corner=False,
                            escape_cells=False,
                            escape_headers=True,
                            slim=True,
                        )
                        return panel_view(
                            header=t,
                            primary_text=table_obj,
                            theme_css_vars="",
                            extra_css="",
                            slim=True,
                            plaque_fill=uses_plaque,
                            plaque_border=uses_plaque_border,
                            plaque_content=False,
                            **corner_kwargs,
                        ).to_html()

                    sections.append(("panel", _series_panel))

    if show_singularities is not False and getattr(refAlg, "_singularities", False):

        def singularities_panel(corner_kwargs):
            type_dict = [
                ("radical", "radical"),
                ("LD", "Levi decomposition"),
                ("derived_series", "derived series"),
                ("simple_ideals", "simple subalgebras"),
                ("center", "center"),
                ("subalgebra_ranks", "subalgebra ranks"),
                ("structure", "structure coefficients"),
            ]
            items_sing = []
            max_operands = 1000
            for key, label in type_dict:
                if show_singularities is not True:
                    sings = refAlg._singularities.get(key, set())
                    printable = True
                    for sing in sings:
                        node_count = fast_printable(
                            sing, max_nodes=max_operands, return_count=True
                        )
                        max_operands -= node_count
                        if max_operands < 0:
                            printable = False
                            break
                    if not printable:
                        numb = len(sings)
                        plur = ("y", "it") if numb == 1 else ("ies", "them")
                        items_sing.append(
                            f"From {label}: {numb} singularit{plur[0]} omitted from report. (set `show_singularities=True` to display {plur[1]}. Warning: typically very long.)"
                        )
                        continue
                terms = list(refAlg._singularities.get(key, []))
                if terms:
                    formatted = (
                        LaTeX_list(terms, math_mode="$")
                        if use_latex
                        else ", ".join([repr(x) for x in terms])
                    )
                    items_sing.append(f"From {label}: {formatted}")
            return panel_view(
                header="Parameter space singularities",
                itemized_text=items_sing,
                theme_css_vars="",
                extra_css="",
                slim=True,
                plaque_fill=uses_plaque,
                plaque_border=uses_plaque_border,
                plaque_content=True,
                **corner_kwargs,
            ).to_html()

        sections.append(("singularities", singularities_panel))

    built_blocks = [
        builder(_corners_for(i, len(sections)))
        for i, (_, builder) in enumerate(sections)
    ]
    return latex_in_html(
        _HTMLWrapper(_stack_many(built_blocks)),
        extra_support_for_math_in_tables=extra_support_for_math_in_tables,
    )


def _classify_simple_by_dim_rank(dim, rank):
    """
    Return a Dynkin-type tag like 'A1', 'D4', 'G2', or 'B3 or C3', else None.
    """
    try:
        d = int(dim)
        r = int(rank)
    except Exception:
        return None

    if (r + 1) ** 2 - 1 == d:
        return f"A{r}"
    if (2 * r + 1) * r == d:
        return f"B{r} or C{r}"
    if (2 * r - 1) * r == d:
        return f"D{r}"

    if r == 2 and d == 14:
        return "G2"
    if r == 4 and d == 52:
        return "F4"
    if r == 6 and d == 78:
        return "E6"
    if r == 7 and d == 133:
        return "E7"
    if r == 8 and d == 248:
        return "E8"

    return None


def _simple_iso_label(dim, rank, *, use_latex: bool):
    typ = _classify_simple_by_dim_rank(dim, rank)
    if typ is None:
        return None

    if not use_latex:
        return typ

    if typ.startswith("A") and " or " not in typ:
        r = int(typ[1:])
        return rf"$\mathfrak{{sl}}_{{{r + 1}}}$"

    if typ.startswith("D") and " or " not in typ:
        r = int(typ[1:])
        return rf"$\mathfrak{{so}}_{{{2 * r}}}$"

    if " or " in typ and typ.startswith("B"):
        r = int(typ.split()[0][1:])
        return rf"$\mathfrak{{so}}_{{{2 * r + 1}}}$ or $\mathfrak{{sp}}_{{{2 * r}}}$"

    if typ in {"G2", "F4", "E6", "E7", "E8"}:
        return rf"$\operatorname{{Lie}}({typ})$"

    return rf"${typ}$"


# -----------------------------------------------------------------------------
# linear algebra for list/tuple processing
# -----------------------------------------------------------------------------
def multiply_matrices(A, B):
    """
    Multiplies two matrices A and B, represented as lists of lists.

    Parameters
    ----------
    A : list of lists
        The first matrix (m x n).
    B : list of lists
        The second matrix (n x p).

    Returns
    -------
    list of lists
        The resulting matrix (m x p) after multiplication.

    Raises
    ------
    ValueError
        If the number of columns in A is not equal to the number of rows in B.
    """
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if cols_A != rows_B:
        raise ValueError(
            "Incompatible matrix dimensions: A is {}x{}, B is {}x{}".format(
                rows_A, cols_A, rows_B, cols_B
            )
        ) from None
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result


def fast_rank(mat, surface_singularities=False, simplify_singularities=None) -> int:
    M = _as_matrix_dgcv(mat)
    if M is None:
        M = matrix_dgcv(mat)
    return M.rank(
        allow_formal_inverse=surface_singularities,
        simplify_steps=False
        if not surface_singularities
        else simplify_singularities
        if simplify_singularities is not None
        else True,
        record_divisors=surface_singularities,
    )


def trace_matrix(A):
    """
    Computes the trace of a square matrix A (sum of the diagonal elements).

    Parameters
    ----------
    A : list of lists
        The square matrix.

    Returns
    -------
    trace_value
        The trace of the matrix (sum of the diagonal elements).

    Raises
    ------
    ValueError
        If the matrix is not square.
    """
    rows_A, cols_A = len(A), len(A[0])
    if rows_A != cols_A:
        raise ValueError(
            "Trace can only be computed for square matrices. The matrix is {}x{}.".format(
                rows_A, cols_A
            )
        ) from None
    trace_value = sum(A[i][i] for i in range(rows_A))
    return trace_value


def _structure_array(data, dim):
    return array_dgcv(
        data,
        shape=(dim, dim),
        null_return=freeze_matrix(matrix_dgcv.zeros(dim, 1)),
    )


def _flatten_structure_data(structure_data, _source="algebra_class"):
    sdd = dict()
    unspool = structure_data._unspool
    for idx, val in structure_data._data.items():
        val_data = getattr(val, "_data", None)
        if val_data is None:
            raise TypeError(
                f"The `{_source}` initializer received data in an unsupported format."
            )
        idx1, idx2 = unspool(idx)
        for idx3, v in val_data.items():
            sdd[(idx1, idx2, idx3)] = v
    return sdd


def _harvest_structure_singularities(structure_data, parameters):
    struct_sing = set()
    for slot in structure_data._data.values():
        for v in slot._data.values():
            _, d = as_numer_denom(v)
            if get_free_symbols(d):
                struct_sing.add(d)
    if get_dgcv_settings_registry().get("simplify_singularity_ideals_by_default", True):
        return expr_union_primitives(
            struct_sing,
            order_coordinates(parameters),
            process_rationals=True,
            fail_quietly=True,
        )
    return list(struct_sing)


def _ordered_union(first, second):
    out = list(first)
    seen = set(out)
    for item in second:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _fresh_solve_variables(count):
    pref = "v" + uuid.uuid4().hex[:8]
    return [symbol(f"{pref}{j}") for j in range(count)]


def _solve_weight_kwargs(
    heavy, surface_singularities, simplify_singularities, method="linsolve"
):
    kwargs = {"method": method, "simplify_result": False}
    if surface_singularities:
        kwargs["return_divisors"] = True
        kwargs["pass_to_symbolic_engine"] = False
        if heavy:
            kwargs["simplify_pivots"] = True
        else:
            kwargs["simplify_pivots"] = (
                simplify_singularities if simplify_singularities is not None else True
            )
    elif heavy:
        kwargs["simplify_pivots"] = True
    return kwargs


def _indep_check(
    elems,
    newE,
    return_decomp_coeffs=False,
    print_solve_stats=False,
    method="linsolve",
    _solve_variables=None,
    surface_singularities=False,
    simplify_singularities=None,
    _force_eqn_simiplify=False,
    force_heavy_solve=False,
):
    if not isinstance(elems, (list, tuple)) or len(elems) == 0:
        if return_decomp_coeffs:
            return (True, {}, []) if surface_singularities else (True, {})
        return (True, []) if surface_singularities else True
    if _scalar_is_zero(newE):
        if return_decomp_coeffs:
            return (False, [{}], []) if surface_singularities else (False, [{}])
        return (False, []) if surface_singularities else False
    count = len(elems)
    if _solve_variables is None or len(_solve_variables) < count:
        variables = _fresh_solve_variables(count)
    else:
        variables = _solve_variables[:count]
    eqn = zip_sum(variables, elems) - newE
    if _force_eqn_simiplify or force_heavy_solve:
        eqn = simplify(eqn)

    solve_kwargs = _solve_weight_kwargs(
        force_heavy_solve,
        surface_singularities,
        simplify_singularities,
        method=method,
    )
    if surface_singularities:
        sol, sing = solve_dgcv(
            eqn,
            variables,
            print_solve_stats=print_solve_stats,
            **solve_kwargs,
        )
    else:
        sol = solve_dgcv(
            eqn,
            variables,
            print_solve_stats=print_solve_stats,
            **solve_kwargs,
        )
    if len(sol) == 0:
        if return_decomp_coeffs:
            return (True, [], sing) if surface_singularities else (True, [])
        return (True, sing) if surface_singularities else True
    if surface_singularities:
        sing = [subs(v, sol[0]) for v in sing]
    if return_decomp_coeffs:
        s = sol[0]
        coeffs = {idx: s.get(var, 0) for idx, var in enumerate(variables)}
        var_set = set(variables)
        free_vars = set()
        for c in coeffs.values():
            free_vars |= get_free_symbols(c)
        free_vars &= var_set
        if len(free_vars) == 0:
            coeffs = [coeffs]
        else:
            zeroing = {u: 0 for u in free_vars}
            expanded = []
            for v in sorted(free_vars, key=str):
                rule = {**zeroing, v: 1}
                expanded.append({idx: c.subs(rule) for idx, c in coeffs.items()})
            coeffs = expanded
        return (False, coeffs, sing) if surface_singularities else (False, coeffs)
    return (False, sing) if surface_singularities else False


def _elem_scale(elem, surface_singularities=False):
    coeffs = getattr(elem, "coeffs", None)
    if isinstance(coeffs, (list, tuple)):
        for c in coeffs:
            if not _scalar_is_zero(c):
                try:
                    out = elem / c
                    if surface_singularities:
                        if get_free_symbols(c):
                            return out, [c]
                        else:
                            return out, []
                    return out
                except Exception:
                    return elem
    return elem


def _basis_builder(
    elems,
    newE,
    ALBS=False,
    print_solve_stats=False,
    method="linsolve",
    _solve_variables=None,
    surface_singularities=False,
    simplify_singularities=None,
    force_heavy_solve=False,
):
    if _scalar_is_zero(newE):
        return (list(elems), []) if surface_singularities else list(elems)
    if ALBS is True:
        newE = _elem_scale(newE, surface_singularities=surface_singularities)
        if surface_singularities:
            newE, sing = newE
    elif surface_singularities:
        sing = []
    if not isinstance(elems, (list, tuple)):
        raise TypeError(
            f"_basis_builder expects `elems` to be a list, recieved {elems} of type {type(elems)}"
        )
    if len(elems) == 0:
        out = ([newE], sing) if surface_singularities else [newE]
        return out
    check = _indep_check(
        elems,
        newE,
        print_solve_stats=print_solve_stats,
        method=method,
        return_decomp_coeffs=False,
        _solve_variables=_solve_variables,
        surface_singularities=surface_singularities,
        simplify_singularities=simplify_singularities,
        force_heavy_solve=force_heavy_solve,
    )
    if surface_singularities:
        check, sing2 = check
    if check is True:
        return (
            (list(elems) + [newE], _ordered_union(sing, sing2))
            if surface_singularities
            else list(elems) + [newE]
        )
    else:
        return (
            (list(elems), _ordered_union(sing, sing2))
            if surface_singularities
            else list(elems)
        )


def _extract_basis(
    element_list,
    ALBS=False,
    print_solve_stats=False,
    method="linsolve",
    _solve_variables=None,
    return_indices=False,
    surface_singularities=False,
    simplify_singularities=None,
    force_heavy_solve=False,
):
    if not isinstance(element_list, (list, tuple)):
        element_list = list(element_list)
    basis = []
    idxs = [] if return_indices else None
    sing = []
    if _solve_variables is None and len(element_list) > 0:
        _solve_variables = _fresh_solve_variables(len(element_list))
    for i, newE in enumerate(element_list):
        old_len = len(basis)
        basis = _basis_builder(
            basis,
            newE,
            ALBS=ALBS,
            print_solve_stats=print_solve_stats,
            method=method,
            _solve_variables=_solve_variables,
            surface_singularities=surface_singularities,
            simplify_singularities=simplify_singularities,
            force_heavy_solve=force_heavy_solve,
        )

        if surface_singularities:
            basis, new_sing = basis
            sing = _ordered_union(sing, new_sing)

        if return_indices and len(basis) == old_len + 1:
            idxs.append(i)
    if surface_singularities:
        out = (basis, idxs, sing) if return_indices else (basis, sing)
    else:
        out = (basis, idxs) if return_indices else basis
    return out


def _generate_gl_structure_data(vs):
    n = len(vs.basis) - 1
    matrix_dim = n + 1

    # Basis elements
    hBasis = {"elems": dict(), "grading": dict()}
    offDiag = {"elems": dict(), "grading": dict()}

    def elemWeights(idx1, idx2):
        wVec = []
        for idx in range(n):
            if idx1 <= idx:
                if idx2 <= idx:
                    wVec.append(0)
                else:
                    wVec.append(1)
            else:
                if idx2 <= idx:
                    wVec.append(-1)
                else:
                    wVec.append(0)
        return wVec

    for j in range(n + 1):
        for k in range(j, n + 1):
            if j == k and j < n:
                M = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                for idx in range(n + 1):
                    if idx > j:
                        M[idx, idx] = -rational(j + 1, n + 1)
                    else:
                        M[idx, idx] = 1 - rational(j + 1, n + 1)
                hBasis["elems"][(j, k, 0)] = M
                hBasis["grading"][(j, k, 0)] = [0] * n
            elif j == n and k == n:
                M = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                for idx in range(n + 1):
                    M[idx, idx] = 1
                hBasis["elems"][(j, k, 0)] = M
                hBasis["grading"][(j, k, 0)] = [0] * n
            elif j != k:
                MPlus = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                MMinus = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                MPlus[j, k] = 1
                MMinus[k, j] = 1
                offDiag["elems"][(j, k, 1)] = MPlus
                offDiag["grading"][(j, k, 1)] = elemWeights(j, k)
                offDiag["elems"][(k, j, 1)] = MMinus
                offDiag["grading"][(k, j, 1)] = elemWeights(k, j)

    indexingKey = dict(
        enumerate(list(hBasis["grading"].keys()) + list(offDiag["grading"].keys()))
    )
    indexingKeyRev = {j: k for k, j in indexingKey.items()}
    LADimension = len(indexingKey)

    def _structureCoeffs(idx1, idx2):
        coeffs = matrix_dgcv({}, shape=(LADimension, 1))
        if idx2 == idx1:
            return coeffs
        if idx2 < idx1:
            reSign = -1
            idx2, idx1 = idx1, idx2
        else:
            reSign = 1
        p10, p11, p12 = indexingKey[idx1]
        p20, p21, p22 = indexingKey[idx2]
        if p12 == 0:
            if p22 == 1 and (p10 != n or p11 != n):
                val = reSign * (
                    int(p10 == p20)
                    - int(p10 == p21)
                    + int(p10 + 1 == p21)
                    - int(p10 + 1 == p20)
                )
                if val != 0:
                    coeffs[idx2] += val
        elif p12 == 1:
            if p22 == 1:
                if p11 == p20:
                    if p10 == p21:
                        if p10 < p11:
                            for idx in range(p10, p11):
                                coeffs[indexingKeyRev[(idx, idx, 0)]] = reSign
                        else:
                            for idx in range(p11, p10):
                                coeffs[indexingKeyRev[(idx, idx, 0)]] = -reSign
                    else:
                        coeffs[indexingKeyRev[(p10, p21, 1)]] = reSign
                elif p10 == p21:
                    coeffs[indexingKeyRev[(p20, p11, 1)]] = -reSign
        return coeffs

    _structure_data = array_dgcv(
        dict(),
        shape=(LADimension, LADimension),
        null_return=freeze_matrix(matrix_dgcv.zeros(LADimension, 1)),
    )
    for j in range(LADimension):
        for k in range(j + 1, LADimension):
            scoeffs = _structureCoeffs(j, k)
            if len(scoeffs._data) > 0:
                _structure_data[(j, k)] = scoeffs

    CartanSubalg = list(hBasis["elems"].values())
    matrixBasis = CartanSubalg + list(offDiag["elems"].values())

    def obGen(j, k):
        if j == k:
            if j < n:
                tp = (1 - rational(j + 1, n + 1)) * vs.basis[0] @ (vs.basis[0].dual())
                for idx in range(1, n + 1):
                    if idx > j:
                        tp += (
                            -rational(j + 1, n + 1)
                            * vs.basis[idx]
                            @ (vs.basis[idx].dual())
                        )
                    else:
                        tp += (
                            (1 - rational(j + 1, n + 1))
                            * vs.basis[idx]
                            @ (vs.basis[idx].dual())
                        )
                return tp
            return sum(
                [vs.basis[j] @ (vs.basis[j].dual()) for j in range(n)],
                vs.basis[n] @ (vs.basis[n].dual()),
            )
        else:
            return vs.basis[j] @ (vs.basis[k].dual())

    operatorBasis = [
        obGen(indexingKey[idx][0], indexingKey[idx][1]) for idx in range(LADimension)
    ]
    gradingVecs = list(hBasis["grading"].values()) + list(offDiag["grading"].values())
    return (
        _structure_data,
        list(zip(*gradingVecs)),
        CartanSubalg,
        matrixBasis,
        operatorBasis,
    )


class vector_space_endomorphisms(algebra_class):
    def __init__(self, vector_space):
        self.domain = vector_space
        self._dgcv_categories = {"endomorphism_space"}
        structure_data, grading, _, matrix_representation, tensor_representation = (
            _generate_gl_structure_data(vector_space)
        )
        super().__init__(
            structure_data,
            grading=grading,
            format_sparse=False,
            process_matrix_rep=False,
            preferred_representation=None,
            simplify_products_by_default=None,
            assume_skew=False,
            matrix_representation=matrix_representation,
            tensor_representation=tensor_representation,
            _basis_labels_parent=None,
            _label=None,
            _basis_labels=None,
            _calledFromCreator=retrieve_passkey(),
            _callLock=retrieve_passkey(),
            _print_warning=None,
            _child_print_warning=None,
            _exclude_from_VMF=retrieve_passkey(),
            _markers={
                "endo": True,
                "endo_label": getattr(vector_space, "label", "algebra_instance"),
                "endo_tex": vector_space._repr_latex_(raw=True, abbrev=True),
            },
        )


class linear_representation(dgcv_class):
    def __init__(self, hom: homomorphism):
        self.structureData, self.antihomomorphism, params = self._validate_hom(hom)
        self.homomorphism = hom
        self.domain = hom.domain
        self._parameters = params | (
            set(hom._parameters) if getattr(hom, "_parameters") else set()
        )
        self.representation_space = hom.codomain.domain

    @classmethod
    def _validate_hom(cls, hom):
        params = set()
        assert query_dgcv_categories(
            hom.codomain, {"endomorphism_space", "tensor_proxy"}
        )
        skew = getattr(hom.domain, "is_skew_symmetric", False)
        amb_dim = hom.domain.dimension + hom.codomain.domain.dimension
        dom_dim = hom.domain.dimension
        anti = None
        is_zero_map = getattr(hom, "_zero_map", False)

        def _equal(a, b):
            return getattr(a - b, "is_zero", False) or a == b

        for c, e1 in enumerate(hom.domain.basis):
            lidx = c + 1 if skew else 0
            for e2 in hom.domain.basis[lidx:]:
                if is_zero_map:
                    anti = False
                    break
                else:
                    p1 = hom(e1 * e2)
                    p2 = hom(e1) * hom(e2)
                    if anti is None and _equal(p1, p2) and not _equal(p1, 0 * p1):
                        anti = False
                    if not _equal(p1, p2):
                        if anti is None and _equal(p1, -p2):
                            anti = True
                        elif anti is True and _equal(p1, -p2):
                            pass
                        else:
                            raise ValueError(
                                f"The `hom` parameter given to the `linear_representation` initializer does not define an algebra homomorphism. The identity hom(v*w)=hom(v)*hom(w) fails for basis elements {e1} and {e2}, producing hom(v*w)={p1} and hom(v)*hom(w)={p2}"
                            )
                    if anti is None:
                        anti = False

        out_sd = matrix_dgcv(
            dict(),
            shape=(amb_dim, amb_dim),
            null_return=matrix_dgcv({}, shape=(amb_dim, 1)),
        )
        for k, v in getattr(hom.domain, "structureDataDict", dict()).items():
            new_key = (k[0], k[1])
            if new_key in out_sd:
                out_sd[new_key][k[2]] = v
            else:
                out_sd[new_key] = matrix_dgcv({k[2]: v}, shape=(amb_dim, 1))
        for k, v in getattr(hom.codomain.domain, "structureDataDict", dict()).items():
            new_key = (k[0] + dom_dim, k[1] + dom_dim)
            if new_key in out_sd:
                out_sd[new_key][k[2] + dom_dim] = v
            else:
                out_sd[new_key] = matrix_dgcv({k[2] + dom_dim: v}, shape=(amb_dim, 1))
        if not is_zero_map:
            for j in range(dom_dim):
                for k in range(amb_dim - dom_dim):
                    image = hom(hom.domain.basis[j])(hom.codomain.domain.basis[k])
                    if _scalar_is_zero(image):
                        continue
                    for idx, value in image.coeff_dict.items():
                        new_key = (j, k)
                        if new_key in out_sd:
                            out_sd[new_key][idx] = value
                            out_sd[(k, j)][idx] = -value
                        else:
                            out_sd[new_key] = matrix_dgcv(
                                {idx: value}, shape=(amb_dim, 1)
                            )
                            out_sd[(k, j)] = matrix_dgcv(
                                {idx: -value}, shape=(amb_dim, 1)
                            )

        return out_sd, anti, params

    def semidirect_sum(
        self,
        grading=None,
        label=None,
        basis_labels=None,
        register_in_vmf=False,
        initial_basis_index=None,
        simplify_products_by_default=None,
        _markers=None,
    ):
        if simplify_products_by_default is None:
            simplify_products_by_default = getattr(
                self.domain, "simplify_products_by_default", False
            )
        if grading is None:
            g1 = tuple(next(iter(self.domain.grading)))
            g2 = tuple(next(iter(self.representation_space.grading)))
            grading = [g1 + g2]
        if isinstance(basis_labels, (tuple, list)):
            if (
                not all(isinstance(elem, str) for elem in basis_labels)
                or len(basis_labels)
                != self.domain.dimension + self.representation_space.dimension
            ):
                dgcv_warning(
                    f"`basis_labels` is in an unsupported format and was ignored. Recieved {basis_labels}, types: {[type(lab) for lab in basis_labels]}, target length {self.domain.dimension}+{self.representation_space.dimension}"
                )
                basis_labels = None

        def _pref(el):
            if el[0] == "_":
                return "_I" + el
            return "_" + el

        def _preftex(el):
            if el[:2] == r"\_":
                return "\\_|" + el
            return "\\_" + el

        if _markers is None:
            _markers = {"sum": True, "lockKey": retrieve_passkey()}
            if label is None:
                label = f"{self.domain.label}_semidir_{self.representation_space.label}"
                _markers["_tex_label"] = (
                    f"{self.domain._repr_latex_(raw=True, abbrev=True)}\\ltimes {self.representation_space._repr_latex_(raw=True, abbrev=True)}"
                )
            if basis_labels is None:
                basis_labels = [elem.__repr__() for elem in self.domain.basis] + [
                    elem.__repr__() for elem in self.representation_space.basis
                ]
                _markers["_tex_basis_labels"] = [
                    elem._repr_latex_(raw=True) for elem in self.domain.basis
                ] + [
                    elem._repr_latex_(raw=True)
                    for elem in self.representation_space.basis
                ]
        elif not isinstance(basis_labels, (tuple, list)):
            if not isinstance(basis_labels, str):
                basis_labels = [elem.__repr__() for elem in self.domain.basis] + [
                    elem.__repr__() for elem in self.representation_space.basis
                ]
            else:
                pref = basis_labels
                IIdx = (
                    initial_basis_index
                    if isinstance(initial_basis_index, numbers.Integral)
                    else 1
                )
                basis_labels = [
                    f"{pref}{i + IIdx}"
                    for i in range(
                        self.domain.dimension + self.representation_space.dimension
                    )
                ]
        if not isinstance(label, str) or label == "":
            label = "Alg_" + create_key()

        _markers["semidirect_decomposition"] = (
            self.domain,
            self.representation_space,
            self.homomorphism,
        )
        _markers["_parameters"] = self._parameters
        if register_in_vmf is True:
            from .algebras_secondary import createAlgebra

            return createAlgebra(
                self.structureData,
                label,
                basis_labels=basis_labels,
                grading=grading,
                return_created_object=True,
                simplify_products_by_default=simplify_products_by_default,
                _markers=_markers,
            )
        else:
            _markers["registered"] = False
            return algebra_class(
                self.structureData,
                grading=grading,
                simplify_products_by_default=simplify_products_by_default,
                _label=label,
                _basis_labels=basis_labels,
                _calledFromCreator=retrieve_passkey(),
                _markers=_markers,
            )

    def __call__(self, *args, **kwds):
        return self.homomorphism.__call__(*args, **kwds)


class algebra_structure_data(dgcv_class):
    """
    Lazy algebra structure data container.

    shape: None (default), "skew", or "symmetric".
      - "skew": call gen only for i<j, mirror with minus; diagonal forced 0.
      - "symmetric": call gen only for i<=j, mirror with plus.

    cache_max_pairs: LRU size for cached (i,j) results.
    """

    def __init__(self, dim, gen, *, shape=None, cache_max_pairs=20000):
        if dim <= 0:
            raise ValueError("dim must be positive")
        if shape not in (None, "skew", "symmetric"):
            raise ValueError("shape must be None, 'skew', or 'symmetric'")
        self.dim = dim
        self._gen = gen
        self._shape = shape

        @lru_cache(maxsize=cache_max_pairs)
        def _pair_cache(i, j):
            i0, j0, sign = self._normalize_pair(i, j)
            if sign == 0:
                return ()  # skew-diagonal short-circuit
            payload = self._normalize_output(self._gen(i0, j0))
            if sign == -1:
                return tuple((k, -v) for (k, v) in payload)
            return payload

        self._pair_cache = _pair_cache

        @lru_cache(maxsize=cache_max_pairs)
        def _vector_cache(i, j):
            if self._normalize_pair(i, j)[2] == 0:
                return tuple(0 for _ in range(self.dim))
            vec = [0] * self.dim
            for k, v in self._pair_cache(i, j):
                if 0 <= k < self.dim and not _scalar_is_zero(v):
                    vec[k] = v
            return tuple(vec)

        self._vector_cache = _vector_cache

    def pair(self, i, j):
        self._check_bounds(i, j)
        return dict(self._pair_cache(i, j))

    def coeff(self, i, j, k):
        self._check_bounds(i, j, k)
        return self._vector_cache(i, j)[k]

    def vector(self, i, j):
        self._check_bounds(i, j)
        return list(self._vector_cache(i, j))

    def clear_cache(self):
        self._pair_cache.cache_clear()
        self._vector_cache.cache_clear()

    def stats(self):
        return {
            "dim": self.dim,
            "shape": self._shape,
            "pairs_cached": self._pair_cache.cache_info().currsize,
            "pair_cache_info": self._pair_cache.cache_info(),
            "vectors_cached": self._vector_cache.cache_info().currsize,
            "vector_cache_info": self._vector_cache.cache_info(),
        }

    def _check_bounds(self, i, j, k=None):
        if not (0 <= i < self.dim and 0 <= j < self.dim):
            raise IndexError(f"(i,j)=({i},{j}) out of bounds for dim={self.dim}")
        if k is not None and not (0 <= k < self.dim):
            raise IndexError(f"k={k} out of bounds for dim={self.dim}")

    def _normalize_output(self, raw):
        if isinstance(raw, dict):
            return tuple((k, v) for k, v in raw.items() if not _scalar_is_zero(v))
        if isinstance(raw, list):
            if len(raw) != self.dim:
                raise ValueError("Dense generator output length != dim")
            return tuple((k, v) for k, v in enumerate(raw) if not _scalar_is_zero(v))
        return tuple((k, v) for (k, v) in raw if not _scalar_is_zero(v))

    def _normalize_pair(self, i, j):
        if self._shape is None:
            return i, j, 1
        if self._shape == "skew":
            if i == j:
                return i, j, 0
            return (i, j, 1) if i < j else (j, i, -1)
        return (i, j, 1) if i <= j else (j, i, 1)

    # Back-compat: sd[i][j][k]
    def __getitem__(self, i):
        self._check_bounds(i, 0)
        return _RowView(self, i)


class _RowView:
    __slots__ = ("_sd", "_i")

    def __init__(self, sd, i):
        self._sd, self._i = sd, i

    def __getitem__(self, j):
        self._sd._check_bounds(self._i, j)
        return _ColumnView(self._sd, self._i, j)


class _ColumnView:
    __slots__ = ("_sd", "_i", "_j")

    def __init__(self, sd, i, j):
        self._sd, self._i, self._j = sd, i, j

    def __getitem__(self, k):
        return self._sd.coeff(self._i, self._j, k)

    def __iter__(self):
        return iter(self._sd.vector(self._i, self._j))

    def __len__(self):
        return self._sd.dim

    def __repr__(self):
        return f"ColumnView({self._i},{self._j}): {self._sd.vector(self._i, self._j)}"


class algebra_rep_data(dgcv_class):
    def __init__(self, dim, matrix_gen, operator_gen, *, cache_max=50000):
        self.dim = dim
        self._matrix_gen = matrix_gen
        self._operator_gen = operator_gen

        @lru_cache(maxsize=cache_max)
        def _cache(mode, i):
            if not (0 <= i < self.dim):
                raise IndexError(f"index {i} out of bounds for dim={self.dim}")
            if mode == "matrix":
                return self._matrix_gen(i)
            if mode == "operator":
                return self._operator_gen(i)
            raise KeyError("mode must be 'matrix' or 'operator'")

        self._cache = _cache

    def __getitem__(self, mode):
        if mode not in ("matrix", "operator"):
            raise KeyError("use ['matrix'] or ['operator']")
        return _RepModeView(self, mode)

    def clear_cache(self):
        self._cache.cache_clear()

    def stats(self):
        return {
            "dim": self.dim,
            "items_cached": self._cache.cache_info().currsize,
            "cache_info": self._cache.cache_info(),
        }


class _RepModeView:
    __slots__ = ("_rep", "_mode")

    def __init__(self, rep, mode):
        self._rep = rep
        self._mode = mode

    def __getitem__(self, i):
        return self._rep._cache(self._mode, i)

    def __len__(self):
        return self._rep.dim

    def __iter__(self):
        for i in range(self._rep.dim):
            yield self[i]

    def __repr__(self):
        return f"algebra_rep_view(mode={self._mode}, dim={self._rep.dim})"


class _lazy_SD(Mapping):
    def __init__(self, structure_data):
        self._data = structure_data
        self._cache = {}

    def __getitem__(self, key):
        # key is expected to be (i, j, k)
        if key in self._cache:
            return self._cache[key]
        i, j, k = key
        val = self._data[i][j][k]
        self._cache[key] = val
        return val

    def __iter__(self):
        for i in range(len(self._data)):
            for j in range(len(self._data[i])):
                for k in range(len(self._data[i][j])):
                    yield (i, j, k)

    def __len__(self):
        return sum(len(middle) * len(middle[0]) for middle in self._data)

    def values(self):
        # Overrides Mapping.values() to lazily iterate through values
        for key in self:
            yield self[key]

    def items(self):
        for key in self:
            yield key, self[key]


def _mat_to_tensor(mat, domain, codomain):
    mat_m = _as_matrix_dgcv(mat)
    if mat_m is None:
        return mat

    if domain.dimension != mat_m.nrows or codomain.dimension != mat_m.ncols:
        raise TypeError(
            "`mat` should be a r-by-s matrix where domain and codomain have dimensions r and s."
        )

    tp = 0
    for j in range(domain.dimension):
        for k in range(codomain.dimension):
            tp += mat_m[j, k] * codomain.basis[k] @ domain.basis[j]
    return tp
