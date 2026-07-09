"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: special_fields

module: dgcv.special_fields.filtered_structures

---
Author (of this module): David Gamble Sykes
Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

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
from collections.abc import Iterable
from typing import Any, Literal, Sequence, Tuple

from .._aux._backends._display_engine import is_rich_displaying_available
from .._aux._backends._numeric_router import zeroish
from .._aux._backends._polynomials import expr_union_primitives
from .._aux._backends._symbolic_router import (
    clear_denominators,
    conjugate,
    get_free_symbols,
    simplify,
    subs,
)
from .._aux._backends._types_and_constants import expr_numeric_types, rational, symbol
from .._aux._utilities._config import (
    dgcv_warning,
    get_dgcv_settings_registry,
    latex_in_html,
)
from .._aux._utilities._misc import linear_combination
from .._aux._utilities._styles import get_style
from .._aux._vmf._safeguards import (
    create_key,
    get_dgcv_category,
    query_dgcv_categories,
    retrieve_passkey,
)
from .._aux._vmf.vmf import order_coordinates
from .._aux.printing._tables import build_plain_table
from .._aux.printing.printing._dgcv_display import LaTeX, show
from ..algebras.algebras_core import (
    _extract_basis,
    _indep_check,
    algebra_class,
    algebra_subspace_class,
)
from ..algebras.algebras_secondary import createAlgebra, subalgebra_class
from ..core.arrays.arrays import array_dgcv, freeze_matrix, matrix_dgcv
from ..core.base import dgcv_class
from ..core.conversions.conversions import allToReal, allToSym, symToHol
from ..core.dgcv_core.dgcv_core import tensor_field_class, wedge
from ..core.solvers.solvers import solve_dgcv
from ..core.vector_fields_and_differential_forms.vector_fields_and_differential_forms import (
    LieDerivative,
    _extract_basis_by_wedge_vectorized,
    annihilator,
    decompose,
)

__all__ = ["Tanaka_symbol", "distribution"]


# -----------------------------------------------------------------------------
# distributions
# -----------------------------------------------------------------------------
class Tanaka_symbol(dgcv_class):
    """
    dgcv class representing a symbol-like object for Tanaka prolongation.

    Parameters
    ----------
        GLA

    Methods
    -------
        prolong
    """

    def __init__(
        self,
        GLA,
        nonnegParts=[],
        assume_FGLA=False,
        subspace=None,
        distinguished_subspaces=None,
        prolongation_label_prefix: str = None,
        validate_aggresively=False,
        index_threshold=None,
        precompute_generators=False,
        _validated=None,
        _internal_parameters=set(),
        _internal_singularities=None,
    ):
        class dynamic_dict(dict):
            def __init__(self, dict_data, initial_index=None):
                super().__init__(dict_data)
                self.index_threshold = initial_index

            def __getitem__(self, key):
                if isinstance(key, numbers.Integral) and (
                    self.index_threshold is None or key >= self.index_threshold
                ):
                    return super().get(key, [])
                return super().get(key, [])

            def _set_index_thr(self, new_threshold):
                if not (
                    isinstance(new_threshold, expr_numeric_types())
                    or new_threshold is None
                ):
                    raise TypeError("index_threshold must be an integer or None.")
                self.index_threshold = new_threshold

            def copy(self):
                return dynamic_dict(self, initial_index=self.index_threshold)

        # validation
        def valence_check(tp):
            for j in tp.coeff_dict:
                valence = j[len(j) // 3 : 2 * len(j) // 3]
                if valence[0] != 1:
                    return False
                if not all(j == 0 for j in valence[1:]):
                    return False
            return True

        if _validated != retrieve_passkey():
            if get_dgcv_category(GLA) not in {
                "algebra",
                "algebra_subspace",
                "subalgebra",
            }:
                raise TypeError(
                    "`Tanaka_symbol` expects `GLA` (which represents a generalized graded Lie algebra) to be an `algebra`, `sualgebra`, or `algebra_subspace_class`, and the first element of `GLA.grading` must contain negative weights (-depth,...,-1)."
                )
            elif not hasattr(GLA, "grading") or len(GLA.grading) == 0:
                raise TypeError(
                    "`Tanaka_symbol` expects `GLA` to be a graded Lie algebra, but the supplied `GLA` has no grading assigned."
                )
            if len(nonnegParts) != 0:  ###!!! patch __init__ to allow this branch
                if (
                    isinstance(GLA.grading[0], (list, tuple))
                    and max(GLA.grading[0]) >= 0
                ):
                    raise TypeError(
                        "While `Tanaka_symbol` supports two syntax formats for encoding non-negative weighted components, they cannot be combined. Either `GLA.grading` should have only non-negative weights or no value for the optional `nonnegParts` parameter should be given."
                    )
            if isinstance(nonnegParts, dict):
                NNPList = list(nonnegParts.values())
            elif (
                isinstance(nonnegParts, (list, tuple))
                and any(not isinstance(entry, (list, tuple)) for entry in nonnegParts)
            ) or len(nonnegParts) == 0:
                NNPList = [nonnegParts]
            else:
                raise TypeError(
                    "`Tanaka_symbol` expects `nonnegParts` to be a list of `tensorProduct` instances built from the `algebra_class` given for `GLA` with `valence` of the form (1,0,...,0). Or it can be a dictionary whose keys are non-negative weights, and whose key-values are such lists."
                )
            for NNP in NNPList:
                if not all(
                    get_dgcv_category(j) == "tensorProduct"
                    and j.vector_space == GLA
                    and valence_check(j)
                    for j in NNP
                ):
                    raise TypeError(
                        "`Tanaka_symbol` expects `nonnegParts` to be a list of `tensorProduct` instances built from the `algebra` given for `GLA` with `valence` of the form (1,0,...,0). Or it can be a dictionary whose keys are non-negative weights, and whose key-values are such lists."
                    )
        else:
            if isinstance(nonnegParts, dict):
                NNPList = list(nonnegParts.values())
            elif isinstance(nonnegParts, (list, tuple)):
                NNPList = [nonnegParts]
        if isinstance(GLA.grading[0], (list, tuple)):
            # if _validated != retrieve_passkey() and not any(j==-1 for j in GLA.grading[0]):
            #     raise TypeError(
            #         f"`Tanaka_symbol` expects `GLA` to be a Z-graded algebra (`algebra_class`, `algebra_subspace_class`, or `sualgebra_class` in particular) with the weight -1 among its weights in the first element of `GLA.grading`. Recieved grading data: {GLA.grading[0]}"
            #     )
            primary_grading = GLA.grading[0]
        else:
            if _validated != retrieve_passkey() and not any(
                j == -1 for j in GLA.grading
            ):  ###!!! review
                raise TypeError(
                    f"`Tanaka_symbol` expects `GLA` to be a Z-graded algebra (`algebra_class`, `algebra_subspace_class`, or `sualgebra_class` in particular) with the weight -1 among its weights in the first element of `GLA.grading`. Recieved grading data: {GLA.grading}"
                )
            primary_grading = GLA.grading
        non_neg_GLA = True if max(primary_grading) >= 0 else False

        raiseWarning = False
        if subspace is None:
            (
                subIndices,
                si_count,
                filtered_grading,
                truncateIndices,
                nonnegPartsTemp,
                index_map,
            ) = [], 0, [], dict(), dict(), dict()
            for count, weight in enumerate(primary_grading):
                if weight < 0:
                    truncateIndices[count] = si_count
                    subIndices.append(count)
                    index_map[count] = si_count
                    filtered_grading.append(weight)
                    si_count += 1
                else:
                    nonnegPartsTemp[weight] = nonnegPartsTemp.get(weight, []) + [
                        GLA.basis[count]
                    ]
            if si_count == 0:
                raise ValueError(
                    "`Tanaka_symbol` objects cannot be initialized from GLA data that has no negative weight components."
                )
            if len(nonnegPartsTemp) > 0:

                def truncateBySubInd(li):
                    return [li[j] for j in subIndices]

                def restrict_structure_data(data):
                    new_data = dict()
                    inner_shape = (si_count, 1)
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
                            elif v is not None and v != 0:
                                raise TypeError(
                                    "The GLA data given to the `Tanaka_symbol` initializer appears to not be compatible with its grading."
                                )
                    return array_dgcv(
                        new_data,
                        shape=(si_count, si_count),
                        null_return=freeze_matrix(matrix_dgcv.zeros(si_count, 1)),
                    )

                ###!!! generalize for vector space GLA
                subspace = subalgebra_class(
                    truncateBySubInd(GLA.basis),
                    GLA,
                    grading=[filtered_grading],
                    _compressed_structure_data=restrict_structure_data(
                        GLA.structureDataDict
                    ),
                    _internal_lock=retrieve_passkey(),
                )
            else:
                subspace = GLA
        else:
            if not isinstance(subspace, Iterable):
                raise TypeError(
                    "`Tanaka_symbol` expects `subpsace` if given to be a list of algebra_element_class instances belonging to the algebra_class `GLA` or tensor products of such elements, or a similar subspace-like object."
                )
            typeCheck = {
                "subalgebra_element",
                "algebra_element",
                "vector_space_element",
            }
            negative_basis, filtered_grading, nonnegPartsTemp = [], [], {}
            for elem in subspace:
                dgcvType = get_dgcv_category(elem)
                if dgcvType in typeCheck and elem.vectorSpace == GLA:
                    w = elem.check_element_weight(
                        test_weights=[primary_grading], flatten_weights=True
                    )
                    if w == "NoW":
                        raise TypeError(
                            "`Tanaka_symbol` expects the spanning set of elements given to define `subpsace` to be weighted homogeneous w.r.t. the primary grading."
                        )
                    elif isinstance(w, numbers.Integral):
                        if w < 0:
                            negative_basis.append(elem)
                            filtered_grading.append(w)
                        else:
                            nonnegPartsTemp[w] = nonnegPartsTemp.get(w, []) + [elem]
                elif (
                    dgcvType == "tensorProduct"
                    and elem.vectorSpace == GLA
                    and valence_check(elem) is True
                ):
                    if raiseWarning is False and non_neg_GLA is True:
                        raiseWarning = True
                    w = elem.compute_weight(
                        test_weights=[primary_grading], flatten_weights=True
                    )
                    if w == "NoW":
                        raise TypeError(
                            "`Tanaka_symbol` expects the spanning set of elements given to define `subpsace` to be weighted homogeneous w.r.t. the primary grading."
                        )
                    elif (
                        elem.max_degree > 1 and isinstance(w, numbers.Number) and w < 0
                    ):
                        raise TypeError(
                            "negatively-graded elements among those given to define `subpsace` should be bare algebra_element_class/vecor_space_element instances, rather than tensor products of such."
                        )
                    elif isinstance(w, numbers.Integral):
                        if w < 0:
                            negative_basis.append(elem)
                            filtered_grading.append(w)
                        else:
                            nonnegPartsTemp[w] = nonnegPartsTemp.get(w, []) + [elem]
                    try:
                        subspace = subalgebra_class(
                            negative_basis, GLA, grading=[filtered_grading]
                        )
                    except ValueError:
                        raise TypeError(
                            "`Tanaka_symbol` expects `subpsace` if given to be have the subspace within its given `GLA` spanned by its negatively graded elements be closed under Lie brackets."
                        )
                    if raiseWarning is True:
                        dgcv_warning(
                            "The graded algebra `GLA` given to `Tanaka_symbol` has non-negative components, but the supplied `subspace` had some non-negative degree elements formatted has tensor products rather than elements of the provided `GLA`. This mixing of formatting results in slower prolongation algorithm, so it is recommended to instead either supply `subset` as formal elements in the `GLA` or give `GLA`  as just its negative component and then additionally supply non-negative components as tensor products via the optional `nonnegParts` parameter."
                        )
                        subspace = algebra_subspace_class(
                            negative_basis,
                            parent_algebra=GLA,
                            _grading=[filtered_grading],
                            _internal_lock=retrieve_passkey(),
                        )
                else:
                    raise TypeError(
                        "`Tanaka_symbol` expects `subpsace` if given to be a list of algebra_element_class instances belonging to the algebra_class `GLA` or tensor products of such elements, or a similar subspace-like object."
                    )

        if len(nonnegPartsTemp) > 0:
            if len(nonnegParts) > 0:
                dgcv_warning(
                    "The `GLA` or `subspace` parameter provided to `Tanaka_symbol` has nonnegatively weighted components. If providing such `GLA` or `subspace` data then the optional `nonnegParts` cannot be manually set. So the provided manual setting for `nonnegParts` is being ignored."
                )
            nonnegParts = nonnegPartsTemp
            for w in nonnegParts.keys():
                nonnegParts[w] = [
                    _GAE_to_hom_formatting(j, subspace, test_weights=[primary_grading])
                    for j in nonnegParts[w]
                ]

        if distinguished_subspaces and _validated != retrieve_passkey():
            ds_params = set()
            total_basis = list(subspace.basis) + sum(
                NNPList, []
            )  # NNPList formatted as tensors
            newDS = []
            _fast_process_DS = []
            _standard_process_DS = []
            _slow_process_DS = []
            if not isinstance(distinguished_subspaces, (list, tuple)):
                raise TypeError(
                    "`Tanaka_symbol` expects `distinguished_subspaces` to be a list of lists of algebra_element_class instances or tensor products belonging to the provided basis of the symbol."
                )
            else:
                for subS in distinguished_subspaces:
                    process = "fast"  # can be 'fast', 'standard', or 'slow'
                    subSLevels = dict()
                    idx_cap = None
                    if not (
                        isinstance(subS, (list, tuple))
                        or get_dgcv_category(subS) == "algebra_subspace"
                    ):
                        raise TypeError(
                            "`Tanaka_symbol` expects `distinguished_subspaces` to be a list of lists of algebra_element_class instances or tensor products built from the provided basis of the symbol, or some subspace class."
                        )
                    DSList = []
                    for elem in subS:
                        ds_params |= get_free_symbols(elem)
                        reformElem, weights = _GAE_to_hom_formatting(
                            elem,
                            subspace,
                            test_weights=[primary_grading],
                            return_weights=True,
                        )  # weights may have len>1 indicated different weighted components
                        DSList.append(reformElem)
                        if process == "fast" and reformElem in total_basis:
                            newE = (
                                reformElem
                                if weights[0] < 0
                                else _fast_tensor_products(reformElem)
                            )
                            subSLevels[weights[0]] = subSLevels.get(weights[0], []) + [
                                newE
                            ]
                            if idx_cap is None or idx_cap < weights[0]:
                                idx_cap = weights[0]
                        elif process != "slow" and len(weights) == 1:
                            process = "standard"
                            newE = (
                                reformElem
                                if weights[0] < 0
                                else _fast_tensor_products(reformElem)
                            )
                            subSLevels[weights[0]] = subSLevels.get(weights[0], []) + [
                                newE
                            ]
                            if idx_cap is None or idx_cap < weights[0]:
                                idx_cap = weights[0]
                        else:
                            process == "slow"
                            if validate_aggresively is True:
                                ###!!! check for reformElem in span of total basis
                                pass
                    newDS.append(DSList)
                    if process == "fast":
                        _fast_process_DS.append(
                            dynamic_dict(subSLevels, initial_index=idx_cap + 1)
                        )
                    elif process == "standard":
                        rich_dict = dynamic_dict(dict(), initial_index=idx_cap + 1)
                        for k, v in subSLevels.items():
                            rich_dict[k] = dict()
                            genelement, newv = linear_combination(v)
                            rich_dict[k]["vars"] = newv
                            rich_dict[k]["element"] = genelement
                            rich_dict[k]["spanners"] = v
                        _standard_process_DS.append(rich_dict)
                    else:
                        genelement, newv = linear_combination(DSList)
                        rich_dict = {
                            "vars": newv,
                            "element": genelement,
                            "spanners": DSList,
                        }
                        _slow_process_DS.append(rich_dict)
            self._fast_process_DS = _fast_process_DS
            self._standard_process_DS = _standard_process_DS
            self._slow_process_DS = _slow_process_DS
        else:
            ds_params = set()
            newDS = [[]]
            self._fast_process_DS = []
            self._standard_process_DS = []
            self._slow_process_DS = []
        self._parameters = GLA._parameters | ds_params | _internal_parameters

        distinguished_subspaces = newDS
        maxDSW = -1
        for subS in self._fast_process_DS:
            maxDSW = max(maxDSW, max(subS.keys()))
        for subSData in self._standard_process_DS:
            maxDSW = max(maxDSW, max(subSData.keys()))
        if (
            len(nonnegParts) > 0 and distinguished_subspaces is not None
        ) or maxDSW >= 0:
            self._default_to_characteristic_space_reductions = True
        else:
            self._default_to_characteristic_space_reductions = False

        self.negativePart = subspace
        self.ambientGLA = GLA
        self.assume_FGLA = assume_FGLA
        self.nonnegParts = nonnegParts
        negWeights = sorted([j for j in set(primary_grading) if j < 0])
        # if negWeights[-1]!=-1:
        #     raise AttributeError('`Tanaka_symbol` expects negatively graded LA to have a weight -1 component.')
        self.negWeights = tuple(negWeights)
        if isinstance(nonnegParts, dict):
            nonNegWeights = sorted([k for k, v in nonnegParts.items() if len(v) != 0])
        else:
            nonNegWeights = sorted(
                tuple(
                    set(
                        [
                            j.compute_weight(test_weights=[primary_grading])[0]
                            for j in nonnegParts
                        ]
                    )
                )
            )
        if len(nonNegWeights) == 0:
            self.height = -1
        else:
            self.height = nonNegWeights[-1]
        self.depth = negWeights[0]
        self.weights = negWeights + nonNegWeights
        GLA_levels = dict()
        grad = (
            filtered_grading
            if get_dgcv_category(self.negativePart) == "subalgebra"
            else primary_grading
        )
        for elem in self.negativePart.basis:
            w = elem.check_element_weight(test_weights=[grad])[0]
            GLA_levels[w] = GLA_levels.get(w, []) + [elem]
        self.GLA_levels = GLA_levels
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "Tanaka_symbol"

        if isinstance(nonnegParts, dict):
            self.nonneg_levels = nonnegParts
        else:
            nonneg_levels = dict()
            for elem in nonnegParts:
                w = elem.compute_weight(test_weights=[primary_grading])[0]
                nonneg_levels[w] = nonneg_levels.get(w, []) + [elem]
            self.nonneg_levels = nonneg_levels
        levels = dict(sorted((self.GLA_levels | self.nonneg_levels).items()))

        self._GLA_structure = dynamic_dict
        self.levels = dynamic_dict(levels, initial_index=index_threshold)
        self.dimension = sum(len(level) for level in self.levels.values())
        self.distinguished_subspaces = distinguished_subspaces
        self._test_commutators = None
        self._GLA_generators = None
        self._plp = (
            prolongation_label_prefix
            if isinstance(prolongation_label_prefix, str)
            else "eta"
        )
        self._singularities = (
            _internal_singularities if _internal_singularities else dict()
        )
        if precompute_generators is True:
            _ = self.GLA_generators

    @property
    def test_commutators(self):
        if self._test_commutators is None:
            if self.assume_FGLA:
                deeper_levels = sum(
                    [self.GLA_levels[j] for j in self.negWeights[:-1]], []
                )
                f_level = self.GLA_levels[-1]
                first_commutators = [
                    (f_level[j], f_level[k], f_level[j] * f_level[k])
                    for j in range(len(f_level))
                    for k in range(j + 1, len(f_level))
                ]
                remaining_comm = [(j, k, j * k) for j in f_level for k in deeper_levels]
                self._test_commutators = first_commutators + remaining_comm
            elif self._GLA_generators is not None:
                first_commutators = sum(self._GLA_generators["triples"].values(), [])
                remaining_comm = [
                    (j, k, j * k)
                    for j in sum(self._GLA_generators["generators"].values(), [])
                    for k in self._GLA_generators["generated"]
                ]
                self._test_commutators = first_commutators + remaining_comm
            else:
                neg_levels = sum([list(j) for j in (self.GLA_levels).values()], [])
                self._test_commutators = [
                    (neg_levels[j], neg_levels[k], neg_levels[j] * neg_levels[k])
                    for j in range(len(neg_levels))
                    for k in range(j + 1, len(neg_levels))
                ]
        return self._test_commutators

    @property
    def GLA_generators(self):
        if self._GLA_generators is None:
            self._test_commutators = None
            self._GLA_generators = {"generators": {-1: self.levels[-1]}}
            self._GLA_generators["map"] = {-1: [(j, j, 1) for j in self.levels[-1]]}
            self._GLA_generators["triples"] = dict()
            nRange = range(-1, min(self.negWeights) - 1, -1)
            generated = []
            for w in nRange[1:]:
                w_level = []
                w_level_brackets = []
                w_level_triples = []
                brackets = self.ambientGLA.subspace()
                for idx1 in range(-1, w // 2 - 1, -1):
                    idx2 = w - idx1
                    for c1, eT1 in enumerate(self._GLA_generators["map"].get(idx1, [])):
                        newC = c1 + 1 if idx1 == idx2 else 0
                        for eT2 in self._GLA_generators["map"].get(idx2, [])[newC:]:
                            tuple1, e1, dep1 = eT1
                            tuple2, e2, dep2 = eT2
                            dep3 = max(dep1, dep2) + 1
                            d1 = brackets.dimension
                            eProd = e1 * e2
                            brackets.append(eProd)
                            if dep3 == 2:
                                w_level_triples.append((tuple1, tuple2, eProd))
                            if brackets.dimension - d1 > 0:
                                generated.append(eProd)
                                w_level_brackets.append(([tuple1, tuple2], eProd, dep3))
                for elem in self.levels[w]:
                    d1 = brackets.dimension
                    brackets.append(elem)
                    if brackets.dimension - d1 > 0:
                        w_level.append(elem)
                        w_level_brackets.append((elem, elem, 1))
                if len(w_level) > 0:
                    self._GLA_generators["generators"][w] = w_level
                self._GLA_generators["map"][w] = w_level_brackets
                if len(w_level_triples) > 0:
                    self._GLA_generators["triples"][w] = w_level_triples
            self._GLA_generators["generated"] = generated
            if (
                self.assume_FGLA is True
                and min(self._GLA_generators["generators"]) < -1
            ):
                self.assume_FGLA = False
                dgcv_warning(
                    "The parameter setting `assume_FGLA=True` has been overwritten because a diognostic has shown the symbol is not fundamental."
                )
        return self._GLA_generators

    @property
    def basis(self):
        return sum(list(self.levels.values()), [])

    def __iter__(self):
        return iter(self.basis)

    def _fast_prolong_by_1(
        self,
        levels,
        height,
        distinguished_s_weight_bound=-1,
        with_characteristic_space_reductions=False,
        ADS=None,
        surface_singularities=None,
        simplify_pivots=None,
    ):  # height must match levels structure
        distinguished_s_weight_bound = min(distinguished_s_weight_bound, height)
        if len(self._parameters) > 0:
            if surface_singularities is not False:
                surface_singularities = True
        else:
            surface_singularities = False
        if simplify_pivots is None:
            simplify_ideals = get_dgcv_settings_registry().get(
                "simplify_singularity_ideals_by_default", True
            )
            simplify_pivots = surface_singularities
        else:
            simplify_ideals = simplify_pivots
        if ADS is None:
            fast_DS = self._fast_process_DS
            standard_DS = self._standard_process_DS
            ADS = False
        else:
            fast_DS = ADS[0]
            standard_DS = ADS[1]
            ADS = True
        if self.assume_FGLA and len(levels[height]) == 0:  # stability check
            new_levels = levels
            new_levels._set_index_thr(height)
            stable = True
        elif (
            self._GLA_generators is not None
            and all(
                len(levels[height - j]) == 0
                for j in range(-min(self._GLA_generators.get("generators", levels)))
            )
            and min(self._GLA_generators.get("generators", levels)) >= -1 - height
        ):
            new_levels = levels
            new_levels._set_index_thr(height)
            stable = True
        elif min(j for j in levels) >= -1 - height and all(
            len(levels[height - j]) == 0 for j in range(-min(j for j in levels))
        ):  # stability check
            new_levels = levels
            new_levels._set_index_thr(height)
            stable = True
        else:

            def fast_validate_for_DS(tp, basisVec, w1, w2):
                for subS in fast_DS:
                    if subS[w2] is not None and basisVec in subS[w2]:
                        if _indep_check(subS[w1], tp):
                            return False
                return True

            def stamp(j, k, jidx, kidx, jdeg, kdeg):
                obj = _fast_tensor_products(k) @ j  # removed .dual() for fast algorithm
                obj._hom_id = [{(jidx, kidx, jdeg, kdeg): 1}, ""]
                return obj

            if self._GLA_generators is None:
                ambient_basis = []
                for weight in self.negWeights:
                    kdeg = height + 1 + weight
                    ambient_basis += [
                        stamp(j, k, jidx, kidx, weight, kdeg)
                        for jidx, j in enumerate(self.GLA_levels[weight])
                        for kidx, k in enumerate(levels[kdeg])
                        if height + 1 + weight > distinguished_s_weight_bound
                        or fast_validate_for_DS(k, j, kdeg, weight)
                    ]
            else:
                preBasis = []
                ambient_basis = []
                for weight, comp in self._GLA_generators["generators"].items():
                    kdeg = height + 1 + weight
                    preBasis += [
                        stamp(j, k, jidx, kidx, weight, kdeg)
                        for jidx, j in enumerate(comp)
                        for kidx, k in enumerate(levels[kdeg])
                        if kdeg > distinguished_s_weight_bound
                        or fast_validate_for_DS(k, j, kdeg, weight)
                    ]

                def _iter_expand(elem, nested):
                    if isinstance(nested, list):
                        return _iter_expand(
                            _iter_expand(elem, nested[0]), nested[1]
                        ) + _iter_expand(nested[0], _iter_expand(elem, nested[1]))
                    return elem * nested

                def _complete(elem):
                    new_terms = []
                    for w, comp in self._GLA_generators["map"].items():
                        if w == -1:
                            continue
                        for trip in comp:
                            if trip[2] > 1:
                                new_terms.append(
                                    _fast_tensor_products(_iter_expand(elem, trip[0]))
                                    @ trip[1]  # removed .dual() for fast algo
                                )
                    return sum(new_terms, elem)

                ambient_basis = [_complete(j) for j in preBasis]

            ###!!! add fast validation for nonnegative weight DS components
            for subSData in standard_DS:
                if len(ambient_basis) == 0:
                    break
                ambGE, ambVars = linear_combination(ambient_basis)
                for w, level in subSData.items():
                    current_degree = height + w + 1
                    if current_degree <= distinguished_s_weight_bound:
                        dsSpanners = subSData[current_degree]["spanners"]
                        eqns = []
                        esVars = ambVars.copy()
                        for elem in level["spanners"]:
                            newGE, newVars = linear_combination(dsSpanners)
                            esVars += newVars
                            eqns.append(ambGE * elem + newGE)
                        if surface_singularities:
                            sol, sing = solve_dgcv(
                                eqns,
                                esVars,
                                method="linsolve",
                                return_divisors=True,
                                pass_to_symbolic_engine=False,
                                simplify_pivots=simplify_pivots,
                            )
                            self._singularities["prolongation"] = expr_union_primitives(
                                list(self._singularities.get("prolongation", []))
                                + [v for v in sing if get_free_symbols(v)],
                                self._parameters,
                                process_rationals=True,
                                fail_quietly=True,
                                bypass=not simplify_ideals,
                            )

                        else:
                            sol = solve_dgcv(eqns, esVars, method="linsolve")
                        ambient_basis = []
                        if len(sol) > 0:
                            solGE = subs(ambGE, sol[0])
                            freeVars = set()
                            for c in solGE.coeffs:
                                freeVars |= get_free_symbols(c)
                            if self._parameters:
                                freeVars = set(
                                    filter(
                                        lambda x: x not in self._parameters, freeVars
                                    )
                                )
                            zeroing = {var: 0 for var in freeVars}
                            for v in freeVars:
                                ambient_basis.append(subs(solGE, {**zeroing, v: 1}))
            if len(self._slow_process_DS) > 0:
                dgcv_warning(
                    "At least one of the distinguished subspaces was given by a spanning set of elements containing some element that is not weighted-homogeneous. The algorithm for preserving subspaces in such a format is not yet implemented in this version of `dgcv`, so the subspace is being disregarding."
                )

            if len(ambient_basis) == 0:
                ambient_basis = [0 * self.basis[0]]

            general_elem, tVars = linear_combination(ambient_basis)

            eqns = []
            for triple in self.test_commutators:
                t0, t1, t2 = triple[0], triple[1], triple[2]
                derivation_rule = (
                    (general_elem * t0) * t1
                    + t0 * (general_elem * t1)
                    - general_elem * t2
                )
                if getattr(derivation_rule, "is_zero", False) or derivation_rule == 0:
                    continue
                if get_dgcv_category(derivation_rule) in {
                    "fastTensorProduct",
                    "tensorProduct",
                    "algebra_element",
                    "subalgebra_element",
                    "vector_space_element",
                }:
                    eqns += list(derivation_rule.coeff_dict.values())
                else:
                    dgcv_warning(
                        f"The derivation rule value {derivation_rule} is outside of expected class. Recieved type: {type(derivation_rule)}",
                        wc_label="debug_log",
                    )

            if eqns == [0] or eqns == []:
                solution = [{}]
            else:
                if surface_singularities:
                    solution, sing = solve_dgcv(
                        eqns,
                        tVars,
                        method="linsolve",
                        return_divisors=True,
                        pass_to_symbolic_engine=False,
                        simplify_pivots=simplify_pivots,
                    )

                    self._singularities["prolongation"] = expr_union_primitives(
                        list(self._singularities.get("prolongation", []))
                        + [v for v in sing if get_free_symbols(v)],
                        self._parameters,
                        process_rationals=True,
                        fail_quietly=True,
                        bypass=not simplify_ideals,
                    )

                else:
                    solution = solve_dgcv(eqns, tVars, method="linsolve")

            if len(solution) == 0:
                dgcv_warning(
                    f"At breakpoint in prolongation algorithm: The equation system was {eqns} w.r.t. {tVars}; return solution data was {solution}",
                    wc_label="debug_log",
                )
                raise RuntimeError(
                    "`Tanaka_symbol.prolongation` failed at a step where a symbolic solver (e.g., sympy.solve if using the default sympy) was being applied."
                )
            solution = solution[0]
            el_sol = subs(general_elem, solution)
            if not isinstance(el_sol, _fast_tensor_products):
                el_sol = _fast_tensor_products(el_sol)

            fv = set()
            for j in el_sol.coeff_dict.values():
                fv |= get_free_symbols(j)
            if self._parameters:
                fv = set(filter(lambda x: x not in self._parameters, fv))
            new_level = []
            zeroing = {v: 0 for v in fv}
            for v in fv:
                basis_element = subs(el_sol, {**zeroing, v: 1})
                coeffs = [subs(solution.get(v, 0), {**zeroing, v: 1}) for v in tVars]
                # _hom_id format: [(source_label,((x,y) for x,y in zip(coeffs,target_labels)), label]
                _hom_id = {}
                new_level.append(basis_element)

            if ADS is True:
                new_elems = []
                for subS in fast_DS:
                    if height + 1 in subS:
                        new_elems += [
                            v for v in subS[height + 1]["spanners"]
                        ]  ###!!! formerly deepcopy
                        subS[height + 1] = _extract_basis(
                            subS[height + 1]["spanners"] + new_level
                        )
                for subSData in standard_DS:
                    if height + 1 in subSData:
                        new_elems += [
                            v for v in subSData[height + 1]["spanners"]
                        ]  ###!!! formerly deepcopy
                        subSData[height + 1]["spanners"] = _extract_basis(
                            subSData[height + 1]["spanners"] + new_level
                        )
                new_level = _extract_basis(new_level + list(new_elems))

            if with_characteristic_space_reductions is True and height == -1:
                z_level = new_level
            else:
                z_level = levels[0]
            if (
                len(new_level) > 0
                and with_characteristic_space_reductions is True
                and len(z_level) > 0
            ):
                stabilized = False
                while stabilized is False:
                    ambient_basis = new_level
                    varLabel = create_key(prefix="_cv")
                    tVars = [
                        symbol(f"{varLabel}{j}") for j in range(len(ambient_basis))
                    ]
                    solVars = list(tVars)
                    general_elem = sum(
                        [tVars[j] * ambient_basis[j] for j in range(len(tVars))]
                    )
                    eqns = []
                    for idx, dzElem in enumerate(z_level):
                        varLabel2 = varLabel + f"{idx}_"
                        vars2 = [
                            symbol(f"{varLabel2}{j}") for j in range(len(ambient_basis))
                        ]
                        solVars += vars2
                        general_elem2 = sum(
                            [vars2[j] * ambient_basis[j] for j in range(len(tVars))]
                        )._convert_to_tp()

                        commutator = (
                            general_elem._convert_to_tp() * dzElem._convert_to_tp()
                            - general_elem2
                        )
                        if get_dgcv_category(commutator) == "tensorProduct":
                            eqns += list(commutator.coeff_dict.values())
                        elif get_dgcv_category(commutator) == "algebra_element_class":
                            eqns += commutator.coeffs
                    if surface_singularities:
                        solution, sing = solve_dgcv(
                            eqns,
                            solVars,
                            method="linsolve",
                            return_divisors=True,
                            pass_to_symbolic_engine=False,
                            simplify_pivots=simplify_pivots,
                        )
                        self._singularities["prolongation"] = expr_union_primitives(
                            list(self._singularities.get("prolongation", []))
                            + [v for v in sing if get_free_symbols(v)],
                            self._parameters,
                            process_rationals=True,
                            fail_quietly=True,
                            bypass=not simplify_ideals,
                        )
                    else:
                        solution = solve_dgcv(eqns, solVars, method="linsolve")
                    if len(solution) == 0:
                        raise RuntimeError(
                            f"`Tanaka_symbol.prolongation` failed at a step where a symbolic solver (e.g., sympy.solve if using the default sympy) was being applied. The equation system was {eqns} w.r.t. {solVars}"
                        )
                    solution = solution[0]
                    solCoeffs = [solution.get(j, 0) for j in tVars]

                    free_variables = tuple(
                        set.union(
                            set(),
                            *[
                                get_free_symbols(j) - self._parameters
                                for j in solCoeffs
                            ],
                        )
                    )
                    # new_vectors = []
                    filtered_vectors = []
                    zeroingDict = {other_var: 0 for other_var in free_variables}
                    for var in free_variables:
                        basis_element = [
                            subs(j, zeroingDict | {var: 1}) for j in solCoeffs
                        ]
                        filtered_vectors.append(clear_denominators(basis_element))
                        # new_vectors.append(basis_element)
                    # columns = (sp.Matrix(new_vectors).T).columnspace()
                    # filtered_vectors = [
                    #     list(sp.nsimplify(col, rational=True)) for col in columns
                    # ]

                    # def reScale(vec):
                    #     denom = 1
                    #     for t in vec:
                    #         if hasattr(t, "denominator") and denom < t.denominator:
                    #             denom = t.denominator
                    #     if denom == 1:
                    #         return list(vec)
                    #     else:
                    #         return [t * denom for t in vec]

                    # filtered_vectors = [reScale(vec) for vec in filtered_vectors]

                    new_basis = []
                    for coeffs in filtered_vectors:
                        new_basis.append(
                            sum(
                                [
                                    coeffs[j] * ambient_basis[j]
                                    for j in range(len(ambient_basis))
                                ]
                            )
                        )
                    if len(new_basis) == 0:
                        new_level = []
                        stabilized = True
                    elif len(new_basis) < len(new_level):
                        new_level = new_basis
                    else:
                        new_level = new_basis
                        stabilized = True
            new_levels = self._GLA_structure(
                levels | {height + 1: new_level}, levels.index_threshold
            )
            stable = False
        ssd = [fast_DS, standard_DS] if ADS is True else None
        return new_levels, stable, ssd

    def _prolong_by_1(
        self,
        levels,
        height,
        distinguished_s_weight_bound=-1,
        with_characteristic_space_reductions=False,
        ADS=None,
        surface_singularities=None,
    ):  # height must match levels structure
        if len(self._parameters) > 0 and surface_singularities is not False:
            surface_singularities = True

        if ADS is None:
            fast_DS = self._fast_process_DS
            standard_DS = self._standard_process_DS
            ADS = False
        else:
            fast_DS = ADS[0]
            standard_DS = ADS[1]
            ADS = True
        if self.assume_FGLA and len(levels[height]) == 0:  # stability check
            new_levels = levels
            new_levels._set_index_thr(height)
            stable = True
        elif (
            self._GLA_generators is not None
            and all(
                len(levels[height - j]) == 0
                for j in range(-min(self._GLA_generators.get("generators", levels)))
            )
            and min(self._GLA_generators.get("generators", levels)) >= -1 - height
        ):
            new_levels = levels
            new_levels._set_index_thr(height)
            stable = True
        elif min(j for j in levels) >= -1 - height and all(
            len(levels[height - j]) == 0 for j in range(-min(j for j in levels))
        ):  # stability check
            new_levels = levels
            new_levels._set_index_thr(height)
            stable = True
        else:

            def fast_validate_for_DS(tp, basisVec, w1, w2):
                for subS in fast_DS:
                    if subS[w2] is not None and basisVec in subS[w2]:
                        if _indep_check(subS[w1], tp):
                            return False
                return True

            if self._GLA_generators is None:
                ambient_basis = []
                for weight in self.negWeights:
                    ambient_basis += [
                        k @ (j.dual())
                        for j in self.GLA_levels[weight]
                        for k in levels[height + 1 + weight]
                        if height + 1 + weight > distinguished_s_weight_bound
                        or fast_validate_for_DS(k, j, height + 1 + weight, weight)
                    ]
            else:
                preBasis = []
                ambient_basis = []
                for weight, comp in self._GLA_generators["generators"].items():
                    preBasis += [
                        k @ (j.dual())
                        for j in comp
                        for k in levels[height + 1 + weight]
                        if height + 1 + weight > distinguished_s_weight_bound
                        or fast_validate_for_DS(k, j, height + 1 + weight, weight)
                    ]

                def _iter_expand(elem, nested):
                    if isinstance(nested, list):
                        return _iter_expand(
                            _iter_expand(elem, nested[0]), nested[1]
                        ) + _iter_expand(nested[0], _iter_expand(elem, nested[1]))
                    return elem * nested

                def _complete(elem):
                    new_terms = []
                    for w, comp in self._GLA_generators["map"].items():
                        if w == -1:
                            continue
                        for trip in comp:
                            if trip[2] > 1:
                                new_terms.append(
                                    _iter_expand(elem, trip[0]) @ (trip[1].dual())
                                )
                    return sum(new_terms, elem)

                ambient_basis = [_complete(j) for j in preBasis]

            ###!!! add fast validation for nonnegative weight DS components
            for subSData in standard_DS:
                if len(ambient_basis) == 0:
                    break
                ambGE, ambVars = linear_combination(ambient_basis)
                for w, level in subSData.items():
                    if height + w + 1 <= distinguished_s_weight_bound:
                        dsSpanners = subSData[height + w + 1]["spanners"]
                        eqns = []
                        esVars = ambVars.copy()
                        for elem in level["spanners"]:
                            newGE, newVars = linear_combination(dsSpanners)
                            esVars += newVars
                            eqns.append(ambGE * elem + newGE)
                        if surface_singularities:
                            sol, sing = solve_dgcv(
                                eqns,
                                esVars,
                                method="linsolve",
                                return_divisors=True,
                                pass_to_symbolic_engine=False,
                            )
                            self._singularities["prolongation"] = (
                                self._singularities.get("prolongation", set())
                                | {v for v in sing if get_free_symbols(v)}
                            )
                        else:
                            sol = solve_dgcv(
                                eqns,
                                esVars,
                                method="linsolve",
                            )
                        ambient_basis = []
                        if len(sol) > 0:
                            solGE = ambGE.subs(sol[0])
                            freeVars = set()
                            for c in solGE.coeffs:
                                freeVars |= get_free_symbols(c)
                            if self._parameters:
                                freeVars = set(
                                    filter(
                                        lambda x: x not in self._parameters, freeVars
                                    )
                                )
                            zeroing = {var: 0 for var in freeVars}
                            for var in freeVars:
                                ambient_basis.append(solGE.subs({var: 1}).subs(zeroing))
            if len(self._slow_process_DS) > 0:
                dgcv_warning(
                    "At least one of the distinguished subspaces was given by a spanning set of elements containing some element that is not weighted-homogeneous. The algorithm for preserving subspaces in such a format is not yet implemented in this version of `dgcv`, so the subspace is being disregarding."
                )

            if len(ambient_basis) == 0:
                ambient_basis = [0 * self.basis[0]]

            general_elem, tVars = linear_combination(ambient_basis)

            eqns = []
            for triple in self.test_commutators:
                t0, t1, t2 = triple[0], triple[1], triple[2]
                derivation_rule = (
                    (general_elem * t0) * t1
                    + t0 * (general_elem * t1)
                    - general_elem * t2
                )
                if getattr(derivation_rule, "is_zero", False) or derivation_rule == 0:
                    continue
                if get_dgcv_category(derivation_rule) in {
                    "fastTensorProduct",
                    "tensorProduct",
                    "algebra_element",
                    "subalgebra_element",
                    "vector_space_element",
                }:
                    eqns += list(derivation_rule.coeff_dict.values())
                else:
                    dgcv_warning(
                        f"The derivation rule value {derivation_rule} is outside of expected class. Recieved type: {type(derivation_rule)}",
                        wc_label="debug_log",
                    )

            if eqns == [0] or eqns == []:
                solution = [{}]
            else:
                if surface_singularities:
                    solution, sing = solve_dgcv(
                        eqns,
                        tVars,
                        method="linsolve",
                        return_divisors=True,
                        pass_to_symbolic_engine=False,
                    )
                    self._singularities["prolongation"] = self._singularities.get(
                        "prolongation", set()
                    ) | {v for v in sing if get_free_symbols(v)}
                else:
                    solution = solve_dgcv(eqns, tVars, method="linsolve")

            if len(solution) == 0:
                raise RuntimeError(
                    f"`Tanaka_symbol.prolongation` failed at a step where a symbolic solver (e.g., sympy.solve if using the default sympy) was being applied. The equation system was {eqns} w.r.t. {tVars}; return solution data was {solution}"
                )

            el_sol = subs(general_elem, solution[0])
            if hasattr(el_sol, "_convert_to_tp"):
                el_sol = el_sol._convert_to_tp()

            fv = set()
            for j in el_sol.coeff_dict.values():
                fv |= get_free_symbols(j)
            if self._parameters:
                fv = set(filter(lambda x: x not in self._parameters, fv))
            new_level = []
            zeroing = {v: 0 for v in fv}
            for v in fv:
                basis_element = subs(el_sol, {**zeroing, v: 1})
                new_level.append(basis_element)

            if ADS is True:
                new_elems = []
                for subS in fast_DS:
                    if height + 1 in subS:
                        new_elems += [
                            v for v in subS[height + 1]
                        ]  ###!!! formerly deepcopy
                        subS[height + 1] = _extract_basis(subS[height + 1] + new_level)
                for subSData in standard_DS:
                    if height + 1 in subSData:
                        new_elems += [
                            v for v in subSData[height + 1]["spanners"]
                        ]  ###!!! formerly deepcopy
                        subSData[height + 1]["spanners"] = _extract_basis(
                            subSData[height + 1]["spanners"] + new_level
                        )
                new_level = _extract_basis(new_level + list(new_elems))

            if with_characteristic_space_reductions is True and height == -1:
                z_level = new_level
            else:
                z_level = levels[0]
            if (
                len(new_level) > 0
                and with_characteristic_space_reductions is True
                and len(z_level) > 0
            ):
                stabilized = False
                while stabilized is False:
                    ambient_basis = new_level
                    varLabel = create_key(prefix="_cv")
                    tVars = [
                        symbol(f"{varLabel}{j}") for j in range(len(ambient_basis))
                    ]
                    solVars = list(tVars)
                    general_elem = sum(
                        [tVars[j] * ambient_basis[j] for j in range(len(tVars))]
                    )
                    eqns = []
                    for idx, dzElem in enumerate(z_level):
                        varLabel2 = varLabel + f"{idx}_"
                        vars2 = [
                            symbol(f"{varLabel2}{j}") for j in range(len(ambient_basis))
                        ]
                        solVars += vars2
                        general_elem2 = sum(
                            [vars2[j] * ambient_basis[j] for j in range(len(tVars))]
                        )

                        commutator = general_elem * dzElem - general_elem2
                        if get_dgcv_category(commutator) == "tensorProduct":
                            eqns += list(commutator.coeff_dict.values())
                        elif get_dgcv_category(commutator) == "algebra_element_class":
                            eqns += commutator.coeffs
                    if surface_singularities:
                        solution, sing = solve_dgcv(
                            eqns,
                            solVars,
                            method="linsolve",
                            return_divisors=True,
                            pass_to_symbolic_engine=False,
                        )
                        self._singularities["prolongation"] = self._singularities.get(
                            "prolongation", set()
                        ) | {v for v in sing if get_free_symbols(v)}
                    else:
                        solution = solve_dgcv(eqns, solVars)
                    if len(solution) == 0:
                        raise RuntimeError(
                            f"`Tanaka_symbol.prolongation` failed at a step where a symbolic solver (e.g., sympy.solve if using the default sympy) was being applied. The equation system was {eqns} w.r.t. {solVars}"
                        )
                    solCoeffs = [j.subs(solution[0]) for j in tVars]

                    free_variables = tuple(
                        set.union(
                            set(),
                            *[
                                get_free_symbols(j) - self._parameters
                                for j in solCoeffs
                            ],
                        )
                    )
                    # new_vectors = []
                    filtered_vectors = []
                    zeroingDict = {other_var: 0 for other_var in free_variables}
                    for var in free_variables:
                        basis_element = [
                            j.subs({var: 1}).subs(zeroingDict) for j in solCoeffs
                        ]
                        filtered_vectors.append(clear_denominators(basis_element))
                        # new_vectors.append(basis_element)
                    # columns = (sp.Matrix(new_vectors).T).columnspace()
                    # filtered_vectors = [
                    #     list(sp.nsimplify(col, rational=True)) for col in columns
                    # ]

                    # def reScale(vec):
                    #     denom = 1
                    #     for t in vec:
                    #         if hasattr(t, "denominator") and denom < t.denominator:
                    #             denom = t.denominator
                    #     if denom == 1:
                    #         return list(vec)
                    #     else:
                    #         return [t * denom for t in vec]

                    # filtered_vectors = [reScale(vec) for vec in filtered_vectors]

                    new_basis = []
                    for coeffs in filtered_vectors:
                        new_basis.append(
                            sum(
                                [
                                    coeffs[j] * ambient_basis[j]
                                    for j in range(1, len(ambient_basis))
                                ],
                                coeffs[0] * ambient_basis[0],
                            )
                        )
                    if len(new_basis) == 0:
                        new_level = []
                        stabilized = True
                    elif len(new_basis) < len(new_level):
                        new_level = new_basis
                    else:
                        new_level = new_basis
                        stabilized = True
            new_levels = self._GLA_structure(
                levels | {height + 1: new_level}, levels.index_threshold
            )
            stable = False
        ssd = [fast_DS, standard_DS] if ADS is True else None
        return new_levels, stable, ssd

    def prolong(
        self,
        iterations: int,
        return_symbol: bool = True,
        report_progress: bool = False,
        report_progress_and_return_nothing: bool = False,
        with_characteristic_space_reductions: bool = None,
        absorb_distinguished_subspaces: bool = False,
        surface_singularities: bool = None,
        simplify_singularities: bool = None,
        max_report_columns: int = 13,
        _fast_algorithm=True,
    ) -> Tanaka_symbol:
        """
        Computes a number of prolongations above the highest level stored in the current Tanaka_symbol data (starting with the first nonnegative integer, so if highest level is -2, it starts with 0, e.g.), up to the number given in `iterations`. Will stop earlier if stabelization is detected.

        New prolongation levels are stored in the symbols internal data. Can alternatively return a new Tanaka_symbol object with the new prolongation levels stored in its data.

        parameters:
        -----------
            iterations: int
        Determines how many prolongation levels to compute

            return_symbol: bool (optional, default=True)
        If True, the algorithm will return a new Tanaka_symbol object with the new prolongation levels stored in its data. If False returns a dictionary whose keys are level weights and values are corresponding prolongation levels (including all initial levels in the symbol data).

            report_progress:
        Reports progress summaries as prolongations are computed in printed output

            report_progress_and_return_nothing: bool (optional, default=False)
        If True, overides `return_symbol` and `surface_singularities`, reports progress summaries of prolongation in printed output, and returns nothing further.

            with_characteristic_space_reductions: bool (defaults to True if relevant - False if theoretically irrelivant)
        If True and `distinguished_subspaces` was registered in the Tanaka_symbol obj creationg then the prolongation algorithm reduces computed prolongation subspaces to the maximal subspaces allowing the result to be a Lie algebra. This is relavent because if any subspace in distinguished_subspaces has elements of nonnegative weight then the standard subspace-preserving prolongation need not yield an algebra. While this reduction can be geometrically motivated, omitting it permits a less demanding algorithm that will produce a larger prolongation space. A typical application for omitting it would be confirming finite prolongation dimensionality with a less demanding algorithm.

            absorb_distinguished_subspaces:bool (default=False)
        This is relevant when distinguished_subspaces include weighted components of higher weight than the given levels data. In this case those higher weighted components are interpreted as subspaces in a tensor algebra where the theoretical prolongation lives, and on which it also operates. The prolongation algorithm proceeds respecting these in the sense that it honors the axiom that distinguished subspaces are invariant under the action of computed prolongation levels. When absorb_distinguished_subspaces=True, then higher weighted levels are artificially added to computed prolongation levels of the same weight. Setting this to True is only natural in niche geometry motivated contexts, so leaving it as False is likely correct if unsure.

            surface_singularities:bool (optional, default=None)
        If paramater space is empty then this is irrelevant for the prolongation algorithm, but affects the output signature. If True (and not overridden by report_progress_and_return_nothing), additionally returns a list of functions whose zeros are singularities in parameter space found during prolongation. i.e., prolongation will be correct for parameters outside of these singularities. If not set to anything (i.e., left as default None) then when relevant all singularities are still found and saved internally. If set to False then tracking of singularities will be skipped, which may improve the algorithm speed slightly.

            max_report_columns: int (positive)
        This controls the maximum number of columns displayed in the output tables when report_progress is true.

        returns:
        --------
            A Tanaka_symbol object with new prolongation levels stored in its data. Returns a dictionary of unformatted prolongation level data if return_symbol=False.

            Additionally returns a list of singularities in parameter space if surface_singularities
        """
        if (
            not isinstance(max_report_columns, numbers.Integral)
            or max_report_columns < 5
        ):
            max_report_columns = 5
        if absorb_distinguished_subspaces is True:
            subspace_data = [
                [
                    directory.copy() for directory in self._standard_process_DS
                ],  # formerly deepcopy
                [directory.copy() for directory in self._standard_process_DS],
            ]
            if len(self._slow_process_DS) > 0:
                dgcv_warning(
                    "Some of the symbols distinguished subspaces (DS) were given by a spanning set of elements that are not all homogeneous. The `absorb_distinguished_subspaces` algorithm can not process such DS, so those DS will not be aborbed into the prolongation. They will still be used for reductions, however."
                )
        else:
            subspace_data = None
        if with_characteristic_space_reductions is None:
            with_characteristic_space_reductions = (
                self._default_to_characteristic_space_reductions
            )
        if report_progress_and_return_nothing is True:
            report_progress = True
        if not isinstance(iterations, numbers.Integral) or iterations < 1:
            raise TypeError("`prolong` expects `iterations` to be a positive int.")
        levels = self.levels
        height = self.height
        distinguished_s_weight_bound = self.height
        for subS in self._fast_process_DS:
            distinguished_s_weight_bound = max(
                distinguished_s_weight_bound, max(subS.keys())
            )
        for subSData in self._standard_process_DS:
            distinguished_s_weight_bound = max(
                distinguished_s_weight_bound, max(subSData.keys())
            )
        stable = False
        if report_progress:
            prol_counter = 1

            def count_to_str(count):
                return f"{count}{'st' if count == 1 else 'nd' if count == 2 else 'rd' if count == 3 else 'th'}"

        if _fast_algorithm is True:
            for w in levels:
                if w >= 0:
                    levels[w] = [_fast_tensor_products(j) for j in levels[w]]
        for j in range(iterations):
            if stable:
                break
            if _fast_algorithm is True:
                levels, stable, subspace_data = self._fast_prolong_by_1(
                    levels,
                    height,
                    distinguished_s_weight_bound=distinguished_s_weight_bound,
                    with_characteristic_space_reductions=with_characteristic_space_reductions,
                    ADS=subspace_data,
                    surface_singularities=surface_singularities,
                    simplify_pivots=simplify_singularities,
                )
            else:
                levels, stable, subspace_data = self._prolong_by_1(
                    levels,
                    height,
                    distinguished_s_weight_bound=distinguished_s_weight_bound,
                    with_characteristic_space_reductions=with_characteristic_space_reductions,
                    ADS=subspace_data,
                    surface_singularities=surface_singularities,
                )
            if report_progress:
                keys = list(levels.keys())
                values = list(levels.values())
                n_cols = len(keys)
                elision_cell = " … "
                elision_border = "     "

                if n_cols > max_report_columns:
                    seg = max_report_columns // 2
                    display_keys = keys[:seg] + [None] + keys[-seg:]
                    display_values = values[:seg] + [None] + values[-seg:]
                else:
                    display_keys = keys
                    display_values = values

                max_len = max(
                    max(len(str(k)) for k in keys),
                    max(len(str(len(v))) for v in values),
                )
                is_elision = [item is None for item in display_keys]

                def fmt_row(label, items):
                    cells = []
                    for item, elide in zip(items, is_elision):
                        cells.append(
                            elision_cell if elide else str(item).ljust(max_len)
                        )
                    return f"│ {label} │ " + " │ ".join(cells) + " │"

                def fmt_border(left, mid, right, junction):
                    segments = []
                    for elide in is_elision:
                        segments.append(
                            elision_border if elide else "─" * (max_len + 2)
                        )
                    header_fill = "─" * len("Weights    │")
                    return f"{left}{header_fill}{mid}" + junction.join(segments) + right

                weight_strs = [None if k is None else str(k) for k in display_keys]
                dim_strs = [None if v is None else str(len(v)) for v in display_values]

                print(f"After {count_to_str(prol_counter)} iteration:")
                print(fmt_border("┌", "┬", "┐", "┬"))
                print(fmt_row("Weights   ", weight_strs))
                print(fmt_border("├", "┼", "┤", "┼"))
                print(fmt_row("Dimensions", dim_strs))
                print(fmt_border("└", "┴", "┘", "┴"))
                prol_counter += 1
            height += 1
        if _fast_algorithm is True:
            for w in levels:
                if w >= 0:
                    levels[w] = [
                        j._convert_to_tp(
                            _hom_id_map=(self._plp, self.levels),
                            _hom_id_label=f"{self._plp}_{c + 1}__{{[{w}]}}",
                        )
                        for c, j in enumerate(levels[w])
                    ]
        if report_progress_and_return_nothing is not True:
            if return_symbol:
                new_nonneg_parts = []
                for key, value in levels.items():
                    if key >= 0:
                        new_nonneg_parts += value
                out = Tanaka_symbol(
                    self.negativePart,
                    new_nonneg_parts,
                    assume_FGLA=self.assume_FGLA,
                    distinguished_subspaces=self.distinguished_subspaces,
                    index_threshold=levels.index_threshold,
                    _validated=retrieve_passkey(),
                    _internal_parameters=self._parameters,
                    _internal_singularities=self._singularities,
                )
                if surface_singularities is True:
                    return out, self._singularities.get("prolongation", set())
                return out
            else:
                if surface_singularities is True:
                    return levels, self._singularities.get("prolongation", set())
                return levels

    def summary(
        self,
        theme=None,
        use_latex=None,
        display_length=500,
        table_scroll=False,
        cell_scroll=False,
        plain_text: bool | None = None,
        condense_tensor_labels: bool = True,
        return_displayable: bool = False,
        **kwargs,
    ):
        dgcvSR = get_dgcv_settings_registry()
        extra_support_for_math_in_tables = bool(
            dgcvSR.get("extra_support_for_math_in_tables") is True
        )
        params = len(self._parameters) > 0
        sing = len(self._singularities.get("prolongation", set())) > 0

        if use_latex is None:
            use_latex = dgcvSR.get("use_latex", False)
        if plain_text is None:
            plain_text = not bool(use_latex)
        if not is_rich_displaying_available():
            plain_text = True

        levels = self.levels or {}
        have_prolongations = any(w >= 0 for w in levels.keys())
        paramessage = "Parametric " if params else ""
        main_title = paramessage + (
            "Tanaka Symbol (+ prolongation) Components"
            if have_prolongations
            else "Tanaka Symbol Components"
        )

        if plain_text:

            def _header_block(title: str, inner_width: int) -> str:
                inner = f" {title} "
                pad = max(0, inner_width - len(inner))
                left = pad // 2
                right = pad - left
                return "===" + ("=" * left) + inner + ("=" * right) + "==="

            dsubs = getattr(self, "distinguished_subspaces", []) or []
            render_panel = not (
                len(dsubs) == 0 or (len(dsubs) == 1 and len(dsubs[0]) == 0)
            )
            sub_title = "Distinguished Subspaces"
            paratitle = "Parameters"
            singtitle = "Singularities in Parameter Space"

            inner_width = max(
                len(f" {main_title} "),
                len(f" {sub_title} ") if render_panel else 0,
                len(f" {paratitle} ") if params else 0,
                len(f" {singtitle} ") if sing else 0,
            )
            lines = [_header_block(main_title, inner_width)]

            for w, basis in sorted(levels.items(), key=lambda kv: kv[0]):
                dim_here = len(basis or [])
                lines.append(f"• graded level {w} ({dim_here} dimensional)")
                basis_str = ", ".join(str(b) for b in (basis or []))
                if display_length is not None and len(basis_str) > display_length:
                    basis_str = "output too long to display; raise `display_length` to a higher bound if needed."
                lines.append(f"  ◦ [{basis_str}]")

            if render_panel:
                lines.append("")
                lines.append(_header_block(sub_title, inner_width))
                for sub in dsubs:
                    inner = ", ".join(str(e) for e in (sub or [])) if sub else "∅"
                    lines.append(f"• [{inner}]")
            if params:
                lines.append("")
                lines.append(_header_block(paratitle, inner_width))
                inner = ", ".join(str(e) for e in order_coordinates(self._parameters))
                lines.append(f"• [{inner}]")
            if sing:
                lines.append("")
                lines.append(_header_block(singtitle, inner_width))
                inner = ", ".join(
                    str(e) for e in self._singularities.get("prolongation", [])
                )
                lines.append(f"• [{inner}]")
            out = "\n".join(lines)
            if return_displayable:
                return out
            print(out)
            return

        # HTML path
        if not isinstance(theme, str):
            style_key = kwargs.get("style", None) or dgcvSR.get("theme", "dark")
        else:
            style_key = theme
        theme_vars = get_style(style_key, legacy=False)
        if use_latex and len(self.nonneg_levels) < 2:
            condense_tensor_labels = False

        def _to_string(e, ul=False):
            if ul:
                s = e._repr_latex_(verbose=False, alias=condense_tensor_labels)
                if s.startswith("$") and s.endswith("$"):
                    s = s[1:-1]
                s = (
                    s.replace(r"\\displaystyle", "")
                    .replace(r"\displaystyle", "")
                    .strip()
                )
                return f"${s}$"
            return str(e)

        rows = []
        sum_computed_dimensions = 0
        for w, basis in sorted(levels.items(), key=lambda kv: kv[0]):
            basis_str = ", ".join(_to_string(b, ul=use_latex) for b in basis)
            if display_length is not None and len(basis_str) > display_length:
                basis_str = "output too long to display; raise `display_length` to a higher bound if needed."
            dim_here = len(basis)
            sum_computed_dimensions += dim_here
            rows.append([str(w), str(dim_here), basis_str])

        footer = [
            [
                {
                    "html": f"Total dimension: {sum_computed_dimensions}",
                    "attrs": {"colspan": 3},
                }
            ]
        ]

        dsubs = getattr(self, "distinguished_subspaces", []) or []
        render_panel = not (
            (len(dsubs) == 0 or (len(dsubs) == 1 and len(dsubs[0]) == 0))
            and not params
            and not sing
        )
        secondary_panel_html = None
        if render_panel:

            def _to_math(e):
                if use_latex:
                    return f"${LaTeX(e)}$"
                return str(e)

            def _panel_section(title, items_html, include_divider):
                # Clean, semantic layout nodes with no presentation styling properties
                divider = (
                    "<div class='dgcv-side-divider'></div>" if include_divider else ""
                )

                lis = "".join(f"<li>{item}</li>" for item in items_html)
                return (
                    f"{divider}"
                    f"<div class='dgcv-side-title'>{title}</div>"
                    f"<ul class='dgcv-side-list'>{lis}</ul>"
                )

            sections = []

            if not (len(dsubs) == 0 or (len(dsubs) == 1 and len(dsubs[0]) == 0)):
                if use_latex:
                    dsub_items = []
                    for sub in dsubs:
                        if sub:
                            inner = ", ".join(
                                (e._repr_latex_(raw=True) or "").strip() for e in sub
                            )
                        else:
                            inner = r"\varnothing"
                        dsub_items.append(f"$\\left\\langle {inner} \\right\\rangle$")
                else:
                    import html

                    dsub_items = [
                        f"[{', '.join(html.escape(str(e)) for e in sub) if sub else '∅'}]"
                        for sub in dsubs
                    ]
                sections.append(
                    _panel_section("Distinguished Subspaces", dsub_items, False)
                )

            if params:
                param_items = [_to_math(e) for e in order_coordinates(self._parameters)]
                sections.append(
                    _panel_section(
                        "Parameters", [", ".join(param_items)], len(sections) > 0
                    )
                )

            if sing:
                sing_items = [
                    _to_math(e) for e in self._singularities.get("prolongation", [])
                ]
                sections.append(
                    _panel_section(
                        "Singularities in Parameter Space",
                        [", ".join(sing_items)],
                        len(sections) > 0,
                    )
                )

            raw_content = "".join(sections)

            if "--plaque-fill" in (theme_vars or ""):
                content_html = f"<div class='dgcv-side-plaque'>{raw_content}</div>"
            else:
                content_html = raw_content

            secondary_panel_html = f"<div class='dgcv-side-panel'>{content_html}</div>"

        # All structural presentation rules cleanly centralized inside one target block
        extra_css = """
.dgcv-data-table th:nth-child(1), .dgcv-data-table td:nth-child(1),
.dgcv-data-table th:nth-child(2), .dgcv-data-table td:nth-child(2) {
    width: 1%;
    white-space: nowrap;
}
.dgcv-side-panel {
    border: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main);
    background: var(--dgcv-special-background, var(--dgcv-bg-surface));
    color: var(--dgcv-special-text, var(--dgcv-text-heading));
    padding: 12px;
    height: 100%;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
}
.dgcv-side-plaque {
    display: block;
    box-sizing: border-box;
    width: 100%;
    padding: 0.75rem 1rem;
    background: var(--plaque-fill);
    border-radius: inherit;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
    overflow-x: auto;
    overflow-y: hidden;
    border: var(--dgcv-border-width, 1px) solid var(--plaque-border, var(--dgcv-border-main));
}
.dgcv-side-divider {
    border-top: calc(var(--dgcv-border-width, 1px) * 2) solid var(--dgcv-border-main);
}
.dgcv-side-title {
    padding: 12px;
    border-bottom: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main);
    font-weight: bold;
}
.dgcv-side-list {
    margin: 0;
    padding: 0;
    list-style: none;
    overflow-y: visible;
    height: fit-content;
}
.dgcv-side-list li {
    overflow-x: auto;
    overflow-y: hidden;
    white-space: nowrap;
    padding: 8px;
    border-bottom: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main);
    display: list-item;
    list-style-type: disc;
    list-style-position: inside;
}
.dgcv-side-scroll-area { overflow-x: auto; width: 100%; }
.dgcv-side-panel h3 { margin: 0; font-weight: bold; font-size: 1.1em; }
.dgcv-side-panel hr { border: 0; border-top: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main); margin: 8px 0; }
.dgcv-data-table tfoot td {
    text-align: left;
    font-weight: bold;
    background-color: var(--dgcv-bg-primary);
    color: var(--dgcv-text-heading);
    border-top: 2px solid var(--dgcv-border-main);
}
"""

        table = build_plain_table(
            columns=["Weight", "Dimension", "Basis"],
            rows=rows,
            caption=main_title,
            theme_css_vars=theme_vars,
            extra_css=extra_css,
            table_attrs='style="table-layout:auto;"',
            cell_align="center",
            escape_cells=False,
            escape_headers=True,
            secondary_panel_html=secondary_panel_html,
            layout="row",
            gap_px=10,
            side_width="340px",
            container_id="tanaka-summary",
            footer_rows=footer,
            table_scroll=table_scroll,
            cell_scroll=cell_scroll,
            ul=12,
            ur=12,
            ll=12,
            lr=12,
        )

        out = latex_in_html(
            table, extra_support_for_math_in_tables=extra_support_for_math_in_tables
        )
        if return_displayable:
            return out
        show(out)

    def __str__(self):
        levels = self.levels
        total_dim = sum(len(basis) for basis in levels.values())

        all_weights = list(levels.keys()) + ["total"]
        all_dims = [len(basis) for basis in levels.values()] + [total_dim]

        max_len = max(
            max(len(str(w)) for w in all_weights),
            max(len(str(d)) for d in all_dims),
        )

        weights_row = " │ ".join(str(w).ljust(max_len) for w in all_weights)
        dims_row = " │ ".join(str(d).ljust(max_len) for d in all_dims)

        weights_line = f"Weights    │ {weights_row}"
        dims_line = f"Dimensions │ {dims_row}"
        line_len = max(len(weights_line), len(dims_line)) + 1
        header_len = len("Weights    │ ")

        top = f"┌{'─' * (header_len - 1)}┬{'─' * (1 + line_len - header_len)}┐"
        middle = f"├{'─' * (header_len - 1)}┼{'─' * (1 + line_len - header_len)}┤"
        bottom = f"└{'─' * (header_len - 1)}┴{'─' * (1 + line_len - header_len)}┘"

        result = [
            "Tanaka Symbol:",
            top,
            f"│ {weights_line} │",
            middle,
            f"│ {dims_line} │",
            bottom,
        ]
        return "\n".join(result)

    def _repr_latex_(self):
        levels = self.levels
        weights = list(levels.keys())
        dims = [len(basis) for basis in levels.values()]
        total_dim = sum(dims)

        # Format weights and dims as strings
        weights_row = " & ".join(map(str, weights)) + r" & \text{total} \\"
        dims_row = " & ".join(map(str, dims)) + rf" & {total_dim} \\"

        lines = [
            r"\textbf{Tanaka Symbol}\\[0.5em]",
            r"\begin{array}{|c||" + "c" * (len(weights)) + r"|c|}",
            r"\hline",
            r"\text{Weights} & " + weights_row,
            r"\hline",
            r"\text{Dimensions} & " + dims_row,
            r"\hline",
            r"\end{array}",
        ]
        return "$" + "\n".join(lines) + "$"

    def _sympystr(self, printer):
        result = ["Tanaka Symbol:"]
        result.append("Weights and Dimensions:")
        for weight, basis in self.levels.items():
            dim = len(basis)
            basis_str = ", ".join(printer.doprint(b) for b in basis)
            result.append(f"  {weight}: Dimension {dim}, Basis: [{basis_str}]")
        return "\n".join(result)

    def export_algebra_data(
        self,
        preserve_negative_part_basis=True,
        _internal_call_lock=None,
        try_hard=False,
    ):
        grading_vec = []
        indexBands = dict()
        dimen = 0
        complimentWeights = {}
        permutation = []
        for weight, level in self.levels.items():
            if (
                preserve_negative_part_basis
                and weight < 0
                and not (level == [0] or level is None)
            ):
                permutation += [self.negativePart.basis.index(elem) for elem in level]
            lLength = 0 if (level == [0] or level is None) else len(level)
            nextDim = dimen + lLength
            for j in range(dimen, nextDim):
                indexBands[j] = (weight, j - dimen)
            complimentWeights[weight] = (dimen, self.dimension - nextDim)
            dimen = nextDim
            grading_vec += [weight] * lLength
        if preserve_negative_part_basis:
            invPerm = [permutation.index(j) for j in range(len(permutation))]
            tail = list(range(self.negativePart.dimension, self.dimension))
            back = invPerm + tail
            forward = permutation + tail

        def flatToLayered(idx):
            return indexBands[idx]

        def bracket_decomp(idx1, idx2, try_hard=False):
            w1, sId1 = flatToLayered(idx1)
            w2, sId2 = flatToLayered(idx2)
            newElem = (
                (self.levels[w1][sId1]) * (self.levels[w2][sId2])
            )  ###!!! review for ambient_rep requirements
            newWeight = w1 + w2
            if self.levels[newWeight] is not None:
                ambient_basis = [
                    j for j in self.levels[newWeight]
                ]  ###!!! review for ambient_rep requirements
            nLDim = (
                0
                if (ambient_basis is None or ambient_basis == [0])
                else len(ambient_basis)
            )
            if nLDim == 0:
                if getattr(newElem, "is_zero", False) or newElem == 0:
                    return [0] * dimen
                else:
                    return "NoSol"
            general_elem, tVars = linear_combination(ambient_basis)
            eqns = [newElem - general_elem]
            sol = solve_dgcv(eqns, tVars, method="linsolve")
            if len(sol) == 0:
                if try_hard:
                    dgcv_warning(
                        "symbol --> algebra method failed to find structure eqns with fast default algorithm"
                    )
                    sol = solve_dgcv(
                        [simplify(eqn) for eqn in eqns],
                        tVars,
                        method="linsolve",
                    )
                    if len(sol) == 0:
                        return "NoSol"
                else:
                    return "NoSol"
            coeffVec = [sol[0].get(var, var) for var in tVars]

            #   newWeight should be in complimentWeights by construction
            start = [0] * complimentWeights[newWeight][0]
            end = [0] * complimentWeights[newWeight][1]
            return start + coeffVec + end

        str_data = array_dgcv(
            dict(),
            shape=(dimen, dimen),
            null_return=freeze_matrix(matrix_dgcv.zeros(dimen, 1)),
        )
        for j in range(dimen):
            for k in range(j + 1, dimen):
                bracket_data = bracket_decomp(k, j, try_hard=try_hard)
                if bracket_data == "NoSol":
                    warningStr = f"due to failure to confirm if the symbol data is closed under brackets between basis elements {j} and {k}."
                    if _internal_call_lock != retrieve_passkey():
                        dgcv_warning(
                            "Unable to extract algebra structure, "
                            + warningStr
                            + " So `None` was returned by `export_algebra_data`."
                        )
                        return None
                    return (
                        "Unable to extract algebra structure from `Tanaka_symbol` object, "
                        + warningStr
                    )
                new_mat = matrix_dgcv(bracket_data)
                if new_mat:
                    str_data[(k, j)] = new_mat
                    str_data[(j, k)] = -new_mat

        if preserve_negative_part_basis:

            def permute_structure_data(SD, perm):
                d = SD.shape[0]
                new_sd = dict()
                perm = list(perm)
                sddd = SD._data
                for idx, v in sddd.items():
                    i, j = SD._unspool(idx)
                    new_key = (perm[i], perm[j])
                    inner_shp = (d, 1)
                    for k, value in enumerate(v):
                        if value != 0:
                            if new_key in new_sd:
                                new_sd[new_key][perm[k]] = value
                            else:
                                new_sd[new_key] = matrix_dgcv(
                                    {perm[k]: value}, shape=inner_shp
                                )

                return array_dgcv(
                    new_sd,
                    shape=(d, d),
                    null_return=freeze_matrix(matrix_dgcv.zeros(d, 1)),
                )

            return {
                "structure_data": permute_structure_data(str_data, forward),
                "grading": [
                    (grading_vec[back[j]] if j < len(back) else grading_vec[j])
                    for j in range(self.dimension)
                ],
            }
        return {"structure_data": str_data, "grading": [grading_vec]}


def _GAE_to_hom_formatting(elem, nilradical, test_weights=None, return_weights=False):
    if get_dgcv_category(elem) not in {
        "algebra_element",
        "subalgebra_element",
        "vector_space_element",
    }:
        if return_weights:
            if get_dgcv_category(elem) == "tensorProduct":
                try:
                    return elem, elem.compute_weight(
                        test_weights=test_weights, _return_mixed_weight_list=True
                    )[0]
                except Exception:
                    raise ValueError(
                        "Unable to process given distinguished subspaces. Cannot infer weights of components in some of the given elements "
                    )
            else:
                return elem, []
        return elem
    test_switch = get_dgcv_category(nilradical) == "subalgebra"
    if test_weights is None:
        test_weights = [elem.algebra.grading[0]]
    wd = elem.weighted_decomposition(test_weights=test_weights, flatten_weights=True)
    weights = list(wd.keys())
    if all(w < 0 for w in weights):
        if test_switch and get_dgcv_category(elem) == "algebra_element":
            elem = nilradical._class_builder(
                nilradical.contains(elem, return_basis_coeffs=True), elem.valence
            )
        if return_weights:
            return elem, weights
        return elem
    terms = []
    for weight, term in wd.items():
        if weight < 0:
            if test_switch and get_dgcv_category(term) == "algebra_element":
                term = nilradical._class_builder(
                    nilradical.contains(term, return_basis_coeffs=True), term.valence
                )
            terms.append(term)
        else:
            for testEl in nilradical:
                terms.append(
                    _GAE_to_hom_formatting(
                        term * testEl, nilradical, test_weights=test_weights
                    )
                    @ testEl.dual()
                )
    if return_weights:
        return sum(terms[1:], terms[0]), weights
    return sum(terms[1:], terms[0])


class _fast_tensor_products:
    def __init__(self, coeff_dict, alg=None, _validated=None, _hom_id=None):
        if isinstance(coeff_dict, _fast_tensor_products):
            coeff_dict, alg, _validated, _hom_id = (
                coeff_dict.coeff_dict,
                coeff_dict.algebra,
                coeff_dict.degree,
                coeff_dict._hom_id,
            )
        self.algebra = alg
        if _validated is None:
            if get_dgcv_category(coeff_dict) == "tensorProduct":
                if alg is None:
                    self.algebra = coeff_dict.vector_space
                self.coeff_dict = dict()
                self.degree = 0
                for k, v in coeff_dict.coeff_dict.items():
                    newkey = k[: len(k) // 3]
                    self.coeff_dict[newkey] = v
                    self.degree = max(self.degree, len(newkey))
            elif isinstance(coeff_dict, dict):
                self.coeff_dict = dict()
                self.degree = 0
                for k, v in coeff_dict.items():
                    if v != 0 or k == tuple():
                        self.coeff_dict[k] = v
                        self.degree = max(self.degree, len(k))
            elif get_dgcv_category(coeff_dict) in {
                "algebra_element",
                "subalgebra_element",
                "vectorspace_element",
            }:
                if alg is None:
                    self.algebra = coeff_dict.algebra
                self.degree = 1
                self.coeff_dict = {
                    (k,): v for k, v in enumerate(coeff_dict.coeffs) if v != 0
                }
            else:
                self.coeff_dict = dict()
        else:
            self.degree = _validated
            self.coeff_dict = coeff_dict
        if len(self.coeff_dict) == 0:
            self.coeff_dict = {tuple(): 0}
            self.degree = 0
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "fastTensorProduct"
        self._is_zero = None
        self._coeffs = None
        if self.degree < max(len(k) for k in self.coeff_dict):
            raise TypeError("ftp init fail")
        if _hom_id:
            self._hom_id = (
                _hom_id  # _hom_id format: [{(jidx, kidx, jdeg, kdeg): val,...}, label]
            )
        else:
            self._hom_id = None

    @property
    def is_zero(self):
        if self._is_zero is None:
            self._is_zero = (
                False if any(v != 0 for v in self.coeff_dict.values()) else True
            )
        return self._is_zero

    @property
    def coeffs(self):
        if self._coeffs is None:
            self._coeffs = list(self.coeff_dict.values())
        return self._coeffs

    def _to_algebra(self, alg=None):
        if alg is None:
            alg = self.algebra
        ae = 0
        for k, v in self.coeff_dict.items():
            if len(k) == 1:
                ae += v * alg.basis[k[0]]
        return ae

    def _convert_to_tp(self, _hom_id_map=None, _hom_id_label=None):
        """
        _hom_id_map should be a pair of (str pref, map from neg weights to components)
        """
        from ..core.tensors.tensors import tensorProduct

        new_dict = dict()
        for k, v in self.coeff_dict.items():
            k2 = tuple(1 if idx == 0 else 0 for idx in range(len(k)))
            k3 = (self.algebra.dgcv_vs_id,) * len(k)
            newkey = k + k2 + k3
            new_dict[newkey] = v
        if self._hom_id and _hom_id_map:
            decomp, label = self._hom_id
            if _hom_id_label:
                label = _hom_id_label
            pref, ngla = _hom_id_map
            hidd = dict()
            valid = True
            for key, v in decomp.items():
                jidx, kidx, jdeg, kdeg = key
                try:
                    jstr = str(ngla[jdeg][jidx])
                    if kdeg < 0:
                        kstr = str(ngla[kdeg][kidx])
                    else:
                        kstr = f"{pref}_{kidx + 1}__{{[{kdeg}]}}"
                    hidd[(jstr, kstr)] = v
                except Exception:
                    valid = False
                    break
            if valid:
                homid = [hidd, label]
            else:
                homid = None
        else:
            homid = None
        return tensorProduct([], new_dict, _hom_id=homid)

    def __add__(self, other):
        if other == 0 or getattr(other, "is_zero", False):
            return self
        if isinstance(other, _fast_tensor_products):
            new_dict = dict(self.coeff_dict)
            deg = self.degree
            for k, v in other.coeff_dict.items():
                deg = max(len(k), deg)
                new_dict[k] = self.coeff_dict.get(k, 0) + v

            if self._hom_id and other._hom_id:
                nhidd = {k: v for k, v in self._hom_id[0].items()}
                for k, v in other._hom_id[0].items():
                    newv = nhidd.get(k, 0) + v
                    if newv == 0:
                        nhidd.pop(k)
                    else:
                        nhidd[k] = newv
                hom_id = [nhidd, ""]
            else:
                hom_id = None

            return _fast_tensor_products(
                new_dict, self.algebra, _validated=deg, _hom_id=hom_id
            )
        if get_dgcv_category(other) in {
            "algebra_element",
            "subalgebra_element",
            "vector_space_element",
        }:
            return self + _fast_tensor_products(other)
        return NotImplemented

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return (self).__add__(-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, expr_numeric_types()):
            if other == 0:
                return _fast_tensor_products(
                    {tuple(): 0},
                    self.algebra,
                    _validated=0,
                    _hom_id=[{}, ""] if self._hom_id else None,
                )
            homid = (
                [{k: other * v for k, v in self._hom_id[0].items()}, ""]
                if self._hom_id
                else None
            )
            return _fast_tensor_products(
                {k: other * v for k, v in self.coeff_dict.items()},
                self.algebra,
                _validated=self.degree,
                _hom_id=homid,
            )
        if isinstance(other, _fast_tensor_products):
            if other.degree == 0:
                return sum(v * self for v in other.coeff_dict.values())
            if self.degree == 1:
                return other * (-self._to_algebra())
            if self.degree == 0:
                return sum(v * other for v in self.coeff_dict.values())
            if other.degree == 1:
                return self * other._to_algebra()
            new_dict = dict()
            deg = 0
            ae1 = 0
            ae2 = 0
            for k1, v1 in self.coeff_dict.items():
                if len(k1) == 1:
                    ae1 += v1 * self.algebra.basis[k1[0]]
                    continue
                k1L, k1A, k1B, k1T = k1[0], k1[:-1], k1[1:], k1[-1]
                for k2, v2 in other.coeff_dict.items():
                    if len(k2) == 1:
                        ae2 += v2 * other.algebra.basis[k2[0]]
                        continue
                    k2L, k2A, k2B, k2T = k2[0], k2[:-1], k2[1:], k2[-1]
                    if k1T == k2L:
                        newkey = k1B + k2A
                        newval = new_dict.get(newkey, 0) + v1 * v2
                        if newval != 0:
                            deg = max(len(newkey), deg)
                            new_dict[newkey] = newval
                        else:
                            new_dict.pop(newkey, None)
                    if k1L == k2T:
                        newkey = k2B + k1A
                        newval = new_dict.get(newkey, 0) - v1 * v2
                        if newval != 0:
                            deg = max(len(newkey), deg)
                            new_dict[newkey] = newval
                        else:
                            new_dict.pop(newkey, None)
            return (
                _fast_tensor_products(new_dict, self.algebra, _validated=deg)
                + ae1 * other
                + self * ae2
            )
        if get_dgcv_category(other) in {
            "algebra_element",
            "subalgebra_element",
            "vector_space_element",
        }:
            if self.degree == 0:
                return sum(v * other for v in self.coeff_dict.values())
            if self.degree == 1:
                return self._to_algebra() * other
            new_dict = dict()
            ac = other.coeffs
            ae1 = 0
            for k1, v1 in self.coeff_dict.items():
                if len(k1) == 1:
                    ae1 += v1 * self.algebra.basis[k1[0]]
                    continue
                k1A, k1T = k1[:-1], k1[-1]
                newval = new_dict.get(k1A, 0) + ac[k1T] * v1
                if newval != 0:
                    new_dict[k1A] = newval
                else:
                    new_dict.pop(k1A, None)
            if self.degree == 2:
                return (
                    _fast_tensor_products(
                        new_dict, self.algebra, _validated=self.degree - 1
                    )._to_algebra()
                    + ae1 * other
                )
            return (
                _fast_tensor_products(
                    new_dict, self.algebra, _validated=self.degree - 1
                )
                + ae1 * other
            )
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, expr_numeric_types()):
            if other == 0:
                return _fast_tensor_products(
                    dict(),
                    self.algebra,
                    _validated=0,
                    _hom_id=[{}, ""] if self._hom_id else None,
                )
            homid = (
                [{k: other * v for k, v in self._hom_id[0].items()}, ""]
                if self._hom_id
                else None
            )
            return _fast_tensor_products(
                {k: other * v for k, v in self.coeff_dict.items()},
                self.algebra,
                _validated=self.degree,
                _hom_id=homid,
            )
        if self.degree == 0:
            return sum(v * other for v in self.coeff_dict.values())
        return self * (-other)

    def __neg__(self):
        return _fast_tensor_products(
            {k: -v for k, v in self.coeff_dict.items()},
            self.algebra,
            _validated=self.degree,
            _hom_id=[{k: -v for k, v in self._hom_id[0].items()}, ""]
            if self._hom_id
            else None,
        )

    def __matmul__(self, other):
        if isinstance(other, expr_numeric_types()):
            return self * other
        if get_dgcv_category(other) in {
            "algebra_element",
            "subalgebra_element",
            "vector_space_element",
        }:
            ac = other.coeffs
            new_dict = dict()
            for k, v in self.coeff_dict.items():
                for idx, c in enumerate(ac):
                    if c != 0:
                        newkey = k + (idx,)
                        newval = new_dict.get(newkey, 0) + c * v
                        if newval != 0:
                            new_dict[newkey] = newval
                        else:
                            new_dict.pop(newkey, None)
            return _fast_tensor_products(
                new_dict, self.algebra, _validated=self.degree + 1
            )
        if isinstance(other, _fast_tensor_products):
            ac = other.coeff_dict
            new_dict = dict()
            for k, v in self.coeff_dict.items():
                for idx, c in other.coeff_dict.items():
                    newkey = k + idx
                    newval = new_dict.get(newkey, 0) + c * v
                    if newval != 0:
                        new_dict[newkey] = newval
                    else:
                        new_dict.pop(newkey, None)
            return _fast_tensor_products(
                new_dict, self.algebra, _validated=self.degree + other.degree
            )
        return NotImplemented

    def __rmatmul__(self, other):
        return self.__matmul__(other)

    def subs(self, subs_data):
        new_dict = {k: subs(v, subs_data) for k, v in self.coeff_dict.items()}
        if self._hom_id:
            hidd = dict()
            for k, v in self._hom_id[0].items():
                newv = subs(v, subs_data)
                if newv != 0:
                    hidd[k] = newv
            homid = [hidd, ""]
        else:
            homid = None
        return _fast_tensor_products(
            new_dict, self.algebra, _validated=self.degree, _hom_id=homid
        )


class distribution(dgcv_class):
    def __init__(
        self,
        spanning_vf_set=None,
        spanning_df_set=None,
        assume_compatibility: bool = False,
        check_compatibility_aggressively: bool = False,
        _assume_minimal_Data=None,
        *,
        coordinate_space: None | Sequence[Any] = None,
        find_basis: bool = False,
        find_polynomial_spanners=False,
        assume_starting_objs_polynomial=False,
        assume_spanning_sections_linearly_indep=False,
        formatting: None | Literal["complex", "real"] = None,
        dimension_hint=None,
    ):
        """
        The optional `dimension hint` keyword is used to minimize expensive linear independence
        checks when extracting a basis. Only set this to a known upper bound on possible
        distribution rank or else computed basis may be wrong.
        """
        if assume_spanning_sections_linearly_indep:
            _assume_minimal_Data = retrieve_passkey()
        if spanning_vf_set is not None:
            if not isinstance(spanning_vf_set, (list, tuple)):
                raise TypeError(
                    "`distribution` spanning_vf_set must be a list/tuple of vector fields."
                )
            spanning_vf_set = tuple(spanning_vf_set)
            if not all(
                query_dgcv_categories(vf, {"vector_field"}) for vf in spanning_vf_set
            ):
                raise TypeError(
                    "`distribution` spanning_vf_set must contain only vector fields."
                )

        if spanning_df_set is not None:
            if not isinstance(spanning_df_set, (list, tuple)):
                raise TypeError(
                    "`distribution` spanning_df_set must be a list/tuple of degree-1 differential forms."
                )
            spanning_df_set = tuple(spanning_df_set)
            if not all(
                query_dgcv_categories(df, {"differential_form"})
                and getattr(df, "degree", None) == 1
                for df in spanning_df_set
            ):
                raise TypeError(
                    "`distribution` spanning_df_set must contain only degree-1 differential forms."
                )

        if spanning_vf_set is None and spanning_df_set is None:
            self._prefered_data_type = 1
            self._spanning_vf_set = tuple()
            self._spanning_df_set = tuple()
            self.coordinates = tuple()
            self.formatting = None
            self._vf_basis = (
                tuple() if _assume_minimal_Data == retrieve_passkey() else None
            )
            self._df_basis = (
                tuple() if _assume_minimal_Data == retrieve_passkey() else None
            )
            self._derived_flag = None
            self._wderived_flag = None
            return

        self._simplifying_preference = find_polynomial_spanners
        if formatting not in (None, "complex", "real"):
            formatting = None

        self._prefered_data_type = 1 if spanning_df_set is None else 0

        vfs = (
            None
            if spanning_vf_set is None
            else self._normalize_spanning_set(
                spanning_vf_set,
                formatting=formatting,
                scale_to_poly=find_polynomial_spanners
                and not assume_starting_objs_polynomial,
            )
        )
        dfs = (
            None
            if spanning_df_set is None
            else self._normalize_spanning_set(
                spanning_df_set,
                formatting=formatting,
                scale_to_poly=find_polynomial_spanners
                and not assume_starting_objs_polynomial,
            )
        )

        if vfs is not None and dfs is not None and assume_compatibility is False:
            for df in dfs:
                for vf in vfs:
                    val = df(vf)
                    if check_compatibility_aggressively:
                        val = simplify(val)
                    if val != 0 or not getattr(val, "is_zero", False):
                        raise TypeError(
                            "Unable to verify that the provided vector fields and differential forms annihilate each other. "
                            "Use assume_compatibility=True to bypass this check, or set check_compatibility_aggressively=True."
                        )

        def _flatten_unique(seq_of_seqs):
            out = []
            seen = set()
            for seq in seq_of_seqs:
                for v in seq:
                    if v in seen:
                        continue
                    seen.add(v)
                    out.append(v)
            return tuple(out)

        obj_spanners = (
            spanning_vf_set
            if spanning_vf_set
            else spanning_df_set
            if spanning_df_set
            else []
        )
        inferred_min_vs = _flatten_unique(
            obj.infer_minimal_varSpace() for obj in obj_spanners
        )

        if coordinate_space is not None:
            if not isinstance(coordinate_space, (list, tuple, set)):
                raise TypeError(
                    "coordinate_space must be a list/tuple/set if provided."
                )
            vs = tuple(coordinate_space)
            missing = [v for v in inferred_min_vs if v not in vs]
            if missing:
                raise ValueError(
                    "Provided coordinate_space does not contain all inferred minimal coordinates: "
                    + ", ".join(str(x) for x in missing)
                )
            self.coordinates = order_coordinates(vs)
        else:
            self.coordinates = order_coordinates(inferred_min_vs)
        self._dim_hint = dimension_hint
        self._spanning_vf_set = vfs
        self._spanning_df_set = dfs
        self._characteristic = None
        self._ext_power_class_cache = None
        self.formatting = formatting
        self._anticanonical_bundle = None
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "distribution"

        if _assume_minimal_Data == retrieve_passkey():
            self._vf_basis = self._spanning_vf_set
            self._df_basis = self._spanning_df_set
        else:
            self._vf_basis = None
            self._df_basis = None

        self._derived_flag = None
        self._wderived_flag = None

        if find_basis is True:
            if vfs:
                self._spanning_vf_set = self.vf_basis
            else:
                self._spanning_df_set = self.df_basis

        self.varSpace = self.coordinates  # deprecated

    @staticmethod
    def _infer_minimal_vs(obj) -> Tuple[Any, ...]:
        f = getattr(obj, "infer_minimal_varSpace", None)
        if callable(f):
            vs = f()
            if isinstance(vs, tuple):
                return vs
            if isinstance(vs, (list, set)):
                return tuple(vs)
        vs2 = getattr(obj, "varSpace", None)
        if isinstance(vs2, tuple):
            return vs2
        if isinstance(vs2, (list, set)):
            return tuple(vs2)
        return tuple()

    @property
    def anticanonical_bundle(self):
        if self._anticanonical_bundle is None:
            vf_basis = self.vf_basis
            if len(vf_basis) == 0:
                self._anticanonical_bundle = tensor_field_class(coeff_dict={tuple(): 1})
            else:
                self._anticanonical_bundle = wedge(*vf_basis)
        return self._anticanonical_bundle

    @property
    def rank(self):
        return len(self.vf_basis)

    @staticmethod
    def _validated_format_of(obj) -> str | None:
        fmt = getattr(obj, "_validated_format", None)
        if fmt in ("standard", "complex", "real"):
            return fmt
        return None

    @staticmethod
    def _needs_mixed_conversion(formats: set[str]) -> bool:
        return ("real" in formats) and ("complex" in formats)

    @staticmethod
    def _preferred_formatting_from_settings() -> str:
        pref = get_dgcv_settings_registry().get("preferred_variable_format", None)
        return "real" if pref == "real" else "complex"

    @staticmethod
    def _convert_obj(obj, target: str):
        if target == "real":
            return allToReal(obj)
        return allToSym(obj)

    def _normalize_spanning_set(
        self, spanning_set, *, formatting: None | str, scale_to_poly: bool = False
    ):
        elems = tuple(spanning_set)
        if not elems:
            return tuple()

        if formatting is not None:
            target = formatting
            out = []
            for e in elems:
                ef = self._validated_format_of(e)
                if ef != target and ef in ("real", "complex"):
                    new_e = self._convert_obj(e, target)
                    if scale_to_poly:
                        new_e = new_e.scale_to_polynomial_attempt()
                    out.append(new_e)
                else:
                    if scale_to_poly:
                        new_e = e.scale_to_polynomial_attempt()
                        out.append(new_e)
                    else:
                        out.append(e)
            return tuple(out)

        fmts = set()
        new_elems = []
        for e in elems:
            if scale_to_poly:
                e = e.scale_to_polynomial_attempt()
            new_elems.append(e)
            ef = self._validated_format_of(e)
            if ef is not None:
                fmts.add(ef)
        elems = new_elems

        if not self._needs_mixed_conversion(fmts):
            return elems

        target = self._preferred_formatting_from_settings()
        out = []
        for e in elems:
            ef = self._validated_format_of(e)
            if ef in ("real", "complex") and ef != target:
                out.append(self._convert_obj(e, target))
            else:
                out.append(e)
        return tuple(out)

    @property
    def spanning_vf_set(self):
        if self._spanning_vf_set is None:
            vfs = annihilator(
                self.df_basis,
                coordinate_space=self.coordinates,
                coherent_coordinates_checked=False,
                polynomial_bases=self._simplifying_preference,
            )
            self._spanning_vf_set = vfs
            self._vf_basis = vfs
        return self._spanning_vf_set

    @property
    def vf_basis(self):
        if self._vf_basis is None:
            if self._spanning_vf_set is None:
                return self.spanning_vf_set
            self._vf_basis = _extract_basis_by_wedge_vectorized(
                self._spanning_vf_set, dimension_hint=self._dim_hint
            )
        return self._vf_basis

    @property
    def df_basis(self):
        if self._df_basis is None:
            if self._spanning_df_set is None:
                return self.spanning_df_set
            self._df_basis = _extract_basis_by_wedge_vectorized(
                self._spanning_df_set, dimension_hint=self._dim_hint
            )
        return self._df_basis

    @property
    def spanning_df_set(self):
        if self._spanning_df_set is None:
            sdf = annihilator(
                self.vf_basis,
                coordinate_space=self.coordinates,
                coherent_coordinates_checked=False,
                polynomial_bases=self._simplifying_preference,
            )
            self._spanning_df_set = sdf
            self._df_basis = sdf
        return self._spanning_df_set

    def intersection(self, other, formatting=None):
        if not isinstance(other, distribution):
            raise TypeError(
                "dgcv.distribution.intersection can only operate on other dgcv.distribution instances."
            )
        svf = self._vf_basis if self._vf_basis is not None else self.spanning_vf_set
        sdf = self._df_basis if self._df_basis is not None else self.spanning_df_set
        ovf = other._vf_basis if other._vf_basis is not None else other.spanning_vf_set
        odf = other._df_basis if other._df_basis is not None else other.spanning_df_set
        if len(svf) * len(odf) < len(ovf) * len(sdf):
            interVF = annihilator(odf, control_distribution=svf)
        else:
            interVF = annihilator(sdf, control_distribution=ovf)
        formatting = (
            self.formatting
            if formatting is None and self.formatting == other.formatting
            else formatting
        )
        return distribution(
            interVF,
            formatting=formatting,
        )

    def union(self, other, extract_basis=False):
        if not isinstance(other, distribution):
            raise TypeError(
                "dgcv.distribution.intersection can only operate on other dgcv.distribution instances."
            )
        return distribution(
            self.spanning_vf_set + other.spanning_vf_set, extract_basis=extract_basis
        )

    def derived_flag(
        self,
        find_polynomial_spanners=True,
        max_iterations=10,
        use_numeric_methods=False,
    ):
        use_numeric = use_numeric_methods or bool(
            get_dgcv_settings_registry().get("use_numeric_methods", False)
        )
        if self._derived_flag is None:
            tiered_list = [list(self.vf_basis)]

            def derive_extension(tieredList, obstruction=None):
                flattenedTL = sum(tieredList, [])
                newTeir = []
                topLevel = tieredList[-1]
                obstr = obstruction if obstruction else simplify(wedge(*flattenedTL))
                for vf1 in flattenedTL:
                    for vf2 in topLevel:
                        nb = LieDerivative(vf1, vf2)
                        if find_polynomial_spanners is True:
                            nb = nb.scale_to_polynomial_attempt(factor=True)
                        new_obs = obstr * nb if use_numeric else simplify(obstr * nb)
                        if use_numeric:
                            if zeroish(new_obs):
                                continue
                        elif new_obs == 0 or getattr(new_obs, "is_zero", False):
                            continue
                        obstr = new_obs
                        newTeir.append(nb)
                return list(tieredList) + [newTeir], obstr

            obstr = None
            for _ in range(max_iterations):
                tiered_list, obstr = derive_extension(tiered_list, obstr)
                if len(tiered_list[-1]) == 0:
                    tiered_list = tiered_list[:-1]
                    break
            self._derived_flag = tiered_list
        return self._derived_flag

    def weak_derived_flag(
        self,
        find_polynomial_spanners=False,
        max_iterations=10,
        use_numeric_methods=False,
    ):
        use_numeric = use_numeric_methods or bool(
            get_dgcv_settings_registry().get("use_numeric_methods", False)
        )
        if self._wderived_flag is None:
            tiered_list = [list(self.vf_basis)]

            def derive_extension(tieredList, obstruction=None):
                baseL = list(tieredList[0])
                flattenedTL = sum(tieredList, [])
                newTeir = []
                topLevel = list(tieredList[-1])
                obstr = obstruction if obstruction else wedge(*flattenedTL)
                for vf1 in baseL:
                    for vf2 in topLevel:
                        nb = LieDerivative(vf1, vf2)
                        if find_polynomial_spanners is True:
                            nb = nb.scale_to_polynomial_attempt(factor=True)
                        new_obs = obstr * nb if use_numeric else simplify(obstr * nb)
                        if use_numeric:
                            if zeroish(new_obs):
                                continue
                        elif new_obs == 0 or getattr(new_obs, "is_zero", False):
                            continue
                        obstr = new_obs
                        newTeir.append(nb)
                return list(tieredList) + [newTeir], obstr

            obstr = None
            for _ in range(max_iterations):
                tiered_list, obstr = derive_extension(tiered_list, obstr)
                if len(tiered_list[-1]) == 0:
                    tiered_list = tiered_list[:-1]
                    break
            self._wderived_flag = tiered_list
        return self._wderived_flag

    def nilpotent_approximation(
        self,
        approximation_point=None,
        label=None,
        basis_labels=None,
        exclude_from_VMF=False,
        return_created_object=True,
        randomize_approximation_point=False,
        use_numeric_methods=False,
        **kwargs,
    ):
        if randomize_approximation_point:
            from random import randint

            approximation_point = dict()
            for var in self.coordinates:
                in1 = randint(1, 20)
                in2 = randint(in1 + 1, in1 + 20)
                ins = [in1, in2]
                idx = randint(0, 1)
                approximation_point[var] = rational(ins[idx], ins[1 - idx])
            # Add plain text printing
            from .._aux.printing.printing._dgcv_display import (
                LaTeX_eqn_system,
                show,
            )

            print("Evaluating nilpotent approximation at the randomly chosen point:")
            show(LaTeX_eqn_system(approximation_point, one_line=True))
        approximation_point = kwargs.get(
            "expansion_point", approximation_point
        )  # old syntax support
        if approximation_point is None:
            approximation_point = {var: 0 for var in self.coordinates}

        dimension = len(self.coordinates)
        derFlag = self.weak_derived_flag(use_numeric_methods=use_numeric_methods)
        evaluated_flag = [
            list([subs(vf, approximation_point) for vf in level]) for level in derFlag
        ]
        evaluated_basis = _extract_basis_by_wedge_vectorized(
            sum(evaluated_flag, []), use_numeric_methods=use_numeric_methods
        )
        depth = len(derFlag)
        basisVF = sum(derFlag, [])

        discrep = len(self.coordinates) - len(evaluated_basis)
        if discrep > 0:
            dgcv_warning(
                f"The distribution is not bracket generating or the expansion point is a growth-vector singularity singularity (note: currently `dgcv.distribution` methods are not intended for analysis at such singularities). A complement to its bracket-generated envelope has been assigned weight {-depth} and added to the nilpotent approximation as a component commuting with everything."
            )
        elif discrep < 0:  # old logic, never happens; refactor reminder
            raise TypeError(
                f"The distribution is singular at the point {approximation_point}. Nilpotent approximations are not yet supported for singular distributions."
            )
        vlabel = create_key("var")
        vars = [symbol(f"{vlabel}{j}") for j in range(len(evaluated_basis))]
        gen_elem = sum(coef * elem for coef, elem in zip(vars, evaluated_basis))

        def _decomp(elem, ge=gen_elem, variables=vars):
            eqns = list((elem - ge).coeff_dict.values())
            sol = solve_dgcv(eqns, variables)
            return sol

        level_dimensions = [len(level) for level in derFlag]

        def i_to_w_rule(idx):
            cap = 0
            for level, ld in enumerate(level_dimensions):
                cap += ld
                if idx < cap:
                    return -1 - level
            return -depth

        idx_to_weight_assignment = {j: i_to_w_rule(j) for j in range(dimension)}
        grading_vec = [idx_to_weight_assignment[idx] for idx in range(dimension)]
        VFC_enum = list(enumerate(basisVF))
        algebra_data = dict()
        for count1, elem1 in VFC_enum:
            for count2, elem2 in VFC_enum[count1 + 1 :]:
                newLevelWeight = (
                    idx_to_weight_assignment[count1] + idx_to_weight_assignment[count2]
                )
                if newLevelWeight < -depth:
                    coeffs = [0] * len(self.coordinates)
                else:
                    eqns = subs(LieDerivative(elem1, elem2), approximation_point)
                    coeff_sol = _decomp(eqns)
                    if len(coeff_sol) == 0:
                        raise RuntimeError(
                            "failed to extract algebra structure during nilpotent approximation."
                        )
                    coeffs = [
                        coeff_sol[0].get(var, var)
                        if idx_to_weight_assignment[idx] == newLevelWeight
                        else 0
                        for idx, var in enumerate(vars)
                    ] + ([0] * discrep)
                algebra_data[(count1, count2)] = coeffs
                algebra_data[(count2, count1)] = [-j for j in coeffs]
        if label is None:
            if basis_labels is not None:
                dgcv_warning(
                    "`basis_labels` was provided but no `label` was provided; `basis_labels` is ignored."
                )
            printWarning = (
                "This algebra was initialized via `distribution.nilpotent_approximation` with no label; "
                "automatic labels were assigned. Provide `label=...` (and optionally `basis_labels=...`) to control labeling, "
                "or use exclude_from_VMF=True to suppress warnings."
            )
            childPrintWarning = (
                "This algebraElement's parent algebra was initialized via `distribution.nilpotent_approximation` with no label; "
                "automatic labels were assigned."
            )
            exclusionPolicy = retrieve_passkey() if exclude_from_VMF is True else None
            return algebra_class(
                algebra_data,
                grading=[grading_vec],
                assume_skew=True,
                _callLock=retrieve_passkey(),
                _print_warning=printWarning,
                _child_print_warning=childPrintWarning,
                _exclude_from_VMF=exclusionPolicy,
            )

        return createAlgebra(
            algebra_data,
            label,
            basis_labels=basis_labels,
            grading=[grading_vec],
            assume_skew=True,
            return_created_object=return_created_object,
            forgo_vmf_registry=exclude_from_VMF,
        )

    def contains_section(
        self,
        section,
        section_parameters: list = None,
        linear_section_parameters: bool = False,
    ):
        if query_dgcv_categories(section, {"vector_field"}):
            bas = self.vf_basis
        elif query_dgcv_categories(section, {"differential_form"}):
            bas = self.df_basis
        if len(bas) == 0 and section_parameters is None:
            return section.is_zero
        indepcheck = decompose(
            section,
            bas,
            assume_basis=True,
            variables_to_constrain=section_parameters,
            assume_VTC_linear=linear_section_parameters,
        )
        if section_parameters is None:
            return len(indepcheck[0]) != 0
        out = len(indepcheck[0]) != 0 and indepcheck[2]
        if isinstance(out, bool):  # DEBUG
            from .._aux._utilities._config import get_globals

            get_globals()["DEBUG"] = (
                section,
                bas,
                section_parameters,
                linear_section_parameters,
            )
        return out

    @property
    def _ext_power_class(self):
        if self._ext_power_class_cache is None:
            self._ext_power_class_cache = simplify(wedge(*self.vf_basis))
        return self._ext_power_class_cache

    @property
    def characteristic(self):
        if self._characteristic is None:
            vfs = self.vf_basis
            genVF, section_par = linear_combination(vfs)
            sp_contraints = []
            for vf in vfs:
                bracket = LieDerivative(genVF, vf)
                deco = self.contains_section(
                    bracket,
                    section_parameters=section_par,
                    linear_section_parameters=True,
                )
                sp_contraints += [k - v for k, v in deco.items()]
            sol = solve_dgcv(sp_contraints, section_par)[0]
            solution = subs(genVF, sol)
            free_vars = set()
            for val in sol.values():
                free_vars |= get_free_symbols(val)
            free_vars = set(section_par) & free_vars
            zeroing = {vari: 0 for vari in free_vars}
            char_dist = [
                subs(solution, zeroing | {vari: 1}).scale_to_polynomial_attempt()
                for vari in free_vars
            ]
            self._characteristic = char_dist
        return self._characteristic

    def __add__(self, other):
        if get_dgcv_category(other) == "distribution":
            return distribution(self.spanning_vf_set + other.spanning_vf_set)
        return NotImplemented

    def __mul__(self, other):
        if get_dgcv_category(other) == "distribution":
            lbs = [
                LieDerivative(vf1, vf2)
                for vf1 in self.vf_basis
                for vf2 in other.vf_basis
            ]
            return distribution(
                list(self.spanning_vf_set) + list(other.spanning_vf_set) + lbs,
                find_polynomial_spanners=True,
                find_basis=True,
            )
        return NotImplemented

    def __pow__(self, other):
        if not isinstance(other, numbers.Integral) or other >= 0:
            return NotImplemented
        out = distribution([])
        for _ in range(other):
            out *= self
        return out

    def __str__(self):
        reg = get_dgcv_settings_registry()
        vlp = bool(reg.get("verbose_label_printing", False))

        max_dim = 20
        vs = getattr(self, "varSpace", None) or tuple()
        fmt = getattr(self, "formatting", None)

        if getattr(self, "_prefered_data_type", 1) == 1:
            span = getattr(self, "_spanning_vf_set", None)
            if span is None:
                span = self.spanning_vf_set
        else:
            span = getattr(self, "_spanning_df_set", None)
            if span is None:
                span = self.spanning_df_set

        span = tuple(span) if span is not None else tuple()

        def _trunc(seq):
            if len(seq) <= max_dim:
                return seq
            k = max_dim // 2
            return seq[:k] + ("...",) + seq[-k:]

        core = "<" + ", ".join(str(e) for e in _trunc(span)) + ">"

        if not vlp:
            return core

        vs_core = "<" + ", ".join(str(v) for v in _trunc(vs)) + ">"
        tag = "distribution"
        if fmt in ("complex", "real"):
            tag += f"[{fmt}]"
        return f"{tag} on {vs_core}: {core}"

    def _repr_latex_(self, raw: bool = False, abbrev: bool = False, **kwargs):
        reg = get_dgcv_settings_registry()
        vlp = bool(reg.get("verbose_label_printing", False))

        max_dim = 20
        vs = getattr(self, "varSpace", None) or tuple()
        fmt = getattr(self, "formatting", None)

        if getattr(self, "_prefered_data_type", 1) == 1:
            span = getattr(self, "_spanning_vf_set", None)
            if span is None:
                span = self.spanning_vf_set
        else:
            span = getattr(self, "_spanning_df_set", None)
            if span is None:
                span = self.spanning_df_set

        span = tuple(span) if span is not None else tuple()

        def _trunc(seq):
            if len(seq) <= max_dim:
                return seq
            k = max_dim // 2
            return seq[:k] + (r"\dots",) + seq[-k:]

        def _tex(obj):
            if get_dgcv_settings_registry().get("compile_latex_conjugation", True):
                f = getattr(
                    symToHol(obj, convert_everything=False), "_repr_latex_", None
                )
            else:
                f = getattr(obj, "_repr_latex_", None)
            if callable(f):
                s = f(raw=True)
                return str(s).replace("$", "").replace(r"\displaystyle", "")
            return str(obj)

        if abbrev:
            out = r"\mathcal{D}"
            return out if raw else rf"$\displaystyle {out}$"

        inner = ", ".join(_tex(e) for e in _trunc(span))
        core = rf"\left\langle {inner}\right\rangle"

        if not vlp:
            out = core
            return out if raw else rf"$\displaystyle {out}$"

        vs_inner = ", ".join(_tex(v) for v in _trunc(vs))
        vs_core = rf"\left\langle {vs_inner}\right\rangle"

        tag = r"\mathcal{D}"
        if fmt == "real":
            tag = r"\mathcal{D}_{\mathbb{R}}"
        elif fmt == "complex":
            tag = r"\mathcal{D}_{\mathbb{C}}"

        out = rf"{tag}\ \text{{on}}\ {vs_core}:\ {core}"
        return out if raw else rf"$\displaystyle {out}$"

    def _latex(self, printer=None, raw: bool = True, **kwargs):
        return self._repr_latex_(raw=raw, **kwargs)

    def apply(self, operator, *args, **kwds):
        return distribution(
            [operator(vf, *args, **kwds) for vf in self.vf_basis],
            formatting=self.formatting,
        )

    def __dgcv_apply__(self, operator):
        return self.apply(operator)

    def __dgcv_conjugate__(self, symbolic=False):
        return self.apply(conjugate, symbolic=symbolic)


class filtration_class(dgcv_class):
    def __init__(
        self,
        spanning_vf_sets: list[list],
        assume_spanning_sections_linearly_indep=False,
    ):
        def sublist_check(sl):
            return isinstance(sl, (list, set, tuple)) and all(
                query_dgcv_categories(vf, {"vector_field"}) for vf in sl
            )

        if not isinstance(spanning_vf_sets, (list, set, tuple)) or not all(
            sublist_check(sl) for sl in spanning_vf_sets
        ):
            raise TypeError(
                "`filtration init expects `spanning_vf_sets` to be a list of lists of vector fields.`"
            )
        spanning_vf_sets = list(spanning_vf_sets)
        self.depth = len(spanning_vf_sets)
        if len(spanning_vf_sets) == 0:
            spanning_vf_sets = [[]]
        distros = [
            distribution(
                spanning_vf_set=spanning_vf_sets[0],
                assume_spanning_sections_linearly_indep=assume_spanning_sections_linearly_indep,
            )
        ]
        for sl in spanning_vf_sets[1:]:
            distros.append(
                distribution(
                    list(distros[-1].vf_basis) + list(sl),
                    assume_spanning_sections_linearly_indep=assume_spanning_sections_linearly_indep,
                )
            )
        self.distributions = tuple(distros)
        self.growth_vector = tuple(distro.rank for distro in self.distributions)
        self._frame_torsion = None
        self._graded_frame_torsion = None
        self._singularities = dict()
        super().__init__()

    @property
    def vf_basis(self):
        return self.distributions[-1].vf_basis if self.depth > 0 else []

    @property
    def associated_graded_bases(self):
        vfb = self.vf_basis
        out = [[]]
        level = 0
        for idx, vf in enumerate(vfb):
            if idx < self.growth_vector[level]:
                out[-1].append(vf)
                continue
            gap = min(
                gvidx - level
                for gvidx in range(self.depth)
                if self.growth_vector[gvidx] > self.growth_vector[level]
            )
            level += gap
            if gap > 1:
                out += [[] for _ in range(gap - 1)]
            out.append([vf])
        return out

    @property
    def frame_torsion(self):
        if self._frame_torsion is None:
            vfb = self.vf_basis
            dim = len(vfb)
            ft = array_dgcv(
                dict(),
                shape=(dim, dim),
                null_return=freeze_matrix(matrix_dgcv.zeros(dim, 1)),
            )
            for c1, vf1 in enumerate(vfb):
                for c, vf2 in enumerate(vfb[c1 + 1 :]):
                    c2 = c1 + 1 + c
                    bracket = LieDerivative(vf1, vf2)
                    coeffs = decompose(bracket, vfb, assume_basis=True)[0]
                    if len(coeffs) != dim:
                        print(type(bracket))
                        raise ValueError(
                            f"The filtration's largest level is not involutive. VFs in indices {c1} and {c2} fail bracket closure."
                        )
                    coeffs = matrix_dgcv(coeffs)
                    ft[c1, c2] = coeffs
                    ft[c2, c1] = -coeffs
            self._frame_torsion = ft
        return self._frame_torsion

    @property
    def associated_graded_frame_torsion(self):
        if self._graded_frame_torsion is None:
            ft = self.frame_torsion
            dim = ft.shape[0]
            nft = array_dgcv(
                dict(),
                shape=(dim, dim),
                null_return=freeze_matrix(matrix_dgcv.zeros(dim, 1)),
            )

            def find_level(idx):
                for level, ldim in enumerate(self.growth_vector):
                    if idx < ldim:
                        return level + 1
                return self.depth

            def trim(clist, level, c1, c2):
                if level > self.depth:
                    return matrix_dgcv({}, shape=(dim, 1))
                l_idx = level - 1
                if l_idx == 0:
                    ld, ld_inc = 0, self.growth_vector[l_idx]
                else:
                    ld, ld_inc = (
                        self.growth_vector[l_idx - 1],
                        self.growth_vector[l_idx],
                    )
                part2, part3 = clist[ld:ld_inc], clist[ld_inc:]
                if any(coef != 0 for coef in part3):
                    if any(simplify(coef) != 0 for coef in part3):
                        raise ValueError(
                            f"The filtration is not compatible with Lie brackets, i.e., [F_i,F_j] is not in F_{{i+j}} for i={-find_level(c1)} and j={-find_level(c2)}. The problem occurs at index pair {(c1, c2)}."
                        )
                out = matrix_dgcv(
                    {idx + ld: coef for idx, coef in enumerate(part2)}, shape=(dim, 1)
                )
                return out

            for k, v in ft._data.items():
                c1, c2 = ft._unspool(k)
                if c1 > c2:
                    continue
                nc = trim(v, find_level(c1) + find_level(c2), c1, c2)
                nft[c1, c2] = nc
                nft[c2, c1] = -nc
            self._graded_frame_torsion = nft
        return self._graded_frame_torsion

    def nilpotent_approximation(
        self,
        approximation_point=None,
        label=None,
        basis_labels=None,
        exclude_from_VMF: bool = None,
        return_created_object: bool = True,
        Tanaka_symbol_format: bool = False,
        randomize_approximation_point=False,
        **kwargs,
    ) -> algebra_class | Tanaka_symbol:
        if exclude_from_VMF is None:
            exclude_from_VMF = Tanaka_symbol_format
        if exclude_from_VMF or Tanaka_symbol_format:
            return_created_object = True
        agft = self.associated_graded_frame_torsion
        if randomize_approximation_point:
            coordinates = get_free_symbols(agft)
            from random import randint

            approximation_point = dict()
            for var in coordinates:
                in1 = randint(1, 20)
                in2 = randint(in1 + 1, in1 + 20)
                ins = [in1, in2]
                idx = randint(0, 1)
                approximation_point[var] = rational(ins[idx], ins[1 - idx])
            # Add plain text printing
            from .._aux.printing.printing._dgcv_display import (
                LaTeX_eqn_system,
                show,
            )

            print("Evaluating nilpotent approximation at the randomly chosen point:")
            show(LaTeX_eqn_system(approximation_point, one_line=True))
        if approximation_point is not None:
            if not isinstance(approximation_point, dict):
                dgcv_warning(
                    "approximation_point was given in an unsuported format, and was ignored."
                )
            else:
                agft = subs(agft, approximation_point)
        grading = []
        for level, ldim in enumerate(self.growth_vector):
            if level > 0:
                ldim = ldim - self.growth_vector[level - 1]
            grading += [-level - 1 for _ in range(ldim)]
        if label is None:
            if basis_labels is not None:
                dgcv_warning(
                    "`basis_labels` was provided but no `label` was provided; `basis_labels` is ignored."
                )
            printWarning = (
                "This algebra was initialized via `filtration_class.nilpotent_approximation` with no label; "
                "automatic labels were assigned. Provide `label=...` (and optionally `basis_labels=...`) to control labeling, "
                "or use exclude_from_VMF=True to suppress warnings."
            )
            childPrintWarning = (
                "This algebraElement's parent algebra was initialized via `filtration_class.nilpotent_approximation` with no label; "
                "automatic labels were assigned."
            )
            exclusionPolicy = retrieve_passkey() if exclude_from_VMF is True else None
            outalg = algebra_class(
                agft,
                grading=[grading],
                assume_skew=True,
                _callLock=retrieve_passkey(),
                _print_warning=printWarning,
                _child_print_warning=childPrintWarning,
                _exclude_from_VMF=exclusionPolicy,
            )
        else:
            outalg = createAlgebra(
                agft,
                label,
                basis_labels=basis_labels,
                grading=[grading],
                assume_skew=True,
                return_created_object=True,
                forgo_vmf_registry=exclude_from_VMF,
            )
        if return_created_object:
            if Tanaka_symbol_format:
                return Tanaka_symbol(outalg)
            return outalg
