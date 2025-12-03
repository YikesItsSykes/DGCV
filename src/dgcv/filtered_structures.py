import copy
import numbers
import warnings
from collections.abc import Iterable
from html import escape as _esc

import sympy as sp

from ._config import get_dgcv_settings_registry, latex_in_html
from ._safeguards import (
    _cached_caller_globals,
    create_key,
    get_dgcv_category,
    query_dgcv_categories,
    retrieve_passkey,
)
from ._tables import build_plain_table
from .algebras.algebras_core import (
    _extract_basis,
    _indep_check,
    algebra_class,
    algebra_subspace_class,
)
from .algebras.algebras_secondary import createAlgebra, subalgebra_class
from .backends._caches import _get_expr_num_types
from .dgcv_core import VF_coeffs, allToHol, allToReal, changeVFBasis, variableProcedure
from .solvers import solve_dgcv
from .styles import get_style
from .vector_fields_and_differential_forms import LieDerivative, annihilator, decompose
from .vmf import clearVar, listVar


class Tanaka_symbol:
    """
    dgcv class representing a symbol-like object for Tanaka prolongation.

    Parameters
    ----------

    Methods
    -------
    prolong

    Examples
    --------
    """

    def __init__(
        self,
        GLA,
        nonnegParts=[],
        assume_FGLA=False,
        subspace=None,
        distinguished_subspaces=None,
        validate_aggresively=False,
        index_threshold=None,
        precompute_generators=False,
        _validated=None,
        _internal_parameters={},
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
                    isinstance(new_threshold, _get_expr_num_types())
                    or new_threshold is None
                ):
                    raise TypeError("index_threshold must be an integer or None.")
                self.index_threshold = new_threshold

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
            if len(nonnegParts) != 0:
                if (
                    isinstance(GLA.grading[0], (list, tuple))
                    and max(GLA.grading[0]) >= 0
                ):
                    raise TypeError(
                        "While `Tanaka_symbol` supports two syntax formats for encoding non-negative weighted components, they cannot be combined. Either `GLA.grading` should have only non-negative weights or no value for the optional `nonnegParts` parameter should be given."
                    )
            if isinstance(nonnegParts, dict):
                NNPList = list(nonnegParts.values())
            elif isinstance(nonnegParts, (list, tuple)):
                NNPList = [nonnegParts]
            else:
                raise TypeError(
                    "`Tanaka_symbol` expects `nonnegParts` to be a list of `tensorProduct` built from the `algebra` given for `GLA` with `valence` of the form (1,0,...,0). Or it can be a dictionary whose keys are non-negative weights, and whose key-values are such lists."
                )
            for NNP in NNPList:
                if not all(
                    get_dgcv_category(j) == "tensorProduct" for j in NNP
                ) or not all(j.vector_space == GLA for j in NNP):
                    raise TypeError(
                        "`Tanaka_symbol` expects `nonnegParts` to be a list of `tensorProduct` instances built from the `algebra` given for `GLA` with `valence` of the form (1,0,...,0). Or it can be a dictionary whose keys are non-negative weights, and whose key-values are such lists."
                    )
            if not all(valence_check(j) for j in nonnegParts):
                raise TypeError(
                    "`Tanaka_symbol` expects `nonnegParts` to be a list of `tensorProduct` instances built from the `algebra` given for `GLA` with `valence` of the form (1,0,...0)."
                )
        else:
            if isinstance(nonnegParts, dict):
                NNPList = list(nonnegParts.values())
            elif isinstance(nonnegParts, (list, tuple)):
                NNPList = [nonnegParts]
        self._parameters = GLA._parameters
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
            subIndices = []
            filtered_grading = []
            truncateIndices = {}
            nonnegPartsTemp = {}
            for count, weight in enumerate(primary_grading):
                if weight < 0:
                    truncateIndices[count] = len(subIndices)
                    subIndices.append(count)
                    filtered_grading.append(weight)
                else:
                    nonnegPartsTemp[weight] = nonnegPartsTemp.get(weight, []) + [
                        GLA.basis[count]
                    ]
            if len(nonnegPartsTemp) > 0:

                def truncateBySubInd(li):
                    return [li[j] for j in subIndices]

                ###!!! generalize for vector space GLA
                structureData = truncateBySubInd(GLA.structureData)
                structureData = [truncateBySubInd(plane) for plane in structureData]
                structureData = [
                    [truncateBySubInd(li) for li in plane] for plane in structureData
                ]
                subspace = subalgebra_class(
                    truncateBySubInd(GLA.basis),
                    GLA,
                    grading=[filtered_grading],
                    _compressed_structure_data=structureData,
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
            negative_basis = []
            filtered_grading = []
            nonnegPartsTemp = {}
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
                            "`Tanaka_symbol` expects `subpsace` if given to be have the subspace within it spanned by its negatively graded elements be closed under Lie brackets."
                        )
                    if raiseWarning is True:
                        warnings.warn(
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
                warnings.warn(
                    "The `GLA` or `subspace` parameter provided to `Tanaka_symbol` has nonnegatively weighted components. If providing such `GLA` or `subspace` data then the optional `nonnegParts` cannot be manually set. So the provided manual setting for `nonnegParts` is being ignored."
                )
            nonnegParts = nonnegPartsTemp
            for w in nonnegParts.keys():
                nonnegParts[w] = [
                    _GAE_to_hom_formatting(j, subspace, test_weights=[primary_grading])
                    for j in nonnegParts[w]
                ]

        if distinguished_subspaces and _validated != retrieve_passkey():
            total_basis = list(subspace.basis) + sum(NNPList, [])
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
                    if not (isinstance(subS, (list, tuple)) or get_dgcv_category(subS) == "algebra_subspace"):
                        raise TypeError(
                            "`Tanaka_symbol` expects `distinguished_subspaces` to be a list of lists of algebra_element_class instances or tensor products belonging to the provided basis of the symbol."
                        )
                    DSList = []
                    for elem in subS:
                        reformElem, weights = _GAE_to_hom_formatting(
                            elem,
                            subspace,
                            test_weights=[primary_grading],
                            return_weights=True,
                        )
                        DSList.append(reformElem)
                        if process == "fast" and reformElem in total_basis:
                            subSLevels[weights[0]] = subSLevels.get(weights[0], []) + [reformElem]
                            if idx_cap is None or idx_cap < weights[0]:
                                idx_cap = weights[0]
                        elif process != "slow" and len(weights) == 1:
                            process = "standard"
                            subSLevels[weights[0]] = subSLevels.get(weights[0], []) + [reformElem]
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
                        var_pre = create_key(prefix="var")
                        rich_dict = dynamic_dict(dict(), initial_index=idx_cap + 1)
                        for k, v in subSLevels.items():
                            rich_dict[k] = dict()
                            rich_dict[k]["vars"] = [sp.Symbol(f"{var_pre}{j}") for j in range(len(v))]
                            terms_list = [var * elem for var, elem in zip(rich_dict[k]["vars"], v)]
                            rich_dict[k]["element"] = sum(terms_list[1:], terms_list[0])
                            rich_dict[k]["spanners"] = v
                        _standard_process_DS.append(rich_dict)
                    else:
                        var_pre = create_key(prefix="var")
                        rich_dict = dict()
                        rich_dict["vars"] = [sp.Symbol(f"{var_pre}{j}") for j in range(len(DSList))]
                        terms_list = [var * elem for var, elem in zip(rich_dict["vars"], DSList)]
                        rich_dict["element"] = sum(terms_list)
                        rich_dict["spanners"] = DSList
                        _slow_process_DS.append(rich_dict)
            self._fast_process_DS = _fast_process_DS
            self._standard_process_DS = _standard_process_DS
            self._slow_process_DS = _slow_process_DS
        else:
            newDS = [[]]
            self._fast_process_DS = []
            self._standard_process_DS = []
            self._slow_process_DS = []
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
                warnings.warn(
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
    ):  # height must match levels structure
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
        elif (self._GLA_generators is not None and all(len(levels[height - j]) == 0 for j in range(-min(self._GLA_generators.get("generators", levels)))) and min(self._GLA_generators.get("generators", levels)) >= -1 - height):
            new_levels = levels
            new_levels._set_index_thr(height)
            stable = True
        elif min(j for j in levels) >= -1 - height and all(len(levels[height - j]) == 0 for j in range(-min(j for j in levels))):  # stability check
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
                        _fast_tensor_products(k) @ j     # removed .dual() for fast algorithm
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
                        _fast_tensor_products(k) @ j   # removed .dual() for fast algorithm
                        for j in comp
                        for k in levels[height + 1 + weight]
                        if height + 1 + weight > distinguished_s_weight_bound
                        or fast_validate_for_DS(k, j, height + 1 + weight, weight)
                    ]

                def _iter_expand(elem, nested):
                    if isinstance(nested, list):
                        return _iter_expand(_iter_expand(elem, nested[0]), nested[1]) + _iter_expand(nested[0], _iter_expand(elem, nested[1]))
                    return elem * nested
                def _complete(elem):
                    new_terms = []
                    for w, comp in self._GLA_generators["map"].items():
                        if w == -1:
                            continue
                        for trip in comp:
                            if trip[2] > 1:
                                new_terms.append(
                                    _fast_tensor_products(_iter_expand(elem, trip[0])) @ trip[1]    # removed .dual() for fast algo
                                )
                    return sum(new_terms, elem)

                ambient_basis = [_complete(j) for j in preBasis]

            ###!!! add fast validation for nonnegative weight DS components
            for subSData in standard_DS:
                if len(ambient_basis) == 0:
                    break
                vLab = create_key(prefix="vLab")
                ambVars = [sp.Symbol(f"{vLab}{j}") for j in range(len(ambient_basis))]
                ds_terms = [var * elem for var, elem in zip(ambVars, ambient_basis)]
                ambGE = sum(ds_terms)
                for w, level in subSData.items():
                    if height + w + 1 <= distinguished_s_weight_bound:
                        dsGE = subSData[height + w + 1]["element"]
                        dsVars = subSData[height + w + 1]["vars"]
                        eqns = []
                        esVars = ambVars.copy()
                        for count, elem in enumerate(level["spanners"]):
                            newVars = [
                                sp.Symbol(f"_{count}{vLab}{j}")
                                for j in range(len(dsVars))
                            ]
                            esVars += newVars
                            eqn = ambGE * elem + (dsGE.subs(dict(zip(dsVars, newVars))))
                            eqns.append(eqn)
                        sol = solve_dgcv(eqns, esVars)
                        ambient_basis = []
                        if len(sol) > 0:
                            solGE = ambGE.subs(sol[0])
                            freeVars = set()
                            for c in solGE.coeffs:
                                freeVars |= getattr(c, "free_symbols", set())
                            zeroing = {var: 0 for var in freeVars}
                            for var in freeVars:
                                ambient_basis.append(solGE.subs({var: 1}).subs(zeroing))
            if len(self._slow_process_DS) > 0:
                warnings.warn(
                    "At least one of the distinguished subspaces was given by a spanning set of elements containing some element that is not weighted-homogeneous. The algorithm for preserving subspaces in such a format is not yet implemented in this version of `dgcv`, so the subspace is being disregarding."
                )

            if len(ambient_basis) == 0:
                ambient_basis = [0 * self.basis[0]]

            varLabel = create_key(prefix="center_var")  # label for temparary variables
            tVars = [sp.Symbol(f"{varLabel}{j}") for j in range(len(ambient_basis))]
            general_elem = sum([tVars[j] * ambient_basis[j] for j in range(len(tVars))])


            eqns = []
            for triple in self.test_commutators:
                derivation_rule = (general_elem * (triple[0])) * triple[1] + triple[0] * (general_elem * (triple[1])) - general_elem * (triple[2])
                if getattr(derivation_rule, "is_zero", False) or derivation_rule == 0:
                    continue
                if get_dgcv_category(derivation_rule) in {"fastTensorProduct","tensorProduct"}:
                    eqns += list(derivation_rule.coeff_dict.values())
                elif get_dgcv_category(derivation_rule) in {"algebra_element","subalgebra_element","vector_space_element"}:
                    eqns += derivation_rule.coeffs

            if eqns == [0] or eqns == []:
                solution = [{}]
            else:
                solution = solve_dgcv(eqns, tVars)

            if len(solution) == 0:
                raise RuntimeError(
                    f"`Tanaka_symbol.prolongation` failed at a step where a symbolic solver (e.g., sympy.solve if using the default sympy) was being applied. The equation system was {eqns} w.r.t. {tVars}; return solution data was {solution}"
                )
            el_sol = general_elem.subs(solution[0])
            if not isinstance(el_sol,_fast_tensor_products):
                el_sol = _fast_tensor_products(el_sol)

            free_variables = tuple(var for var in set.union(set(),*[getattr(j, "free_symbols", set()) for j in el_sol.coeff_dict.values()]) if var not in self._parameters)

            new_level = []
            for var in free_variables:
                basis_element = el_sol.subs({var: 1}).subs([(other_var, 0) for other_var in free_variables if other_var != var])
                new_level.append(basis_element)

            if ADS is True:
                new_elems = []
                for subS in fast_DS:
                    if height + 1 in subS:
                        new_elems += copy.deepcopy(subS[height + 1])
                        subS[height + 1] = _extract_basis(subS[height + 1] + new_level)
                for subSData in standard_DS:
                    if height + 1 in subSData:
                        new_elems += copy.deepcopy(subSData[height + 1]["spanners"])
                        subSData[height + 1]["spanners"] = _extract_basis(
                            subSData[height + 1]["spanners"] + new_level
                        )
                new_level = _extract_basis(new_level + list(new_elems))

            if with_characteristic_space_reductions is True and height == -1:
                z_level = new_level
            else:
                z_level = levels[0]
            if (len(new_level) > 0 and with_characteristic_space_reductions is True and len(z_level) > 0):
                stabilized = False
                while stabilized is False:
                    ambient_basis = new_level
                    varLabel = create_key(prefix="_cv")
                    tVars = [sp.Symbol(f"{varLabel}{j}") for j in range(len(ambient_basis))]
                    solVars = list(tVars)
                    general_elem = sum([tVars[j] * ambient_basis[j] for j in range(len(tVars))])
                    eqns = []
                    for idx, dzElem in enumerate(z_level):
                        varLabel2 = varLabel + f"{idx}_"
                        vars2 = [
                            sp.Symbol(f"{varLabel2}{j}")
                            for j in range(len(ambient_basis))
                        ]
                        solVars += vars2
                        general_elem2 = sum([vars2[j] * ambient_basis[j] for j in range(len(tVars))])._convert_to_tp()

                        commutator = general_elem._convert_to_tp() * dzElem._convert_to_tp() - general_elem2
                        if get_dgcv_category(commutator) == "tensorProduct":
                            eqns += list(commutator.coeff_dict.values())
                        elif get_dgcv_category(commutator) == "algebra_element_class":
                            eqns += commutator.coeffs
                    # eqns = list(set(eqns))
                    solution = solve_dgcv(eqns, solVars)
                    if len(solution) == 0:
                        raise RuntimeError(
                            f"`Tanaka_symbol.prolongation` failed at a step where sympy.linsolve was being applied. The equation system was {eqns} w.r.t. {solVars}"
                        )
                    solCoeffs = [j.subs(solution[0]) for j in tVars]

                    free_variables = tuple(set.union(set(), *[set(sp.sympify(j).free_symbols) for j in solCoeffs]))
                    new_vectors = []
                    zeroingDict = {other_var: 0 for other_var in free_variables}
                    for var in free_variables:
                        basis_element = [j.subs({var: 1}).subs(zeroingDict) for j in solCoeffs]
                        new_vectors.append(basis_element)
                    columns = (sp.Matrix(new_vectors).T).columnspace()
                    filtered_vectors = [list(sp.nsimplify(col, rational=True)) for col in columns]

                    def reScale(vec):
                        denom = 1
                        for t in vec:
                            if hasattr(t, "denominator") and denom < t.denominator:
                                denom = t.denominator
                        if denom == 1:
                            return list(vec)
                        else:
                            return [t * denom for t in vec]

                    filtered_vectors = [reScale(vec) for vec in filtered_vectors]

                    new_basis = []
                    for coeffs in filtered_vectors:
                        new_basis.append(sum([coeffs[j] * ambient_basis[j] for j in range(len(ambient_basis))]))
                    if len(new_basis) == 0:
                        new_level = []
                        stabilized = True
                    elif len(new_basis) < len(new_level):
                        new_level = new_basis
                    else:
                        new_level = new_basis
                        stabilized = True
            new_levels = self._GLA_structure(levels | {height + 1: new_level}, levels.index_threshold)
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
    ):  # height must match levels structure
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
                vLab = create_key(prefix="vLab")
                ambVars = [sp.Symbol(f"{vLab}{j}") for j in range(len(ambient_basis))]
                ds_terms = [var * elem for var, elem in zip(ambVars, ambient_basis)]
                ambGE = sum(ds_terms)
                for w, level in subSData.items():
                    if height + w + 1 <= distinguished_s_weight_bound:
                        dsGE = subSData[height + w + 1]["element"]
                        dsVars = subSData[height + w + 1]["vars"]
                        eqns = []
                        esVars = ambVars.copy()
                        for count, elem in enumerate(level["spanners"]):
                            newVars = [
                                sp.Symbol(f"_{count}{vLab}{j}")
                                for j in range(len(dsVars))
                            ]
                            esVars += newVars
                            eqn = ambGE * elem + (dsGE.subs(dict(zip(dsVars, newVars))))
                            eqns.append(eqn)
                        sol = solve_dgcv(eqns, esVars)
                        ambient_basis = []
                        if len(sol) > 0:
                            solGE = ambGE.subs(sol[0])
                            freeVars = set()
                            for c in solGE.coeffs:
                                freeVars |= getattr(c, "free_symbols", set())
                            zeroing = {var: 0 for var in freeVars}
                            for var in freeVars:
                                ambient_basis.append(solGE.subs({var: 1}).subs(zeroing))
            if len(self._slow_process_DS) > 0:
                warnings.warn(
                    "At least one of the distinguished subspaces was given by a spanning set of elements containing some element that is not weighted-homogeneous. The algorithm for preserving subspaces in such a format is not yet implemented in this version of `dgcv`, so the subspace is being disregarding."
                )

            if len(ambient_basis) == 0:
                ambient_basis = [0 * self.basis[0]]

            varLabel = create_key(prefix="center_var")  # label for temparary variables
            tVars = [sp.Symbol(f"{varLabel}{j}") for j in range(len(ambient_basis))]
            general_elem = sum([tVars[j] * ambient_basis[j] for j in range(len(tVars))])

            eqns = []
            for triple in self.test_commutators:
                derivation_rule = (general_elem * (triple[0])) * triple[1] + triple[0] * (general_elem * (triple[1])) - general_elem * (triple[2])
                if getattr(derivation_rule, "is_zero", False) or derivation_rule == 0:
                    continue
                if get_dgcv_category(derivation_rule) == "tensorProduct":
                    eqns += list(derivation_rule.coeff_dict.values())
                elif get_dgcv_category(derivation_rule) in {
                    "algebra_element",
                    "subalgebra_element",
                    "vector_space_element",
                }:
                    eqns += derivation_rule.coeffs

            if eqns == [0] or eqns == []:
                solution = [{}]
            else:
                solution = solve_dgcv(eqns, tVars)

            if len(solution) == 0:
                raise RuntimeError(
                    f"`Tanaka_symbol.prolongation` failed at a step where a symbolic solver (e.g., sympy.solve if using the default sympy) was being applied. The equation system was {eqns} w.r.t. {tVars}; return solution data was {solution}"
                )
            el_sol = general_elem.subs(solution[0])
            if hasattr(el_sol, "_convert_to_tp"):
                el_sol = el_sol._convert_to_tp()

            free_variables = tuple(
                var
                for var in set.union(
                    set(),
                    *[
                        getattr(j, "free_symbols", set())
                        for j in el_sol.coeff_dict.values()
                    ],
                )
                if var not in self._parameters
            )

            new_level = []
            for var in free_variables:
                basis_element = el_sol.subs({var: 1}).subs(
                    [(other_var, 0) for other_var in free_variables if other_var != var]
                )
                new_level.append(basis_element)

            if ADS is True:
                new_elems = []
                for subS in fast_DS:
                    if height + 1 in subS:
                        new_elems += copy.deepcopy(subS[height + 1])
                        subS[height + 1] = _extract_basis(subS[height + 1] + new_level)
                for subSData in standard_DS:
                    if height + 1 in subSData:
                        new_elems += copy.deepcopy(subSData[height + 1]["spanners"])
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
                    tVars = [sp.Symbol(f"{varLabel}{j}") for j in range(len(ambient_basis))]
                    solVars = list(tVars)
                    general_elem = sum([tVars[j] * ambient_basis[j] for j in range(len(tVars))])
                    eqns = []
                    for idx, dzElem in enumerate(z_level):
                        varLabel2 = varLabel + f"{idx}_"
                        vars2 = [sp.Symbol(f"{varLabel2}{j}") for j in range(len(ambient_basis))]
                        solVars += vars2
                        general_elem2 = sum([vars2[j] * ambient_basis[j] for j in range(len(tVars))])

                        commutator = general_elem * dzElem - general_elem2
                        if get_dgcv_category(commutator) == "tensorProduct":
                            eqns += list(commutator.coeff_dict.values())
                        elif get_dgcv_category(commutator) == "algebra_element_class":
                            eqns += commutator.coeffs
                    # eqns = list(set(eqns))
                    solution = solve_dgcv(eqns, solVars)
                    if len(solution) == 0:
                        raise RuntimeError(
                            f"`Tanaka_symbol.prolongation` failed at a step where sympy.linsolve was being applied. The equation system was {eqns} w.r.t. {solVars}"
                        )
                    solCoeffs = [j.subs(solution[0]) for j in tVars]

                    free_variables = tuple(
                        set.union(
                            set(), *[set(sp.sympify(j).free_symbols) for j in solCoeffs]
                        )
                    )
                    new_vectors = []
                    zeroingDict = {other_var: 0 for other_var in free_variables}
                    for var in free_variables:
                        basis_element = [
                            j.subs({var: 1}).subs(zeroingDict) for j in solCoeffs
                        ]
                        new_vectors.append(basis_element)
                    columns = (sp.Matrix(new_vectors).T).columnspace()
                    filtered_vectors = [
                        list(sp.nsimplify(col, rational=True)) for col in columns
                    ]

                    def reScale(vec):
                        denom = 1
                        for t in vec:
                            if hasattr(t, "denominator") and denom < t.denominator:
                                denom = t.denominator
                        if denom == 1:
                            return list(vec)
                        else:
                            return [t * denom for t in vec]

                    filtered_vectors = [reScale(vec) for vec in filtered_vectors]

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
        iterations,
        return_symbol=True,
        report_progress=False,
        report_progress_and_return_nothing=False,
        with_characteristic_space_reductions=None,
        absorb_distinguished_subspaces=False,
        _fast_algorithm=True
    ):
        _cached_caller_globals['PROFILE']=[]
        if absorb_distinguished_subspaces is True:
            subspace_data = [
                copy.deepcopy(self._fast_process_DS),
                copy.deepcopy(self._standard_process_DS),
            ]
            if len(self._slow_process_DS) > 0:
                warnings.warn(
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
                if w>=0:
                    levels[w]=[_fast_tensor_products(j) for j in levels[w]]
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
                )
            else:
                levels, stable, subspace_data = self._prolong_by_1(
                    levels,
                    height,
                    distinguished_s_weight_bound=distinguished_s_weight_bound,
                    with_characteristic_space_reductions=with_characteristic_space_reductions,
                    ADS=subspace_data,
                )
            if report_progress:
                max_len = max(
                    max(len(str(weight)) for weight in levels.keys()),
                    max(len(str(len(basis))) for basis in levels.values()),
                )

                weights = " │ ".join(
                    [str(weight).ljust(max_len) for weight in levels.keys()]
                )
                dimensions = " │ ".join(
                    [str(len(basis)).ljust(max_len) for basis in levels.values()]
                )
                weights = f"Weights    │ {weights}"
                dimensions = f"Dimensions │ {dimensions}"
                line_length = max(len(weights), len(dimensions)) + 1

                header_length = len("Weights    │ ")
                top_border = f"┌{'─' * (header_length - 1)}┬{'─' * (1+line_length - header_length)}┐"
                middle_border = f"├{'─' * (header_length - 1)}┼{'─' * (1+line_length - header_length)}┤"
                bottom_border = f"└{'─' * (header_length - 1)}┴{'─' * (1+line_length - header_length)}┘"

                print(f"After {count_to_str(prol_counter)} iteration:")
                print(top_border)
                print(f"│ {weights} │")
                print(middle_border)
                print(f"│ {dimensions} │")
                print(bottom_border)
                prol_counter += 1
            height += 1
        if _fast_algorithm is True:
            for w in levels:
                if w>=0:
                    levels[w]=[j._convert_to_tp() for j in levels[w]]
        if report_progress_and_return_nothing is not True:
            if return_symbol:
                new_nonneg_parts = []
                for key, value in levels.items():
                    if key >= 0:
                        new_nonneg_parts += value
                return Tanaka_symbol(
                    self.negativePart,
                    new_nonneg_parts,
                    assume_FGLA=self.assume_FGLA,
                    distinguished_subspaces=self.distinguished_subspaces,
                    index_threshold=levels.index_threshold,
                    _validated=retrieve_passkey(),
                )
            else:
                return levels

    def summary(
        self,
        style=None,
        use_latex=None,
        display_length=500,
        table_scroll=False,
        cell_scroll=False,
    ):
        dgcvSR = get_dgcv_settings_registry()
        if dgcvSR.get("apply_awkward_workarounds_to_fix_VSCode_display_issues") is True:
            _apply_VScode_display_workaround_with_JS_deliver = True
        else:
            _apply_VScode_display_workaround_with_JS_deliver = False
        if use_latex is None:
            use_latex = dgcvSR.get("use_latex", False)
        style_key = style or dgcvSR.get("theme", "dark")

        def _to_string(e, ul=False):
            if ul:
                s = e._repr_latex_(verbose=False)
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
        for w, basis in sorted(self.levels.items(), key=lambda kv: kv[0]):
            basis_str = ", ".join(_to_string(b, ul=use_latex) for b in basis)
            if display_length is not None and len(basis_str) > display_length:
                basis_str = "output too long to display; raise `display_length` to a higher bound if needed."
            dim_here = len(basis)
            sum_computed_dimensions += dim_here
            rows.append([str(w), str(dim_here), basis_str])

        columns = ["Weight", "Dimension", "Basis"]

        footer = [
            [
                {
                    "html": f"Total dimension: {sum_computed_dimensions}",
                    "attrs": {
                        "colspan": len(columns),
                        # "style": "text-align:right; font-weight:bold;"
                    },
                }
            ]
        ]

        loc_style = get_style(style_key)

        def _get_prop(sel, prop):
            for sd in loc_style:
                if sd.get("selector") == sel:
                    for k, v in sd.get("props", []):
                        if k == prop:
                            return v
            return None

        header_bg = _get_prop("thead th", "background-color") or _get_prop(
            "th.col_heading.level0", "background-color"
        )
        header_col = _get_prop("thead th", "color") or _get_prop(
            "th.col_heading.level0", "color"
        )
        header_ff = _get_prop("thead th", "font-family") or _get_prop(
            "th.col_heading.level0", "font-family"
        )
        header_fs = _get_prop("thead th", "font-size") or _get_prop(
            "th.col_heading.level0", "font-size"
        )
        col_heading_color = _get_prop("th.col_heading.level0", "color")
        col_heading_ff = _get_prop("th.col_heading.level0", "font-family")
        col_heading_fs = _get_prop("th.col_heading.level0", "font-size")
        col_heading_bg = _get_prop("th.col_heading.level0", "background-color")

        border_val = None
        for sd in loc_style:
            if sd.get("selector") == "table":
                for k, v in sd.get("props", []):
                    if k in (
                        "border-bottom",
                        "border-right",
                        "border-left",
                        "border-top",
                        "border",
                    ):
                        border_val = v
                        break
            if border_val:
                break
        parts = (border_val or "1px solid #ccc").split()
        thickness = parts[0] if parts else "1px"
        border_color = parts[-1] if parts else "#ccc"

        # build the panel
        dsubs = getattr(self, "distinguished_subspaces", []) or []
        render_panel = not (len(dsubs) == 0 or (len(dsubs) == 1 and len(dsubs[0]) == 0))
        secondary_panel_html = None
        if render_panel:
            if use_latex:
                items = []
                for sub in dsubs:
                    labels = [(e._repr_latex_(raw=True) or "").strip() for e in sub]
                    inner = ", ".join(labels) if labels else r"\varnothing"
                    items.append(f"<li>$\\left\\langle {inner} \\right\\rangle$</li>")
            else:
                items = []
                for sub in dsubs:
                    labels = [repr(e) for e in sub]
                    inner = ", ".join(_esc(s) for s in labels) if labels else "∅"
                    items.append(f"<li>[{inner}]</li>")

            secondary_panel_html = (
                "<div class='dgcv-side-panel'>"
                "<h3>Distinguished Subspaces</h3>"
                "<hr/>"
                "<ul>" + "".join(items) + "</ul>"
                "</div>"
            )

        preface_html = "<div class='dgcv-preface'>Summary of Tanaka Symbol (with prolongations)</div>"

        extra = [
            {
                "selector": ".dgcv-table-wrap, .dgcv-side-panel",
                "props": [
                    ("box-sizing", "border-box"),
                ],
            },
            {
                "selector": ".dgcv-table-wrap",
                "props": [
                    ("display", "inline-block"),
                    # ("width", "fit-content"),
                    ("max-width", "100%"),
                    ("margin", "0"),
                    ("border", f"{thickness} solid transparent"),
                ],
            },
            {
                "selector": ".dgcv-side-panel",
                "props": [
                    ("border", f"{thickness} solid {border_color}"),
                    ("background-color", col_heading_bg or header_bg or "transparent"),
                    ("width", f"calc(100% - 2 * {thickness})"),
                    ("color", header_col or "inherit"),
                    ("padding", "4px 4px"),
                    ("margin", "0"),
                    ("overflow-y", "visible"),
                    ("height", f"calc(100% - 2 * {thickness})"),
                    ("max-height", "none"),
                ],
            },
            {
                "selector": ".dgcv-side-panel *",
                "props": [
                    ("color", header_col or "inherit"),
                ],
            },
            {
                "selector": ".dgcv-side-panel h3",
                "props": [
                    ("margin", "0"),
                    ("color", col_heading_color or header_col or "inherit"),
                    ("font-family", col_heading_ff or header_ff or "inherit"),
                    ("font-size", col_heading_fs or header_fs or "inherit"),
                    ("font-weight", "bold"),
                ],
            },
            {
                "selector": ".dgcv-side-panel hr",
                "props": [
                    ("border", "0"),
                    ("border-top", f"{thickness} solid {border_color}"),
                    ("margin", "6px 0 8px"),
                ],
            },
            {
                "selector": ".dgcv-side-panel ul",
                "props": [
                    ("margin", "8px 0 0 18px"),
                    ("padding", "0"),
                    ("overflow-wrap", "scroll"),
                    ("overflow-y", "hidden"),
                ],
            },
            {
                "selector": ".dgcv-side-panel li::marker",
                "props": [
                    ("color", col_heading_color or header_col or border_color),
                ],
            },
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "separate"),
                    ("border-spacing", "0"),
                ],
            },
            {
                "selector": "tbody tr",
                "props": [
                    ("will-change", "transform"),
                    ("transform-origin", "center"),
                    ("backface-visibility", "hidden"),
                    ("contain", "paint"),
                ],
            },
            {
                "selector": "tfoot td",
                "props": [
                    ("text-align", "left"),
                    ("font-weight", "bold"),
                    ("padding", "6px 8px"),
                    ("border-top", f"{thickness} solid {border_color}"),
                    ("background-color", col_heading_bg or header_bg or "transparent"),
                    ("color", header_col or "inherit"),
                ],
            },
            {
                "selector": ".dgcv-data-table",
                "props": [
                    ("border-collapse", "separate"),
                ],
            },
        ]

        table = build_plain_table(
            columns=columns,
            rows=rows,
            caption="",
            preface_html=preface_html,
            theme_styles=loc_style,
            extra_styles=extra,
            table_attrs='style="table-layout:fixed; overflow-x:visible;"',
            cell_align="center",
            escape_cells=False,
            escape_headers=True,
            nowrap=False,
            truncate_chars=None,
            secondary_panel_html=secondary_panel_html,
            layout="row",  # side-by-side
            gap_px=10,
            side_width="340px",
            breakpoint_px=900,  # stacked <= 900px;
            container_id="tanaka-summary",
            footer_rows=footer,
            table_scroll=table_scroll,
            cell_scroll=cell_scroll,
        )

        return latex_in_html(
            table,
            apply_VSCode_workarounds=_apply_VScode_display_workaround_with_JS_deliver,
        )

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

        top = f"┌{'─' * (header_len - 1)}┬{'─' * (1+line_len - header_len)}┐"
        middle = f"├{'─' * (header_len - 1)}┼{'─' * (1+line_len - header_len)}┤"
        bottom = f"└{'─' * (header_len - 1)}┴{'─' * (1+line_len - header_len)}┘"

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

    def __repr__(self):
        return f"Tanaka_symbol(ambientGLA={repr(self.ambientGLA)}, levels={len(self.levels)} levels)"

    def export_algebra_data(
        self, preserve_negative_part_basis=True, _internal_call_lock=None
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
            permutation = invPerm + list(
                range(self.negativePart.dimension, self.dimension)
            )

        def flatToLayered(idx):
            return indexBands[idx]

        def bracket_decomp(idx1, idx2):
            w1, sId1 = flatToLayered(idx1)
            w2, sId2 = flatToLayered(idx2)
            newElem = (self.levels[w1][sId1]) * (self.levels[w2][sId2])  ###!!! review for ambient_rep requirements
            newWeight = w1 + w2
            if self.levels[newWeight] is not None:
                ambient_basis = [j for j in self.levels[newWeight]]  ###!!! review for ambient_rep requirements
            nLDim = 0 if (ambient_basis is None or ambient_basis == [0]) else len(ambient_basis)
            if nLDim == 0:
                if getattr(newElem, "is_zero", False) or newElem == 0:
                    return [0] * dimen
                else:
                    return "NoSol"
            varLabel = create_key(prefix="_cv")
            tVars = variableProcedure(varLabel, nLDim, _tempVar=retrieve_passkey(),return_created_object=True)[0]
            general_elem = sum([tVars[j] * ambient_basis[j] for j in range(len(tVars))])
            eqns = [newElem - general_elem]
            sol = solve_dgcv(eqns, tVars)
            if len(sol) == 0:
                return "NoSol"
            coeffVec = [var.subs(sol[0]) for var in tVars]
            clearVar(*listVar(temporary_only=True), report=False)

            #   newWeight should be in complimentWeights by construction
            start = [0] * complimentWeights[newWeight][0]
            end = [0] * complimentWeights[newWeight][1]
            return start + coeffVec + end

        zeroVec = [0] * dimen
        str_data = [[bracket_decomp(k, j) if j < k else list(zeroVec) for j in range(dimen)] for k in range(dimen)]
        for j in range(dimen):
            for k in range(j + 1, dimen):
                skew_data = str_data[k][j]
                if skew_data == "NoSol":
                    warningStr = f"due to failure to confirm if the symbol data is closed under brackets between basis elements {j} and {k}."
                    if _internal_call_lock != retrieve_passkey():
                        warnings.warn(
                            "Unable to extract algebra structure, "
                            + warningStr
                            + " So `None` was returned by `export_algebra_data`."
                        )
                        return None
                    return (
                        "Unable to extract algebra structure from `Tanaka_symbol` object, "
                        + warningStr
                    )
                str_data[j][k] = [-entry for entry in skew_data]
        if preserve_negative_part_basis:

            def permute_structure_data(SD, perm):
                d = len(SD)  # len(perm) == d
                perm = list(perm)
                return [
                    [
                        [SD[perm[i]][perm[j]][perm[m]] for m in range(d)]
                        for j in range(d)
                    ]
                    for i in range(d)
                ]

            return {
                "structure_data": permute_structure_data(str_data, permutation),
                "grading": [
                    (
                        grading_vec[permutation[j]]
                        if j < len(permutation)
                        else grading_vec[j]
                    )
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
                return elem, elem.compute_weight(
                    test_weights=test_weights, _return_mixed_weight_list=True
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
    def __init__(self,coeff_dict,alg=None,_validated=None):
        if get_dgcv_category(coeff_dict)=='fastTensorProduct':
            coeff_dict,alg,_validated=coeff_dict.coeff_dict,coeff_dict.algebra,coeff_dict.degree
        self.algebra=alg
        if _validated is None:
            if get_dgcv_category(coeff_dict)=='tensorProduct':
                if alg is None:
                    self.algebra=coeff_dict.vector_space
                self.coeff_dict=dict()
                self.degree=0
                for k,v in coeff_dict.coeff_dict.items():
                    newkey=k[:len(k)//3]
                    self.coeff_dict[newkey]=v
                    self.degree=max(self.degree,len(newkey))                
            elif isinstance(coeff_dict,dict):
                self.coeff_dict=dict()
                self.degree=0
                for k,v in coeff_dict.items():
                    if v!=0 or k==tuple():
                        self.coeff_dict[k]=v
                        self.degree=max(self.degree,len(k))
            elif get_dgcv_category(coeff_dict) in {'algebra_element','subalgebra_element','vectorspace_element'}:
                if alg is None:
                    self.algebra=coeff_dict.algebra
                self.degree=1
                self.coeff_dict={(k,):v for k,v in enumerate(coeff_dict.coeffs) if v!=0}
            else:
                self.coeff_dict=dict()
        else:
            self.degree=_validated
            self.coeff_dict=coeff_dict
        if len(self.coeff_dict)==0:
            self.coeff_dict={tuple():0}
            self.degree=0
        self._dgcv_class_check=retrieve_passkey()
        self._dgcv_category='fastTensorProduct'
        self._is_zero=None
        self._coeffs=None
        if self.degree<max(len(k) for k in self.coeff_dict):
            raise TypeError('ftp init fail')
    @property
    def is_zero(self):
        if self._is_zero is None:
            self._is_zero=False if any(v!=0 for v in self.coeff_dict.values()) else True
        return self._is_zero
    @property
    def coeffs(self):
        if self._coeffs is None:
            self._coeffs=list(self.coeff_dict.values())
        return self._coeffs
    def _to_algebra(self,alg=None):
        if alg is None:
            alg=self.algebra
        ae=0
        for k,v in self.coeff_dict.items():
            if len(k)==1:
                ae+=v*alg.basis[k[0]]
        return ae

    def _convert_to_tp(self):
        from .tensors import tensorProduct
        new_dict=dict()
        for k,v in self.coeff_dict.items():
            k2=tuple(1 if idx==0 else 0 for idx in range(len(k)))
            k3=(self.algebra.dgcv_vs_id,)*len(k)
            newkey=k+k2+k3
            new_dict[newkey]=v
        return tensorProduct([],new_dict)

    def __add__(self,other):
        if other==0 or getattr(other,'is_zero',False):
            return self
        if isinstance(other,_fast_tensor_products):
            new_dict=dict(self.coeff_dict)
            deg=self.degree
            for k,v in other.coeff_dict.items():
                deg=max(len(k),deg)
                new_dict[k]=self.coeff_dict.get(k,0)+v
            return _fast_tensor_products(new_dict,self.algebra,_validated=deg)
        if get_dgcv_category(other) in {'algebra_element','subalgebra_element','vector_space_element'}:
            return self +_fast_tensor_products(other)
        return NotImplemented

    def __radd__(self,other):
        return self+other
    def __sub__(self, other):
        return (self).__add__(-other)
    def __rsub__(self, other):
        return (-self) + other
    def __mul__(self,other):
        if isinstance(other,_get_expr_num_types()):
            if other==0:
                return _fast_tensor_products({tuple():0},self.algebra,_validated=0)
            return _fast_tensor_products({k:other*v for k,v in self.coeff_dict.items()},self.algebra,_validated=self.degree)
        if isinstance(other,_fast_tensor_products):
            if other.degree==0:
                return sum(v*self for v in other.coeff_dict.values())
            if self.degree==1:
                return other*(-self._to_algebra())
            if self.degree==0:
                return sum(v*other for v in self.coeff_dict.values())
            if other.degree==1:
                return self*other._to_algebra()
            new_dict=dict()
            deg=0
            ae1=0
            ae2=0
            for k1,v1 in self.coeff_dict.items():
                if len(k1)==1:
                    ae1+=v1*self.algebra.basis[k1[0]]
                    continue
                k1L,k1A,k1B,k1T=k1[0],k1[:-1],k1[1:],k1[-1]
                for k2,v2 in other.coeff_dict.items():
                    if len(k2)==1:
                        ae2+=v2*other.algebra.basis[k2[0]]
                        continue
                    k2L,k2A,k2B,k2T=k2[0],k2[:-1],k2[1:],k2[-1]
                    if k1T==k2L:
                        newkey=k1B+k2A
                        newval=new_dict.get(newkey,0)+v1*v2
                        if newval!=0:
                            deg=max(len(newkey),deg)
                            new_dict[newkey]=newval
                        else:
                            new_dict.pop(newkey, None)
                    if k1L==k2T:
                        newkey=k2B+k1A
                        newval=new_dict.get(newkey,0)-v1*v2
                        if newval!=0:
                            deg=max(len(newkey),deg)
                            new_dict[newkey]=newval
                        else:
                            new_dict.pop(newkey, None)
            return _fast_tensor_products(new_dict,self.algebra,_validated=deg)+ae1*other+self*ae2
        if get_dgcv_category(other) in {'algebra_element','subalgebra_element','vector_space_element'}:
            if self.degree==0:
                return sum(v*other for v in self.coeff_dict.values())
            if self.degree==1:
                return self._to_algebra()*other
            new_dict = dict()
            ac=other.coeffs
            ae1=0
            for k1,v1 in self.coeff_dict.items():
                if len(k1)==1:
                    ae1+=v1*self.algebra.basis[k1[0]]
                    continue
                k1A,k1T=k1[:-1],k1[-1]
                newval=new_dict.get(k1A,0)+ac[k1T]*v1
                if newval!=0:
                    new_dict[k1A]=newval
                else:
                    new_dict.pop(k1A, None)
            if self.degree==2:
                return _fast_tensor_products(new_dict,self.algebra,_validated=self.degree-1)._to_algebra()+ae1*other
            return _fast_tensor_products(new_dict,self.algebra,_validated=self.degree-1)+ae1*other
        return NotImplemented
    def __rmul__(self,other):
        if isinstance(other,_get_expr_num_types()):
            if other==0:
                return _fast_tensor_products(dict(),self.algebra,_validated=0)
            return _fast_tensor_products({k:other*v for k,v in self.coeff_dict.items()},self.algebra,_validated=self.degree)
        if self.degree==0:
            return sum(v*other for v in self.coeff_dict.values())
        return self*(-other)
    def __neg__(self):
        return _fast_tensor_products({k:-v for k,v in self.coeff_dict.items()},self.algebra,_validated=self.degree)
    def __matmul__(self,other):
        if isinstance(other,_get_expr_num_types()):
            return self*other
        if get_dgcv_category(other) in {'algebra_element','subalgebra_element','vector_space_element'}:
            ac=other.coeffs
            new_dict=dict()
            for k,v in self.coeff_dict.items():
                for idx,c in enumerate(ac):
                    if c!=0:
                        newkey=k+(idx,)
                        newval=new_dict.get(newkey,0)+c*v
                        if newval!=0:
                            new_dict[newkey]=newval
                        else:
                            new_dict.pop(newkey, None)
            return _fast_tensor_products(new_dict,self.algebra,_validated=self.degree+1)
        if isinstance(other,_fast_tensor_products):
            ac=other.coeff_dict
            new_dict=dict()
            for k,v in self.coeff_dict.items():
                for idx,c in other.coeff_dict.items():
                    newkey=k+idx
                    newval=new_dict.get(newkey,0)+c*v
                    if newval!=0:
                        new_dict[newkey]=newval
                    else:
                        new_dict.pop(newkey, None)
            return _fast_tensor_products(new_dict,self.algebra,_validated=self.degree+other.degree)
        return NotImplemented

    def __rmatmul__(self,other):
        return self.__matmul__(other)

    def subs(self,subs_data):
        new_dict = {k:sp.sympify(v).subs(subs_data) for k,v in self.coeff_dict.items()}
        return _fast_tensor_products(new_dict,self.algebra,_validated=self.degree)

class distribution:

    def __init__(
        self,
        spanning_vf_set=None,
        spanning_df_set=None,
        assume_compatibility=False,
        check_compatibility_aggressively=False,
        _assume_minimal_Data=None,
    ):
        # validating
        if spanning_vf_set is not None:
            if isinstance(spanning_vf_set, (list, tuple)):
                spanning_vf_set = tuple(spanning_vf_set)
                if not all(
                    query_dgcv_categories(vf, "vector_field") for vf in spanning_vf_set
                ):
                    raise TypeError(
                        "The `spanning_vf_set` keyword in `distribution` can only be assigned a list or tuple of `VFClass` instances"
                    )
            else:
                raise TypeError(
                    "The `spanning_vf_set` keyword in `distribution` can only be assigned a list or tuple of `VFClass` instances"
                )
        if spanning_df_set is not None:
            if isinstance(spanning_df_set, (list, tuple)):
                spanning_df_set = tuple(spanning_df_set)
                if not all(
                    query_dgcv_categories(df, "differential_form") and df.degree == 1
                    for df in spanning_df_set
                ):
                    raise TypeError(
                        "The `spanning_df_set` keyword in `distribution` can only be assigned a list or tuple of `DFClass` instances. And they must all be degree 1."
                    )
            else:
                raise TypeError(
                    "The `spanning_df_set` keyword in `distribution` can only be assigned a list or tuple of `DFClass` instances"
                )
        if (
            spanning_vf_set is not None
            and spanning_df_set is not None
            and assume_compatibility is False
        ):
            for df in spanning_df_set:
                for vf in spanning_vf_set:
                    if check_compatibility_aggressively is True:
                        val = sp.simplify(df(vf))
                        if val != 0:
                            raise TypeError(
                                f"Unnable to verify if the provided vector fields and differential forms annihilate each other. This may be due a failure in the program to recognize if {val} is zero. If that value in nonzero then the two sets do not define a comonn distribution. Set `assume_compatibility = True` to force initialization despite this compatibility check failing."
                            )
                    else:
                        if df(vf) != 0:
                            raise TypeError(
                                "Unnable to verify if the provided vector fields and differential forms annihilate each other. This may be due a failure in the program to recognize if complex expression is zero. Set `check_compatibility_aggressively = True` to implement more expensive simplify methods in this step. Or set `assume_compatibility = True` to force initialization despite this compatibility check failing."
                            )

        self._prefered_data_type = 1 if spanning_df_set is None else 0
        if spanning_vf_set is None and spanning_df_set is None:
            self._spanning_vf_set = tuple()
            self._spanning_df_set = tuple()
        if spanning_vf_set is not None:
            self._spanning_vf_set, varSpace1, self._varSpace_type = (
                self._validate_spanning_sets(spanning_vf_set)
            )
            if spanning_df_set is not None:
                self._spanning_df_set, varSpace2, _ = self._validate_spanning_sets(
                    spanning_df_set, target_type=self._varSpace_type
                )
                varSpace1.extend(varSpace2)
                self.varSpace = tuple(dict.fromkeys(varSpace1))
            self._spanning_df_set = None
            self.varSpace = tuple(varSpace1)
        else:
            self._spanning_vf_set = None
            self._spanning_df_set, self.varSpace, self._varSpace_type = (
                self._validate_spanning_sets(spanning_df_set)
            )
        if _assume_minimal_Data == retrieve_passkey():
            self._vf_basis = self._spanning_vf_set
            self._df_basis = self._spanning_df_set
        else:
            self._vf_basis = None
            self._df_basis = None
        self._derived_flag = None
        self._wderived_flag = None

    def _validate_spanning_sets(self, spanning_set, target_type=None):
        standardList = []
        realList = []
        complexList = []
        if target_type == "real" or target_type == "complex":
            primaryType = target_type
        else:
            primaryType = "standard"
        for elem in spanning_set:
            if elem._varSpace_type == "standard":
                standardList.append(elem)
            if elem._varSpace_type == "real":
                realList.append(elem)
                if primaryType == "standard":
                    primaryType = "real"
            if elem._varSpace_type == "complex":
                complexList.append(elem)
                if primaryType == "standard":
                    primaryType = "complex"
        if primaryType == "complex":
            formattedList = tuple(
                standardList + complexList + [allToHol(elem) for elem in realList]
            )
        elif primaryType == "real":
            formattedList = tuple(
                standardList + realList + [allToReal(elem) for elem in complexList]
            )
        else:
            formattedList = tuple(standardList)

        varSpaceLoc = []
        for j in formattedList:
            varSpaceLoc.extend(j.varSpace)
        varSpaceList = list(dict.fromkeys(varSpaceLoc))
        return formattedList, varSpaceList, primaryType

    @property
    def spanning_vf_set(self):
        if self._spanning_vf_set is None:
            self._spanning_vf_set = [
                changeVFBasis(vf, self.varSpace)
                for vf in annihilator(self.df_basis, self.varSpace)
            ]
            self._vf_basis = self._spanning_vf_set
        return self._spanning_vf_set

    @property
    def vf_basis(self):
        if self._vf_basis is None:
            if self._spanning_vf_set is None:
                return self.spanning_vf_set
            vfBasis = []
            for vf in self._spanning_vf_set:
                if decompose(vf, vfBasis, only_check_decomposability=True) is False:
                    vfBasis.append(vf)
            self._vf_basis = vfBasis
        return self._vf_basis

    @property
    def spanning_df_set(self):
        if self._spanning_df_set is None:
            self._spanning_df_set = annihilator(self.vf_basis, self.varSpace)
            self._df_basis = annihilator(self._spanning_df_set, self.varSpace)
        return self._spanning_df_set

    @property
    def df_basis(self):
        if self._df_basis is None:
            if self._spanning_df_set is None:
                return self.spanning_df_set
            dfBasis = []
            for df in self._spanning_df_set:
                if decompose(df, dfBasis, only_check_decomposability=True) is False:
                    dfBasis.append(df)
            self._df_basis = dfBasis
        return self._df_basis

    def derived_flag(self, max_iterations=10):
        if self._derived_flag is None:
            tiered_list = [list(self.vf_basis)]
            if self._prefered_data_type == 0:
                pass

            def derive_extension(tieredList):
                flattenedTL = sum(tieredList, [])
                baseL = flattenedTL
                newTeir = []
                topLevel = tieredList[-1]
                for vf1 in baseL:
                    for vf2 in topLevel:
                        nb = LieDerivative(vf1, vf2)
                        if (
                            decompose(nb, flattenedTL, only_check_decomposability=True)
                            is False
                        ):
                            flattenedTL.append(nb)
                            newTeir.append(nb)
                return list(tieredList) + [newTeir]

            for _ in range(max_iterations):
                newTL = derive_extension(tiered_list)
                if len(newTL[-1]) == 0:
                    break
                else:
                    tiered_list = newTL
            self._derived_flag = tiered_list
        return self._derived_flag

    def weak_derived_flag(self, max_iterations=10):
        if self._wderived_flag is None:
            tiered_list = [list(self.vf_basis)]
            if self._prefered_data_type == 0:
                pass

            def derive_extension(tieredList):
                baseL = list(tieredList[0])
                flattenedTL = sum(tieredList, [])
                newTeir = []
                topLevel = list(tieredList[-1])
                for vf1 in baseL:
                    for vf2 in topLevel:
                        nb = LieDerivative(vf1, vf2)
                        if (
                            decompose(nb, flattenedTL, only_check_decomposability=True)
                            is False
                        ):
                            flattenedTL.append(nb)
                            newTeir.append(nb)
                return list(tieredList) + [newTeir]

            for _ in range(max_iterations):
                newTL = derive_extension(tiered_list)
                if len(newTL[-1]) == 0:
                    break
                else:
                    tiered_list = newTL
            self._wderived_flag = tiered_list
        return self._wderived_flag

    def nilpotent_approximation(
        self,
        expansion_point=None,
        label=None,
        basis_labels=None,
        exclude_from_VMF=False,
    ):
        """expansion point should be a dictionary assigning numeric (float, int) values to the variables `distribution.varSpace`"""
        if expansion_point is None:
            expansion_point = {var: 0 for var in self.varSpace}
        derFlag = self.weak_derived_flag()
        depth = len(derFlag)
        basisVF = sum(derFlag, [])
        if len(basisVF) < len(self.varSpace):
            warnings.warn(
                f"The distribution is not bracket generating. A compliment to its bracket-generated envelope has been assigned weight {-depth} and added to the nilpotent approximation as a component commuting with everything."
            )
        elif len(basisVF) > len(self.varSpace):
            raise TypeError(
                f"The distribution is singular at the point {expansion_point}. Nilpotent approximations are not yet supported for singular distributions."
            )
        VFCoeffs = [
            [
                tuple(VF_coeffs((vf.subs(expansion_point)), self.varSpace))
                for vf in level
            ]
            for level in derFlag
        ]
        VFCFlattened = sum(VFCoeffs, [])
        level_dimensions = [len(level) for level in derFlag]

        def i_to_w_rule(idx):
            cap = 0
            for level, ld in enumerate(level_dimensions):
                cap += ld
                if idx < cap:
                    return -1 - level
            return -depth

        idx_to_weight_assignment = {j: i_to_w_rule(j) for j in range(len(VFCFlattened))}
        weight_to_level_assignment = {
            w: -1 - w for w in idx_to_weight_assignment.values()
        }
        grading_vec = [
            idx_to_weight_assignment[idx] for idx in range(len(self.varSpace))
        ]
        rank = sp.Matrix(VFCFlattened).rank()
        if rank < len(VFCFlattened):
            raise TypeError(
                f"The distribution is singular at the point {expansion_point}. Nilpotent approximations are not yet supported for singular distributions."
            )
        algebra_data = dict()
        VFC_enum = list(enumerate(basisVF))
        varLabel = create_key(prefix="temp_var_")  # label for temparary variables
        variableProcedure(varLabel, len(VFCFlattened), _tempVar=retrieve_passkey())
        tVars = _cached_caller_globals[varLabel]  # pointer to tuple of coef vars
        for count1, elem1 in VFC_enum:
            for count2, elem2 in VFC_enum[count1 + 1 :]:
                newLevelWeight = (
                    idx_to_weight_assignment[count1] + idx_to_weight_assignment[count2]
                )
                if newLevelWeight < min(weight_to_level_assignment.keys()):
                    resulting_coeffs = [0] * len(self.varSpace)
                else:
                    newCoeffs = (
                        LieDerivative(elem1, elem2).subs(expansion_point)
                    ).coeffs
                    if len(self.varSpace) != len(newCoeffs):
                        raise SystemError(
                            "DEBUG: distribution VF var spaces need alignment (add it to the distribution initializer). Please report this bug to `dgcv` maintainers."
                        )
                    index_cap = sum(
                        level_dimensions[
                            : weight_to_level_assignment[newLevelWeight] + 1
                        ]
                    )
                    spanning_set = VFCFlattened[:index_cap]
                    general_elem = [
                        sum(
                            [
                                tVars[count] * coeff[idx]
                                for count, coeff in enumerate(spanning_set)
                            ]
                        )
                        for idx in range(len(self.varSpace))
                    ]
                    eqns = [j - k for j, k in zip(newCoeffs, general_elem)]
                    sol = solve_dgcv(eqns, tVars[: len(spanning_set)])
                    if len(sol) == 0:
                        raise RuntimeError(
                            f"Unable to compute the nilpotent approximation due to failure by internal solvers processing the equations {eqns}."
                        )
                    resulting_coeffs = [
                        (
                            0
                            if idx_to_weight_assignment[count] != newLevelWeight
                            else sol[0][var]
                        )
                        for count, var in enumerate(tVars)
                    ]
                    algebra_data[(count1, count2)] = resulting_coeffs
        clearVar(*listVar(temporary_only=True), report=False)
        if label is None:
            if basis_labels is not None:
                warnings.warn(
                    "The `distribution.nilpotent_approximation` method was given a `basis_labels` parameter value but no `label` parameter value. The former is ignored unless a `label` value is provided."
                )
            printWarning = "This algebra was initialized via the `distribution.nilpotent_approximation` method, but no labeling was assigned. Intentionally obscur labels were therefore assigned automatically. Use optional keywords `distribution.nilpotent_approximation(label='custom_label',basis_labels='list_of_labels')` for better labels. If such non-labeled initialization is really wanted, then use `distribution.nilpotent_approximation(exclude_from_VMF=True)` instead to suppress warnings such as this."
            childPrintWarning = "This algebraElement's parent algebra was initialized via the `distribution.nilpotent_approximation` method, but no labeling was assigned. Intentionally obscur labels were therefore assigned automatically. Use optional keywords `distribution.nilpotent_approximation(label='custom_label',basis_labels='list_of_labels')` for better labels. If such non-labeled initialization is really wanted, then use `distribution.nilpotent_approximation(exclude_from_VMF=True)` instead to suppress warnings such as this."
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
        else:
            return createAlgebra(
                algebra_data,
                label,
                basis_labels=basis_labels,
                grading=[grading_vec],
                assume_skew=True,
                return_created_obj=True,
            )
