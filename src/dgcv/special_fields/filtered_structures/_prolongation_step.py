from __future__ import annotations

from ..._aux._backends._polynomials import expr_union_primitives
from ..._aux._backends._symbolic_router import get_free_symbols, subs
from ..._aux._utilities._config import dgcv_warning, get_dgcv_settings_registry
from ..._aux._utilities._misc import linear_combination
from ..._aux._vmf._safeguards import get_dgcv_category
from ...algebras import _extract_basis
from ...core.solvers import solve_dgcv
from ._tensor_products import _fast_tensor_products


class _symbol_prolongation_step:
    def _fast_prolong_by_1(
        self,
        levels,
        height,
        alias_counter,
        with_characteristic_space_reductions=False,
        DS_records=None,
        absorb_DS=False,
        surface_singularities=None,
        simplify_pivots=None,
    ):  # height must match levels structure
        get_alias_id = alias_counter

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
        if DS_records is None:
            DS_records = []
        ADS = absorb_DS is True
        if (
            self.assume_FGLA
            and len(levels[height]) == 0
            or (
                self._GLA_generators is not None
                and all(
                    len(levels[height - j]) == 0
                    for j in range(-min(self._GLA_generators.get("generators", levels)))
                )
                and min(self._GLA_generators.get("generators", levels)) >= -1 - height
            )
            or min(j for j in levels) >= -1 - height
            and all(
                len(levels[height - j]) == 0 for j in range(-min(j for j in levels))
            )
        ):  # stability check
            new_levels = levels
            new_levels._set_index_thr(height)
            stable = True
        else:
            ambient_basis = self._prolongation_ambient_basis(levels, height)

            if len(ambient_basis) == 0:
                ambient_basis = [0 * self.basis[0]]

            general_elem_terse, tVars = linear_combination(ambient_basis)
            general_elem = self._aliased_expansion(general_elem_terse, partial=True)

            eqns = []
            esVars = list(tVars)

            def _accumulate(expr):
                if getattr(expr, "is_zero", False) or expr == 0:
                    return
                if get_dgcv_category(expr) in {
                    "fastTensorProduct",
                    "tensorProduct",
                    "algebra_element",
                    "subalgebra_element",
                }:
                    eqns.extend(expr.coeff_dict.values())
                else:
                    dgcv_warning(
                        f"The constraint value {expr} is outside of expected class. Recieved type: {type(expr)}",
                        wc_label="debug_log",
                    )

            if len(DS_records) > 0:
                ambGE = None
                for record in DS_records:
                    source_bound = record.cap - height - 1
                    for (low, high), source in record.components.items():
                        if high >= 0 or low > source_bound:
                            continue
                        if low == high:
                            target = record.component(low + height + 1)
                            dsSpanners = target.spanners if target else []
                            sources = source.spanners
                        else:
                            dsSpanners = record.target_spanners(
                                low + height + 1,
                                min(high, source_bound) + height + 1,
                            )
                            sources = (
                                source.spanners
                                if high <= source_bound
                                else source.truncated_spanners(source_bound)
                            )
                        if ambGE is None:
                            ambGE = self._aliased_expansion(general_elem_terse)
                        for elem in sources:
                            if dsSpanners:
                                newGE, newVars = linear_combination(dsSpanners)
                                esVars += newVars
                                _accumulate(ambGE * elem + newGE)
                            else:
                                _accumulate(ambGE * elem)

            for triple in self.test_commutators:
                t0, t1, t2 = triple[0], triple[1], triple[2]
                _accumulate(
                    self._aliased_expansion(general_elem * t0, partial=True) * t1
                    + t0 * self._aliased_expansion(general_elem * t1, partial=True)
                    - general_elem * t2
                )

            if eqns == [0] or eqns == []:
                solution = [{}]
            else:
                if surface_singularities:
                    solution, sing = solve_dgcv(
                        eqns,
                        esVars,
                        method="linsolve",
                        return_divisors=True,
                        pass_to_symbolic_engine=False,
                        simplify_pivots=simplify_pivots,
                        simplify_result=False,
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
                    solution = solve_dgcv(
                        eqns, esVars, method="linsolve", simplify_result=False
                    )

            if len(solution) == 0:
                dgcv_warning(
                    f"At breakpoint in prolongation algorithm: The equation system was {eqns} w.r.t. {esVars}; return solution data was {solution}",
                    wc_label="debug_log",
                )
                raise RuntimeError(
                    "`Tanaka_symbol.prolongation` failed at a step where a symbolic solver (e.g., sympy.solve if using the default sympy) was being applied."
                )
            solution = solution[0]
            el_sol = subs(general_elem_terse, solution)
            if not isinstance(el_sol, _fast_tensor_products):
                el_sol = _fast_tensor_products(el_sol)

            fv_possibles = set(esVars)
            fv = set()
            for variable in tVars:
                fv |= get_free_symbols(solution.get(variable, variable))
            new_level = []
            zeroing = {v: 0 for v in fv if v in fv_possibles}
            for v in zeroing:
                basis_element = subs(el_sol, {**zeroing, v: 1})
                new_level.append(basis_element)

            expansions = [self._aliased_expansion(el) for el in new_level]
            if ADS is True:
                absorbed = []
                for record in DS_records:
                    component = record.component(height + 1)
                    if component is not None:
                        absorbed += list(component.spanners)
                if absorbed:
                    _, kept_idxs = _extract_basis(
                        expansions + absorbed, return_indices=True
                    )
                    offset = len(expansions)
                    kept = [absorbed[i - offset] for i in kept_idxs if i >= offset]
                    new_level = new_level + kept
                    expansions = expansions + kept

            new_level, expansions = self._characteristic_space_reduction(
                new_level,
                expansions,
                levels,
                height,
                with_characteristic_space_reductions,
                surface_singularities,
                simplify_pivots,
                simplify_ideals,
            )
            atomized_level = []
            for el, expanded in zip(new_level, expansions):
                new_idx = get_alias_id()
                alias_data = {
                    "expanded": expanded
                    if isinstance(expanded, _fast_tensor_products)
                    else _fast_tensor_products(expanded)
                }
                if el is not expanded:
                    alias_data["operator"] = self._aliased_expansion(el, partial=True)
                self._aliasing[new_idx] = alias_data
                atom = _fast_tensor_products({(new_idx,): 1}, _atomic_index=new_idx)
                atomized_level.append(atom)
            new_level = atomized_level

            self._recoordinatize_DS_components(
                DS_records, height, atomized_level, expansions
            )

            new_levels = self._GLA_structure(
                levels | {height + 1: new_level}, levels.index_threshold
            )
            stable = False
        return new_levels, stable, DS_records
