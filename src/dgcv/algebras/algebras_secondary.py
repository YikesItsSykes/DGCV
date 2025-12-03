import numbers
import warnings

import sympy as sp

from .._config import (
    dgcv_exception_note,
    get_dgcv_settings_registry,
    get_variable_registry,
)
from .._safeguards import (
    _cached_caller_globals,
    create_key,
    get_dgcv_category,
    query_dgcv_categories,
    retrieve_passkey,
    retrieve_public_key,
    unique_label,
    validate_label,
    validate_label_list,
)
from ..backends._caches import _get_expr_num_types, _get_expr_types
from ..combinatorics import carProd
from ..dgcv_core import variableProcedure
from ..solvers import simplify_dgcv, solve_dgcv
from ..tensors import mergeVS, tensorProduct
from ..vmf import clearVar, listVar
from .algebras_aux import _validate_structure_data
from .algebras_core import (
    _lazy_SD,
    algebra_class,
    algebra_element_class,
    algebra_subspace_class,
    homomorphism,
    killingForm,
    linear_representation,
)


class subalgebra_class(algebra_subspace_class):
    def __init__(
        self,
        basis,
        alg,
        grading=None,
        _compressed_structure_data=None,
        _internal_lock=None,
        span_warning=True,
        simplify_basis=False,
        simplify_products_by_default=None,
        _markers={},
        **kwargs,
    ):
        super().__init__(
            basis,
            alg,
            test_weights=None,
            _grading=grading,
            _internal_lock=_internal_lock,
            span_warning=span_warning,
            simplify_basis=False,
        )

        _markers['_educed_properties']=dict()



        basis = self.filtered_basis
        self.structureData = None
        params=set()
        def _tuple_scan(elems,par:set):
            for elem in elems:
                par|=getattr(elem,"free_symbols",set())
            return tuple(elems)

        if _internal_lock == retrieve_passkey():
            if _compressed_structure_data is not None:
                self.structureData = tuple(
                    tuple(_tuple_scan(inner,params) for inner in middle)
                    for middle in _compressed_structure_data
                )  ###!!! optimize by always preprocessing _compressed_structure_data elsewhere
        if self.structureData is None:
            valSD = self.is_subalgebra(return_structure_data=True)["structure_data"]
            self.structureData = tuple(
                tuple(_tuple_scan(inner,params) for inner in middle) for middle in valSD
            )
        # self._structureData = tuple(map(tuple, self.structureData))
        self._parameters=params
        self.subindices_to_ambient_dict = {
            count: elem for count, elem in enumerate(basis)
        }
        self.basis_in_ambient_alg = tuple(basis)
        self.basis = [
            subalgebra_element(
                self,
                [1 if j == count else 0 for j in range(self.dimension)],
                elem.valence,
            )
            for count, elem in enumerate(basis)
        ]
        if all(elem in self.ambient for elem in basis):
            self.basis_labels = [elem.__str__() for elem in self.basis]
        else:
            self.basis_labels = [f"_e_{j+1}" for j in range(self.dimension)]
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "subalgebra"
        self.structureDataDict = _lazy_SD(self.structureData)
        if (
            simplify_products_by_default is True
            or self.ambient.simplify_products_by_default is True
        ):
            self.simplify_products_by_default = True
        else:
            self.simplify_products_by_default = simplify_products_by_default
        self._registered=self.ambient._registered
        # cached_properties
        self._jacobi_identity_cache = None
        self._skew_symmetric_cache = None
        self._lie_algebra_cache = None
        self._killing_form = None
        self._derived_subalg_cache = None
        self._derived_series_cache = None
        self._lower_central_series_cache = None
        self._radical_cache = None
        self._Levi_deco_cache = None
        self._is_semisimple_cache = None
        self._is_simple_cache = None
        self._is_nilpotent_cache = None
        self._is_abelian_cache = None
        self._is_solvable_cache = None
        self._rank_approximation = None
        self._graded_components = None
        self._educed_properties = dict()
        ep = getattr(self.ambient,'_educed_properties',dict())
        t_message='True by inheritance: parent algebra --> subalgebra'
        if ep.get('is_Lie_algebra',None) is not None:
            self._educed_properties['is_Lie_algebra']=t_message
        if ep.get('is_skew',None) is not None:
            self._educed_properties['is_skew']=t_message
        if ep.get('satisfies_Jacobi_ID',None) is not None:
            self._educed_properties['satisfies_Jacobi_ID']=t_message
        if ep.get('is_nilpotent',None) is not None:
            self._educed_properties['is_nilpotent']=t_message
        if ep.get('is_solvable',None) is not None:
            self._educed_properties['is_solvable']=t_message
        if ep.get('special_type',None) in {'abelian','solvable','nilpotent'}:
            self._educed_properties['special_type']=ep.get('special_type',None)

    @property
    def zero_element(self):
        return subalgebra_element(self,(0,)*self.dimension,1)

    def __contains__(self, item):
        return item in self.basis

    def __iter__(self):
        return iter(self.basis)

    def __getitem__(self, indices):
        if isinstance(indices, int):
            return self.basis[indices]
        elif isinstance(indices, list):
            if len(indices) == 1:
                return self.basis[indices[0]]
            elif isinstance(indices, list) and len(indices) == 2:
                return self.structureData[indices[0]][indices[1]]
            elif isinstance(indices, list) and len(indices) == 3:
                return self.structureData[indices[0]][indices[1]][indices[2]]
        else:
            raise TypeError(
                f"To access a subalgebra element or structure data component, provide one index for an element from the basis, two indices for a list of coefficients from the product  of two basis elements, or 3 indices for the corresponding entry in the structure array. Instead of an integer of list of integers, the following was given: {indices}"
            ) from None

    # (self,alg,coeffs,valence,ambient_rep=None,_internalLock=None)
    def _class_builder(self, coeffs, valence):
        return subalgebra_element(self, coeffs, valence)

    def _set_product_protocol(self):
        if self.simplify_products_by_default is None:
            if any(
                not isinstance(j, numbers.Number)
                for j in self.structureDataDict.values()
            ):
                self.simplify_products_by_default = True
            else:
                self.simplify_products_by_default = False
        elif self.simplify_products_by_default is not True:
            self.simplify_products_by_default is False

    def __eq__(self, other):
        if not isinstance(other, subalgebra_class):
            return NotImplemented
        return self.dgcv_vs_id == other.dgcv_vs_id

    def __hash__(self):
        return hash(self.dgcv_vs_id)

    def multiplication_table(
        self, elements=None, restrict_to_subspace=False, style=None, use_latex=None
    ):
        if elements is None:
            newElements = [elem.ambient_rep for elem in self.basis]
        elif isinstance(elements, (list, tuple)):
            warningMessage = ""
            newElements = []
            for elem in elements:
                elemTest = (
                    elem.ambient_rep if isinstance(elem, subalgebra_element) else elem
                )
                if self.contains(elemTest) is False:
                    if warningMessage == "":
                        warningMessage += "Some elements in the `elements` list were not in the span of the subalgebra's basis, so they were omitted from the multiplication table."
                else:
                    newElements.append(elemTest)
            if warningMessage != 0 and len(newElements) > 0:
                warnings.warn(warningMessage)
            else:
                raise TypeError(
                    "No elements from the provided `elements` list belong to the subalgebra, so a multiplication table will not be produced."
                ) from None
        else:
            raise TypeError(
                "If provided, the `elements` parameter in `subalgebra_class.multiplication_table` must be a list."
            ) from None

        return self.ambient.multiplication_table(
            elements=newElements,
            restrict_to_subspace=restrict_to_subspace,
            style=style,
            use_latex=use_latex,
            _called_from_subalgebra={
                "internalLock": retrieve_passkey(),
                "basis": self.basis,
            },
        )

    def subalgebra(self, basis, grading=None, span_warning=False, simplify_basis=False, simplify_products_by_default=None):
        elems = [
            (
                elem.ambient_rep
                if get_dgcv_category(elem) == "subalgebra_element"
                else elem
            )
            for elem in basis
        ]
        return self.ambient.subalgebra(elems, grading=grading, simplify_basis=simplify_basis, span_warning=span_warning, simplify_products_by_default=simplify_products_by_default)

    def subspace(self, basis:list|tuple=[], grading=None, span_warning=True):
        elems = [
            (
                elem.ambient_rep
                if get_dgcv_category(elem) == "subalgebra_element"
                else elem
            )
            for elem in basis
        ]
        return self.ambient.subspace(elems, grading=grading, span_warning=span_warning)

    def contains(
        self, items, return_basis_coeffs=False
    ):  ###!!! optimize for high volume calls with batched coeff creation
        if not isinstance(items, (list, tuple)):
            items = [items]
        for item in items:
            if get_dgcv_category(item) == "subalgebra_element":
                if item.algebra == self:
                    bas = self.basis
                elif item.algebra.ambient == self.ambient:
                    item = item.ambient_rep
                    bas = self.basis_in_ambient_alg
                else:
                    return False
            elif (
                get_dgcv_category(item) == "algebra_element"
                and item.algebra == self.ambient
            ):
                bas = self.basis_in_ambient_alg
            else:
                return False
            if item not in bas:
                if len(bas)==0:
                    return False
                tempVarLabel = "T" + retrieve_public_key()
                vars=variableProcedure(tempVarLabel, len(bas), _tempVar=retrieve_passkey(),return_created_object=True)[0]
                genElement = sum([vars[j + 1] * elem for j, elem in enumerate(bas[1:])], vars[0] * (bas[0]))
                sol = solve_dgcv(item - genElement, vars)
                if len(sol) == 0:
                    clearVar(*listVar(temporary_only=True), report=False)
                    return False
            else:
                if return_basis_coeffs is True:
                    idx = bas.index(item)
                    return [1 if _ == idx else 0 for _ in range(len(bas))]
        if return_basis_coeffs is True:
            vec = [var.subs(sol[0]) for var in vars]
            clearVar(*listVar(temporary_only=True), report=False)
            return vec
        clearVar(*listVar(temporary_only=True), report=False)
        return True

    def copy(
        self,
        label=None,
        basis_labels=None,
        register_in_vmf=False,
        initial_basis_index=None,
        simplify_products_by_default=None,
    ):
        if simplify_products_by_default is None:
            simplify_products_by_default = self.simplify_products_by_default
        if not isinstance(label, str) or label == "":
            label = "Alg_" + create_key()
        if isinstance(basis_labels, (tuple, list)):
            if (
                not all(isinstance(elem, str) for elem in basis_labels)
                or len(basis_labels) != self.dimension
            ):
                warnings.warn(
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
            basis_labels = [f"{pref}{i+IIdx}" for i in range(self.dimension)]
        if not isinstance(self._grading,(list,tuple)) or len(self._grading) == 0:
            grad = None
        else:
            grad = self._grading
        if register_in_vmf is True:
            return createAlgebra(
                self.structureData,
                label,
                basis_labels=basis_labels,
                grading=grad,
                return_created_obj=True,
                simplify_products_by_default=simplify_products_by_default,
            )
        else:
            return algebra_class(
                self.structureData,
                grading=grad,
                simplify_products_by_default=simplify_products_by_default,
                _label=label,
                _basis_labels=basis_labels,
                _calledFromCreator=retrieve_passkey(),
            )

    def is_skew_symmetric(self, verbose=False, _return_proof_path=False):
        """
        Checks if the algebra is skew-symmetric.
        """
        if not self._registered and verbose:
            if self._callLock == retrieve_passkey() and isinstance(
                self._print_warning, str
            ):
                print(self._print_warning)
            else:
                print(
                    "Warning: This algebra instance is unregistered. Initialize algebra objects with createFiniteAlg instead to register them."
                )

        if isinstance(self._educed_properties.get('is_skew',None),str):
            t_message=self._educed_properties.get('is_skew',None)
            self._skew_symmetric_cache = (True,None)
        else:
            t_message=''

        if self._skew_symmetric_cache is None:
            result, failure = self._check_skew_symmetric()
            self._skew_symmetric_cache = (result, failure)
        else:
            result, failure = self._skew_symmetric_cache

        if verbose:
            if result:
                if self.ambient.label is None:
                    print("The algebra is skew-symmetric.")
                else:
                    print(f"The subalgebra in {self.ambient.label} is skew-symmetric.")
            else:
                i, j, k = failure
                print(
                    f"Skew symmetry fails for basis elements {i} and {j}, at coefficient index {k}."
                )
        if _return_proof_path is True:
            return result, t_message
        return result

    def _check_skew_symmetric(self):
        for i in range(self.dimension):
            for j in range(i, self.dimension):
                for k in range(self.dimension):
                    vector_sum_element = (
                        self.structureData[i][j][k] + self.structureData[j][i][k]
                    )
                    if vector_sum_element != 0:
                        return False, (i, j, k)
        return True, None

    def satisfies_jacobi_identity(self, verbose=False, _return_proof_path=False):
        """
        Checks if the algebra satisfies the Jacobi identity.
        Includes a warning for unregistered instances only if verbose=True.
        """
        if not self._registered and verbose:
            if self._callLock == retrieve_passkey() and isinstance(
                self._print_warning, str
            ):
                print(self._print_warning)
            else:
                print(
                    "Warning: This algebra instance is unregistered. Initialize algebra objects with createFiniteAlg instead to register them."
                )

        if isinstance(self._educed_properties.get('satisfies_Jacobi_ID',None),str):
            t_message=self._educed_properties.get('satisfies_Jacobi_ID',None)
            self._jacobi_identity_cache = (True,None)
        else:
            t_message=''


        if self._jacobi_identity_cache is None:
            result, fail_list = self._check_jacobi_identity()
            self._jacobi_identity_cache = (result, fail_list)
        else:
            result, fail_list = self._jacobi_identity_cache

        if verbose:
            if result:
                if self.ambient.label is None:
                    print("The subalgebra satisfies the Jacobi identity.")
                else:
                    print(f"The subalgebra in {self.ambient.label} satisfies the Jacobi identity.")
            else:
                print(f"Jacobi identity fails for the following triples: {fail_list}")

        if _return_proof_path is True:
            return result, t_message
        return result

    def _check_jacobi_identity(self):
        skew = self.is_skew_symmetric()
        fail_list = []
        for i in range(self.dimension):
            lower_j = i + 1 if skew else 0
            for j in range(lower_j, self.dimension):
                lower_k = j + 1 if skew else 0
                for k in range(lower_k, self.dimension):
                    if not (
                        self.basis[i] * self.basis[j] * self.basis[k]
                        + self.basis[j] * self.basis[k] * self.basis[i]
                        + self.basis[k] * self.basis[i] * self.basis[j]
                    ).is_zero:
                        fail_list.append((i, j, k))
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
        - This helper method is intended for internal use.
        - Use it in methods where associativity is assumed but not explicitly verified.
        """
        warnings.warn(
            f"{method_name} assumes the subalgebra is associative. "
            "If it is not then unexpected results may occur.",
            UserWarning,
        )

    def is_Lie_algebra(self, verbose=False, return_bool=True,_return_proof_path=False):
        """
        Checks if the algebra is a Lie algebra.
        Includes a warning for unregistered instances only if verbose=True.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints detailed information about the check.
        return_bool : bool, optional
            Affects whether or not a boolian value is returned. If False, nothing is returned, which may be used in combination with verbose=True to have the function simply print a report.

        Returns
        -------
        bool or nothing
            True if the algebra is a Lie algebra, False otherwise. Nothing is returned if return_bool=False is set.
        """
        if not self._registered and verbose:
            if self._callLock == retrieve_passkey() and isinstance(
                self._print_warning, str
            ):
                print(self._print_warning)
            else:
                print(
                    "Warning: This algebra instance is unregistered. Initialize algebra objects with createFiniteAlg instead to register them."
                )

        if isinstance(self._educed_properties.get('is_Lie_algebra',None),str):
            t_message=self._educed_properties.get('is_Lie_algebra',None)
            self._lie_algebra_cache = True
            self._jacobi_identity_cache = True
            self._skew_symmetric_cache = True
        else:
            t_message=''

        if self._lie_algebra_cache is not None:
            if verbose:
                print(
                    f"Cached result: {'Previously verified the subalgebra is a Lie algebra' if self._lie_algebra_cache else 'Previously verified the subalgebra is not a Lie algebra'}."
                )
            if _return_proof_path is True:
                return self._lie_algebra_cache, t_message
            return self._lie_algebra_cache

        if not self.is_skew_symmetric(verbose=verbose):
            self._lie_algebra_cache = False
            if return_bool is True:
                if _return_proof_path is True:
                    return False, t_message
                return False
        if not self.satisfies_jacobi_identity(verbose=verbose):
            self._lie_algebra_cache = False
            if return_bool is True:
                if _return_proof_path is True:
                    return False, t_message
                return False
        if self._lie_algebra_cache is None:
            self._lie_algebra_cache = True

        if verbose:
            if self.ambient.label is None:
                print("The algebra is a Lie algebra.")
            else:
                print(f"The subalgebra in {self.ambient.label} is a Lie algebra.")

        if return_bool is True:
            if _return_proof_path is True:
                return self._lie_algebra_cache, t_message
            return self._lie_algebra_cache

    def _require_lie_algebra(self, method_name):
        """
        Checks that the subalgebra is a Lie algebra before proceeding.
        """
        if not self.is_Lie_algebra():
            raise ValueError(
                f"{method_name} can only be applied to Lie algebras."
            ) from None

    def is_semisimple(self, verbose=False, return_bool=True):
        """
        Checks if the algebra is semisimple.
        Nothing is returned if return_bool=False is set.
        """
        if not self.ambient._registered and verbose:
            if self.ambient._callLock == retrieve_passkey() and isinstance(
                self.ambient._print_warning, str
            ):
                print(self.ambient._print_warning)
            else:
                print(
                    "Warning: This algebra instance is unregistered. Initialize algebra objects with createFiniteAlg instead to register them."
                )

        if self._is_simple_cache is True:
            self._is_semisimple_cache=True

        if self._is_semisimple_cache is not None:
            if verbose:
                print(
                    f"Cached result: {f'Previously verified {self.label} is a semisimple Lie algebra' if self._is_semisimple_cache else f'Previously verified {self.label} is not a semisimple Lie algebra.'}."
                )
            if return_bool is True:
                return self._is_semisimple_cache
            else:
                return


        if not self.is_Lie_algebra(verbose=verbose):
            self._is_semisimple_cache = False
            if return_bool is True:
                return False
            else:
                return

        if verbose is True:
            print("Progress update: computing determinant of the Killing form...")
        det = sp.simplify(killingForm(self).det())  ###!!! Optimize, removing simplify

        if verbose:
            if det != 0:
                self._is_semisimple_cache = True
                self._educed_properties['special_type'] = 'semisimple'
                self._is_nilpotent_cache = False
                self._is_solvable_cache = False
                print("The subalgebra is semisimple.")
            else:
                self._is_semisimple_cache = False
                self._is_simple_cache = False
                print("The subalgebra is not semisimple.")
        if return_bool is True:
            return det != 0

    def is_simple(self, verbose=False, bypass_semisimple_check=False):
        if bypass_semisimple_check is False and self._is_semisimple_cache is None:
            self.is_semisimple(verbose=verbose)
        if self._is_simple_cache is None:
            self.compute_simple_subalgebras(verbose=verbose)
            if self._Levi_deco_cache['LD_components'][1].dimension == 0:
                self._is_semisimple_cache = True
                self._is_nilpotent_cache = False
                self._is_solvable_cache = False
                if len(self._Levi_deco_cache['simple_ideals'])==1:
                    self._is_simple_cache = True
                    self._educed_properties['special_type'] = 'simple'
                else:
                    self._is_simple_cache = False
                    self._educed_properties['special_type'] = 'semisimple'
            else:
                self._is_semisimple_cache = False
                self._is_simple_cache = False
                if self._Levi_deco_cache['LD_components'][0].dimension==0:
                    self._is_solvable_cache = True
                    if self._educed_properties['special_type'] is None:
                        self._educed_properties['special_type'] = 'solvable'
        return self._is_simple_cache

    def is_subspace_subalgebra(
        self, elements, return_structure_data=False, check_linear_independence=False
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

        elems = [
            (
                elem.ambient_rep
                if get_dgcv_category(elem) == "subalgebra_element"
                else elem
            )
            for elem in elements
        ]
        return self.ambient.is_subspace_subalgebra(
            elems,
            return_structure_data=return_structure_data,
            check_linear_independence=check_linear_independence,
        )

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

        if for_associative_alg is True:
            assume_Lie_algebra is False
        elif assume_Lie_algebra is False and not self.is_Lie_algebra():
            raise ValueError(
                "This algebra is not a Lie algebra. To compute the center for an associative algebra, set for_associative_alg=True."
            ) from None

        temp_label = create_key(prefix="center_var")
        variableProcedure(temp_label, self.dimension, _tempVar=retrieve_passkey())
        temp_vars = _cached_caller_globals[temp_label]

        el = sum(
            (temp_vars[i] * self.basis[i] for i in range(self.dimension)),
            self.basis[0] * 0,
        )

        if for_associative_alg:
            eqns = sum(
                [list((el * other - other * el).coeffs) for other in self.basis], []
            )
        else:
            eqns = sum([list((el * other).coeffs) for other in self.basis], [])

        solutions = solve_dgcv(eqns, temp_vars)
        if not solutions:
            warnings.warn(
                "The internal solver (which defaults to sympy.solve or sympy.linsolve unless running another symbolic engine) returned no solutions, indicating that this computation of the center failed, as solutions do exist. An empty list is being returned."
            )
            return []

        el_sol = el.subs(solutions[0])

        free_variables = tuple(set.union(*[set(j.free_symbols) for j in el_sol.coeffs]))

        return_list = []
        for var in free_variables:
            basis_element = el_sol.subs({var: 1}).subs(
                [(other_var, 0) for other_var in free_variables if other_var != var]
            )
            return_list.append(basis_element)

        clearVar(*listVar(temporary_only=True), report=False)

        return return_list  ###!!! return subalgebra instead

    def compute_derived_algebra(self, from_subalg=None):
        if from_subalg is None:
            from_subalg = self
        return self.ambient.compute_derived_algebra(from_subalg=from_subalg)

    def lower_central_series(
        self,
        max_depth=None,
        format_as_subalgebras=False,
        from_subalg=None,
        align_nested_bases=False,
    ):
        if from_subalg is None:
            from_subalg = self
        return self.ambient.lower_central_series(
            max_depth=max_depth,
            format_as_subalgebras=format_as_subalgebras,
            from_subalg=from_subalg,
            align_nested_bases=align_nested_bases,
        )

    def derived_series(
        self,
        max_depth=None,
        format_as_subalgebras=False,
        from_subalg=None,
        align_nested_bases=False,
    ):
        if from_subalg is None:
            from_subalg = self
        return self.ambient.derived_series(
            max_depth=max_depth,
            format_as_subalgebras=format_as_subalgebras,
            from_subalg=from_subalg,
            align_nested_bases=align_nested_bases,
        )

    def is_nilpotent(self,**kwargs):
        """
        Checks if the algebra is nilpotent.

        Returns
        -------
        bool
            True if the algebra is nilpotent, False otherwise.
        """
        if self._is_nilpotent_cache is None:
            series = self.lower_central_series()
            if len(series[-1])<2: # to allow different conventions for formatting a trivial level basis
                self._is_nilpotent_cache=True
                self._educed_properties['special_type'] = 'nilpotent'
                self._is_semisimple_cache=False
                self._is_simple_cache=False
            else:
                self._is_nilpotent_cache=False
                self._is_abelian_cache=True
        return self._is_nilpotent_cache

    def is_solvable(self,**kwargs):
        """
        Checks if the algebra is solvable.

        Returns
        -------
        bool
            True if the algebra is solvable, False otherwise.
        """
        if self._is_solvable_cache is None:
            if self._is_nilpotent_cache is None:
                series = self.derived_series()
                if len(series[-1])<2: # to allow different conventions for formatting a trivial level basis
                    self._is_solvable_cache=True
                    self._is_semisimple_cache=False
                    self._is_simple_cache=False
                    self._educed_properties['special_type'] = 'solvable'
                else:
                    self._is_solvable_cache=False
                    self._is_abelian_cache=False
                    self._is_nilpotent_cache=False
            else:
                self._is_solvable_cache=self._is_nilpotent_cache
        return self._is_solvable_cache

    def is_abelian(self,**kwargs):
        if self._is_abelian_cache is None:
            self._is_abelian_cache = all(elem == 0 for elem in self.structureDataDict.values())
            if self._is_abelian_cache is True:
                self._educed_properties['special_type'] = 'abelian'
                self._is_nilpotent_cache=True
                self._is_solvable_cache=True
                self._is_semisimple_cache=False
                self._is_simple_cache=False
        return self._is_abelian_cache

    def _require_lie_algebra(self, method_name):
        if not self.is_Lie_algebra():
            raise ValueError(
                f"{method_name} can only be applied to Lie algebras."
            ) from None

    def radical(self, from_subalg=None, assume_Lie_algebra=False):
        if from_subalg is None:
            from_subalg = self
        return self.ambient.radical(
            from_subalg=from_subalg, assume_Lie_algebra=assume_Lie_algebra
        )

    def killing_form_product(self, elem1, elem2, assume_Lie_algebra=False):
        kf = killingForm(self, assume_Lie_algebra=assume_Lie_algebra)
        vec1 = sp.Matrix(elem1.coeffs)
        vec2 = sp.Matrix(elem2.coeffs)
        return (vec2.transpose() * kf * vec1)[0]

    @property
    def graded_components(self):
        if self._graded_components is None:
            gradings = sorted(list(set([tuple(j) for j in zip(*self.grading)])))
            gc = {}
            for key in gradings:
                gc[key] = self.weighted_component(key)
            self._graded_components=gc
        return self._graded_components

    def compute_graded_component_wrt_weight_index(self,idx=0):
        if idx not in range(len(self.grading)):
            warnings.warn('The provided index is out of range. `compute_graded_component_wrt_weight_index` is using 0 instead.')
            idx=0
        wc = dict()
        for idxs,comp in self.graded_components.items():
            wc[idxs[idx]]=wc.get(idxs[idx],self.subspace([]))+comp 
        return wc

    def grading_summary(self):
        from .._dgcv_display import show

        gradingNumber = len(self.grading)
        graded_components = self.graded_components
        # gradings = sorted(list(set([tuple(j) for j in zip(*self.grading)])))
        # graded_components = {}
        # for key in gradings:
        #     graded_components[key] = self.weighted_component(key)
        pref = self._repr_latex_(abbrev=True).replace("$", "")
        if "_" in pref:
            prefi = f"\\left({pref} \\right)_"
        else:
            prefi = f"{pref}_"
        strings = []
        for k, v in graded_components.items():
            inner = ''.join(str(j) for j in k)
            latex = v._repr_latex_()
            latex = latex.replace('$', '').replace('\\displaystyle', '')
            strings.append(f" $$ {prefi}{{{inner}}} = {latex},$$")
        if len(strings) > 1:
            strings.insert(-1, "and")
        strings[-1] = strings[-1][:-3] + ".$$"
        if gradingNumber == 0:
            show(f"The algebra ${pref}$ has no assigend grading.")
        else:
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

    def weighted_component(self, weights, test_weights=None, trust_test_weight_format=False, from_subalg=None):
        if from_subalg is None:
            from_subalg=self
        return self.ambient.weighted_component(weights, test_weights=test_weights, trust_test_weight_format=trust_test_weight_format, from_subalg=from_subalg)

    def compute_simple_subalgebras(self,verbose=False):
        _ = self.Levi_decomposition(decompose_semisimple_fully=True,verbose=verbose)
        return self._Levi_deco_cache['simple_ideals']

    def Levi_decomposition(self,        
                           decompose_semisimple_fully = False,
                           _bust_cache=False,
                           assume_Lie_algebra=False,
                           _try_multiple_times=None,
                           verbose=False
                           ):
        return self.ambient.Levi_decomposition(from_subalg=self,decompose_semisimple_fully=decompose_semisimple_fully,verbose=verbose,_bust_cache=_bust_cache,assume_Lie_algebra=assume_Lie_algebra,_try_multiple_times=_try_multiple_times)

    def approximate_rank(self,check_semisimple=False,assume_semisimple=False,_use_cache=False,**kwargs):
        return self.ambient.approximate_rank(check_semisimple=check_semisimple,assume_semisimple=assume_semisimple,_use_cache=_use_cache,from_subalg=self)
        # if self.dimension==0:
        #     self._rank_approximation=0
        #     return 0
        # if check_semisimple is True:
        #     ssc=self.is_semisimple()
        #     if ssc is True:
        #         assume_semisimple = True
        #     elif assume_semisimple is True:
        #         print('approximate_rank recieved parameters `check_semisimple=True` and `assume_semisimple=True`, but the semisimple check returned false. The algorithm is proceeding with the `assume_semisimple` logic applied, but this is likely not wanted, and should be prevented by setting those parameters differently. Note, just setting `check_semisimple=True` is enough to use optimized algorithms in the event that the semisimple check returns true, whereas `assume_semisimple` should only be used in applications where forgoing the semisimple check entirely is wanted.')
        # if _use_cache and self._rank_approximation is not None:
        #     return self._rank_approximation
        # power=1 if (assume_semisimple or self._is_semisimple_cache is True) else self.dimension
        # elem = sp.Matrix(self.structureData[0])    # test element
        # bound=min(100,10*self.dimension)
        # for elem2 in self.structureData[1:]:
        #     elem+=random.randint(0,bound)*sp.Matrix(elem2)
        # rank = self.dimension-(elem**power).rank()
        # if not isinstance(self._rank_approximation,numbers.Integral) or self._rank_approximation>rank:
        #     self._rank_approximation=rank
        # return self._rank_approximation

    def direct_sum(self,other,grading=None,label=None,basis_labels=None,register_in_vmf=False,initial_basis_index=None,simplify_products_by_default=None,build_all_gradings=False):
        if get_dgcv_category(other) in {'algebra','vectorspace','subalgebra','algebra_subspace','vector_subspace'}:
            _markers={'sum':True,'lockKey':retrieve_passkey()}
            if build_all_gradings is not True:
                grad1 = self.grading[:1] or [[0]*self.dimension]
                grad2 = other.grading[:1] or [[0]*other.dimension]
            else:
                grad1 = self.grading or [[0]*self.dimension]
                grad2 = other.grading or [[0]*other.dimension]
            builtG=[]
            for gl1 in grad1:
                for gl2 in grad2:
                    builtG.append(list(gl1)+list(gl2))
            if not isinstance(grading,(list,tuple)):
                grading=[]
            if isinstance(grading, (list,tuple)):
                if all(isinstance(elem,(list,tuple)) for elem in grading):
                    grading = [list(elem) for elem in grading]+builtG
                elif all(isinstance(elem,_get_expr_num_types()) for elem in grading):
                    grading = [list(grading)]+builtG
                elif grading is not None:
                    warnings.warn('The supplied grading data format is incompatible, and was ignored.')
                    grading=builtG
                else:
                    grading=builtG


            if label is None:
                label = f'{self.label}_plus_{other.label}'
                _markers['_tex_label']=f'{self._repr_latex_(raw=True,abbrev=True)}\\oplus {other._repr_latex_(raw=True,abbrev=True)}'
            if basis_labels is None:
                basis_labels = [elem.__repr__() for elem in self.basis]+[elem.__repr__() for elem in other.basis]
                _markers['_tex_basis_labels']=[elem._repr_latex_(raw=True) for elem in self.basis]+[elem._repr_latex_(raw=True) for elem in other.basis]

            return linear_representation(homomorphism(self,other.endomorphism_algebra)).semidirect_sum(grading=grading,label=label,basis_labels=basis_labels,register_in_vmf=register_in_vmf,initial_basis_index=initial_basis_index,simplify_products_by_default=simplify_products_by_default,_markers=_markers)
        else: 
            return NotImplemented

    def __add__(self,other):
        if other==0 or getattr(other,'is_zero',False):
            return self
        if get_dgcv_category(other) in {'algebra','vectorspace','subalgebra','algebra_subspace','vector_subspace'}:
            return self.direct_sum(other)
        return NotImplemented

    def __radd__(self,other):
        if other==0 or getattr(other,'is_zero',False):
            return self
        return NotImplemented

    def tensor_product(self,other,grading=None,label=None,basis_labels=None,register_in_vmf=False,initial_basis_index=None,simplify_products_by_default=None,build_all_gradings=False):
        if get_dgcv_category(other) in {'algebra','vectorspace','subalgebra','algebra_subspace','vector_subspace'}: 
            if simplify_products_by_default is None:
                simplify_products_by_default = getattr(self,'simplify_products_by_default',False)
            if build_all_gradings is not True:
                grad1 = self.grading[:1] or [[0]*self.dimension]
                grad2 = other.grading[:1] or [[0]*other.dimension]
            else:
                grad1 = self.grading or [[0]*self.dimension]
                grad2 = other.grading or [[0]*other.dimension]
            builtG=[]
            for gl1 in grad1:
                for gl2 in grad2:
                    builtG.append([w1+w2 for w1 in gl1 for w2 in gl2])
            if not isinstance(grading,(list,tuple)):
                grading=[]
            if isinstance(grading, (list,tuple)):
                if all(isinstance(elem,(list,tuple)) for elem in grading):
                    grading = [list(elem) for elem in grading]+builtG
                elif all(isinstance(elem,_get_expr_num_types()) for elem in grading):
                    grading = [list(grading)]+builtG
                elif grading is not None:
                    warnings.warn('The supplied grading data format is incompatible, and was ignored.')
                    grading=builtG
                else:
                    grading=builtG

            if isinstance(basis_labels, (tuple, list)):
                if (
                    not all(isinstance(elem, str) for elem in basis_labels)
                    or len(basis_labels) != self.dimension*other.dimension
                ):
                    warnings.warn(
                        f"`basis_labels` is in an unsupported format and was ignored. Recieved {basis_labels}, types: {[type(lab) for lab in basis_labels]}, target length {self.dimension}*{other.dimension}"
                    )
                    basis_labels = None
            _markers={'prod':True,'lockKey':retrieve_passkey(),'tensor_decomposition':(self,other)}
            if label is None:
                label = f'{self.label}_tensor_{other.label}'
                _markers['_tex_label']=f'{self._repr_latex_(raw=True,abbrev=True)}\\otimes {other._repr_latex_(raw=True,abbrev=True)}'
            if basis_labels is None or not isinstance(basis_labels,str):
                basis_labels = [f'{elem1.__repr__()}_tensor_{elem2.__repr__()}' for elem1 in self.basis for elem2 in other.basis]
                _markers['_tex_basis_labels']=[f'{elem1._repr_latex_(raw=True)}\\otimes {elem2._repr_latex_(raw=True)}' for elem1 in self.basis for elem2 in other.basis]
            if isinstance(basis_labels,str):
                pref = basis_labels
                IIdx = (
                    initial_basis_index
                    if isinstance(initial_basis_index, numbers.Integral)
                    else 1
                )
                basis_labels = [f"{pref}{i+IIdx}" for i in range(self.dimension*other.dimension)]
            if not isinstance(label, str) or label == "":
                label = "Alg_" + create_key()

            if register_in_vmf is True:
                from .algebras_secondary import createAlgebra
                return createAlgebra(
                    self.dimension*other.dimension,
                    label,
                    basis_labels=basis_labels,
                    grading=grading,
                    return_created_obj=True,
                    simplify_products_by_default=simplify_products_by_default,
                    _markers=_markers
                )
            else:
                _markers['registered'] = False
                return algebra_class(
                    self.dimension*other.dimension,
                    grading=grading,
                    simplify_products_by_default=simplify_products_by_default,
                    _label=label,
                    _basis_labels=basis_labels,
                    _calledFromCreator=retrieve_passkey(),
                    _markers=_markers
                )
        elif isinstance(other,_get_expr_num_types()):
            return self._convert_to_tp().__matmul__(other)
        else:
            return NotImplemented

    def __matmul__(self,other):
        if get_dgcv_category(other) in {'algebra','vectorspace','subalgebra','algebra_subspace','vector_subspace'}:
            return self.tensor_product(other)
        return NotImplemented

    def __rmatmul__(self,other):
        if isinstance(other,_get_expr_num_types()):
            return self._convert_to_tp().__rmatmul__(other)

    def summary(self, generate_full_report=False, style=None, use_latex=None):
        return self.ambient.summary(generate_full_report=generate_full_report, style=style, use_latex=use_latex, _from_subalg=self, _IL=retrieve_passkey())

class subalgebra_element:
    def __init__(self, alg, coeffs, valence, ambient_rep=None, _internalLock=None):
        self.algebra = alg
        self.vectorSpace = alg
        if valence not in (0, 1):
            raise ValueError(f"valence must be 0 or 1, got {valence!r}")
        self.valence = valence
        if isinstance(coeffs, (list, tuple)):
            self.coeffs = tuple(coeffs)
        else:
            raise TypeError(
                "subalgebra_element expects coeffs to be a list or tuple."
            ) from None
        if _internalLock == retrieve_passkey():
            self._ambient_rep = ambient_rep
        else:
            self._ambient_rep = None
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "subalgebra_element"
        self.dgcv_vs_id=self.vectorSpace.dgcv_vs_id
        self._natural_weight=None

    @property
    def ambient_rep(self):
        if self._ambient_rep is None:
            self._ambient_rep = sum(
                [
                    coeff * self.algebra.subindices_to_ambient_dict[j + 1]
                    for j, coeff in enumerate(self.coeffs[1:])
                ],
                self.coeffs[0] * self.algebra.subindices_to_ambient_dict[0],
            )
        return self._ambient_rep

    def __eq__(self, other):
        if not isinstance(other, subalgebra_element):
            return NotImplemented
        return (
            self.algebra == other.algebra
            and self.coeffs == other.coeffs
            and self.valence == other.valence
        )

    def __hash__(self):
        return hash((self.algebra, self.coeffs, self.valence))

    def __str__(self):
        return self.ambient_rep.__str__()

    def _repr_latex_(self, verbose=False, raw=False):
        return self.ambient_rep._repr_latex_(verbose=verbose,raw=raw)

    def _latex(self, printer=None):
        return self._repr_latex_()

    def _sympystr(self):
        return self.ambient_rep._sympystr()

    def _latex_verbose(self, printer=None):
        return self.ambient_rep._latex_verbose(printer=printer)

    def __repr__(self):
        return self.ambient_rep.__repr__()

    @property
    def label(self):
        return self.__repr__()

    @property
    def is_zero(self):
        for j in self.coeffs:
            if sp.simplify(j) != 0:
                return False
        else:
            return True

    def _si_wrap(self, obj):
        if self.algebra.simplify_products_by_default is True:
            return simplify_dgcv(obj)
        else:
            return obj

    def _eval_simplify(self,*args,**kwargs):
        newCoeffs = [simplify_dgcv(j) for j in self.coeffs]
        return subalgebra_element(self.algebra, newCoeffs, self.valence)

    def subs(self, subsData):
        newCoeffs = [sp.sympify(j).subs(subsData) for j in self.coeffs]
        return subalgebra_element(self.algebra, newCoeffs, self.valence)

    def dual(self):
        return subalgebra_element(self.algebra, self.coeffs, (self.valence + 1) % 2)

    def _convert_to_tp(self):
        return tensorProduct(
            tuple([self.dgcv_vs_id]),
            {(j, self.valence,self.dgcv_vs_id): self.coeffs[j] for j in range(self.algebra.dimension)},
        )

    def _recursion_contract_hom(self, other):
        return self._convert_to_tp()._recursion_contract_hom(other)

    def _fast_add(self, other):
        """
        Internal-only: assumes `other` is a subalgebra_element_class
        from the same subalgebra with the same valence.
        No type or safety checks.
        No simplification.
        """
        coeffs = [a + b for a, b in zip(self.coeffs, other.coeffs)]
        return subalgebra_element(
            self.algebra,
            coeffs,
            self.valence,
            format_sparse=self.is_sparse,
        )

    def __add__(self, other):
        if getattr(other, "is_zero", False) or other==0:
            return self
        if get_dgcv_category(other)=='subalgebra_element':
            if self.algebra == other.algebra and self.valence == other.valence:
                coeffs = [a + b for a, b in zip(self.coeffs, other.coeffs)]
                return subalgebra_element(
                    self.algebra,
                    coeffs,
                    self.valence
                )
            elif other.algebra.ambient==self.algebra.ambient:
                return self.ambient_rep+self.ambient_rep
            else:
                other = other._convert_to_tp()
        if get_dgcv_category(other) in {'algebra_element','vector_space_element','tensorProduct'} or isinstance(other,_get_expr_num_types()):
            if self.algebra.ambient==getattr(other,'algebra',None):
                return self.ambient_rep+other
            return self._convert_to_tp()+other
        if get_dgcv_category(other)=='fastTensorProduct':
            return other+self
        return self.ambient_rep.__add__(other)
    def __radd__(self, other):
        if getattr(other, "is_zero", False) or other==0:
            return self
        if isinstance(other,_get_expr_num_types()):
            return self._convert_to_tp().__radd__(other)
        return NotImplemented

    def __sub__(self, other):
        return (self).__add__(-other)

    def __rsub__(self, other):
        return (-self).__radd__(other)

    def __mul__(self, other):
        if get_dgcv_category(other)== 'subalgebra_element':
            if self.algebra == other.algebra and self.valence == other.valence:
                sign = 1 if self.valence == 1 else -1
                alg = self.algebra
                dim = alg.dimension
                coeffs1 = self.coeffs
                coeffs2 = other.coeffs
                struct = alg.structureData
                raw_result = [0] * dim
                for i in range(dim):
                    ci = coeffs1[i]
                    if ci == 0:
                        continue
                    for j in range(dim):
                        cj = coeffs2[j]
                        if cj == 0:
                            continue
                        scalar = sign * ci * cj
                        row = struct[i][j]
                        for k in range(dim):
                            c_ijk = row[k]
                            if c_ijk != 0:
                                raw_result[k] += scalar * c_ijk
                if self.algebra.simplify_products_by_default:
                    result_coeffs = [self._si_wrap(c) for c in raw_result]
                else:
                    result_coeffs = raw_result
                return subalgebra_element(self.algebra, result_coeffs, self.valence)
            elif other.algebra.ambient==self.algebra.ambient:
                return self.ambient_rep*other.ambient_rep
            else:
                return self._convert_to_tp().__mul__(other)
        elif isinstance(other, _get_expr_num_types()):
            new_coeffs = [self._si_wrap(coeff * other) for coeff in self.coeffs]
            return subalgebra_element(self.algebra, new_coeffs, self.valence)
        elif get_dgcv_category(other)=='algebra_element':
            return self.ambient_rep*other
        elif get_dgcv_category(other) in {'vector_space_element','tensorProduct'}:
            return self._convert_to_tp().__mul__(other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, _get_expr_num_types()):
            return self * other
        return NotImplemented
    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return self._si_wrap(sp.Rational(1, other) * self)
        elif isinstance(other, _get_expr_types()):
            return self._si_wrap((1 / other) * self)
        else:
            raise TypeError(
                f"True division `/` of subalgebra elements by another object is only supported for scalars, not {type(other)}"
            ) from None

    def __matmul__(self, other):
        """Overload @ operator for tensor product."""
        if get_dgcv_category(other)=='tensorProduct':
            return self._convert_to_tp() @ other
        if isinstance(other,_get_expr_num_types()):
            return other*self
        if get_dgcv_category(other) not in {'algebra_element','subalgebra_element','vector_space_element'}:
            return self._convert_to_tp().__matmul__(other)
        new_dict = {
            (j, k, self.valence, other.valence, self.dgcv_vs_id, other.dgcv_vs_id): self.coeffs[j] * other.coeffs[k]
            for j in range(self.algebra.dimension)
            for k in range(other.algebra.dimension)
        }
        return self._si_wrap(tensorProduct(mergeVS([self.dgcv_vs_id],[other.dgcv_vs_id]), new_dict))
    def __rmatmul__(self,other):
        if isinstance(other,_get_expr_num_types()):
            return other*self
        return self._convert_to_tp().__rmatmul__(other) 

    def __xor__(self, other):
        if other == "":
            return self.dual()
        raise ValueError("Invalid operation. Use `^''` to denote the dual.") from None

    def __neg__(self):
        return -1 * self

    def __call__(self, other, **kwds):
        if (
            get_dgcv_category(other) == "subalgebra_element"
            and other.algebra == self.algebra
        ):
            return sum([j * k for j, k in zip(self.coeffs, other.coeffs)])
        else:
            return self.ambient_rep(other)

    def check_element_weight(self, test_weights=None, flatten_weights=False):
        """
        Determines the weight vector of this subalgebra_element with respect to its ambient algebra's grading vectors.

        Returns
        -------
        list
            A list of weights corresponding to the grading vectors of the parent algebra.
            Each entry is either an integer, sympy.Expr (weight), the string 'AllW' if the element is the zero element,
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
        for idx, coeff in enumerate(self.coeffs):
            if coeff != 0:
                elem = self.algebra.basis[idx]
                w = elem.check_element_weight(
                    test_weights=test_weights, flatten_weights=flatten_weights
                )
                if isinstance(w, list):
                    w = tuple(w)
                weighted_components[w] = weighted_components.get(w, 0) + coeff * elem
        return weighted_components

    def terms(self):
        if self._terms is None:
            terms=[]
            for idx,c in enumerate(self.coeffs):
                if c==0:
                    continue
                terms.append(c*self.algebra.basis[idx])
            self._terms=[self] if len(terms)<2 else terms
        return self._terms

    def dual_pairing(self,other):
        return self._convert_to_tp().dual_pairing(other)

class simple_Lie_algebra(algebra_class):
    def __init__(
        self,
        structure_data,
        grading=None,
        format_sparse=False,
        process_matrix_rep=False,
        preferred_representation=None,
        _label=None,
        _basis_labels=None,
        _calledFromCreator=None,
        _callLock=None,
        _print_warning=None,
        _child_print_warning=None,
        _exclude_from_VMF=None,
        _simple_data=None,
        _basis_labels_parent=None
    ):
        if _calledFromCreator != retrieve_passkey():
            raise RuntimeError(
                "`simple_Lie_algebra` class instances can only be initialized by internal `dgcv` functions indirectly. To instantiate a simple Lie algebra, use dgcv `creator` functions"
            ) from None
        t_message='True by construction: instantiated from `simple_Lie_algebra` class constructor'
        super().__init__(
            structure_data,
            grading=grading,
            format_sparse=format_sparse,
            process_matrix_rep=process_matrix_rep,
            preferred_representation=preferred_representation,
            _label=_label,
            _basis_labels=_basis_labels,
            _calledFromCreator=_calledFromCreator,
            _callLock=_callLock,
            _print_warning=_print_warning,
            _child_print_warning=_child_print_warning,
            _exclude_from_VMF=_exclude_from_VMF,
            _basis_labels_parent=_basis_labels_parent,
            _markers={'simple':True,'_educed_properties':{'is_simple':t_message,'is_Lie_algebra':t_message,'is_semisimple':t_message,'special_type':'simple','is_skew':t_message,'satisfies_Jacobi_ID':t_message}}
        )

        self.roots = []
        self.simpleRoots = []
        self.rootSpaces = {(0,) * len(self.grading): []}

        def isSimpleRoot(vec):
            if vec.count(0) == len(vec) - 1 and vec.count(1) == 1:
                return True
            else:
                return False

        for elem in self.basis:
            root = tuple(elem.check_element_weight())
            if root in self.rootSpaces:
                self.rootSpaces[root].append(elem)
            else:
                self.rootSpaces[root] = [elem]
                self.roots.append(root)
                if isSimpleRoot(root):
                    self.simpleRoots.append(root)
        self.simpleRootSpaces = {
            root: self.rootSpaces[root] for root in self.simpleRoots
        }
        seriesLabel, rank = _simple_data["type"]
        self.rank = rank
        self.Cartan_subalgebra = self.basis[0:rank]
        self.simpleLieType = f"{seriesLabel}{rank}"  # example: "A3", "D4", ... etch

    def root_space_summary(self):
        def pluralize(idx):
            if idx != 1:
                return "s"
            else:
                return ""

        def rootString(idx):
            if idx == 1:
                return "(r_1)"
            if idx == 2:
                return "(r_1, r_2)"
            if idx == 3:
                return "(r_1, r_2, r_3)"
            else:
                return f"(r_1, ..., r_{idx})"

        print(
            f"This simple algebra {self.simpleLieType} has {self.rank} root{pluralize(self.rank)} {rootString(self.rank)}, which are dual to the Cartan subalgebra basis {self.Cartan_subalgebra}. These roots correspond to vertices in the Dynkin diagram as follows:\n"
        )

        if self.simpleLieType[0] == "D":
            n = self.rank
            if n == 2:
                print(
                    "Dynkin diagram for D2 is just two disconnected vertices corresponding to a direct sum of two u(2) copies."
                )
            else:
                lines = []
                horiz = "   "
                if n > 7:
                    mid_nodes = ["r_1 r_2", f"r_{n-4}", f"r_{n-3}", f"r_{n-2}"]
                    latter_rules = [
                        " ◯───◯─" + " ┅ ─",
                        "─" * (len(mid_nodes[1])),
                        "─" * (len(mid_nodes[2])),
                        "─" * (len(mid_nodes[3])),
                        "",
                    ]
                    horiz += "◯".join(latter_rules)
                    top_labels = f"{' '*4}{mid_nodes[0]}{' '*3}{mid_nodes[1]} {mid_nodes[2]} {mid_nodes[3]}"
                    fork_pos = 16 + len(latter_rules[1]) + len(latter_rules[2])
                elif n > 1:
                    horiz += "───".join("◯" for _ in range(n - 2))
                    top_labels = "   " + " ".join([f"r_{i+1}" for i in range(n - 2)])
                    horiz += "───◯"
                    fork_pos = 4 * (n - 2) - 1
                else:
                    horiz += "───".join("◯" for _ in range(n - 2))
                    top_labels = "   " + " ".join([f"r_{i+1}" for i in range(n - 2)])
                    horiz += "───◯"
                    fork_pos = 4 * (n - 2) - 1

                top_labels += " " + f"r_{n-1}"

                # Final node
                final_line = " " * fork_pos + "│"
                final_node = " " * fork_pos + f"◯ r_{n}"

                # bounding box
                width_bound = len(top_labels)
                title = "│" + self.simpleLieType.center(width_bound) + " │"
                border_top = "┌" + "─" * width_bound + "─┐"
                head_sep = "╞" + "═" * width_bound + "═╡"
                top_labels = "│" + top_labels + " │"
                horiz = "│" + horiz.ljust(width_bound) + " │"
                final_line = "│" + final_line.ljust(width_bound) + " │"
                final_node = "│" + final_node.ljust(width_bound) + " │"
                border_bottom = "└" + "─" * width_bound + "─┘"

                lines.append(border_top)
                lines.append(title)
                lines.append(head_sep)
                lines.append(top_labels)
                lines.append(horiz)
                lines.append(final_line)
                lines.append(final_node)
                lines.append(border_bottom)

                print("\n".join(lines))
        elif self.simpleLieType[0] == "B":
            n = self.rank
            lines = []
            horiz = "   "
            if n > 7:
                mid_nodes = ["r_1 r_2", f"r_{n-3}", f"r_{n-2}", f"r_{n-1}"]
                latter_rules = [
                    " ◯───◯─" + " ⋯ ─",
                    "─" * (len(mid_nodes[1])),
                    "─" * (len(mid_nodes[2])),
                    "═" * (len(mid_nodes[3]) - 2) + "═>",
                    "",
                ]
                horiz += "◯".join(latter_rules)
                top_labels = f"{' '*4}{mid_nodes[0]}{' '*3}{mid_nodes[1]} {mid_nodes[2]} {mid_nodes[3]}"
            else:
                horiz += "───".join("◯" for _ in range(n - 1))
                top_labels = "   " + " ".join([f"r_{i+1}" for i in range(n - 1)])
                horiz += "══>◯"
            top_labels += " " + f"r_{n}"

            # bounding box
            width_bound = len(top_labels) + 1
            title = "│" + self.simpleLieType.center(width_bound) + " │"
            border_top = "┌" + "─" * width_bound + "─┐"
            head_sep = "╞" + "═" * width_bound + "═╡"
            top_labels = "│" + top_labels + "  │"
            horiz = "│" + horiz.ljust(width_bound) + " │"
            border_bottom = "└" + "─" * width_bound + "─┘"

            lines.append(border_top)
            lines.append(title)
            lines.append(head_sep)
            lines.append(top_labels)
            lines.append(horiz)
            lines.append(border_bottom)

            print("\n".join(lines))
        elif self.simpleLieType[0] == "C":
            n = self.rank
            lines = []
            horiz = "   "
            if n > 7:
                mid_nodes = ["r_1 r_2", f"r_{n-3}", f"r_{n-2}", f"r_{n-1}"]
                latter_rules = [
                    " ◯───◯─" + " ⋯ ─",
                    "─" * (len(mid_nodes[1])),
                    "─" * (len(mid_nodes[2])),
                    "═" * (len(mid_nodes[3]) - 2) + "<═",
                    "",
                ]
                horiz += "◯".join(latter_rules)
                top_labels = f"{' '*4}{mid_nodes[0]}{' '*3}{mid_nodes[1]} {mid_nodes[2]} {mid_nodes[3]}"
            else:
                horiz += "───".join("◯" for _ in range(n - 1))
                top_labels = "   " + " ".join([f"r_{i+1}" for i in range(n - 1)])
                horiz += "══>◯"
            top_labels += " " + f"r_{n}"

            # bounding box
            width_bound = len(top_labels) + 1
            title = "│" + self.simpleLieType.center(width_bound) + " │"
            border_top = "┌" + "─" * width_bound + "─┐"
            head_sep = "╞" + "═" * width_bound + "═╡"
            top_labels = "│" + top_labels + "  │"
            horiz = "│" + horiz.ljust(width_bound) + " │"
            border_bottom = "└" + "─" * width_bound + "─┘"

            lines.append(border_top)
            lines.append(title)
            lines.append(head_sep)
            lines.append(top_labels)
            lines.append(horiz)
            lines.append(border_bottom)

            print("\n".join(lines))
        elif self.simpleLieType[0] == "A":
            n = self.rank
            lines = []
            horiz = "   "
            if n > 7:
                mid_nodes = ["r_1 r_2", f"r_{n-3}", f"r_{n-2}", f"r_{n-1}"]
                latter_rules = [
                    " ◯───◯─" + " ⋯ ─",
                    "─" * (len(mid_nodes[1])),
                    "─" * (len(mid_nodes[2])),
                    "─" * (len(mid_nodes[3])),
                    "",
                ]
                horiz += "◯".join(latter_rules)
                top_labels = f"{' '*4}{mid_nodes[0]}{' '*3}{mid_nodes[1]} {mid_nodes[2]} {mid_nodes[3]}"
            else:
                horiz += "───".join("◯" for _ in range(n - 1))
                top_labels = "   " + " ".join([f"r_{i+1}" for i in range(n - 1)])
                horiz += "───◯"
            top_labels += " " + f"r_{n}"

            # bounding box
            width_bound = len(top_labels) + 1
            title = "│" + self.simpleLieType.center(width_bound) + " │"
            border_top = "┌" + "─" * width_bound + "─┐"
            head_sep = "╞" + "═" * width_bound + "═╡"
            top_labels = "│" + top_labels + "  │"
            horiz = "│" + horiz.ljust(width_bound) + " │"
            border_bottom = "└" + "─" * width_bound + "─┘"

            lines.append(border_top)
            lines.append(title)
            lines.append(head_sep)
            lines.append(top_labels)
            lines.append(horiz)
            lines.append(border_bottom)

            print("\n".join(lines))

    def parabolic_grading(self, roots=None):
        if roots is None:
            roots = []
        if isinstance(roots, int):
            roots = [roots]
        elif not isinstance(roots, (list, tuple)):
            raise TypeError(
                f"The `roots` parameter in `simple_Lie_algebra.parabolic_grading(roots)` should be either `None`, an `int`, or a list of integers in the range (1,...,{self.rank}) representing indices of simple roots as enumerated in the algebras Dynkin diagram (see `simple_Lie_algebra.root_space_summary()` for a summary of this indexing)."
            ) from None
        gradingVector = [
            sum([self.grading[idx - 1][j] for idx in roots])
            for j in range(self.dimension)
        ]
        denom = 1
        for weight in gradingVector:
            if isinstance(weight, sp.Rational):
                if denom < weight.denominator:
                    denom = weight.denominator
        if denom > 1:
            gradingVector = [denom * weight for weight in gradingVector]
        return gradingVector

    def parabolic_subalgebra(
        self,
        roots=None,
        label=None,
        basis_labels=None,
        register_in_vmf=None,
        return_created_obj=False,
        use_non_positive_weights=False,
        format_as_subalgebra_class=False,
    ):
        if roots is None:
            roots = []
        if isinstance(roots, int):
            roots = [roots]
        if not isinstance(roots, (list, tuple)) or not all(root-1 in range(self.rank) for root in roots):
            raise TypeError(
                f"The `roots` parameter in `simple_Lie_algebra.parabolic_subalgebra(roots)` should be either `None`, an `int`, or a list of integers in the range (1,...,{self.rank}) representing indices of simple roots as enumerated in the algebras Dynkin diagram (see `simple_Lie_algebra.root_space_summary()` for a summary of this indexing)."
            ) from None
        marked = set(roots)
        newGrading = [sum([self.grading[idx - 1][j] for idx in marked]) for j in range(self.dimension)]
        if format_as_subalgebra_class is True:
            parabolic = []
        subIndices = []
        filtered_grading = []
        if not isinstance(use_non_positive_weights, bool):
            use_non_positive_weights = False
        # With H_i dual to simple roots, self.grading stores the simple-root coefficients n_i.
        # Sigma = marked nodes. Standard parabolic keeps Sigma-height ≥ 0; opposite keeps Sigma-height ≤ 0.
        sign = -1 if use_non_positive_weights else 1
        for count, weight in enumerate(newGrading):
            if sign * weight >= 0:
                if format_as_subalgebra_class is True:
                    parabolic.append(self.basis[count])
                subIndices.append(count)
                filtered_grading.append(weight)
        denom = 1
        for weight in filtered_grading:
            if isinstance(weight, sp.Rational):
                if denom < weight.denominator:
                    denom = weight.denominator
        if denom > 1:
            filtered_grading = [denom * weight for weight in filtered_grading]

        def truncateBySubInd(li):
            return [li[j] for j in subIndices]

        structureData = truncateBySubInd(self.structureData)
        structureData = [truncateBySubInd(plane) for plane in structureData]
        structureData = [[truncateBySubInd(li) for li in plane] for plane in structureData]
        if format_as_subalgebra_class is True:
            ignoredList = []
            if label is not None:
                ignoredList.append("label")
            if basis_labels is not None:
                ignoredList.append("basis_labels")
            if register_in_vmf is True:
                ignoredList.append("register_in_vmf")
            if len(ignoredList) == 1:
                warnings.warn(
                    f"A parameter value was supplied for `{ignoredList[0]}`, but `format_as_subalgebra_class=True` was set. The `subalgebra_class` is not tracked in the vmf, so this parameter value was ignored. A subalgebra_class instance was returned instead."
                )
            elif len(ignoredList) == 2:
                warnings.warn(
                    f"Parameter values were supplied for `{ignoredList[0]}` and `{ignoredList[1]}`, but `format_as_subalgebra_class=True` was set. The `subalgebra_class` is not tracked in the vmf, so these parameter values were ignored. A subalgebra_class instance was returned instead."
                )
            elif len(ignoredList) == 3:
                warnings.warn(
                    f"Parameter values were supplied for `{ignoredList[0]}`, `{ignoredList[1]}`, and `{ignoredList[2]}`, but `format_as_subalgebra_class=True` was set. The `subalgebra_class` is not tracked in the vmf, so these parameter values were ignored. `A subalgebra_class instance was returned instead.`"
                )
            return subalgebra_class(
                parabolic,
                self,
                grading=[filtered_grading],
                # _compressed_structure_data=structureData,
                # _internal_lock=retrieve_passkey(),
            )
        if isinstance(label, str) or isinstance(basis_labels, (list, tuple, str)):
            register_in_vmf = True
        if register_in_vmf is True:
            if label is None:
                label = self.label + "_parabolic"
            if basis_labels is None:
                basis_labels = label
            elif (
                isinstance(basis_labels, (list, tuple))
                and not all(isinstance(elem, str) for elem in basis_labels)
            ) or not isinstance(basis_labels, str):
                raise TypeError(
                    "If supplying the optional parameter `basis_labels` to `simple_Lie_algebra.parabolic_subalgebra` then it should be either a string or list of strings"
                ) from None
            createAlgebra(
                structureData,
                label=label,
                basis_labels=basis_labels,
                grading=filtered_grading,
            )
            if return_created_obj is True:
                return _cached_caller_globals[label]
        if return_created_obj is True:
            return algebra_class(structureData, grading=[filtered_grading])
        elif register_in_vmf is not True:
            warnings.warn(
                "Optional keywords for the `parabolic_subalgebra` method indicate that nothing should be return returned or registered in the vmf. Probably that is not intended, in which case at least one keyword `label`, `basis_labels`, `register_in_vmf`, `return_created_obj`, or `format_as_subalgebra_class` should be set differently."
            )

def createSimpleLieAlgebra(
    series: str,
    label: str = None,
    basis_labels: list = None,
    build_standard_mat_rep=False,
    return_created_obj=False,
    forgo_vmf_registry=False
):
    """
    Creates a simple (with 2 exceptions) complex Lie algebra specified from the classical
    series
        - A_n = sl(n+1)     for n>0
        - B_n = so(2n+1)    for n>0
        - C_n = sp(2n)      for n>0
        - D_n = so(2n)      for n>0 (not simple for n=1,2)


    Parameters
    ----------
    series : str
        The type and rank of the Lie algebra, e.g., "A1", "A2", ..., "Dn".
    label : str, optional
        Custom label for the Lie algebra. If not provided, defaults to a standard notation,
        like sl2 for A2 etc.
    basis_labels : list, optional
        Custom labels for the basis elements. If not provided, default labels will be generated.

    Returns
    -------
    algebra
        The resulting Lie algebra as an algebra instance.

    Raises
    ------
    ValueError
        If the series label is not recognized or not implemented.

    Notes
    -----
    - Currently supports only the A,B, and D series (special linear Lie algebras: A_n = sl(n+1), etc.).
    """
    try:
        series_type, rank = series[0], int(series[1:])
        series_type = "".join(c.upper() if c.islower() else c for c in series_type)
    except (IndexError, ValueError):
        raise ValueError(
            f"Invalid series format: {series}. Expected a letter 'A', 'B', 'C', 'D', 'E', 'F', or 'G' followed by a positive integer, like 'A1', 'B5', etc."
        ) from None
    if rank <= 0:
        raise ValueError(
            f"Sequence index must be a positive integer, but got: {rank}."
        ) from None

    def _generate_A_series_structure_data(n):
        matrix_dim = n + 1

        # Basis elements
        hBasis = {"elems": dict(), "grading": dict()}
        offDiag = {"elems": dict(), "grading": dict()}

        repMatrix = [[0] * matrix_dim for _ in range(matrix_dim)]

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
                # Diagonal (Cartan) element
                if j == k and j < n:
                    M = [row[:] for row in repMatrix]
                    for idx in range(n + 1):
                        if idx > j:
                            M[idx][idx] = -sp.Rational(j+1,n+1)
                        else:
                            M[idx][idx] = 1-sp.Rational(j+1,n+1)
                    hBasis["elems"][(j, k, 0)] = M
                    hBasis["grading"][(j, k, 0)] = [0] * n
                elif j != k:
                    # off diagonal generators
                    MPlus = [row[:] for row in repMatrix]
                    MMinus = [row[:] for row in repMatrix]
                    MPlus[j][k] = 1
                    MMinus[k][j] = 1
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
            coeffs = [0] * LADimension
            if idx2 == idx1:
                return coeffs
            if idx2 < idx1:
                reSign = -1
                idx2, idx1 = idx1, idx2
            else:
                reSign = 1
            p10, p11, p12 = indexingKey[idx1]
            p20, p21, p22 = indexingKey[idx2]
            if p12 == 0:    # implies p10 = p11
                if p22 == 1:
                    if p20<=p10 and p21>p10:
                        coeffs[idx2] = reSign
                    elif p21<=p10 and p20>p10:
                        coeffs[idx2] = -reSign
            elif p12 == 1:
                if p22 == 1:
                    if p11 == p20:
                        if p10 == p21:
                            if p10 < p11:
                                if 0<p10:
                                    coeffs[indexingKeyRev[(p10-1, p10-1, 0)]] = -reSign
                                if p10==p11-1:
                                    coeffs[indexingKeyRev[(p10, p10, 0)]] = 2*reSign
                                else:
                                    coeffs[indexingKeyRev[(p10, p10, 0)]] = reSign
                                    coeffs[indexingKeyRev[(p11-1, p11-1, 0)]] = reSign
                                if p11<n:
                                    coeffs[indexingKeyRev[(p11, p11, 0)]] = -reSign
                            else:
                                if 0<p11:
                                    coeffs[indexingKeyRev[(p11-1, p11-1, 0)]] = reSign
                                if p11==p10-1:
                                    coeffs[indexingKeyRev[(p11, p11, 0)]] = -2*reSign
                                else:
                                    coeffs[indexingKeyRev[(p11, p11, 0)]] = -reSign
                                    coeffs[indexingKeyRev[(p10-1, p10-1, 0)]] = -reSign
                                if p10<n:
                                    coeffs[indexingKeyRev[(p10, p10, 0)]] = reSign
                        else:
                            coeffs[indexingKeyRev[(p10, p21, 1)]] = reSign
                    elif p10 == p21:
                        coeffs[indexingKeyRev[(p20, p11, 1)]] = -reSign
            return coeffs

        _structure_data = [[_structureCoeffs(k, j) for j in range(LADimension)] for k in range(LADimension)]
        CartanSubalg = list(hBasis["elems"].values())
        matrixBasis = CartanSubalg + list(offDiag["elems"].values())
        gradingVecs = list(hBasis["grading"].values()) + list(
            offDiag["grading"].values()
        )
        return _structure_data, list(zip(*gradingVecs)), CartanSubalg, matrixBasis

    def _generate_B_series_structure_data(n):
        matrix_dim = 2 * n + 1

        # Basis elements
        hBasis = {"elems": dict(), "grading": dict()}
        GPlus = {"elems": dict(), "grading": dict()}
        GMinus = {"elems": dict(), "grading": dict()}
        DPlus = {"elems": dict(), "grading": dict()}
        DMinus = {"elems": dict(), "grading": dict()}

        skew_symmetric = [[0] * matrix_dim for _ in range(matrix_dim)]

        def gPlusWeights(idx1, idx2):
            wVec = []
            for idx in range(n - 1):
                if (idx1 <= idx and idx2 <= idx) or (idx1 > idx and idx2 > idx):
                    wVec.append(0)
                elif idx1 <= idx:
                    wVec.append(1)
                else:
                    wVec.append(-1)
            wVec.append(0)
            return wVec

        def gMinusWeights(idx1, idx2):
            wVec = []
            sign = 1 if idx2 < idx1 else -1
            for idx in range(n - 1):
                if idx1 <= idx and idx2 <= idx:
                    if idx==0:
                        wVec.append(sign)
                    else:
                        wVec.append(2*sign)
                elif idx1 > idx and idx2 > idx:
                    wVec.append(0)
                elif idx1<=idx:
                    wVec.append(-1)
                elif idx2<=idx:
                    wVec.append(1)
                else:   # should never trigger
                    wVec.append(0)
            wVec.append(2*sign)
            return wVec

        def DWeights(idx1, sign):
            wVec = []
            for idx in range(n - 1):
                if idx1 <= idx:
                    wVec.append(-sign)
                else:
                    wVec.append(0)
            wVec.append(-sign)
            return wVec

        for j, k in carProd(range(n), range(n)):
            # Diagonal (Cartan) element
            if j == k and j < n - 1:
                M = [row[:] for row in skew_symmetric]
                for idx in range(n):
                    if idx < j+1:
                        M[2 * idx][2 * idx + 1] = sp.I
                        M[2 * idx + 1][2 * idx] = -sp.I
                hBasis["elems"][(j, k, 0)] = M
                hBasis["grading"][(j, k, 0)] = [0] * n
                if j + 2 == n:
                    M = [row[:] for row in skew_symmetric]
                    for idx in range(n):
                        M[2*idx][2*idx+1] = sp.I
                        M[2*idx+1][2*idx] = -sp.I
                    hBasis["elems"][(j + 1, k + 1, 0)] = M
                    hBasis["grading"][(j + 1, k + 1, 0)] = [0] * n
            elif j != k:
                # “+” generator
                MPlus = [row[:] for row in skew_symmetric]
                MPlus[2 * j][2 * k] = 1
                MPlus[2 * k][2 * j] = -1
                MPlus[2 * j + 1][2 * k + 1] = 1
                MPlus[2 * k + 1][2 * j + 1] = -1
                MPlus[2 * j][2 * k + 1] = sp.I
                MPlus[2 * k + 1][2 * j] = -sp.I
                MPlus[2 * j + 1][2 * k] = -sp.I
                MPlus[2 * k][2 * j + 1] = sp.I
                GPlus["elems"][(j, k, 1)] = MPlus
                GPlus["grading"][(j, k, 1)] = gPlusWeights(j, k)

                # “–” generator
                if j < k:
                    MMinus = [row[:] for row in skew_symmetric]
                    MMinus[2 * j][2 * k] = 1
                    MMinus[2 * k][2 * j] = -1
                    MMinus[2 * j + 1][2 * k + 1] = -1
                    MMinus[2 * k + 1][2 * j + 1] = 1
                    MMinus[2 * j][2 * k + 1] = sp.I
                    MMinus[2 * k + 1][2 * j] = -sp.I
                    MMinus[2 * j + 1][2 * k] = sp.I
                    MMinus[2 * k][2 * j + 1] = -sp.I
                    GMinus["elems"][(j, k, -1)] = MMinus
                    GMinus["grading"][(j, k, -1)] = gMinusWeights(j, k)
                else:  # k<j
                    MMinus = [row[:] for row in skew_symmetric]
                    MMinus[2 * k][2 * j] = 1
                    MMinus[2 * j][2 * k] = -1
                    MMinus[2 * k + 1][2 * j + 1] = -1
                    MMinus[2 * j + 1][2 * k + 1] = 1
                    MMinus[2 * k][2 * j + 1] = -sp.I
                    MMinus[2 * j + 1][2 * k] = sp.I
                    MMinus[2 * k + 1][2 * j] = -sp.I
                    MMinus[2 * j][2 * k + 1] = sp.I
                    GMinus["elems"][(j, k, -1)] = MMinus
                    GMinus["grading"][(j, k, -1)] = gMinusWeights(j, k)
        for j in range(n):
            MPlus = [row[:] for row in skew_symmetric]
            MMinus = [row[:] for row in skew_symmetric]
            MPlus[2 * j][2 * n] = 1
            MPlus[2 * n][2 * j] = -1
            MPlus[2 * j + 1][2 * n] = sp.I
            MPlus[2 * n][2 * j + 1] = -sp.I
            MMinus[2 * j][2 * n] = 1
            MMinus[2 * n][2 * j] = -1
            MMinus[2 * j + 1][2 * n] = -sp.I
            MMinus[2 * n][2 * j + 1] = sp.I
            DPlus["elems"][(j, 2 * n, 2)] = MPlus
            DPlus["grading"][(j, 2 * n, 2)] = DWeights(j, 1)
            DMinus["elems"][(j, 2 * n, -2)] = MMinus
            DMinus["grading"][(j, 2 * n, -2)] = DWeights(j, -1)

        indexingKey = dict(
            enumerate(
                list(hBasis["grading"].keys())
                + list(GPlus["grading"].keys())
                + list(GMinus["grading"].keys())
                + list(DPlus["grading"].keys())
                + list(DMinus["grading"].keys())
            )
        )
        indexingKeyRev = {j: k for k, j in indexingKey.items()}
        LADimension = len(indexingKey)
        CSDict = {
            idx: {0: 1} if idx == 0 else {idx: 1, idx - 1: -1}
            for idx in range(n)
        }  # Cartan subalgebra basis transform indexing
        CSDictInv = {
            idx: {j: 0 if j > idx else 1 for j in range(n)} for idx in range(n - 1)
        } | {n - 1: {j:1 for j in range(n)}}

        def _structureCoeffs(idx1, idx2):
            coeffs = [0] * LADimension
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
                for term, scale in CSDictInv[p10].items():
                    if p22 == 1:
                        coeffs[idx2] += (
                            scale
                            * reSign
                            * (int(term == p20) - int(term == p21))
                        )
                    elif p22 == -1:
                        sign = -reSign if p20 < p21 else reSign
                        coeffs[idx2] += (scale * sign * (int(term == p20) + int(term == p21))
                        )
                    elif p22 == 2:
                        coeffs[idx2] += (
                            -scale * (int(term == p20)) * reSign
                        )
                    elif p22 == -2:
                        coeffs[idx2] += (
                            scale * (int(term == p20)) * reSign
                        )
            elif p12 == 1:
                if p22 == 1:
                    if p11 == p20:
                        if p10 == p21:
                            # l(p10)-l(p11)
                            for t, s in CSDict[p10].items():
                                coeffs[t] += reSign * 4 * s
                            for t, s in CSDict[p11].items():
                                coeffs[t] += -reSign * 4 * s
                        else:
                            coeffs[indexingKeyRev[(p10, p21, 1)]] += 2 * reSign
                    elif p10 == p21:
                        coeffs[indexingKeyRev[(p20, p11, 1)]] += -2 * reSign
                elif p22 == -1:
                    slope1 = 1 if p10 < p11 else -1
                    slope2 = 1 if p20 < p21 else -1
                    if p10 == p20:
                        if not (slope1 == -1 and slope2 == -1):
                            if p11 < p21:
                                coeffs[indexingKeyRev[(p11, p21, -1)]] += -2 * reSign
                            elif p21 < p11:
                                if not (slope1 == 1 and slope2 == -1):
                                    coeffs[indexingKeyRev[(p21, p11, -1)]] += 2 * reSign
                    elif p11 == p21:
                        if not (slope1 == 1 and slope2 == 1):
                            if p10 < p20:
                                coeffs[indexingKeyRev[(p20, p10, -1)]] += 2 * reSign
                            elif p20 < p10:
                                if not (slope1 == -1 and slope2 == 1):
                                    coeffs[indexingKeyRev[(p10, p20, -1)]] += (
                                        -2 * reSign
                                    )
                    elif p11 == p20:
                        if not (slope1 == 1 and slope2 == 1) and not (
                            slope1 == -1 and slope2 == 1
                        ):
                            if p10 < p21:
                                coeffs[indexingKeyRev[(p21, p10, -1)]] = -2 * reSign
                            elif p21 < p10:
                                coeffs[indexingKeyRev[(p10, p21, -1)]] = 2 * reSign
                    elif p10 == p21:
                        if not (slope1 == -1 and slope2 == -1) and not (
                            slope1 == 1 and slope2 == -1
                        ):
                            if p11 < p20:
                                coeffs[indexingKeyRev[(p11, p20, -1)]] = 2 * reSign
                            elif p20 < p11:
                                coeffs[indexingKeyRev[(p20, p11, -1)]] = -2 * reSign
                elif p22 == 2:
                    if p10 == p20:
                        coeffs[indexingKeyRev[(p11, p21, 2)]] = -2 * reSign
                elif p22 == -2:
                    if p11 == p20:
                        coeffs[indexingKeyRev[(p10, p21, -2)]] = 2 * reSign
            elif p12 == -1:
                slope1 = 1 if p10 < p11 else -1
                slope2 = 1 if p20 < p21 else -1
                if p22 == -1:
                    sign2 = 1 if p10 < p11 else -1
                    if (p10 < p11 and p20 < p21) or (p10 > p11 and p20 > p21):
                        pass
                    elif p11 == p20:
                        if p10 == p21:
                            # plus/minus (l(p10)+l(p11))
                            for t, s in CSDict[p10].items():
                                coeffs[t] += sign2 * reSign * 4 * s
                            for t, s in CSDict[p11].items():
                                coeffs[t] += sign2 * reSign * 4 * s
                        else:
                            if sign2 == 1:
                                coeffs[indexingKeyRev[(p21, p10, 1)]] += (
                                    2 * reSign * sign2
                                )
                            else:
                                coeffs[indexingKeyRev[(p10, p21, 1)]] += (
                                    2 * reSign * sign2
                                )
                    elif p10 == p21:
                        if sign2 == 1:
                            coeffs[indexingKeyRev[(p20, p11, 1)]] += 2 * reSign * sign2
                        else:
                            coeffs[indexingKeyRev[(p11, p20, 1)]] += 2 * reSign * sign2
                    elif p10 == p20 and p21 != p11:
                        if sign2 == 1:
                            coeffs[indexingKeyRev[(p21, p11, 1)]] += -2 * reSign * sign2
                        else:
                            coeffs[indexingKeyRev[(p11, p21, 1)]] += -2 * reSign * sign2
                    elif p11 == p21 and p10 != p20:
                        if sign2 == 1:
                            coeffs[indexingKeyRev[(p20, p10, 1)]] += -2 * reSign * sign2
                        else:
                            coeffs[indexingKeyRev[(p10, p20, 1)]] += -2 * reSign * sign2
                elif p22 == 2:
                    if slope1 == -1:
                        if p11 == p20:
                            coeffs[indexingKeyRev[(p10, p21, -2)]] = -2 * reSign
                        elif p10 == p20:
                            coeffs[indexingKeyRev[(p11, p21, -2)]] = 2 * reSign
                elif p22 == -2:
                    if p10 == p20 and slope1 == 1:
                        coeffs[indexingKeyRev[(p11, p21, 2)]] = -2 * reSign
                    if p11 == p20:
                        if slope1 == 1:
                            coeffs[indexingKeyRev[(p10, p21, 2)]] = 2 * reSign
            elif p12 == 2:
                if p22 == 2:
                    if p10 < p20:
                        coeffs[indexingKeyRev[(p10, p20, -1)]] = -reSign
                    elif p20 < p10:
                        coeffs[indexingKeyRev[(p20, p10, -1)]] = reSign
                if p22 == -2:
                    if p10 == p20:
                        for term, scale in CSDict[p10].items():
                            coeffs[term] = 2 * scale * reSign
                    elif p10 < p20:
                        coeffs[indexingKeyRev[(p20, p10, 1)]] = reSign
                    else:
                        coeffs[indexingKeyRev[(p20, p10, 1)]] = reSign
            elif p12 == -2:
                if p22 == -2:
                    if p10 < p20:
                        coeffs[indexingKeyRev[(p20, p10, -1)]] = -reSign
                    elif p20 < p10:
                        coeffs[indexingKeyRev[(p10, p20, -1)]] = reSign
            return coeffs

        _structure_data = [
            [_structureCoeffs(k, j) for j in range(LADimension)]
            for k in range(LADimension)
        ]
        CartanSubalg = list(hBasis["elems"].values())
        matrixBasis = (
            CartanSubalg
            + list(GPlus["elems"].values())
            + list(GMinus["elems"].values())
            + list(DPlus["elems"].values())
            + list(DMinus["elems"].values())
        )
        gradingVecs = (
            list(hBasis["grading"].values())
            + list(GPlus["grading"].values())
            + list(GMinus["grading"].values())
            + list(DPlus["grading"].values())
            + list(DMinus["grading"].values())
        )
        return _structure_data, list(zip(*gradingVecs)), CartanSubalg, matrixBasis

    def _generate_C_series_structure_data(n):
        matrix_dim = 2 * n

        # Basis elements
        hBasis = {"elems": dict(), "grading": dict()}
        offDiag = {"elems": dict(), "grading": dict()}

        symplectic = [[0] * matrix_dim for _ in range(matrix_dim)]

        def gPlusWeights(idx1, idx2):
            wVec = []
            for idx in range(idx1):
                wVec.append(0)
            for idx in range(idx1,idx2):
                wVec.append(1)
            for idx in range(idx2,n-1):
                wVec.append(2)
            wVec.append(1)
            return wVec

        def gMinusWeights(idx1, idx2):
            wVec = []
            for idx in range(idx1):
                wVec.append(0)
            for idx in range(idx1,idx2):
                wVec.append(-1)
            for idx in range(idx2,n-1):
                wVec.append(-2)
            wVec.append(-1)
            return wVec


        def GLWeights(idx1, idx2):
            if idx1<idx2:
                wVec=[1 if idx1<=idx and idx<idx2 else 0 for idx in range(n)]
            else:
                wVec=[-1 if idx2<=idx and idx<idx1 else 0 for idx in range(n)]
            return wVec

        for j in range(n):
            for k in range(j,n):
                if j == k:
                    M = [row[:] for row in symplectic]
                    if j<n-1:
                        for idx in range(j+1):
                            M[idx][idx] = 1
                            M[n+idx][n+idx] = -1
                    else:
                        for idx in range(n):
                            M[idx][idx] = sp.Rational(1,2)
                            M[n+idx][n+idx] = -sp.Rational(1,2)
                    hBasis["elems"][(j, k, 0)] = M
                    hBasis["grading"][(j, k, 0)] = [0] * n

                    M = [row[:] for row in symplectic]
                    M[j][n+j] = 1
                    offDiag["elems"][(j, k, 1)] = M
                    offDiag["grading"][(j, k, 1)] = gPlusWeights(j, k)

                    M = [row[:] for row in symplectic]
                    M[n+j][j] = 1
                    offDiag["elems"][(j, k, -1)] = M
                    offDiag["grading"][(j, k, -1)] = gMinusWeights(j, k)
                else:
                    M = [row[:] for row in symplectic]
                    M[j][k] = 1
                    M[n+k][n+j] = -1
                    offDiag["elems"][(j, k, 2)] = M
                    offDiag["grading"][(j, k, 2)] = GLWeights(j, k)

                    M = [row[:] for row in symplectic]
                    M[k][j] = 1
                    M[n+j][n+k] = -1
                    offDiag["elems"][(k, j, 2)] = M
                    offDiag["grading"][(k, j, 2)] = GLWeights(k, j)

                    M = [row[:] for row in symplectic]
                    M[j][n+k] = 1
                    M[k][n+j] = 1
                    offDiag["elems"][(j, k, 1)] = M
                    offDiag["grading"][(j, k, 1)] = gPlusWeights(j, k)

                    M = [row[:] for row in symplectic]
                    M[n+j][k] = 1
                    M[n+k][j] = 1
                    offDiag["elems"][(j, k, -1)] = M
                    offDiag["grading"][(j, k, -1)] = gMinusWeights(j, k)

        indexingKey = dict(
            enumerate(
                list(hBasis["grading"].keys())
                + list(offDiag["grading"].keys())
            )
        )
        indexingKeyRev = {j: k for k, j in indexingKey.items()}
        LADimension = len(indexingKey)

        def minmaxtuple(id1,id2,id3):
            if id1<id2:
                return (id1,id2,id3)
            return (id2,id1,id3)
        def _structureCoeffs(idx1, idx2):
            coeffs = [0] * LADimension
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
                if p22 == 1:
                    coeffs[idx2] += offDiag["grading"][indexingKey[idx2]][p10]
                elif p22 == -1:
                    coeffs[idx2] += offDiag["grading"][indexingKey[idx2]][p10]
                elif p22 == 2:
                    coeffs[idx2] += offDiag["grading"][indexingKey[idx2]][p10]
            elif p12 == 1:
                if p22 == -1:
                    if p11 == p20:
                        coeffs[indexingKeyRev[(p10,p21,2*int(p10!=p21))]] += reSign
                    elif p11 == p21:
                        coeffs[indexingKeyRev[(p10,p20,2*int(p10!=p20))]] += reSign
                    if p11!=p10:
                        if p10 == p20:
                            coeffs[indexingKeyRev[(p11,p21,2*int(p11!=p21))]] += reSign
                        elif p10 == p21:
                            coeffs[indexingKeyRev[(p11,p20,2*int(p11!=p20))]] += reSign
                elif p22 == 2:
                    if p11 == p21 and p10==p20:     ###!!! check second condition
                        coeffs[indexingKeyRev[minmaxtuple(p10,p20,1)]] += -reSign
                    if p10 == p21:
                        coeffs[indexingKeyRev[minmaxtuple(p11,p20,1)]] += -reSign
                    if p11!=p10:
                        if p10 == p21 and p11==p20: ###!!! check second condition
                            coeffs[indexingKeyRev[minmaxtuple(p11,p20,1)]] += -reSign
                        if p11 == p21:
                            coeffs[indexingKeyRev[minmaxtuple(p10,p20,1)]] += -reSign

            elif p12 == -1:
                if p22 == 1:
                    if p11 == p20:
                        coeffs[indexingKeyRev[(p21,p10,2*int(p10!=p21))]] += -reSign
                    elif p11 == p21:
                        coeffs[indexingKeyRev[(p20,p10,2*int(p10!=p20))]] += -reSign
                    if p11!=p10:
                        if p10 == p20:
                            coeffs[indexingKeyRev[(p21,p11,2*int(p11!=p21))]] += -reSign
                        elif p10 == p21:
                            coeffs[indexingKeyRev[(p20,p11,2*int(p11!=p20))]] += -reSign
                elif p22 == 2:
                    if p11 == p20 and p10==p21:     ###!!! check second condition
                        coeffs[indexingKeyRev[minmaxtuple(p10,p21,-1)]] += reSign
                    if p10 == p20:
                        coeffs[indexingKeyRev[minmaxtuple(p11,p21,-1)]] += reSign
                    if p11!=p10:
                        if p10 == p20 and p11 == p21:   ###!!! check second condition
                            coeffs[indexingKeyRev[minmaxtuple(p11,p21,-1)]] += reSign
                        if p11 == p20:
                            coeffs[indexingKeyRev[minmaxtuple(p10,p21,-1)]] += reSign

            elif p12 == 2:
                if p22 == 1:
                    if p21 == p11 and p20==p10:     ###!!! check second condition
                        coeffs[indexingKeyRev[minmaxtuple(p20,p10,1)]] += reSign
                    if p20 == p11:
                        coeffs[indexingKeyRev[minmaxtuple(p21,p10,1)]] += reSign
                    if p20!=p21:
                        if p20 == p11 and p21==p10: ###!!! check second condition
                            coeffs[indexingKeyRev[minmaxtuple(p21,p10,1)]] += reSign
                        if p21 == p11:
                            coeffs[indexingKeyRev[minmaxtuple(p20,p10,1)]] += reSign
                elif p22 == -1:
                    if p21 == p10 and p20==p11:     ###!!! check second condition
                        coeffs[indexingKeyRev[minmaxtuple(p20,p11,-1)]] += -reSign
                    if p20 == p10:
                        coeffs[indexingKeyRev[minmaxtuple(p21,p11,-1)]] += -reSign
                    if p20!=p21:
                        if p20 == p10 and p21==p11:  ###!!! check second condition
                            coeffs[indexingKeyRev[minmaxtuple(p21,p11,-1)]] += -reSign
                        if p21 == p10:
                            coeffs[indexingKeyRev[minmaxtuple(p20,p11,-1)]] += -reSign

                elif p22 == 2:
                    if p11 == p20:
                        coeffs[indexingKeyRev[(p10,p21,2*int(p10!=p21))]] += reSign
                    if p10 == p21:
                        coeffs[indexingKeyRev[(p20,p11,2*int(p20!=p11))]] += -reSign
            return coeffs
        _structure_data = [
            [_structureCoeffs(k, j) for j in range(LADimension)]
            for k in range(LADimension)
        ]
        CartanSubalg = list(hBasis["elems"].values())
        matrixBasis = CartanSubalg + list(offDiag["elems"].values())
        gradingVecs = list(hBasis["grading"].values()) + list(
            offDiag["grading"].values()
        )
        return _structure_data, list(zip(*gradingVecs)), CartanSubalg, matrixBasis

    def _generate_D_series_structure_data(n):
        matrix_dim = 2 * n

        # Basis elements
        hBasis = {"elems": dict(), "grading": dict()}
        GPlus = {"elems": dict(), "grading": dict()}
        GMinus = {"elems": dict(), "grading": dict()}

        skew_symmetric = [[0] * matrix_dim for _ in range(matrix_dim)]

        def gPlusWeights(idx1, idx2):
            wVec = []
            for idx in range(n - 2):
                if (idx1 <= idx and idx2 <= idx) or (idx1 > idx and idx2 > idx):
                    wVec.append(0)
                elif idx1 <= idx:
                    wVec.append(1)
                else:
                    wVec.append(-1)
            if (idx1 < n - 1 and idx2 < n-1) or (idx1 > n - 2 and idx2 > n - 2):
                wVec.append(0)
            elif idx1 < n - 1:
                wVec.append(1)
            else:
                wVec.append(-1)
            wVec.append(0)
            return wVec

        def gMinusWeights(idx1, idx2):
            wVec = []
            sign = 1 if idx2 < idx1 else -1
            for idx in range(n - 2):
                if idx1 <= idx and idx2 <= idx:
                    wVec.append(2*sign)
                elif idx1 > idx and idx2 > idx:
                    wVec.append(0)
                else:
                    wVec.append(sign)
            if idx1 < n - 1 and idx2 < n - 1:
                wVec.append(sign)
            elif idx1 > n - 2 and idx2 > n - 2:
                wVec.append(-sign)
            else:
                wVec.append(0)
            wVec.append(sign)
            return wVec

        for j, k in carProd(range(n), range(n)):
            # Diagonal (Cartan) element
            if j == k and j < n - 1:
                M = [row[:] for row in skew_symmetric]
                if j<n-2:
                    for idx in range(j+1):
                        M[2 * idx][2 * idx + 1] = sp.I
                        M[2 * idx + 1][2 * idx] = -sp.I
                    hBasis["elems"][(j, k, 0)] = M
                    hBasis["grading"][(j, k, 0)] = [0] * n
                else:
                    for idx in range(n):
                        if idx > j:
                            M[2 * idx][2 * idx + 1] = -sp.I / 2
                            M[2 * idx + 1][2 * idx] = sp.I / 2
                        else:
                            M[2 * idx][2 * idx + 1] = sp.I / 2
                            M[2 * idx + 1][2 * idx] = -sp.I / 2
                    hBasis["elems"][(j, k, 0)] = M
                    hBasis["grading"][(j, k, 0)] = [0] * n
                    M = [row[:] for row in skew_symmetric]
                    for idx in range(n):
                        M[2 * idx][2 * idx + 1] = sp.I / 2
                        M[2 * idx + 1][2 * idx] = -sp.I / 2
                    hBasis["elems"][(j + 1, k + 1, 0)] = M
                    hBasis["grading"][(j + 1, k + 1, 0)] = [0] * n
            elif j != k:
                # “+” generator
                MPlus = [row[:] for row in skew_symmetric]
                MPlus[2 * j][2 * k] = 1
                MPlus[2 * k][2 * j] = -1
                MPlus[2 * j + 1][2 * k + 1] = 1
                MPlus[2 * k + 1][2 * j + 1] = -1
                MPlus[2 * j][2 * k + 1] = sp.I
                MPlus[2 * k + 1][2 * j] = -sp.I
                MPlus[2 * j + 1][2 * k] = -sp.I
                MPlus[2 * k][2 * j + 1] = sp.I
                GPlus["elems"][(j, k, 1)] = MPlus
                GPlus["grading"][(j, k, 1)] = gPlusWeights(j, k)

                # “–” generator
                if j < k:
                    MMinus = [row[:] for row in skew_symmetric]
                    MMinus[2 * j][2 * k] = 1
                    MMinus[2 * k][2 * j] = -1
                    MMinus[2 * j + 1][2 * k + 1] = -1
                    MMinus[2 * k + 1][2 * j + 1] = 1
                    MMinus[2 * j][2 * k + 1] = sp.I
                    MMinus[2 * k + 1][2 * j] = -sp.I
                    MMinus[2 * j + 1][2 * k] = sp.I
                    MMinus[2 * k][2 * j + 1] = -sp.I
                    GMinus["elems"][(j, k, -1)] = MMinus
                    GMinus["grading"][(j, k, -1)] = gMinusWeights(j, k)
                else:  # k<j
                    MMinus = [row[:] for row in skew_symmetric]
                    MMinus[2 * k][2 * j] = 1
                    MMinus[2 * j][2 * k] = -1
                    MMinus[2 * k + 1][2 * j + 1] = -1
                    MMinus[2 * j + 1][2 * k + 1] = 1
                    MMinus[2 * k][2 * j + 1] = -sp.I
                    MMinus[2 * j + 1][2 * k] = sp.I
                    MMinus[2 * k + 1][2 * j] = -sp.I
                    MMinus[2 * j][2 * k + 1] = sp.I
                    GMinus["elems"][(j, k, -1)] = MMinus
                    GMinus["grading"][(j, k, -1)] = gMinusWeights(j, k)

        indexingKey = dict(
            enumerate(
                list(hBasis["grading"].keys())
                + list(GPlus["grading"].keys())
                + list(GMinus["grading"].keys())
            )
        )
        indexingKeyRev = {j: k for k, j in indexingKey.items()}
        LADimension = len(indexingKey)
        if n==1:
            CSDict = {0:{0:2}}  # Cartan subalgebra basis transform indexing
        elif n==2:
            CSDict = {0:{0:1,1:1},1:{0:-1,1:1}}
        else:
            CSDict = {idx: {0: 1} if idx == 0 else {idx: 1, idx - 1: -1}
                for idx in range(n-2)} | {n-2:{n-2:1,n-1:1,n-3:-1},n-1:{n-2:-1,n-1:1}}
        CSDictInv = {
            idx: {j: 0 if j > idx else 1 for j in range(n)} for idx in range(n - 2)
        } | {n - 1: {j: sp.Rational(1, 2) for j in range(n)}}
        if n>2:
            CSDictInv|={n-2:{j: sp.Rational(-1, 2) if j > n-2 else sp.Rational(1, 2) for j in range(n)}}

        def _structureCoeffs(idx1, idx2):
            coeffs = [0] * LADimension
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
                for term, scale in CSDictInv[p10].items():
                    if p22 == 1:
                        coeffs[idx2] += (
                            scale
                            * reSign
                            * (int(term == p20) - int(term == p21))
                        )
                    elif p22 == -1:
                        if p20<p21:
                            if p21<=p10:
                                sign=-2
                            else:
                                sign=-1
                        else:
                            if p21<=p10:
                                sign=-2
                            else:
                                sign=2
                        sign = -reSign if p20 < p21 else reSign
                        coeffs[idx2] += (
                            scale
                            * sign
                            * (int(term == p20) + int(term == p21))
                        )
            elif p12 == 1:
                if p22 == 1:
                    if p11 == p20:
                        if p10 == p21:
                            # l(p10)-l(p11)
                            for t, s in CSDict[p10].items():
                                coeffs[t] += reSign * 4 * s
                            for t, s in CSDict[p11].items():
                                coeffs[t] += -reSign * 4 * s
                        else:
                            coeffs[indexingKeyRev[(p10, p21, 1)]] += 2 * reSign
                    elif p10 == p21:
                        coeffs[indexingKeyRev[(p20, p11, 1)]] += -2 * reSign
                else:
                    slope1 = 1 if p10 < p11 else -1
                    slope2 = 1 if p20 < p21 else -1
                    if p10 == p20:
                        if not (slope1 == -1 and slope2 == -1):
                            if p11 < p21:
                                coeffs[indexingKeyRev[(p11, p21, -1)]] += -2 * reSign
                            elif p21 < p11:
                                if not (slope1 == 1 and slope2 == -1):
                                    coeffs[indexingKeyRev[(p21, p11, -1)]] += 2 * reSign
                    elif p11 == p21:
                        if not (slope1 == 1 and slope2 == 1):
                            if p10 < p20:
                                coeffs[indexingKeyRev[(p20, p10, -1)]] += 2 * reSign
                            elif p20 < p10:
                                if not (slope1 == -1 and slope2 == 1):
                                    coeffs[indexingKeyRev[(p10, p20, -1)]] += (
                                        -2 * reSign
                                    )
                    elif p11 == p20:
                        if not (slope1 == 1 and slope2 == 1) and not (
                            slope1 == -1 and slope2 == 1
                        ):
                            if p10 < p21:
                                coeffs[indexingKeyRev[(p21, p10, -1)]] = -2 * reSign
                            elif p21 < p10:
                                coeffs[indexingKeyRev[(p10, p21, -1)]] = 2 * reSign
                    elif p10 == p21:
                        if not (slope1 == -1 and slope2 == -1) and not (
                            slope1 == 1 and slope2 == -1
                        ):
                            if p11 < p20:
                                coeffs[indexingKeyRev[(p11, p20, -1)]] = 2 * reSign
                            elif p20 < p11:
                                coeffs[indexingKeyRev[(p20, p11, -1)]] = -2 * reSign
            else:
                sign2 = 1 if p10 < p11 else -1
                if (p10 < p11 and p20 < p21) or (p10 > p11 and p20 > p21):
                    pass
                elif p11 == p20:
                    if p10 == p21:
                        # plus/minus (l(p10)+l(p11))
                        for t, s in CSDict[p10].items():
                            coeffs[t] += sign2 * reSign * 4 * s
                        for t, s in CSDict[p11].items():
                            coeffs[t] += sign2 * reSign * 4 * s
                    else:
                        if sign2 == 1:
                            coeffs[indexingKeyRev[(p21, p10, 1)]] += 2 * reSign * sign2
                        else:
                            coeffs[indexingKeyRev[(p10, p21, 1)]] += 2 * reSign * sign2
                elif p10 == p21:
                    if sign2 == 1:
                        coeffs[indexingKeyRev[(p20, p11, 1)]] += 2 * reSign * sign2
                    else:
                        coeffs[indexingKeyRev[(p11, p20, 1)]] += 2 * reSign * sign2
                elif p10 == p20 and p21 != p11:
                    if sign2 == 1:
                        coeffs[indexingKeyRev[(p21, p11, 1)]] += -2 * reSign * sign2
                    else:
                        coeffs[indexingKeyRev[(p11, p21, 1)]] += -2 * reSign * sign2
                elif p11 == p21 and p10 != p20:
                    if sign2 == 1:
                        coeffs[indexingKeyRev[(p20, p10, 1)]] += -2 * reSign * sign2
                    else:
                        coeffs[indexingKeyRev[(p10, p20, 1)]] += -2 * reSign * sign2
            return coeffs

        _structure_data = [
            [_structureCoeffs(k, j) for j in range(LADimension)]
            for k in range(LADimension)
        ]
        CartanSubalg = list(hBasis["elems"].values())
        matrixBasis = (
            CartanSubalg
            + list(GPlus["elems"].values())
            + list(GMinus["elems"].values())
        )
        gradingVecs = (
            list(hBasis["grading"].values())
            + list(GPlus["grading"].values())
            + list(GMinus["grading"].values())
        )
        return _structure_data, list(zip(*gradingVecs)), CartanSubalg, matrixBasis

    if series_type == "A":
        default_label = f"sl{rank + 1}" if label is None else label
        structure_data, grading, CartanSubalgebra, matrixBasis = _generate_A_series_structure_data(rank)
        passkey = retrieve_passkey()
        if build_standard_mat_rep is True:
            return createAlgebra(
                matrixBasis,
                label=default_label,
                basis_labels=basis_labels,
                grading=grading,
                process_matrix_rep=True,
                preferred_representation=matrixBasis,
                _simple={
                    "lockKey": passkey,
                    "CartanSubalgebra": CartanSubalgebra,
                    "type": [series_type, rank],
                },
                return_created_obj=return_created_obj,
                forgo_vmf_registry=forgo_vmf_registry
            )
        else:
            return createAlgebra(
                structure_data,
                label=default_label,
                basis_labels=basis_labels,
                grading=grading,
                preferred_representation=matrixBasis,
                _simple={
                    "lockKey": passkey,
                    "CartanSubalgebra": CartanSubalgebra,
                    "type": [series_type, rank],
                },
                return_created_obj=return_created_obj,
                forgo_vmf_registry=forgo_vmf_registry
            )

    elif series_type == "B":
        default_label = f"so{2*rank + 1}" if label is None else label
        structure_data, grading, CartanSubalgebra, matrixBasis = (
            _generate_B_series_structure_data(rank)
        )
        passkey = retrieve_passkey()
        if build_standard_mat_rep is True:
            return createAlgebra(
                matrixBasis,
                label=default_label,
                basis_labels=basis_labels,
                grading=grading,
                process_matrix_rep=True,
                preferred_representation=matrixBasis,
                _simple={
                    "lockKey": passkey,
                    "CartanSubalgebra": CartanSubalgebra,
                    "type": [series_type, rank],
                },
                return_created_obj=return_created_obj,
                forgo_vmf_registry=forgo_vmf_registry
            )
        else:
            return createAlgebra(
                structure_data,
                label=default_label,
                basis_labels=basis_labels,
                grading=grading,
                preferred_representation=matrixBasis,
                _simple={
                    "lockKey": passkey,
                    "CartanSubalgebra": CartanSubalgebra,
                    "type": [series_type, rank],
                },
                return_created_obj=return_created_obj,
                forgo_vmf_registry=forgo_vmf_registry
            )

    elif series_type == "C":
        default_label = f"sp{rank}" if label is None else label
        structure_data, grading, CartanSubalgebra, matrixBasis = (
            _generate_C_series_structure_data(rank)
        )

        if build_standard_mat_rep is True:
            return createAlgebra(
                matrixBasis,
                label=default_label,
                basis_labels=basis_labels,
                grading=grading,
                process_matrix_rep=True,
                preferred_representation=matrixBasis,
                _simple={
                    "lockKey": retrieve_passkey(),
                    "CartanSubalgebra": CartanSubalgebra,
                    "type": [series_type, rank],
                },
                return_created_obj=return_created_obj,
                forgo_vmf_registry=forgo_vmf_registry
            )
        else:
            return createAlgebra(
                structure_data,
                label=default_label,
                basis_labels=basis_labels,
                grading=grading,
                preferred_representation=matrixBasis,
                _simple={
                    "lockKey": retrieve_passkey(),
                    "CartanSubalgebra": CartanSubalgebra,
                    "type": [series_type, rank],
                },
                return_created_obj=return_created_obj,
                forgo_vmf_registry=forgo_vmf_registry
            )

    elif series_type == "D":
        default_label = f"so{2*rank}" if label is None else label
        structure_data, grading, CartanSubalgebra, matrixBasis = (
            _generate_D_series_structure_data(rank)
        )
        passkey = retrieve_passkey()
        if build_standard_mat_rep is True:
            return createAlgebra(
                matrixBasis,
                label=default_label,
                basis_labels=basis_labels,
                grading=grading,
                process_matrix_rep=True,
                preferred_representation=matrixBasis,
                _simple={
                    "lockKey": passkey,
                    "CartanSubalgebra": CartanSubalgebra,
                    "type": [series_type, rank],
                },
                return_created_obj=return_created_obj,
                forgo_vmf_registry=forgo_vmf_registry
            )
        else:
            return createAlgebra(
                structure_data,
                label=default_label,
                basis_labels=basis_labels,
                grading=grading,
                preferred_representation=matrixBasis,
                _simple={
                    "lockKey": passkey,
                    "CartanSubalgebra": CartanSubalgebra,
                    "type": [series_type, rank],
                },
                return_created_obj=return_created_obj,
                forgo_vmf_registry=forgo_vmf_registry
            )

    elif series_type + str(rank) in {"G2", "F4", "E6", "E7", "E8"}:
        raise ValueError(
            "Exceptional Lie algebras are not yet supported by `createSimpleLieAlgebra`."
        ) from None

    else:
        raise ValueError(
            f"Invalid series parameter format: {series}. Expected a letter 'A', 'B', 'C', 'D', 'E', 'F', or 'G' followed by a positive integer, like 'A1', 'B5', etc. For the exceptional LA labels 'E', 'F', and 'G' the integer must be among the classified types (i.e., only 'G2', 'F4', 'E6', 'E7', and 'E8' are admissible)."
        ) from None


def createFiniteAlg(
    obj,
    label,
    basis_labels=None,
    grading=None,
    format_sparse=False,
    process_matrix_rep=False,
    preferred_representation=None,
    verbose=False,
    assume_skew=False,
    assume_Lie_alg=False,
    basis_order_for_supplied_str_eqns=None,
    _simple=None,
):
    warnings.warn(
        "`createFiniteAlg` has been deprecated as it is being replaced with a more genera function. "
        "It will be removed in 2026. Use `createAlgebra` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return createAlgebra(
        obj,
        label,
        basis_labels=basis_labels,
        grading=grading,
        format_sparse=format_sparse,
        process_matrix_rep=process_matrix_rep,
        preferred_representation=preferred_representation,
        verbose=verbose,
        assume_skew=assume_skew,
        assume_Lie_alg=assume_Lie_alg,
        basis_order_for_supplied_str_eqns=basis_order_for_supplied_str_eqns,
        _simple=_simple,
    )

def createAlgebra(
    obj,
    label,
    basis_labels=None,
    grading=None,
    format_sparse=False,
    process_matrix_rep=False,
    preferred_representation=None,
    matrix_representation = None,
    tensor_representation = None,
    verbose=False,
    assume_skew=False,
    assume_Lie_alg=False,
    basis_order_for_supplied_str_eqns=None,
    _simple=None,
    return_created_obj=False,
    forgo_vmf_registry=False,
    simplify_products_by_default=None,
    initial_basis_index=1,
    allow_natural_basis_reordering:bool|None=None,
    _basis_labels_parent=None,
    _markers={}
):
    """
    Registers an algebra object and its basis elements in the caller's global namespace,
    and adds them to the variable_registry for tracking in the Variable Management Framework.

    Parameters
    ----------
    obj : algebra, structure data, or list of algebra_element_class
        The algebra object (an instance of algebra), the structure data used to create one,
        or a list of algebra_element_class instances with the same parent algebra.
    label : str
        The label used to reference the algebra object in the global namespace.
    basis_labels : list, optional
        A list of custom labels for the basis elements of the algebra.
        If not provided, default labels will be generated.
    grading : list of lists or list, optional
        A list specifying the grading(s) of the algebra.
    format_sparse : bool, optional
        Whether to use sparse arrays when creating the algebra object.
    process_matrix_rep : bool, optional
        Whether to compute and store the matrix representation of the algebra.
    verbose : bool, optional
        If True, provides information during the creation process.
    """
    notes={}
    _markers['_educed_properties']=dict()
    if get_dgcv_category(obj) == "Tanaka_symbol":
        t_message='True by construction: data --> `Tanaka_symbol` --> `createAlgebra`'
        _markers['_educed_properties']['is_Lie_algebra']=t_message
        _markers['_educed_properties']['is_skew']=t_message
        _markers['_educed_properties']['satisfies_Jacobi_ID']=t_message
        if grading is not None:
            warnings.warn(
                "When processing a `Tanaka_symbol` object, `createAlgebra` uses the symbol's internally defined grading rather than a manually supplied grading. You are getting this warning because an additional grading was manually supplied. To apply the custom grading instead, extract the symbol object's structure data using `Tanaka_symbol.export_algebra_data()`, and then pass that to `createAlgebra` -- create the algebra first and extract the data from the created `algebra_class` attributes."
            )
        if allow_natural_basis_reordering is None:
            allow_natural_basis_reordering=False
        preserve_negative_part_basis = not allow_natural_basis_reordering
        symbolData = obj.export_algebra_data(_internal_call_lock=retrieve_passkey(),preserve_negative_part_basis=preserve_negative_part_basis)
        if isinstance(symbolData, str):
            raise TypeError(symbolData + " So no `createAlgebra` did not instantiate a new algebra.") from None
        obj = symbolData["structure_data"]
        grading = symbolData["grading"]

    passkey = retrieve_passkey()
    if (_markers.get('sum', False) or _markers.get('prod', False)) and _markers.get('lockKey', None) == passkey:
        incoming_tex_label = _markers.get('_tex_label', None)
        if incoming_tex_label is None:
            label = unique_label(label)
        else:
            label, _markers['_tex_label'] = unique_label(label, tex_label=incoming_tex_label)
    if (label in listVar(algebras_only=True) and get_dgcv_settings_registry()["forgo_warnings"] is not True):
        if isinstance(_simple, dict) and _simple.get("lockKey", None) == passkey:
            callFunction = "createSimpleLieAlgebra"
        else:
            callFunction = "createAlgebra"
        warnings.warn(
            f"`{callFunction}` was called with a `label` parameter already assigned to another algebra, so `{callFunction}` will overwrite the other algebra in the VMF and global namespace."
        )
        clearVar(label)

    def extract_structure_from_elements(elements,markers):
        """
        Computes structure constants and validates linear independence from a list of algebra_element_class.

        Parameters
        ----------
        elements : list of algebra_element_class
            A list of algebra_element_class instances.

        Returns
        -------
        structure_data : list of lists of lists
            The structure constants for the subalgebra spanned by the elements.

        Raises
        ------
        ValueError
            If the elements are not linearly independent or not closed under the algebra product.
        """
        if isinstance(elements, (list, tuple)):
            elements = [(elem.ambient_rep if get_dgcv_category(elem) == "subalgebra_element" else elem) for elem in elements]
        if not elements or not all(isinstance(el, algebra_element_class) for el in elements):
            raise ValueError(
                "Invalid input: All elements must be instances of algebra_element_class."
            ) from None
        parent_algebra = elements[0].algebra
        if parent_algebra._lie_algebra_cache is True:
            t_message='True by inheritance: subalgebra of Lie algebra'
            markers['_educed_properties']['is_Lie_algebra']=t_message
            markers['_educed_properties']['is_skew']=t_message
            markers['_educed_properties']['satisfies_Jacobi_ID']=t_message
        else:
            if parent_algebra._jacobi_identity_cache is True:
                markers['_educed_properties']['satisfies_Jacobi_ID']='True by inheritance: subalgebra of Jacobi satisfying algebra'
            if parent_algebra._skew_symmetric_cache is True:
                markers['_educed_properties']['is_skew']='True by inheritance: subalgebra of a skew symmetric algebra'

        if not all(el.algebra == parent_algebra for el in elements):
            raise ValueError("All algebra_element_class instances must share the same parent algebra.") from None
        try:
            result = parent_algebra.is_subspace_subalgebra(elements, return_structure_data=True)
        except ValueError as e:
            raise ValueError(
                "Error during subalgebra validation. "
                "The input list of algebra_element_class instances must be linearly independent and closed under the algebra product. "
                f"Original error: {e}"
            ) from e
        if not result["linearly_independent"]:
            raise ValueError("The input elements are not linearly independent. ") from None
        if not result["closed_under_product"]:
            raise ValueError("The input elements are not closed under the algebra product. ") from None
        return result["structure_data"]

    if get_dgcv_category(obj) == 'algebra_subspace':
        try:
            obj=obj.ambient.subalgebra(obj)
        except dgcv_exception_note as e:
           raise SystemExit(e)

    if isinstance(obj,numbers.Integral) and obj>=0:
        obj=(((0,)*obj,)*obj,)*obj
        t_message='True by construction: abelian data --> `createAlgebra`'
        _markers['_educed_properties']['is_Lie_algebra']=t_message
        _markers['_educed_properties']['is_skew']=t_message
        _markers['_educed_properties']['satisfies_Jacobi_ID']=t_message
        _markers['_educed_properties']['is_nilpotent']=t_message
        _markers['_educed_properties']['is_solvable']=t_message
        _markers['_educed_properties']['special_type']='abelian'
    if get_dgcv_category(obj) in {'algebra','subalgebra'}:
        if verbose:
            print(f"Using existing algebra instance: {label}")
        _markers['_educed_properties']=getattr(obj,'_educed_properties',dict())
        structure_data = obj.structureData
        dimension = obj.dimension
        if grading is None:
            grading = getattr(obj,'grading',None)
    elif isinstance(obj, (list, tuple)) and len(obj)==0:
        structure_data=tuple()
        dimension=0
    elif isinstance(obj, (list, tuple)) and all(get_dgcv_category(el) in {'algebra_element_class','subalgebra_element_class'} for el in obj):
        if verbose:
            print("Creating algebra from list of algebra_element_class instances.")
        structure_data = extract_structure_from_elements(obj,_markers)
        dimension = len(obj)
    elif (isinstance(obj, (list, tuple)) and all(get_dgcv_category(el) == 'tensorProduct' for el in obj)):
        notes['process_tensor_rep']=True
        if verbose:
            print("Creating algebra from list of tensorProduct instances.")
        try:
            vsd = _validate_structure_data(obj,process_matrix_rep=False,assume_skew=assume_skew,assume_Lie_alg=assume_Lie_alg,basis_order_for_supplied_str_eqns=basis_order_for_supplied_str_eqns,process_tensor_rep=True
            )
            if tensor_representation is not None:
                warnings.warn('The primary object given to `createAlgebra` was a list of tensorProduct instances, but a secondary value fo `tensor_representation` representation was given. The latter was ignored.')
            structure_data,tensor_representation = vsd[0][0],vsd[0][1]

        except dgcv_exception_note as e:
            raise SystemExit(e)
        dimension = len(structure_data)
    else:
        if verbose:
            print("processing structure data...")
        try:
            vsd = _validate_structure_data(
                obj,
                process_matrix_rep=process_matrix_rep,
                assume_skew=assume_skew,
                assume_Lie_alg=assume_Lie_alg,
                basis_order_for_supplied_str_eqns=basis_order_for_supplied_str_eqns,
            )
            if vsd[-1]=='matrix':
                t_message='True by construction: list of matrices --> `createAlgebra`'
                _markers['_educed_properties']['is_Lie_algebra']=t_message
                _markers['_educed_properties']['is_skew']=t_message
                _markers['_educed_properties']['satisfies_Jacobi_ID']=t_message
                _markers['parameters']=vsd[0][2]
                structure_data,matrix_representation = vsd[0][0],vsd[0][1]
            elif vsd[-1]=='tensor':
                notes['process_tensor_rep']=True
                _markers['parameters']=vsd[0][2]
                structure_data, tensor_representation = vsd[0][0], vsd[0][1]
            else:
                if isinstance(obj,(list,tuple)) and len(obj)>1 and query_dgcv_categories(obj[-1],'vector_field'):
                    t_message='True by construction: list of vector fields --> `createAlgebra`'
                    _markers['_educed_properties']['is_Lie_algebra']=t_message
                    _markers['_educed_properties']['is_skew']=t_message
                    _markers['_educed_properties']['satisfies_Jacobi_ID']=t_message
                structure_data=vsd[0]
                _markers['parameters']=vsd[1]

        except dgcv_exception_note as e:
            raise SystemExit(e)
        dimension = len(structure_data)

    if (_markers.get('sum', False) or _markers.get('prod', False)) and _markers.get('lockKey', None) == passkey:
        if basis_labels is None:
            initial_names = [f"{label}_{i+1}" for i in range(dimension)]
            _basis_labels_parent = True
        elif isinstance(basis_labels, str):
            initial_names = [f"{basis_labels}_{i+initial_basis_index}" for i in range(dimension)]
        else:
            initial_names = list(basis_labels)

        incoming_tex_basis = list(_markers.get('_tex_basis_labels', []) or [])
        have_tex_basis = len(incoming_tex_basis) == len(initial_names) and len(initial_names) > 0

        batch_protected = {label}
        new_basis = []
        new_tex_basis = [] if have_tex_basis else None
        for idx, base_lbl in enumerate(initial_names):
            if have_tex_basis:
                bl, tl = unique_label(base_lbl, tex_label=incoming_tex_basis[idx], protected=batch_protected)
                new_basis.append(bl)
                new_tex_basis.append(tl)
                batch_protected.add(bl)
            else:
                bl = unique_label(base_lbl, protected=batch_protected)
                new_basis.append(bl)
                batch_protected.add(bl)

        basis_labels = new_basis
        if have_tex_basis:
            _markers['_tex_basis_labels'] = new_tex_basis
    else:# check for redundancy here
        if basis_labels is None:
            basis_labels = [validate_label(f"{label}_{i+1}") for i in range(dimension)]
            _basis_labels_parent = True
        elif isinstance(basis_labels, str):
            basis_labels = [validate_label(f"{basis_labels}_{i+initial_basis_index}") for i in range(dimension)]
        else:
            validate_label_list(basis_labels)

    if grading is None:
        if notes.get('process_tensor_rep',False) is True:
            w=None
            changed=None
            weights=[]
            for elem in tensor_representation:
                wts=elem.compute_weight()
                if isinstance(wts,str):
                    changed='break'
                    break
                weights.append(wts)
                if w is None:
                    w=len(wts)
                    changed=False
                elif len(wts)<w:
                    w=len(wts)
                    if changed is False:
                        changed = True
            if changed!='break':
                if changed is True:
                    weights=[elem[:w] for elem in weights]
                grading = list(zip(*weights))
        else:
            grading = [tuple([0] * dimension)]
    elif isinstance(grading, (list, tuple)) and all(
        isinstance(w, _get_expr_num_types()) for w in grading
    ):
        if len(grading) != dimension:
            raise ValueError(
                f"Grading vector length ({len(grading)}) must match the algebra dimension ({dimension})."
            ) from None
        grading = [tuple(grading)]
    elif isinstance(grading, (list, tuple)) and all(
        isinstance(vec, (list, tuple)) for vec in grading
    ):
        for vec in grading:
            if len(vec) != dimension:
                raise ValueError(
                    f"Grading vector length ({len(vec)}) must match the algebra dimension ({dimension})."
                ) from None
        grading = [tuple(vec) for vec in grading]
    else:
        raise ValueError(
            f"Grading must be a single vector or a list of vectors. Recieved {grading}"
        ) from None

    if isinstance(_simple, dict) and _simple.get("lockKey", None) == passkey:
        algebra_obj = simple_Lie_algebra(
            structure_data=structure_data,
            grading=grading,
            format_sparse=format_sparse,
            process_matrix_rep=process_matrix_rep,
            preferred_representation=preferred_representation,
            _label=label,
            _basis_labels=basis_labels,
            _calledFromCreator=passkey,
            _simple_data=_simple,
            _basis_labels_parent=_basis_labels_parent
        )
    else:
        if _markers.get("lockKey", None) == passkey:
            _markers = {k:v for k,v in _markers.items() if k!="lockKey"}  
        elif '_educed_properties' in _markers: 
            _markers={'_educed_properties':_markers['_educed_properties'],'parameters':_markers.get('parameters',set())}
        else:
            _markers={'parameters':_markers.get('parameters',set())}
        algebra_obj = algebra_class(
            structure_data=structure_data,
            grading=grading,
            format_sparse=format_sparse,
            process_matrix_rep=process_matrix_rep,
            preferred_representation=preferred_representation,
            simplify_products_by_default=simplify_products_by_default,
            matrix_representation = matrix_representation,
            tensor_representation = tensor_representation,
            _label=label,
            _basis_labels=basis_labels,
            _calledFromCreator=passkey,
            _basis_labels_parent=_basis_labels_parent,
            _markers=_markers
        )

    if forgo_vmf_registry is False:
        _cached_caller_globals.update({label: algebra_obj})
        _cached_caller_globals.update(zip(basis_labels, algebra_obj.basis))

        variable_registry = get_variable_registry()
        variable_registry["finite_algebra_systems"][label] = {
            "family_type": "algebra",
            "family_names": tuple(basis_labels),
            "family_values": tuple(algebra_obj.basis),
            "dimension": dimension,
            "grading": grading,
            "basis_labels": basis_labels,
            "structure_data": structure_data,
        }
        variable_registry["_labels"][label] = {
            "path": ("finite_algebra_systems", label),
            "children": set(basis_labels),
        }

    if verbose:
        if forgo_vmf_registry is False:
            print(f"Algebra '{label}' registered successfully.")
        print(
            f"Created an algebra with the following properties. Dimension: {dimension}, Grading: {grading}, Basis Labels: {basis_labels}"
        )
    if return_created_obj is True:
        return algebra_obj
