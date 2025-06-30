############## dependencies
import re
import warnings

import pandas as pd
import sympy as sp

from .._config import (
    _cached_caller_globals,
    dgcv_exception_note,
    get_dgcv_settings_registry,
    greek_letters,
)
from .._safeguards import (
    create_key,
    get_dgcv_category,
    retrieve_passkey,
    retrieve_public_key,
)
from ..dgcv_core import variableProcedure
from ..solvers import solve_dgcv
from ..styles import get_style
from ..tensors import tensorProduct
from ..vmf import clearVar, listVar
from .algebras_aux import _validate_structure_data

############## Algebras


# finite dimensional algebra class
class algebra_class(sp.Basic):
    def __new__(cls, structure_data, *args, **kwargs):
        if kwargs.get("_calledFromCreator", False) == retrieve_passkey():
            validated_structure_data = structure_data
        else:
            try:
                structure_data = validated_structure_data = _validate_structure_data(
                    structure_data, process_matrix_rep=kwargs.get("process_matrix_rep", False), assume_skew=kwargs.get("assume_skew", False), assume_Lie_alg=kwargs.get("assume_Lie_alg", False),basis_order_for_supplied_str_eqns=kwargs.get("basis_order_for_supplied_str_eqns", False)
                )
            except dgcv_exception_note as e:
                raise SystemExit(e)

        validated_structure_data = tuple(map(tuple, validated_structure_data))

        obj = sp.Basic.__new__(cls, validated_structure_data)

        obj.structureData = validated_structure_data

        return obj

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
    ):
        if _calledFromCreator == retrieve_passkey():
            self.label = _label
            self.basis_labels = _basis_labels
            self._registered = True
        else:
            self.label = "Alg_" + create_key()
            self.basis_labels = [f"_e{i+1}" for i in range(len(self.structureData))]
            self._registered = False
        self._callLock = _callLock
        self._print_warning = _print_warning
        self._child_print_warning = _child_print_warning
        self._exclude_from_VMF = _exclude_from_VMF
        self.is_sparse = format_sparse
        self.dimension = len(self.structureData)
        self._built_from_matrices = process_matrix_rep
        self._dgcv_class_check=retrieve_passkey()
        self._dgcv_category='algebra'

        def validate_and_adjust_grading_vector(vector, dimension):
            if not isinstance(vector, (list, tuple)):
                raise ValueError(
                    "Grading vector must be a list or tuple."
                ) from None

            vector = list(vector)

            if len(vector) < dimension:
                warnings.warn(
                    f"Grading vector is shorter than the dimension ({len(vector)} < {dimension}). "
                    f"Padding with zeros to match the dimension.",
                    UserWarning,
                )
                vector += [0] * (dimension - len(vector))
            elif len(vector) > dimension:
                warnings.warn(
                    f"Grading vector is longer than the dimension ({len(vector)} > {dimension}). "
                    f"Truncating to match the dimension.",
                    UserWarning,
                )
                vector = vector[:dimension]

            for i, component in enumerate(vector):
                if not isinstance(component, (int, float, sp.Basic)):
                    raise ValueError(
                        f"Invalid component in grading vector at index {i}: {component}. "
                        f"Expected int, float, or sympy.Expr."
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

        if preferred_representation is not None and (not isinstance(preferred_representation,(list,tuple)) or len(preferred_representation)!=self.dimension):
            raise TypeError('unsupported format for `preferred_representation`.') from None
        if preferred_representation is not None and all(isinstance(elem,sp.Matrix) for elem in preferred_representation):
            self._preferred_rep_type = 'matrix'
            self._preferred_representation = preferred_representation
        elif preferred_representation is not None and all(isinstance(elem,tensorProduct) for elem in preferred_representation):
            self._preferred_rep_type = 'tensor'
            self._preferred_representation = preferred_representation
        elif preferred_representation is not None and all(isinstance(elem,(list,tuple)) for elem in preferred_representation):
            self._preferred_rep_type = 'matrix'
            self._preferred_representation = [sp.Matrix(elem) for elem in preferred_representation]
        elif preferred_representation is not None:
            raise TypeError('unsupported format for `preferred_representation`.') from None

        self.basis = tuple([
            algebra_element_class(
                self,
                [1 if i == j else 0 for j in range(self.dimension)],
                1,
                format_sparse=format_sparse,
            )
            for i in range(self.dimension)
        ])
        #immutables
        self._structureData = tuple(map(tuple, structure_data))
        self._basis_labels = tuple(_basis_labels) if _basis_labels else None
        self._grading = tuple(map(tuple, self.grading))
        # Caches for check methods
        self._skew_symmetric_cache = None
        self._jacobi_identity_cache = None
        self._lie_algebra_cache = None
        self._derived_algebra_cache = None
        self._center_cache = None
        self._lower_central_series_cache = None
        self._derived_series_cache = None
        self._grading_compatible = None
        self._grading_report = None

    def _class_builder(self,coeffs,valence,format_sparse=False):
        return algebra_element_class(self,coeffs,valence,format_sparse=format_sparse)

    @property
    def preferred_representation(self):
        if self._preferred_representation is None:
            warnings.warn('A preferred representation format for this algebra was not specified, to it has been converted to its adjoint representation.')
            self._preferred_rep_type = 'matrix'
            self._preferred_representation = adjointRepresentation(self)
        return self._preferred_representation

    def __eq__(self, other):
        if not isinstance(other, algebra_class):
            return NotImplemented
        return (
            self._structureData == other._structureData and
            self.label == other.label and
            self._basis_labels == other._basis_labels and
            self._grading == other._grading and
            self.basis == other.basis
        )

    def __hash__(self):
        return hash((self.label, self._basis_labels, self._grading))        #!!! add hashable structure data

    def __contains__(self, item):
        return item in self.basis

    def __iter__(self):
        return iter(self.basis) 

    def __getitem__(self, indices):
        if isinstance(indices,int):
            return self.basis[indices]
        elif isinstance(indices,list): 
            if len(indices)==1:
                return self.basis[indices[0]]
            elif isinstance(indices,list) and len(indices)==2:
                return self.structureData[indices[0]][indices[1]]
            elif isinstance(indices,list) and len(indices)==3:
                return self.structureData[indices[0]][indices[1]][indices[2]]
        else:
            raise TypeError(f'To access an algebra element or structure data component, provide one index for an element from the basis, two indices for a list of coefficients from the product  of two basis elements, or 3 indices for the corresponding entry in the structure array. Instead of an integer of list of integers, the following was given: {indices}') from None

    def __repr__(self):
        if not self._registered:
            if self._exclude_from_VMF == retrieve_passkey() or get_dgcv_settings_registry()['forgo_warnings'] is True:
                pass
            elif self._callLock == retrieve_passkey() and isinstance(self._print_warning,str):
                warnings.warn(
                    self._print_warning,
                    UserWarning,
                )
            else:
                warnings.warn(
                    "This algebra instance was initialized without an assigned label. "
                    "It is recommended to initialize algebra objects with dgcv creator functions like `createFiniteAlg` instead.",
                    UserWarning,
                )
        return (
            f"algebra_class(dim={self.dimension}, grading={self.grading}, "
            f"label={self.label}, basis_labels={self.basis_labels}, "
            f"struct_data={self.structureData})"
        )

    def _structure_data_summary(self):
        if self.dimension <= 3:
            return self.structureData
        return (
            "Structure data is large. Access the `structureData` attribute for details."
        )

    def __str__(self):
        if not self._registered:
            if self._exclude_from_VMF == retrieve_passkey() or get_dgcv_settings_registry()['forgo_warnings'] is True:
                pass
            elif self._callLock == retrieve_passkey() and isinstance(self._print_warning,str):
                warnings.warn(
                    self._print_warning,
                    UserWarning,
                )
            else:
                warnings.warn(
                    "This algebra instance was initialized without an assigned label. "
                    "It is recommended to initialize algebra objects with dgcv creator functions like `createFiniteAlg` instead.",
                    UserWarning,
                )

        formatted_label = self.label if self.label else "Unnamed Algebra"
        formatted_basis_labels = (
            ", ".join(list(self.basis_labels))
            if self.basis_labels
            else "No basis labels assigned"
        )
        return (
            f"Algebra: {formatted_label}\n"
            f"Dimension: {self.dimension}\n"
            f"Grading: {self.grading}\n"
            f"Basis: {formatted_basis_labels}"
        )

    def _display_DGCV_hook(self):
        if not self._registered:
            if self._exclude_from_VMF == retrieve_passkey() or get_dgcv_settings_registry()['forgo_warnings'] is True:
                pass
            elif self._callLock == retrieve_passkey() and isinstance(self._print_warning,str):
                warnings.warn(
                    self._print_warning,
                    UserWarning,
                )
            else:
                warnings.warn(
                    "This algebra instance was initialized without an assigned label. "
                    "It is recommended to initialize algebra objects with dgcv creator functions like `createFiniteAlg` instead.",
                    UserWarning,
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

    def _repr_latex_(self,verbose=False,abbrev=False):
        if not self._registered:
            if self._exclude_from_VMF == retrieve_passkey() or get_dgcv_settings_registry()['forgo_warnings'] is True:
                pass
            elif self._callLock == retrieve_passkey() and isinstance(self._print_warning,str):
                warnings.warn(
                    self._print_warning,
                    UserWarning,
                )
            else:
                warnings.warn(
                    "This algebra instance was initialized without an assigned label. "
                    "It is recommended to initialize algebra objects with dgcv creator functions like `createFiniteAlg` instead.",
                    UserWarning,
                )

        def format_algebra_label(label):
            r"""
            Formats an algebra label for LaTeX. Handles:
            1. Labels with an underscore, splitting into two parts:
            - The first part goes into \mathfrak{} if it is lowercase.
            - The second part becomes a LaTeX subscript.
            2. Labels without an underscore:
            - Checks if the label ends in a numeric tail for subscripting.
            - Otherwise wraps the label in \mathfrak{} if it is entirely lowercase.

            Parameters
            ----------
            label : str
                The algebra label to format.

            Returns
            -------
            str
                A LaTeX-formatted algebra label.
            """
            if not label:
                return "\\text{{Unnamed Algebra}}"

            if "_" in label:
                # Split the label at the first underscore
                main_part, subscript_part = label.split("_", 1)
                if main_part.islower():
                    return rf"\mathfrak{{{main_part}}}_{{{subscript_part}}}"
                return rf"{main_part}_{{{subscript_part}}}"

            if label[-1].isdigit():
                # Split into text and numeric parts for subscripting
                label_text = "".join(filter(str.isalpha, label))
                label_number = "".join(filter(str.isdigit, label))
                if label_text.islower():
                    return rf"\mathfrak{{{label_text}}}_{{{label_number}}}"
                return rf"{label_text}_{{{label_number}}}"

            if label.islower():
                # Wrap entirely lowercase labels in \mathfrak{}
                return rf"\mathfrak{{{label}}}"

            # Return the label as-is if no special conditions apply
            return label

        if abbrev is True:
            return f'${format_algebra_label(self.label)}$'

        def format_basis_label(label, idx):
            return rf"{label}" if label else f"e_{idx}"
        if verbose is True:
            formatted_label = format_algebra_label(self.label)
            formatted_basis_labels = (
                ", ".join([format_basis_label(bl,idx) for idx,bl in enumerate(self.basis_labels)])
                if self.basis_labels
                else "No basis labels assigned"
            )
            return (
                f"Algebra: ${formatted_label}$, Basis: ${formatted_basis_labels}$, "
                f"Dimension: ${self.dimension}$, Grading: ${sp.latex(self.grading)}$"
            )
        else:
            formatted_str = f'\\langle{', '.join(elem._repr_latex_() for elem in self.basis)}\\rangle'.replace('$','').replace('\\displaystyle','')
            if self.label:
                return f'$\\displaystyle {format_algebra_label(self.label)}={formatted_str}$'
            else:
                return f'$\\displaystyle {formatted_str}$'

    def _latex(self, printer=None):
        return self._repr_latex_().replace('$','').replace('\\displaystyle','')

    def _sympystr(self):
        if not self._registered:
            if self._exclude_from_VMF == retrieve_passkey() or get_dgcv_settings_registry()['forgo_warnings'] is True:
                pass
            elif self._callLock == retrieve_passkey() and isinstance(self._print_warning,str):
                warnings.warn(
                    self._print_warning,
                    UserWarning,
                )
            else:
                warnings.warn(
                    "This algebra instance was initialized without an assigned label. "
                    "It is recommended to initialize algebra objects with dgcv creator functions like `createFiniteAlg` instead.",
                    UserWarning,
                )

        if self.label:
            return f"algebra_class({self.label}, dim={self.dimension})"
        else:
            return f"algebra_class(dim={self.dimension})"

    def _structure_data_summary_latex(self):
        try:
            # Check if structureData contains only symbolic or numeric elements
            if self._is_symbolic_matrix(self.structureData):
                return sp.latex(
                    sp.Matrix(self.structureData)
                )  # Convert to matrix if valid
            else:
                return str(
                    self.structureData
                )  # Fallback to basic string representation
        except Exception:
            return str(self.structureData)  # Fallback in case of an error

    def _is_symbolic_matrix(self, data):
        """
        Checks if the matrix contains only symbolic or numeric entries.
        """
        return all(all(isinstance(elem, sp.Basic) for elem in row) for row in data)

    def is_skew_symmetric(self, verbose=False):
        """
        Checks if the algebra is skew-symmetric.
        Includes a warning for unregistered instances only if verbose=True.
        """
        if not self._registered and verbose:
            if self._callLock == retrieve_passkey() and isinstance(self._print_warning,str):
                print(self._print_warning)
            else:
                print(
                    "Warning: This algebra instance is unregistered. Initialize algebra objects with createFiniteAlg instead to register them."
                )

        if self._skew_symmetric_cache is None:
            result, failure = self._check_skew_symmetric()
            self._skew_symmetric_cache = (result, failure)
        else:
            result, failure = self._skew_symmetric_cache

        if verbose:
            if result:
                if self.label is None:
                    print("The algebra is skew-symmetric.")
                else:
                    print(f"{self.label} is skew-symmetric.")
            else:
                i, j, k = failure
                print(
                    f"Skew symmetry fails for basis elements {i}, {j}, at vector index {k}."
                )

        return result

    def _check_skew_symmetric(self):
        for i in range(self.dimension):
            for j in range(self.dimension):
                for k in range(len(self.structureData[i][j])):
                    vector_sum_element = sp.simplify(
                        self.structureData[i][j][k] + self.structureData[j][i][k]
                    )
                    if vector_sum_element != 0:
                        return False, (i, j, k)
        return True, None

    def satisfies_jacobi_identity(self, verbose=False):
        """
        Checks if the algebra satisfies the Jacobi identity.
        Includes a warning for unregistered instances only if verbose=True.
        """
        if not self._registered and verbose:
            if self._callLock == retrieve_passkey() and isinstance(self._print_warning,str):
                print(self._print_warning)
            else:
                print(
                    "Warning: This algebra instance is unregistered. Initialize algebra objects with createFiniteAlg instead to register them."
                )

        if self._jacobi_identity_cache is None:
            result, fail_list = self._check_jacobi_identity()
            self._jacobi_identity_cache = (result, fail_list)
        else:
            result, fail_list = self._jacobi_identity_cache

        if verbose:
            if result:
                if self.label is None:
                    print("The algebra satisfies the Jacobi identity.")
                else:
                    print(f"{self.label} satisfies the Jacobi identity.")
            else:
                print(f"Jacobi identity fails for the following triples: {fail_list}")

        return result

    def _check_jacobi_identity(self):
        fail_list = []
        for i in range(self.dimension):
            for j in range(self.dimension):
                for k in range(self.dimension):
                    if not (
                        self.basis[i] * self.basis[j] * self.basis[k]
                        + self.basis[j] * self.basis[k] * self.basis[i]
                        + self.basis[k] * self.basis[i] * self.basis[j]
                    ).is_zero():
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
        import warnings

        warnings.warn(
            f"{method_name} assumes the algebra is associative. "
            "If it is not then unexpected results may occur.",
            UserWarning,
        )

    def is_lie_algebra(self, verbose=False, return_bool=True):
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
            if self._callLock == retrieve_passkey() and isinstance(self._print_warning,str):
                print(self._print_warning)
            else:
                print(
                    "Warning: This algebra instance is unregistered. Initialize algebra objects with createFiniteAlg instead to register them."
                )

        # Check the cache
        if self._lie_algebra_cache is not None:
            if verbose:
                print(
                    f"Cached result: {'Lie algebra' if self._lie_algebra_cache else 'Not a Lie algebra'}."
                )
            return self._lie_algebra_cache

        # Perform the checks
        if not self.is_skew_symmetric(verbose=verbose):
            self._lie_algebra_cache = False
            if return_bool is True:
                return False
        if not self.satisfies_jacobi_identity(verbose=verbose):
            self._lie_algebra_cache = False
            if return_bool is True:
                return False

        # If both checks pass, cache the result and return True
        self._lie_algebra_cache = True

        if verbose:
            if self.label is None:
                print("The algebra is a Lie algebra.")
            else:
                print(f"{self.label} is a Lie algebra.")

        if return_bool is True:
            return True

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
        if not self.is_lie_algebra():
            raise ValueError(f"{method_name} can only be applied to Lie algebras.") from None

    def is_semisimple(self, verbose=False, return_bool=True):
        """
        Checks if the algebra is semisimple.
        Includes a warning for unregistered instances only if verbose=True.
        Nothing is returned if return_bool=False is set.
        """
        if not self._registered and verbose:
            if self._callLock == retrieve_passkey() and isinstance(self._print_warning,str):
                print(self._print_warning)
            else:
                print(
                    "Warning: This algebra instance is unregistered. Initialize algebra objects with createFiniteAlg instead to register them."
                )

        # Check if the algebra is a Lie algebra first
        if not self.is_lie_algebra(verbose=verbose):
            if return_bool is True:
                return False
            else:
                return

        # Compute the determinant of the Killing form
        if verbose is True:
            print('Progress update: computing determinant of the Killing form...')
        det = sp.simplify(killingForm(self).det())

        if verbose:
            if det != 0:
                if self.label is None:
                    print("The algebra is semisimple.")
                else:
                    print(f"{self.label} is semisimple.")
            else:
                if self.label is None:
                    print("The algebra is not semisimple.")
                else:
                    print(f"{self.label} is not semisimple.")
        if return_bool is True:
            return det != 0

    def is_subspace_subalgebra(self, elements, return_structure_data=False, check_linear_independence=False):
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

        # Perform linear independence check
        filtered_elem = self.filter_independent_elements(elements)
        span_matrix = sp.Matrix([list(el.coeffs) for el in filtered_elem]).transpose()

        linearly_independent = len(elements)==len(filtered_elem)

        # Check closure under product and build structure data
        dim = len(filtered_elem)
        structure_data = [
            [[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)
        ]
        closed_under_product = True

        for i, el1 in enumerate(filtered_elem):
            if closed_under_product is False:
                break
            for j, el2 in enumerate(filtered_elem):
                product = el1 * el2
                prodVec = sp.Matrix(product.coeffs)
                solution = span_matrix.solve_least_squares(sp.Matrix(product.coeffs))
                if any(entry!=0 for entry in span_matrix*solution-prodVec):
                    closed_under_product = False
                    structure_data = None
                    break
                for k, coeff in enumerate(solution):
                    coeff_simplified = sp.nsimplify(coeff)
                    structure_data[i][j][k] = coeff_simplified

        if return_structure_data:
            return {
                "linearly_independent": linearly_independent,
                "closed_under_product": closed_under_product,
                "structure_data": structure_data,
            }
        if check_linear_independence:
            return linearly_independent and closed_under_product
        else:
            return closed_under_product

    def check_element_weight(self, element, test_weights = None):
        """
        Determines the weight vector of an algebra_element_class with respect to the grading vectors. Weight can be instead computed against another grading vector passed a list of weights as the keyword `test_weights`.

        Parameters
        ----------
        element : algebra_element_class
            The algebra_element_class to analyze.
        test_weights : list of int or sympy.Expr, optional (default: None)

        Returns
        -------
        list
            A list of weights corresponding to the grading vectors of this algebra (or test_weights if provided).
            Each entry is either an integer, sympy.Expr (weight), the string 'AllW' (i.e., All Weights) if the element is the zero element,
            or 'NoW' (i.e., No Weights) if the element is not homogeneous.

        Notes
        -----
        - 'AllW' (meaning, All Weights) is returned for zero elements, which are compatible with all weights.
        - 'NoW' (meaning, No Weights) is returned for non-homogeneous elements that do not satisfy the grading constraints.
        """
        if not isinstance(element, algebra_element_class) or element.algebra!=self:
            raise TypeError("Input in `algebra_class.check_element_weight` must be an `algebra_element_class` instance belonging to the `algebra` instance whose `check_element_weight` is being called.") from None
        if not test_weights:
            if not hasattr(self, "grading") or self._gradingNumber == 0:
                raise ValueError("This algebra instance has no assigned grading vectors.") from None
        if all(coeff == 0 for coeff in element.coeffs):
            return ["AllW"] * self._gradingNumber
        if test_weights:
            if not isinstance(test_weights,(list,tuple)):
                raise TypeError('`check_element_weight` expects `test_weights` to be None or a list/tuple of lists/tuples of weight values (int,float, or sp.Expr).') from None
            for weight in test_weights:
                if not isinstance(weight,(list,tuple)):
                    raise TypeError('`check_element_weight` expects `test_weights` to be None or a list/tuple of lists/tuples of weight values (int,float, or sp.Expr).') from None
                if self.dimension != len(weight) or not all([isinstance(j,(int,float,sp.Expr)) for j in weight]):
                    raise TypeError('`check_element_weight` expects `test_weights` to be None or a list/tuple of lists/tuples of weight values (int,float, or sp.Expr).') from None
            GVs = test_weights
        else:
            GVs = self.grading
        weights = []
        for grading_vector in GVs:
            non_zero_indices = [i for i, coeff in enumerate(element.coeffs) if coeff != 0]
            basis_weights = [grading_vector[i] for i in non_zero_indices]
            if len(set(basis_weights)) == 1:
                weights.append(basis_weights[0])
            else:
                weights.append("NoW")
        return weights

    def check_grading_compatibility(self, verbose=False):
        """
        Checks if the algebra's structure constants are compatible with the assigned grading.

        Parameters
        ----------
        verbose : bool, optional (default=False)
            If True, prints detailed information about incompatibilities.

        Returns
        -------
        bool
            True if the algebra is compatible with all assigned grading vectors, False otherwise.

        Notes
        -----
        - Zero products (weights labeled as 'AllW') are treated as compatible with all grading vectors.
        - Non-homogeneous products (weights labeled as 'NoW') are treated as incompatible.
        """
        if not self._gradingNumber:
            raise ValueError(
                "No grading vectors are assigned to this algebra instance."
            ) from None
        if isinstance(self._grading_compatible,bool) and self._grading_report:
            compatible = self._grading_compatible
            failure_details = self._grading_report
        else:
            compatible = True
            failure_details = []

            for i, el1 in enumerate(self.basis):
                for j, el2 in enumerate(self.basis):
                    # Compute the product of basis elements
                    product = el1 * el2
                    product_weights = self.check_element_weight(product)

                    for g, grading_vector in enumerate(self.grading):
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
            self._grading_compatible = compatible
            self._grading_report = failure_details

        if verbose and not compatible:
            print("Grading Compatibility Check Failed:")
            for failure in failure_details:
                print(
                    f"- Grading Vector {failure['grading_vector_index']}: "
                    f"Basis elements {failure['basis_elements'][0]} and {failure['basis_elements'][1]} "
                    f"(weights: {failure['weights'][0]}, {failure['weights'][1]}) "
                    f"produced weight {failure['actual_weight']}, expected {failure['expected_weight']}."
                )

        return compatible

    def compute_center(self, for_associative_alg=False):
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

        if not for_associative_alg and not self.is_lie_algebra():
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
                'Using sympy.solve returned no solutions, indicating that this computation of the center failed, as solutions do exist.'
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

        return return_list

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
        self._require_lie_algebra("compute_derived_algebra")

        # Compute commutators only for j < k
        commutators = []
        for j, el1 in enumerate(self.basis):
            for k, el2 in enumerate(self.basis):
                if j < k:  # Only compute for j < k
                    commutators.append(el1 * el2)

        # Filter for linearly independent commutators
        subalgebra_data = self.is_subspace_subalgebra(
            commutators, return_structure_data=True
        )

        if not subalgebra_data["linearly_independent"]:
            raise ValueError(
                "Failed to compute the derived algebra: commutators are not linearly independent."
            ) from None
        if not subalgebra_data["closed_under_product"]:
            raise ValueError(
                "Failed to compute the derived algebra: commutators are not closed under the product."
            ) from None

        # Extract independent generators and structure data
        independent_generators = subalgebra_data.get(
            "independent_elements", commutators
        )
        structure_data = subalgebra_data["structure_data"]

        # Create the derived algebra
        return algebra_class(
            structure_data=structure_data,
            grading=self.grading,
            format_sparse=self.is_sparse,
            _label="Derived_Algebra",
            _basis_labels=[f"c_{i}" for i in range(len(independent_generators))],
            _calledFromCreator=retrieve_passkey(),
        )

    def filter_independent_elements(self, elements):
        """
        Filters a set of elements to retain only linearly independent and unique ones.

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
        if not isinstance(elements,(list,tuple)):
            warning_message += "\n The given value for `elements` is not a list or tuple"
        else:
            nonAE = []
            wrongAlgebra = []
            typeCheck={'algebra_element','subalgebra_element'}
            for elem in elements:
                if get_dgcv_category(elem) not in typeCheck:
                    nonAE.append(elem)
                elif elem.algebra!=self or (get_dgcv_category(elem)=='subalgebra_element' and elem.algebra.ambiant!=self):
                    wrongAlgebra.append(elem)
            if len(nonAE)>0 or len(wrongAlgebra)>0:
                if len(nonAE)>0:
                    warning_message += f"\n • These list elements are not `algebra_element_class` type: {nonAE}"
                if len(wrongAlgebra)>0:
                    warning_message += f"\n • These list elements are `algebra_element_class` type, but belong to a different algebra: {wrongAlgebra}"
        if warning_message:
            raise ValueError("The `algebra` method `filter_independent_elements` can only be applied to lists of elements belonging to the parent algebra the method is called from. Given data has the following problems:"+warning_message) from None

        # Remove duplicate elements based on their coefficients
        unique_elements = []
        seen_coeffs = set()
        for el in elements:
            if get_dgcv_category(el)=='subalgebra_element':
                el = el.ambiant_rep
            coeff_tuple = tuple(el.coeffs)  # Convert coeffs to a tuple for hashability
            if coeff_tuple not in seen_coeffs:
                seen_coeffs.add(coeff_tuple)
                unique_elements.append(el)

        # Create a matrix where each column is the coefficients of an element
        coeff_matrix = sp.Matrix([list(el.coeffs) for el in unique_elements]).transpose()

        # Get the column space (linearly independent vectors)
        independent_vectors = coeff_matrix.columnspace()

        # Match independent vectors with original columns
        independent_indices = []
        for vec in independent_vectors:
            for i in range(coeff_matrix.cols):
                if list(coeff_matrix[:, i]) == list(vec):
                    independent_indices.append(i)
                    break

        # Retrieve the corresponding elements
        independent_elements = [unique_elements[i] for i in independent_indices]

        return independent_elements

    def lower_central_series(self, max_depth=None):
        """
        Computes the lower central series of the algebra.

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
        if max_depth is None:
            max_depth = self.dimension

        series = []
        current_basis = self.basis
        previous_length = len(current_basis)

        for _ in range(max_depth):
            series.append(current_basis)  # Append the current basis level

            # Compute the commutators for the next level
            lower_central = []
            for el1 in current_basis:
                for el2 in self.basis:  # Bracket with the original algebra
                    commutator = el1 * el2
                    lower_central.append(commutator)

            # Filter for linear independence
            independent_generators = self.filter_independent_elements(lower_central)

            # Handle termination conditions
            if len(independent_generators) == 0:
                series.append([0 * self.basis[0]])  # Add the zero level
                break
            if len(independent_generators) == previous_length:
                break  # Series has stabilized

            # Update for the next iteration
            current_basis = independent_generators
            previous_length = len(independent_generators)

        return series

    def derived_series(self, max_depth=None):
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
        if max_depth is None:
            max_depth = self.dimension

        series = []
        current_basis = self.basis
        previous_length = len(current_basis)

        for _ in range(max_depth):
            series.append(current_basis)  # Append the current basis level

            # Compute the commutators for the next level
            derived = []
            for el1 in current_basis:
                for el2 in current_basis:  # Bracket with itself
                    commutator = el1 * el2
                    derived.append(commutator)

            # Filter for linear independence
            independent_generators = self.filter_independent_elements(derived)

            # Handle termination conditions
            if len(independent_generators) == 0:
                series.append([0 * self.basis[0]])  # Add the zero level
                break
            if len(independent_generators) == previous_length:
                break  # Series has stabilized

            # Update for the next iteration
            current_basis = independent_generators
            previous_length = len(independent_generators)

        return series

    def is_nilpotent(self, max_depth=10):
        """
        Checks if the algebra is nilpotent.

        Parameters
        ----------
        max_depth : int, optional
            Maximum depth to check for the lower central series.

        Returns
        -------
        bool
            True if the algebra is nilpotent, False otherwise.
        """
        series = self.lower_central_series(max_depth=max_depth)
        return (
            series[-1][0] == 0 * self.basis[0]
        )  # Nilpotent if the series terminates at {0}

    def is_solvable(self, max_depth=10):
        """
        Checks if the algebra is solvable.

        Parameters
        ----------
        max_depth : int, optional
            Maximum depth to check for the derived series.

        Returns
        -------
        bool
            True if the algebra is solvable, False otherwise.
        """
        series = self.derived_series(max_depth=max_depth)
        return (
            series[-1][0] == 0 * self.basis[0]
        )  # Solvable if the series terminates at {0}

    def get_structure_matrix(self, table_format=True, style=None):
        """
        Computes the structure matrix for the algebra.

        Parameters
        ----------
        table_format : bool, optional
            If True (default), returns a pandas DataFrame for a nicely formatted table.
            If False, returns a raw list of lists.
        style : str, optional
            A string key to retrieve a custom pandas style from the style_guide.

        Returns
        -------
        list of lists or pandas.DataFrame
            The structure matrix as a list of lists or a pandas DataFrame
            depending on the value of `table_format`.

        Notes
        -----
        - The (j, k)-entry of the structure matrix is the result of `basis[j] * basis[k]`.
        - If `basis_labels` is None, defaults to "_e1", "_e2", ..., "_e{d}".
        """
        import pandas as pd

        dimension = self.dimension
        basis_labels = self.basis_labels or [f"_e{i+1}" for i in range(dimension)]
        structure_matrix = [
            [(self.basis[j] * self.basis[k]) for k in range(dimension)]
            for j in range(dimension)
        ]

        if table_format:
            # Create a pandas DataFrame for a nicely formatted table
            data = {
                basis_labels[j]: [str(structure_matrix[j][k]) for k in range(dimension)]
                for j in range(dimension)
            }
            df = pd.DataFrame(data, index=basis_labels)
            df.index.name = "[e_j, e_k]"

            # Retrieve the style from get_style()
            if style is not None:
                pandas_style = get_style(style)
            else:
                pandas_style = get_style("default")

            # Apply the style to the DataFrame
            styled_df = df.style.set_caption("Structure sp.Matrix").set_table_styles(
                pandas_style
            )
            return styled_df
        return structure_matrix

    def is_ideal(self, subspace_elements):
        """
        Checks if the given list of elgebra elements spans an ideal.

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
        # Checks that all subspace elements belong to this algebra
        for el in subspace_elements:
            if not isinstance(el, algebra_element_class) or el.algebra != self:
                raise ValueError("All elements in subspace_elements must belong to this algebra.") from None

        # Check the ideal condition
        for el in subspace_elements:
            for other in self.basis:
                # Compute the product and check if it is in the span of subspace_elements
                product = el * other
                if not self.is_in_span(product, subspace_elements):
                    return False
        return True

    def is_in_span(self, element, subspace_elements):
        """
        Checks if a given algebra_element_class is in the span of subspace_elements.

        Parameters
        ----------
        element : algebra_element_class
            The element to check.
        subspace_elements : list
            A list of algebra_element_class instances representing the subspace they span.

        Returns
        -------
        bool
            True if the element is in the span of subspace_elements, False otherwise.
        """
        # Build a matrix where columns are the coefficients of subspace_elements
        span_matrix = sp.Matrix([list(el.coeffs) for el in subspace_elements]).transpose()

        # Solve for the coefficients that express `element` as a linear combination
        product_vector = sp.Matrix(element.coeffs)
        solution = span_matrix.solve_least_squares(product_vector)

        # Check if the solution satisfies the equation
        return span_matrix * solution == product_vector

    def weighted_component(self, weights):
        if isinstance(weights, (list,tuple)):
            if all(isinstance(weight, (int,float,sp.Expr)) for weight in weights):
                weights = [[weight] for weight in weights]
            elif not all(isinstance(weight, (list, tuple)) for weight in weights):
                raise ValueError('The `weights` parameter in `algebra_class.weighted_component` must be a list/tuple of weights/multi-weights. If giving a single multi-weight, it should be a length-1 list/tuple of lists/tuples, as otherwise a bare mult-weight tuple will be interpreted as a list of singleton weights.') from None
        else:
            raise ValueError('The `weights` parameter in `algebra_class.weighted_component` must be a list/tuple of weights/multi-weights. If giving a single multi-weight, it should be a length-1 list/tuple of lists/tuples, as otherwise a bare mult-weight tuple will be interpreted as a list of singleton weights.') from None
        component = []
        weights = [list(weight) for weight in weights]
        for elem in self.basis:
            if list(elem.check_element_weight()) in weights:
                component.append(elem)
        return algebra_subspace_class(component,self,self.grading)

    def multiplication_table(self, elements=None, restrict_to_subspace=False, style=None, use_latex=None, _called_from_subalgebra = None):
        """
        Generates a multiplication table for the algebra and the given elements.

        Parameters:
        -----------
        elements : list[algebra_element_class]
            A list of algebra_element_class instances to include in the multiplication table.
        restrict_to_subspace : bool, optional
            If True, restricts the multiplication table to the given elements as basis.
        style : str, optional
            A key to retrieve a pandas style via `get_style()`. If None, defaults to the current theme from DGCV settings.
        use_latex : bool, optional
            If True, wraps table contents in `$…$`. If None, defaults from DGCV settings.

        Returns:
        --------
        pandas.DataFrame
            A DataFrame representing the multiplication table.

        style : str, optional
            A key to retrieve a pandas style via `get_style()`. If None, defaults to the current theme from DGCV settings.
        use_latex : bool, optional
            If True, wraps table contents in `$…$`. If None, defaults from DGCV settings.
        """
        if elements is None:
            elements = self.basis
        elif not all(isinstance(elem, algebra_element_class) and elem.algebra == self for elem in elements):
            raise ValueError("All elements must be instances of algebraElement.") from None
        if restrict_to_subspace is True:
            basis_elements = elements
        elif isinstance(_called_from_subalgebra,dict) and _called_from_subalgebra.get('internalLock',None)==retrieve_passkey():
            basis_elements=_called_from_subalgebra['basis']
        else:
            basis_elements = self.basis

        # Determine LaTeX formatting
        if use_latex is None:
            use_latex = get_dgcv_settings_registry()['use_latex']
        def _to_string(element, ul=False):
            if ul:
                latex_str = element._repr_latex_(verbose=False)
                if latex_str.startswith('$') and latex_str.endswith('$'):
                    latex_str = latex_str[1:-1]
                latex_str = latex_str.replace(r'\\displaystyle', '').replace(r'\displaystyle', '').strip()
                return f'${latex_str}$'
            else:
                return str(element)
        # Create the table headers and initialize an empty data list
        headers = [_to_string(elem,ul=use_latex) for elem in elements]
        index_headers = [_to_string(elem,ul=use_latex) for elem in basis_elements]
        data = []

        # Populate the multiplication table
        for left_element in basis_elements:
            row = [_to_string(left_element * right_element, ul=use_latex) for right_element in elements]
            data.append(row)

        # Create a DataFrame for the multiplication table
        df = pd.DataFrame(data, columns=headers, index=index_headers)

        # Determine style key
        style_key = style or get_dgcv_settings_registry()['theme']
        pandas_style = get_style(style_key)

        # Determine outer border style from theme (fallback to 1px solid #ccc)
        border_style = "1px solid #ccc"
        for sd in pandas_style:
            if sd.get("selector") == "table":
                for prop_name, prop_value in sd.get("props", []):
                    if prop_name == "border":
                        border_style = prop_value
                        break
                break

        # Define additional styles: outer border, header bottom, index-column right border
        additional_styles = [
            {"selector": "",          "props": [("border-collapse", "collapse"), ("border", border_style)]},
            {"selector": "thead th",  "props": [("border-bottom", border_style)]},
            {"selector": "tbody th",  "props": [("border-right", border_style)]},
        ]

        table_styles = pandas_style + additional_styles

        # Build styled DataFrame
        styled = (
            df.style
            .set_caption("Multiplication Table")
            .set_table_styles(table_styles)
        )

        return styled

    def subalgebra(self,basis,grading=None):
        from .algebras_secondary import subalgebra_class
        basis_set = set(basis)
        newBasis = []
        for elem in self.basis:
            if elem in basis_set:
                newBasis.append(elem)
            if len(newBasis) == len(basis):
                break
        if len(newBasis) == len(basis):
            basis = newBasis
            subIndices = []
            for count, elem in enumerate(self.basis):
                if elem in basis:
                    subIndices.append(count)
            def truncateBySubInd(li,check_compl=False):
                if check_compl is True:
                    new_li = []
                    for count, elem in enumerate(li):
                        if count in subIndices:
                            new_li.append(elem)
                        elif elem!=0:
                            raise TypeError('The basis provided to the `algebra_class.subalgebra` method does not span a subalgebra. Suggestion: use `algebra_class.subspace` instead.') from None
                    return new_li
                return [li[j] for j in subIndices]
            if isinstance(grading,(list,tuple)) and all(isinstance(elem,(list,tuple)) for elem in grading):
                gradings = grading
            else:
                if grading is not None:
                    warnings.warn('The `gradings` keyword given to `algebra_class.subalgebra` was in an unsupported format (i.e., not list of lists), so a valid alternate gradings vector was computed instead.')
                gradings = [truncateBySubInd(grading) for grading in self.grading]
            structureData = truncateBySubInd(self._structureData)
            structureData = [truncateBySubInd(plane) for plane in structureData]
            structureData = [[truncateBySubInd(li,check_compl=True) for li in plane] for plane in structureData]
            return subalgebra_class(basis,self,grading=gradings,_compressed_structure_data=structureData,_internal_lock=retrieve_passkey())
        testStruct = self.is_subspace_subalgebra(basis,return_structure_data=True)
        if testStruct['closed_under_product'] is not True:
            raise TypeError('The basis provided to the `algebra_class.subalgebra` method does not span a subalgebra. Suggestion: use `algebra_class.subspace` instead.') from None
        return subalgebra_class(basis,self,grading=gradings,_compressed_structure_data=testStruct['structure_data'],_internal_lock=retrieve_passkey())

    def subspace(self,basis,grading=None):
        return algebra_subspace_class(self,basis,parent_algebra=self,test_weights=grading)


class algebra_element_class(sp.Basic):
    def __new__(cls, alg, coeffs, valence, format_sparse=False):

        if not isinstance(alg, algebra_class):
            raise TypeError(
                "`algebra_element_class` expects the first argument to be an instance of the `algebra` class."
            ) from None
        if valence not in {0, 1}:
            raise TypeError("vector_space_element expects third argument to be 0 or 1.") from None

        coeffs = tuple(coeffs)

        obj = sp.Basic.__new__(cls, alg, coeffs, valence, format_sparse)
        return obj

    def __init__(self, alg, coeffs, valence, format_sparse=False):
        self.algebra = alg
        self.vectorSpace = alg
        self.valence = valence
        self.is_sparse = format_sparse
        self._dgcv_class_check=retrieve_passkey()
        self._dgcv_category='algebra_element'
        # Store coeffs as an immutable tuple of tuples
        if isinstance(coeffs, (list, tuple)):  
            self.coeffs = tuple(coeffs)
        else:
            raise TypeError("algebra_element_class expects coeffs to be a list or tuple.") from None

    def __eq__(self, other):
        if not isinstance(other, algebra_element_class):
            return NotImplemented
        return (
            self.algebra == other.algebra and
            self.coeffs == other.coeffs and
            self.valence == other.valence and
            self.is_sparse == other.is_sparse
        )

    def __hash__(self):
        return hash((self.algebra, self.coeffs, self.valence, self.is_sparse))
    def __str__(self):
        if self.algebra.basis_labels is None:
            # Fallback to __str__ when basis_labels is None
            return str(self)

        terms = []
        for coeff, basis_label in zip(self.coeffs, self.algebra.basis_labels):
            if coeff == 0:
                continue
            elif coeff == 1:
                if self.valence==1:
                    terms.append(f"{basis_label}")
                else:
                    terms.append(f"{basis_label}^\'\'")
            elif coeff == -1:
                if self.valence==1:
                    terms.append(f"-{basis_label}")
                else:
                    terms.append(f"-{basis_label}^\'\'")
            else:
                if isinstance(coeff, sp.Expr) and len(coeff.args) > 1:
                    if self.valence==1:
                        terms.append(f"({coeff}) * {basis_label}")
                    else:
                        terms.append(f"({coeff}) * {basis_label}^\'\'")
                else:
                    if self.valence==1:
                        terms.append(f"{coeff} * {basis_label}")
                    else:
                        terms.append(f"{coeff} * {basis_label}^\'\'")
        if not terms:
            return '0'
        return " + ".join(terms).replace("+ -", "- ")


    def _class_builder(self,coeffs,valence,format_sparse=False):
        return algebra_element_class(self.algebra,coeffs,valence,format_sparse=format_sparse)

    def _repr_latex_(self,verbose=False):
        if not self.algebra._registered:
            if self.algebra._exclude_from_VMF == retrieve_passkey() or get_dgcv_settings_registry()['forgo_warnings'] is True:
                pass
            elif self.algebra._callLock == retrieve_passkey() and isinstance(self.algebra._child_print_warning,str):
                warnings.warn(self.algebra._child_print_warning,UserWarning)
            else:
                warnings.warn(
                    "This algebra_element_class's parent vector space (algebra_class) was initialized without an assigned label. "
                    "It is recommended to initialize `algebra_class` objects with dgcv creator functions like `createAlgebra` instead.",
                    UserWarning,
                )

        terms = []
        for coeff, basis_label in zip(
            self.coeffs,
            self.algebra.basis_labels
            or [f"e_{{{i+1}}}" for i in range(self.algebra.dimension)],
        ):
            if "_" not in basis_label and basis_label and basis_label[-1].isdigit():
                basis_label = re.sub(r"^(.+?)(\d+)$", r"\1_\2", basis_label)
            basis_label = format_latex_subscripts(basis_label)

            if coeff == 0:
                continue
            elif coeff == 1:
                if self.valence==1:
                    terms.append(rf"{basis_label}")
                else:
                    terms.append(rf"{basis_label}^*")
            elif coeff == -1:
                if self.valence==1:
                    terms.append(rf"-{basis_label}")
                else:
                    terms.append(rf"-{basis_label}^*")
            else:
                if isinstance(coeff, sp.Expr) and len(coeff.args) > 1:
                    if self.valence==1:
                        terms.append(rf"({sp.latex(coeff)}) \cdot {basis_label}")
                    else:
                        terms.append(rf"({sp.latex(coeff)}) \cdot {basis_label}^*")
                else:
                    if self.valence==1:
                        terms.append(rf"{sp.latex(coeff)} \cdot {basis_label}")
                    else:
                        terms.append(rf"{sp.latex(coeff)} \cdot {basis_label}^*")

        if not terms:
            if verbose:
                return rf"$0 \cdot {self.algebra.basis_labels[0] if self.algebra.basis_labels else 'e_1'}$"
            else:
                return "$0$"

        result = " + ".join(terms).replace("+ -", "- ")

        return rf"$\displaystyle {result}$"

    def _latex(self,printer=None):
        return self._repr_latex_()

    def _sympystr(self):
        """
        SymPy string representation for algebra_element_class.
        Handles unregistered parent algebra by raising a warning.
        """
        if not self.algebra._registered:
            if self.algebra._exclude_from_VMF == retrieve_passkey() or get_dgcv_settings_registry()['forgo_warnings'] is True:
                pass
            elif self.algebra._callLock == retrieve_passkey() and isinstance(self.algebra._child_print_warning,str):
                warnings.warn(
                    self.algebra._child_print_warning,
                    UserWarning,
                )
            else:
                warnings.warn(
                    "This algebra_element_class's parent algebra (`algebra` class) was initialized without an assigned label. "
                    "It is recommended to initialize `algebra` class objects with dgcv creator functions like `createFiniteAlg` instead.",
                    UserWarning,
                )

        coeffs_str = ", ".join(map(str, self.coeffs))
        if self.algebra.label:
            return f"algebra_element_class({self.algebra.label}, coeffs=[{coeffs_str}])"
        else:
            return f"algebra_element_class(coeffs=[{coeffs_str}])"

    def _latex_verbose(self, printer=None):
        if not self.algebra._registered:
            if self.algebra._exclude_from_VMF == retrieve_passkey() or get_dgcv_settings_registry()['forgo_warnings'] is True:
                pass
            elif self.algebra._callLock == retrieve_passkey() and isinstance(self.algebra._child_print_warning,str):
                warnings.warn(
                    self.algebra._child_print_warning,
                    UserWarning,
                )
            else:
                warnings.warn(
                    "This algebra_element_class's parent vector space (an `algebra` class instance) was initialized without an assigned label. "
                    "It is recommended to initialize `algebra` class objects with dgcv creator functions like `createFiniteAlg` instead.",
                    UserWarning,
                )

        terms = []
        for coeff, basis_label in zip(
            self.coeffs,
            self.algebra.basis_labels
            or [f"e_{i+1}" for i in range(self.algebra.dimension)],
        ):
            if coeff == 0:
                continue
            elif coeff == 1:
                if self.valence == 1:
                    terms.append(rf"{basis_label}")
                else:
                    terms.append(rf"{basis_label}^*")
            elif coeff == -1:
                if self.valence == 1:
                    terms.append(rf"-{basis_label}")
                else:
                    terms.append(rf"-{basis_label}^*")
            else:
                if isinstance(coeff, sp.Expr) and len(coeff.args) > 1:
                    if self.valence == 1:
                        terms.append(rf"({sp.latex(coeff)}) \cdot {basis_label}")
                    else:
                        terms.append(rf"({sp.latex(coeff)}) \cdot {basis_label}^*")
                else:
                    if self.valence == 1:
                        terms.append(rf"{sp.latex(coeff)} \cdot {basis_label}")
                    else:
                        terms.append(rf"{sp.latex(coeff)} \cdot {basis_label}^*")

        if not terms:
            return rf"0 \cdot {self.algebra.basis_labels[0] if self.algebra.basis_labels else 'e_1'}"

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

    def __repr__(self):
        if self.algebra.basis_labels is None:
            # Fallback to __str__ when basis_labels is None
            return str(self)

        terms = []
        for coeff, basis_label in zip(self.coeffs, self.algebra.basis_labels):
            if coeff == 0:
                continue
            elif coeff == 1:
                if self.valence==1:
                    terms.append(f"{basis_label}")
                else:
                    terms.append(f"{basis_label}^\'\'")
            elif coeff == -1:
                if self.valence==1:
                    terms.append(f"-{basis_label}")
                else:
                    terms.append(f"-{basis_label}^\'\'")
            else:
                if isinstance(coeff, sp.Expr) and len(coeff.args) > 1:
                    if self.valence==1:
                        terms.append(f"({coeff}) * {basis_label}")
                    else:
                        terms.append(f"({coeff}) * {basis_label}^\'\'")
                else:
                    if self.valence==1:
                        terms.append(f"{coeff} * {basis_label}")
                    else:
                        terms.append(f"{coeff} * {basis_label}^\'\'")

        if not terms:
            if self.valence==1:
                return f"0*{self.algebra.basis_labels[0]}"
            else:
                return f"0*{self.algebra.basis_labels[0]}^\'\'"

        return " + ".join(terms).replace("+ -", "- ")

    def is_zero(self):
        for j in self.coeffs:
            if sp.simplify(j) != 0:
                return False
        else:
            return True

    def subs(self, subsData):
        newCoeffs = [sp.sympify(j).subs(subsData) for j in self.coeffs]
        return algebra_element_class(self.algebra, newCoeffs, self.valence, format_sparse=self.is_sparse)

    def dual(self):
        return algebra_element_class(self.algebra, self.coeffs, (self.valence+1)%2,format_sparse=self.is_sparse)

    def _convert_to_tp(self):
        return tensorProduct(self.algebra,{(j,self.valence):self.coeffs[j] for j in range(self.algebra.dimension)})

    def _recursion_contract_hom(self, other):
        return self._convert_to_tp()._recursion_contract_hom(other)

    def __add__(self, other):
        if hasattr(other,"is_zero") and other.is_zero():
            return self
        if get_dgcv_category(other)=='subalgebra_element':
            other = other.ambiant_rep
        if isinstance(other, algebra_element_class):
            if self.algebra == other.algebra and self.valence==other.valence:
                return algebra_element_class(
                    self.algebra,
                    [self.coeffs[j] + other.coeffs[j] for j in range(len(self.coeffs))],
                    self.valence,
                    format_sparse=self.is_sparse,
                )
            else:
                raise TypeError(
                    "algebra_element_class operands for + must belong to the same algebra."
                ) from None
        if isinstance(other,tensorProduct):
            if other.max_degree==1 and other.min_degree==1 and other.vector_space==self.algebra:
                pt = other.prolongation_type
                coeffs = [other.coeff_dict[(j,pt)] if (j,pt) in other.coeff_dict else 0 for j in range(other.vector_space.dimension)]
                LA_elem = other.vector_space._class_builder(coeffs,pt,format_sparse=False)
                return self+LA_elem
            else:
                return self._convert_to_tp()+other
        else:
            raise TypeError(
                "Unsupported operand type(s) for + with the algebra_element_class"
            ) from None

    def __sub__(self, other):
        if hasattr(other,"is_zero") and other.is_zero():
            return self
        if get_dgcv_category(other)=='subalgebra_element':
            other = other.ambiant_rep
        if isinstance(other, algebra_element_class):
            if self.algebra == other.algebra and self.valence==other.valence:
                return algebra_element_class(
                    self.algebra,
                    [self.coeffs[j] - other.coeffs[j] for j in range(len(self.coeffs))],
                    self.valence,
                    format_sparse=self.is_sparse,
                )
            else:
                raise TypeError(
                    "algebra_element_class operands for - must belong to the same algebra."
                ) from None
        if isinstance(other,tensorProduct):
            if other.max_degree==1 and other.min_degree==1:
                if other.vector_space==self.algebra:
                    pt = other.prolongation_type
                    coeffs = [other.coeff_dict[(j,pt)] if (j,pt) in other.coeff_dict else 0 for j in range(other.vector_space.dimension)]
                    LA_elem = other.vector_space._class_builder(coeffs,pt,format_sparse=False)
                    return self-LA_elem
        else:
            raise TypeError(
                f"Unsupported operand type(s) {type(other)} for - with the algebra_element_class"
            ) from None

    def __mul__(self, other):
        if get_dgcv_category(other)=='subalgebra_element':
            other = other.ambiant_rep
        if isinstance(other, algebra_element_class):
            if self.algebra == other.algebra and self.valence==other.valence:
                result_coeffs = [0] * self.algebra.dimension
                for i in range(self.algebra.dimension):
                    for j in range(self.algebra.dimension):
                        scalar_product = self.coeffs[i] * other.coeffs[j]
                        structure_vector_product = [scalar_product * element for element in self.algebra.structureData[i][j]]
                        result_coeffs = [
                            sp.sympify(result_coeffs[k] + structure_vector_product[k])
                            for k in range(len(result_coeffs))
                        ]
                return algebra_element_class(self.algebra, result_coeffs, self.valence, format_sparse=self.is_sparse)
            else:
                raise TypeError(
                    "Both operands for * must be algebra_element_class instances from the same algebra."
                ) from None
        elif isinstance(other, tensorProduct):
            return (self._convert_to_tp())*other
        elif isinstance(other, (int, float, sp.Expr)):
            # Scalar multiplication case
            new_coeffs = [coeff * other for coeff in self.coeffs]
            # Return a new algebra_element_class with the updated coefficients
            return algebra_element_class(
                self.algebra, new_coeffs, self.valence, format_sparse=self.is_sparse
            )
        else:
            raise TypeError(
                f"Multiplication is only supported for scalars and the AlegebraElement class, not {type(other)}"
            ) from None

    def __rmul__(self, other):
        if get_dgcv_category(other)=='subalgebra_element':
            other = other.ambiant_rep
        if isinstance(other, (int, float, sp.Expr)):
            return self * other
        elif isinstance(other, algebra_element_class):
            other * self
        elif isinstance(other, tensorProduct):
            return other*(self._convert_to_tp())
        else:
            raise TypeError(
                f"Right multiplication is only supported for scalars and the AlegebraElement class, not {type(other)}"
            ) from None

    def __matmul__(self, other):
        """Overload @ operator for tensor product."""
        if get_dgcv_category(other)=='subalgebra_element':
            other = other.ambiant_rep
        if not isinstance(other, algebra_element_class) or other.algebra!=self.algebra:
            raise TypeError('`@` only supports tensor products between algebra_element_class instances with the same `algebra` attribute') from None
        new_dict = {(j,k,self.valence,other.valence):self.coeffs[j]*other.coeffs[k] for j in range(self.algebra.dimension) for k in range(self.algebra.dimension)}
        return tensorProduct(self.algebra, new_dict)

    def __neg__(self):
        return -1*self

    def __xor__(self, other):
        if other == '':
            return self.dual()
        raise ValueError("Invalid operation. Use `^''` to denote the dual.") from None


    def check_element_weight(self):
        """
        Determines the weight vector of this algebra_element_class with respect to its algebra' grading vectors.

        Returns
        -------
        list
            A list of weights corresponding to the grading vectors of the parent algebra.
            Each entry is either an integer, sympy.Expr (weight), the string 'AllW' if the element is the zero element,
            or 'NoW' if the element is not homogeneous.

        Notes
        -----
        - This method calls the parentt algebra' check_element_weight method.
        - 'AllW' is returned for zero elements, which are compaible with all weights.
        - 'NoW' is returned for non-homogeneous elements that do not satisfy the grading constraints.
        """

        return self.algebra.check_element_weight(self)

class algebra_subspace_class(sp.Basic):
    def __new__(cls, basis, parent_algebra=None, test_weights=None,_grading=None,_internal_lock=None,**kwargs):
        if not isinstance(basis,(list,tuple)):
            raise TypeError('algebra_subspace_class expects first argument to a be a list or tuple of algebra_element_class instances') from None
        typeCheck = {'subalgebra_element', 'algebra_element'}
        if (not all(get_dgcv_category(j) in typeCheck for j in basis)):
            raise TypeError('algebra_subspace_class expects first argument to a be a list or tuple of algebra_element_class instances') from None
        if parent_algebra is None:
            if len(basis)>0:
                if get_dgcv_category(basis[0].algebra)!='algebra':
                    if all(j.algebra==basis[0].algebra for j in basis[1:]):
                        parent_alg = basis[0].algebra.ambiant
                        if test_weights and isinstance(test_weights,(list,tuple)) and len(test_weights[0])==len(basis[0].algebra):
                            weight_subspace = basis[0].algebra
                        else:
                            weight_subspace = parent_alg
                    else:
                        parent_alg = basis[0].algebra.ambiant
                        weight_subspace = parent_alg
                else:
                    parent_alg=basis[0].algebra
                    weight_subspace = parent_alg
            else: 
                weight_subspace = []
        elif get_dgcv_category(parent_algebra) in {'subalgebra', 'algebra_subspace'}:
            parent_alg = parent_algebra.ambiant
            weight_subspace = parent_algebra if test_weights else parent_alg
        elif get_dgcv_category(parent_algebra)=='algebra':
            parent_alg = parent_algebra
            weight_subspace = parent_algebra
        else:
            raise TypeError('algebra_subspace_class expects second argument to an algebra instance or algebra subspace or subalgebra.') from None

        if not all([j.algebra == weight_subspace for j in basis]):
            if get_dgcv_category(weight_subspace)=='algebra':
                new_basis = []
                for elem in basis:
                    if get_dgcv_category(elem)=='algebra_element':
                        new_basis.append(elem)
                    else:
                        new_basis.append(elem.ambiant_rep)
                basis=new_basis
            if not all([j.algebra == weight_subspace for j in basis]):
                raise TypeError('algebra_subspace_class expects all (sub)algebra_element_class instances given in the first argument to have the same `(sub)algebra_element_class.algebra` value as the second argument.') from None
        if test_weights and not(_internal_lock==retrieve_passkey() and _grading is not None):
            if not isinstance(test_weights,(list,tuple)):
                raise TypeError('`algebra_subspace_class` initializer expects `test_weight` to be None or a list/tuple of lists of weight values (int,float, or sp.Expr).') from None
            for tw in test_weights:
                if not isinstance(tw,(list,tuple)):
                    raise TypeError('`algebra_subspace_class` initializer expects `test_weight` to be None or a list/tuple of lists of weight values (int,float, or sp.Expr).') from None
                if weight_subspace.dimension != len(tw) or not all([isinstance(j,(int,float,sp.Expr)) for j in tw]):
                    raise TypeError('`algebra_subspace_class` initializer expects `test_weight` to be None or a length {alg.dimension} list/tuple of weight values (int,float, or sp.Expr).') from None
        filtered_basis = parent_alg.filter_independent_elements(basis)
        if len(filtered_basis)<len(basis):
            basis = filtered_basis
            warnings.warn('The given list for `basis` was not linearly independent, so the algebra_subspace_class initializer computed a basis for its span to use instead.')

        # Create the new instance
        obj = sp.Basic.__new__(cls, basis, parent_algebra, test_weights)
        obj.filtered_basis = filtered_basis
        obj.basis = tuple(filtered_basis)
        obj.dimension = len(filtered_basis)
        obj.ambiant = parent_alg
        grading_per_elem = []
        if test_weights is None and _internal_lock==retrieve_passkey() and _grading is not None:
            obj.grading=_grading
        else:
            for elem in filtered_basis:
                weight = weight_subspace.check_element_weight(elem,test_weights=test_weights)
                grading_per_elem.append(weight)
            obj.grading = list(zip(*grading_per_elem))
            obj.grading = [elem for elem in obj.grading if 'NoW' not in elem]
        return obj

    def __init__(self, basis, parent_algebra=None, test_weights=None,_grading=None,_internal_lock=None):
        self.original_basis = basis
        self._dgcv_class_check=retrieve_passkey()
        self._dgcv_category='algebra_subspace'

        # immutables
        self._grading = tuple(self.grading)
        self._gradingNumber = len(self._grading)

        # attribute caches
        self._is_subalgebra = None

    def __eq__(self, other):
        if not isinstance(other, algebra_subspace_class):
            return NotImplemented
        return (
            self.ambiant == other.ambiant and
            self.basis == other.basis and
            self._grading == other._grading
        )
    def __hash__(self):
        return hash((self.ambiant, self.basis, self._grading))
    def __contains__(self, item):
        return item in self.basis

    def check_element_weight(self, element, test_weights = None):
        """
        Determines the weight vector of an algebra_element_class with respect to the grading vectors assigned to an algebra_subspace_class. Weight can be instead computed against another grading vector passed a list of weights as the keyword `test_weights`.

        Parameters
        ----------
        element : (sub)algebra_element_class
            The (sub)algebra_element_class to analyze.
        test_weights : list of int or sympy.Expr, optional (default: None)

        Returns
        -------
        list
            A list of weights corresponding to the grading vectors of this algebra subspace (or test_weights if provided).
            Each entry is either an integer, sympy.Expr (weight), the string 'AllW' (i.e., All Weights) if the element is the zero element,
            or 'NoW' (i.e., No Weights) if the element is not homogeneous.

        Notes
        -----
        - 'AllW' (meaning, All Weights) is returned for zero elements, which are compatible with all weights.
        - 'NoW' (meaning, No Weights) is returned for non-homogeneous elements that do not satisfy the grading constraints.
        """
        if not isinstance(element, algebra_element_class) or element.algebra!=self:
            if get_dgcv_category(element)=='subalgebra_element' and element.algebra.ambiant==self.ambiant:
                pass
            else:
                raise TypeError("Input in `algebra_subspace_class.check_element_weight` must be an `(sub)algebra_element_class` instance belonging to the `(sub)algebra_class` instance whose `check_element_weight` is being called.") from None
        if not test_weights:
            if self._gradingNumber == 0:
                raise ValueError("This algebra subspace instance has no assigned grading vectors to test weighting w.r.t..") from None
        if all(coeff == 0 for coeff in element.coeffs):
            return ["AllW"] * self._gradingNumber
        if test_weights:
            if not isinstance(test_weights,(list,tuple)):
                raise TypeError('`check_element_weight` expects `test_weights` to be None or a list/tuple of lists/tuples of weight values (int,float, or sp.Expr).') from None
            for weight in test_weights:
                if not isinstance(weight,(list,tuple)):
                    raise TypeError('`check_element_weight` expects `test_weights` to be None or a list/tuple of lists/tuples of weight values (int,float, or sp.Expr).') from None
                if self.dimension != len(weight) or not all([isinstance(j,(int,float,sp.Expr)) for j in weight]):
                    raise TypeError('`check_element_weight` expects `test_weights` to be None or a list/tuple of lists/tuples of weight values (int,float, or sp.Expr).') from None
            GVs = test_weights
        else:
            GVs = self.grading
        weights = []
        for grading_vector in GVs:
            non_zero_indices = [i for i, coeff in enumerate(element.coeffs) if coeff != 0]
            basis_weights = [grading_vector[i] for i in non_zero_indices]
            if len(set(basis_weights)) == 1:
                weights.append(basis_weights[0])
            else:
                weights.append("NoW")
        return weights

    def contains(self, items, return_basis_coeffs = False):
        if not isinstance(items,(list,tuple)):
            items = [items]
        for item in items:
            if get_dgcv_category(item)=='subalgebra_element':
                item=item.ambiant_rep
            if not isinstance(item,algebra_element_class) or item.algebra!=self.ambiant:
                return False
            if item not in self.basis:
                tempVarLabel = "T" + retrieve_public_key()
                variableProcedure(tempVarLabel, len(self.basis), _tempVar=retrieve_passkey())
                genElement = sum([_cached_caller_globals[tempVarLabel][j+1] * elem for j,elem in enumerate(self.basis[1:])],_cached_caller_globals[tempVarLabel][0]*(self.basis[0]))
                sol = solve_dgcv(item-genElement,_cached_caller_globals[tempVarLabel])
                if len(sol)==0:
                    clearVar(*listVar(temporary_only=True))
                    return False
            else:
                if return_basis_coeffs is True:
                    idx = (self.basis).index(item)
                    return [1 if _==idx else 0 for _ in range(len(self.basis))]
        if return_basis_coeffs is True:
            vec=[var.subs(sol[0]) for var in _cached_caller_globals[tempVarLabel]]
            clearVar(*listVar(temporary_only=True))
            return vec
        clearVar(*listVar(temporary_only=True))
        return True

    def __iter__(self):
        return iter(self.basis) 

    def __getitem__(self, index):
        return self.basis[index]

    def is_subalgebra(self,return_structure_data=False):
        if self._is_subalgebra is None:
            self._is_subalgebra = self.ambiant.is_subspace_subalgebra(self.filtered_basis,return_structure_data=return_structure_data)
        return self._is_subalgebra

    def __str__(self):
        return f'span{{{', '.join(elem.__str__() for elem in self.basis)}}}'

    def __repr__(self):
        return self.__str__()

    def _repr_latex_(self):
        formatted_str = f'\\langle{', '.join(elem._repr_latex_() for elem in self.basis)}\\rangle'.replace('$','').replace('\\displaystyle','')
        return f'$\\displaystyle {formatted_str}$'

    def _latex(self, printer=None):
        return self._repr_latex_().replace('$','').replace('\\displaystyle','')


############## algebra tools

def killingForm(alg, list_processing=False):
    if get_dgcv_category(alg) in {"algebra","subalgebra"}:
        # Convert the structure data to a mutable array
        if not alg.is_lie_algebra():
            raise Exception(
                "killingForm expects argument to be a Lie algebra instance of the algebra"
            ) from None
        if list_processing:
            aRepLoc = alg.structureData
            return [
                [
                    trace_matrix(multiply_matrices(aRepLoc[j], aRepLoc[k]))
                    for k in range(alg.dimension)
                ]
                for j in range(alg.dimension)
            ]
        else:
            aRepLoc = adjointRepresentation(alg)
            return sp.Matrix(
                alg.dimension,
                alg.dimension,
                lambda j, k: (aRepLoc[j] * aRepLoc[k]).trace(),
            )
    else:
        raise Exception("killingForm expected to receive an algebra instance.") from None


def adjointRepresentation(alg, list_format=False):
    if get_dgcv_category(alg) in {"algebra","subalgebra"}:
        # Convert the structure data to a mutable array
        if not alg.is_lie_algebra():
            warnings.warn(
                "Caution: The algebra passed to adjointRepresentation is not a Lie algebra."
            )
        if list_format:
            return alg.structureData
        return [sp.Matrix(j) for j in alg.structureData]
    else:
        raise Exception(
            "adjointRepresentation expected to receive an algebra instance."
        ) from None


############## helpers

def convert_to_greek(var_name):
    for name, greek in greek_letters.items():
        if var_name.lower().startswith(name):
            return var_name.replace(name, greek, 1)
    return var_name
def format_latex_subscripts(var_name, nest_braces=False):
    """for use_latex branches"""
    if var_name[-1]=='_':
        var_name=var_name[:-1]
    parts = var_name.split("_")
    if len(parts) == 1:
        return convert_to_greek(var_name)
    base = convert_to_greek(parts[0])
    subscript = ", ".join(parts[1:])
    if nest_braces is True:
        return f"{{{base}_{{{subscript}}}}}"
    else:
        return f"{base}_{{{subscript}}}"

############## linear algebra list processing


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
    # Get the dimensions of the matrices
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    # Check if matrices are compatible for multiplication
    if cols_A != rows_B:
        raise ValueError(
            "Incompatible matrix dimensions: A is {}x{}, B is {}x{}".format(
                rows_A, cols_A, rows_B, cols_B
            )
        ) from None

    # Initialize the result matrix with zeros
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    # Perform matrix multiplication
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):  # or range(rows_B), since cols_A == rows_B
                result[i][j] += A[i][k] * B[k][j]

    return result

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
    # Get the dimensions of the matrix
    rows_A, cols_A = len(A), len(A[0])

    # Check if the matrix is square
    if rows_A != cols_A:
        raise ValueError(
            "Trace can only be computed for square matrices. sp.Matrix is {}x{}.".format(
                rows_A, cols_A
            )
        ) from None

    # Compute the trace (sum of the diagonal elements)
    trace_value = sum(A[i][i] for i in range(rows_A))

    return trace_value
