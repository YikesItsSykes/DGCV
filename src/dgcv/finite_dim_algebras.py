############## dependencies
import random
import warnings

import sympy as sp

from ._config import _cached_caller_globals
from ._safeguards import (
    create_key,
    get_variable_registry,
    retrieve_passkey,
    retrieve_public_key,
    validate_label,
    validate_label_list,
)
from .dgcv_core import VFClass, addVF, allToReal, clearVar, listVar, variableProcedure
from .styles import get_style
from .tensors import tensorProduct
from .vector_fields_and_differential_forms import VF_bracket

############## Algebras


# finite dimensional algebra class
class FAClass(sp.Basic):
    def __new__(cls, structure_data, *args, **kwargs):
        # Validate structure data
        validated_structure_data = cls.validate_structure_data(
            structure_data, process_matrix_rep=kwargs.get("process_matrix_rep", False)
        )
        validated_structure_data = tuple(map(tuple, validated_structure_data))

        # Create the new instance
        obj = sp.Basic.__new__(cls, validated_structure_data)

        # Attach structureData directly for use in __init__
        obj.structureData = validated_structure_data

        return obj

    def __init__(
        self,
        structure_data,
        grading=None,
        format_sparse=False,
        process_matrix_rep=False,
        _label=None,
        _basis_labels=None,
        _calledFromCreator=None,
    ):
        # Handle structure_data in __init__
        self.structureData = structure_data

        # Detect if initialized from creator (using the passkey)
        if _calledFromCreator == retrieve_passkey():
            self.label = _label
            self.basis_labels = _basis_labels
            self._registered = True
        else:
            self.label = "Alg_" + create_key()  # Assign a random label
            self.basis_labels = None
            self._registered = False

        # Other attributes
        self.is_sparse = format_sparse
        self.dimension = len(self.structureData)
        self._built_from_matrices = process_matrix_rep

        def validate_and_adjust_grading_vector(vector, dimension):
            """
            Validates and adjusts a grading vector to match the algebra's dimension.

            Parameters
            ----------
            vector : list, tuple, or sympy.Tuple
                The grading vector to validate and adjust.
            dimension : int
                The dimension of the algebra.

            Returns
            -------
            sympy.Tuple
                The validated and adjusted grading vector.
            """
            # Check that vector is a list, tuple, or SymPy Tuple
            if not isinstance(vector, (list, tuple, sp.Tuple)):
                raise ValueError(
                    "Grading vector must be a list, tuple, or SymPy Tuple."
                )

            # Convert to list for uniform handling
            vector = list(vector)

            # Adjust length to match dimension
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

            # Validate components
            for i, component in enumerate(vector):
                if not isinstance(component, (int, float, sp.Basic)):
                    raise ValueError(
                        f"Invalid component in grading vector at index {i}: {component}. "
                        f"Expected int, float, or sympy.Expr."
                    )

            return sp.Tuple(*vector)

        # Process grading
        if grading is None:
            # Default to a single grading vector [0, 0, ..., 0]
            self.grading = [sp.Tuple(*([0] * self.dimension))]
        else:
            # Handle single or multiple grading vectors
            if isinstance(grading, (list, tuple)) and all(
                isinstance(g, (list, tuple, sp.Tuple)) for g in grading
            ):
                # Multiple grading vectors provided
                self.grading = [
                    validate_and_adjust_grading_vector(vector, self.dimension)
                    for vector in grading
                ]
            else:
                # Single grading vector provided
                self.grading = [
                    validate_and_adjust_grading_vector(grading, self.dimension)
                ]

        # Set the number of grading vectors
        self._gradingNumber = len(self.grading)

        # Initialize basis
        self.basis = tuple([
            AlgebraElement(
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

    @staticmethod
    def validate_structure_data(data, process_matrix_rep=False):
        """
        Validates the structure data and converts it to a list of lists of lists.

        Parameters
        ----------
        data : list or array-like
            The structure data to be validated. Can be:
            - A 3D list of structure constants (list of lists of lists)
            - A list of VFClass objects
            - A list of square matrices

        process_matrix_rep : bool, optional
            If True, the function will interpret the data as a list of square matrices
            and pass them to algebraDataFromMatRep to compute the structure data.

        Returns
        -------
        list
            The validated structure data as a list of lists of lists.

        Raises
        ------
        ValueError
            If the structure data is invalid or cannot be processed into a valid 3D list.
        """
        # Case 1: If process_matrix_rep is True, handle matrix representation
        if process_matrix_rep:
            if all(
                isinstance(sp.Matrix(obj), sp.Matrix)
                and sp.Matrix(obj).shape[0] == sp.Matrix(obj).shape[1]
                for obj in data
            ):
                return algebraDataFromMatRep(data)
            else:
                raise ValueError(
                    "sp.Matrix representation requires a list of square matrices."
                )

        # Case 2: If the input is a list of VFClass objects, handle vector fields
        if all(isinstance(obj, VFClass) for obj in data):
            return algebraDataFromVF(data)

        # Case 3: Validate and return the data as a list of lists of lists
        try:
            # Check that the data is a 3D list-like structure
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                if len(data) == len(data[0]) == len(data[0][0]):
                    return data  # Return as a validated 3D list
                else:
                    raise ValueError("Structure data must have 3D shape (x, x, x).")
            else:
                raise ValueError("Structure data format must be a 3D list of lists.")
        except Exception as e:
            raise ValueError(f"Invalid structure data format: {type(data)} - {e}")

    def __eq__(self, other):
        if not isinstance(other, FAClass):
            return NotImplemented
        return (
            self._structureData == other._structureData and
            self.label == other.label and
            self._basis_labels == other._basis_labels and
            self._grading == other._grading and
            self.basis == other.basis
        )

    def __hash__(self):
        return hash((self._structureData, self._label, self._basis_labels, self._grading, self.basis))

    def __contains__(self, item):
        return item in self.basis

    def __iter__(self):
        return iter(self.basis) 

    def __getitem__(self, indices):
        return self.structureData[indices[0]][indices[1]][indices[2]]

    def __repr__(self):
        """
        Provides a detailed representation of the FAClass object.
        Raises a warning if the instance is unregistered.
        """
        if not self._registered:
            warnings.warn(
                "This FAClass instance was initialized without an assigned label. "
                "It is recommended to initialize FAClass objects with dgcv creator functions like `createFiniteAlg` instead.",
                UserWarning,
            )
        return (
            f"FAClass(dim={self.dimension}, grading={self.grading}, "
            f"label={self.label}, basis_labels={self.basis_labels}, "
            f"struct_data={self.structureData})"
        )

    def _structure_data_summary(self):
        """
        Provides a summary of the structure data to avoid printing large matrices.
        """
        if self.dimension <= 3:
            return self.structureData
        return (
            "Structure data is large. Access the `structureData` attribute for details."
        )

    def __str__(self):
        """
        Provides a string representation of the FAClass object.
        Raises a warning if the instance is unregistered.
        """
        if not self._registered:
            warnings.warn(
                "This FAClass instance was initialized without an assigned label. "
                "It is recommended to initialize FAClass objects with dgcv creator functions like `createFiniteAlg` instead.",
                UserWarning,
            )

        def format_basis_label(label):
            return label

        formatted_label = self.label if self.label else "Unnamed Algebra"
        formatted_basis_labels = (
            ", ".join([format_basis_label(bl) for bl in self.basis_labels])
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
        """
        Hook for dgcv-specific display customization.
        Raises a warning if the instance is unregistered.
        """
        if not self._registered:
            warnings.warn(
                "This FAClass instance was initialized without an assigned label. "
                "It is recommended to initialize FAClass objects with dgcv creator functions like `createFiniteAlg` instead.",
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

    def _repr_latex_(self):
        """
        Provides a LaTeX representation of the FAClass object for Jupyter notebooks.
        Raises a warning if the instance is unregistered.
        """
        if not self._registered:
            warnings.warn(
                "This FAClass instance was initialized without an assigned label. "
                "It is recommended to initialize FAClass objects with dgcv creator functions like `createFiniteAlg` instead.",
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
                return "Unnamed Algebra"

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

        def format_basis_label(label):
            return rf"{label}" if label else "e_i"

        formatted_label = format_algebra_label(self.label)
        formatted_basis_labels = (
            ", ".join([format_basis_label(bl) for bl in self.basis_labels])
            if self.basis_labels
            else "No basis labels assigned"
        )
        return (
            f"Algebra: ${formatted_label}$, Basis: ${formatted_basis_labels}$, "
            f"Dimension: ${self.dimension}$, Grading: ${sp.latex(self.grading)}$"
        )

    def _sympystr(self):
        """
        SymPy string representation for FAClass.
        Raises a warning if the instance is unregistered.
        """
        if not self._registered:
            warnings.warn(
                "This FAClass instance was initialized without an assigned label. "
                "It is recommended to initialize FAClass objects with dgcv creator functions like `createFiniteAlg` instead.",
                UserWarning,
            )

        if self.label:
            return f"FAClass({self.label}, dim={self.dimension})"
        else:
            return f"FAClass(dim={self.dimension})"

    def _structure_data_summary_latex(self):
        """
        Provides a LaTeX summary of the structure data.
        Converts only numerical/symbolic elements to a matrix when possible.
        """
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
            print(
                "Warning: This FAClass instance is unregistered. Use createFiniteAlg to register it."
            )

        if self._skew_symmetric_cache is None:
            result, failure = self._check_skew_symmetric()
            self._skew_symmetric_cache = (result, failure)
        else:
            result, failure = self._skew_symmetric_cache

        if verbose:
            if result:
                print("The algebra is skew-symmetric.")
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
            print(
                "Warning: This FAClass instance is unregistered. Use createFiniteAlg to register it."
            )

        if self._jacobi_identity_cache is None:
            result, fail_list = self._check_jacobi_identity()
            self._jacobi_identity_cache = (result, fail_list)
        else:
            result, fail_list = self._jacobi_identity_cache

        if verbose:
            if result:
                print("The algebra satisfies the Jacobi identity.")
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

    def is_lie_algebra(self, verbose=False):
        """
        Checks if the algebra is a Lie algebra.
        Includes a warning for unregistered instances only if verbose=True.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints detailed information about the check.

        Returns
        -------
        bool
            True if the algebra is a Lie algebra, False otherwise.
        """
        if not self._registered and verbose:
            print(
                "Warning: This FAClass instance is unregistered. Use createFiniteAlg to register it."
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
            return False
        if not self.satisfies_jacobi_identity(verbose=verbose):
            self._lie_algebra_cache = False
            return False

        # If both checks pass, cache the result and return True
        self._lie_algebra_cache = True

        if verbose:
            if self.label is None:
                print("The algebra is a Lie algebra.")
            else:
                print(f"{self.label} is a Lie algebra.")

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
            raise ValueError(f"{method_name} can only be applied to Lie algebras.")

    def is_semisimple(self, verbose=False):
        """
        Checks if the algebra is semisimple.
        Includes a warning for unregistered instances only if verbose=True.
        """
        if not self._registered and verbose:
            print(
                "Warning: This FAClass instance is unregistered. Use createFiniteAlg to register it."
            )

        # Check if the algebra is a Lie algebra first
        if not self.is_lie_algebra(verbose=False):
            return False

        # Compute the determinant of the Killing form
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

        return det != 0

    def is_subspace_subalgebra(self, elements, return_structure_data=False):
        """
        Checks if a set of elements is a subspace subalgebra.

        Parameters
        ----------
        elements : list
            A list of AlgebraElement instances.
        return_structure_data : bool, optional
            If True, returns the structure constants for the subalgebra.

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
        span_matrix = sp.Matrix.hstack(*[el.coeffs for el in elements])
        linearly_independent = span_matrix.rank() == len(elements)

        if not linearly_independent:
            if return_structure_data:
                return {
                    "linearly_independent": False,
                    "closed_under_product": False,
                    "structure_data": None,
                }
            return False

        # Check closure under product and build structure data
        dim = len(elements)
        structure_data = [
            [[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)
        ]
        closed_under_product = True

        for i, el1 in enumerate(elements):
            for j, el2 in enumerate(elements):
                product = el1 * el2
                solution = span_matrix.solve_least_squares(product.coeffs)

                for k, coeff in enumerate(solution):
                    # Apply nsimplify to enforce exact representation
                    coeff_simplified = sp.nsimplify(coeff)
                    structure_data[i][j][k] = coeff_simplified

        if return_structure_data:
            return {
                "linearly_independent": linearly_independent,
                "closed_under_product": closed_under_product,
                "structure_data": structure_data,
            }

        return linearly_independent and closed_under_product

    def check_element_weight(self, element, test_weights = None):
        """
        Determines the weight vector of an AlgebraElement with respect to the grading vectors. Weight can be instead computed against another grading vector passed a list of weights as the keyword `test_weights`.

        Parameters
        ----------
        element : AlgebraElement
            The AlgebraElement to analyze.
        test_weights : list of int or sympy.Expr, optional (default: None)

        Returns
        -------
        list
            A list of weights corresponding to the grading vectors of this FAClass (or test_weights if provided).
            Each entry is either an integer, sympy.Expr (weight), the string 'AllW' (i.e., All Weights) if the element is the zero element,
            or 'NoW' (i.e., No Weights) if the element is not homogeneous.

        Notes
        -----
        - 'AllW' (meaning, All Weights) is returned for zero elements, which are compatible with all weights.
        - 'NoW' (meaning, No Weights) is returned for non-homogeneous elements that do not satisfy the grading constraints.
        """
        if not isinstance(element, AlgebraElement):
            raise TypeError("Input must be an instance of AlgebraElement.")

        if not test_weights:
            if not hasattr(self, "grading") or self._gradingNumber == 0:
                raise ValueError("This FAClass instance has no assigned grading vectors.")

        # Detect zero element
        if all(coeff == 0 for coeff in element.coeffs):
            return ["AllW"] * self._gradingNumber

        if test_weights:
            if not isinstance(test_weights,(list,tuple)):
                raise TypeError('`check_element_weight` expects `check_element_weight` to be None or a list/tuple of weight values (int,float, or sp.Expr).')
            if len(self.dimension) != len(test_weights) or not all([isinstance(j,(int,float,sp.Expr)) for j in test_weights]):
                raise TypeError('`check_element_weight` expects `check_element_weight` to be None or a length {self.dimension} list/tuple of weight values (int,float, or sp.Expr).')
            GVs = [test_weights]
        else:
            GVs = self.grading

        weights = []
        for g, grading_vector in enumerate(GVs):
            # Compute contributions of the element's basis components
            non_zero_indices = [
                i for i, coeff in enumerate(element.coeffs) if coeff != 0
            ]

            # Check homogeneity
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
                "No grading vectors are assigned to this FAClass instance."
            )
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
            A list of AlgebraElement instances that span the center of the algebra.

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
            )

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

        solutions = sp.solve(eqns, temp_vars, dict=True)
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
        FAClass
            A new FAClass instance representing the derived algebra.

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
            )
        if not subalgebra_data["closed_under_product"]:
            raise ValueError(
                "Failed to compute the derived algebra: commutators are not closed under the product."
            )

        # Extract independent generators and structure data
        independent_generators = subalgebra_data.get(
            "independent_elements", commutators
        )
        structure_data = subalgebra_data["structure_data"]

        # Create the derived algebra
        return FAClass(
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
        elements : list of AlgebraElement
            The set of elements to filter.

        Returns
        -------
        list of AlgebraElement
            A subset of the input elements that are linearly independent and unique.
        """

        # Remove duplicate elements based on their coefficients
        unique_elements = []
        seen_coeffs = set()
        for el in elements:
            coeff_tuple = tuple(el.coeffs)  # Convert coeffs to a tuple for hashability
            if coeff_tuple not in seen_coeffs:
                seen_coeffs.add(coeff_tuple)
                unique_elements.append(el)

        # Create a matrix where each column is the coefficients of an element
        coeff_matrix = sp.Matrix.hstack(*[el.coeffs for el in unique_elements])

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
        - If `basis_labels` is None, defaults to "e1", "e2", ..., "ed".
        """
        import pandas as pd

        # Dimension of the algebra
        dimension = self.dimension

        # Default labels if basis_labels is None
        basis_labels = self.basis_labels or [f"e{i+1}" for i in range(dimension)]

        # Initialize the structure matrix as a list of lists
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

        # Return as a list of lists
        return structure_matrix

    def is_ideal(self, subspace_elements):
        """
        Checks if the given list of elgebra elements spans an ideal.

        Parameters
        ----------
        subspace_elements : list
            A list of AlgebraElement instances representing the subspace
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
            if not isinstance(el, AlgebraElement) or el.algebra != self:
                raise ValueError("All elements in subspace_elements must belong to this algebra.")

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
        Checks if a given AlgebraElement is in the span of subspace_elements.

        Parameters
        ----------
        element : AlgebraElement
            The element to check.
        subspace_elements : list
            A list of AlgebraElement instances representing the subspace they span.

        Returns
        -------
        bool
            True if the element is in the span of subspace_elements, False otherwise.
        """
        # Build a matrix where columns are the coefficients of subspace_elements
        span_matrix = sp.Matrix.hstack(*[el.coeffs for el in subspace_elements])

        # Solve for the coefficients that express `element` as a linear combination
        product_vector = sp.Matrix(element.coeffs)
        solution = span_matrix.solve_least_squares(product_vector)

        # Check if the solution satisfies the equation
        return span_matrix * solution == product_vector

# algebra element class
class AlgebraElement(sp.Basic):
    def __new__(cls, algebra, coeffs, valence, format_sparse=False):

        if not isinstance(algebra, FAClass):
            raise TypeError(
                "AlgebraElement expects the first argument to be an instance of FAClass."
            )

        if valence not in {0, 1}:
            raise TypeError("AlgebraElement expects valence to be 0 or 1.")

        coeffs = tuple(coeffs)

        obj = sp.Basic.__new__(cls, algebra, coeffs, valence, format_sparse)
        return obj

    def __init__(self, algebra, coeffs, valence, format_sparse=False):
        self.algebra = algebra
        self.vectorSpace = algebra
        self.valence = valence
        self.is_sparse = format_sparse

        # Store coeffs as an immutable tuple of tuples
        if isinstance(coeffs, (list, tuple)):  
            self.coeffs = tuple(coeffs)
        else:
            raise TypeError("AlgebraElement expects coeffs to be a list or tuple.")

    def __eq__(self, other):
        if not isinstance(other, AlgebraElement):
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
        """
        Custom string representation for vectorSpaceElement.
        Displays the linear combination of basis elements with coefficients.
        Handles unregistered parent vector space by raising a warning.
        """
        if not self.algebra._registered:
            warnings.warn(
                "This vectorSpaceElement's parent vector space (vectorSpace) was initialized without an assigned label. "
                "It is recommended to initialize vectorSpace objects with dgcv creator functions like `createVectorSpace` instead.",
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
                return f"0 * {self.algebra.basis_labels[0] if self.algebra.basis_labels else 'e_1'}"
            else:
                return f"0 * {self.algebra.basis_labels[0] if self.algebra.basis_labels else 'e_1'}^\'\'"

        return " + ".join(terms).replace("+ -", "- ")

    def _class_builder(self,coeffs,valence,format_sparse=False):
        return AlgebraElement(self.algebra,coeffs,valence,format_sparse=False)

    def _repr_latex_(self):
        """
        Provides a LaTeX representation of vectorSpaceElement for Jupyter notebooks.
        Handles unregistered parent vector space by raising a warning.
        """
        if not self.algebra._registered:
            warnings.warn(
                "This vectorSpaceElement's parent vector space (vectorSpace) was initialized without an assigned label. "
                "It is recommended to initialize vectorSpace objects with dgcv creator functions like `createVectorSpace` instead.",
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
            return rf"$0 \cdot {self.algebra.basis_labels[0] if self.algebra.basis_labels else 'e_1'}$"

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

        return rf"$\text{{Element of }} {format_algebra_label(self.algebra.label)}: {result}$"

    def _latex(self):
        return self._repr_latex_()

    def _sympystr(self):
        """
        SymPy string representation for AlgebraElement.
        Handles unregistered parent algebra by raising a warning.
        """
        if not self.algebra._registered:
            warnings.warn(
                "This AlgebraElement's parent algebra (FAClass) was initialized without an assigned label. "
                "It is recommended to initialize FAClass objects with dgcv creator functions like `createFiniteAlg` instead.",
                UserWarning,
            )

        coeffs_str = ", ".join(map(str, self.coeffs))
        if self.algebra.label:
            return f"AlgebraElement({self.algebra.label}, coeffs=[{coeffs_str}])"
        else:
            return f"AlgebraElement(coeffs=[{coeffs_str}])"

    def _latex(self, printer):
        """
        Provides a LaTeX representation of AlgebraElement for SymPy's latex() function.
        Displays only the element itself without the algebra label text.
        """
        if not self.algebra._registered:
            warnings.warn(
                "This AlgebraElement's parent vector space (FAClass) was initialized without an assigned label. "
                "It is recommended to initialize FAClass objects with dgcv creator functions like `createFiniteAlg` instead.",
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
        return result

    def _latex_verbose(self, printer):
        """
        Provides a LaTeX representation of AlgebraElement for SymPy's latex() function.
        Handles unregistered parent vector space by raising a warning.
        """
        if not self.algebra._registered:
            warnings.warn(
                "This AlgebraElement's parent vector space (FAClass) was initialized without an assigned label. "
                "It is recommended to initialize FAClass objects with dgcv creator functions like `createFiniteAlg` instead.",
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
        """
        Representation of vectorSpaceElement.
        Shows the linear combination of basis elements with coefficients.
        Falls back to __str__ if basis_labels is None.
        """
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
        return AlgebraElement(self.algebra, newCoeffs, format_sparse=self.is_sparse)

    def dual(self):
        return AlgebraElement(self.algebra, self.coeffs, (self.valence+1)%2,format_sparse=self.is_sparse)

    def _convert_to_tp(self):
        return tensorProduct(self.algebra,{(j,self.valence):self.coeffs[j] for j in range(self.algebra.dimension)})

    def _recursion_contract_hom(self, other):
        return self._convert_to_tp()._recursion_contract_hom(other)

    def __add__(self, other):
        if hasattr(other,"is_zero") and other.is_zero():
            return self
        if isinstance(other, AlgebraElement):
            if self.algebra == other.algebra and self.valence==other.valence:
                return AlgebraElement(
                    self.algebra,
                    [self.coeffs[j] + other.coeffs[j] for j in range(len(self.coeffs))],
                    self.valence,
                    format_sparse=self.is_sparse,
                )
            else:
                raise TypeError(
                    "AlgebraElement operands for + must belong to the same FAClass."
                )
        if isinstance(other,tensorProduct) and other.max_degree==1 and other.min_degree==1 and other.vector_space==self.algebra:
            pt = other.prolongation_type
            coeffs = [other.coeff_dict[(j,pt)] if (j,pt) in other.coeff_dict else 0 for j in range(other.vector_space.dimension)]
            LA_elem = other._class_builder(coeffs,pt,format_sparse=False)
            return self+LA_elem
        else:
            raise TypeError(
                "Unsupported operand type(s) for + with the AlgebraElement class"
            )

    def __sub__(self, other):
        if hasattr(other,"is_zero") and other.is_zero():
            return self
        if isinstance(other, AlgebraElement):
            if self.algebra == other.algebra and self.valence==other.valence:
                return AlgebraElement(
                    self.algebra,
                    [self.coeffs[j] - other.coeffs[j] for j in range(len(self.coeffs))],
                    self.valence,
                    format_sparse=self.is_sparse,
                )
            else:
                raise TypeError(
                    "AlgebraElement operands for - must belong to the same FAClass."
                )
        if isinstance(other,tensorProduct):
            if other.max_degree==1 and other.min_degree==1:
                if other.vector_space==self.algebra:
                    pt = other.prolongation_type
                    coeffs = [other.coeff_dict[(j,pt)] if (j,pt) in other.coeff_dict else 0 for j in range(other.vector_space.dimension)]
                    LA_elem = other._class_builder(coeffs,pt,format_sparse=False)
                    return self-LA_elem
        else:
            raise TypeError(
                f"Unsupported operand type(s) {type(other)} for - with the AlgebraElement class"
            )

    def __mul__(self, other):
        """
        Multiplies two AlgebraElement objects by multiplying their coefficients
        and summing the results based on the algebra's structure constants. Also handles
        multiplication with scalars.

        Args:
            other (AlgebraElement) or (scalar): The algebra element or scalar to multiply with.

        Returns:
            AlgebraElement: The result of the multiplication.
        """
        if isinstance(other, AlgebraElement):
            if self.algebra == other.algebra and self.valence==other.valence:
                # Initialize result coefficients as a list of zeros
                result_coeffs = [0] * self.algebra.dimension

                # Loop over each pair of basis element coefficients
                for i in range(self.algebra.dimension):
                    for j in range(self.algebra.dimension):
                        # Compute the scalar product of the current coefficients
                        scalar_product = self.coeffs[i] * other.coeffs[j]

                        # Multiply scalar_product with the corresponding vector from structureData
                        structure_vector_product = [
                            scalar_product * element
                            for element in self.algebra.structureData[i][j]
                        ]

                        # Sum the resulting vector into result_coeffs element-wise
                        result_coeffs = [
                            sp.sympify(result_coeffs[k] + structure_vector_product[k])
                            for k in range(len(result_coeffs))
                        ]

                # Return a new AlgebraElement with the updated coefficients
                return AlgebraElement(
                    self.algebra, result_coeffs, self.valence, format_sparse=self.is_sparse
                )
            else:
                raise TypeError(
                    "Both operands for * must be AlgebraElement instances from the same FAClass."
                )
        elif isinstance(other, tensorProduct):
            return (self._convert_to_tp())*other
        elif isinstance(other, (int, float, sp.Expr)):
            # Scalar multiplication case
            new_coeffs = [coeff * other for coeff in self.coeffs]
            # Return a new AlgebraElement with the updated coefficients
            return AlgebraElement(
                self.algebra, new_coeffs, self.valence, format_sparse=self.is_sparse
            )
        else:
            raise TypeError(
                f"Multiplication is only supported for scalars and the AlegebraElement class, not {type(other)}"
            )

    def __rmul__(self, other):
        # If other is a scalar, treat it as commutative
        if isinstance(
            other, (int, float, sp.Expr)
        ):
            return self * other  # Calls __mul__ (which is already implemented)
        elif isinstance(other, AlgebraElement):
            -1 * other * self
        elif isinstance(other, tensorProduct):
            return other*(self._convert_to_tp())
        else:
            raise TypeError(
                f"Right multiplication is only supported for scalars and the AlegebraElement class, not {type(other)}"
            )

    def __matmul__(self, other):
        """Overload @ operator for tensor product."""
        if not isinstance(other, AlgebraElement) or other.algebra!=self.algebra:
            raise TypeError('`@` only supports tensor products between AlgebraElement instances with the same algebra attribute')
        new_dict = {(j,k,self.valence,other.valence):self.coeffs[j]*other.coeffs[k] for j in range(self.algebra.dimension) for k in range(self.algebra.dimension)}
        return tensorProduct(self.algebra, new_dict)


    def __xor__(self, other):
        if other == '':
            return self.dual()
            raise ValueError("Invalid operation. Use `^ ''` to denote the dual.")


    def check_element_weight(self):
        """
        Determines the weight vector of this AlgebraElement with respect to its FAClass' grading vectors.

        Returns
        -------
        list
            A list of weights corresponding to the grading vectors of the parent FAClass.
            Each entry is either an integer, sympy.Expr (weight), the string 'AllW' if the element is the zero element,
            or 'NoW' if the element is not homogeneous.

        Notes
        -----
        - This method calls the parentt FAClass' check_element_weight method.
        - 'AllW' is returned for zero elements, which are compaible with all weights.
        - 'NoW' is returned for non-homogeneous elements that do not satisfy the grading constraints.
        """
        if not hasattr(self, "algebra") or not isinstance(self.algebra, FAClass):
            raise ValueError(
                "This AlgebraElement is not associated with a valid FAClass."
            )

        return self.algebra.check_element_weight(self)

# subspaces in FAClass
class algebraSubspace(sp.Basic):
    def __new__(cls,basis,alg, test_weights=None):
        if not isinstance(alg, FAClass):
            raise TypeError('algebraSubspace expects second argument to an FAClass instance.')
        if not isinstance(basis,(list,tuple)):
            raise TypeError('algebraSubspace expects first argument to a be a list or tuple of AlgebraElement instances')
        if not all(isinstance(j,AlgebraElement) for j in basis):
            raise TypeError('algebraSubspace expects first argument to a be a list or tuple of AlgebraElement instances')
        if not all([j.algebra == alg for j in basis]):
            raise TypeError('algebraSubspace expects all AlgebraElement instances given in the first argument to have the same `AlgebraElement.algebra` value as the second argument.')
        if test_weights:
            if not isinstance(test_weights,(list,tuple)):
                raise TypeError('`check_element_weight` expects `check_element_weight` to be None or a list/tuple of weight values (int,float, or sp.Expr).')
            if len(alg.dimension) != len(test_weights) or not all([isinstance(j,(int,float,sp.Expr)) for j in test_weights]):
                raise TypeError('`check_element_weight` expects `check_element_weight` to be None or a length {alg.dimension} list/tuple of weight values (int,float, or sp.Expr).')
        filtered_basis = alg.filter_independent_elements(basis)
        if len(filtered_basis)<len(basis):
            basis = filtered_basis
            warnings.warn('The given list for `basis` was not linearly independent, so the algebraSubspace initializer computed a basis for its span to use instead.')

        # Create the new instance
        obj = sp.Basic.__new__(cls, basis, alg, test_weights)

        return obj

    def __init__(self, basis, alg, test_weights=None):
        self.ambiant = alg
        self.basis = tuple(basis)
        grading = []
        for elem in basis:
            weight = alg.check_element_weight(elem,test_weights=test_weights)
            if weight == 'NoW':
                raise ValueError('`algebraSubspace` was given a basis that is not compatible with the given weights in `test_weights`.')
            grading.append(weight)
        self.grading = grading

        # immutables
        self._grading = tuple(grading)
        self._test_weights = tuple(test_weights) if test_weights else None
    def __eq__(self, other):
        if not isinstance(other, algebraSubspace):
            return NotImplemented
        return (
            self.ambiant == other.ambiant and
            self.basis == other.basis and
            self._grading == other._grading and
            self._test_weights == other._test_weights
        )
    def __hash__(self):
        return hash((self.ambiant, self.basis, self._grading, self._test_weights))
############## finite algebra creation 


def createFiniteAlg(
    obj,
    label,
    basis_labels=None,
    grading=None,
    format_sparse=False,
    process_matrix_rep=False,
    verbose=False,
):
    """
    Registers an algebra object and its basis elements in the caller's global namespace,
    and adds them to the variable_registry for tracking in the Variable Management Framework.

    Parameters
    ----------
    obj : FAClass, structure data, or list of AlgebraElement
        The algebra object (an instance of FAClass), the structure data used to create one,
        or a list of AlgebraElement instances with the same parent algebra.
    label : str
        The label used to reference the algebra object in the global namespace.
    basis_labels : list, optional
        A list of custom labels for the basis elements of the algebra.
        If not provided, default labels will be generated.
    grading : list of lists or list, optional
        A list specifying the grading(s) of the algebra.
    format_sparse : bool, optional
        Whether to use sparse arrays when creating the FAClass object.
    process_matrix_rep : bool, optional
        Whether to compute and store the matrix representation of the algebra.
    verbose : bool, optional
        If True, provides detailed feedback during the creation process.
    """

    if label in listVar(algebras_only=True):
        warnings.warn('`createFiniteAlg` was called with a `label` already assigned to another algebra, so `createFiniteAlg` will overwrite the other algebra.')
        clearVar(label)

    def validate_structure_data(data, process_matrix_rep=False):
        """
        Validates the structure data and converts it to a list of lists of lists.
        """
        # Case 1: If process_matrix_rep is True, handle matrix representation
        if process_matrix_rep:
            if all(
                isinstance(sp.Matrix(obj), sp.Matrix)
                and sp.Matrix(obj).shape[0] == sp.Matrix(obj).shape[1]
                for obj in data
            ):
                return algebraDataFromMatRep(data)
            else:
                raise ValueError(
                    "sp.Matrix representation requires a list of square matrices."
                )

        # Case 2: If the input is a list of VFClass objects, handle vector fields
        if all(isinstance(obj, VFClass) for obj in data):
            return algebraDataFromVF(data)

        # Case 3: Validate and return the data as a list of lists of lists
        try:
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                if len(data) == len(data[0]) == len(data[0][0]):
                    return data
                else:
                    raise ValueError("Structure data must have 3D shape (x, x, x).")
            else:
                raise ValueError("Structure data format must be a 3D list of lists.")
        except Exception as e:
            raise ValueError(f"Invalid structure data format: {type(data)} - {e}")

    def extract_structure_from_elements(elements):
        """
        Computes structure constants and validates linear independence from a list of AlgebraElement.

        Parameters
        ----------
        elements : list of AlgebraElement
            A list of AlgebraElement instances.

        Returns
        -------
        structure_data : list of lists of lists
            The structure constants for the subalgebra spanned by the elements.

        Raises
        ------
        ValueError
            If the elements are not linearly independent or not closed under the algebra product.
        """
        if not elements or not all(isinstance(el, AlgebraElement) for el in elements):
            raise ValueError(
                "Invalid input: All elements must be instances of AlgebraElement."
            )

        # Check that all elements have the same parent algebra
        parent_algebra = elements[0].algebra
        if not all(el.algebra == parent_algebra for el in elements):
            raise ValueError(
                "All AlgebraElement instances must share the same parent algebra."
            )

        try:
            # Use the parent algebra's is_subspace_subalgebra method
            result = parent_algebra.is_subspace_subalgebra(
                elements, return_structure_data=True
            )
        except ValueError as e:
            raise ValueError(
                "Error during subalgebra validation. "
                "The input list of AlgebraElement instances must be linearly independent and closed under the algebra product. "
                f"Original error: {e}"
            ) from e

        if not result["linearly_independent"]:
            raise ValueError(
                "The input elements are not linearly independent. "
            )

        if not result["closed_under_product"]:
            raise ValueError(
                "The input elements are not closed under the algebra product. "
            )

        # Return structure data
        return result["structure_data"]

    # Validate or create the FAClass object
    if isinstance(obj, FAClass):
        if verbose:
            print(f"Using existing FAClass instance: {label}")
        structure_data = obj.structureData
        dimension = obj.dimension
    elif isinstance(obj, list) and all(isinstance(el, AlgebraElement) for el in obj):
        if verbose:
            print("Creating algebra from list of AlgebraElement instances.")
        structure_data = extract_structure_from_elements(obj)
        dimension = len(obj)
    else:
        if verbose:
            print("Validating or processing structure data.")
        structure_data = validate_structure_data(
            obj, process_matrix_rep=process_matrix_rep
        )
        dimension = len(structure_data)

    # Create or validate basis labels
    if basis_labels is None:
        basis_labels = [validate_label(f"{label}_{i+1}") for i in range(dimension)]
    validate_label_list(basis_labels)

    # Process grading
    if grading is None:
        grading = [sp.Tuple(*([0] * dimension))]  # Default grading: all zeros
    elif isinstance(grading, (list, tuple)) and all(
        isinstance(w, (int, sp.Expr)) for w in grading
    ):
        # Single grading vector
        if len(grading) != dimension:
            raise ValueError(
                f"Grading vector length ({len(grading)}) must match the algebra dimension ({dimension})."
            )
        grading = [sp.Tuple(*grading)]  # Wrap single vector in a list
    elif isinstance(grading, list) and all(
        isinstance(vec, (list, tuple)) for vec in grading
    ):
        # List of grading vectors
        for vec in grading:
            if len(vec) != dimension:
                raise ValueError(
                    f"Grading vector length ({len(vec)}) must match the algebra dimension ({dimension})."
                )
        grading = [sp.Tuple(*vec) for vec in grading]  # Convert each vector to Tuple
    else:
        raise ValueError("Grading must be a single vector or a list of vectors.")

    # Create the FAClass object
    passkey = retrieve_passkey()
    algebra_obj = FAClass(
        structure_data=structure_data,
        grading=grading,
        format_sparse=format_sparse,
        process_matrix_rep=process_matrix_rep,
        _label=label,
        _basis_labels=basis_labels,
        _calledFromCreator=passkey,
    )

    # Make sure algebra object and its basis are fully initialized
    assert (
        algebra_obj.basis is not None
    ), "Algebra object basis elements must be initialized."

    # Register in _cached_caller_globals
    _cached_caller_globals.update({label: algebra_obj})
    _cached_caller_globals.update(zip(basis_labels, algebra_obj.basis))

    # Register in the variable registry
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

    if verbose:
        print(f"Algebra '{label}' registered successfully.")
        print(
            f"Dimension: {dimension}, Grading: {grading}, Basis Labels: {basis_labels}"
        )

    return algebra_obj


def algebraDataFromVF(vector_fields):
    """
    Create the structure data array for a Lie algebra from a list of vector fields in *vector_fields*.

    This function computes the Lie algebra structure constants from a list of vector fields
    (instances of VFClass) defined on the same variable space. The returned structure data
    can be used to initialize an FAClass instance.

    Parameters
    ----------
    vector_fields : list
        A list of VFClass instances, all defined on the same variable space with respect to the same basis.

    Returns
    -------
    list
        A 3D array-like list of lists of lists representing the Lie algebra structure data.

    Raises
    ------
    Exception
        If the vector fields do not span a Lie algebra or are not defined on a common basis.

    Notes
    -----
    This function dynamically chooses its approach to solve for the structure constants:
    - For smaller dimensional algebras, it substitutes pseudo-arbitrary values for the variables in `varSpaceLoc`
      based on a power function to create a system of linear equations.
    - For larger systems, where `len(varSpaceLoc)` raised to `len(vector_fields)` exceeds a threshold (default is 10,000),
      random rational numbers are used for substitution to avoid performance issues caused by large numbers.
    """
    # Define the product threshold for switching to random sampling
    product_threshold = 1

    # Check if all vector fields are defined on the same variable space
    if len(set([vf.varSpace for vf in vector_fields])) != 1:
        raise Exception(
            "algebraDataFromVF requires vector fields defined with respect to a common basis."
        )

    complexHandling = any(vf.dgcvType == "complex" for vf in vector_fields)
    if complexHandling:
        vector_fields = [allToReal(j) for j in vector_fields]
    varSpaceLoc = vector_fields[0].varSpace

    # Create temporary variables for solving structure constants
    tempVarLabel = "T" + retrieve_public_key()
    variableProcedure(tempVarLabel, len(vector_fields), _tempVar=retrieve_passkey())
    combiVFLoc = addVF(
        *[
            _cached_caller_globals[tempVarLabel][j] * vector_fields[j]
            for j in range(len(_cached_caller_globals[tempVarLabel]))
        ]
    )

    def computeBracket(j, k):
        """
        Compute and return the Lie bracket [vf_j, vf_k] and structure constants.

        Parameters
        ----------
        j : int
            Index of the first vector field.
        k : int
            Index of the second vector field.

        Returns
        -------
        list
            Structure constants for the Lie bracket of vf_j and vf_k.
        """
        if k <= j:
            return [0] * len(_cached_caller_globals[tempVarLabel])

        # Compute the Lie bracket
        bracket = VF_bracket(vector_fields[j], vector_fields[k]) - combiVFLoc

        if complexHandling:
            bracket = [allToReal(expr) for expr in bracket.coeffs]
        else:
            bracket = bracket.coeffs

        if len(varSpaceLoc) ** len(vector_fields) <= product_threshold:
            # Use the current system of pseudo-arbitrary substitutions
            bracketVals = list(
                set(
                    sum(
                        [
                            [
                                expr.subs(
                                    [
                                        (
                                            varSpaceLoc[i],
                                            sp.Rational((i + 1) ** sampling_index, 32),
                                        )
                                        for i in range(len(varSpaceLoc))
                                    ]
                                )
                                for expr in bracket
                            ]
                            for sampling_index in range(len(vector_fields))
                        ],
                        [],
                    )
                )
            )
        else:
            # Use random sampling system for larger cases
            def random_rational():
                return sp.Rational(random.randint(1, 1000), random.randint(1001, 2000))            
            bracketVals = list(
                set(
                    sum(
                        [
                            [
                                expr if not hasattr(expr,'subs') else
                                expr.subs(
                                    [
                                        (varSpaceLoc[i], random_rational())
                                        for i in range(len(varSpaceLoc))
                                    ]
                                )
                                for expr in bracket
                            ]
                            for _ in range(len(vector_fields))
                        ],
                        [],
                    )
                )
            )

        # Solve the system of equations
        solutions = list(sp.linsolve(bracketVals, _cached_caller_globals[tempVarLabel]))

        if len(solutions) == 1:
            # Extract the solution and substitute into all temporary variables
            sol_values = solutions[0]

            # Substitute back into the original bracket
            substituted_constants = [
                expr.subs(zip(_cached_caller_globals[tempVarLabel], sol_values))
                for expr in _cached_caller_globals[tempVarLabel]
            ]

            return substituted_constants
        else:
            raise Exception(
                f"Fields at positions {j} and {k} are not closed under Lie brackets."
            )

    # Precompute all necessary Lie brackets and store as 3D list
    structure_data = [
        [[0 for _ in vector_fields] for _ in vector_fields] for _ in vector_fields
    ]

    for j in range(len(vector_fields)):
        for k in range(j + 1, len(vector_fields)):
            structure_data[j][k] = computeBracket(j, k)
            structure_data[k][j] = [-elem for elem in structure_data[j][k]]

    # Clean up temporary variables
    clearVar(*listVar(temporary_only=True), report=False)

    return structure_data


def algebraDataFromMatRep(mat_list):
    """
    Create the structure data array for a Lie algebra from a list of matrices in *mat_list*.

    This function computes the Lie algebra structure constants from a matrix representation of a Lie algebra.
    The returned structure data can be used to initialize an FAClass instance.

    Parameters
    ----------
    mat_list : list
        A list of square matrices of the same size representing the Lie algebra.

    Returns
    -------
    list
        A 3D list of lists of lists representing the Lie algebra structure data.

    Raises
    ------
    Exception
        If the matrices do not span a Lie algebra, or if the matrices are not square and of the same size.
    """
    if isinstance(mat_list, list):
        mListLoc = [
            sp.Matrix(j) for j in mat_list
        ]  # Convert input to sympy sp.Matrix objects
        shapeLoc = mListLoc[0].shape[0]

        # Check that all matrices are square and of the same size
        if all(j.shape == (shapeLoc, shapeLoc) for j in mListLoc):
            # Temporary variables for solving the commutators
            tempVarLabel = "T" + retrieve_public_key()
            variableProcedure(tempVarLabel, len(mat_list), _tempVar=retrieve_passkey())

            # Create a symbolic matrix to solve for commutators
            combiMatLoc = sum(
                [
                    _cached_caller_globals[tempVarLabel][j] * mListLoc[j]
                    for j in range(len(_cached_caller_globals[tempVarLabel]))
                ],
                sp.zeros(shapeLoc, shapeLoc),
            )

            def pairValue(j, k):
                """
                Compute the commutator [m_j, m_k] and match with the combination matrix.

                Parameters
                ----------
                j : int
                    Index of the first matrix in the commutator.
                k : int
                    Index of the second matrix in the commutator.

                Returns
                -------
                list
                    The coefficients representing the structure constants.
                """
                bracketVals = list(
                    set(
                        (
                            mListLoc[j] * mListLoc[k]
                            - mListLoc[k] * mListLoc[j]
                            - combiMatLoc
                        ).vec()
                    )
                )

                solLoc = list(
                    sp.linsolve(bracketVals, _cached_caller_globals[tempVarLabel])
                )

                if len(solLoc) == 1:
                    return [
                        expr.subs(zip(_cached_caller_globals[tempVarLabel], solLoc[0]))
                        for expr in _cached_caller_globals[tempVarLabel]
                    ]
                else:
                    raise Exception(
                        f"Unable to determine if matrices are closed under commutators. "
                        f"Problem matrices are in positions {j} and {k}."
                    )

            # Assemble the structure data array from commutators and store as 3D list
            structure_data = [
                [
                    pairValue(j, k)
                    for j in range(len(_cached_caller_globals[tempVarLabel]))
                ]
                for k in range(len(_cached_caller_globals[tempVarLabel]))
            ]

            # Clear all temporary variables
            clearVar(*listVar(temporary_only=True), report=False)

            return structure_data
        else:
            raise Exception(
                "algebraDataFromMatRep expects a list of square matrices of the same size."
            )
    else:
        raise Exception("algebraDataFromMatRep expects a list of square matrices.")


def createClassicalLA(
    series: str,
    label: str = None,
    basis_labels: list = None,
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
    FAClass
        The resulting Lie algebra as an FAClass instance.

    Raises
    ------
    ValueError
        If the series is not recognized or not implemented.

    Notes
    -----
    - Currently supports only the A series (special linear Lie algebras: A_n = sl(n+1)).
    """
    # Extract series type and rank
    try:
        series_type, rank = series[0], int(series[1:])
    except (IndexError, ValueError):
        raise ValueError(f"Invalid series format: {series}. Expected a letter 'A', 'B', 'C', or 'D' followed by a positive integer, like 'A1', 'B5', etc.")
    if rank <= 0:
            raise ValueError(f"Sequence index must be a positive integer, but got: {rank}.")

    def generate_A_series_structure_data(n):
        """
        Generates the structure data and weights for the A_n series (special linear: sl(n+1)).

        Parameters
        ----------
        n : int
            The relevant term in the A-series

        Returns
        -------
        tuple
            - basis (list): A 3-dimensional list representing the structure data for sl(n+1).
                            Each element is a 2D list of lists representing a trace-free (n+1)x(n+1) matrix.
            - weight_vectors (list): A list containing one inner list representing the weight vectors of the basis.

        Notes
        -----
        - Basis includes off-diagonal elementary matrices E_{j,k} (j < k first, then j > k).
        - Diagonal basis elements are trace-free combinations of E_{j,j}.
        - Weight vectors:
            - Off-diagonal E_{j,k} is assigned weight j-k.
            - Diagonal trace-free matrices are assigned weight 0.
        """
        # Dimension of the matrices
        matrix_dim = n + 1

        # Basis elements and weight vectors
        basis = []
        weight_vectors = []

        # Add off-diagonal elements with j < k
        for j in range(matrix_dim):
            for k in range(j + 1, matrix_dim):
                E_jk = [[0] * matrix_dim for _ in range(matrix_dim)]
                E_jk[j][k] = 1  # Set the (j, k)-entry to 1
                basis.append(E_jk)
                weight_vectors.append(j - k)

        # Add diagonal trace-free elements
        for j in range(matrix_dim - 1):
            E_diag = [[0] * matrix_dim for _ in range(matrix_dim)]
            E_diag[j][j] = 1       # +1 for the j-th diagonal entry
            E_diag[j + 1][j + 1] = -1  # -1 for the (j+1)-th diagonal entry
            basis.append(E_diag)
            weight_vectors.append(0)

        # Add off-diagonal elements with j > k
        for j in range(matrix_dim):
            for k in range(j):
                E_jk = [[0] * matrix_dim for _ in range(matrix_dim)]
                E_jk[j][k] = 1  # Set the (j, k)-entry to 1
                basis.append(E_jk)
                weight_vectors.append(j - k)

        # Wrap weight vectors in a list (for compatibility with multiple weight systems)
        weight_vectors = [weight_vectors]

        return basis, weight_vectors

    def generate_B_series_structure_data(n):
        """
        Generates the structure data the B_n series 
        (special orthogonal: so(2n+1)).

        Parameters
        ----------
        n : int
            The rank of the B-series Lie algebra (2n+1 is the matrix dimension).

        Returns
        -------
        tuple
            - basis (list): A 3-dimensional list representing the structure data for so(2n+1).
                            Each element is a 2D list of lists representing a skew-symmetric (2n+1)x(2n+1) matrix.
            - weight_vectors (list): An empty list (weights not assigned for B-series).

        Notes
        -----
        - Basis includes skew-symmetric matrices E_{j,k} - E_{k,j} (j < k).
        - Weight vector assignment for B-series is non-trivial and left empty for now.
        """
        # Dimension of the matrices
        matrix_dim = 2 * n + 1

        # Basis elements
        basis = []

        # Generate skew-symmetric off-diagonal matrices E_{j,k} - E_{k,j}
        for j in range(matrix_dim):
            for k in range(j + 1, matrix_dim):
                # Create a zero matrix
                skew_symmetric = [[0] * matrix_dim for _ in range(matrix_dim)]
                skew_symmetric[j][k] = 1   # Set the (j, k)-entry to 1
                skew_symmetric[k][j] = -1  # Set the (k, j)-entry to -1
                basis.append(skew_symmetric)

        # Return basis and empty weight_vectors
        return basis, None

    def generate_C_series_structure_data(n):
        """
        Generates the structure data and weight vectors for the C_n series 
        (symplectic Lie algebra: sp(2n)).

        Parameters
        ----------
        n : int
            The rank of the C-series Lie algebra (2n is the matrix dimension).

        Returns
        -------
        tuple
            - basis (list): A 3-dimensional list representing the structure data for sp(2n).
                            Each element is a 2D list of lists representing a matrix.
            - weight_vectors (list): A single weight vector for the basis elements, 
                                    representing a grading of the algebra.

        Notes
        -----
        - Basis matrices are partitioned into nxn blocks and constructed in three groups:
        1. Lower block triangular: [[0, 0], [S_{j,k}, 0]] (weight = -1).
        2. Block diagonal: [[E_{j,k}, 0], [0, -E_{k,j}]] (weight = 0).
        3. Upper block triangular: [[0, S_{j,k}], [0, 0]] (weight = 1).
        """
        # Dimension of the full matrices
        matrix_dim = 2 * n

        # Basis elements and weight vector
        basis = []
        weight_vector = []

        # Step 1: Create symmetric matrices S_{j,k} = E_{j,k} + E_{k,j} for j ≤ k
        symmetric_matrices = []
        for j in range(n):
            for k in range(j, n):
                S = [[0] * n for _ in range(n)]
                S[j][k] = 1
                if j != k:
                    S[k][j] = 1
                symmetric_matrices.append(S)

        # Step 2: Create pairs P_{j,k} = (E_{j,k}, -E_{k,j})
        matrix_pairs = []
        for j in range(n):
            for k in range(n):
                P1 = [[0] * n for _ in range(n)]
                P2 = [[0] * n for _ in range(n)]
                P1[j][k] = 1
                P2[k][j] = -1
                matrix_pairs.append((P1, P2))

        # Step 3: Create basis matrices in three groups
        # Group 1: Lower block triangular [[0, 0], [S_{j,k}, 0]] (weight = -1)
        for S in symmetric_matrices:
            lower_triangular = [[0] * matrix_dim for _ in range(matrix_dim)]
            # Insert S into the lower-left block
            for i in range(n):
                for j in range(n):
                    lower_triangular[n + i][j] = S[i][j]
            basis.append(lower_triangular)
            weight_vector.append(-1)

        # Group 2: Block diagonal [[E_{j,k}, 0], [0, -E_{k,j}]] (weight = 0)
        for P1, P2 in matrix_pairs:
            block_diagonal = [[0] * matrix_dim for _ in range(matrix_dim)]
            # Insert P1 into the top-left block and P2 into the bottom-right block
            for i in range(n):
                for j in range(n):
                    block_diagonal[i][j] = P1[i][j]
                    block_diagonal[n + i][n + j] = P2[i][j]
            basis.append(block_diagonal)
            weight_vector.append(0)

        # Group 3: Upper block triangular [[0, S_{j,k}], [0, 0]] (weight = 1)
        for S in symmetric_matrices:
            upper_triangular = [[0] * matrix_dim for _ in range(matrix_dim)]
            # Insert S into the upper-right block
            for i in range(n):
                for j in range(n):
                    upper_triangular[i][n + j] = S[i][j]
            basis.append(upper_triangular)
            weight_vector.append(1)

        # Return basis and weight vector wrapped in a list (to allow multiple gradings)
        return basis, [weight_vector]

    def generate_D_series_structure_data(n):
        """
        Generates the structure data and an empty weight vector for the D_n series 
        (special orthogonal Lie algebra: so(2n)).

        Parameters
        ----------
        n : int
            The rank of the D-series Lie algebra (2n is the matrix dimension).

        Returns
        -------
        tuple
            - basis (list): A 3-dimensional list representing the structure data for so(2n).
                            Each element is a 2D list of lists representing a matrix.
            - weight_vectors (list): An empty list (weights not assigned for D-series).

        Notes
        -----
        - Reuses the skew-symmetric construction logic from the B-series generator.
        - The weight vector assignment for D-series is left empty for now.
        """
        return generate_B_series_structure_data(n)

    # A-series implementation
    if series_type == "A":
        default_label = f"sl{rank + 1}" if label is None else label
        # Compute structure data for sl(n+1)
        structure_data, grading = generate_A_series_structure_data(rank)
        # Create and return the Lie algebra
        return createFiniteAlg(
            structure_data,
            default_label,
            basis_labels=basis_labels,
            grading=grading,
            process_matrix_rep=True
        )

    # B-series implementation
    elif series_type == "B":
        default_label = f"so{2*rank + 1}" if label is None else label
        # Compute structure data for so(2n+1)
        structure_data, grading = generate_B_series_structure_data(rank)
        # Create and return the Lie algebra
        return createFiniteAlg(
            structure_data,
            default_label,
            basis_labels=basis_labels,
            grading=grading,
            process_matrix_rep=True
        )

    # C-series implementation
    elif series_type == "C":
        default_label = f"sp{2*rank}" if label is None else label
        # Compute structure data for sp(2n)
        structure_data, grading = generate_C_series_structure_data(rank)
        # Create and return the Lie algebra
        return createFiniteAlg(
            structure_data,
            default_label,
            basis_labels=basis_labels,
            grading=grading,
            process_matrix_rep=True
        )

    # D-series implementation
    elif series_type == "D":
        default_label = f"so{2*rank}" if label is None else label
        # Compute structure data for so(2n)
        structure_data, grading = generate_D_series_structure_data(rank)
        # Create and return the Lie algebra
        return createFiniteAlg(
            structure_data,
            default_label,
            basis_labels=basis_labels,
            grading=grading,
            process_matrix_rep=True
        )

    # Raise an error for unrecognized series
    else:
        raise ValueError(f"Series type '{series_type}' is not recognized. Expected 'A', 'B', 'C', or 'D'.")


############## algebra tools

def killingForm(arg1, list_processing=False):
    if arg1.__class__.__name__ == "FAClass":
        # Convert the structure data to a mutable array
        if not arg1.is_lie_algebra():
            raise Exception(
                "killingForm expects argument to be a Lie algebra instance of the FAClass"
            )
        if list_processing:
            aRepLoc = arg1.structureData
            return [
                [
                    trace_matrix(multiply_matrices(aRepLoc[j], aRepLoc[k]))
                    for k in range(arg1.dimension)
                ]
                for j in range(arg1.dimension)
            ]
        else:
            aRepLoc = adjointRepresentation(arg1)
            return sp.Matrix(
                arg1.dimension,
                arg1.dimension,
                lambda j, k: (aRepLoc[j] * aRepLoc[k]).trace(),
            )
    else:
        raise Exception("killingForm expected to receive an FAClass instance.")


def adjointRepresentation(arg1, list_format=False):
    if arg1.__class__.__name__ == "FAClass":
        # Convert the structure data to a mutable array
        if not arg1.is_lie_algebra():
            warnings.warn(
                "Caution: The algebra passed to adjointRepresentation is not a Lie algebra."
            )
        if list_format:
            return arg1.structureData
        return [sp.Matrix(j) for j in arg1.structureData]
    else:
        raise Exception(
            "adjointRepresentation expected to receive an FAClass instance."
        )


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
        )

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
        )

    # Compute the trace (sum of the diagonal elements)
    trace_value = sum(A[i][i] for i in range(rows_A))

    return trace_value
