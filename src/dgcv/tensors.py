import warnings

import sympy as sp

from ._config import _cached_caller_globals, get_variable_registry
from ._safeguards import (
    create_key,
    get_dgcv_category,
    retrieve_passkey,
    validate_label,
    validate_label_list,
)
from ._tensor_field_printers import (
    tensor_latex_helper,
    tensor_VS_printer,
)
from .combinatorics import shufflings
from .vmf import clearVar, listVar


# vector space class
class vector_space_class(sp.Basic):
    def __new__(cls, dimension, *args, **kwargs):
        if not isinstance(dimension, int) or dimension < 0:
            raise TypeError("vector_space_class expected dimension to be a positive int.")

        # Create the new instance
        obj = sp.Basic.__new__(cls, dimension)
        return obj

    def __init__(
        self,
        dimension,
        grading=None,
        _label=None,
        _basis_labels=None,
        _calledFromCreator=None,
    ):
        self.dimension = dimension
        self._dgcv_class_check=retrieve_passkey()
        self._dgcv_category='vectorSpace'

        # Detect if initialized from creator (using the passkey)
        if _calledFromCreator == retrieve_passkey():
            self.label = _label
            self.basis_labels = tuple(_basis_labels) if _basis_labels else None
            self._registered = True
        else:
            self.label = "Alg_" + create_key()  # Assign a random label
            self.basis_labels = None
            self._registered = False

        def validate_and_adjust_grading_vector(vector, dimension):
            """
            Validates and adjusts a grading vector to match the vector space's dimension.

            Parameters
            ----------
            vector : list, tuple, or sympy.Tuple
                The grading vector to validate and adjust.
            dimension : int
                The dimension of the vector space.

            Returns
            -------
            sympy.Tuple
                The validated and adjusted grading vector.
            """
            if not isinstance(vector, (list, tuple, sp.Tuple)):
                raise ValueError(
                    "Grading vector must be a list, tuple, or SymPy Tuple."
                )
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
            self.grading = (tuple([0] * self.dimension),)
        else:
            # Handle single or multiple grading vectors
            if isinstance(grading, (list, tuple)) and all(
                isinstance(g, (list, tuple, sp.Tuple)) for g in grading
            ):
                # Multiple grading vectors provided
                self.grading = tuple(
                    validate_and_adjust_grading_vector(vector, self.dimension)
                    for vector in grading
                )
            else:
                # Single grading vector provided
                self.grading = (
                    validate_and_adjust_grading_vector(grading, self.dimension),
                )

        # Set the number of grading vectors
        self._gradingNumber = len(self.grading)

        # Initialize basis
        self.basis = tuple(
            vector_space_element(
                self,
                [1 if i == j else 0 for j in range(self.dimension)],
                1
            )
            for i in range(self.dimension)
        )

    def __eq__(self, other):
        if not isinstance(other, vector_space_class):
            return NotImplemented
        return (
            self.dimension == other.dimension and
            self.label == other.label and
            self.basis_labels == other.basis_labels and
            self.grading == other.grading and
            self.basis == other.basis
        )

    def __hash__(self):
        return hash((self.dimension, self.label, self.basis_labels, self.grading, self.basis))

    def __iter__(self):
        return iter(self.basis) 

    def __contains__(self, item):
        return item in self.basis

    def __repr__(self):
        """
        Provides a detailed representation of the vector_space_class object.
        Raises a warning if the instance is unregistered.
        """
        if not self._registered:
            warnings.warn(
                "This vector_space_class instance was initialized without an assigned label. "
                "It is recommended to initialize vector_space_class objects with dgcv creator functions like `createVectorSpace` instead.",
                UserWarning,
            )
        return (
            f"vector_space_class(dim={self.dimension}, grading={self.grading}, "
            f"label={self.label}, basis_labels={self.basis_labels}"
        )

    def __str__(self):
        """
        Provides a string representation of the vector_space_class object.
        Raises a warning if the instance is unregistered.
        """
        if not self._registered:
            warnings.warn(
                "This vector_space_class instance was initialized without an assigned label. "
                "It is recommended to initialize vector_space_class objects with dgcv creator functions like `createVectorSpace` instead.",
                UserWarning,
            )

        def format_basis_label(label):
            return label

        formatted_label = self.label if self.label else "Unnamed VS"
        formatted_basis_labels = (
            ", ".join([format_basis_label(bl) for bl in self.basis_labels])
            if self.basis_labels
            else "No basis labels assigned"
        )
        return (
            f"Vector Space: {formatted_label}\n"
            f"Dimension: {self.dimension}\n"
            f"Grading: {self.grading}\n"
            f"Basis: {formatted_basis_labels}"
        )

    def _display_dgcv_hook(self):
        """
        Hook for dgcv-specific display customization.
        Raises a warning if the instance is unregistered.
        """
        if not self._registered:
            warnings.warn(
                "This vector_space_class instance was initialized without an assigned label. "
                "It is recommended to initialize vector_space_class objects with dgcv creator functions like `createVectorSpace` instead.",
                UserWarning,
            )

        def format_VS_label(label):
            r"""Wrap the vector space label in \mathfrak{} if all characters are lowercase, and subscript any numeric suffix."""
            if label and label[-1].isdigit():
                label_text = "".join(filter(str.isalpha, label))
                label_number = "".join(filter(str.isdigit, label))
                if label_text.islower():
                    return rf"\mathfrak{{{label_text}}}_{{{label_number}}}"
                return rf"{label_text}_{{{label_number}}}"
            elif label and label.islower():
                return rf"\mathfrak{{{label}}}"
            return label or "Unnamed Vector Space"

        return format_VS_label(self.label)

    def _repr_latex_(self,**kwargs):
        """
        Provides a LaTeX representation of the vector_space_class object for Jupyter notebooks.
        Raises a warning if the instance is unregistered.
        """
        if not self._registered:
            warnings.warn(
                "This vector_space_class instance was initialized without an assigned label. "
                "It is recommended to initialize vector_space_class objects with dgcv creator functions like `createVectorSpace` instead.",
                UserWarning,
            )

        def format_VS_label(label):
            r"""
            Formats a vector space label for LaTeX. Handles:
            1. Labels with an underscore, splitting into two parts:
            - The first part goes into \mathfrak{} if it is lowercase.
            - The second part becomes a LaTeX subscript.
            2. Labels without an underscore:
            - Checks if the label ends in a numeric tail for subscripting.
            - Otherwise wraps the label in \mathfrak{} if it is entirely lowercase.

            Parameters
            ----------
            label : str
                The vector space label to format.

            Returns
            -------
            str
                A LaTeX-formatted vector space label.
            """
            if not label:
                return "Unnamed Vector Space"

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

        formatted_label = format_VS_label(self.label)
        formatted_basis_labels = (
            ", ".join([format_basis_label(bl) for bl in self.basis_labels])
            if self.basis_labels
            else "No basis labels assigned"
        )
        return (
            f"Vector Space: ${formatted_label}$, Basis: ${formatted_basis_labels}$, "
            f"Dimension: ${self.dimension}$, Grading: ${sp.latex(self.grading)}$"
        )

    def _sympystr(self):
        """
        SymPy string representation for vector_space_class.
        Raises a warning if the instance is unregistered.
        """
        if not self._registered:
            warnings.warn(
                "This vector_space_class instance was initialized without an assigned label. "
                "It is recommended to initialize vector_space_class objects with dgcv creator functions like `createVectorSpace` instead.",
                UserWarning,
            )

        if self.label:
            return f"vector_space_class({self.label}, dim={self.dimension})"
        else:
            return f"vector_space_class(dim={self.dimension})"

    def subspace_basis(self, elements):
        """
        Computes a basis of subspace spanned by given set of elements.

        Parameters
        ----------
        elements : list
            A list of vector_space_element instances.

        Returns
        -------
        list of vector_space_element
            basis of subspace
        """

        if not all(isinstance(j,vector_space_element) for j in elements) or not all(j.vectorSpace==self for j in elements) or len(set([j.valence for j in elements]))!=1:
            raise TypeError('vector_space_class.subspace_basis expects a list of elements from the calling vector_space_class instance.')

        # Perform linear independence check
        span_matrix = sp.Matrix.hstack(*[el.coeffs for el in elements])
        linearly_independent = span_matrix.rank() == len(elements)

        if linearly_independent:
            return elements

        rref_matrix, pivot_columns = span_matrix.rref()

        # Extract the linearly independent basis
        return [vector_space_element(self,elements[i],) for i in pivot_columns]

# vector space element class
class vector_space_element(sp.Basic):
    def __new__(cls, VS, coeffs, valence):
        if not isinstance(VS, vector_space_class):
            raise TypeError(
                "vector_space_element expects the first argument to be an instance of vector_space_class."
            )
        if valence not in {0, 1}:
            raise TypeError("vector_space_element expects third argument to be 0 or 1.")

        coeffs = tuple(coeffs)

        obj = sp.Basic.__new__(cls, VS, coeffs, valence)
        return obj

    def __init__(self, VS, coeffs, valence):
        self.vectorSpace = VS
        self.coeffs = tuple(coeffs)  # Store coeffs as a tuple instead of a Matrix
        self.valence = valence
        self._dgcv_class_check=retrieve_passkey()
        self._dgcv_category='vector_space_element'

    def __eq__(self, other):
        if not isinstance(other, vector_space_element):
            return NotImplemented
        return (
            self.vectorSpace == other.vectorSpace and
            self.coeffs == other.coeffs and
            self.valence == other.valence
        )

    def __hash__(self):
        return hash((self.vectorSpace, self.coeffs, self.valence))

    def __str__(self):
        """
        Custom string representation for vector_space_element.
        Displays the linear combination of basis elements with coefficients.
        Handles unregistered parent vector space by raising a warning.
        """
        if not self.vectorSpace._registered:
            warnings.warn(
                "This vector_space_element's parent vector space (vector_space_class) was initialized without an assigned label. "
                "It is recommended to initialize vector_space_class objects with dgcv creator functions like `createVectorSpace` instead.",
                UserWarning,
            )

        terms = []
        for coeff, basis_label in zip(
            self.coeffs,
            self.vectorSpace.basis_labels
            or [f"e_{i+1}" for i in range(self.vectorSpace.dimension)],
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
                return f"0 * {self.vectorSpace.basis_labels[0] if self.vectorSpace.basis_labels else 'e_1'}"
            else:
                return f"0 * {self.vectorSpace.basis_labels[0] if self.vectorSpace.basis_labels else 'e_1'}^\'\'"

        return " + ".join(terms).replace("+ -", "- ")

    def _repr_latex_(self,**kwargs):
        """
        Provides a LaTeX representation of vector_space_element for Jupyter notebooks.
        Handles unregistered parent vector space by raising a warning.
        """
        if not self.vectorSpace._registered:
            warnings.warn(
                "This vector_space_element's parent vector space (vector_space_class) was initialized without an assigned label. "
                "It is recommended to initialize vector_space_class objects with dgcv creator functions like `createVectorSpace` instead.",
                UserWarning,
            )

        terms = []
        for coeff, basis_label in zip(
            self.coeffs,
            self.vectorSpace.basis_labels
            or [f"e_{i+1}" for i in range(self.vectorSpace.dimension)],
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
            return rf"$0 \cdot {self.vectorSpace.basis_labels[0] if self.vectorSpace.basis_labels else 'e_1'}$"

        result = " + ".join(terms).replace("+ -", "- ")

        def format_VS_label(label):
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

        return rf"$\text{{Element of }} {format_VS_label(self.vectorSpace.label)}: {result}$"

    def _sympystr(self):
        """
        SymPy string representation for vector_space_element.
        Handles unregistered parent vector space by raising a warning.
        """
        if not self.vectorSpace._registered:
            warnings.warn(
                "This vector_space_element's parent vector space (vector_space_class) was initialized without an assigned label. "
                "It is recommended to initialize vector_space_class objects with dgcv creator functions like `createVectorSpace` instead.",
                UserWarning,
            )

        coeffs_str = ", ".join(map(str, self.coeffs))
        if self.vectorSpace.label:
            return f"vector_space_element({self.vectorSpace.label}, coeffs=[{coeffs_str}])"
        else:
            return f"vector_space_element(coeffs=[{coeffs_str}])"

    def __repr__(self):
        """
        Representation of vector_space_element.
        Shows the linear combination of basis elements with coefficients.
        Falls back to __str__ if basis_labels is None.
        """
        if self.vectorSpace.basis_labels is None:
            # Fallback to __str__ when basis_labels is None
            return str(self)

        terms = []
        for coeff, basis_label in zip(self.coeffs, self.vectorSpace.basis_labels):
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
                return f"0*{self.vectorSpace.basis_labels[0]}"
            else:
                return f"0*{self.vectorSpace.basis_labels[0]}^\'\'"

        return " + ".join(terms).replace("+ -", "- ")

    def is_zero(self):
        for j in self.coeffs:
            if sp.simplify(j) != 0:
                return False
        else:
            return True

    def dual(self):
        return vector_space_element(self.vectorSpace, self.coeffs, (self.valence+1)%2)

    def _convert_to_tp(self):
        return tensorProduct(self.vectorSpace,{(j,self.valence):self.coeffs[j] for j in range(self.vectorSpace.dimension)})

    def _recursion_contract_hom(self, other):
        return self._convert_to_tp()._recursion_contract_hom(other)

    def subs(self, subsData):
        newCoeffs = [sp.sympify(j).subs(subsData) for j in self.coeffs]
        return vector_space_element(self.vectorSpace, newCoeffs)

    def __call__(self, other):
        if not isinstance(other, vector_space_element) or other.vectorSpace!=self.vectorSpace or other.valence==self.valence:
            raise TypeError('`vector_space_element.call()` can only be applied to `vector_space_element` instances with the same vector_space_class but different valence attributes.')
        return sum([self.coeffs[j]*other.coeffs[j] for j in range(self.vectorSpace.dimension)])

    def __add__(self, other):
        if isinstance(other, vector_space_element):
            if self.vectorSpace == other.vectorSpace and self.valence == other.valence:
                return vector_space_element(
                    self.vectorSpace,
                    [self.coeffs[j] + other.coeffs[j] for j in range(len(self.coeffs))],
                    self.valence
                )
            else:
                raise TypeError(
                    "vector_space_element operands for + must belong to the same vector_space_class."
                )
        else:
            raise TypeError(
                "Unsupported operand type(s) for + with the vector_space_element class"
            )

    def __sub__(self, other):
        if isinstance(other, vector_space_element):
            if self.vectorSpace == other.vectorSpace and self.valence == other.valence:
                return vector_space_element(
                    self.vectorSpace,
                    [self.coeffs[j] - other.coeffs[j] for j in range(len(self.coeffs))],
                    self.valence
                )
            else:
                raise TypeError(
                    "vector_space_element operands for - must belong to the same vector_space_class."
                )
        else:
            raise TypeError(
                "Unsupported operand type(s) for - with the vector_space_element class"
            )

    def __mul__(self, other):
        """
        Multiplies two vector_space_element objects by multiplying their coefficients
        and summing the results based on the vector space's structure constants. Also handles
        multiplication with scalars.

        Args:
            other (vector_space_element) or (scalar): The vector space element or scalar to multiply with.

        Returns:
            vector_space_element: The result of the multiplication.
        """
        if isinstance(other, (int, float, sp.Expr)):
            # Scalar multiplication case
            new_coeffs = [coeff * other for coeff in self.coeffs]
            return vector_space_element(
                self.vectorSpace, new_coeffs, self.valence
            )
        else:
            raise TypeError(
                f"Multiplication is only supported for scalars, not {type(other)}"
            )

    def __rmul__(self, other):
        # If other is a scalar, treat it as commutative
        if isinstance(
            other, (int, float, sp.Expr)
        ):  # Handles numeric types and SymPy scalars
            return self * other  # Calls __mul__ (which is already implemented)
        else:
            raise TypeError(
                f"Right multiplication is only supported for scalars not {type(other)}"
            )

    def __matmul__(self, other):
        """Overload @ operator for tensor product."""
        if not isinstance(other, vector_space_element) or other.vectorSpace!=self.vectorSpace:
            raise TypeError('`@` only supports tensor products between vector_space_elements instances with the same vector_space_class attribute')
        return tensorProduct(self.vectorSpace, {(j,k,self.valence,other.valence):self.coeffs[j]*other.coeffs[k] for j in range(self.vectorSpace.dimension) for k in range(self.vectorSpace.dimension)})

    def __xor__(self, other):
        if other == '':
            return self.dual()
            raise ValueError("Invalid operation. Use `^ ''` to denote the dual.")

    def __neg__(self):
        return -1*self

class tensorProduct(sp.Basic):
    def __new__(cls, vector_space, coeff_dict):

        if not isinstance(coeff_dict, dict):
            raise ValueError("Coefficient dictionary must be a dictionary.")

        # Process the coefficient dictionary
        try:
            result = tensorProduct._process_coeffs_dict(coeff_dict)

            if not isinstance(result, tuple):
                raise TypeError(f"DEBUG: `tensorProduct._process_coeffs_dict` is expected to return a tuple, but produced type {type(result)}")

            if len(result) != 5:
                raise ValueError(f"DEBUG: `tensorProduct._process_coeffs_dict` is expected to return 5 values, but returned {len(result)}: {result}")

            processed_coeff_dict, max_degree, min_degree, prolongation_type, homogeneous_dicts = result

        except ValueError as ve:
            print(f"ValueError: {ve}\nDebug info: Check the return statement of _process_coeffs_dict.")

        except TypeError as te:
            print(f"TypeError: {te}\nDebug info: Make sure _process_coeffs_dict returns a tuple.")

        except Exception as e:
            print(f"Unexpected error: {e}\nDebug info: {result if 'result' in locals() else 'Function did not return any value.'}")

        obj = sp.Basic.__new__(cls, vector_space,  processed_coeff_dict)
        obj.vector_space = vector_space
        obj.coeff_dict = processed_coeff_dict
        obj.max_degree = max_degree
        obj.min_degree = min_degree
        obj.prolongation_type = prolongation_type
        obj.homogeneous_dicts = homogeneous_dicts
        obj.homogeneous = True if len(homogeneous_dicts)==1 else False
        return obj

    def __init__(self, vector_space, coeff_dict):
        self._weights = None
        self._leading_valence = None
        self._trailing_valence = None
        self._dgcv_class_check=retrieve_passkey()
        self._dgcv_category='tensorProduct'

    @staticmethod
    def _process_coeffs_dict(coeff_dict):
        """Process the coefficient dictionary."""
        if not coeff_dict:
            return {(0,) * 0: 0}, 0, 0, 0,[{(0,) * 0: 0}]

        max_degree = 0
        min_degree = -1
        valence = tuple()
        prolongation_type = None
        processed_dict = dict()
        homogeneous_dicts = dict()
        for key, value in coeff_dict.items():
            if not isinstance(key, tuple) or len(key)%2!=0 or not all(j in {0,1} for j in key[len(key)//2:]) or not all(isinstance(j,(int,float,sp.Expr)) for j in key[:len(key)//2]):
                raise ValueError("Keys in coeff_dict must be tuples of even length whose last half contain only 0s and 1s (indicating valence of the first half).")
            if 2*max_degree<len(key):
                max_degree = len(key)//2
                valence = key[len(key)//2:]
            if len(key)//2 in homogeneous_dicts:
                homogeneous_dicts[len(key)//2][key]=value
            else:
                homogeneous_dicts[len(key)//2]={key:value}
            if value != 0:
                processed_dict[key] = value
                if min_degree<0 or 2*min_degree>len(key):
                    min_degree = len(key)//2
                if prolongation_type!=-1 and all((key[len(key)//2]+j)%2==1 for j in key[len(key)//2+1:]):
                    if prolongation_type is None:
                        prolongation_type = key[len(key)//2]
                    elif prolongation_type!=key[len(key)//2]:
                        prolongation_type=-1
                else:
                    prolongation_type=-1
            if min_degree<0:
                min_degree=max_degree
        if prolongation_type is None:
            prolongation_type=-1

        # If all keys are removed (tensor is zero), restore a zero key
        if not processed_dict:
            processed_dict[((0,) * max_degree)+valence] = 0
        return processed_dict, max_degree, min_degree, prolongation_type, homogeneous_dicts

    @property
    def leading_valence(self):
        if self._leading_valence:
            return self._leading_valence
        lv = set()
        for k in self.coeff_dict:
            if len(k)>1:
                lv = lv|{k[len(k)//2]}
        if len(lv)==1:
            self._leading_valence = list(lv)[0]
        else:
            self._leading_valence = -1      # denoting exceptional cases
        return self._leading_valence

    @property
    def trailing_valence(self):
        if self._trailing_valence:
            return self._trailing_valence
        lv = set()
        for k in self.coeff_dict:
            if len(k)>0:
                lv = lv|{k[-1]}
        if len(lv)==1:
            self._trailing_valence = list(lv)[0]
        else:
            self._trailing_valence = -1     # denoting exceptional cases
        return self._trailing_valence

    @property
    def homogeneous_components(self):
        return [tensorProduct(self.vector_space,cd) for cd in self.homogeneous_dicts.values()]

    @property
    def free_vectors(self):
        vec_idx = set()
        for idx_t in self.coeff_dict:
            vec_idx = vec_idx | set(idx_t[:len(idx_t//2)])
        return set([self.vector_space.basis[idx] for idx in vec_idx])

    def _compute_weights(self):
        """Lazily compute weights for all terms in coeff_dict for each weight vector."""
        if not hasattr(self, "_weights"):
            self._weights = {}

        if not self._weights:  # Only compute if weights have not been cached
            self._weights = {}
            grading = self.vector_space.grading  # List of weight vectors

            for key, _ in self.coeff_dict.items():  # algo requires valence 1 for vec and 0 for covec
                weight_list = []
                for weight_vector in grading:
                    weight = 0
                    for j, index in enumerate(key[:len(key)//2]):
                        weight += weight_vector[index] * (key[(len(key)//2)+j] * 2 - 1)
                    weight_list.append(weight)
                self._weights[key] = tuple(weight_list)

        return self._weights

    def compute_weight(self, _return_mixed_weight_list = False):
        weights = list(set((self._compute_weights()).values()))
        if _return_mixed_weight_list is True:
            return weights
        if len(weights)==1:
            return weights[0]
        else:
            return "NoW"

    def get_weighted_components(self, weight_list):
        """
        Return a new tensorProduct with components matching the given weight_list.

        Parameters:
        - weight_list: A list or tuple of weights to match against.

        Returns:
        - A new tensorProduct with a filtered coeff_dict.
        """
        self._compute_weights()

        # Filter components whose weights match the given weight_list
        filtered_coeff_dict = {
            key: value for key, value in self.coeff_dict.items() if self._weights[key] == tuple(weight_list)
        }

        return tensorProduct(self.vector_space, filtered_coeff_dict)

    def tp(self, other):
        """Tensor product with another tensorProduct."""
        if get_dgcv_category(other)=='subalgebra_element' and other.algebra.ambiant == self.vector_space:
            other = other._convert_to_tp()
        elif hasattr(other,"algebra") and other.algebra == self.vector_space:
            other = other._convert_to_tp()
        elif hasattr(other,"vectorSpace") and other.vectorSpace == self.vector_space:
            other = other._convert_to_tp()
        if not isinstance(other, tensorProduct):
            raise ValueError("The other object must be a tensorProduct instance.")
        if self.vector_space != other.vector_space:
            raise ValueError(f"Both tensors must have the same vector_space. Recieved objects {self} and {other} with vector_spaces: {self.vector_space} and {other.vector_space}.")

        # Compute new coefficient dictionary
        new_coeff_dict = {}
        for (key1, value1) in self.coeff_dict.items():
            for (key2, value2) in other.coeff_dict.items():
                new_key = key1[:len(key1)//2] + key2[:len(key2)//2] + key1[len(key1)//2:] + key2[len(key2)//2:]
                new_coeff_dict[new_key] = value1 * value2

        return tensorProduct(self.vector_space, new_coeff_dict)

    def is_zero(self):
        for v in self.coeff_dict.values():
            if sp.simplify(v) != 0:
                return False
        else:
            return True

    def __str__(self):
        return tensor_VS_printer(self)

    def __repr__(self):
        return tensor_VS_printer(self)

    def _latex(self, printer=None,**kwargs):
        """
        Defines the LaTeX representation for SymPy's latex() function.
        """
        return f'$\\displaystyle {tensor_latex_helper(self)}$'

    def _repr_latex_(self,**kwargs):
        return self._latex()

    def _sympystr(self, printer):
        return self.__repr__()

    def subs(self, subs_data):
        new_dict = {j:sp.sympify(k).subs(subs_data) for j,k in self.coeff_dict.items()}
        return tensorProduct(self.vector_space,new_dict)

    def simplify(self):
        new_dict = {j:sp.simplify(k) for j,k in self.coeff_dict.items()}
        return tensorProduct(self.vector_space,new_dict)

    def _eval_simplify(self, ratio=1.7, measure=None, rational=True, **kwargs):
        """
        Custom simplification method for sympy.simplify.
        """
        # Apply sympy.simplify to each value in coeff_dict
        new_dict = {key: sp.simplify(value) for key, value in self.coeff_dict.items()}
        # Return a new instance of the class (using tensorProduct or another constructor)
        return tensorProduct(self.vector_space, new_dict)

    def __add__(self,other):
        if get_dgcv_category(other)=='subalgebra_element':
            if other.algebra == self.vector_space:
                other = other._convert_to_tp()
            elif other.algebra.ambiant == self.vector_space:
                other = other.ambiant_rep._convert_to_tp()
            ###!!!! add logic branch for when self.vector_space is a subalgebra
        if get_dgcv_category(other)=='algebra_element' and other.algebra == self.vector_space:
            other = other._convert_to_tp()
        elif hasattr(other,"vectorSpace") and other.vectorSpace == self.vector_space:
            other = other._convert_to_tp()
        if not isinstance(other,tensorProduct) or (isinstance(other,tensorProduct) and other.vector_space != self.vector_space):
            raise TypeError('`+` requires other tensor to be formed from the same vector space.')
        new_dict = dict(self.coeff_dict)
        for key, val in other.coeff_dict.items():
            if key in new_dict:
                new_dict[key] = new_dict[key]+val
            else:
                new_dict[key] = val
        return tensorProduct(self.vector_space,new_dict)

    def __sub__(self,other):
        if get_dgcv_category(other)=='subalgebra_element':
            if other.algebra == self.vector_space:
                other = other._convert_to_tp()
            elif other.algebra.ambiant == self.vector_space:
                other = other.ambiant_rep._convert_to_tp()
        if get_dgcv_category(other)=='algebra_element' and other.algebra == self.vector_space:
            other = other._convert_to_tp()
        elif hasattr(other,"vectorSpace") and other.vectorSpace == self.vector_space:
            other = other._convert_to_tp()
        if not isinstance(other,tensorProduct) or (isinstance(other,tensorProduct) and other.vector_space != self.vector_space):
            raise TypeError('`+` requires other tensor to be formed from the same vector space.')
        new_dict = dict(self.coeff_dict)
        for key, val in other.coeff_dict.items():
            if key in new_dict:
                new_dict[key] = new_dict[key]-val
            else:
                new_dict[key] = -val
        return tensorProduct(self.vector_space,new_dict)

    def __truediv__(self, other):
        if isinstance(other,int):
            return sp.Rational(1,other)*self
        if isinstance(other,(float,sp.Expr)):
            return (1/other)*self

    def __matmul__(self, other):
        """Overload @ operator for tensor product."""
        return self.tp(other)

    def contract_call(self, other):
        """
        Contract the last index of self with the first index of other or handle algebra_element.
        """
        if self.is_zero():
            return self
        if other.is_zero():
            return 0*self
        if isinstance(other, tensorProduct):
            if self.vector_space != other.vector_space:
                raise ValueError("Both tensors must be defined w.r.t. the same vector_space.")

            if self.trailing_valence + other.leading_valence != 1:
                raise ValueError("Contraction requires the first tensor factor of every term in other to have leading valence different from the last entry tensor factor from terms of self.")

            new_dict = {}
            for key1, value1 in self.coeff_dict.items():
                for key2, value2 in other.coeff_dict.items():
                    if key1[(len(key1)//2)-1] == key2[0]:  # Matching indices for contraction
                        new_key = key1[:(len(key1)//2)-1] + key2[1:(len(key2)//2)] + key1[(len(key1)//2):-1] + key2[1+(len(key2)//2):] # Remove the contracted indices
                        new_value = value1 * value2
                        new_dict[new_key] = new_dict.get(new_key, 0) + new_value  # Accumulate values for duplicate keys

            return tensorProduct(self.vector_space, new_dict)

        if get_dgcv_category(other)=='subalgebra_element' and other.algebra != self.vector_space:
            other = other.ambiant_rep

        elif hasattr(other, "algebra") and self.vector_space == other.algebra:
            if self.trailing_valence != 0:
                raise ValueError(f"Operating on algebra_element requires all terms in self to end in covariant tensor factor. Recieved self: {self} and other: {other}")
            other_as_tensor = other._convert_to_tp()
            return self.contract_call(other_as_tensor)
        else:
            raise ValueError("The other object must be a tensorProduct or an algebra_element with matching algebra.")

    def _recursion_contract_hom(self,other):
        if self.is_zero():
            return self
        if self.prolongation_type == 0:
            vs = tuple([j.dual() for j in self.vector_space])
            vsDual = self.vector_space.basis
        elif self.prolongation_type == 1:
            vs = self.vector_space.basis
            vsDual = tuple([j.dual() for j in self.vector_space])
        else:
            raise TypeError(f'`_recursion_contract_hom` does not operate on arguments with mixed `prolongation_type` e.g., {self} has type {self.prolongation_type}')
        if self.max_degree==1 or other.max_degree==1:
            return self*other
        otherContract = other*vs[0]
        if hasattr(otherContract,'_convert_to_tp'):
            otherContract=otherContract._convert_to_tp()
        image_part = ((self*vs[0])._recursion_contract_hom(other)+self._recursion_contract_hom(otherContract))
        domain_part = vsDual[0]
        contraction = image_part@domain_part
        for vec,vecD in  zip(vs[1:],vsDual[1:]):
            otherContract = other*vec
            if hasattr(otherContract,'_convert_to_tp'):
                otherContract=otherContract._convert_to_tp()
            contraction += ((self*vec)._recursion_contract_hom(other)+self._recursion_contract_hom(otherContract))@vecD
        return contraction

    def _recursion_contract(self,other):
        if self.is_zero():
            return self
        if other.is_zero():
            return 0*self
        if isinstance(other, tensorProduct):
            if self.prolongation_type != other.prolongation_type:
                raise type(f'`tensorProduct` contraction is only supported between instances with matching `prolongation types`, not types: {self.prolongation_type} and {other.prolongation_type}')
            if self.vector_space != other.vector_space:
                return 0*self
            hc1 = self.homogeneous_components
            hc2 = other.homogeneous_components
            terms = [t1._recursion_contract_hom(t2) for t1,t2 in zip(hc1,hc2)]
            return sum(terms[1:],terms[0])

    def _bracket(self,other):
        if self.is_zero():
            return self
        if other.is_zero():
            return 0*self
        if get_dgcv_category(other)=='subalgebra_element' and other.algebra != self.vector_space:
            other = other.ambiant_rep
        if isinstance(other, tensorProduct):
            if self.vector_space != other.vector_space:
                raise ValueError("In `tensorProduct._bracket` both tensors must be defined w.r.t. the same vector_space.")

            if self.prolongation_type!=other.prolongation_type or self.prolongation_type==-1:
                raise ValueError("`tensorProduct._bracket` requires bracket components to have matching prolongation types.")

            complimentType = 1 if self.prolongation_type==0 else 0

            new_dict = {}
            for key1, value1 in self.coeff_dict.items():
                degree1 = len(key1) // 2
                for key2, value2 in other.coeff_dict.items():
                    degree2 = len(key2) // 2
                    for idx in range(1, degree1 - 1):
                        if key1[idx] == key2[0] and len(key2) > 3:   # Check index matching before contraction
                            new_value = value1 * value2
                            k1_start = key1[:idx]
                            k2_start = key2[1:2]
                            k1_tail_inputs = key1[idx+1:degree1]
                            k2_inputs = key2[2:degree2]
                            new_tails = shufflings(k1_tail_inputs, k2_inputs)
                            valence = (self.prolongation_type,) + (complimentType,)*(degree1+degree2-3)
                            new_keys = [tuple(k1_start + k2_start + tuple(tail) + valence) for tail in new_tails]
                            for key in new_keys:
                                new_dict[key] = new_dict.get(key, 0) + new_value  # Accumulate values for duplicate keys
                    if key1[degree1-1] == key2[0]:   # Check index matching before contraction
                        new_value = value1 * value2
                        k1_start = key1[:degree1-1]
                        k2_inputs = key2[1:degree2]
                        valence = (self.prolongation_type,) + (complimentType,)*(degree1+degree2-3)
                        new_key = tuple(k1_start + k2_inputs + valence)
                        new_dict[new_key] = new_dict.get(new_key, 0) + new_value

                    for idx in range(1, degree2 - 1):
                        if key2[idx] == key1[0] and len(key1) > 3:   # Check index matching before contraction
                            new_value = -value1 * value2
                            k2_start = key2[:idx]
                            k1_start = key1[1:2]
                            k2_tail_inputs = key2[idx+1:degree2]
                            k1_inputs = key1[2:degree1]
                            new_tails = shufflings(k2_tail_inputs, k1_inputs)
                            valence = (self.prolongation_type,) + (complimentType,)*(degree1+degree2-3)
                            new_keys = [tuple(k2_start + k1_start + tuple(tail) + valence) for tail in new_tails]
                            for key in new_keys:
                                new_dict[key] = new_dict.get(key, 0) + new_value  # Accumulate values for duplicate keys
                    if key2[degree2-1] == key1[0]:   # Check index matching before contraction
                        new_value = -value1 * value2
                        k2_start = key2[:degree2-1]
                        k1_inputs = key1[1:degree1]
                        valence = (self.prolongation_type,) + (complimentType,)*(degree2+degree1-3)
                        new_key = tuple(k2_start + k1_inputs + valence)
                        new_dict[new_key] = new_dict.get(new_key, 0) + new_value
            return tensorProduct(self.vector_space, new_dict)

        elif hasattr(other, "algebra") and self.vector_space == other.algebra:
            if self.vector_space != other.algebra:
                raise ValueError("In `tensorProduct._bracket` both tensors must be defined w.r.t. the same vector_space.")
            if self.prolongation_type != other.valence:
                raise ValueError("`tensorProduct._bracket` operating on algebra_element requires all terms in self to end in covariant tensor factor.")
            other_as_tensor = other._convert_to_tp()
            other_index,other_value = list(other_as_tensor.coeff_dict.items())[0]
            new_dict = {}
            for key, value in self.coeff_dict.items():
                if key[(len(key)//2)-1] == other_index:  # Matching indices for contraction
                    new_value = value * other_value
                    key_truncated = tuple(key[:(len(key)//2)-1]+key[(len(key)//2):-1])
                    new_dict[key_truncated] = new_dict.get(key_truncated, 0) + new_value
            return tensorProduct(self.vector_space, new_dict)
        else:
            raise ValueError("In `tensorProduct._bracket` the second factor must be a tensorProduct or an algebra_element with matching algebra.")

    def __mul__(self, other):
        """Overload * to compute the Lie bracket, with special logic for algebra_element."""
        if self.max_degree==0:
            coef=self.coeff_dict[tuple()]
            if coef!=0:      # max_degree loses relevance when coef is zero 
                return coef*other
        if isinstance(other, (int, float, sp.Expr)):
            new_coeff_dict = {key: value * other for key, value in self.coeff_dict.items()}
            return tensorProduct(self.vector_space, new_coeff_dict)

        if (hasattr(other,'is_zero') and other.is_zero()) or self.is_zero():
            new_coeff_dict = {key:0 for key in self.coeff_dict.keys()}
            return tensorProduct(self.vector_space, new_coeff_dict)

        if get_dgcv_category(other) in {'subalgebra_element', 'vectorSpace', 'algebra_element'}:
            other = other._convert_to_tp()

        if isinstance(other, tensorProduct):
            # Lie bracket for two tensorProducts
            if self.vector_space != other.vector_space:
                return 0*self
            if other.max_degree==0:
                return other.coeff_dict[tuple()]*self

            if other.max_degree==1 and other.min_degree==1:
                if self.max_degree==1 and self.min_degree==1:
                    if self.prolongation_type == other.prolongation_type:
                        if isinstance(self.vector_space,vector_space_class):
                            return 0*self
                        pt = self.prolongation_type
                        coeffs1 = [self.coeff_dict.get((j, pt), 0) for j in range(self.vector_space.dimension)]
                        coeffs2 = [other.coeff_dict.get((j, pt), 0) for j in range(self.vector_space.dimension)]
                        LA_elem1 = self.vector_space._class_builder(coeffs1,pt)
                        LA_elem2 = self.vector_space._class_builder(coeffs2,pt)
                        return (LA_elem1*LA_elem2)._convert_to_tp()
                    else:
                        ###!!! mixed prolongation type needs an additional logic branch
                        pt1 = self.prolongation_type
                        pt2 = other.prolongation_type
                        warnings.warn('DEBUG NOTE: incomplete logic branch triggered...')
                        return sum([self.coeff_dict[(j,pt1)]*other.coeff_dict[(j,pt2)] for j in range(self.vector_space.dimension)])
                else:
                    if self.trailing_valence != other.prolongation_type and self.trailing_valence!=-1:
                        cd = {}
                        for t,tv in other.coeff_dict.items():
                            for key,val in {tuple(k[:(len(k)//2)-1]+k[(len(k)//2):-1]):v*tv for k,v in self.coeff_dict.items() if k[(len(k)//2)-1]==t[0]}.items():
                                cd[key] = cd.get(key,0)+val
                        return tensorProduct(self.vector_space,cd)
                    else:
                        raise TypeError(f'* cannot operate on `tensorProduct` pairs if: \n second arg is degree 1 \n first argument has degree>1 \n first\'s `trailing_valence` doesn\'t match the second\'s `prolongation_type`.\n `trailing_valence` of {self}: {self.trailing_valence} \n `prolongation_type` of {other}: {other.prolongation_type}')
            if self.max_degree==1 and self.min_degree==1:
                return -1 * other * self
            return self._bracket(other)
        else:
            raise ValueError(f"Unsupported operation for * between the given object types: {type(self)} and {type(other)}")

    def __rmul__(self, other):
        if isinstance(other, (int, float, sp.Expr)):
            return self.__mul__(other)
        if get_dgcv_category(other)=='subalgebra_element' and other.algebra != self.vector_space:
            other = other.ambiant_rep
        if hasattr(other, "algebra") and other.algebra == self.vector_space:
            return -self.__mul__(other)
        else:
            return NotImplemented

    def __neg__(self):
        return -1*self


################ creator functions
def createVectorSpace(
    obj,
    label,
    basis_labels=None,
    grading=None,
    verbose=False
):
    """
    Registers a vector space object and its basis elements in the caller's global namespace,
    and adds them to the variable_registry for tracking in the Variable Management Framework.

    Parameters
    ----------
    obj : int, vector_space_class, list of vector_space_elements
        vector space dimension
    label : str
        The label used to reference the VS object in the global namespace.
    basis_labels : list, optional
        A list of custom labels for the basis elements of the VS.
        If not provided, default labels will be generated.
    grading : list of lists or list, optional
        A list specifying the grading(s) of the VS.
    verbose : bool, optional
        If True, provides detailed feedback during the creation process.
    """

    if label in listVar(algebras_only=True):
        warnings.warn('`createFiniteAlg` was called with a `label` already assigned to another algebra, so `createFiniteAlg` will overwrite the other algebra.')
        clearVar(label)

    # Validate or create the vector_space_class object
    if isinstance(obj, vector_space_class):
        if verbose:
            print(f"Using existing vector_space_class instance: {label}")
        dimension = obj.dimension
    elif isinstance(obj, list) and all(isinstance(el, vector_space_element) for el in obj):
        if verbose:
            print("Creating VS from list of vector_space_element instances.")
        dimension = len(obj)
    elif isinstance(obj,int) and 0<=obj:
        dimension = obj

    # Create or validate basis labels
    if basis_labels is None:
        basis_labels = [validate_label(f"{label}_{i+1}") for i in range(dimension)]
    validate_label_list(basis_labels)

    # Process grading
    if grading is None:
        grading = [(0,) * dimension]  # Default grading: all zeros
    elif isinstance(grading, (list, tuple)) and all(
        isinstance(w, (int, sp.Expr)) for w in grading
    ):
        # Single grading vector
        if len(grading) != dimension:
            raise ValueError(
                f"Grading vector length ({len(grading)}) must match the VS dimension ({dimension})."
            )
        grading = [tuple(grading)]  # Wrap single vector in a list
    elif isinstance(grading, list) and all(
        isinstance(vec, (list, tuple)) for vec in grading
    ):
        # List of grading vectors
        for vec in grading:
            if len(vec) != dimension:
                raise ValueError(
                    f"Grading vector length ({len(vec)}) must match the VS dimension ({dimension})."
                )
        grading = [tuple(vec) for vec in grading]  # Convert each vector to Tuple
    else:
        raise ValueError("Grading must be a single vector or a list of vectors.")

    # Create the vector_space_class object
    passkey = retrieve_passkey()

    vs_obj = vector_space_class(
        dimension,
        grading=grading,
        _label=label,
        _basis_labels=basis_labels,
        _calledFromCreator=passkey,
    )

    # initialize vector space and its basis
    assert (
        vs_obj.basis is not None
    ), "VS object basis elements must be initialized."

    # Register in _cached_caller_globals
    _cached_caller_globals.update({label: vs_obj})
    _cached_caller_globals.update(zip(basis_labels, vs_obj.basis))

    # Register in the variable registry
    variable_registry = get_variable_registry()
    variable_registry["finite_algebra_systems"][label] = {
        "family_type": "algebra",
        "family_names": tuple(basis_labels),
        "family_values": tuple(vs_obj.basis),
        "dimension": dimension,
        "grading": grading,
        "basis_labels": basis_labels,
        "structure_data": {(0,0,0):0},
    }

    if verbose:
        print(f"Vector Space '{label}' registered successfully.")
        print(
            f"Dimension: {dimension}, Grading: {grading}, Basis Labels: {basis_labels}"
        )

    return vs_obj





