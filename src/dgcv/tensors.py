import numbers
import warnings
from collections import Counter
from collections.abc import MutableMapping
from typing import Any, Iterable, Iterator, Mapping, Optional, Tuple, Union

import sympy as sp

from ._config import (
    _cached_caller_globals,
    _vsr_inh_idx,
    from_vsr,
    get_variable_registry,
    get_vs_registry,
)
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
from .backends._caches import _get_expr_num_types
from .combinatorics import permSign, shufflings
from .solvers import simplify_dgcv
from .vmf import clearVar, listVar


class vector_space_class:
    def __init__(
        self,
        dimension,
        grading=None,
        _label=None,
        _basis_labels=None,
        _calledFromCreator=None,
    ):
        if not isinstance(dimension, int) or dimension < 0:
            raise TypeError("vector_space_class expected dimension to be a positive int.")
        self.dimension = dimension
        self._dgcv_class_check=retrieve_passkey()
        self._dgcv_category='vectorSpace'

        if _calledFromCreator == retrieve_passkey():
            self.label = _label
            self.basis_labels = tuple(_basis_labels) if _basis_labels else None
            self._registered = True
        else:
            self.label = "Alg_" + create_key()  # Assign a random label
            self.basis_labels = None
            self._registered = False

        def validate_and_adjust_grading_vector(vector, dimension):
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

            for i, component in enumerate(vector):
                if not isinstance(component, (int, float, sp.Basic)):
                    raise ValueError(
                        f"Invalid component in grading vector at index {i}: {component}. "
                        f"Expected int, float, or sympy.Expr."
                    )

            return sp.Tuple(*vector)

        if grading is None:
            self.grading = (tuple([0] * self.dimension),)
        else:
            if isinstance(grading, (list, tuple)) and all(
                isinstance(g, (list, tuple, sp.Tuple)) for g in grading
            ):
                # Multiple grading vectors provided
                self.grading = tuple(
                    validate_and_adjust_grading_vector(vector, self.dimension)
                    for vector in grading
                )
            else:
                self.grading = (
                    validate_and_adjust_grading_vector(grading, self.dimension),
                )

        self._gradingNumber = len(self.grading)

        self.basis = tuple(
            vector_space_element(
                self,
                [1 if i == j else 0 for j in range(self.dimension)],
                1
            )
            for i in range(self.dimension)
        )
        vsr=get_vs_registry()
        self.dgcv_vs_id = len(vsr)
        vsr.append(self)

    @property
    def zero_element(self):
        return vector_space_element(self,(0,)*self.dimension,1)
    @property
    def ambient(self):
        return self

    def update_grading(self,new_weight_vectors_list,replace_instead_of_add=False):
        if isinstance(new_weight_vectors_list,(list,tuple)):
            if all(isinstance(elem,(list,tuple)) for elem in new_weight_vectors_list):
                   if replace_instead_of_add is True:
                       self.grading = [tuple(elem) for elem in new_weight_vectors_list]
                   else:
                       grad=list(self.grading)+[tuple(elem) for elem in new_weight_vectors_list]
                       self.grading=grad
            else:
                raise TypeError(f'update_grading expects first parameter to be a list of lists. The inner lists should have length {self.dimension}')
        else:
            raise TypeError(f'update_grading expects first parameter to be a list of lists. The inner lists should have length {self.dimension}')


    def __eq__(self, other):
        if not isinstance(other, vector_space_class):
            return NotImplemented
        return (self.dgcv_vs_id == other.dgcv_vs_id)

    def __hash__(self):
        return hash(self.dgcv_vs_id)

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

    def check_element_weight(self, element, test_weights = None, flatten_weights=False):
        """
        Determines the weight vector of an vector_space_element with respect to the grading vectors. Weight can be instead computed against another grading vector passed a list of weights as the keyword `test_weights`.

        Parameters
        ----------
        element : vector_space_element
            The vector_space_element to analyze.
        test_weights : list of int or sympy.Expr, optional (default: None)
        flatten_weights : (default: False) If True, returns contents of a list if otherwise would have returned a length 1 list

        Returns
        -------
        list or weight value
            A list of weights corresponding to the grading vectors of this vector_space_class (or test_weights if provided).
            Each entry is either an integer, variable expression representing weight value, the string 'AllW' (i.e., All Weights) if the element is the zero element,
            or 'NoW' (i.e., No Weights) if the element is not homogeneous.
            If the list is length 1 and flatten_weights=True then only the contents of the list is returned.

        Notes
        -----
        - 'AllW' (meaning, All Weights) is returned for zero elements, which are compatible with all weights.
        - 'NoW' (meaning, No Weights) is returned for non-homogeneous elements that do not satisfy the grading constraints.
        """
        if not get_dgcv_category(element)=='vector_space_element'  or element.vectorSpace!=self:
            raise TypeError("Input in `vector_space_class.check_element_weight` must be a `vector_space_element` instance belonging to the `vector_space_class` instance whose `check_element_weight` is being called.") from None
        if not test_weights:
            if not hasattr(self, "grading") or self._gradingNumber == 0:
                raise ValueError("This vector_space_class instance has no assigned grading vectors.") from None
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
        if flatten_weights and len(weights)==1:
            return weights[0]
        return weights

class vector_space_element:
    def __init__(self, VS, coeffs, valence):
        if not isinstance(VS, vector_space_class):
            raise TypeError(
                "vector_space_element expects the first argument to be an instance of vector_space_class."
            )
        if valence not in {0, 1}:
            raise TypeError("vector_space_element expects third argument to be 0 or 1.")
        coeffs = tuple(coeffs)
        self.vectorSpace = VS
        self.coeffs = tuple(coeffs)
        self.valence = valence
        self._dgcv_class_check=retrieve_passkey()
        self._dgcv_category='vector_space_element'
        self.dgcv_vs_id=self.vectorSpace.dgcv_vs_id

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

    def _repr_latex_(self,raw=False,**kwargs):
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

        return result if raw else rf"$\text{{Element of }} {format_VS_label(self.vectorSpace.label)}: {result}$"

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

    def contains(self, items, return_basis_coeffs=False,strict_types=True):
        if isinstance(items, (list, tuple)):
            return [self.contains(item,return_basis_coeffs=return_basis_coeffs) for item in items]
        if strict_types is False and items==0:
            if return_basis_coeffs is True:
                return [0]*self.dimension
            return True

        if strict_types is False and get_dgcv_category(items)=='tensorProduct':
            if next(iter(items._vs_spring))==self.dgcv_vs_id and len(items.vector_spaces)==1:
                k,v=next(iter(items.coeff_dict.items()))
                if len(k)==3:
                    ne = v*(from_vsr(k[2]).basis[k[0]])
                    if k[1]==0:
                        ne=ne.dual()
                    return self.contains(ne)
        if get_dgcv_category(items)=='vector_space_element' and items.dgcv_vs_id==self.dgcv_vs_id:
            if return_basis_coeffs:
                return list(items.coeffs)
            else:
                return True
        return False

    def dual(self):
        return vector_space_element(self.vectorSpace, self.coeffs, (self.valence+1)%2)

    def _convert_to_tp(self):
        return tensorProduct([self.dgcv_vs_id],{(j,self.valence,self.dgcv_vs_id):self.coeffs[j] for j in range(self.vectorSpace.dimension)})

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
            new_coeffs = [coeff * other for coeff in self.coeffs]
            return vector_space_element(
                self.vectorSpace, new_coeffs, self.valence
            )
        else:
            raise TypeError(
                f"Multiplication is only supported for scalars, not {type(other)}"
            )

    def __rmul__(self, other):
        if isinstance(
            other, (int, float, sp.Expr)
        ):  # Handles numeric types and SymPy scalars
            return self * other
        else:
            raise TypeError(
                f"Right multiplication is only supported for scalars not {type(other)}"
            )

    def __matmul__(self, other):
        """Overload @ operator for tensor product."""
        if isinstance(other,_get_expr_num_types()):
            return self.__matmul__(tensorProduct('_',{tuple():other}))
        if get_dgcv_category(other) not in {'vector_space_element','algebra_element','subalgebra_element'}:
            raise TypeError('`@` only supports tensor products between vector_space_elements instances with the same vector_space_class attribute')
        return tensorProduct('_',{(j,k,self.valence,other.valence,self.dgcv_vs_id,other.dgcv_vs_id):self.coeffs[j]*other.coeffs[k] for j in range(self.vectorSpace.dimension) for k in range(other.vectorSpace.dimension)})

    def __rmatmul__(self, other):
        """Overload @ operator for tensor product."""
        return self._convert_to_tp().__rmatmul__(other)

    def __xor__(self, other):
        if other == '':
            return self.dual()
            raise ValueError("Invalid operation. Use `^ ''` to denote the dual.")

    def __neg__(self):
        return -1*self

    def check_element_weight(self, test_weights=None, flatten_weights=False):
        """
        Determines the weight vector of this vector_space_element with respect to its vector_space's grading vectors.

        Returns
        -------
        list
            A list of weights corresponding to the grading vectors of the parent vector_space.
            Each entry is either an integer, variable representing a weight, the string 'AllW' if the element is the zero element,
            or 'NoW' if the element is not homogeneous.

        Notes
        -----
        - This method calls the parent vector_space's check_element_weight method.
        - 'AllW' is returned for zero elements, which are compaible with all weights.
        - 'NoW' is returned for non-homogeneous elements that do not satisfy the grading constraints.
        """

        return self.vectorSpace.check_element_weight(self, test_weights=test_weights, flatten_weights=flatten_weights)

    def terms(self):
        if self._terms is None:
            terms=[]
            for idx,c in enumerate(self.coeffs):
                if c==0:
                    continue
                terms.append(c*self.vectorSpace.basis[idx])
            self._terms=[self] if len(terms)<2 else terms
        return self._terms

class tensorProduct:
    def __init__(self, depricated_placeholder_param, coeff_dict,_amb_prom=False):

        if not isinstance(coeff_dict, dict):
            raise ValueError("Coefficient dictionary must be a dictionary.")

        try:
            result = tensorProduct._process_coeffs_dict(coeff_dict,_amb_prom=_amb_prom)
        except ValueError as ve:
            print(f"ValueError: {ve}\nDebug info: Check the return statement of _process_coeffs_dict.")
            raise
        except TypeError as te:
            print(f"TypeError: {te}\nDebug info: Make sure _process_coeffs_dict returns a tuple.")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}\nDebug info: "
                f"{result if 'result' in locals() else 'Function did not return any value.'}")
            raise
        processed_coeff_dict, max_degree, min_degree, prolongation_type, homogeneous_dicts, vector_spaces, spring, replacements = result
        if len(vector_spaces)==0 and len(processed_coeff_dict)>0:
            for pcKey in processed_coeff_dict:
                if len(pcKey)>0:
                    vector_spaces = list(set(pcKey[2*(len(pcKey)//3):]))
                    break
        self.vector_spaces = vector_spaces
        self.vector_space = from_vsr(vector_spaces[0]) if len(vector_spaces)>0 else from_vsr(0) ### depricate soon
        self._vs_spring = spring.keys()
        self._unpromoted_spring = {}
        if replacements:
            for k,v in spring.items():
                for idx in v[1]:
                    self._unpromoted_spring|={idx:k}
        self.coeff_dict = processed_coeff_dict
        self.max_degree = max_degree
        self.min_degree = min_degree
        self.prolongation_type = prolongation_type
        self.homogeneous_dicts = homogeneous_dicts
        self.homogeneous = True if len(homogeneous_dicts)==1 else False    
        self._weights = None
        self._leading_valence = None
        self._trailing_valence = None
        self._dgcv_class_check=retrieve_passkey()
        self._dgcv_category='tensorProduct'
        self._terms = [self] if len(self.coeff_dict)==1 else None

    @staticmethod
    def _process_coeffs_dict(coeff_dict,_amb_prom):
        """Process the coefficient dictionary."""
        if not coeff_dict:
            return {tuple(): 0}, 0, 0, 0,[{tuple(): 0}],[0], {0:[]},{}
        vector_spaces=[]
        spring=dict()
        seen_vs=[]
        max_degree = 0
        min_degree = -1
        prolongation_type = None ###!!! depricating soon
        processed_dict = dict()
        homogeneous_dicts = dict()
        for key, value in coeff_dict.items():
            if not isinstance(key,tuple):
                raise ValueError(f"Keys in coeff_dict must be tuples of length divisible by 3 whose middle third contains only 0s and 1s (indicating valence of the first third). Reicieved keys: {list(coeff_dict.keys())}")
            kl=len(key)
            deg=kl//3
            if kl%3!=0 or not all(j==0 or j==1 for j in key[deg:2*deg]) or not all(isinstance(j,numbers.Integral) for j in key[2*deg:]) or not all(isinstance(j,_get_expr_num_types()) for j in key[:deg]):
                raise ValueError(f"Keys in coeff_dict must be tuples of length divisible by 3 whose middle third contains only 0s and 1s (indicating valence of the first third). Reicieved keys: {list(coeff_dict.keys())}")
            if max_degree<deg:
                max_degree = deg
            for idx in key[2*deg:]:
                if idx not in seen_vs:
                    seen_vs.append(idx)
                    vector_spaces.append(idx)
                    nidx=_vsr_inh_idx(idx)
                    if idx==nidx:
                        _,b=spring.get(nidx,[0,[]])
                        spring[nidx]=[1,b]
                    else:
                        a,b=spring.get(nidx,[0,[]])
                        spring[nidx]=[a,b+[idx]]

            if deg in homogeneous_dicts:
                homogeneous_dicts[deg][key]=value
            else:
                homogeneous_dicts[deg]={key:value}
            if value != 0:
                processed_dict[key] = processed_dict.get(key,0)+value
                if min_degree<0 or 3*min_degree>kl:
                    min_degree = deg
                if prolongation_type!=-1 and all((key[deg]+j)%2==1 for j in key[deg+1:2*deg]):
                    if kl>0 and prolongation_type is None:
                        prolongation_type = key[deg]
                    elif kl>0 and prolongation_type!=key[deg]:
                        prolongation_type=-1
                else:
                    prolongation_type=-1
            if min_degree<0:
                min_degree=max_degree
        if prolongation_type is None:
            prolongation_type=-1

        # If all keys are removed (tensor is zero), restore a zero key
        if not processed_dict:
            processed_dict[tuple()] = 0
        replacements=dict()
        for k,v in spring.items():
            if _amb_prom is False and v[0]==0 and len(v[1])<2:
                continue
            for nkey in v[1]:
                replacements[nkey]=k
        if replacements:
            def replace(key,tgts,val):
                if len(tgts)==0:
                    return {key:val}
                deg=len(key)//3
                terms=[list(key)+[val]]
                for c in tgts:
                    new_terms=[]
                    for elem in terms:
                        kidx,kv,kvs,kval=elem[:deg],elem[deg:2*deg],elem[2*deg:3*deg],elem[3*deg]
                        elidx,tgtvs=kidx[c],kvs[c]
                        coeffs = from_vsr(tgtvs).basis[elidx].ambient_rep.coeffs
                        for cou,co in enumerate(coeffs):
                            if co==0:
                                continue
                            nidx,nvs,nval=list(kidx),list(kvs),co*kval
                            nidx[c]=cou
                            nvs[c]=replacements[tgtvs]
                            new_terms.append(nidx+kv+nvs+[nval])
                    terms = new_terms
                nd=Counter()
                for term in terms:
                    nd[tuple(term[:-1])]+=term[-1]
                return nd
            final_dict = Counter()
            for k,v in processed_dict.items():
                targets=[]
                for c,idx in enumerate(k[2*len(k)//3:]):
                    if idx in replacements:
                        targets.append(c)
                for nk, nv in replace(k, targets, v).items():
                    final_dict[nk] += nv
            processed_dict=final_dict
            vector_spaces=[vsidx for vsidx in vector_spaces if vsidx not in replacements]
        return processed_dict, max_degree, min_degree, prolongation_type, homogeneous_dicts, vector_spaces, spring, replacements

    @property
    def leading_valence(self):
        if self._leading_valence:
            return self._leading_valence
        lv = set()
        for k in self.coeff_dict:
            if len(k)>1:
                lv = lv|{k[len(k)//3]}
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
                lv = lv|{k[((2*len(k))//3)-1]}
        if len(lv)==1:
            self._trailing_valence = list(lv)[0]
        else:
            self._trailing_valence = -1     # denoting exceptional cases
        return self._trailing_valence

    @property
    def homogeneous_components(self):
        return [tensorProduct(self.vector_spaces,cd) for cd in self.homogeneous_dicts.values()]

    @property
    def free_vectors(self):
        vec_idx = set()
        for idx_t in self.coeff_dict:
            vec_idx = vec_idx | set(tuple(zip(idx_t[:len(idx_t)//3],idx_t[len(2*idx_t)//3:])))
        return set([from_vsr(self.vector_spaces[idx[1]]).basis[idx[0]] for idx in vec_idx])

    @property
    def coeffs(self):
        return tuple(self.coeff_dict.values())

    @property
    def terms(self):
        if self._terms is None:
            tList=[]
            for k,v in self.coeff_dict.items():
                if v==0:
                    continue
                if k==tuple():
                    tList.append(tensorProduct(self.vector_spaces,{k:v}))
                else:
                    deg=len(k)//3
                    seen=[]
                    newVS=[]
                    for idx in k[2*deg:]:
                        if idx in seen:
                            continue
                        newVS.append(idx)
                        seen.append(idx)
                    tList.append(tensorProduct(newVS,{k:v}))
            self._terms = tList if len(tList)>0 else [self]
        return self._terms

    def _convert_to_tp(self):
        return self

    def _compute_weights(self,test_weights=None,bound_number_of_weights=None,use_aggresive_weight_search=False):
        """Lazily compute weights for all terms in coeff_dict for each weight vector of the parent vector space or in a given list `test_weights`."""
        if test_weights is None:
            default_test = True
            test_weights={}
            mw=None
            changed = None
            for idx in self.vector_spaces:
                gl=from_vsr(idx).grading
                if mw is None:
                    if bound_number_of_weights is None:
                        mw=len(gl)
                    else:
                        mw=min(bound_number_of_weights,len(gl))
                    changed is False
                elif mw>len(gl):
                    mw=len(gl)
                    if changed is False:
                        changed is True
                test_weights[idx]=gl
            if changed is True:
                for idx in self.vector_spaces:
                    test_weights[idx]=test_weights[idx][:mw]
        else:
            default_test = False
            if isinstance(test_weights,(list,tuple)):
                mw=len(test_weights)
                test_weights={idx:test_weights for idx in self.vector_spaces}   # supporting old format
            else:
                mw = len(next(iter(test_weights.values())))
        if default_test is False or self._weights is None:
            weight_dict = {}
            for key, _ in self.coeff_dict.items():  # algo requires valence 1 for vec and 0 for covec
                weight_list = []
                k_deg=len(key)//3
                for w_idx in range(mw):
                    weight = 0
                    for j, index in enumerate(key[:k_deg]):
                        twL1=test_weights[key[j+2*k_deg]]
                        twL2=twL1[w_idx]
                        twL3=twL2[index]
                        weight += twL3 * (key[k_deg+j] * 2 - 1)
                    weight_list.append(weight)
                weight_dict[key] = tuple(weight_list)
            if default_test is True:
                self._weights=weight_dict
            else:
                return weight_dict

        return self._weights

    def compute_weight(self, test_weights=None, _return_mixed_weight_list = False, bound_number_of_weights=None):
        weights = list(set((self._compute_weights(test_weights=test_weights,bound_number_of_weights=bound_number_of_weights)).values()))
        if _return_mixed_weight_list is True:
            return weights
        if len(weights)==1:
            return weights[0]
        else:
            return "NoW"

    def get_weighted_components(self, weight_list, test_weights=None):
        """
        Return a new tensorProduct with components matching the given weight_list.

        Parameters:
        - weight_list: A list or tuple of weights to match against.

        Returns:
        - A new tensorProduct with a filtered coeff_dict.
        """
        wd = self._compute_weights(test_weights=test_weights)

        filtered_coeff_dict = {
            key: value for key, value in self.coeff_dict.items() if wd[key] == tuple(weight_list)
        }

        return tensorProduct(self.vector_space, filtered_coeff_dict)

    def tp(self, other):
        """Tensor product with another tensorProduct."""
        if get_dgcv_category(other) in {'subalgebra_element','algebra_element','vector_space_element'}:
            other = other._convert_to_tp()
        if isinstance(other,_get_expr_num_types()):
            return other*self
        if not isinstance(other, tensorProduct):
            raise ValueError(f"The other object must be a tensorProduct instance or vector space element related class. Revieved {type(other)} instead.")

        # Compute new coefficient dictionary
        new_coeff_dict = {}
        for key1, value1 in self.coeff_dict.items():
            for key2, value2 in other.coeff_dict.items():
                kl1,kl2=len(key1),len(key2)
                new_key = key1[:kl1//3] + key2[:kl2//3] + key1[kl1//3:2*kl1//3] + key2[kl2//3:2*kl2//3] + key1[2*kl1//3:] + key2[2*kl2//3:]
                new_coeff_dict[new_key] = value1 * value2

        return tensorProduct(self.vector_spaces, new_coeff_dict)

    @property
    def is_zero(self):
        for v in self.coeff_dict.values():
            if simplify_dgcv(v) != 0:
                return False
        else:
            return True

    def __str__(self):
        return tensor_VS_printer(self)

    def __repr__(self):
        return tensor_VS_printer(self)

    def _latex(self, printer=None,raw=False,**kwargs):
        """
        Defines the LaTeX representation for SymPy's latex() function.
        """
        return tensor_latex_helper(self) if raw else f'$\\displaystyle {tensor_latex_helper(self)}$'

    def _repr_latex_(self,raw=False,**kwargs):
        return self._latex(raw=raw)

    def _sympystr(self, printer):
        return self.__repr__()

    def subs(self, subs_data):
        def simsub(elem,sd):
            ne=simplify_dgcv(elem)
            if hasattr(ne,'subs'):
                return ne.subs(sd)
            else:
                return ne
        new_dict = {j:simsub(k,subs_data) for j,k in self.coeff_dict.items()}
        return tensorProduct(self.vector_spaces,new_dict)

    def simplify(self):
        new_dict = {j:simplify_dgcv(k) for j,k in self.coeff_dict.items()}
        return tensorProduct(self.vector_spaces,new_dict)

    def _eval_simplify(self, ratio=1.7, measure=None, rational=True, **kwargs):
        new_dict = {key: sp.simplify(value) for key, value in self.coeff_dict.items()}
        return tensorProduct(self.vector_spaces, new_dict)

    def __add__(self,other):
        if get_dgcv_category(other) in {'algebra_element','vector_space_element','subalgebra_element'}:
            other = other.ambient_rep._convert_to_tp()
        if isinstance(other, _get_expr_num_types()):
            other = tensorProduct('_',{tuple():other})
        if not isinstance(other,tensorProduct):
            raise TypeError('`+` can only combine `tensorProduct` elements with elements from variants of dgcv vector space type classes.')
        new_dict = dict(self.coeff_dict)
        for key, val in other.coeff_dict.items():
            if key in new_dict:
                new_dict[key] = new_dict[key]+val
            else:
                new_dict[key] = val
        return tensorProduct(self.vector_spaces,new_dict)

    def __radd__(self,other):
        if isinstance(other, _get_expr_num_types()):
            return tensorProduct('_',{tuple():other}) + self

    def __sub__(self,other):
        if get_dgcv_category(other) in {'algebra_element','vector_space_element','subalgebra_element'}:
            other = other._convert_to_tp()
        elif isinstance(other,_get_expr_num_types()):
            other = tensorProduct('_',{tuple():other})
        if not isinstance(other,tensorProduct):
            raise TypeError('`-` can only combine `tensorProduct` elements with elements from variants of dgcv vector space type classes.')
        new_dict = dict(self.coeff_dict)
        for key, val in other.coeff_dict.items():
            if key in new_dict:
                new_dict[key] = new_dict[key]-val
            else:
                new_dict[key] = -val
        return tensorProduct(self.vector_spaces,new_dict)

    def __rsub__(self,other):
        if isinstance(other, _get_expr_num_types()):
            return tensorProduct('_',{tuple():other}) - self

    def __truediv__(self, other):
        if isinstance(other,int):
            return sp.Rational(1,other)*self
        if isinstance(other,(float,sp.Expr)):
            return (1/other)*self

    def __matmul__(self, other):
        """Overload @ operator for tensor product."""
        return self.tp(other)

    def __rmatmul__(self, other):
        """Overload @ operator for tensor product."""
        if isinstance(other,_get_expr_num_types()):
            return tensorProduct('_',{tuple():other}).__matmul__(self)
        elif get_dgcv_category(other)=='tensorProduct':
            return other.__matmul__(self)
        elif get_dgcv_category(other) in {'algebra_element','subalgebra_element','vector_space_element'}:
            return other._convert_to_tp().__matmul__(self)

    def dual(self):
        def keyflip(key):
            def BInv(j,le,b):
                if j<kl//3 or j>=2*kl//3:
                    return b
                elif b==1:
                    return 0
                return 1
            kl=len(key)
            return tuple([BInv(c,kl,idx) for c,idx in enumerate(key)])
        return tensorProduct(self.vector_spaces,{keyflip(k):v for k,v in self.coeff_dict.items()})

    def __xor__(self,other):
        if other == "":
            return self.dual()
        raise ValueError("Invalid operation. Use `^''` to denote the dual.") from None

    @property
    def ambient_rep(self):
        return tensorProduct('_',self.coeff_dict,_amb_prom=True)

    def contract_call(self, other):
        """
        Contract the last index of self with the first index of other or handle algebra_element.
        """
        if self.is_zero:
            return self
        if other.is_zero:
            return 0*self
        if isinstance(other, tensorProduct):
            if len(self.vector_spaces)!=1 or tuple(self.vector_spaces) != tuple(other.vector_spaces):
                raise ValueError("Both tensors must be defined w.r.t. the same single vector space.")### generalize

            if self.trailing_valence + other.leading_valence != 1:
                raise ValueError("Contraction requires the first tensor factor of every term in other to have leading valence different from the last entry tensor factor from terms of self.")

            new_dict = {}
            for key1, value1 in self.coeff_dict.items():
                for key2, value2 in other.coeff_dict.items():
                    kl1,kl2=len(key1),len(key2)
                    if key1[(kl1//3)-1] == key2[0]:
                        new_key = key1[:(kl1//3)-1] + key2[1:(kl2//3)] + key1[(kl1//3):2*(kl1//3)-1] + key2[1+(kl2//3):2*kl2//3] + key1[2*(kl1//3):-1] + key2[1+(2*kl2//3):]
                        new_value = value1 * value2
                        new_dict[new_key] = new_dict.get(new_key, 0) + new_value

            return tensorProduct(self.vector_spaces, new_dict)

        if get_dgcv_category(other)=='subalgebra_element' and other.algebra != self.vector_space:
            other = other.ambient_rep

        elif hasattr(other, "algebra") and self.vector_space == other.algebra:
            if self.trailing_valence != 0:
                raise ValueError(f"Operating on algebra_element requires all terms in self to end in covariant tensor factor. Recieved self: {self} and other: {other}")
            other_as_tensor = other._convert_to_tp()
            return self.contract_call(other_as_tensor)
        else:
            raise ValueError("The other object must be a tensorProduct or an algebra_element with matching algebra.")

    def _recursion_contract_hom(self,other):
        if self.is_zero:
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
        if self.is_zero:
            return self
        if other.is_zero:
            return 0*self
        if isinstance(other, tensorProduct):
            if self.prolongation_type != other.prolongation_type:
                raise type(f'`tensorProduct` contraction is only supported between instances with matching `prolongation types`, not types: {self.prolongation_type} and {other.prolongation_type}')
            if tuple(self.vector_spaces) != tuple(other.vector_spaces):
                return 0*self       ### critical logic
            hc1 = self.homogeneous_components
            hc2 = other.homogeneous_components
            terms = [t1._recursion_contract_hom(t2) for t1,t2 in zip(hc1,hc2)]
            return sum(terms[1:],terms[0])

    def _bracket(self,other):
        if self.is_zero:
            return self
        if other.is_zero:
            return 0*self
        if get_dgcv_category(other)=='subalgebra_element' and other.algebra != self.vector_space:
            other = other.ambient_rep
        if isinstance(other, tensorProduct):
            if len(self.vector_spaces)!=1 or tuple(self.vector_spaces) != tuple(other.vector_spaces):
                raise ValueError("In `tensorProduct._bracket` both tensors must be defined w.r.t. the same vector_space.")

            if self.prolongation_type!=other.prolongation_type or self.prolongation_type==-1:
                raise ValueError("`tensorProduct._bracket` requires bracket components to have matching prolongation types.")

            complimentType = 1 if self.prolongation_type==0 else 0

            new_dict = {}
            for key1, value1 in self.coeff_dict.items():
                degree1 = len(key1) // 3
                for key2, value2 in other.coeff_dict.items():
                    degree2 = len(key2) // 3
                    for idx in range(1, degree1-1): # double check degree-1
                        if key1[idx] == key2[0] and len(key2) > 3:   # Check index matching before contraction
                            new_value = value1 * value2
                            k1_start = key1[:idx]
                            k2_start = key2[1:2]
                            k1_tail_inputs = key1[idx+1:degree1]
                            k2_inputs = key2[2:degree2]
                            new_tails = shufflings(k1_tail_inputs, k2_inputs)
                            valence = (self.prolongation_type,) + (complimentType,)*(degree1+degree2-3)
                            vfpointers = (self.vector_spaces[0],)*len(valence)
                            new_keys = [tuple(k1_start+k2_start+tuple(tail)+valence+vfpointers) for tail in new_tails]
                            for key in new_keys:
                                new_dict[key] = new_dict.get(key, 0) + new_value  # Accumulate values for duplicate keys
                    if key1[degree1-1] == key2[0]:   # Check index matching before contraction
                        new_value = value1 * value2
                        k1_start = key1[:degree1-1]
                        k2_inputs = key2[1:degree2]
                        valence = (self.prolongation_type,) + (complimentType,)*(degree1+degree2-3)
                        vfpointers = (self.vector_spaces[0],)*len(valence)
                        new_key = tuple(k1_start+k2_inputs+valence+vfpointers)
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
                            vfpointers = (self.vector_spaces[0],)*len(valence)                            
                            new_keys = [tuple(k2_start+k1_start+tuple(tail)+valence+vfpointers) for tail in new_tails]
                            for key in new_keys:
                                new_dict[key] = new_dict.get(key, 0) + new_value  # Accumulate values for duplicate keys
                    if key2[degree2-1] == key1[0]:   # Check index matching before contraction
                        new_value = -value1 * value2
                        k2_start = key2[:degree2-1]
                        k1_inputs = key1[1:degree1]
                        valence = (self.prolongation_type,) + (complimentType,)*(degree2+degree1-3)
                        vfpointers = (self.vector_spaces[0],)*len(valence)                            
                        new_key = tuple(k2_start+k1_inputs+valence+vfpointers)
                        new_dict[new_key] = new_dict.get(new_key, 0) + new_value
            return tensorProduct(self.vector_spaces, new_dict)

        elif hasattr(other, "algebra") and self.vector_space == other.algebra:
            if self.vector_space != other.algebra:
                raise ValueError("In `tensorProduct._bracket` both tensors must be defined w.r.t. the same single vector_space.")
            if self.prolongation_type != other.valence:
                raise ValueError("`tensorProduct._bracket` operating on algebra_element requires all terms in self to end in covariant tensor factor.")
            other_as_tensor = other._convert_to_tp()
            other_index,other_value = list(other_as_tensor.coeff_dict.items())[0]###!!! review
            new_dict = {}
            for key, value in self.coeff_dict.items():
                kl=len(key)
                if key[(kl//3)-1] == other_index:  # Matching indices for contraction
                    new_value = value * other_value
                    key_truncated = tuple(key[:(kl//3)-1]+key[(kl//3):(2*kl//3)-1]+key[(2*kl//3):-1])
                    new_dict[key_truncated] = new_dict.get(key_truncated, 0) + new_value
            return tensorProduct(self.vector_spaces, new_dict)
        else:
            raise ValueError("In `tensorProduct._bracket` the second factor must be a tensorProduct or an algebra_element with matching algebra.")

    def _bracket_gen(self,other):
        if self.is_zero:
            return self
        if other.is_zero:
            return 0*self
        if get_dgcv_category(other)=='subalgebra_element':
            other = other.ambient_rep
        if isinstance(other, tensorProduct):
            new_dict = {}
            for key1, value1 in self.coeff_dict.items():
                if len(key1)==0:
                    continue
                degree1 = len(key1) // 3
                bound_k1=tuple((key1[idx],key1[idx+degree1],key1[idx+2*degree1]) for idx in range(degree1))
                for key2, value2 in other.coeff_dict.items():
                    if len(key2)==0:
                        continue
                    degree2 = len(key2) // 3
                    bound_k2=tuple((key2[idx],key2[idx+degree2],key2[idx+2*degree2]) for idx in range(degree2))
                    for idx in range(1, degree1-1):
                        if key1[idx] == key2[0] and key1[idx+degree1]!=key2[degree2] and key1[idx+2*degree1]!=key2[2*degree2] and degree2>1:
                            new_value = value1 * value2
                            k1_start = bound_k1[:idx]
                            k2_start = bound_k2[1:2]
                            k1_tail_inputs = bound_k1[idx+1:degree1]
                            k2_inputs = bound_k2[2:degree2]
                            new_tails = shufflings(k1_tail_inputs, k2_inputs)
                            new_keys = [sum(zip(*(k1_start+k2_start+tuple(tail))),()) for tail in new_tails]
                            for key in new_keys:
                                new_dict[key] = new_dict.get(key, 0) + new_value
                    if degree1==1 and degree2==1 and key1[1:]==key2[1:]:
                        algB=from_vsr(key1[-1]).basis
                        if key1[1]==0:
                            elem1,elem2=algB[key1[0]].dual(),algB[key2[0]].dual()
                        else:
                            elem1,elem2=algB[key1[0]].dual(),algB[key2[0]].dual()
                        newElem=elem1*elem2
                        for k,v in newElem._convert_to_tp().coeff_dict.items():
                            new_dict[k] = new_dict.get(k, 0) + v
                    else:
                        if key1[degree1-1] == key2[0] and key1[2*degree1-1] != key2[degree2] and key1[-1] == key2[2*degree2]:
                            new_value = value1 * value2
                            k1_start = bound_k1[:degree1-1]
                            k2_inputs = bound_k2[1:degree2]
                            new_key = sum(zip(*(k1_start+k2_inputs)),())
                            new_dict[new_key] = new_dict.get(new_key, 0) + new_value
                        if key2[degree2-1] == key1[0] and key2[2*degree2-1] != key1[degree1] and key2[-1] == key1[2*degree1]:
                            new_value = -value2 * value1
                            k2_start = bound_k2[:degree2-1]
                            k1_inputs = bound_k1[1:degree1]
                            new_key = sum(zip(*(k2_start+k1_inputs)),())
                            new_dict[new_key] = new_dict.get(new_key, 0) + new_value


                    for idx in range(1, degree2-1):
                        if key2[idx] == key1[0] and key2[idx+degree2]!=key1[degree1] and key2[idx+2*degree2]!=key1[2*degree1]and degree1>1:
                            new_value = -value2 * value1
                            k2_start = bound_k2[:idx]
                            k1_start = bound_k1[1:2]
                            k2_tail_inputs = bound_k2[idx+1:degree2]
                            k1_inputs = bound_k1[2:degree1]
                            new_tails = shufflings(k2_tail_inputs, k1_inputs)
                            new_keys = [sum(zip(*(k2_start+k1_start+tuple(tail))),()) for tail in new_tails]
                            for key in new_keys:
                                new_dict[key] = new_dict.get(key, 0) + new_value
            return tensorProduct(self.vector_spaces, new_dict)

        elif hasattr(other, "algebra"):
            other_as_tensor = other._convert_to_tp()
            new_dict = {}
            for other_index,other_value in other_as_tensor.coeff_dict.items():
                for key, value in self.coeff_dict.items():
                    deg=len(key)//3
                    if key[deg-1] == other_index[0] and key[2*deg-1]==other_index[1] and key[-1]==other_index[2]:
                        new_value = value * other_value
                        key_truncated = tuple(key[:deg-1]+key[deg:2*deg-1]+key[2*deg:-1])
                        new_dict[key_truncated] = new_dict.get(key_truncated, 0) + new_value
                    if key[0] == other_index[0] and key[deg]==other_index[1] and key[2*deg]==other_index[2]:
                        new_value = -value * other_value
                        key_truncated = tuple(key[1:deg]+key[deg+1:2*deg]+key[2*deg+1:])
                        new_dict[key_truncated] = new_dict.get(key_truncated, 0) + new_value

            return tensorProduct(self.vector_spaces, new_dict)
        else:
            raise ValueError("In `tensorProduct._bracket` the second factor must be a tensorProduct or an algebra_element with matching algebra.")

    def _contraction_product(self,other, include_Lie_brackets=True):
        if len(self.terms)>1:
            tl=[term._contraction_product(other) for term in self.terms]
            return sum(tl)
        k, v = next(iter(self.coeff_dict.items()))
        deg=len(k)//3
        if deg==0:
            return v*other
        if deg==1:
            if get_dgcv_category(other)=='subalgebra_element':
                if k[1] == other.valence:
                    elem = v*(from_vsr(k[2]).basis[k[0]]) if k[1]==1 else v*(from_vsr(k[2]).basis[k[0]].dual())
                    if k[2]==other.dgcv_vs_id:
                        return elem*other
                    elif k[2]==other.ambient_rep.dgcv_vs_id:
                        return elem*(other.ambient_rep)
                    elif getattr(elem,'ambient',elem).dgcv_vs_id==other.ambient_rep.dgcv_vs_id:
                        return (elem.ambient_rep)*(other.ambient_rep)
                    else:
                        return 0*other
                else:
                    if k[2]==other.dgcv_vs_id:
                        return v*other.coeffs[k[0]]
                    elif k[2]==other.ambient_rep.dgcv_vs_id:
                        return v*other.ambient_rep.coeffs[k[0]]
                    else:
                        elem = v*(from_vsr(k[2]).basis[k[0]]) if k[1]==1 else v*(from_vsr(k[2]).basis[k[0]].dual())
                        if getattr(elem,'ambient',elem).dgcv_vs_id==other.ambient_rep.dgcv_vs_id:
                            return elem.ambient_rep._convert_to_tp()._contraction_product(other.ambient_rep)
                        else:
                            return 0*self
            elif get_dgcv_category(other)=='algebra_element':
                if k[1] == other.valence:
                    elem = v*(from_vsr(k[2]).basis[k[0]]) if k[1]==1 else v*(from_vsr(k[2]).basis[k[0]].dual())
                    if k[2]==other.dgcv_vs_id:
                        return elem*other
                    else:
                        if getattr(elem,'ambient',elem).dgcv_vs_id==other.dgcv_vs_id:
                            return (elem.ambient_rep)*(other)
                        else:
                            return 0*self
                else:
                    if k[2]==other.dgcv_vs_id:
                        return v*other.coeffs[k[0]]
                    else:
                        elem = v*(from_vsr(k[2]).basis[k[0]]) if k[1]==1 else v*(from_vsr(k[2]).basis[k[0]].dual())
                        if getattr(elem,'ambient',elem).dgcv_vs_id==other.dgcv_vs_id:
                            return elem.ambient_rep._convert_to_tp()._contraction_product(other)
                        else:
                            return 0*self
            elif get_dgcv_category(other)=='vectorSpace':
                if k[1] == other.valence:
                    return 0*self
                else:
                    if k[2]==other.dgcv_vs_id:
                        return v*other.coeffs[k[0]]
                    else:
                        elem = from_vsr(k[2])
                        if getattr(elem,'ambient',elem).dgcv_vs_id==other.dgcv_vs_id:
                            return elem.ambient_rep._convert_to_tp()._contraction_product(other)
                        else:
                            return 0*self
            elif get_dgcv_category(other)=='tensorProduct':
                newDict=Counter()
                for k2,v2 in other.coeff_dict.items():
                    deg2=len(k2)//3
                    if deg2==0:
                        newDict[k]+=v*v2
                    elif deg2==1:
                        if k[2]==k2[2]:
                            if k[1]!=k2[1]:
                                if k[0]==k2[0]:
                                    newDict[tuple()]+=v*v2
                            else:
                                elem1 = v*(from_vsr(k[2]).basis[k[0]]) if k[1]==1 else v*(from_vsr(k[2]).basis[k[0]].dual())
                                elem2 = v2*(from_vsr(k2[2]).basis[k2[0]]) if k2[1]==1 else v*(from_vsr(k2[2]).basis[k2[0]].dual())
                                nd=(elem1*elem2)
                                if hasattr(nd,'_convert_to_tp'):
                                    nd=nd._convert_to_tp().coeff_dict
                                elif hasattr(nd,'coeff_dict'):
                                    nd=nd.coeff_dict
                                elif isinstance(nd,_get_expr_num_types()):
                                    nd={tuple(),nd}
                                else:
                                    raise RuntimeError('unanticipated edge case in tensorProduct._contraction_product algo')
                                for nk,nv in nd.items():
                                    newDict[nk]+=nv
                        elif _vsr_inh_idx(k[2])==_vsr_inh_idx(k2[2]):
                            if k[2]==_vsr_inh_idx(k[2]):
                                nd=self._contraction_product(tensorProduct('_',{k2:v2},_amb_prom=True))
                            else:
                                nd=self.ambient_rep._contraction_product(tensorProduct('_',{k2:v2},_amb_prom=True))
                            if hasattr(nd,'_convert_to_tp'):
                                nd=nd._convert_to_tp().coeff_dict
                            elif hasattr(nd,'coeff_dict'):
                                nd=nd.coeff_dict
                            elif isinstance(nd,_get_expr_num_types()):
                                nd={tuple(),nd}
                            else:
                                raise RuntimeError('unanticipated edge case in tensorProduct._contraction_product algo')
                            for nk,nv in nd.items():
                                newDict[nk]+=nv
                    else:   # deg2>1
                        if k[1]!=k2[deg2]:
                            if k[2]==k2[2*deg2]:
                                if k[0]==k2[0]:
                                    newDict[tuple(k2[j] for j in range(1,3*deg2) if j not in {deg2,2*deg2})]+=v*v2
                            elif _vsr_inh_idx(k[2])==_vsr_inh_idx(k2[2*deg2]):
                                if k[2]==_vsr_inh_idx(k[2]):
                                    nd=self._contraction_product(tensorProduct('_',{k2:v2},_amb_prom=True))
                                else:
                                    nd=self.ambient_rep._contraction_product(tensorProduct('_',{k2:v2},_amb_prom=True))
                                if hasattr(nd,'_convert_to_tp'):
                                    nd=nd._convert_to_tp().coeff_dict
                                elif hasattr(nd,'coeff_dict'):
                                    nd=nd.coeff_dict
                                elif isinstance(nd,_get_expr_num_types()):
                                    nd={tuple(),nd}
                                else:
                                    raise RuntimeError('unanticipated edge case in tensorProduct._contraction_product algo')
                                for nk,nv in nd.items():
                                    newDict[nk]+=nv
                        if k[1]!=k2[2*deg2-1]:
                            if k[2]==k2[-1]:
                                if k[0]==k2[deg2-1]:
                                    newDict[tuple(k2[j] for j in range(3*deg2-1) if j not in {deg2-1,2*deg2-1})]+=-v*v2
                            elif _vsr_inh_idx(k[2])==_vsr_inh_idx(k2[-1]):
                                if k[2]==_vsr_inh_idx(k[2]):
                                    nd=self._contraction_product(tensorProduct('_',{k2:v2},_amb_prom=True))
                                else:
                                    nd=self.ambient_rep._contraction_product(tensorProduct('_',{k2:v2},_amb_prom=True))
                                if hasattr(nd,'_convert_to_tp'):
                                    nd=nd._convert_to_tp().coeff_dict
                                elif hasattr(nd,'coeff_dict'):
                                    nd=nd.coeff_dict
                                elif isinstance(nd,_get_expr_num_types()):
                                    nd={tuple(),nd}
                                else:
                                    raise RuntimeError('unanticipated edge case in tensorProduct._contraction_product algo')
                                for nk,nv in nd.items():
                                    newDict[nk]+=nv
                return tensorProduct('_',newDict)
        else:
            if get_dgcv_category(other) in {'subalgebra_element','algebra_element','vectorSpace'}:
                if _vsr_inh_idx(k[-1])!=k[-1] and k[-1]!=other.dgcv_vs_id and k[2*deg-1]!=other.valence and getattr(other,'ambient_rep',other).dgcv_vs_id == _vsr_inh_idx(k[-1]):
                    return self.ambient_rep._contraction_product(other)
                if _vsr_inh_idx(k[2*deg])!=k[2*deg] and k[deg]!=other.valence and k[2*deg]!=other.dgcv_vs_id and getattr(other,'ambient_rep',other).dgcv_vs_id == _vsr_inh_idx(k[2*deg]) :
                    return self.ambient_rep._contraction_product(other)
                newDict=Counter()

                if k[2*deg-1]!=other.valence:
                    locElem=other.ambient_rep if k[-1]==other.algebra.ambient.dgcv_vs_id else other
                    if k[-1]==locElem.dgcv_vs_id:
                        newDict[tuple(k[j] for j in range(3*deg-1) if j not in {deg-1,2*deg-1})]+=locElem.coeffs[k[deg-1]]*v
                if k[deg]!=other.valence:
                    locElem=other.ambient_rep if k[2*deg]==other.algebra.ambient.dgcv_vs_id else other
                    if k[2*deg]==locElem.dgcv_vs_id:
                        newDict[tuple(k[j] for j in range(1,3*deg) if j not in {deg,2*deg})]+=-locElem.coeffs[k[0]]*v
                return tensorProduct('_',newDict)
            elif get_dgcv_category(other)=='tensorProduct':
                newDict=Counter()
                for key,val in self._unpromoted_spring:
                    if val in other.vector_spaces:
                        return self.ambient_rep._contraction_product(other)
                    elif val in other._vs_spring and key not in other._unpromoted_spring:
                        return self.ambient_rep._contraction_product(other.ambient_rep)

                bound_k=tuple((k[idx],k[idx+deg],k[idx+2*deg]) for idx in range(deg)) 
                for k2,v2 in other.coeff_dict.items():
                    deg2=len(k2)//3
                    if deg2==0:
                        newDict[k]+=v*v2
                    elif deg2==1:
                        if k2[1]==0:
                            nd = self._contraction_product(v2*from_vsr(k2[2]).basis[k2[0]].dual())
                        else:
                            nd = self._contraction_product(v2*from_vsr(k2[2]).basis[k2[0]])
                        if hasattr(nd,'coeff_dict'):
                            nd=nd.coeff_dict
                        elif hasattr(nd,'_convert_to_tp'):
                            nd=nd._convert_to_tp().coeff_dict
                        elif isinstance(nd,_get_expr_num_types()):
                            nd={tuple(),nd}
                        else:
                            raise RuntimeError('unanticipated edge case in tensorProduct._contraction_product algo')
                        for nk,nv in nd.items():
                            newDict[nk]+=nv
                    else:
                        bound_k2=tuple((k2[idx],k2[idx+deg2],k2[idx+2*deg2]) for idx in range(deg2)) 
                        for idx in range(1, deg-1):
                            if k[idx] == k2[0] and k[idx+deg]!=k2[deg2] and k[idx+2*deg]==k2[2*deg2]:
                                new_value = v * v2
                                k1_start = bound_k[:idx]
                                k2_start = bound_k2[1:2]
                                k1_tail_inputs = bound_k[idx+1:deg]
                                k2_inputs = bound_k2[2:deg2]
                                new_tails = shufflings(k1_tail_inputs, k2_inputs)
                                new_keys = [sum(zip(*(k1_start+k2_start+tuple(tail))),()) for tail in new_tails]
                                for key in new_keys:
                                    newDict[key]+=new_value
                        if k[deg-1] == k2[0] and k[2*deg-1] != k2[deg2] and k[-1] == k2[2*deg2]:
                            new_value = v * v2
                            k1_start = bound_k[:deg-1]
                            k2_inputs = bound_k2[1:deg2]
                            new_key = sum(zip(*(k1_start+k2_inputs)),())
                            newDict[new_key]+=new_value
                        if k2[deg2-1] == k[0] and k2[2*deg2-1] != k[deg] and k2[-1] == k[2*deg]:
                            new_value = -v2 * v
                            k2_start = bound_k2[:deg2-1]
                            k1_inputs = bound_k[1:deg]
                            new_key = sum(zip(*(k2_start+k1_inputs)),())
                            newDict[new_key]+=new_value
                        for idx in range(1, deg2-1):
                            if k2[idx] == k[0] and k2[idx+deg2]!=k[deg] and k2[idx+2*deg2]==k[2*deg]:
                                new_value = -v2 * v
                                k2_start = bound_k2[:idx]
                                k1_start = bound_k[1:2]
                                k2_tail_inputs = bound_k2[idx+1:deg2]
                                k1_inputs = bound_k[2:deg]
                                new_tails = shufflings(k2_tail_inputs, k1_inputs)
                                new_keys = [sum(zip(*(k2_start+k1_start+tuple(tail))),()) for tail in new_tails]
                                for key in new_keys:
                                    newDict[key]+=new_value
            return tensorProduct(self.vector_spaces, newDict)

    @property
    def free_symbols(self):
        fs = set()
        for c in self.coeff_dict.values():
            fs |= getattr(c, "free_symbols", set())
        return fs

    def dual_pairing(self,other):
        if get_dgcv_category(other) in {'algebra_element','subalgebra_element','vectorpace_element'}:
            other=other._convert_to_tp()
        if get_dgcv_category(other)!='tensorProduct':
            raise TypeError(f'cannot apply dual_pairing to type {type(other)}')
        terms1,terms2 = self.terms, other.terms
        result = 0
        for t1 in terms1:
            for t2 in terms2:
                cd1,cd2=next(iter(t1.coeff_dict)),next(iter(t2.coeff_dict))
                deg1,deg2=len(cd1)//3,len(cd2)//3
                if deg1!=deg2:
                    continue
                if all(j==k for j,k in zip(cd1[:deg1],cd2[:deg2])) and all(j+k==1 for j,k in zip(cd1[deg1:2*deg1],cd2[deg2:2*deg2])) and all(j==k for j,k in zip(cd1[2*deg1:],cd2[2*deg2:])):
                    result+=t1.coeff_dict[cd1]*t2.coeff_dict[cd2]
        return result



    def __mul__(self, other):
        """Overload * to compute the contraction product, with special logic for algebra_element."""
        if isinstance(other, _get_expr_num_types()):
            new_coeff_dict = {key: value * other for key, value in self.coeff_dict.items()}
            return tensorProduct(self.vector_spaces, new_coeff_dict)
        if get_dgcv_category(other) in {'subalgebra_element', 'vector_space_element', 'algebra_element', 'tensorProduct'}:
            return self._contraction_product(other)
        else:
            raise ValueError(f"Unsupported operation for * between the given object types: {type(self)} and {type(other)}; `other` dgcv type is {get_dgcv_category(other)}; DEBUG data: {get_dgcv_category(other)}")

    def __rmul__(self, other):
        if isinstance(other, _get_expr_num_types()):
            new_coeff_dict = {key: value * other for key, value in self.coeff_dict.items()}
            return tensorProduct(self.vector_spaces, new_coeff_dict)
        if get_dgcv_category(other) in {'subalgebra_element', 'vector_space_element', 'algebra_element'}:
            return other._convert_to_tp()._contraction_product(self)
        else:
            raise ValueError(f"Unsupported operation for * between the given object types: {type(self)} and {type(other)}; `other` dgcv type is {get_dgcv_category(other)}; DEBUG data: {get_dgcv_category(other)}")

    # def _legacy_mul__(self, other):
    #     """Overload * to compute the contraction product, with special logic for algebra_element."""
    #     if self.max_degree==0:
    #         coef=self.coeff_dict[tuple()]
    #         if coef!=0:      # max_degree loses relevance when coef is zero 
    #             return coef*other
    #     if isinstance(other, _get_expr_num_types()):
    #         new_coeff_dict = {key: value * other for key, value in self.coeff_dict.items()}
    #         return tensorProduct(self.vector_spaces, new_coeff_dict)

    #     if getattr(other,'is_zero',False) or self.is_zero:
    #         return 0*self

    #     if get_dgcv_category(other) in {'subalgebra_element', 'vector_space_element', 'algebra_element'}:
    #         other = other._convert_to_tp()

    #     if isinstance(other, tensorProduct):
    #         # Lie bracket for two tensorProducts
    #         if self.vector_space != other.vector_space:
    #             return 0*self
    #         if other.max_degree==0:
    #             return other.coeff_dict[tuple()]*self

    #         if other.max_degree==1 and other.min_degree==1:
    #             if self.max_degree==1 and self.min_degree==1:
    #                 if self.prolongation_type == other.prolongation_type:
    #                     if isinstance(self.vector_space,vector_space_class):
    #                         return 0*self
    #                     pt = self.prolongation_type     ###!!! Note: vector_spaces attribute brittle structure dependence
    #                     coeffs1 = [self.coeff_dict.get((j, pt, self.vector_spaces[0]), 0) for j in range(self.vector_space.dimension)]
    #                     coeffs2 = [other.coeff_dict.get((j, pt, self.vector_spaces[0]), 0) for j in range(self.vector_space.dimension)]
    #                     LA_elem1 = self.vector_space._class_builder(coeffs1,pt)
    #                     LA_elem2 = self.vector_space._class_builder(coeffs2,pt)
    #                     rTP=LA_elem1*LA_elem2
    #                     _cached_caller_globals['DEBU']=rTP
    #                     return rTP._convert_to_tp()
    #                 else:
    #                     ###!!! mixed prolongation type needs an additional logic branch
    #                     pt1 = self.prolongation_type
    #                     pt2 = other.prolongation_type
    #                     if any(pt not in [0,1] for pt in [pt1,pt2]): 
    #                         raise RuntimeError('prolongation type error')
    #                     warnings.warn('DEBUG NOTE: incomplete logic branch triggered...')
    #                     return sum([self.coeff_dict[(j,pt1,self.vector_spaces[0])]*other.coeff_dict[(j,pt2,other.vector_spaces[0])] for j in range(self.vector_space.dimension)])
    #             else:
    #                 if self.trailing_valence != other.prolongation_type and self.trailing_valence!=-1:
    #                     cd = {}
    #                     for t,tv in other.coeff_dict.items():
    #                         for key,val in {tuple(k[:(len(k)//3)-1]+k[(len(k)//3):(2*len(k)//3)-1]+k[(2*len(k)//3):-1]):v*tv for k,v in self.coeff_dict.items() if k[(len(k)//3)-1]==t[0]}.items():
    #                             cd[key] = cd.get(key,0)+val
    #                     return tensorProduct(self.vector_spaces,cd)
    #                 else:
    #                     raise TypeError(f'* cannot operate on `tensorProduct` pairs if: \n second arg is degree 1 \n first argument has degree>1 \n first\'s `trailing_valence` doesn\'t match the second\'s `prolongation_type`.\n `trailing_valence` of {self}: {self.trailing_valence} \n `prolongation_type` of {other}: {other.prolongation_type}')
    #         if self.max_degree==1 and self.min_degree==1:
    #             return -1 * other * self
    #         return self._bracket(other)
    #     else:
    #         raise ValueError(f"Unsupported operation for * between the given object types: {type(self)} and {type(other)}; `other` dgcv type is {get_dgcv_category(other)}; DEBUG data: {get_dgcv_category(other)}")

    # def _legacy_rmul__(self, other):
    #     if isinstance(other, _get_expr_num_types()):
    #         return self.__mul__(other)
    #     if get_dgcv_category(other)=='subalgebra_element' and other.algebra != self.vector_space:
    #         other = other.ambient_rep
    #     if hasattr(other, "_convert_to_tp"):
    #         return other._convert_to_tp()*self
    #     else:
    #         return NotImplemented

    def __neg__(self):
        return -1*self

    def __call__(self, *args, contract_from_left=False,demote_to_VS_when_possible=True):
        if len(args)>self.min_degree:
            return 'UNDEF'
        if len(args)==0:
            return self
        if len(self.terms)>1:
            return sum(term(*args,contract_from_left=contract_from_left,demote_to_VS_when_possible=demote_to_VS_when_possible) for term in self.terms)
        if len(args)>1:
            return self.__call(args[0],contract_from_left=contract_from_left,demote_to_VS_when_possible=demote_to_VS_when_possible).__call__(*args[1:],contract_from_left=contract_from_left,demote_to_VS_when_possible=demote_to_VS_when_possible)
        parents = [idx for idx in self._vs_spring if idx not in self.vector_spaces]
        for arg in args:
            if get_dgcv_category(arg)=='tensorProduct':
                tidx=arg.vector_spaces
            else:
                tidx=[getattr(arg,'dgcv_vs_id',None)]
            for t in tidx:
                if t not in self.vector_spaces and _vsr_inh_idx(t) in parents:
                    return tensorProduct('_',self.coeff_dict,_amb_prom=True).__call__(*args, contract_from_left=contract_from_left,demote_to_VS_when_possible=demote_to_VS_when_possible)
        k,v = next(iter(self.coeff_dict.items()))
        deg = len(k)//3
        kc,kv,kvs = k[:deg],k[deg:2*deg],k[2*deg:]
        factor = 1
        elem = args[0]
        if contract_from_left is True:
            idx=0
        else:
            idx=deg-1
        if get_dgcv_category(elem)=='tensorProduct': 
            if elem.max_degree==1 and elem.min_degree==1:
                if any((vs!=_vsr_inh_idx(vs) and _vsr_inh_idx(vs) in self.vector_spaces) for vs in elem.vector_spaces):
                    elem = tensorProduct('_',elem.coeff_dict,_amb_prom=True)
                accu=0
                for kappa,nu in elem.coeff_dict.items():
                    if kappa[2]==kvs[idx]:
                        if kappa[1]!=kv[idx] and kappa[0]==kc[idx]:
                            accu+=nu
                factor=factor*accu
            else:
                raise TypeError('tensorProduct.__call__() can only operatate on lists of vector space like elements, but it was applied to a tensorProduct element of degree≠1.')
        elif kv[idx]==getattr(elem,'valence',kv[idx]):
            factor=0
        elif isinstance(getattr(elem,'dgcv_vs_id',None),numbers.Integral):
            if kvs[idx]==_vsr_inh_idx(elem.dgcv_vs_id) and elem.dgcv_vs_id!=_vsr_inh_idx(elem.dgcv_vs_id):
                elem=elem.ambient_rep
            if kvs[idx]==elem.dgcv_vs_id:
                coeffs=getattr(elem,'coeffs',[])
                if isinstance(coeffs,(list,tuple)) and len(coeffs)>kc[idx]:
                    factor=factor*coeffs[kc[idx]]
            else:
                factor=0
        else:
            factor=0
        if factor==0:
            return 0
        if contract_from_left is True:
            nk=tuple(kc[1:]+kv[1:]+kvs[1:])
        else:        
            nk=tuple(kc[:-1]+kv[:-1]+kvs[:-1])
        nv=v*factor
        if demote_to_VS_when_possible is True and len(nk)==3:
            ne=nv*(from_vsr(nk[2]).basis[nk[0]])
            if nk[1]==0:
                ne=ne.dual()
            return ne
        else:
            return tensorProduct('_',{nk:nv})

        # for k,v in self.coeff_dict.items():
        #     kl = len(k)//3
        #     kc = k[:kl]
        #     kv = k[kl:2*kl]
        #     kvs = k[2*kl:]
        #     factor = 1
        #     for c,elem in enumerate(args):
        #         if contract_from_left is True:
        #             idx=c
        #         else:
        #             idx=kl-1-c
        #         if elem.valence==kv[idx] or elem.dgcv_vs_id!=kvs[idx]:
        #             factor=0
        #             break
        #         else:
        #             factor = elem.coeffs[kc[idx]]*factor
        #     if contract_from_left is True:
        #         newKey = tuple(kc[len(args):]+kv[len(args):]+kvs[len(args):])
        #     else:
        #         newKey = tuple(kc[:kl-len(args)]+kv[:kl-len(args)]+kvs[:kl-len(args)])
        #     new_cd[newKey]+=factor*v
        # if len(new_cd)==1 and tuple() in new_cd:
        #     return new_cd[tuple()]
        # elif demote_to_VS_when_possible is True and len(new_cd)>0 and all(len(t)==3 for t in new_cd) and len(set([t[-1] for t in new_cd]))==1:
        #     kvL=list(new_cd.items())
        #     vsB=get_vs_registry()[kvL[0][0][-1]].basis
        #     def genTerm(kvD,vsD):
        #         elem=kvD[1]*vsD[kvD[0][0]]
        #         if kvD[0][1]==1:
        #             return elem
        #         else:
        #             return elem.dual()
        #     term=genTerm(kvL[0],vsB)
        #     terms=[term]
        #     for kv in kvL[1:]:
        #         term+=genTerm(kv,vsB)
        #         terms.append(genTerm(kv,vsB))
        #     return term
        # else:
        #     return tensorProduct(self.vector_spaces,new_cd)

def multi_tensor_product(*tp):
    product = 1
    tpTypes = {'tensorProduct','algebra_element','subalgebra_element','vector_space_element'}
    for elem in tp:
        if get_dgcv_category(elem) not in tpTypes:
            raise TypeError(f'multi_tensor_product only excepts arguments that `dgcv` can process as factors in a tensor product. Recieved type {type(elem)}')
        product = product @ elem
    return product



KeyFlat = Tuple[Any, ...]
KeyLike = KeyFlat
class _tensor_structure_data(MutableMapping):
    __slots__ = ("shape", "_data")

    def __init__(
        self,
        init: Optional[Union[Mapping[KeyLike, Any], Iterable[Tuple[KeyLike, Any]], "_tensor_structure_data"]] = None,
        *,
        shape: Optional[str] = None,
        _validated: Any = None,
    ):
        if shape not in (None, "symmetric", "skew"):
            shape=None
        self.shape = shape
        self._data: dict[KeyFlat, Any] = {}

        if init is None:
            return

        it = init.items() if isinstance(init, Mapping) else (init.items() if isinstance(init, _tensor_structure_data) else init)
        is_validated = (_validated == retrieve_passkey())

        if self.shape is None:
            for k, v in it:
                if v != 0:
                    self._data[k] = self._data.get(k, 0) + v
            return

        if self.shape == "symmetric":
            for k, v in it:
                if v == 0:
                    continue
                _, fkc = self._canon_symmetric_for_access(k)
                self._data[fkc] = self._data.get(fkc, 0) + v
            return

        # skew
        if is_validated:
            for k, v in it:
                if v == 0:
                    continue
                s, fkc = self._canon_skew_for_access(k)
                if s is None:
                    raise ValueError("Nonzero skew term has repeated indices.")
                self._data[fkc] = self._data.get(fkc, 0) + s * v
        else:
            for k, v in it:
                if v == 0:
                    continue
                s, fkc = self._canon_skew_for_access(k)
                if s is None:
                    raise ValueError("Nonzero skew term has repeated indices.")
                self._data[fkc] = self._data.get(fkc, 0) + s * v
        if len(self._data)==0:
            self._data[tuple()]=0

    def __getitem__(self, key: KeyLike) -> Any:
        if self.shape is None:
            return self._data.get(key, 0)
        if self.shape == "symmetric":
            _, fkc = self._canon_symmetric_for_access(key)
            return self._data.get(fkc, 0)
        s, fkc = self._canon_skew_for_access(key)
        if s is None:
            return 0
        return s * self._data.get(fkc, 0)

    def __setitem__(self, key: KeyLike, value: Any) -> None:
        if value == 0:
            try:
                del self[key]
            except KeyError:
                pass
            return

        if self.shape is None:
            self._data[key] = value
            return

        if self.shape == "symmetric":
            _, fkc = self._canon_symmetric_for_access(key)
            self._data[fkc] = value
            return

        s, fkc = self._canon_skew_for_access(key)
        if s is None:
            raise ValueError("Cannot set nonzero skew term with repeated indices.")
        self._data[fkc] = value * s

    def __delitem__(self, key: KeyLike) -> None:
        if self.shape is None:
            del self._data[key]
            return

        if self.shape == "symmetric":
            _, fkc = self._canon_symmetric_for_access(key)
            del self._data[fkc]
            return

        s, fkc = self._canon_skew_for_access(key)
        if s is None:
            raise KeyError(key)
        del self._data[fkc]

    def __iter__(self) -> Iterator[KeyFlat]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        try:
            return self[key] != 0
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"_tensor_structure_data(shape={self.shape!r}, size={len(self)})"

    def add(self, key: KeyLike, delta: Any) -> None:
        if delta == 0:
            return
        if self.shape is None:
            self._data[key] = self._data.get(key, 0) + delta
            return
        if self.shape == "symmetric":
            _, fkc = self._canon_symmetric_for_access(key)
            self._data[fkc] = self._data.get(fkc, 0) + delta
            return
        s, fkc = self._canon_skew_for_access(key)
        if s is None:
            raise ValueError("Cannot add nonzero skew term with repeated indices.")
        self._data[fkc] = self._data.get(fkc, 0) + s * delta

    def update(self, other: Optional[Union[Mapping[KeyLike, Any], Iterable[Tuple[KeyLike, Any]]]] = None, **kwargs: Any) -> None:
        if other is not None:
            it = other.items() if isinstance(other, Mapping) else other
            for k, v in it:
                self.add(k, v)
        for k, v in kwargs.items():
            self.add(k, v)

    @staticmethod
    def _bundle(flat: KeyFlat) -> list[Tuple[Any, Any, Any]]:
        if not flat:
            return []
        d = len(flat) // 3
        return [(flat[j], flat[j + d], flat[j + 2 * d]) for j in range(d)]

    @staticmethod
    def _unbundle(trip: list[Tuple[Any, Any, Any]]) -> KeyFlat:
        if not trip:
            return tuple()
        i, v, vs = zip(*trip)
        return tuple(i) + tuple(v) + tuple(vs)

    def _canon_symmetric_for_access(self, flat: KeyFlat) -> Tuple[int, KeyFlat]:
        trip = self._bundle(flat)
        _, sorted_trip = permSign(trip, returnSorted=True)
        return 1, self._unbundle(sorted_trip)

    def _canon_skew_for_access(self, flat: KeyFlat) -> Tuple[Optional[int], KeyFlat]:
        trip = self._bundle(flat)
        if len(trip) != len(set(trip)):
            _, sorted_trip = permSign(trip, returnSorted=True)
            return None, self._unbundle(sorted_trip)
        s, sorted_trip = permSign(trip, returnSorted=True)
        return int(s), self._unbundle(sorted_trip)

    def __iadd__(self, other):
        if isinstance(other, _tensor_structure_data) or isinstance(other, Mapping):
            self.update(other)
            return self
        try:
            iter(other)
        except TypeError:
            return NotImplemented
        self.update(other)
        return self

    def __isub__(self, other):
        if isinstance(other, _tensor_structure_data) or isinstance(other, Mapping):
            for k, v in other.items():
                self.add(k, -v)
            return self
        try:
            it = iter(other)
        except TypeError:
            return NotImplemented
        for k, v in it:
            self.add(k, -v)
        return self


def mergeVS(L1,L2):
    return '_'
    # filtered=[]
    # for vs in L2:
    #     if vs not in L1:
    #         filtered.append(vs)
    # return tuple(list(L1)+filtered)

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





